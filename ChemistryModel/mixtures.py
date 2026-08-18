# ============================================================
# Starting mixtures
# ============================================================
#
# Kept in their own module so the headless batch runner does not
# have to import the viewer, and therefore does not need Qt or
# OpenGL installed to run.
#
#   "atoms"     scatters loose atoms of the given elements
#   "molecules" builds real geometry from build_box.BUILDERS
#
# Loose atoms bond immediately and heat themselves, so they need
# no spark. Molecular mixtures are stable until something breaks
# them open, which is what the lightning channel is for.

import json
import os
from collections import Counter
from numbers import Integral


# Anything defined in mixtures.json is merged in on top of the
# built-in list. The control panel writes that file rather than
# editing this one, so a mixture invented in the interface is
# visible to the batch runner and the viewer without any of them
# having to edit Python.

CUSTOM_FILE = "mixtures.json"


def supported_species(kind):
    """Return species accepted by the existing runtime for one mixture kind."""
    if kind == "atoms":
        import reactive
        return tuple(reactive.ELEMENTS)
    if kind == "molecules":
        import build_box
        return tuple(sorted(build_box.BUILDERS))
    return ()


def validate_contents(kind, contents):
    if kind not in ("atoms", "molecules"):
        raise ValueError("composition type must be atoms or molecules")
    allowed = set(supported_species(kind))
    normalised = {}
    for species, amount in dict(contents).items():
        species = str(species).strip()
        if species not in allowed:
            raise ValueError(f"{species or '(blank)'} is not a supported {kind} species")
        if isinstance(amount, bool) or not isinstance(amount, Integral) or int(amount) <= 0:
            raise ValueError(f"amount for {species} must be a positive integer")
        normalised[species] = normalised.get(species, 0) + int(amount)
    if not normalised:
        raise ValueError("a mixture needs at least one species")
    return normalised


def parse_definition(text, kind):
    """Parse the legacy one-species/count-per-line representation strictly."""
    contents = {}
    for number, line in enumerate(str(text).splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"line {number}: expected 'species amount'")
        try:
            amount = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"line {number}: amount must be an integer") from exc
        contents[parts[0]] = contents.get(parts[0], 0) + amount
    return validate_contents(kind, contents)


def format_definition(contents):
    return "\n".join(f"{species} {int(amount)}" for species, amount in contents.items())


def element_totals(kind, contents):
    contents = validate_contents(kind, contents)
    totals = Counter()
    if kind == "atoms":
        totals.update(contents)
    else:
        import build_box
        for molecule, amount in contents.items():
            symbols, _ = build_box.BUILDERS[molecule]()
            for symbol in symbols:
                totals[str(symbol)] += int(amount)
    return dict(totals)


def atom_count(kind, contents):
    return int(sum(element_totals(kind, contents).values()))


def composition_metrics(kind, contents, box_size):
    elements = element_totals(kind, contents)
    atoms = int(sum(elements.values()))
    molecules = int(sum(contents.values())) if kind == "molecules" else None
    box_size = float(box_size)
    density = atoms / box_size ** 3 if atoms and box_size > 0 else 0.0
    return {
        "atoms": atoms,
        "molecules": molecules,
        "box_A": box_size,
        "density_atoms_per_A3": density,
        "elements": elements,
    }


def species_atom_sizes(kind, contents):
    contents = validate_contents(kind, contents)
    if kind == "atoms":
        return {species: 1 for species in contents}
    import build_box
    return {
        species: len(build_box.BUILDERS[species]()[0])
        for species in contents
    }


def scale_to_density(kind, contents, box_size, target_density):
    """Scale integer species counts at fixed box, preserving their ratios.

    Counts are apportioned around one common scale factor. Every species that
    was present remains present. Molecular mixtures may not hit an arbitrary
    atom total exactly because one added molecule can contain several atoms.
    """
    contents = validate_contents(kind, contents)
    box_size = float(box_size)
    target_density = float(target_density)
    if box_size <= 0:
        raise ValueError("box size must be positive")
    if target_density <= 0:
        raise ValueError("target density must be positive")
    sizes = species_atom_sizes(kind, contents)
    current_atoms = sum(sizes[name] * amount for name, amount in contents.items())
    target_atoms = max(1, int(round(target_density * box_size ** 3)))
    minimum_atoms = sum(sizes.values())
    if target_atoms < minimum_atoms:
        raise ValueError(
            f"target loading is about {target_atoms} atoms, but retaining all "
            f"{len(contents)} selected species needs at least {minimum_atoms} atoms"
        )
    scale = target_atoms / current_atoms
    ideal = {name: amount * scale for name, amount in contents.items()}
    scaled = {name: max(1, int(value // 1)) for name, value in ideal.items()}

    def atoms(values):
        return sum(sizes[name] * amount for name, amount in values.items())

    def ratio_error(values):
        return sum((values[name] - ideal[name]) ** 2 for name in values)

    # Add/remove one whole species unit only when it improves the atom-total
    # error. Ratio error breaks ties, giving largest-remainder-like behaviour.
    while True:
        present_atoms = atoms(scaled)
        present_error = abs(present_atoms - target_atoms)
        candidates = []
        for name in scaled:
            for change in (-1, 1):
                if change < 0 and scaled[name] <= 1:
                    continue
                candidate_atoms = present_atoms + change * sizes[name]
                error = abs(candidate_atoms - target_atoms)
                if error > present_error:
                    continue
                candidate = dict(scaled)
                candidate[name] += change
                candidates.append((error, ratio_error(candidate), name, change))
        if not candidates:
            break
        best = min(candidates)
        if best[0] == present_error and best[1] >= ratio_error(scaled) - 1e-15:
            break
        scaled[best[2]] += best[3]

    result_atoms = atoms(scaled)
    return {
        "contents": scaled,
        "current_atoms": int(current_atoms),
        "target_atoms": int(target_atoms),
        "result_atoms": int(result_atoms),
        "scale": float(scale),
        "target_density_atoms_per_A3": target_density,
        "result_density_atoms_per_A3": result_atoms / box_size ** 3,
        "box_A": box_size,
    }


BUILT_IN = {
    "loose H + O": ("atoms", {"H": 40, "O": 20}),
    "loose C H N O": ("atoms", {"C": 8, "H": 44, "N": 6, "O": 8}),
    "Miller-Urey": (
        "molecules", {"CH4": 6, "NH3": 4, "H2O": 6, "H2": 8}
    ),
    "water box": ("molecules", {"H2O": 24}),
    "methane box": ("molecules", {"CH4": 12}),
    "H rich loose": ("atoms", {"C": 8, "H": 60, "N": 6, "O": 8}),
    "H rich x5": ("atoms", {"C": 40, "H": 300, "N": 30, "O": 40}),
    "carbon rich": ("atoms", {"C": 80, "H": 200, "N": 20, "O": 30}),
    "carbon skeleton growth": ("atoms", {"C": 94, "H": 236}),
    "oxygenated carbon growth": (
        "atoms", {"C": 73, "H": 183, "O": 73}
    ),
    "amino carbon growth": ("atoms", {"C": 83, "H": 206, "N": 41}),
    "amino alcohol growth": (
        "atoms", {"C": 66, "H": 197, "N": 33, "O": 33}
    ),
    "balanced complex CHNO": (
        "atoms", {"C": 63, "H": 188, "N": 31, "O": 47}
    ),
    "[calibration] NH2 radicals": ("molecules", {"NH2": 24}),
    "[calibration] OH radicals": ("molecules", {"OH": 24}),
    "[validation] stable small molecules": (
        "molecules", {"H2": 4, "CH4": 4, "NH3": 4, "H2O": 4}
    ),
    "[validation] nitrogen radicals": (
        "molecules", {"NH3": 8, "NH2": 8, "H2": 4}
    ),
    "[validation] oxygen radicals": (
        "molecules", {"H2O": 8, "OH": 8, "H2": 4}
    ),
}


def load_custom(path=CUSTOM_FILE):
    if not os.path.exists(path):
        return {}

    try:
        with open(path) as handle:
            stored = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}

    mixtures = {}

    for name, entry in stored.items():
        kind = entry.get("kind", "atoms")
        contents = entry.get("contents", {})

        if kind in ("atoms", "molecules") and contents:
            mixtures[name] = (kind, contents)

    return mixtures


def save_custom(mixtures, path=CUSTOM_FILE):
    stored = {
        name: {"kind": kind, "contents": contents}
        for name, (kind, contents) in mixtures.items()
    }

    with open(path, "w") as handle:
        json.dump(stored, handle, indent=1)


def all_mixtures():
    combined = dict(BUILT_IN)
    combined.update(load_custom())

    return combined


STARTS = all_mixtures()
