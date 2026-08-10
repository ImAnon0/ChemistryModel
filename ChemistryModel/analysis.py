import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import reactive as R
import fingerprint as F
import amphiphile as A
import bonding


# ============================================================
# Analysing a recorded run
# ============================================================
#
# The viewer works one frame at a time, which is fine when a
# human is driving it. Going through thousands of frames needs
# the bond search vectorised, so everything here works on all
# pairs at once rather than looping over atoms in Python.


HEAVY = ("C", "N", "O")

EXPECTED_BONDS = {"H": 1, "C": 4, "N": 3, "O": 2}


def pair_indices(count):
    return np.triu_indices(count, k=1)


def cutoffs_for(types, first, second):
    return (
        R.CUTOFF_INNER[types[first], types[second]],
        R.CUTOFF_OUTER[types[first], types[second]],
    )


def bonds_in_frame(positions, box_size, first, second,
                   inner, outer, threshold=0.35):
    offsets = positions[second] - positions[first]
    offsets -= box_size * np.round(offsets / box_size)

    distances = np.linalg.norm(offsets, axis=1)

    taper = R.smooth_cutoff(distances, inner, outer)

    keep = taper > threshold

    return first[keep], second[keep]


def label_frame(count, bond_first, bond_second):
    if len(bond_first) == 0:
        return np.arange(count)

    graph = coo_matrix(
        (np.ones(len(bond_first)), (bond_first, bond_second)),
        shape=(count, count)
    )

    _, labels = connected_components(graph, directed=False)

    return labels


def formula_for(symbols, members):
    counts = {}

    for member in members:
        symbol = symbols[member]
        counts[symbol] = counts.get(symbol, 0) + 1

    return "".join(
        symbol + (str(counts[symbol]) if counts[symbol] > 1 else "")
        for symbol in ["C", "N", "O", "H"]
        if symbol in counts
    )



def bond_counts(positions, types, box_size, threshold=0.35):
    # How many bonds each atom is actually using, counting a
    # double bond as two.
    #
    # Counting partners instead gets multiple bonds wrong: the
    # carbon in CO2 has two neighbours but four bonds, and would
    # be reported as a radical when it is perfectly satisfied.

    count = len(positions)

    offsets = positions[None, :, :] - positions[:, None, :]
    offsets -= box_size * np.round(offsets / box_size)

    distance_squared = np.sum(offsets ** 2, axis=2)
    np.fill_diagonal(distance_squared, np.inf)

    distances = np.sqrt(distance_squared)

    inner = R.CUTOFF_INNER[np.ix_(types, types)]
    outer = R.CUTOFF_OUTER[np.ix_(types, types)]

    taper = R.smooth_cutoff(distances, inner, outer)
    np.fill_diagonal(taper, 0.0)

    order, _ = R.bond_orders(taper, types)

    counted = np.where(taper > threshold, order, 0.0)

    return np.rint(np.sum(counted, axis=1)).astype(int)


def analyse(recorder, stride=1, threshold=0.35, late_fs=2500.0,
            structures=True, persistent=True):
    symbols = recorder.symbols
    types = recorder.types

    # The last frame's cell, used for the final-state analysis.
    # Individual frames use their own, since the box can change
    # during a run.

    box = recorder.box_at(len(recorder) - 1)

    count = len(symbols)

    first, second = pair_indices(count)
    inner, outer = cutoffs_for(types, first, second)

    times = np.array(recorder.times)

    seen = {}
    heavy_events = []

    previous_heavy = set()
    examined = 0

    # Bonds are judged by how long a pair stays within range
    # rather than by whether it happens to be close in one frame.
    # See bonding.py for why: a distance test cannot tell a bond
    # from a collision once the box is crowded, and reports
    # structures of two hundred atoms that do not exist.

    tracker = (
        bonding.BondTracker(types, threshold=threshold)
        if persistent else None
    )

    # The tracker begins with nothing bonded and needs one
    # formation time before it can confirm anything, so its first
    # frames would report every atom as a separate species. Those
    # frames are counted for the tracker but left out of the
    # census.

    warm_up = (
        bonding.FORMATION_TIME if tracker is not None else 0.0
    )

    census_from = float(times[0]) + warm_up

    # Fingerprints are gathered first and named at the end, so a
    # label never has to change halfway through.

    registry = F.Registry()

    keyed = {}

    for index in range(0, len(recorder), stride):
        positions = recorder.positions[index]

        frame_box = recorder.box_at(index)

        if tracker is not None:
            bond_first, bond_second = tracker.update(
                positions, frame_box, float(times[index])
            )
        else:
            bond_first, bond_second = bonds_in_frame(
                positions, frame_box, first, second,
                inner, outer, threshold
            )

        if float(times[index]) < census_from:
            continue

        labels = label_frame(count, bond_first, bond_second)

        examined += 1

        if structures:
            table, _, _ = F.bond_table(
                positions, types, frame_box, threshold
            )

        present = {}

        for label in np.unique(labels):
            members = np.where(labels == label)[0]

            if structures:
                name = registry.register(symbols, members, table)
            else:
                name = formula_for(symbols, members)

            present[name] = present.get(name, 0) + 1

        for name, number in present.items():
            record = seen.setdefault(
                name,
                {
                    "first_fs": float(times[index]),
                    "last_fs": float(times[index]),
                    "frames": 0,
                    "max_count": 0,
                }
            )

            record["last_fs"] = float(times[index])
            record["frames"] += 1
            record["max_count"] = max(record["max_count"], number)

        # Bonds between heavy atoms build structure. Hydrogen
        # comes and goes constantly and would drown the log.

        heavy_now = set()

        for a, b in zip(bond_first, bond_second):
            if symbols[a] in HEAVY and symbols[b] in HEAVY:
                heavy_now.add((int(a), int(b)))

        for pair in sorted(heavy_now - previous_heavy):
            heavy_events.append((
                float(times[index]), "formed",
                symbols[pair[0]], pair[0],
                symbols[pair[1]], pair[1]
            ))

        for pair in sorted(previous_heavy - heavy_now):
            heavy_events.append((
                float(times[index]), "broke",
                symbols[pair[0]], pair[0],
                symbols[pair[1]], pair[1]
            ))

        previous_heavy = heavy_now

    # ---- the final frame, in detail ----

    positions = recorder.positions[len(recorder) - 1]

    if tracker is not None:
        bond_first, bond_second = tracker.confirmed_now()
    else:
        bond_first, bond_second = bonds_in_frame(
            positions, box, first, second, inner, outer, threshold
        )

    labels = label_frame(count, bond_first, bond_second)

    degree = bond_counts(positions, types, box, threshold)

    if structures:
        final_table, _, _ = F.bond_table(
            positions, types, box, threshold
        )

    final = []

    for label in np.unique(labels):
        members = np.where(labels == label)[0]

        radicals = 0
        over = 0

        for member in members:
            want = EXPECTED_BONDS[symbols[member]]

            if degree[member] < want:
                radicals += 1
            elif degree[member] > want:
                over += 1

        if structures:
            key = registry.register(symbols, members, final_table)
        else:
            key = formula_for(symbols, members)

        final.append({
            "key": key,
            "formula": formula_for(symbols, members),
            "atoms": int(len(members)),
            "members": [int(m) for m in members],
            "radicals": radicals,
            "over_coordinated": over,
            "closed_shell": radicals == 0 and over == 0,
            "heavy": int(
                sum(1 for m in members if symbols[m] in HEAVY)
            ),
            "first_seen_fs": 0.0,
        })

    if structures:
        registry.finalise()

        # Names first, everywhere, before anything looks a species
        # up by name.

        for entry in final:
            entry["structure"] = registry.structure(entry["key"])
            entry["formula"] = registry.name(entry["key"])

        seen = {
            registry.name(key): record
            for key, record in seen.items()
        }

    for entry in final:
        entry["first_seen_fs"] = float(
            seen.get(
                entry["formula"], {"first_fs": times[-1]}
            )["first_fs"]
        )

    final.sort(key=lambda item: (-item["atoms"], item["formula"]))

    # ---- summary numbers for comparing conditions ----
    #
    # Everything before late_fs is shared between a strike run
    # and its matched control, since they start from identical
    # boxes and only diverge when the first discharge lands.
    # Counting from there onward is the only fair comparison.

    late_formed = sum(
        1 for event in heavy_events
        if event[1] == "formed" and event[0] >= late_fs
    )

    late_broke = sum(
        1 for event in heavy_events
        if event[1] == "broke" and event[0] >= late_fs
    )

    # A pair that forms and later breaks, or breaks and reforms,
    # is the box being actively churned rather than settling once.

    history = {}

    for time, kind, sa, ia, sb, ib in heavy_events:
        history.setdefault((ia, ib), []).append(kind)

    turnovers = sum(
        1 for record in history.values() if len(record) > 1
    )

    closed = [
        entry for entry in final
        if entry["closed_shell"] and entry["heavy"] >= 2
    ]

    largest_closed = max(
        (entry["atoms"] for entry in closed), default=0
    )

    largest_closed_heavy = max(
        (entry["heavy"] for entry in closed), default=0
    )

    # The largest thing built, whether or not it closed its shell.
    #
    # Counting only closed-shell molecules turned out to hide the
    # most interesting products. A discharge leaves radicals
    # behind by design, and in a cold box with most of the
    # hydrogen locked up as H2 there is nothing left to cap them
    # with. So a run can assemble a six-carbon skeleton and still
    # report a largest molecule of five atoms, because the
    # skeleton is one hydrogen short of counting.

    any_molecule = [
        entry for entry in final if entry["heavy"] >= 2
    ]

    largest_any = max(
        (entry["atoms"] for entry in any_molecule), default=0
    )

    largest_any_heavy = max(
        (entry["heavy"] for entry in any_molecule), default=0
    )

    # Carbon chains specifically, since chain length is what
    # separates interesting organic chemistry from small molecules.

    most_carbon = max(
        (
            sum(
                1 for index in entry["members"]
                if symbols[index] == "C"
            )
            for entry in final
        ),
        default=0
    )

    # ---- lipid shape ----
    #
    # Whether anything in the box looks like a membrane lipid: a
    # polar group at the end of a run of carbons. This is what
    # would have to exist before the coarse-grained membrane model
    # could be handed a molecule and asked to assemble it.

    if structures:
        lipid = A.scan(final, final_table, symbols)
    else:
        table, _, _ = F.bond_table(
            positions, types, box, threshold
        )

        lipid = A.scan(final, table, symbols)

    # ---- integration health ----
    #
    # Energy can only move between potential and kinetic. If the
    # total jumps upward, it came from nowhere, and the only place
    # it can come from is the integrator.
    #
    # This happens when two atoms get close enough that the Morse
    # repulsion is astronomically steep. Velocity Verlet conserves
    # energy well, but only while the potential is smooth across a
    # timestep, and at very short range it is not. One step lands
    # an atom deep inside the wall and the force there launches it.
    #
    # Measured across several healthy runs the largest honest
    # single-frame rise is about 40 eV, mostly during the opening
    # exotherm. A failure produces something like 1200 eV in one
    # frame. The threshold below sits comfortably between.

    totals = (
        np.array(recorder.potential) + np.array(recorder.kinetic)
    )

    rises = np.diff(totals)

    threshold = max(
        80.0, 0.08 * abs(float(recorder.potential[-1]))
    )

    bad = np.where(rises > threshold)[0]

    energy_jumps = int(len(bad))

    largest_jump = float(rises.max()) if len(rises) else 0.0

    first_jump_fs = (
        float(times[bad[0] + 1]) if len(bad) else 0.0
    )

    return {
        "frames_examined": examined,
        "unconfirmed_pairs": (
            tracker.pending if tracker is not None else 0
        ),
        "final_density": count / (box ** 3),
        "energy_jumps": energy_jumps,
        "largest_energy_jump": round(largest_jump, 1),
        "first_energy_jump_fs": first_jump_fs,
        "stable": energy_jumps == 0,
        "box_changed": recorder.box_changed,
        "box_start": recorder.box_at(0),
        "box_end": box,
        "late_fs": late_fs,
        "late_formed": late_formed,
        "late_broke": late_broke,
        "turnovers": turnovers,
        "largest_closed": largest_closed,
        "largest_closed_heavy": largest_closed_heavy,
        "largest_any": largest_any,
        "largest_any_heavy": largest_any_heavy,
        "most_carbon": most_carbon,
        "species_count": len(seen),
        "best_tail": lipid["best_tail"],
        "best_chain": lipid["best_chain"],
        "amphiphiles": lipid["amphiphiles"],
        "best_amphiphile": lipid["best_formula"],
        "best_head": lipid["best_head"],
        "best_shape": lipid["best_shape"],
        "vesicle_ready": lipid["vesicle_ready"],
        "isomers": registry.isomers() if structures else {},
        "structures": {
            registry.name(key): registry.structure(key)
            for key in registry.details
        } if structures else {},
        "duration_fs": float(times[-1] - times[0]),
        "final": final,
        "seen": seen,
        "heavy_events": heavy_events,
        "temperature": {
            "min": float(np.min(recorder.temperature)),
            "max": float(np.max(recorder.temperature)),
            "final": float(recorder.temperature[-1]),
        },
        "potential": {
            "start": float(recorder.potential[0]),
            "final": float(recorder.potential[-1]),
            "min": float(np.min(recorder.potential)),
        },
    }


def headline(result):
    # One line describing what the run produced, for a list.

    interesting = [
        entry for entry in result["final"]
        if entry["heavy"] >= 2 and entry["closed_shell"]
    ]

    if interesting:
        biggest = max(interesting, key=lambda item: item["atoms"])

        return (
            f"{biggest['formula']} "
            f"({biggest['atoms']} atoms, closed shell)"
        )

    radicals = [
        entry for entry in result["final"] if entry["heavy"] >= 2
    ]

    if radicals:
        biggest = max(radicals, key=lambda item: item["atoms"])

        return f"{biggest['formula']} (radical)"

    return "nothing larger than one heavy atom"


def summary_lines(result):
    lines = []

    lines.append("=" * 60)
    lines.append("  RUN SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        f"  duration      {result['duration_fs']:>10.0f} fs"
        f"      ({result['frames_examined']} frames examined)"
    )
    lines.append(
        f"  temperature   {result['temperature']['min']:>10.0f} - "
        f"{result['temperature']['max']:.0f} K"
        f"   ended {result['temperature']['final']:.0f} K"
    )
    lines.append(
        f"  potential     {result['potential']['start']:>10.1f} -> "
        f"{result['potential']['final']:.1f} eV"
    )

    if not result.get("stable", True):
        lines.append("")
        lines.append("  " + "!" * 56)
        lines.append(
            f"  INTEGRATION FAILURE: total energy jumped "
            f"{result['largest_energy_jump']:.0f} eV in one frame"
        )
        lines.append(
            f"  at {result['first_energy_jump_fs']:.0f} fs"
            f"{' and ' + str(result['energy_jumps'] - 1) + ' other times' if result['energy_jumps'] > 1 else ''}."
        )
        lines.append(
            "  Energy cannot appear from nowhere, so two atoms got"
        )
        lines.append(
            "  too close and the integrator launched them. Anything"
        )
        lines.append(
            "  after that point happened in a box that was briefly"
        )
        lines.append(
            "  far hotter than intended. Exclude this run."
        )
        lines.append("  " + "!" * 56)

    lines.append("")
    lines.append(
        f"  after {result['late_fs']:.0f} fs   "
        f"{result['late_formed']} heavy bonds formed, "
        f"{result['late_broke']} broke, "
        f"{result['turnovers']} pairs changed twice"
    )
    lines.append(
        f"  largest closed shell   "
        f"{result['largest_closed']} atoms "
        f"({result['largest_closed_heavy']} heavy)"
    )
    lines.append(
        f"  largest of any kind    "
        f"{result.get('largest_any', 0)} atoms "
        f"({result.get('largest_any_heavy', 0)} heavy), "
        f"longest carbon count {result.get('most_carbon', 0)}"
    )
    lines.append(
        f"  distinct species seen  {result['species_count']}"
    )

    if result.get("best_chain", 0) >= 2:
        ready = (
            "  <- long enough to form a vesicle"
            if result.get("vesicle_ready") else ""
        )

        lines.append(
            f"  best lipid shape       "
            f"{result.get('best_amphiphile', '')}, "
            f"{result.get('best_chain', 0)}-carbon chain, "
            f"{result.get('best_tail', 0)} clean, "
            f"{result.get('best_shape', '')}, "
            f"head {result.get('best_head', '')}{ready}"
        )

        if result.get("amphiphiles", 0) > 1:
            lines.append(
                f"                         "
                f"{result['amphiphiles']} molecules with a tail of "
                f"{A.MINIMUM_TAIL} or more"
            )

    lines.append("")
    lines.append("")
    lines.append("-" * 60)
    lines.append("  WHAT SURVIVED TO THE END")
    lines.append("-" * 60)
    lines.append("")

    closed = [
        entry for entry in result["final"]
        if entry["heavy"] >= 1 and entry["closed_shell"]
    ]

    open_shell = [
        entry for entry in result["final"]
        if entry["heavy"] >= 1 and not entry["closed_shell"]
    ]

    if closed:
        lines.append("  stable molecules")

        for entry in closed[:16]:
            lines.append(
                f"    {entry['formula']:<12} "
                f"{entry['atoms']:>2d} atoms"
                f"     from {entry['first_seen_fs']:>7.0f} fs"
                f"   {entry.get('structure', '')}"
            )
    else:
        lines.append("  stable molecules:  none")

    lines.append("")

    if open_shell:
        lines.append("  radicals and fragments")

        for entry in open_shell[:16]:
            note = f"{entry['radicals']} radical"

            if entry["radicals"] != 1:
                note += "s"

            if entry["over_coordinated"]:
                note += (
                    f", {entry['over_coordinated']} over-coordinated"
                )

            lines.append(
                f"    {entry['formula']:<12} "
                f"{entry['atoms']:>2d} atoms"
                f"     {note}"
            )

    small = {}

    for entry in result["final"]:
        if entry["heavy"] == 0:
            small[entry["formula"]] = (
                small.get(entry["formula"], 0) + 1
            )

    if small:
        lines.append("")
        lines.append(
            "  small stuff   "
            + ", ".join(
                f"{name} x{number}"
                for name, number in sorted(small.items())
            )
        )

    if result.get("isomers"):
        lines.append("")
        lines.append("  isomers found: " + ", ".join(
            f"{formula} x{number}"
            for formula, number in sorted(result["isomers"].items())
        ))

    lines.append("")
    lines.append("")
    lines.append("-" * 60)
    lines.append("  EVERY SPECIES SEEN AT ANY POINT")
    lines.append("-" * 60)
    lines.append("")
    lines.append(
        f"    {'formula':<12} {'first':>9} {'last':>9} "
        f"{'frames':>8} {'max':>5}"
    )
    lines.append("    " + "-" * 47)

    ordered = sorted(
        result["seen"].items(),
        key=lambda item: (-len(item[0]), -item[1]["frames"])
    )

    for name, record in ordered[:30]:
        lines.append(
            f"    {name:<12} {record['first_fs']:>9.0f} "
            f"{record['last_fs']:>9.0f} "
            f"{record['frames']:>8d} {record['max_count']:>5d}"
        )

    if len(ordered) > 30:
        lines.append(f"    ... and {len(ordered) - 30} more")

    lines.append("")
    lines.append("")
    lines.append("-" * 60)
    lines.append("  HEAVY-ATOM BONDS (C, N, O)")
    lines.append("-" * 60)
    lines.append("")

    events = result["heavy_events"]

    if not events:
        lines.append("    none formed")
    else:
        formed = sum(1 for e in events if e[1] == "formed")
        broke = len(events) - formed

        lines.append(
            f"    {formed} formed, {broke} broke"
        )
        lines.append("")

        for time, kind, sa, ia, sb, ib in events[:50]:
            arrow = "+" if kind == "formed" else "-"

            lines.append(
                f"    {arrow}  {time:>9.0f} fs    "
                f"{sa}{ia}-{sb}{ib}"
            )

        if len(events) > 50:
            lines.append(f"    ... and {len(events) - 50} more")

    return lines