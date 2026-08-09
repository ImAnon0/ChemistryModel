import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import reactive as R


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


def analyse(recorder, stride=1, threshold=0.35):
    symbols = recorder.symbols
    types = recorder.types
    box = recorder.box_size

    count = len(symbols)

    first, second = pair_indices(count)
    inner, outer = cutoffs_for(types, first, second)

    times = np.array(recorder.times)

    seen = {}
    heavy_events = []

    previous_heavy = set()
    examined = 0

    for index in range(0, len(recorder), stride):
        positions = recorder.positions[index]

        bond_first, bond_second = bonds_in_frame(
            positions, box, first, second, inner, outer, threshold
        )

        labels = label_frame(count, bond_first, bond_second)

        examined += 1

        present = {}

        for label in np.unique(labels):
            members = np.where(labels == label)[0]

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

    bond_first, bond_second = bonds_in_frame(
        positions, box, first, second, inner, outer, threshold
    )

    labels = label_frame(count, bond_first, bond_second)

    degree = bond_counts(positions, types, box, threshold)

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

        name = formula_for(symbols, members)

        final.append({
            "formula": name,
            "atoms": int(len(members)),
            "members": [int(m) for m in members],
            "radicals": radicals,
            "over_coordinated": over,
            "closed_shell": radicals == 0 and over == 0,
            "heavy": int(
                sum(1 for m in members if symbols[m] in HEAVY)
            ),
            "first_seen_fs": float(
                seen.get(name, {"first_fs": times[-1]})["first_fs"]
            ),
        })

    final.sort(key=lambda item: (-item["atoms"], item["formula"]))

    return {
        "frames_examined": examined,
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
                f"     formed at {entry['first_seen_fs']:>7.0f} fs"
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