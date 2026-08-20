from pathlib import Path
import csv
from collections import defaultdict


INPUT = Path("research_data/benchmark/grambow_scores.json")


def mean(values):
    return sum(values) / len(values) if values else 0.0


def load_scores(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def elements_from_row(row):
    # We only have atom count here currently.
    # Later we can join geometry data for exact elements.
    return None


def main():
    rows = load_scores(INPUT)

    print(f"Loaded {len(rows)} benchmark reactions\n")

    # Worst failures
    print("=" * 80)
    print("WORST BARRIER ERRORS")
    print("=" * 80)

    worst_barrier = sorted(
        rows,
        key=lambda r: abs(float(r["barrier_error_eV"])),
        reverse=True
    )

    for r in worst_barrier[:20]:
        print(
            f"{r['reaction_id']:10s} "
            f"atoms={r['atom_count']:>2s} "
            f"model={float(r['model_barrier_eV']):8.2f} "
            f"ref={float(r['reference_barrier_eV']):8.2f} "
            f"error={float(r['barrier_error_eV']):8.2f}"
        )

    print()

    print("=" * 80)
    print("WORST REACTION ENERGY ERRORS")
    print("=" * 80)

    worst_reaction = sorted(
        rows,
        key=lambda r: abs(float(r["reaction_error_eV"])),
        reverse=True
    )

    for r in worst_reaction[:20]:
        print(
            f"{r['reaction_id']:10s} "
            f"atoms={r['atom_count']:>2s} "
            f"model={float(r['model_reaction_eV']):8.2f} "
            f"ref={float(r['reference_reaction_eV']):8.2f} "
            f"error={float(r['reaction_error_eV']):8.2f}"
        )


    print()

    # Atom count grouping
    print("=" * 80)
    print("ERROR BY ATOM COUNT")
    print("=" * 80)

    groups = defaultdict(list)

    for r in rows:
        groups[int(r["atom_count"])].append(
            abs(float(r["barrier_error_eV"]))
        )

    for atoms in sorted(groups):
        print(
            f"{atoms:2d} atoms: "
            f"n={len(groups[atoms]):3d} "
            f"MAE={mean(groups[atoms]):.3f} eV"
        )


    print()

    # Catastrophic collapse detector
    print("=" * 80)
    print("ENERGY COLLAPSE CASES (< -10 eV error)")
    print("=" * 80)

    for r in rows:
        barrier = float(r["barrier_error_eV"])
        reaction = float(r["reaction_error_eV"])

        if barrier < -10 or reaction < -10:
            print(
                f"{r['reaction_id']:10s} "
                f"barrier={barrier:8.2f} "
                f"reaction={reaction:8.2f}"
            )


if __name__ == "__main__":
    main()