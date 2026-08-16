"""Audit ChemistryModel static evaluations before spending QM compute."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

DEFAULT_INPUT = Path("research_data/qm_residual/base_results.csv")


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["base_energy_eV"] = float(row["base_energy_eV"])
            row["base_force_max_eV_per_angstrom"] = float(
                row["base_force_max_eV_per_angstrom"]
            )
            donor = row.get("donor_distance_angstrom", "")
            transfer = row.get("transfer_distance_angstrom", "")
            row["donor_distance_angstrom"] = None if donor == "" else float(donor)
            row["transfer_distance_angstrom"] = None if transfer == "" else float(transfer)
            rows.append(row)
    return rows


def fmt_coord(value):
    return "-" if value is None else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = read_rows(args.input)
    if not rows:
        raise SystemExit("no rows found")

    print(f"rows: {len(rows)}")
    print()

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["system"]].append(row)

    print("PER SYSTEM")
    print(
        f"{'system':14s} {'N':>4s} "
        f"{'E min':>11s} {'E max':>11s} "
        f"{'Fmax med':>11s} {'Fmax max':>11s}"
    )
    for system in sorted(grouped):
        group = grouped[system]
        energies = sorted(row["base_energy_eV"] for row in group)
        forces = sorted(row["base_force_max_eV_per_angstrom"] for row in group)
        middle = len(forces) // 2
        median = (
            forces[middle]
            if len(forces) % 2
            else 0.5 * (forces[middle - 1] + forces[middle])
        )
        print(
            f"{system:14s} {len(group):4d} "
            f"{energies[0]:+11.4f} {energies[-1]:+11.4f} "
            f"{median:11.3f} {forces[-1]:11.3f}"
        )

    print()
    print("FORCE THRESHOLDS")
    for threshold in (10.0, 20.0, 30.0, 50.0, 100.0):
        selected = [
            row for row in rows
            if row["base_force_max_eV_per_angstrom"] > threshold
        ]
        by_system = defaultdict(int)
        for row in selected:
            by_system[row["system"]] += 1
        detail = ", ".join(
            f"{name}={by_system[name]}" for name in sorted(by_system)
        ) or "none"
        print(
            f"> {threshold:5.1f} eV/A : "
            f"{len(selected):3d}/{len(rows)}  {detail}"
        )

    print()
    print(f"TOP {args.top} STATIC FORCES")
    print(
        f"{'#':>3s} {'geometry_id':34s} {'E/eV':>10s} "
        f"{'Fmax/eV/A':>12s} {'donor/A':>9s} {'transfer/A':>11s}"
    )
    ranked = sorted(
        rows,
        key=lambda row: row["base_force_max_eV_per_angstrom"],
        reverse=True,
    )
    for index, row in enumerate(ranked[: args.top], start=1):
        print(
            f"{index:3d} {row['geometry_id']:34s} "
            f"{row['base_energy_eV']:+10.4f} "
            f"{row['base_force_max_eV_per_angstrom']:12.3f} "
            f"{fmt_coord(row['donor_distance_angstrom']):>9s} "
            f"{fmt_coord(row['transfer_distance_angstrom']):>11s}"
        )


if __name__ == "__main__":
    main()
