"""Decompose the largest residual jumps into ChemistryModel energy components.

Reads:
    research_data/qm_residual/qm_residual_dataset.csv
    research_data/qm_residual/base_results.csv

For adjacent structured-grid points, prints:
    dBase total
    dBase bond
    dBase overcoordination
    dBase angle
    dQM
    dResidual

This identifies which ChemistryModel term is responsible for the sharp
reactive-region mismatch before changing the force field or training ML.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_DATASET = Path("research_data/qm_residual/qm_residual_dataset.csv")
DEFAULT_BASE = Path("research_data/qm_residual/base_results.csv")


def load_csv(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["geometry_id"]] = row
    return rows


def f(row, key):
    value = row.get(key, "")
    return None if value == "" else float(value)


def adjacent_pairs(dataset: dict[str, dict]):
    grid = [r for r in dataset.values() if r["sample_kind"] == "grid"]
    by_system = defaultdict(list)
    for row in grid:
        by_system[row["system"]].append(row)

    pairs = []
    for system, group in by_system.items():
        coords = {
            (f(row, "donor_distance_angstrom"),
             f(row, "transfer_distance_angstrom")): row
            for row in group
        }
        donors = sorted({d for d, _ in coords})
        transfers = sorted({t for _, t in coords})

        for donor in donors:
            line = [
                coords[(donor, transfer)]
                for transfer in transfers
                if (donor, transfer) in coords
            ]
            line.sort(key=lambda r: f(r, "transfer_distance_angstrom"))
            for left, right in zip(line, line[1:]):
                pairs.append(("transfer", left, right))

        for transfer in transfers:
            line = [
                coords[(donor, transfer)]
                for donor in donors
                if (donor, transfer) in coords
            ]
            line.sort(key=lambda r: f(r, "donor_distance_angstrom"))
            for left, right in zip(line, line[1:]):
                pairs.append(("donor", left, right))

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--top", type=int, default=16)
    args = parser.parse_args()

    dataset = load_csv(args.dataset)
    base = load_csv(args.base)

    ranked = []
    for axis, left, right in adjacent_pairs(dataset):
        gid_l = left["geometry_id"]
        gid_r = right["geometry_id"]

        d_base = f(right, "base_relative_eV") - f(left, "base_relative_eV")
        d_qm = f(right, "qm_relative_eV") - f(left, "qm_relative_eV")
        d_res = f(right, "residual_target_eV") - f(left, "residual_target_eV")

        b_l = base[gid_l]
        b_r = base[gid_r]

        d_bond = f(b_r, "base_bond_energy_eV") - f(b_l, "base_bond_energy_eV")
        d_over = f(b_r, "base_overcoord_energy_eV") - f(b_l, "base_overcoord_energy_eV")
        d_angle = f(b_r, "base_angle_energy_eV") - f(b_l, "base_angle_energy_eV")

        ranked.append({
            "system": left["system"],
            "axis": axis,
            "left": gid_l,
            "right": gid_r,
            "d_base": d_base,
            "d_qm": d_qm,
            "d_res": d_res,
            "d_bond": d_bond,
            "d_over": d_over,
            "d_angle": d_angle,
            "audit": d_bond + d_over + d_angle - d_base,
        })

    ranked.sort(key=lambda x: abs(x["d_res"]), reverse=True)

    print("LARGEST RESIDUAL JUMPS — BASE COMPONENT DECOMPOSITION")
    print()
    for i, row in enumerate(ranked[:args.top], 1):
        print(
            f"{i:2d}. {row['system']:12s} {row['axis']:8s} "
            f"{row['left']} -> {row['right']}"
        )
        print(f"    dQM       : {row['d_qm']:+.6f} eV")
        print(f"    dBase     : {row['d_base']:+.6f} eV")
        print(f"      bond    : {row['d_bond']:+.6f} eV")
        print(f"      over    : {row['d_over']:+.6f} eV")
        print(f"      angle   : {row['d_angle']:+.6f} eV")
        print(f"      audit   : {row['audit']:+.3e} eV")
        print(f"    dResidual : {row['d_res']:+.6f} eV")
        print()


if __name__ == "__main__":
    main()
