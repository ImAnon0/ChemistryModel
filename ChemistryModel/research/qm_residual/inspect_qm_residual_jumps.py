"""Inspect the largest adjacent jumps in the QM-residual dataset.

This is a diagnostic before any ML.  It decomposes each residual jump into:

    dResidual = dQM - dBase

and flags whether the step crosses one of ChemistryModel's current
distance-taper boundaries for the active bond type.

Run in the normal/base environment:

    python inspect_qm_residual_jumps.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT = Path("research_data/qm_residual/qm_residual_dataset.csv")

# Current ChemistryModel single-bond equilibrium lengths used by the
# reactive taper.  The base engine's taper is:
#     inner = 1.25 * r_e
#     outer = 1.60 * r_e
RE = {
    "H-H": 0.74144,
    "C-H": 1.086,
    "O-H": 0.960,
}

SYSTEM_AXIS_BOND = {
    "h3": {"donor": "H-H", "transfer": "H-H"},
    "formaldehyde": {"donor": "C-H", "transfer": "H-H"},
    "methane": {"donor": "C-H", "transfer": "H-H"},
    "water": {"donor": "O-H", "transfer": "O-H"},
}


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in (
                "donor_distance_angstrom",
                "transfer_distance_angstrom",
                "base_relative_eV",
                "qm_relative_eV",
                "residual_target_eV",
                "base_force_max_eV_per_angstrom",
            ):
                value = row.get(key, "")
                row[key] = None if value == "" else float(value)
            rows.append(row)
    return rows


def crossed(value_a: float, value_b: float, boundary: float) -> bool:
    lo = min(value_a, value_b)
    hi = max(value_a, value_b)
    return lo < boundary <= hi


def taper_note(system: str, axis: str, a: float, b: float) -> str:
    bond = SYSTEM_AXIS_BOND[system][axis]
    re = RE[bond]
    inner = 1.25 * re
    outer = 1.60 * re

    flags = []
    if crossed(a, b, inner):
        flags.append(f"crosses {bond} inner={inner:.4f} A")
    if crossed(a, b, outer):
        flags.append(f"crosses {bond} outer={outer:.4f} A")

    if flags:
        return "; ".join(flags)

    # Helpful when a pair lives inside the taper interval even if this
    # particular discrete step does not cross its exact endpoint.
    midpoint = 0.5 * (a + b)
    if inner < midpoint < outer:
        return f"inside {bond} taper [{inner:.4f}, {outer:.4f}] A"

    return ""


def adjacent_pairs(rows: list[dict]) -> list[dict]:
    grid = [r for r in rows if r["sample_kind"] == "grid"]
    by_system = defaultdict(list)
    for row in grid:
        by_system[row["system"]].append(row)

    result = []

    for system, group in by_system.items():
        coords = {
            (
                row["donor_distance_angstrom"],
                row["transfer_distance_angstrom"],
            ): row
            for row in group
        }

        donors = sorted({d for d, _ in coords})
        transfers = sorted({t for _, t in coords})

        for axis in ("transfer", "donor"):
            if axis == "transfer":
                for donor in donors:
                    line = [
                        coords[(donor, transfer)]
                        for transfer in transfers
                        if (donor, transfer) in coords
                    ]
                    line.sort(key=lambda r: r["transfer_distance_angstrom"])
                    _add_line(result, system, axis, line)
            else:
                for transfer in transfers:
                    line = [
                        coords[(donor, transfer)]
                        for donor in donors
                        if (donor, transfer) in coords
                    ]
                    line.sort(key=lambda r: r["donor_distance_angstrom"])
                    _add_line(result, system, axis, line)

    return result


def _add_line(result, system, axis, line):
    coord_key = (
        "transfer_distance_angstrom"
        if axis == "transfer"
        else "donor_distance_angstrom"
    )

    for left, right in zip(line, line[1:]):
        coord_a = left[coord_key]
        coord_b = right[coord_key]

        d_base = right["base_relative_eV"] - left["base_relative_eV"]
        d_qm = right["qm_relative_eV"] - left["qm_relative_eV"]
        d_res = right["residual_target_eV"] - left["residual_target_eV"]

        result.append({
            "system": system,
            "axis": axis,
            "from": left,
            "to": right,
            "coord_a": coord_a,
            "coord_b": coord_b,
            "d_base": d_base,
            "d_qm": d_qm,
            "d_res": d_res,
            "note": taper_note(system, axis, coord_a, coord_b),
        })


def print_pair(rank: int, pair: dict):
    left = pair["from"]
    right = pair["to"]

    print(
        f"{rank:2d}. {pair['system']:12s} {pair['axis']:8s} "
        f"{left['geometry_id']} -> {right['geometry_id']}"
    )
    print(
        f"    coordinate : {pair['coord_a']:.6f} -> "
        f"{pair['coord_b']:.6f} A"
    )
    print(
        f"    dBase      : {pair['d_base']:+.6f} eV"
    )
    print(
        f"    dQM        : {pair['d_qm']:+.6f} eV"
    )
    print(
        f"    dResidual  : {pair['d_res']:+.6f} eV "
        f"(dQM - dBase = "
        f"{pair['d_qm'] - pair['d_base']:+.6f})"
    )
    print(
        f"    base Fmax  : "
        f"{left['base_force_max_eV_per_angstrom']:.3f} -> "
        f"{right['base_force_max_eV_per_angstrom']:.3f} eV/A"
    )
    if pair["note"]:
        print(f"    taper      : {pair['note']}")
    print()


def region_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["system"], row["region"])].append(
            row["residual_target_eV"]
        )

    print("RESIDUAL BY REGION")
    print(
        f"{'system':14s} {'region':20s} {'N':>4s} "
        f"{'MAE/eV':>10s} {'min/eV':>10s} {'max/eV':>10s}"
    )
    for (system, region), values in sorted(grouped.items()):
        mae = sum(abs(v) for v in values) / len(values)
        print(
            f"{system:14s} {region:20s} {len(values):4d} "
            f"{mae:10.4f} {min(values):+10.4f} {max(values):+10.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--top", type=int, default=16)
    args = parser.parse_args()

    rows = read_rows(args.input)
    pairs = adjacent_pairs(rows)

    ranked = sorted(
        pairs,
        key=lambda p: abs(p["d_res"]),
        reverse=True,
    )

    print("LARGEST ADJACENT RESIDUAL JUMPS")
    print()
    for rank, pair in enumerate(ranked[: args.top], start=1):
        print_pair(rank, pair)

    print()
    region_summary(rows)


if __name__ == "__main__":
    main()
