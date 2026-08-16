"""Analyze the dense QM/base reactive-taper microscope scan.

Inputs:
    research_data/qm_residual/dense_scan_geometries.json
    research_data/qm_residual/dense_scan_base.csv
    research_data/qm_residual/dense_scan_qm.csv

Outputs:
    research_data/qm_residual/dense_scan_analysis.csv

For each scan point it records relative QM/base energies and ChemistryModel's
bond/overcoordination/angle components.  It then prints the steepest adjacent
steps and the values nearest the taper boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_GEOMETRIES = Path(
    "research_data/qm_residual/dense_scan_geometries.json"
)
DEFAULT_BASE = Path(
    "research_data/qm_residual/dense_scan_base.csv"
)
DEFAULT_QM = Path(
    "research_data/qm_residual/dense_scan_qm.csv"
)
DEFAULT_OUTPUT = Path(
    "research_data/qm_residual/dense_scan_analysis.csv"
)


def load_csv(path: Path):
    rows = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["geometry_id"]] = row
    return rows


def num(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{row['geometry_id']}: non-finite {key}")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometries", type=Path, default=DEFAULT_GEOMETRIES)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--qm", type=Path, default=DEFAULT_QM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = json.loads(args.geometries.read_text(encoding="utf-8"))
    base = load_csv(args.base)
    qm = load_csv(args.qm)

    refs = {}
    for geometry in payload["geometries"]:
        if geometry["sample_kind"] == "reactant_reference":
            gid = geometry["geometry_id"]
            refs[geometry["system"]] = {
                "base": num(base[gid], "base_energy_eV"),
                "qm": num(qm[gid], "qm_energy_eV"),
            }

    output_rows = []

    for geometry in payload["geometries"]:
        if geometry["sample_kind"] != "dense_transfer_scan":
            continue

        gid = geometry["geometry_id"]
        system = geometry["system"]
        base_row = base[gid]
        qm_row = qm[gid]
        ref = refs[system]

        base_total = num(base_row, "base_energy_eV")
        qm_total = num(qm_row, "qm_energy_eV")

        scan = geometry["dense_scan"]
        rc = geometry["reaction_coordinate"]

        output_rows.append({
            "geometry_id": gid,
            "system": system,
            "transfer_distance_angstrom": rc[
                "transfer_distance_angstrom"
            ],
            "fixed_donor_distance_angstrom": rc[
                "donor_distance_angstrom"
            ],
            "taper_inner_angstrom": scan["taper_inner_angstrom"],
            "taper_outer_angstrom": scan["taper_outer_angstrom"],
            "base_relative_eV": base_total - ref["base"],
            "qm_relative_eV": qm_total - ref["qm"],
            "residual_target_eV": (
                (qm_total - ref["qm"])
                - (base_total - ref["base"])
            ),
            "base_bond_energy_eV": num(
                base_row, "base_bond_energy_eV"
            ),
            "base_overcoord_energy_eV": num(
                base_row, "base_overcoord_energy_eV"
            ),
            "base_angle_energy_eV": num(
                base_row, "base_angle_energy_eV"
            ),
            "base_force_max_eV_per_angstrom": num(
                base_row, "base_force_max_eV_per_angstrom"
            ),
        })

    fieldnames = list(output_rows[0].keys())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"wrote : {args.output}")
    print(f"rows  : {len(output_rows)}")
    print("")

    systems = sorted({row["system"] for row in output_rows})
    for system in systems:
        rows = sorted(
            [row for row in output_rows if row["system"] == system],
            key=lambda row: row["transfer_distance_angstrom"],
        )

        steps = []
        for left, right in zip(rows, rows[1:]):
            dx = (
                right["transfer_distance_angstrom"]
                - left["transfer_distance_angstrom"]
            )
            d_base = (
                right["base_relative_eV"]
                - left["base_relative_eV"]
            )
            d_qm = (
                right["qm_relative_eV"]
                - left["qm_relative_eV"]
            )
            d_res = (
                right["residual_target_eV"]
                - left["residual_target_eV"]
            )
            d_bond = (
                right["base_bond_energy_eV"]
                - left["base_bond_energy_eV"]
            )
            d_over = (
                right["base_overcoord_energy_eV"]
                - left["base_overcoord_energy_eV"]
            )
            d_angle = (
                right["base_angle_energy_eV"]
                - left["base_angle_energy_eV"]
            )
            steps.append({
                "x0": left["transfer_distance_angstrom"],
                "x1": right["transfer_distance_angstrom"],
                "dx": dx,
                "d_base": d_base,
                "d_qm": d_qm,
                "d_res": d_res,
                "d_bond": d_bond,
                "d_over": d_over,
                "d_angle": d_angle,
            })

        worst = max(steps, key=lambda s: abs(s["d_res"]))
        inner = rows[0]["taper_inner_angstrom"]
        outer = rows[0]["taper_outer_angstrom"]
        near_inner = min(
            rows,
            key=lambda row: abs(
                row["transfer_distance_angstrom"] - inner
            ),
        )
        near_outer = min(
            rows,
            key=lambda row: abs(
                row["transfer_distance_angstrom"] - outer
            ),
        )

        print(system.upper())
        print(
            f"  worst residual step : "
            f"{worst['x0']:.3f}->{worst['x1']:.3f} A"
        )
        print(
            f"    dQM={worst['d_qm']:+.6f}  "
            f"dBase={worst['d_base']:+.6f}  "
            f"dResidual={worst['d_res']:+.6f} eV"
        )
        print(
            f"    base components: "
            f"bond={worst['d_bond']:+.6f}  "
            f"over={worst['d_over']:+.6f}  "
            f"angle={worst['d_angle']:+.6f} eV"
        )
        print(
            f"  near inner taper {inner:.4f} A : "
            f"x={near_inner['transfer_distance_angstrom']:.3f}  "
            f"res={near_inner['residual_target_eV']:+.6f} eV"
        )
        print(
            f"  near outer taper {outer:.4f} A : "
            f"x={near_outer['transfer_distance_angstrom']:.3f}  "
            f"res={near_outer['residual_target_eV']:+.6f} eV"
        )
        print(
            f"  residual range      : "
            f"{min(r['residual_target_eV'] for r in rows):+.6f} .. "
            f"{max(r['residual_target_eV'] for r in rows):+.6f} eV"
        )
        print()


if __name__ == "__main__":
    main()
