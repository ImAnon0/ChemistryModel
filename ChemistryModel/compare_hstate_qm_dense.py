"""Compare H-state models against the dense QM microscope scan.

Run this on the `sapt-h-state` branch with the Windows Python that has Torch:

    py compare_hstate_qm_dense.py

Required files:
    research_data/qm_residual/dense_scan_geometries.json
    research_data/qm_residual/dense_scan_qm.csv

Output:
    research_data/qm_residual/dense_scan_hstate_compare.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from h_state_torch import HStateReferenceBatchedSimulation
from sapt_h_state_torch import SaptHStateBatchedSimulation


DEFAULT_GEOMETRIES = Path("research_data/qm_residual/dense_scan_geometries.json")
DEFAULT_QM = Path("research_data/qm_residual/dense_scan_qm.csv")
DEFAULT_OUTPUT = Path("research_data/qm_residual/dense_scan_hstate_compare.csv")
DEFAULT_BOX_SIZE = 30.0


def load_qm(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                rows[row["geometry_id"]] = row
    return rows


def evaluate(model_class, symbols, coordinates, box_size: float):
    simulation = model_class(
        boxes=[(symbols, coordinates)],
        box_size=box_size,
        target_temperature=0.0,
        friction=0.0,
        device="cpu",
        dtype=torch.float64,
        random_seed=0,
        relax_on_start=False,
    )
    energy = float(simulation.potential_per_box[0])
    forces = simulation.forces.detach().cpu().to(torch.float64)
    force_max = float(torch.max(torch.linalg.norm(forces, dim=1)))
    if not math.isfinite(energy) or not math.isfinite(force_max):
        raise RuntimeError("non-finite H-state result")
    return energy, force_max


def rmse(values):
    return math.sqrt(sum(v * v for v in values) / len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometries", type=Path, default=DEFAULT_GEOMETRIES)
    parser.add_argument("--qm", type=Path, default=DEFAULT_QM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--box-size", type=float, default=DEFAULT_BOX_SIZE)
    args = parser.parse_args()

    payload = json.loads(args.geometries.read_text(encoding="utf-8"))
    geometries = payload["geometries"]
    qm = load_qm(args.qm)

    missing = [g["geometry_id"] for g in geometries if g["geometry_id"] not in qm]
    if missing:
        raise RuntimeError(f"QM missing {len(missing)} rows; first={missing[0]}")

    raw = []
    total = len(geometries)

    print(f"geometries : {total}")
    print("models     : H-state common-core, SAPT H-state")
    print("device     : cpu / float64")
    print()

    for index, geometry in enumerate(geometries, start=1):
        gid = geometry["geometry_id"]
        symbols = geometry["symbols"]
        coordinates = geometry["coordinates_angstrom"]

        h_energy, h_force = evaluate(
            HStateReferenceBatchedSimulation, symbols, coordinates, args.box_size
        )
        s_energy, s_force = evaluate(
            SaptHStateBatchedSimulation, symbols, coordinates, args.box_size
        )
        qm_energy = float(qm[gid]["qm_energy_eV"])

        rc = geometry.get("reaction_coordinate", {})
        raw.append({
            "geometry_id": gid,
            "system": geometry["system"],
            "sample_kind": geometry["sample_kind"],
            "region": geometry["region"],
            "transfer_distance_angstrom": rc.get("transfer_distance_angstrom", ""),
            "donor_distance_angstrom": rc.get("donor_distance_angstrom", ""),
            "qm_energy_eV": qm_energy,
            "hstate_energy_eV": h_energy,
            "sapt_hstate_energy_eV": s_energy,
            "hstate_force_max_eV_per_angstrom": h_force,
            "sapt_hstate_force_max_eV_per_angstrom": s_force,
        })

        if index == 1 or index % 20 == 0 or index == total:
            print(
                f"[{index:3d}/{total:3d}] {gid:<38s} "
                f"H={h_energy:+.6f}  SAPT={s_energy:+.6f} eV"
            )

    refs = {}
    for row in raw:
        if row["sample_kind"] == "reactant_reference":
            refs[row["system"]] = {
                "qm": row["qm_energy_eV"],
                "hstate": row["hstate_energy_eV"],
                "sapt": row["sapt_hstate_energy_eV"],
            }

    final = []
    for row in raw:
        ref = refs[row["system"]]
        qm_rel = row["qm_energy_eV"] - ref["qm"]
        h_rel = row["hstate_energy_eV"] - ref["hstate"]
        s_rel = row["sapt_hstate_energy_eV"] - ref["sapt"]
        merged = dict(row)
        merged.update({
            "qm_relative_eV": qm_rel,
            "hstate_relative_eV": h_rel,
            "sapt_hstate_relative_eV": s_rel,
            "hstate_residual_eV": qm_rel - h_rel,
            "sapt_hstate_residual_eV": qm_rel - s_rel,
        })
        final.append(merged)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final[0].keys()))
        writer.writeheader()
        writer.writerows(final)

    print()
    print(f"wrote      : {args.output}")
    print()
    print("DENSE TRANSFER RESIDUALS VS QM")
    print(
        f"{'system':14s} {'H MAE':>9s} {'H RMSE':>9s} {'H max':>9s} "
        f"{'SAPT MAE':>10s} {'SAPT RMSE':>10s} {'SAPT max':>10s}"
    )

    dense = defaultdict(list)
    for row in final:
        if row["sample_kind"] == "dense_transfer_scan":
            dense[row["system"]].append(row)

    for system in sorted(dense):
        rows = dense[system]
        h = [r["hstate_residual_eV"] for r in rows]
        s = [r["sapt_hstate_residual_eV"] for r in rows]
        print(
            f"{system:14s} "
            f"{statistics.fmean(abs(v) for v in h):9.4f} "
            f"{rmse(h):9.4f} "
            f"{max(abs(v) for v in h):9.4f} "
            f"{statistics.fmean(abs(v) for v in s):10.4f} "
            f"{rmse(s):10.4f} "
            f"{max(abs(v) for v in s):10.4f}"
        )

    print()
    print("WORST ADJACENT 0.02 A RESIDUAL STEP")
    for system in sorted(dense):
        rows = sorted(dense[system], key=lambda r: float(r["transfer_distance_angstrom"]))
        for column, label in (
            ("hstate_residual_eV", "H-state"),
            ("sapt_hstate_residual_eV", "SAPT"),
        ):
            pairs = []
            for left, right in zip(rows, rows[1:]):
                jump = right[column] - left[column]
                pairs.append((abs(jump), jump, left, right))
            _, jump, left, right = max(pairs, key=lambda x: x[0])
            print(
                f"{system:14s} {label:8s} "
                f"{float(left['transfer_distance_angstrom']):.3f}->"
                f"{float(right['transfer_distance_angstrom']):.3f} A  "
                f"dResidual={jump:+.6f} eV"
            )

    print()
    print("PRODUCT REFERENCE ENERGIES")
    print(f"{'system':14s} {'QM':>9s} {'H-state':>9s} {'SAPT':>9s}")
    for system in sorted(refs):
        products = [
            r for r in final
            if r["system"] == system and r["sample_kind"] == "product_reference"
        ]
        if len(products) == 1:
            row = products[0]
            print(
                f"{system:14s} "
                f"{row['qm_relative_eV']:+9.4f} "
                f"{row['hstate_relative_eV']:+9.4f} "
                f"{row['sapt_hstate_relative_eV']:+9.4f}"
            )


if __name__ == "__main__":
    main()
