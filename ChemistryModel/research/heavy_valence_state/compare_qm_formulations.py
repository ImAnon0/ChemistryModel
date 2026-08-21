"""Four-way comparison on the frozen independent QM microscopes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from research.heavy_valence_state.compare_formulations import MODELS
from research.heavy_valence_state.compare_qm_microscopes import _evaluate, _load_qm, _metrics


DEFAULT_GEOMETRIES = Path("research_data/qm_residual/dense_scan_geometries.json")
DEFAULT_QM = Path("research_data/qm_residual/dense_scan_qm.csv")
DEFAULT_OUTPUT = Path(
    "research_data/benchmark/diagnostics/heavy_valence_qm_formulations.csv"
)
DEFAULT_SUMMARY = Path(
    "research_data/benchmark/diagnostics/heavy_valence_qm_formulations.json"
)


def compare(geometries_path, qm_path, box_size, device):
    geometries = json.loads(geometries_path.read_text(encoding="utf-8"))["geometries"]
    qm = _load_qm(qm_path)
    rows = []
    for geometry in geometries:
        gid = geometry["geometry_id"]
        if gid not in qm:
            raise RuntimeError(f"QM missing {gid}")
        coordinate = geometry.get("reaction_coordinate", {})
        row = {
            "geometry_id": gid,
            "system": geometry["system"],
            "sample_kind": geometry["sample_kind"],
            "region": geometry["region"],
            "transfer_distance_angstrom": coordinate.get("transfer_distance_angstrom", ""),
            "qm_energy_eV": qm[gid],
        }
        for name, model in MODELS.items():
            energy, force = _evaluate(model, geometry, box_size, device)
            row[f"{name}_energy_eV"] = energy
            row[f"{name}_force_max_eV_per_angstrom"] = force
        rows.append(row)

    references = {
        row["system"]: row for row in rows
        if row["sample_kind"] == "reactant_reference"
    }
    for row in rows:
        reference = references[row["system"]]
        row["qm_relative_eV"] = row["qm_energy_eV"] - reference["qm_energy_eV"]
        for name in MODELS:
            row[f"{name}_relative_eV"] = (
                row[f"{name}_energy_eV"] - reference[f"{name}_energy_eV"]
            )
            row[f"{name}_residual_eV"] = (
                row["qm_relative_eV"] - row[f"{name}_relative_eV"]
            )

    dense = defaultdict(list)
    for row in rows:
        if row["sample_kind"] == "dense_transfer_scan":
            dense[row["system"]].append(row)
    summary = {"evaluated_geometries": len(rows), "systems": {}, "all_dense": {}}
    for system, system_rows in sorted(dense.items()):
        summary["systems"][system] = {}
        ordered = sorted(
            system_rows,
            key=lambda row: float(row["transfer_distance_angstrom"]),
        )
        for name in MODELS:
            residuals = [row[f"{name}_residual_eV"] for row in ordered]
            steps = [right - left for left, right in zip(residuals, residuals[1:])]
            summary["systems"][system][name] = {
                **_metrics(residuals),
                "worst_adjacent_residual_step_eV": max(
                    (abs(value) for value in steps), default=0.0
                ),
            }
    for name in MODELS:
        values = [
            row[f"{name}_residual_eV"] for row in rows
            if row["sample_kind"] == "dense_transfer_scan"
        ]
        summary["all_dense"][name] = _metrics(values)
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometries", type=Path, default=DEFAULT_GEOMETRIES)
    parser.add_argument("--qm", type=Path, default=DEFAULT_QM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--box-size", type=float, default=30.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    rows, summary = compare(args.geometries, args.qm, args.box_size, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
