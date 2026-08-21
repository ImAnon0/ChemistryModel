"""Compare current/prototype energies on the frozen dense QM microscopes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from research.heavy_valence_state import HeavyValenceStateEnergyPrototype
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


DEFAULT_GEOMETRIES = Path("research_data/qm_residual/dense_scan_geometries.json")
DEFAULT_QM = Path("research_data/qm_residual/dense_scan_qm.csv")
DEFAULT_OUTPUT = Path(
    "research_data/benchmark/diagnostics/heavy_valence_qm_microscopes.csv"
)
DEFAULT_SUMMARY = Path(
    "research_data/benchmark/diagnostics/heavy_valence_qm_microscopes.json"
)


def _load_qm(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            row["geometry_id"]: float(row["qm_energy_eV"])
            for row in csv.DictReader(handle)
            if row.get("status") == "ok"
        }


def _evaluate(model_class, geometry, box_size, device):
    simulation = model_class(
        boxes=[(
            geometry["symbols"], geometry["coordinates_angstrom"]
        )],
        box_size=box_size,
        target_temperature=0.0,
        friction=0.0,
        device=device,
        dtype=torch.float64,
        random_seed=0,
        relax_on_start=False,
    )
    energy = float(simulation.potential_per_box[0])
    force_max = float(torch.linalg.vector_norm(
        simulation.forces, dim=1
    ).max())
    if not math.isfinite(energy) or not math.isfinite(force_max):
        raise ValueError("non-finite model result")
    return energy, force_max


def _metrics(values):
    return {
        "count": len(values),
        "mae_eV": sum(abs(value) for value in values) / len(values),
        "rmse_eV": math.sqrt(sum(value * value for value in values) / len(values)),
        "max_absolute_eV": max(abs(value) for value in values),
    }


def compare(geometries_path, qm_path, box_size, device):
    geometries = json.loads(
        geometries_path.read_text(encoding="utf-8")
    )["geometries"]
    qm = _load_qm(qm_path)
    missing = [row["geometry_id"] for row in geometries if row["geometry_id"] not in qm]
    if missing:
        raise RuntimeError(f"QM missing {len(missing)} geometries; first={missing[0]}")

    rows = []
    for geometry in geometries:
        current_energy, current_force = _evaluate(
            OptimisedValenceStateBatchedSimulation,
            geometry, box_size, device,
        )
        prototype_energy, prototype_force = _evaluate(
            HeavyValenceStateEnergyPrototype,
            geometry, box_size, device,
        )
        coordinate = geometry.get("reaction_coordinate", {})
        rows.append({
            "geometry_id": geometry["geometry_id"],
            "system": geometry["system"],
            "sample_kind": geometry["sample_kind"],
            "region": geometry["region"],
            "transfer_distance_angstrom": coordinate.get(
                "transfer_distance_angstrom", ""
            ),
            "qm_energy_eV": qm[geometry["geometry_id"]],
            "current_energy_eV": current_energy,
            "prototype_energy_eV": prototype_energy,
            "energy_change_eV": prototype_energy - current_energy,
            "current_force_max_eV_per_angstrom": current_force,
            "prototype_force_max_eV_per_angstrom": prototype_force,
        })

    references = {}
    for row in rows:
        if row["sample_kind"] == "reactant_reference":
            references[row["system"]] = row
    for row in rows:
        reference = references[row["system"]]
        for name in ("qm", "current", "prototype"):
            row[f"{name}_relative_eV"] = (
                row[f"{name}_energy_eV"] - reference[f"{name}_energy_eV"]
            )
        row["current_residual_eV"] = (
            row["qm_relative_eV"] - row["current_relative_eV"]
        )
        row["prototype_residual_eV"] = (
            row["qm_relative_eV"] - row["prototype_relative_eV"]
        )

    grouped = defaultdict(list)
    for row in rows:
        if row["sample_kind"] == "dense_transfer_scan":
            grouped[row["system"]].append(row)

    summary = {
        "evaluated_geometries": len(rows),
        "maximum_absolute_energy_change_eV": max(
            abs(row["energy_change_eV"]) for row in rows
        ),
        "systems": {},
    }
    all_current = []
    all_prototype = []
    for system, system_rows in sorted(grouped.items()):
        current = [row["current_residual_eV"] for row in system_rows]
        prototype = [row["prototype_residual_eV"] for row in system_rows]
        all_current.extend(current)
        all_prototype.extend(prototype)
        summary["systems"][system] = {
            "current": _metrics(current),
            "prototype": _metrics(prototype),
            "maximum_absolute_model_change_eV": max(
                abs(row["energy_change_eV"]) for row in system_rows
            ),
        }
    summary["all_dense_scans"] = {
        "current": _metrics(all_current),
        "prototype": _metrics(all_prototype),
    }
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

    rows, summary = compare(
        args.geometries, args.qm, args.box_size, args.device
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.summary.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
