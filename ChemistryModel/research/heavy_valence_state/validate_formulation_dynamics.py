"""Matched NVE checks for all heavy-valence research formulations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from research.heavy_valence_state.compare_formulations import MODELS


DEFAULT_QM = Path("research_data/qm_residual/dense_scan_geometries.json")
DEFAULT_GRAMBOW = Path("research_data/benchmark/grambow_endpoints.json")
DEFAULT_OUTPUT = Path(
    "research_data/benchmark/diagnostics/heavy_valence_formulation_nve.json"
)


def _centred(symbols, coordinates):
    positions = np.asarray(coordinates, dtype=float)
    positions -= positions.mean(axis=0)
    positions += 20.0
    return list(symbols), positions


def _cases(qm_path, grambow_path):
    qm = json.loads(qm_path.read_text(encoding="utf-8"))["geometries"]
    water = min(
        (row for row in qm if row["system"] == "water" and row["sample_kind"] == "dense_transfer_scan"),
        key=lambda row: abs(float(row["reaction_coordinate"]["transfer_distance_angstrom"]) - 1.16),
    )
    grambow = json.loads(grambow_path.read_text(encoding="utf-8"))["geometries"]
    crowded = next(
        row for row in grambow
        if row["reaction_id"] == "rxn000105" and row["region"] == "reactant"
    )
    centre = np.array([20.0, 20.0, 20.0])
    directions = np.asarray([
        [0.0, 0.0, 1.7], [0.0, 0.0, -1.7], [1.7, 0.0, 0.0],
        [-0.85, 1.472243186, 0.0], [-1.0, -1.732050808, 0.0],
    ])
    return {
        "water_transfer": {
            "geometry": _centred(water["symbols"], water["coordinates_angstrom"]),
            "dt": 0.25, "temperature": 100.0, "steps": 250, "no_angles": False,
        },
        "grambow_crowded_reactant": {
            "geometry": _centred(crowded["symbols"], crowded["coordinates_angstrom"]),
            "dt": 0.10, "temperature": 50.0, "steps": 250, "no_angles": False,
        },
        "symmetric_preference_exchange": {
            "geometry": (["C"] * 6, np.vstack((centre, centre + directions))),
            "dt": 0.02, "temperature": 0.0, "steps": 300, "no_angles": True,
        },
    }


def _build(model, case):
    symbols, positions = case["geometry"]
    simulation = model(
        boxes=[(symbols, positions)],
        box_size=40.0,
        time_step=case["dt"],
        target_temperature=case["temperature"],
        friction=0.0,
        device="cpu",
        dtype=torch.float64,
        random_seed=194,
        relax_on_start=False,
    )
    if case["no_angles"]:
        simulation.angle_stiffness.zero_()
        simulation.forces, simulation._potential_energy = simulation.compute_forces()
    simulation.thermostat_is_on = False
    return simulation


def _run(simulation, steps):
    initial = float(simulation.potential_energy + simulation.kinetic_energy)
    drifts = []
    previous_force = simulation.forces.detach().clone()
    max_force_jump = 0.0
    max_force = 0.0
    for _ in range(steps):
        simulation.step()
        total = float(simulation.potential_energy + simulation.kinetic_energy)
        if not math.isfinite(total) or not bool(torch.isfinite(simulation.forces).all()):
            raise RuntimeError("non-finite NVE state")
        drifts.append(total - initial)
        max_force_jump = max(
            max_force_jump,
            float(torch.max(torch.abs(simulation.forces - previous_force)).detach().cpu()),
        )
        max_force = max(
            max_force,
            float(torch.max(torch.abs(simulation.forces)).detach().cpu()),
        )
        previous_force = simulation.forces.detach().clone()
    values = np.asarray(drifts)
    positions = simulation.positions.detach()
    offsets = positions[:, None, :] - positions[None, :, :]
    offsets = offsets - simulation.box_size * torch.round(offsets / simulation.box_size)
    distances = torch.linalg.vector_norm(offsets, dim=2)
    distances = distances + torch.eye(
        len(positions), device=simulation.device, dtype=simulation.dtype
    ) * simulation.box_size
    return {
        "max_absolute_drift_eV": float(np.max(np.abs(values))),
        "rms_drift_eV": float(np.sqrt(np.mean(values * values))),
        "final_drift_eV": float(values[-1]),
        "move_caps": int(simulation.capped_steps),
        "maximum_force_component_eV_per_angstrom": max_force,
        "maximum_force_step_change_eV_per_angstrom": max_force_jump,
        "final_minimum_distance_angstrom": float(distances.min().detach().cpu()),
    }


def validate(qm_path, grambow_path):
    result = {}
    for case_name, case in _cases(qm_path, grambow_path).items():
        reference = _build(MODELS["production"], case)
        initial_velocities = reference.velocities.detach().clone()
        result[case_name] = {}
        for name, model in MODELS.items():
            simulation = _build(model, case)
            simulation.velocities = initial_velocities.clone()
            result[case_name][name] = _run(simulation, case["steps"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qm", type=Path, default=DEFAULT_QM)
    parser.add_argument("--grambow", type=Path, default=DEFAULT_GRAMBOW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = validate(args.qm, args.grambow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
