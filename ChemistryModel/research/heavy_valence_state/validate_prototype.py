"""Focused force/NVE safety checks for the heavy-valence energy prototype."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from research.heavy_valence_state import HeavyValenceStateEnergyPrototype
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


DEFAULT_GEOMETRIES = Path("research_data/qm_residual/dense_scan_geometries.json")
DEFAULT_OUTPUT = Path(
    "research_data/benchmark/diagnostics/heavy_valence_state_safety.json"
)


def _water_competition(path, target=1.16):
    rows = json.loads(path.read_text(encoding="utf-8"))["geometries"]
    eligible = [
        row for row in rows
        if row["system"] == "water"
        and row["sample_kind"] == "dense_transfer_scan"
    ]
    geometry = min(
        eligible,
        key=lambda row: abs(
            float(row["reaction_coordinate"]["transfer_distance_angstrom"])
            - target
        ),
    )
    positions = np.asarray(geometry["coordinates_angstrom"], dtype=float)
    positions += np.array([12.0, 12.0, 12.0])
    return geometry["symbols"], positions, geometry["geometry_id"]


def _build(model_class, symbols, positions, device):
    simulation = model_class(
        boxes=[(symbols, positions)],
        box_size=30.0,
        time_step=0.25,
        target_temperature=100.0,
        friction=0.0,
        device=device,
        dtype=torch.float64,
        random_seed=913,
        relax_on_start=False,
    )
    simulation.thermostat_is_on = False
    return simulation


def _total_energy(simulation):
    return float(simulation.potential_energy + simulation.kinetic_energy)


def _run_nve(simulation, steps):
    initial = _total_energy(simulation)
    max_abs_drift = 0.0
    max_force = 0.0
    for _ in range(steps):
        simulation.step()
        total = _total_energy(simulation)
        if not math.isfinite(total):
            raise RuntimeError("NVE trajectory became non-finite")
        max_abs_drift = max(max_abs_drift, abs(total - initial))
        max_force = max(
            max_force,
            float(torch.max(torch.abs(simulation.forces)).detach().cpu()),
        )
    return {
        "initial_total_eV": initial,
        "final_total_eV": _total_energy(simulation),
        "max_absolute_drift_eV": max_abs_drift,
        "maximum_force_component_eV_per_angstrom": max_force,
        "capped_steps": int(simulation.capped_steps),
        "steps": steps,
        "time_step_fs": float(simulation.time_step),
    }


def validate(geometries, steps, device):
    symbols, positions, geometry_id = _water_competition(geometries)
    current = _build(
        OptimisedValenceStateBatchedSimulation, symbols, positions, device
    )
    prototype = _build(
        HeavyValenceStateEnergyPrototype, symbols, positions, device
    )
    prototype.velocities = current.velocities.detach().clone()

    result = {
        "geometry_id": geometry_id,
        "device": str(device),
        "current": _run_nve(current, steps),
        "prototype": _run_nve(prototype, steps),
    }
    result["gates"] = {
        "finite": True,
        "prototype_max_drift_below_0.05_eV": (
            result["prototype"]["max_absolute_drift_eV"] < 0.05
        ),
        "prototype_no_move_caps": (
            result["prototype"]["capped_steps"] == 0
        ),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometries", type=Path, default=DEFAULT_GEOMETRIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = validate(args.geometries, args.steps, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not all(result["gates"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
