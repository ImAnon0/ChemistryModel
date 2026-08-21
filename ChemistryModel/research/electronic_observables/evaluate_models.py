"""Evaluate existing research/baseline energies and forces on the manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.unified_bond_capacity import UnifiedBondCapacityEnergyPrototype
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


MODELS = {
    "production": OptimisedValenceStateBatchedSimulation,
    "unified_radial": UnifiedBondCapacityEnergyPrototype,
}


def evaluate(model, symbols, coordinates, box_size=30.0):
    positions = np.asarray(coordinates, dtype=float)
    positions -= positions.mean(axis=0)
    positions += box_size / 2.0
    simulation = model(
        boxes=[(symbols, positions)],
        box_size=box_size,
        time_step=0.1,
        target_temperature=0.0,
        friction=0.0,
        device="cpu",
        dtype=torch.float64,
        random_seed=17,
        relax_on_start=False,
    )
    force = simulation.forces.detach().cpu().numpy()
    if force.ndim == 3:
        force = force[0]
    if force.ndim != 2 or force.shape[1] != 3:
        raise RuntimeError(f"unexpected force shape {force.shape}")
    force = force[: len(symbols)]
    return float(simulation.potential_energy), force.tolist()


def _write_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("research_data/electronic_observables/manifest.json"))
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(f"research_data/electronic_observables/{args.model}.json")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["geometries"][: args.limit] if args.limit else manifest["geometries"]
    output = []
    for index, row in enumerate(rows, 1):
        started = time.perf_counter()
        energy, force = evaluate(MODELS[args.model], row["symbols"], row["coordinates_angstrom"])
        output.append({
            "geometry_id": row["geometry_id"],
            "energy_eV": energy,
            "force_eV_per_angstrom": force,
            "wall_seconds": time.perf_counter() - started,
        })
        print(f"[{index:02d}/{len(rows):02d}] {row['geometry_id']:<38s} {energy:+.8f} eV")
    _write_atomic(args.output, {
        "schema_version": 1,
        "model": args.model,
        "records": output,
        "electronic_observables_available": False,
        "note": "Current models expose energies/forces but no physical charge, dipole, or response state.",
    })
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
