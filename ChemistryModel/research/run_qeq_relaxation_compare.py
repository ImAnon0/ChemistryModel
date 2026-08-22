from __future__ import annotations

import torch

from chemistry_engine.config import PhysicsSpec
from chemistry_engine.registry import build
from research.electrostatics_diagnostics import known_diagnostic_cases
from research.unified_bond_capacity import UnifiedBondCapacityEnergyPrototype
from unified_radial_equivalence import build_simulation


def convert_case(name, data):
    positions, atomic_numbers = data
    symbols = {1: "H", 6: "C", 7: "N", 8: "O"}

    return {
        "name": name,
        "symbols": [symbols[number] for number in atomic_numbers],
        "positions": positions,
    }


def build_case(case, extensions):
    sim = build_simulation(
        UnifiedBondCapacityEnergyPrototype,
        [case],
        device="cpu",
        dtype=torch.float64,
        box_size=40.0,
    )

    spec = PhysicsSpec.unified_radial_v1(
        {},
        capacity_temperature=0.01,
        h_regularisation_temperature=1e-4,
    )

    spec = spec.__class__(
        model_id=spec.model_id,
        parameter_sha256=spec.parameter_sha256,
        parameter_payload_json=spec.parameter_payload_json,
        capacity=spec.capacity,
        geometry=spec.geometry,
        enabled_terms=spec.enabled_terms,
        enabled_extensions=extensions,
    )

    sim.chemistry_engine = build(spec.model_id, sim, spec)

    if "electrostatics" in extensions:
        sim.use_chemistry_engine = True

    return sim


def distance(positions, a, b):
    return torch.linalg.norm(positions[a] - positions[b]).item()


def report(name, sim):
    positions = sim.positions.detach().cpu()

    print("energy:", float(sim.potential_energy))

    if name == "H2O":
        print("O-H1:", distance(positions, 0, 1))
        print("O-H2:", distance(positions, 0, 2))

    if name == "CH2O":
        print("C-O :", distance(positions, 0, 1))
        print("C-H1:", distance(positions, 0, 2))
        print("C-H2:", distance(positions, 0, 3))


def run():
    cases = {
        name: convert_case(name, data)
        for name, data in known_diagnostic_cases().items()
        if name in ("H2O", "CH2O")
    }

    for name, case in cases.items():
        print("=" * 40)
        print(name)

        off = build_case(case, ())
        on = build_case(case, ("electrostatics",))
        print("ENGINE EXTENSIONS:", on.chemistry_engine.hamiltonian.extensions)

        off.relax(steps=300, maximum_force=25.0, step_size=0.002)
        on.relax(steps=300, maximum_force=25.0, step_size=0.002)

        print("OFF")
        report(name, off)

        print("ON")
        report(name, on)


if __name__ == "__main__":
    run()
