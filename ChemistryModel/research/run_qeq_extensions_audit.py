from __future__ import annotations

import torch

from chemistry_engine.config import PhysicsSpec
from chemistry_engine.registry import build

from research.unified_bond_capacity import UnifiedBondCapacityEnergyPrototype
from research.electrostatics_diagnostics import known_diagnostic_cases
from unified_radial_equivalence import build_simulation


def convert_case(data):
    positions, atomic_numbers = data

    symbols = {
        1: "H",
        6: "C",
        7: "N",
        8: "O",
    }

    return {
        "symbols": [
            symbols[number]
            for number in atomic_numbers
        ],
        "positions": torch.tensor(
            positions,
            dtype=torch.float64,
        ),
    }


def build_case(case, extensions):
    sim = build_simulation(
        UnifiedBondCapacityEnergyPrototype,
        [case],
        device="cpu",
        dtype=torch.float64,
        box_size=40.0,
    )

    base = PhysicsSpec.unified_radial_v1(
        {},
        capacity_temperature=0.01,
        h_regularisation_temperature=1e-4,
    )

    spec = base.__class__(
        model_id=base.model_id,
        parameter_sha256=base.parameter_sha256,
        parameter_payload_json=base.parameter_payload_json,
        capacity=base.capacity,
        geometry=base.geometry,
        enabled_terms=base.enabled_terms,
        enabled_extensions=extensions,
    )

    sim.chemistry_engine = build(
        spec.model_id,
        sim,
        spec,
    )

    sim.use_chemistry_engine = True

    return sim


def evaluate(sim):
    positions = sim.positions.detach().requires_grad_(True)

    context = sim.build_interaction_context(
        positions
    )

    result = sim.chemistry_engine.energy(
        context
    )

    total = result.per_atom.sum()

    gradient, = torch.autograd.grad(
        total,
        positions,
    )

    return result, gradient


def audit(name, case):
    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    off = build_case(
        case,
        (),
    )

    on = build_case(
        case,
        ("electrostatics",),
    )

    off_result, off_force = evaluate(off)
    on_result, on_force = evaluate(on)

    print("\nOFF COMPONENTS")
    for key, value in off_result.components.items():
        print(
            f"{key:25s}",
            float(value.sum()),
        )

    print("\nON COMPONENTS")
    for key, value in on_result.components.items():
        print(
            f"{key:25s}",
            float(value.sum()),
        )

    print("\nCHECKS")

    electro = on_result.components["electrostatics"]

    reconstructed = (
        on_result.components["base"]
        + on_result.components["capacity_correction"]
        + on_result.components["topology_correction"]
        + electro
    )

    error = (
        reconstructed
        - on_result.components["total"]
    ).abs().max()

    print(
        "energy reconstruction error:",
        float(error),
    )

    print(
        "energy difference:",
        float(
            on_result.components["total"].sum()
            -
            off_result.components["total"].sum()
        ),
    )

    print(
        "force finite:",
        bool(torch.isfinite(on_force).all()),
    )

    print(
        "force changed:",
        bool(
            torch.linalg.norm(
                on_force - off_force
            ) > 1e-12
        ),
    )

    state = on_result.state

    if "charge_sum" in state:
        print(
            "charge sum:",
            float(state["charge_sum"]),
        )

    if "dipole" in state:
        print(
            "dipole:",
            state["dipole"].tolist(),
        )


def main():
    cases = known_diagnostic_cases()

    for name in ("H2O", "CH2O"):
        audit(
            name,
            convert_case(
                cases[name]
            ),
        )


if __name__ == "__main__":
    main()