from __future__ import annotations

import torch

from chemistry_engine.config import PhysicsSpec
from chemistry_engine.registry import build

from research.unified_bond_capacity import UnifiedBondCapacityEnergyPrototype
from unified_radial_equivalence import build_simulation
from research.electrostatics_diagnostics import known_diagnostic_cases


def convert_case(data):
    positions, atomic_numbers = data

    return {
        "symbols": [
            {1: "H", 6: "C", 7: "N", 8: "O"}[z]
            for z in atomic_numbers
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

    base_spec = PhysicsSpec.unified_radial_v1(
        {},
        capacity_temperature=0.01,
        h_regularisation_temperature=1e-4,
    )

    spec = base_spec.__class__(
        model_id=base_spec.model_id,
        parameter_sha256=base_spec.parameter_sha256,
        parameter_payload_json=base_spec.parameter_payload_json,
        capacity=base_spec.capacity,
        geometry=base_spec.geometry,
        enabled_terms=base_spec.enabled_terms,
        enabled_extensions=extensions,
    )

    sim.chemistry_engine = build(
        spec.model_id,
        sim,
        spec,
    )

    if extensions:
        sim.use_chemistry_engine = True

    return sim


def test_extension_registry_builds():
    sim = build_case(
        convert_case(
            known_diagnostic_cases()["H2O"]
        ),
        ("electrostatics",),
    )

    names = [
        term.name
        for term in sim.chemistry_engine.hamiltonian.extensions
    ]

    assert names == ["electrostatics"]


def test_electrostatics_changes_energy():
    case = convert_case(
        known_diagnostic_cases()["H2O"]
    )

    off = build_case(case, ())
    on = build_case(case, ("electrostatics",))

    off.relax(
        steps=50,
        maximum_force=25.0,
        step_size=0.002,
    )

    on.relax(
        steps=50,
        maximum_force=25.0,
        step_size=0.002,
    )

    assert torch.isfinite(
        torch.tensor(off.potential_energy)
    )

    assert torch.isfinite(
        torch.tensor(on.potential_energy)
    )

    assert abs(
        float(on.potential_energy)
        -
        float(off.potential_energy)
    ) > 1e-6


def test_electrostatics_has_forces():
    case = convert_case(
        known_diagnostic_cases()["CH2O"]
    )

    sim = build_case(
        case,
        ("electrostatics",),
    )

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

    assert gradient is not None

    assert torch.isfinite(
        gradient
    ).all()