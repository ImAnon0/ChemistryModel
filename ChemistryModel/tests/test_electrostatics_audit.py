from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import batch_runner
from chemistry_engine.config import PhysicsSpec
from chemistry_engine.registry import build
from chemistry_engine.terms.electrostatics import ElectrostaticEnergyTerm
from research.electrostatics_diagnostics import known_diagnostic_cases
from research.unified_bond_capacity import UnifiedBondCapacityEnergyPrototype
from unified_radial_equivalence import build_simulation


def _context(positions, atomic_numbers, *, assignments=None, box_size=40.0):
    count = len(atomic_numbers)
    return SimpleNamespace(
        positions=positions,
        atomic_numbers=tuple(atomic_numbers),
        batch_assignment=(tuple(assignments) if assignments is not None else (0,) * count),
        box_size=box_size,
    )


def _engine_case(data, extensions):
    positions, atomic_numbers = data
    symbols = {1: "H", 6: "C", 7: "N", 8: "O"}
    case = {
        "symbols": [symbols[number] for number in atomic_numbers],
        "positions": torch.tensor(positions, dtype=torch.float64),
    }
    simulation = build_simulation(
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
    simulation.chemistry_engine = build(spec.model_id, simulation, spec)
    return simulation


def _energy_and_force(positions, atomic_numbers, *, assignments=None, box_size=40.0):
    positions = positions.detach().clone().requires_grad_(True)
    context = _context(
        positions,
        atomic_numbers,
        assignments=assignments,
        box_size=box_size,
    )
    term = ElectrostaticEnergyTerm(enabled=True)
    energy = term.energy(context, torch.zeros((), dtype=positions.dtype))
    force = -torch.autograd.grad(energy, positions)[0]
    return energy.detach(), force.detach(), term.diagnostics()


@pytest.mark.parametrize("name", ("H2", "H2O", "CH4", "CH2O"))
def test_qeq_charge_determinism_neutrality_and_finiteness(name):
    coordinates, atomic_numbers = known_diagnostic_cases()[name]
    positions = torch.tensor(coordinates, dtype=torch.float64)

    first = _energy_and_force(positions, atomic_numbers)
    second = _energy_and_force(positions, atomic_numbers)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2]["charges"], second[2]["charges"])
    assert torch.isfinite(first[0])
    assert torch.isfinite(first[1]).all()
    assert torch.isfinite(first[2]["charges"]).all()
    assert abs(float(first[2]["charge_sum"].detach())) < 1e-12


def test_h3_identical_atoms_remain_neutral_and_finite():
    positions = torch.tensor(
        [[-0.74, 0.0, 0.0], [0.0, 0.0, 0.0], [0.92, 0.0, 0.0]],
        dtype=torch.float64,
    )
    energy, forces, state = _energy_and_force(positions, (1, 1, 1))

    assert torch.isfinite(energy)
    assert torch.isfinite(forces).all()
    assert torch.allclose(state["charges"], torch.zeros(3, dtype=torch.float64), atol=2e-16)


def test_qeq_electrostatic_force_matches_finite_difference():
    coordinates, atomic_numbers = known_diagnostic_cases()["CH2O"]
    positions = torch.tensor(coordinates, dtype=torch.float64)
    _, force, _ = _energy_and_force(positions, atomic_numbers)

    step = 1e-6
    finite_difference = torch.zeros_like(positions)
    for atom in range(len(positions)):
        for axis in range(3):
            plus = positions.clone()
            minus = positions.clone()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            plus_energy = _energy_and_force(plus, atomic_numbers)[0]
            minus_energy = _energy_and_force(minus, atomic_numbers)[0]
            finite_difference[atom, axis] = -(plus_energy - minus_energy) / (2 * step)

    assert torch.allclose(force, finite_difference, atol=2e-8, rtol=2e-7)


def test_qeq_translation_rotation_and_net_force_invariance():
    coordinates, atomic_numbers = known_diagnostic_cases()["H2O"]
    positions = torch.tensor(coordinates, dtype=torch.float64)
    energy, force, _ = _energy_and_force(positions, atomic_numbers)

    translated_energy, translated_force, _ = _energy_and_force(
        positions + torch.tensor([3.2, -1.4, 2.1], dtype=torch.float64),
        atomic_numbers,
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    rotated_energy, rotated_force, _ = _energy_and_force(
        positions @ rotation.T,
        atomic_numbers,
    )

    assert torch.allclose(energy, translated_energy, atol=1e-13, rtol=0.0)
    assert torch.allclose(force, translated_force, atol=1e-13, rtol=0.0)
    assert torch.allclose(energy, rotated_energy, atol=1e-13, rtol=0.0)
    assert torch.allclose(rotated_force, force @ rotation.T, atol=1e-13, rtol=0.0)
    assert torch.allclose(force.sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-13)


def test_qeq_uses_periodic_minimum_image_distances():
    atomic_numbers = (8, 1, 1)
    wrapped = torch.tensor(
        [[0.10, 0.0, 0.0], [29.70, 0.0, 0.0], [0.10, 0.96, 0.0]],
        dtype=torch.float64,
    )
    unwrapped = wrapped.clone()
    unwrapped[1, 0] -= 30.0

    wrapped_result = _energy_and_force(wrapped, atomic_numbers, box_size=30.0)
    unwrapped_result = _energy_and_force(unwrapped, atomic_numbers, box_size=30.0)

    assert torch.allclose(wrapped_result[0], unwrapped_result[0], atol=1e-13, rtol=0.0)
    assert torch.allclose(
        wrapped_result[2]["charges"],
        unwrapped_result[2]["charges"],
        atol=1e-13,
        rtol=0.0,
    )


def test_qeq_batched_boxes_are_independent_neutral_systems():
    water_positions, water_numbers = known_diagnostic_cases()["H2O"]
    methane_positions, methane_numbers = known_diagnostic_cases()["CH4"]
    water = torch.tensor(water_positions, dtype=torch.float64)
    methane = torch.tensor(methane_positions, dtype=torch.float64) + 4.0

    water_result = _energy_and_force(water, water_numbers)
    methane_result = _energy_and_force(methane, methane_numbers)

    combined_positions = torch.cat([water, methane], dim=0)
    combined_numbers = tuple(water_numbers) + tuple(methane_numbers)
    assignments = (0,) * len(water) + (1,) * len(methane)
    combined = _energy_and_force(
        combined_positions,
        combined_numbers,
        assignments=assignments,
    )

    assert torch.allclose(combined[0], water_result[0] + methane_result[0], atol=1e-13)
    assert torch.allclose(combined[2]["charges"][: len(water)], water_result[2]["charges"])
    assert torch.allclose(combined[2]["charges"][len(water) :], methane_result[2]["charges"])
    assert torch.allclose(
        combined[2]["box_charge_sums"],
        torch.zeros(2, dtype=torch.float64),
        atol=1e-12,
    )


def test_electrostatic_diagnostics_are_exposed_by_energy_result():
    simulation = _engine_case(known_diagnostic_cases()["H2O"], ("electrostatics",))
    positions = simulation.positions.detach().requires_grad_(True)
    result = simulation.chemistry_engine.energy(
        simulation.build_interaction_context(positions)
    )

    state = result.state["extensions"]["electrostatics"]
    assert abs(float(state["charge_sum"].detach())) < 1e-12
    assert state["charges"].shape == (3,)
    assert state["charges"].requires_grad is False
    assert "box_solutions" not in state


def test_enabled_extension_is_recorded_in_physics_provenance():
    simulation = _engine_case(known_diagnostic_cases()["H2O"], ("electrostatics",))
    simulation.chemistry_physics_spec = simulation.chemistry_engine.physics

    metadata = batch_runner.simulation_physics_provenance(simulation)

    assert metadata["physics_enabled_extensions"] == ["electrostatics"]
    assert len(metadata["physics_extension_parameter_sha256"]) == 64
    assert '"minimum_image_unshielded_inverse_distance_v1"' in (
        metadata["physics_extension_parameter_payload_json"]
    )
