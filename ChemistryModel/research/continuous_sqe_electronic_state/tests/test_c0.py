from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from research.continuous_sqe_electronic_state import (
    C0_PARAMETERS,
    C0ContinuousSQE,
)
from research.electrostatics_diagnostics import known_diagnostic_cases


DTYPE = torch.float64
MODEL = C0ContinuousSQE()
SUPPORT_STRESS_MODEL = C0ContinuousSQE(
    replace(
        C0_PARAMETERS,
        capacity_radius_scale=4.0,
        capacity_steepness=0.5,
        convention="continuous_sqe_c0_support_stress_test",
    )
)


def _positions(values, *, requires_grad=False):
    result = torch.tensor(values, dtype=DTYPE)
    result.requires_grad_(requires_grad)
    return result


def _case(name, *, forces=False):
    coordinates, atomic_numbers = known_diagnostic_cases()[name]
    return MODEL.evaluate(
        _positions(coordinates), atomic_numbers, calculate_forces=forces
    )


def test_c0_schema_is_exactly_17_unfitted_parameters():
    assert C0_PARAMETERS.independent_parameter_count == 17
    assert C0_PARAMETERS.convention == "continuous_sqe_c0_provisional_seed_v1"
    assert MODEL.research_only is True


@pytest.mark.parametrize("name", ("H2", "CH4", "H2O", "CH2O"))
def test_basic_molecules_are_finite_neutral_and_strictly_convex(name):
    result = _case(name, forces=True)
    assert torch.isfinite(result.energy)
    assert torch.isfinite(result.charges).all()
    assert torch.isfinite(result.forces).all()
    assert abs(float(torch.sum(result.charges.detach()))) < 1e-13
    assert result.diagnostics.minimum_response_eigenvalue >= 1.0 - 1e-12
    assert result.diagnostics.minimum_hardness_eigenvalue > 0.0
    assert result.diagnostics.relative_solve_residual < 1e-12
    assert result.diagnostics.all_finite


def test_h3_is_finite_neutral_and_does_not_acquire_symmetry_breaking_charge():
    result = MODEL.evaluate(
        _positions([[-0.74, 0.0, 0.0], [0.0, 0.0, 0.0], [0.92, 0.0, 0.0]]),
        (1, 1, 1),
        calculate_forces=True,
    )
    assert torch.allclose(result.charges, torch.zeros(3, dtype=DTYPE), atol=1e-15)
    assert abs(float(result.energy.detach())) < 1e-15
    assert result.diagnostics.minimum_response_eigenvalue >= 1.0 - 1e-12


def test_nonzero_total_charge_is_rejected_without_reference_charge_assignment():
    coordinates, atomic_numbers = known_diagnostic_cases()["H2O"]
    with pytest.raises(ValueError, match="neutral-only"):
        MODEL.evaluate(_positions(coordinates), atomic_numbers, total_charge=1.0)


def test_water_equivalent_hydrogens_have_equal_charge():
    bond_length = 0.9572
    half_angle = math.radians(104.52 / 2.0)
    x = bond_length * math.cos(half_angle)
    y = bond_length * math.sin(half_angle)
    result = MODEL.evaluate(
        _positions([[0.0, 0.0, 0.0], [x, y, 0.0], [x, -y, 0.0]]),
        (8, 1, 1),
    )
    assert float(result.charges[1]) == pytest.approx(float(result.charges[2]), abs=1e-13)
    assert result.charges[0] < 0.0
    assert result.charges[1] > 0.0


def test_separated_oh_atoms_have_exactly_zero_charge_and_energy():
    result = MODEL.evaluate(
        _positions([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]),
        (8, 1),
        calculate_forces=True,
    )
    assert torch.equal(result.charges, torch.zeros(2, dtype=DTYPE))
    assert float(result.energy.detach()) == 0.0
    assert torch.equal(result.forces, torch.zeros((2, 3), dtype=DTYPE))
    assert result.diagnostics.active_edge_count == 0
    assert result.diagnostics.minimum_response_eigenvalue == 1.0


def test_separated_h_and_oh_fragments_are_independently_neutral():
    result = MODEL.evaluate(
        _positions([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.97, 0.0, 0.0]]),
        (1, 8, 1),
    )
    assert abs(float(result.charges[0])) < 1e-14
    assert abs(float(torch.sum(result.charges[1:]))) < 1e-14
    assert result.diagnostics.active_edge_count == 1


def test_translation_rotation_and_net_force_invariance():
    coordinates, atomic_numbers = known_diagnostic_cases()["H2O"]
    positions = _positions(coordinates)
    reference = MODEL.evaluate(positions, atomic_numbers, calculate_forces=True)
    shift = torch.tensor([2.7, -1.9, 0.8], dtype=DTYPE)
    translated = MODEL.evaluate(
        positions + shift, atomic_numbers, calculate_forces=True
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=DTYPE
    )
    rotated = MODEL.evaluate(
        positions @ rotation.T, atomic_numbers, calculate_forces=True
    )
    assert torch.allclose(reference.energy, translated.energy, atol=1e-13, rtol=0.0)
    assert torch.allclose(reference.charges, translated.charges, atol=1e-13, rtol=0.0)
    assert torch.allclose(reference.forces, translated.forces, atol=1e-12, rtol=0.0)
    assert torch.allclose(reference.energy, rotated.energy, atol=1e-13, rtol=0.0)
    assert torch.allclose(reference.charges, rotated.charges, atol=1e-13, rtol=0.0)
    assert torch.allclose(rotated.forces, reference.forces @ rotation.T, atol=1e-11)
    assert torch.allclose(
        reference.forces.sum(dim=0), torch.zeros(3, dtype=DTYPE), atol=1e-12
    )
    assert torch.allclose(
        reference.polarizability_A3,
        translated.polarizability_A3,
        atol=1e-11,
        rtol=0.0,
    )
    assert torch.allclose(
        rotated.polarizability_A3,
        rotation @ reference.polarizability_A3 @ rotation.T,
        atol=1e-10,
        rtol=0.0,
    )
    assert torch.allclose(
        reference.polarizability_A3,
        reference.polarizability_A3.T,
        atol=1e-12,
        rtol=0.0,
    )
    assert torch.linalg.eigvalsh(reference.polarizability_A3).min() >= -1e-12


def test_atom_permutation_only_permutes_atomic_observables():
    coordinates, atomic_numbers = known_diagnostic_cases()["CH2O"]
    positions = _positions(coordinates)
    reference = MODEL.evaluate(positions, atomic_numbers, calculate_forces=True)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted_numbers = tuple(atomic_numbers[index] for index in permutation.tolist())
    permuted = MODEL.evaluate(
        positions.index_select(0, permutation),
        permuted_numbers,
        calculate_forces=True,
    )
    assert torch.allclose(reference.energy, permuted.energy, atol=1e-13, rtol=0.0)
    assert torch.allclose(
        reference.charges.index_select(0, permutation),
        permuted.charges,
        atol=1e-13,
        rtol=0.0,
    )
    assert torch.allclose(
        reference.forces.index_select(0, permutation), permuted.forces, atol=1e-11
    )


def test_energy_force_matches_central_finite_difference():
    coordinates, atomic_numbers = known_diagnostic_cases()["CH2O"]
    positions = _positions(coordinates)
    result = MODEL.evaluate(positions, atomic_numbers, calculate_forces=True)
    step = 1e-6
    finite_difference = torch.zeros_like(positions)
    for atom in range(len(positions)):
        for axis in range(3):
            plus = positions.clone()
            minus = positions.clone()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            plus_energy = MODEL.evaluate(plus, atomic_numbers).energy
            minus_energy = MODEL.evaluate(minus, atomic_numbers).energy
            finite_difference[atom, axis] = -(plus_energy - minus_energy) / (2 * step)
    assert torch.allclose(result.forces, finite_difference, atol=2e-8, rtol=2e-7)


def test_deterministic_solution_is_bit_identical():
    coordinates, atomic_numbers = known_diagnostic_cases()["CH2O"]
    first = MODEL.evaluate(_positions(coordinates), atomic_numbers)
    second = MODEL.evaluate(_positions(coordinates), atomic_numbers)
    assert torch.equal(first.energy, second.energy)
    assert torch.equal(first.charges, second.charges)
    assert torch.equal(first.transfer_amplitudes, second.transfer_amplitudes)


def test_oh_distance_scan_is_finite_localising_and_has_no_large_jump():
    distances = torch.linspace(0.75, 5.0, 220, dtype=DTYPE)
    energies = []
    charges = []
    minimum_eigenvalues = []
    for distance in distances:
        result = MODEL.evaluate(
            torch.stack(
                [torch.zeros(3, dtype=DTYPE), torch.tensor([distance, 0.0, 0.0])]
            ),
            (8, 1),
        )
        energies.append(result.energy.detach())
        charges.append(result.charges[0].detach())
        minimum_eigenvalues.append(result.diagnostics.minimum_response_eigenvalue)
    energies = torch.stack(energies)
    charges = torch.stack(charges)
    assert torch.isfinite(energies).all()
    assert torch.isfinite(charges).all()
    assert min(minimum_eigenvalues) >= 1.0 - 1e-12
    assert float(torch.max(torch.abs(torch.diff(energies)))) < 0.15
    assert float(torch.max(torch.abs(torch.diff(charges)))) < 0.03
    assert abs(float(charges[-1])) < 1e-14
    assert abs(float(energies[-1])) < 1e-14


@pytest.mark.parametrize("distance", (3.5, 4.5))
def test_capacity_transition_has_continuous_energy_and_force(distance):
    offsets = (-2e-5, -1e-5, 0.0, 1e-5, 2e-5)
    energies = []
    forces = []
    for offset in offsets:
        result = SUPPORT_STRESS_MODEL.evaluate(
            _positions([[0.0, 0.0, 0.0], [distance + offset, 0.0, 0.0]]),
            (8, 1),
            calculate_forces=True,
        )
        energies.append(float(result.energy.detach()))
        forces.append(float(result.forces[1, 0]))
    assert all(math.isfinite(value) for value in energies + forces)
    # The inner point retains the ordinary physical energy slope. Continuity
    # means the slope/force does not jump, not that the local energy is flat.
    assert max(abs(right - left) for left, right in zip(energies, energies[1:])) < 1e-6
    assert max(abs(right - left) for left, right in zip(forces, forces[1:])) < 2e-6
