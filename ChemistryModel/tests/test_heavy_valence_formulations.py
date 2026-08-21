"""Mathematical guards for the research heavy-valence formulations."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import bond_calibration
from research.heavy_valence_state import (
    HeavyValenceStateEnergyPrototype,
    JointEdgeStateHeavyValencePrototype,
    LocalFreeEnergyHeavyValencePrototype,
)
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


DTYPE = torch.float64
FORMULATIONS = (
    LocalFreeEnergyHeavyValencePrototype,
    JointEdgeStateHeavyValencePrototype,
)


def _evaluate(model, symbols, positions, *, no_angles=False):
    simulation = model(
        boxes=[(symbols, np.asarray(positions, dtype=float))],
        box_size=40.0,
        target_temperature=0.0,
        friction=0.0,
        device="cpu",
        dtype=DTYPE,
        random_seed=0,
        relax_on_start=False,
    )
    if no_angles:
        simulation.angle_stiffness.zero_()
    forces, energy = simulation.compute_forces()
    return simulation, forces.detach(), energy.detach()


def _crowded_carbon(fifth=2.0):
    centre = np.array([20.0, 20.0, 20.0])
    directions = np.asarray([
        [0.0, 0.0, 1.7], [0.0, 0.0, -1.7], [1.7, 0.0, 0.0],
        [-0.85, 1.472243186, 0.0],
        [-0.5 * fifth, -np.sqrt(3.0) * 0.5 * fifth, 0.0],
    ])
    return ["C"] * 6, np.vstack((centre, centre + directions))


def _shared_competition():
    # Both carbons have four H candidates plus the same C-C edge.
    first = np.array([18.0, 20.0, 20.0])
    second = np.array([20.0, 20.0, 20.0])
    offsets = np.asarray([
        [0.0, 0.85, 0.85], [0.0, -0.85, 0.85],
        [0.0, 0.85, -0.85], [0.0, -0.85, -0.85],
    ])
    symbols = ["C", "C"] + ["H"] * 8
    positions = np.vstack((first, second, first + offsets, second + offsets))
    return symbols, positions


@pytest.mark.parametrize("model", FORMULATIONS)
def test_permutation_symmetry_for_equivalent_crowded_contacts(model):
    symbols, positions = _crowded_carbon(1.95)
    _, forces, energy = _evaluate(model, symbols, positions, no_angles=True)
    permutation = np.asarray([0, 4, 2, 5, 1, 3])
    permuted_symbols = [symbols[index] for index in permutation]
    _, permuted_forces, permuted_energy = _evaluate(
        model, permuted_symbols, positions[permutation], no_angles=True
    )
    torch.testing.assert_close(permuted_energy, energy, atol=2e-11, rtol=0)
    torch.testing.assert_close(
        permuted_forces,
        forces[torch.tensor(permutation)],
        atol=2e-9,
        rtol=0,
    )


@pytest.mark.parametrize("model", FORMULATIONS)
@pytest.mark.parametrize("geometry", (_crowded_carbon(2.25), _crowded_carbon(1.90), _shared_competition()))
def test_autograd_force_matches_finite_difference(model, geometry):
    symbols, positions = geometry
    _, forces, _ = _evaluate(model, symbols, positions, no_angles=True)
    atom, axis, epsilon = len(symbols) - 1, 0, 1e-5
    plus = positions.copy()
    minus = positions.copy()
    plus[atom, axis] += epsilon
    minus[atom, axis] -= epsilon
    _, _, e_plus = _evaluate(model, symbols, plus, no_angles=True)
    _, _, e_minus = _evaluate(model, symbols, minus, no_angles=True)
    numerical = -(float(e_plus) - float(e_minus)) / (2.0 * epsilon)
    assert abs(float(forces[atom, axis]) - numerical) < 1e-5


@pytest.mark.parametrize("model", FORMULATIONS)
def test_equivalent_contacts_cross_smoothly(model):
    energies = []
    forces = []
    memberships = []
    for displacement in np.linspace(-0.02, 0.02, 41):
        symbols, positions = _crowded_carbon(2.0)
        centre = positions[0]
        first_direction = positions[4] - centre
        second_direction = positions[5] - centre
        positions[4] = centre + first_direction / np.linalg.norm(first_direction) * (2.0 + displacement)
        positions[5] = centre + second_direction / np.linalg.norm(second_direction) * (2.0 - displacement)
        simulation, live_forces, energy = _evaluate(
            model, symbols, positions, no_angles=True
        )
        energies.append(float(energy))
        forces.append(float(live_forces[4, 0]))
        memberships.append(
            simulation._heavy_valence_energy_diagnostics["membership"][0]
            .detach().cpu().numpy()
        )
    assert np.isfinite(energies).all()
    assert np.isfinite(forces).all()
    assert np.isfinite(memberships).all()
    assert np.max(np.abs(np.diff(energies))) < 0.1

    # The 0.01 eV state temperature permits a physically steep but smooth
    # preference exchange. Resolve the actual left/right limit rather than
    # confusing a large finite slope over 0.001 A with a force jump.
    limiting_forces = []
    for displacement in (-1e-6, 1e-6):
        symbols, positions = _crowded_carbon(2.0)
        centre = positions[0]
        first_direction = positions[4] - centre
        second_direction = positions[5] - centre
        positions[4] = centre + first_direction / np.linalg.norm(first_direction) * (2.0 + displacement)
        positions[5] = centre + second_direction / np.linalg.norm(second_direction) * (2.0 - displacement)
        _, live_forces, _ = _evaluate(model, symbols, positions, no_angles=True)
        limiting_forces.append(float(live_forces[4, 0]))
    assert abs(limiting_forces[1] - limiting_forces[0]) < 0.01


@pytest.mark.parametrize("model", (HeavyValenceStateEnergyPrototype,) + FORMULATIONS)
def test_known_noncompeting_molecules_preserve_current_energy_and_force(model):
    builders = (
        lambda: (["H", "H"], np.asarray([[0.0, 0.0, 0.0], [0.74144, 0.0, 0.0]])),
        bond_calibration.ethane_geometry,
        bond_calibration.methylamine_geometry,
        bond_calibration.methanol_geometry,
        bond_calibration.hydrazine_geometry,
        bond_calibration.hydroxylamine_geometry,
        bond_calibration.hydrogen_peroxide_geometry,
    )
    for builder in builders:
        symbols, positions = builder()
        positions = np.asarray(positions) + 20.0
        _, reference_forces, reference_energy = _evaluate(
            OptimisedValenceStateBatchedSimulation, symbols, positions
        )
        _, candidate_forces, candidate_energy = _evaluate(
            model, symbols, positions
        )
        torch.testing.assert_close(candidate_energy, reference_energy, atol=2e-11, rtol=0)
        torch.testing.assert_close(candidate_forces, reference_forces, atol=2e-9, rtol=0)
