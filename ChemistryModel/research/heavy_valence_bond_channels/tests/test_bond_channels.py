"""Evidence gates for shared incremental bond-order channels."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from research.heavy_valence_bond_channels import (
    ContinuousBondFreeEnergyPrototype,
    SharedBondOrderChannelPrototype,
)
from research.heavy_valence_continuous_edge.tests.test_continuous_edge import (
    _accepted_geometries,
    _crowded_carbon,
    _evaluate,
)
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


def test_channel_reference_is_research_only():
    assert SharedBondOrderChannelPrototype.research_only is True
    assert SharedBondOrderChannelPrototype.physics_model_revision == 0


@pytest.mark.parametrize("geometry", _accepted_geometries())
def test_required_accepted_molecules_are_exactly_unchanged(geometry):
    symbols, positions = geometry
    positions = np.asarray(positions) + 20.0
    _, reference_force, reference_energy = _evaluate(
        OptimisedValenceStateBatchedSimulation, symbols, positions
    )
    simulation, channel_force, channel_energy = _evaluate(
        SharedBondOrderChannelPrototype, symbols, positions
    )
    torch.testing.assert_close(channel_energy, reference_energy, atol=2e-11, rtol=0)
    torch.testing.assert_close(channel_force, reference_force, atol=2e-9, rtol=0)
    assert simulation._continuous_edge_state["solver"]["solver"] in {
        "empty", "unconstrained"
    }


def test_channels_are_shared_capacity_limited_and_hierarchical():
    symbols, positions = _crowded_carbon(1.95)
    simulation, force, _ = _evaluate(
        SharedBondOrderChannelPrototype, symbols, positions, no_angles=True
    )
    state = simulation._continuous_edge_state
    assert torch.isfinite(force).all()
    assert len(state["hierarchy"]) > 0
    assert torch.any(state["occupancy"] < 1.0 - 1e-8)
    assert torch.all(state["incidence"] @ state["capacity"] <= state["residual_capacity"] + 2e-9)
    for lower, upper in state["hierarchy"]:
        assert state["occupancy"][upper] <= state["occupancy"][lower] + 2e-9
    assert state["solver"]["maximum_constraint_violation"] < 2e-9


def test_channel_energy_partition_recovers_full_edge_attraction():
    symbols, positions = _crowded_carbon(2.30)
    simulation, _, _ = _evaluate(
        SharedBondOrderChannelPrototype, symbols, positions, no_angles=True
    )
    state = simulation._continuous_edge_state
    # Incremental shares sum back to the exact live blended attraction.
    for edge_index, variables in enumerate(state["channels_by_edge"]):
        index = torch.tensor(variables)
        torch.testing.assert_close(
            state["channel_attraction"][index].sum(),
            state["edge_full_attraction"][edge_index],
            atol=2e-12,
            rtol=0,
        )
    assert torch.all(state["channel_attraction"] > 0.0)


def test_permutation_symmetry_for_channel_allocation():
    symbols, positions = _crowded_carbon(1.95)
    _, force, energy = _evaluate(
        SharedBondOrderChannelPrototype, symbols, positions, no_angles=True
    )
    permutation = np.asarray([0, 4, 2, 5, 1, 3])
    _, permuted_force, permuted_energy = _evaluate(
        SharedBondOrderChannelPrototype,
        [symbols[index] for index in permutation],
        positions[permutation],
        no_angles=True,
    )
    torch.testing.assert_close(permuted_energy, energy, atol=2e-10, rtol=0)
    torch.testing.assert_close(
        permuted_force, force[torch.tensor(permutation)], atol=2e-8, rtol=0
    )


@pytest.mark.parametrize("distance", (2.25, 1.95))
def test_channel_autograd_force_matches_central_difference(distance):
    symbols, positions = _crowded_carbon(distance)
    _, force, _ = _evaluate(
        SharedBondOrderChannelPrototype, symbols, positions, no_angles=True
    )
    epsilon = 1e-5
    plus = positions.copy()
    minus = positions.copy()
    plus[-1, 0] += epsilon
    minus[-1, 0] -= epsilon
    _, _, e_plus = _evaluate(
        SharedBondOrderChannelPrototype, symbols, plus, no_angles=True
    )
    _, _, e_minus = _evaluate(
        SharedBondOrderChannelPrototype, symbols, minus, no_angles=True
    )
    numerical = -(float(e_plus) - float(e_minus)) / (2.0 * epsilon)
    assert abs(float(force[-1, 0]) - numerical) < 4e-5


def test_channel_preference_exchange_is_continuous():
    energies = []
    forces = []
    for displacement in np.linspace(-0.02, 0.02, 41):
        symbols, positions = _crowded_carbon(2.0)
        centre = positions[0]
        for atom, sign in ((4, 1.0), (5, -1.0)):
            direction = positions[atom] - centre
            positions[atom] = centre + direction / np.linalg.norm(direction) * (
                2.0 + sign * displacement
            )
        _, live_force, energy = _evaluate(
            SharedBondOrderChannelPrototype, symbols, positions, no_angles=True
        )
        energies.append(float(energy))
        forces.append(float(live_force[4, 0]))
    assert np.isfinite(energies).all()
    assert np.isfinite(forces).all()
    assert np.max(np.abs(np.diff(energies))) < 0.1

    limiting = []
    for displacement in (-1e-6, 1e-6):
        symbols, positions = _crowded_carbon(2.0)
        centre = positions[0]
        for atom, sign in ((4, 1.0), (5, -1.0)):
            direction = positions[atom] - centre
            positions[atom] = centre + direction / np.linalg.norm(direction) * (
                2.0 + sign * displacement
            )
        _, live_force, _ = _evaluate(
            SharedBondOrderChannelPrototype, symbols, positions, no_angles=True
        )
        limiting.append(float(live_force[4, 0]))
    assert abs(limiting[1] - limiting[0]) < 0.02


@pytest.mark.parametrize("geometry", _accepted_geometries())
def test_redesigned_bond_free_energy_preserves_accepted_molecules(geometry):
    symbols, positions = geometry
    positions = np.asarray(positions) + 20.0
    _, reference_force, reference_energy = _evaluate(
        OptimisedValenceStateBatchedSimulation, symbols, positions
    )
    _, candidate_force, candidate_energy = _evaluate(
        ContinuousBondFreeEnergyPrototype, symbols, positions
    )
    torch.testing.assert_close(candidate_energy, reference_energy, atol=2e-11, rtol=0)
    torch.testing.assert_close(candidate_force, reference_force, atol=2e-9, rtol=0)


def _cutoff_onset_geometry(distance):
    centre = np.asarray([20.0, 20.0, 20.0])
    directions = np.asarray([
        [1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0],
    ])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    positions = np.vstack((
        centre,
        centre + 1.54 * directions,
        centre + np.asarray([distance, 0.0, 0.0]),
    ))
    return ["C"] * 6, positions


def test_redesigned_surface_is_continuous_at_competition_onset():
    left = _cutoff_onset_geometry(2.440001)
    right = _cutoff_onset_geometry(2.439999)
    _, left_force, left_energy = _evaluate(
        ContinuousBondFreeEnergyPrototype, *left, no_angles=True
    )
    _, right_force, right_energy = _evaluate(
        ContinuousBondFreeEnergyPrototype, *right, no_angles=True
    )
    assert abs(float(right_energy - left_energy)) < 5e-5
    assert abs(float(right_force[-1, 0] - left_force[-1, 0])) < 2e-3


def test_redesigned_surface_force_matches_finite_difference():
    symbols, positions = _crowded_carbon(1.95)
    simulation, force, _ = _evaluate(
        ContinuousBondFreeEnergyPrototype, symbols, positions, no_angles=True
    )
    epsilon = 1e-5
    plus = positions.copy()
    minus = positions.copy()
    plus[-1, 0] += epsilon
    minus[-1, 0] -= epsilon
    _, _, e_plus = _evaluate(
        ContinuousBondFreeEnergyPrototype, symbols, plus, no_angles=True
    )
    _, _, e_minus = _evaluate(
        ContinuousBondFreeEnergyPrototype, symbols, minus, no_angles=True
    )
    numerical = -(float(e_plus) - float(e_minus)) / (2.0 * epsilon)
    assert abs(float(force[-1, 0]) - numerical) < 1e-4
    component = simulation._continuous_bond_data["components"][0]
    assert component["full"]["max_capacity_violation"] < 1e-8
    assert component["reference"]["max_capacity_violation"] < 1e-8


def test_redesigned_surface_is_permutation_symmetric():
    symbols, positions = _crowded_carbon(1.95)
    _, force, energy = _evaluate(
        ContinuousBondFreeEnergyPrototype, symbols, positions, no_angles=True
    )
    permutation = np.asarray([0, 4, 2, 5, 1, 3])
    _, permuted_force, permuted_energy = _evaluate(
        ContinuousBondFreeEnergyPrototype,
        [symbols[index] for index in permutation], positions[permutation],
        no_angles=True,
    )
    torch.testing.assert_close(permuted_energy, energy, atol=2e-8, rtol=0)
    torch.testing.assert_close(
        permuted_force, force[torch.tensor(permutation)], atol=2e-5, rtol=0
    )
