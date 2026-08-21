"""Evidence gates for the research-only continuous shared-edge model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import bond_calibration
from research.heavy_valence_continuous_edge import (
    ContinuousSharedEdgeHeavyValencePrototype,
)
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


DTYPE = torch.float64


def _evaluate(model, symbols, positions, *, no_angles=False):
    simulation = model(
        boxes=[(list(symbols), np.asarray(positions, dtype=float))],
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


def _accepted_geometries():
    methane = (["C", "H", "H", "H", "H"], np.asarray([
        [0.0, 0.0, 0.0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
        [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63],
    ]))
    formaldehyde = (["C", "O", "H", "H"], np.asarray([
        [0.0, 0.0, 0.0], [1.21, 0.0, 0.0],
        [-0.55, 0.9526, 0.0], [-0.55, -0.9526, 0.0],
    ]))
    water = (["O", "H", "H"], np.asarray([
        [0.0, 0.0, 0.0], [0.9572, 0.0, 0.0],
        [-0.2399872, 0.9266272, 0.0],
    ]))
    h3 = (["H", "H", "H"], np.asarray([
        [-0.75, 0.0, 0.0], [0.0, 0.0, 0.0], [0.75, 0.0, 0.0],
    ]))
    return (
        h3, methane, formaldehyde, water,
        bond_calibration.ethane_geometry(),
        bond_calibration.methanol_geometry(),
        bond_calibration.hydroxylamine_geometry(),
        bond_calibration.hydrogen_peroxide_geometry(),
    )


def test_reference_is_research_only():
    assert ContinuousSharedEdgeHeavyValencePrototype.research_only is True
    assert ContinuousSharedEdgeHeavyValencePrototype.physics_model_revision == 0


@pytest.mark.parametrize("geometry", _accepted_geometries())
def test_accepted_chemistry_is_exactly_unchanged(geometry):
    symbols, positions = geometry
    positions = np.asarray(positions) + 20.0
    _, reference_force, reference_energy = _evaluate(
        OptimisedValenceStateBatchedSimulation, symbols, positions
    )
    simulation, candidate_force, candidate_energy = _evaluate(
        ContinuousSharedEdgeHeavyValencePrototype, symbols, positions
    )
    torch.testing.assert_close(candidate_energy, reference_energy, atol=2e-11, rtol=0)
    torch.testing.assert_close(candidate_force, reference_force, atol=2e-9, rtol=0)
    diagnostics = simulation._heavy_valence_energy_diagnostics
    assert diagnostics["solver"] in {"empty", "unconstrained"}


def test_one_fractional_capacity_is_shared_by_both_edge_endpoints():
    symbols, positions = _crowded_carbon(1.95)
    simulation, force, _ = _evaluate(
        ContinuousSharedEdgeHeavyValencePrototype,
        symbols,
        positions,
        no_angles=True,
    )
    state = simulation._continuous_edge_state
    assert torch.isfinite(force).all()
    assert torch.any(state["occupancy"] < 1.0 - 1e-8)
    assert torch.all(state["occupancy"] >= 0.0)
    assert torch.all(state["incidence"] @ state["capacity"] <= state["residual_capacity"] + 2e-9)
    for edge_index, (first, second) in enumerate(state["edges"]):
        assert state["capacity_rows"][first].max() >= state["capacity"][edge_index] - 1e-12
        assert state["capacity_rows"][second].max() >= state["capacity"][edge_index] - 1e-12


def test_permutation_symmetry():
    symbols, positions = _crowded_carbon(1.95)
    _, force, energy = _evaluate(
        ContinuousSharedEdgeHeavyValencePrototype, symbols, positions,
        no_angles=True,
    )
    permutation = np.asarray([0, 4, 2, 5, 1, 3])
    permuted_symbols = [symbols[index] for index in permutation]
    _, permuted_force, permuted_energy = _evaluate(
        ContinuousSharedEdgeHeavyValencePrototype,
        permuted_symbols,
        positions[permutation],
        no_angles=True,
    )
    torch.testing.assert_close(permuted_energy, energy, atol=2e-10, rtol=0)
    torch.testing.assert_close(
        permuted_force, force[torch.tensor(permutation)], atol=2e-8, rtol=0
    )


@pytest.mark.parametrize("distance", (2.25, 1.95))
def test_autograd_force_matches_central_difference(distance):
    symbols, positions = _crowded_carbon(distance)
    _, force, _ = _evaluate(
        ContinuousSharedEdgeHeavyValencePrototype, symbols, positions,
        no_angles=True,
    )
    epsilon = 1e-5
    plus = positions.copy()
    minus = positions.copy()
    plus[-1, 0] += epsilon
    minus[-1, 0] -= epsilon
    _, _, e_plus = _evaluate(
        ContinuousSharedEdgeHeavyValencePrototype, symbols, plus,
        no_angles=True,
    )
    _, _, e_minus = _evaluate(
        ContinuousSharedEdgeHeavyValencePrototype, symbols, minus,
        no_angles=True,
    )
    numerical = -(float(e_plus) - float(e_minus)) / (2.0 * epsilon)
    assert abs(float(force[-1, 0]) - numerical) < 3e-5


def test_equivalent_contacts_cross_without_energy_or_force_jump():
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
            ContinuousSharedEdgeHeavyValencePrototype,
            symbols,
            positions,
            no_angles=True,
        )
        energies.append(float(energy))
        forces.append(float(live_force[4, 0]))
    assert np.isfinite(energies).all()
    assert np.isfinite(forces).all()
    assert np.max(np.abs(np.diff(energies))) < 0.1

    limits = []
    for displacement in (-1e-6, 1e-6):
        symbols, positions = _crowded_carbon(2.0)
        centre = positions[0]
        for atom, sign in ((4, 1.0), (5, -1.0)):
            direction = positions[atom] - centre
            positions[atom] = centre + direction / np.linalg.norm(direction) * (
                2.0 + sign * displacement
            )
        _, live_force, _ = _evaluate(
            ContinuousSharedEdgeHeavyValencePrototype,
            symbols,
            positions,
            no_angles=True,
        )
        limits.append(float(live_force[4, 0]))
    assert abs(limits[1] - limits[0]) < 0.02
