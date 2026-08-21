"""Focused guards for the research-only heavy-valence energy prototype."""

from __future__ import annotations

import numpy as np
import torch

import bond_calibration
from research.heavy_valence_state import HeavyValenceStateEnergyPrototype
from valence_state_optimised_torch import (
    OptimisedValenceStateBatchedSimulation,
)


DTYPE = torch.float64


def _build(simulation_class, symbols, positions):
    simulation = simulation_class(
        boxes=[(symbols, np.asarray(positions, dtype=float))],
        box_size=30.0,
        target_temperature=0.0,
        friction=0.0,
        device="cpu",
        dtype=DTYPE,
        random_seed=0,
        relax_on_start=False,
    )
    return simulation


def _crowded_carbon_geometry(fifth_distance=2.0):
    centre = np.array([10.0, 10.0, 10.0])
    radius = 1.7
    vectors = [
        np.array([0.0, 0.0, radius]),
        np.array([0.0, 0.0, -radius]),
        np.array([radius, 0.0, 0.0]),
        np.array([
            radius * np.cos(2.0 * np.pi / 3.0),
            radius * np.sin(2.0 * np.pi / 3.0),
            0.0,
        ]),
        np.array([
            fifth_distance * np.cos(4.0 * np.pi / 3.0),
            fifth_distance * np.sin(4.0 * np.pi / 3.0),
            0.0,
        ]),
    ]
    return ["C"] * 6, np.asarray([centre] + [centre + v for v in vectors])


def _energy_and_forces(simulation_class, symbols, positions, *, no_angles=False):
    simulation = _build(simulation_class, symbols, positions)
    if no_angles:
        simulation.angle_stiffness.zero_()
    forces, energy = simulation.compute_forces()
    return simulation, forces, energy


def test_prototype_is_explicitly_research_only():
    assert HeavyValenceStateEnergyPrototype.research_only is True
    assert HeavyValenceStateEnergyPrototype.physics_model_revision == 0


def test_no_heavy_competition_is_exactly_unchanged():
    cases = [
        (["H", "H"], [[10.0, 10.0, 10.0], [10.74144, 10.0, 10.0]]),
        (["C", "H", "H", "H", "H"], [
            [10.0, 10.0, 10.0], [10.63, 10.63, 10.63],
            [9.37, 9.37, 10.63], [9.37, 10.63, 9.37],
            [10.63, 9.37, 9.37],
        ]),
        (["O", "H", "H"], [
            [10.0, 10.0, 10.0], [10.957, 10.0, 10.0],
            [9.760, 10.927, 10.0],
        ]),
    ]

    for symbols, positions in cases:
        _, current_forces, current_energy = _energy_and_forces(
            OptimisedValenceStateBatchedSimulation, symbols, positions
        )
        _, prototype_forces, prototype_energy = _energy_and_forces(
            HeavyValenceStateEnergyPrototype, symbols, positions
        )
        torch.testing.assert_close(prototype_energy, current_energy, atol=1e-12, rtol=0)
        torch.testing.assert_close(prototype_forces, current_forces, atol=1e-11, rtol=0)


def test_accepted_heavy_bond_molecules_are_exactly_unchanged():
    builders = (
        bond_calibration.ethane_geometry,
        bond_calibration.methylamine_geometry,
        bond_calibration.methanol_geometry,
        bond_calibration.hydrazine_geometry,
        bond_calibration.hydroxylamine_geometry,
        bond_calibration.hydrogen_peroxide_geometry,
    )
    for builder in builders:
        symbols, positions = builder()
        positions = np.asarray(positions) + 10.0
        _, current_forces, current_energy = _energy_and_forces(
            OptimisedValenceStateBatchedSimulation, symbols, positions
        )
        _, prototype_forces, prototype_energy = _energy_and_forces(
            HeavyValenceStateEnergyPrototype, symbols, positions
        )
        torch.testing.assert_close(
            prototype_energy, current_energy, atol=1e-12, rtol=0
        )
        torch.testing.assert_close(
            prototype_forces, current_forces, atol=1e-11, rtol=0
        )


def test_crowded_contact_remains_energy_bearing():
    symbols, positions = _crowded_carbon_geometry()
    current, _, current_energy = _energy_and_forces(
        OptimisedValenceStateBatchedSimulation,
        symbols,
        positions,
        no_angles=True,
    )
    prototype, forces, prototype_energy = _energy_and_forces(
        HeavyValenceStateEnergyPrototype,
        symbols,
        positions,
        no_angles=True,
    )
    diagnostics = prototype._heavy_valence_energy_diagnostics

    current_over = current._energy_parts["over"].sum()
    rejected_attraction = diagnostics["rejected_attraction_per_atom"].sum()
    prototype_over = diagnostics["prototype_over_per_atom"].sum()
    naive_delete_energy = current_energy - current_over

    assert float(current_over) > 1.0
    assert float(rejected_attraction) > 0.5
    assert float(prototype_over) < 1e-12
    assert float(prototype_energy) > float(naive_delete_energy) + 0.5
    assert torch.isfinite(forces).all()


def test_crowded_contact_force_matches_central_difference():
    symbols, positions = _crowded_carbon_geometry()
    simulation, forces, _ = _energy_and_forces(
        HeavyValenceStateEnergyPrototype,
        symbols,
        positions,
        no_angles=True,
    )

    atom = 5
    axis = 0
    epsilon = 1e-5
    displaced_plus = positions.copy()
    displaced_minus = positions.copy()
    displaced_plus[atom, axis] += epsilon
    displaced_minus[atom, axis] -= epsilon

    plus, _, plus_energy = _energy_and_forces(
        HeavyValenceStateEnergyPrototype,
        symbols,
        displaced_plus,
        no_angles=True,
    )
    minus, _, minus_energy = _energy_and_forces(
        HeavyValenceStateEnergyPrototype,
        symbols,
        displaced_minus,
        no_angles=True,
    )
    del plus, minus, simulation

    finite_difference_force = -(
        float(plus_energy) - float(minus_energy)
    ) / (2.0 * epsilon)
    assert abs(float(forces[atom, axis]) - finite_difference_force) < 2e-6


def test_competition_scan_is_finite_and_continuous():
    energies = []
    forces = []
    for distance in np.linspace(1.90, 2.10, 41):
        symbols, positions = _crowded_carbon_geometry(distance)
        _, live_forces, energy = _energy_and_forces(
            HeavyValenceStateEnergyPrototype,
            symbols,
            positions,
            no_angles=True,
        )
        energies.append(float(energy))
        forces.append(float(torch.linalg.vector_norm(live_forces[5])))

    assert np.isfinite(energies).all()
    assert np.isfinite(forces).all()
    assert np.max(np.abs(np.diff(energies))) < 0.2
    assert np.max(np.abs(np.diff(forces))) < 0.5
