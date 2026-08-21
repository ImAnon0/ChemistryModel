from __future__ import annotations

import numpy as np
import pytest
import torch
import build_box
import bond_calibration

from research.unified_bond_capacity import UnifiedBondCapacityEnergyPrototype
from research.heavy_valence_continuous_edge.tests.test_continuous_edge import (
    _accepted_geometries,
    _crowded_carbon,
    _evaluate,
)
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


def _simulation(symbols, coordinates):
    positions = np.asarray(coordinates, dtype=float)
    positions -= positions.mean(axis=0)
    positions += 15.0
    return UnifiedBondCapacityEnergyPrototype(
        boxes=[(symbols, positions)], box_size=30.0, time_step=0.1,
        target_temperature=0.0, friction=0.0, device="cpu",
        dtype=torch.float64, random_seed=7, relax_on_start=False,
    )


@pytest.mark.parametrize("distance", [0.60, 0.74144, 1.0, 1.8])
def test_h2_energy_force_are_finite(distance):
    sim = _simulation(["H", "H"], [[-distance / 2, 0, 0], [distance / 2, 0, 0]])
    assert np.isfinite(sim.potential_energy)
    assert torch.isfinite(sim.forces).all()


def test_h2_force_matches_finite_difference():
    distance = 0.9
    step = 1e-5
    def energy(value):
        sim = _simulation(["H", "H"], [[-value / 2, 0, 0], [value / 2, 0, 0]])
        return float(sim.potential_energy)
    sim = _simulation(["H", "H"], [[-distance / 2, 0, 0], [distance / 2, 0, 0]])
    derivative = (energy(distance + step) - energy(distance - step)) / (2 * step)
    assert float(sim.forces[1, 0]) == pytest.approx(-derivative, abs=2e-5)


def test_permutation_symmetry_ch4():
    coords = np.asarray([
        [0, 0, 0], [0.629, 0.629, 0.629], [-0.629, -0.629, 0.629],
        [-0.629, 0.629, -0.629], [0.629, -0.629, -0.629],
    ])
    first = _simulation(["C", "H", "H", "H", "H"], coords)
    order = [0, 3, 1, 4, 2]
    second = _simulation(["C", "H", "H", "H", "H"], coords[order])
    assert float(first.potential_energy) == pytest.approx(float(second.potential_energy), abs=1e-10)


def test_ch4_respects_carbon_capacity():
    coords = [[0, 0, 0], [0.629, 0.629, 0.629], [-0.629, -0.629, 0.629],
              [-0.629, 0.629, -0.629], [0.629, -0.629, -0.629]]
    sim = _simulation(["C", "H", "H", "H", "H"], coords)
    box = sim._unified_diagnostics["boxes"][0]
    assert max(np.asarray(box["usage"]) - np.asarray(box["capacity"])) < 1e-8


@pytest.mark.parametrize("geometry", _accepted_geometries())
def test_accepted_molecule_energy_is_preserved(geometry):
    symbols, positions = geometry
    positions = np.asarray(positions) + 20.0
    _, reference_force, reference_energy = _evaluate(
        OptimisedValenceStateBatchedSimulation, symbols, positions
    )
    _, candidate_force, candidate_energy = _evaluate(
        UnifiedBondCapacityEnergyPrototype, symbols, positions
    )
    torch.testing.assert_close(candidate_energy, reference_energy, atol=1e-8, rtol=0)
    torch.testing.assert_close(candidate_force, reference_force, atol=1e-6, rtol=0)


def test_crowded_force_matches_finite_difference():
    symbols, positions = _crowded_carbon(1.95)
    _, force, _ = _evaluate(UnifiedBondCapacityEnergyPrototype, symbols, positions)
    epsilon = 1e-5
    plus = positions.copy()
    minus = positions.copy()
    plus[-1, 0] += epsilon
    minus[-1, 0] -= epsilon
    plus_energy = _evaluate(UnifiedBondCapacityEnergyPrototype, symbols, plus)[2]
    minus_energy = _evaluate(UnifiedBondCapacityEnergyPrototype, symbols, minus)[2]
    numerical = -(float(plus_energy) - float(minus_energy)) / (2.0 * epsilon)
    assert float(force[-1, 0]) == pytest.approx(numerical, abs=1e-4)


@pytest.mark.parametrize("builder", [build_box.BUILDERS["H2"], build_box.BUILDERS["NH3"]])
def test_h2_and_ammonia_are_preserved(builder):
    symbols, positions = builder()
    positions = np.asarray(positions) + 20.0
    _, reference_force, reference_energy = _evaluate(
        OptimisedValenceStateBatchedSimulation, symbols, positions
    )
    _, candidate_force, candidate_energy = _evaluate(
        UnifiedBondCapacityEnergyPrototype, symbols, positions
    )
    torch.testing.assert_close(candidate_energy, reference_energy, atol=1e-8, rtol=0)
    torch.testing.assert_close(candidate_force, reference_force, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    ("symbols", "positions", "expected_order"),
    [
        (*bond_calibration.ethane_geometry(1.525), 1.0),
        (["C", "O"], [[0, 0, 0], [1.21, 0, 0]], 2.0),
        (["N", "N"], [[0, 0, 0], [1.10, 0, 0]], 3.0),
    ],
)
def test_single_double_triple_states_are_represented(symbols, positions, expected_order):
    simulation = _simulation(symbols, positions)
    order = simulation._unified_diagnostics["boxes"][0]["heavy_bond_orders"][0]
    assert order["expected_order"] == pytest.approx(expected_order, abs=1e-6)
