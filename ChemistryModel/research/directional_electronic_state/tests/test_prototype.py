from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.directional_electronic_state import (
    StateConditionedP2CouplingPrototype,
)
from research.heavy_valence_continuous_edge.tests.test_continuous_edge import (
    _accepted_geometries,
    _evaluate,
)
from research.unified_bond_capacity import UnifiedBondCapacityEnergyPrototype


def _water_transfer_geometry():
    rows = json.loads(Path(
        "research_data/qm_residual/dense_scan_geometries.json"
    ).read_text(encoding="utf-8"))["geometries"]
    row = min(
        (
            item for item in rows
            if item["system"] == "water"
            and item["sample_kind"] == "dense_transfer_scan"
        ),
        key=lambda item: abs(float(
            item["reaction_coordinate"]["transfer_distance_angstrom"]
        ) - 1.16),
    )
    return row["symbols"], np.asarray(row["coordinates_angstrom"], dtype=float) + 20.0


def test_probe_is_research_only():
    assert StateConditionedP2CouplingPrototype.research_only


def test_boolean_off_switch_is_exact_unified_radial_control():
    symbols, positions = _water_transfer_geometry()
    _, reference_force, reference_energy = _evaluate(
        UnifiedBondCapacityEnergyPrototype, symbols, positions
    )
    simulation = StateConditionedP2CouplingPrototype(
        boxes=[(symbols, positions)], box_size=40.0,
        target_temperature=0.0, friction=0.0, device="cpu",
        dtype=torch.float64, random_seed=0, relax_on_start=False,
        directional_response=False,
    )
    force, energy = simulation.compute_forces()
    torch.testing.assert_close(energy, reference_energy, atol=1e-10, rtol=0)
    torch.testing.assert_close(force, reference_force, atol=1e-9, rtol=0)


@pytest.mark.parametrize("geometry", _accepted_geometries())
def test_settled_molecules_are_exactly_preserved(geometry):
    symbols, positions = geometry
    positions = np.asarray(positions, dtype=float) + 20.0
    _, reference_force, reference_energy = _evaluate(
        UnifiedBondCapacityEnergyPrototype, symbols, positions
    )
    _, force, energy = _evaluate(
        StateConditionedP2CouplingPrototype, symbols, positions
    )
    torch.testing.assert_close(energy, reference_energy, atol=1e-9, rtol=0)
    torch.testing.assert_close(force, reference_force, atol=1e-8, rtol=0)


def test_water_transfer_has_a_live_directional_signal():
    symbols, positions = _water_transfer_geometry()
    simulation, _, candidate_energy = _evaluate(
        StateConditionedP2CouplingPrototype, symbols, positions
    )
    _, _, reference_energy = _evaluate(
        UnifiedBondCapacityEnergyPrototype, symbols, positions
    )
    assert abs(float(candidate_energy - reference_energy)) > 1e-8
    diagnostics = simulation._directional_electronic_diagnostics
    assert diagnostics["transition_count"] > 0
    assert diagnostics["minimum_transition_factor"] != pytest.approx(1.0, abs=1e-8)


def test_water_force_matches_finite_difference():
    symbols, positions = _water_transfer_geometry()
    _, force, _ = _evaluate(
        StateConditionedP2CouplingPrototype, symbols, positions
    )
    epsilon = 1e-5
    plus = positions.copy()
    minus = positions.copy()
    plus[-1, 0] += epsilon
    minus[-1, 0] -= epsilon
    plus_energy = _evaluate(
        StateConditionedP2CouplingPrototype, symbols, plus
    )[2]
    minus_energy = _evaluate(
        StateConditionedP2CouplingPrototype, symbols, minus
    )[2]
    numerical = -(float(plus_energy) - float(minus_energy)) / (2.0 * epsilon)
    assert float(force[-1, 0]) == pytest.approx(numerical, abs=2e-4)


def test_water_permutation_symmetry():
    symbols, positions = _water_transfer_geometry()
    _, force, energy = _evaluate(
        StateConditionedP2CouplingPrototype, symbols, positions
    )
    oxygen = [index for index, symbol in enumerate(symbols) if symbol == "O"]
    permutation = np.arange(len(symbols))
    permutation[oxygen[0]], permutation[oxygen[1]] = (
        permutation[oxygen[1]], permutation[oxygen[0]]
    )
    _, permuted_force, permuted_energy = _evaluate(
        StateConditionedP2CouplingPrototype,
        [symbols[index] for index in permutation],
        positions[permutation],
    )
    assert float(permuted_energy) == pytest.approx(float(energy), abs=1e-9)
    torch.testing.assert_close(
        permuted_force, force[torch.as_tensor(permutation)], atol=1e-8, rtol=0
    )
