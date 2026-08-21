from __future__ import annotations

import numpy as np
import pytest
import torch
import json
from pathlib import Path

from research.heavy_valence_continuous_edge.tests.test_continuous_edge import (
    _crowded_carbon,
    _evaluate,
)
from research.unified_bond_geometry import (
    PostSolvedWeightedGeometryPrototype,
    VariationalElectronDomainGeometryPrototype,
    VariationalJointGeometryStatePrototype,
    VariationalWeightedGeometryPrototype,
)


def test_all_geometry_models_are_research_only():
    assert PostSolvedWeightedGeometryPrototype.research_only
    assert VariationalWeightedGeometryPrototype.research_only
    assert VariationalElectronDomainGeometryPrototype.research_only
    assert VariationalJointGeometryStatePrototype.research_only


@pytest.mark.parametrize(
    "model",
    [VariationalWeightedGeometryPrototype, VariationalElectronDomainGeometryPrototype],
)
def test_crowded_force_is_finite(model):
    symbols, positions = _crowded_carbon(1.95)
    simulation, force, energy = _evaluate(model, symbols, positions)
    assert torch.isfinite(force).all()
    assert torch.isfinite(energy)
    solver = simulation._geometry_diagnostics["boxes"][0]["solver"]
    assert solver["simplex_violation"] < 2e-7
    assert solver["capacity_violation"] < 2e-7


def test_weighted_geometry_force_matches_finite_difference():
    symbols, positions = _crowded_carbon(1.95)
    _, force, _ = _evaluate(
        VariationalWeightedGeometryPrototype, symbols, positions
    )
    epsilon = 1e-5
    plus = positions.copy()
    minus = positions.copy()
    plus[-1, 0] += epsilon
    minus[-1, 0] -= epsilon
    plus_energy = _evaluate(
        VariationalWeightedGeometryPrototype, symbols, plus
    )[2]
    minus_energy = _evaluate(
        VariationalWeightedGeometryPrototype, symbols, minus
    )[2]
    numerical = -(float(plus_energy) - float(minus_energy)) / (2.0 * epsilon)
    assert float(force[-1, 0]) == pytest.approx(numerical, abs=2e-4)


def _water_transfer_geometry():
    rows = json.loads(Path(
        "research_data/qm_residual/dense_scan_geometries.json"
    ).read_text(encoding="utf-8"))["geometries"]
    row = min(
        (item for item in rows if item["system"] == "water"
         and item["sample_kind"] == "dense_transfer_scan"),
        key=lambda item: abs(float(
            item["reaction_coordinate"]["transfer_distance_angstrom"]
        ) - 1.16),
    )
    return row["symbols"], np.asarray(row["coordinates_angstrom"], dtype=float)


def test_joint_geometry_force_is_finite():
    symbols, positions = _water_transfer_geometry()
    simulation, force, energy = _evaluate(
        VariationalJointGeometryStatePrototype, symbols, positions
    )
    assert torch.isfinite(force).all()
    assert torch.isfinite(energy)
    assert simulation._geometry_diagnostics["boxes"][0]["solver"][
        "equality_violation"
    ] < 2e-7


def test_joint_geometry_force_matches_finite_difference():
    symbols, positions = _water_transfer_geometry()
    _, force, _ = _evaluate(
        VariationalJointGeometryStatePrototype, symbols, positions
    )
    epsilon = 1e-5
    plus = positions.copy()
    minus = positions.copy()
    plus[-1, 0] += epsilon
    minus[-1, 0] -= epsilon
    plus_energy = _evaluate(
        VariationalJointGeometryStatePrototype, symbols, plus
    )[2]
    minus_energy = _evaluate(
        VariationalJointGeometryStatePrototype, symbols, minus
    )[2]
    numerical = -(float(plus_energy) - float(minus_energy)) / (2.0 * epsilon)
    assert float(force[-1, 0]) == pytest.approx(numerical, abs=3e-4)
