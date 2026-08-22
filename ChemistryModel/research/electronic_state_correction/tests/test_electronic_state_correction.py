from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from research.electronic_state_correction import (
    CombinedElectronicStatePrototype,
    LocalElectronicDescriptorPrototype,
    MultipoleDensityPrototype,
    PolarisationResponsePrototype,
)
from research.electronic_state_correction.prototype import (
    numpy_electronic_features,
)
from research.heavy_valence_continuous_edge.tests.test_continuous_edge import (
    _evaluate,
)
from research.unified_bond_capacity import UnifiedBondCapacityEnergyPrototype


ROOT = Path(__file__).resolve().parents[3]
PARAMETERS = ROOT / "research_data/benchmark/diagnostics/electronic_state_parameters.json"
SOURCE = {"H": 0.0, "C": -0.02, "N": 0.0, "O": 0.35}


def test_settled_and_separated_contacts_have_zero_features():
    settled = numpy_electronic_features(
        ["O", "H", "H"],
        [[0, 0, 0], [0.9572, 0, 0], [-0.24, 0.926, 0]],
        SOURCE,
    )
    separated = numpy_electronic_features(
        ["O", "H"], [[0, 0, 0], [5, 0, 0]], SOURCE
    )
    assert max(abs(value) for value in settled.values()) == 0.0
    assert max(abs(value) for value in separated.values()) == 0.0


def test_numpy_features_are_permutation_and_rotation_invariant():
    symbols = ["O", "H", "C"]
    positions = np.asarray([[0, 0, 0], [1.30, 0.2, 0], [-1.4, 0.1, 0.3]])
    reference = numpy_electronic_features(symbols, positions, SOURCE)
    permutation = [2, 0, 1]
    permuted = numpy_electronic_features(
        [symbols[index] for index in permutation], positions[permutation], SOURCE
    )
    angle = 0.71
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    rotated = numpy_electronic_features(symbols, positions @ rotation.T, SOURCE)
    for key in reference:
        assert np.isclose(permuted[key], reference[key], atol=1e-14)
        assert np.isclose(rotated[key], reference[key], atol=1e-14)


def test_zero_response_is_exact_unified_identity():
    symbols = ["O", "H", "O", "H", "H"]
    positions = np.asarray([
        [0, 0, 0], [1.03, 0, 0], [2.33, 0, 0],
        [-0.2, 0.94, 0], [2.55, 0.93, 0],
    ]) + 20.0
    _, base_force, base_energy = _evaluate(
        UnifiedBondCapacityEnergyPrototype, symbols, positions
    )
    _, force, energy = _evaluate(
        partial(
            LocalElectronicDescriptorPrototype,
            electronic_source_values=SOURCE,
            electronic_coefficients={"local_scalar": 0.0},
        ),
        symbols,
        positions,
    )
    assert float(energy) == float(base_energy)
    assert torch.equal(force, base_force)


def test_torch_and_numpy_feature_totals_match():
    symbols = ["O", "H", "C"]
    raw = np.asarray([[0, 0, 0], [1.30, 0.2, 0], [-1.4, 0.1, 0.3]])
    simulation, _, _ = _evaluate(
        partial(
            CombinedElectronicStatePrototype,
            electronic_source_values=SOURCE,
            electronic_coefficients={
                "local_scalar": 0.0,
                "polarisation_vector": 0.0,
                "multipole_tensor": 0.0,
            },
        ),
        symbols,
        raw + 20.0,
    )
    expected = numpy_electronic_features(symbols, raw, SOURCE)
    actual = simulation._electronic_state_diagnostics["feature_totals"]
    for key in expected:
        assert np.isclose(actual[key], expected[key], atol=1e-12)


def test_fitted_parameters_never_read_grambow_and_water_is_holdout():
    payload = json.loads(PARAMETERS.read_text(encoding="utf-8"))
    assert "no Grambow targets read" in payload["scope"]
    assert payload["source_fit"]["unidentifiable_elements"] == ["N"]
    for hypothesis in payload["hypotheses"].values():
        assert hypothesis["energy_fit_systems"] == [
            "formaldehyde", "methane alternating even-index points"
        ]
        assert "water" in hypothesis["energy_holdout_systems"]
        assert hypothesis["fit_observations"] > hypothesis["parameter_count"]


def test_all_candidate_classes_are_research_only():
    for model in (
        LocalElectronicDescriptorPrototype,
        PolarisationResponsePrototype,
        MultipoleDensityPrototype,
        CombinedElectronicStatePrototype,
    ):
        assert model.research_only is True
        assert model.physics_model_name.startswith("research_")
