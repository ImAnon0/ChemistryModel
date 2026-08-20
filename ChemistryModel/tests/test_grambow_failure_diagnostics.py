import json

import pytest
import torch

from research.benchmark import diagnose_grambow_failures as failure
from research.benchmark.microscope_grambow_reactions import (
    _endpoint_diagnostics,
)


def test_score_loader_detects_csv_despite_json_suffix(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(
        "reaction_id,atom_count,barrier_error_eV\nrxn1,2,1.25\n",
        encoding="utf-8",
    )
    rows, detected = failure._load_score_rows(path)
    assert detected == "csv"
    assert rows[0]["reaction_id"] == "rxn1"


def test_score_loader_accepts_real_json(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(
        json.dumps({"rows": [{"reaction_id": "rxn2"}]}),
        encoding="utf-8",
    )
    rows, detected = failure._load_score_rows(path)
    assert detected == "json"
    assert rows == [{"reaction_id": "rxn2"}]


def test_geometric_classifier_identifies_h_transfer_without_model_claim():
    reactant = {
        "symbols": ["O", "H", "H"],
        "coordinates_angstrom": [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [3.0, 0.0, 0.0]],
    }
    product = {
        "symbols": ["O", "H", "H"],
        "coordinates_angstrom": [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
    }
    before = failure._guess_bonds(reactant)
    after = failure._guess_bonds(product)
    change = failure._change_summary(before, after)
    primary, tags = failure._classify(reactant, product, change)
    assert primary == "H transfer (O-H -> H-H)"
    assert "H transfer" in tags


def test_microscope_decomposition_is_observational_and_exact():
    geometry = {
        "geometry_id": "synthetic/h2",
        "symbols": ["H", "H"],
        "coordinates_angstrom": [[0.0, 0.0, 0.0], [0.74144, 0.0, 0.0]],
    }
    result = _endpoint_diagnostics(geometry, "cpu")
    components = result["components_eV"]
    assert components["composition_residual"] == pytest.approx(0.0, abs=1e-12)
    assert components["reported_potential"] == pytest.approx(
        components["base_total"]
        + components["h_state_correction"]
        + components["valence_topology_correction"],
        abs=1e-12,
    )
    assert torch.isfinite(torch.tensor(components["total"]))
