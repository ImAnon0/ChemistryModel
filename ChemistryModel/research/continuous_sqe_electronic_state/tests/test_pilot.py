from __future__ import annotations

import hashlib
import json

import numpy as np

from research.continuous_sqe_electronic_state.fit_pilot import (
    DATA, C0_PARAMETERS, _load, _predict, _predict_fast, _vector,
)


def test_pilot_manifest_is_exactly_ten_percent_and_family_blocked():
    manifest, records = _load()
    assert len(records) == 235
    family_splits = {}
    for record in records:
        family_splits.setdefault(record["family"], set()).add(record["split"])
    assert all(len(splits) == 1 for splits in family_splits.values())
    holdout = [row for row in manifest["geometries"] if row["split"] == "locked_holdout"]
    canonical = json.dumps(holdout, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == manifest["locked_holdout_sha256"]


def test_all_pilot_qm_observables_are_complete():
    _, records = _load()
    assert all(record["qm"]["status"] == "ok" for record in records)
    assert all("external_potential" in record["qm"] for record in records)


def test_fast_fitting_kernel_matches_reference_solver():
    _, records = _load()
    values = _vector(C0_PARAMETERS)
    for record in records[::47]:
        reference = _predict(record, values)
        fast = _predict_fast(record, values)
        assert np.allclose(reference["dipole"], fast["dipole"], atol=1e-12, rtol=0)
        assert np.allclose(reference["alpha"], fast["alpha"], atol=1e-12, rtol=0)
        assert np.allclose(reference["charges"], fast["charges"], atol=1e-12, rtol=0)


def test_fitted_outputs_are_research_only_and_holdout_was_sealed():
    parameters = json.loads((DATA / "fitted_parameters.json").read_text(encoding="utf-8"))
    results = json.loads((DATA / "pilot_results.json").read_text(encoding="utf-8"))
    assert parameters["independent_parameter_count"] == 17
    assert results["research_only"] is True
    assert results["production_integration"] is False
    assert results["locked_holdout_sha256"]
