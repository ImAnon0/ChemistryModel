from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.electronic_observables.build_manifest import build_manifest
from research.electronic_observables.prototype import (
    ElectronicObservableRecord,
    cancellation_warning,
    compare_candidate_rows,
    relative_energies,
    vector_rmse,
)
from research.electronic_observables.validate_dataset import quality_checks


ROOT = Path(__file__).resolve().parents[3]
DENSE = ROOT / "research_data/qm_residual/dense_scan_geometries.json"


def _record(geometry_id, energy, reference="a", family="family"):
    return ElectronicObservableRecord(
        geometry_id=geometry_id,
        family=family,
        reference_geometry_id=reference,
        energy_eV=energy,
    )


def test_manifest_is_small_versioned_and_has_real_holdouts():
    payload = build_manifest(DENSE)
    rows = payload["geometries"]
    ids = {row["geometry_id"] for row in rows}
    assert payload["schema_version"] == 1
    assert len(rows) == 30
    assert len(ids) == len(rows)
    assert {row["role"] for row in rows} == {
        "characterisation", "validation", "final_holdout"
    }
    assert all(row["reference_geometry_id"] in ids for row in rows)
    assert {row["family"] for row in rows} >= {
        "h3_transfer", "water_transfer", "water_angle", "oh_stretch",
        "h_h2_approach", "h_formaldehyde_transfer", "formaldehyde", "n2",
    }


def test_relative_energy_uses_composition_reference():
    rows = [_record("a", -10.0), _record("b", -8.5)]
    assert relative_energies(rows) == {"a": 0.0, "b": 1.5}


def test_comparison_reports_regression_even_if_signed_direction_changes():
    qm = [_record("a", 0.0), _record("b", 1.0)]
    baseline = [_record("a", 5.0), _record("b", 6.4)]
    candidate = [_record("a", -3.0), _record("b", -1.0)]
    rows = compare_candidate_rows(qm, baseline, candidate)
    by_id = {row.geometry_id: row for row in rows}
    assert np.isclose(by_id["b"].baseline_residual_eV, 0.4)
    assert np.isclose(by_id["b"].candidate_residual_eV, 1.0)
    assert by_id["b"].classification == "regressed"


def test_cancellation_warning_and_vector_rmse():
    qm = [_record("a", 0.0), _record("b", 0.0)]
    baseline = [_record("a", 0.0), _record("b", 2.0)]
    candidate = [_record("a", 0.0), _record("b", -2.5)]
    comparisons = compare_candidate_rows(qm, baseline, candidate)
    assert not cancellation_warning(comparisons)
    assert np.isclose(vector_rmse([[1, 2]], [[1, 4]]), np.sqrt(2.0))


def test_quality_checks_conserve_charge_and_preserve_symmetry():
    manifest = {
        "geometries": [{
            "geometry_id": "h2_equilibrium",
            "charge": 0,
            "symbols": ["H", "H"],
        }]
    }
    results = {
        "records": [{
            "geometry_id": "h2_equilibrium",
            "status": "ok",
            "core_status": "ok",
            "mulliken_charges_e": [0.0, 0.0],
            "lowdin_charges_e": [0.0, 0.0],
            "mbis_charges_e": [0.0, 0.0],
            "force_eV_per_angstrom": [[0.1, 0, 0], [-0.1, 0, 0]],
            "mbis_dipole_reconstruction_error_au": 0.0,
            "dipole_magnitude_debye": 0.0,
            "polarizability_status": "ok",
            "polarizability": {
                "antisymmetric_norm_au": 0.0,
                "eigenvalues_au": [1.0, 1.0, 2.0],
            },
        }]
    }
    report = quality_checks(manifest, results)
    assert report["complete"]
    assert report["all_available_checks_pass"]
