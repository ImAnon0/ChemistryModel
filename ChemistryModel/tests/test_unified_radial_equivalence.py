from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import batch_runner
import characterisation_runner
from physics_provenance import physics_source_identity
from research.benchmark import benchmark_reaction_barriers as benchmark
from research.unified_bond_capacity import UnifiedBondCapacityEnergyPrototype
from research.unified_bond_capacity import LegacyUnifiedRadialReference
from unified_radial_equivalence import (
    CUDA_TOLERANCES,
    build_simulation,
    compare_implementation_to_fixture,
    compare_implementations,
    compare_values,
    evaluate_cases,
    load_fixture,
)


def test_unified_radial_is_explicit_opt_in_everywhere():
    options = SimpleNamespace(physics="unified-radial")
    assert (
        batch_runner.grouped_simulation_class(options)
        is UnifiedBondCapacityEnergyPrototype
    )
    assert (
        characterisation_runner.simulation_class_for_physics("unified-radial")
        is UnifiedBondCapacityEnergyPrototype
    )
    assert (
        benchmark.resolve_physics("unified-radial")["class"]
        is UnifiedBondCapacityEnergyPrototype
    )
    assert UnifiedBondCapacityEnergyPrototype.model_id == "unified_radial_v1"


def test_existing_selector_defaults_are_unchanged():
    assert benchmark._normalised_physics_name(None) == "base"
    assert characterisation_runner.simulation_class_for_physics("standard")
    source = Path("batch_runner.py").read_text(encoding="utf-8")
    assert 'default="reactive"' in source


def test_effective_source_manifest_captures_copied_research_dependencies():
    first = physics_source_identity(UnifiedBondCapacityEnergyPrototype)
    second = physics_source_identity(UnifiedBondCapacityEnergyPrototype)
    assert first == second
    required = {
        "reactive.py",
        "reactive_torch.py",
        "h_state_reference.py",
        "research/unified_bond_capacity/prototype.py",
        "research/heavy_valence_bond_channels/bond_state_hamiltonian.py",
        "research/heavy_valence_state/energy_common.py",
    }
    assert required <= set(first["files"])
    assert not any("site-packages" in path for path in first["files"])


def test_characterisation_metadata_persists_model_and_source_identity():
    metadata = characterisation_runner.physics_metadata("unified-radial")
    assert metadata["physics_model_id"] == "unified_radial_v1"
    assert len(metadata["physics_source_sha256"]) == 64
    assert metadata["physics_source_algorithm"] == (
        "chemistrymodel-effective-sources-v1"
    )
    assert len(metadata["physics_parameter_sha256"]) == 64
    assert metadata["physics_capacity_solver"] == (
        "existing_scipy_l_bfgs_b_dual"
    )
    assert "research/unified_bond_capacity/prototype.py" in (
        metadata["physics_source_files"]
    )
    fingerprint = benchmark.physics_fingerprint("unified-radial")
    assert fingerprint["physics_model_id"] == "unified_radial_v1"
    assert fingerprint["source_sha256"] == metadata["physics_source_sha256"]
    assert fingerprint["parameter_sha256"] == metadata[
        "physics_parameter_sha256"
    ]
    assert fingerprint["source_algorithm"] == metadata[
        "physics_source_algorithm"
    ]


def test_batch_run_provenance_includes_source_and_parameter_identities():
    case = next(
        case for case in load_fixture()["cases"]
        if case["name"] == "h2_equilibrium"
    )
    simulation = build_simulation(
        UnifiedBondCapacityEnergyPrototype, [case], device="cpu",
        dtype=torch.float64, box_size=40.0,
    )
    metadata = batch_runner.simulation_physics_provenance(simulation)
    assert metadata["physics_model_id"] == "unified_radial_v1"
    assert metadata["physics_source_sha256"] == physics_source_identity(
        UnifiedBondCapacityEnergyPrototype
    )["sha256"]
    assert metadata["physics_parameter_sha256"] == (
        simulation.chemistry_physics_spec.parameter_sha256
    )


def test_frozen_cpu_reference_matches_single_and_grouped_execution():
    assert compare_implementation_to_fixture() == []


def test_frozen_fixture_retains_pre_extraction_source_provenance():
    fixture = load_fixture()
    assert fixture["model_id"] == "unified_radial_v1"
    assert fixture["source_identity"]["algorithm"] == (
        "chemistrymodel-effective-sources-v1"
    )
    assert len(fixture["source_identity"]["sha256"]) == 64
    current = physics_source_identity(UnifiedBondCapacityEnergyPrototype)
    assert "chemistry_engine/hamiltonian.py" in current["files"]
    assert "research/unified_bond_capacity/prototype.py" in current["files"]


def test_canonical_engine_matches_retained_legacy_route():
    fixture = load_fixture()
    assert compare_implementations(
        LegacyUnifiedRadialReference,
        UnifiedBondCapacityEnergyPrototype,
        fixture["cases"],
    ) == []


def test_observational_capture_does_not_change_energy_or_forces():
    case = next(
        case for case in load_fixture()["cases"]
        if case["name"] == "water_transfer_midpoint"
    )
    plain = build_simulation(
        UnifiedBondCapacityEnergyPrototype, [case], device="cpu",
        dtype=torch.float64, box_size=40.0,
        capture_equivalence_state=False,
    )
    observed = build_simulation(
        UnifiedBondCapacityEnergyPrototype, [case], device="cpu",
        dtype=torch.float64, box_size=40.0,
        capture_equivalence_state=True,
    )
    assert torch.equal(plain._potential_per_atom, observed._potential_per_atom)
    assert torch.equal(plain.forces, observed.forces)


def test_fixture_covers_required_scientific_categories_and_state():
    fixture = load_fixture()
    categories = {case["category"] for case in fixture["cases"]}
    assert {
        "small_system", "stable_molecule", "water_transfer",
        "grambow_representative",
    } <= categories
    names = {case["name"] for case in fixture["cases"]}
    assert {"h2_equilibrium", "h3_symmetric", "h_plus_h2"} <= names
    snapshot = fixture["reference"]["single"]["water_transfer_midpoint"]
    assert snapshot["energy_components_eV"]
    assert snapshot["per_box_energy_eV"] == snapshot["total_energy_eV"]
    assert snapshot["forces_eV_per_angstrom"]
    assert "lambda_eV" in snapshot["state"]
    assert "solver" in snapshot["state"]
    assert "h_state_probabilities" in snapshot["state"]
    assert snapshot["membership"]


def test_comparator_rejects_a_scientific_change():
    fixture = load_fixture()
    expected = fixture["reference"]["single"]
    altered = deepcopy(expected)
    altered["water_transfer_midpoint"]["total_energy_eV"] += 1e-6
    differences = compare_values(
        expected, altered, fixture["tolerances"]["cpu_float64"]
    )
    assert any("water_transfer_midpoint" in item for item in differences)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_single_and_grouped_reference_agree():
    cases = [
        case for case in load_fixture()["cases"]
        if case["name"] in {"h3_symmetric", "h_plus_h2"}
    ]
    single = evaluate_cases(
        UnifiedBondCapacityEnergyPrototype, cases, device="cuda",
        dtype=torch.float32, grouped=False,
    )
    grouped = evaluate_cases(
        UnifiedBondCapacityEnergyPrototype, cases, device="cuda",
        dtype=torch.float32, grouped=True,
    )
    assert compare_values(single, grouped, CUDA_TOLERANCES) == []
    assert compare_implementations(
        LegacyUnifiedRadialReference,
        UnifiedBondCapacityEnergyPrototype,
        cases,
        device="cuda",
        dtype=torch.float32,
        tolerances=CUDA_TOLERANCES,
    ) == []
