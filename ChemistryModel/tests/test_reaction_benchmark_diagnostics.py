import json
import sys

from research.benchmark import benchmark_reaction_barriers as benchmark
from research.diagnostics import probe_oo_topology_exclusion as oo_probe
from reactive_torch import ReactiveSimulation
from valence_state_optimised_torch import (
    OptimisedValenceStateBatchedSimulation,
)


def _endpoint(reaction_id, region, positions):
    return {
        "reaction_id": reaction_id,
        "region": region,
        "symbols": list(oo_probe.SYMBOLS),
        "coordinates_angstrom": positions,
        "reference_energy_eV": 0.0,
    }


def test_benchmark_physics_selector_uses_requested_concrete_class():
    assert benchmark.resolve_physics("base")["class"] is ReactiveSimulation
    assert (
        benchmark.resolve_physics("optimised-valence")["class"]
        is OptimisedValenceStateBatchedSimulation
    )
    base = benchmark.physics_fingerprint("base")
    full = benchmark.physics_fingerprint("optimised-valence")
    assert base["simulation_class"] == "reactive_torch.ReactiveSimulation"
    assert full["simulation_class"].endswith(
        ".OptimisedValenceStateBatchedSimulation"
    )
    assert full["h_state_active"] is True
    assert full["physics_model_revision"] == 1
    assert base["fingerprint_sha256"] != full["fingerprint_sha256"]


def test_topology_fingerprint_records_diagnostic_exclusion():
    current = benchmark.physics_fingerprint("base", "current")
    excluded = benchmark.physics_fingerprint("base", "exclude-oo")
    assert current["effective_topology_exclusions"] == []
    assert ["O", "O"] in excluded["effective_topology_exclusions"]
    assert current["fingerprint_sha256"] != excluded["fingerprint_sha256"]


def test_oo_probe_changes_only_topology_derived_terms():
    result = oo_probe.run_probe()
    assert result["passed"] is True
    assert result["checks"] == {
        "radial_taper_identical": True,
        "radial_energy_identical": True,
        "oo_topology_changed": True,
        "excluded_oo_topology_is_zero": True,
        "overcoordination_changed": True,
        "angle_changed": True,
    }


def test_ab_comparison_has_zero_control_and_responds_to_oo_exclusion(tmp_path):
    positions = oo_probe.POSITIONS.tolist()
    shifted = oo_probe.POSITIONS.copy()
    shifted[3:, 0] -= 0.25
    separated = oo_probe.POSITIONS.copy()
    separated[3:, 0] += 0.45
    payload = {
        "geometries": [
            _endpoint("probe", "reactant", separated.tolist()),
            _endpoint("probe", "transition_state", positions),
            _endpoint("probe", "product", shifted.tolist()),
        ]
    }
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    current, failures, _ = benchmark.score(
        path, limit=None, box_size=20.0, device="cpu",
        physics="base", topology="current",
    )
    same, _, _ = benchmark.score(
        path, limit=None, box_size=20.0, device="cpu",
        physics="base", topology="current",
    )
    excluded, excluded_failures, _ = benchmark.score(
        path, limit=None, box_size=20.0, device="cpu",
        physics="base", topology="exclude-oo",
    )
    assert failures == excluded_failures == []
    control = benchmark.compare_rows(current, same)
    changed = benchmark.compare_rows(current, excluded)
    assert control[0]["barrier_delta_eV"] == 0.0
    assert control[0]["reaction_delta_eV"] == 0.0
    assert (
        abs(changed[0]["barrier_delta_eV"]) > 1e-9
        or abs(changed[0]["reaction_delta_eV"]) > 1e-9
    )


def test_summary_persists_physics_revision_and_topology(tmp_path, monkeypatch):
    h2 = {
        "symbols": ["H", "H"],
        "coordinates_angstrom": [[0, 0, 0], [.74, 0, 0]],
        "reference_energy_eV": 0.0,
    }
    payload = {"geometries": [
        {**h2, "reaction_id": "h2", "region": "reactant"},
        {**h2, "reaction_id": "h2", "region": "transition_state"},
        {**h2, "reaction_id": "h2", "region": "product"},
    ]}
    source = tmp_path / "input.json"
    output = tmp_path / "scores.csv"
    summary = tmp_path / "summary.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "benchmark_reaction_barriers.py",
        "--input", str(source),
        "--output", str(output),
        "--summary", str(summary),
        "--device", "cpu",
        "--physics", "optimised-valence",
        "--topology", "current",
    ])

    benchmark.main()

    stored = json.loads(summary.read_text(encoding="utf-8"))
    fingerprint = stored["physics_fingerprint"]
    assert stored["physics"] == "optimised-valence"
    assert stored["topology"] == "current"
    assert fingerprint["physics_model_revision"] == 1
    assert fingerprint["git_revision"]
    assert fingerprint["source_sha256"]

