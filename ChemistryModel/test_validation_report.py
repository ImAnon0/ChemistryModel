"""Fast invariants for the independent validation report."""

import validation_report as V


def test_fit_targets_are_never_presented_as_independent_validation():
    rows = V.geometry_rows(V.FIT_GEOMETRY, True)
    assert rows
    assert all(row["classification"] == "fit target" for row in rows)
    assert all(row["status"] == V.STATUS["fit"] for row in rows)


def test_holdout_geometry_is_separate_and_quantitative():
    rows = V.geometry_rows(V.HOLDOUT_GEOMETRY, False)
    assert rows
    assert all(row["classification"] == "hold-out" for row in rows)
    assert all(row["percent_error"] >= 0.0 for row in rows)


def test_numpy_and_torch_parameter_tables_are_identical():
    result = V.torch_consistency()
    assert result["status"] == V.STATUS["good"]
    assert result["maximum_absolute_difference"] == 0.0


def test_parameter_audit_covers_all_live_bond_order_rows():
    result = V.parameter_audit()
    expected = len(V.R.BOND_TABLE) + len(V.R.DOUBLE_BOND_TABLE) + len(V.R.TRIPLE_BOND_TABLE)
    assert len(result["pairs"]) == expected


def test_quick_report_refuses_fake_overall_accuracy_number():
    report = V.build_report("quick")
    assert report["rules"]["overall_accuracy_percentage"] is None
    assert report["rules"]["force_field_modified"] is False
    assert report["holdout_geometry"]


def test_comparison_reports_directional_metric_changes():
    old = {"git_revision": "old", "holdout_geometry": [
        {"molecule": "ethanol", "percent_error": 2.0}
    ]}
    new = {"git_revision": "new", "holdout_geometry": [
        {"molecule": "ethanol", "percent_error": 1.5}
    ]}
    result = V.compare_reports(old, new)
    assert result["changes"][0]["difference"] == -0.5


def test_json_safe_normalizes_numpy_values_recursively():
    result = V.json_safe({"array": V.np.array([1.0]), "scalar": V.np.float64(2.0)})
    assert result == {"array": [1.0], "scalar": 2.0}


def test_pair_depth_estimates_are_explicitly_diagnostics():
    rows = V.bond_depth_diagnostics()
    assert rows
    assert all("diagnostic" in row.get("note", "").lower() for row in rows)


def test_quick_report_keeps_whole_model_energy_separate():
    report = V.build_report("quick")
    assert "bond_depth_diagnostics" in report
    assert "whole_model_reaction_energies" in report
    assert "reaction_thermochemistry" not in report


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print("PASS ", test.__name__)
        except Exception as problem:
            failures += 1
            print("FAIL ", test.__name__, problem)
    print(f"\n{len(tests)-failures} passed, {failures} failed")
    raise SystemExit(bool(failures))
