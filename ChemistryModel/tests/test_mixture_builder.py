import json

import pytest

import mixtures


def test_atom_metrics_match_runtime_density_formula():
    contents = {"C": 80, "H": 200, "N": 20, "O": 30}
    metrics = mixtures.composition_metrics("atoms", contents, 19.0)
    assert metrics["atoms"] == 330
    assert metrics["molecules"] is None
    assert metrics["elements"] == contents
    assert metrics["density_atoms_per_A3"] == pytest.approx(330 / 19.0 ** 3)


def test_molecule_metrics_use_actual_runtime_builders():
    contents = {"CH4": 2, "NH2": 3, "OH": 4}
    metrics = mixtures.composition_metrics("molecules", contents, 12.0)
    assert metrics["molecules"] == 9
    assert metrics["atoms"] == 10 + 9 + 8
    assert metrics["elements"] == {"C": 2, "H": 18, "N": 3, "O": 4}


def test_raw_definition_merges_duplicates_and_rejects_invalid_species():
    assert mixtures.parse_definition("C 5\nH 10\nC 2", "atoms") == {
        "C": 7, "H": 10
    }
    with pytest.raises(ValueError, match="positive integer"):
        mixtures.parse_definition("C 0", "atoms")
    with pytest.raises(ValueError, match="not a supported"):
        mixtures.parse_definition("Xe 2", "atoms")
    with pytest.raises(ValueError, match="expected"):
        mixtures.parse_definition("C", "atoms")


def test_legacy_custom_file_round_trips_without_changing_schema(tmp_path):
    path = tmp_path / "mixtures.json"
    legacy = {
        "Legacy": {"kind": "atoms", "contents": {"C": 5, "H": 12}}
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")
    loaded = mixtures.load_custom(path)
    assert loaded == {"Legacy": ("atoms", {"C": 5, "H": 12})}
    mixtures.save_custom(loaded, path)
    assert mixtures.load_custom(path) == loaded
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["Legacy"]["kind"] == "atoms"
    assert stored["Legacy"]["contents"] == {"C": 5, "H": 12}


def test_builtins_are_not_mutated_by_custom_save(tmp_path):
    before = dict(mixtures.BUILT_IN)
    path = tmp_path / "mixtures.json"
    mixtures.save_custom({"Experiment": ("atoms", {"O": 4})}, path)
    assert mixtures.BUILT_IN == before


def test_density_scaling_preserves_atom_ratios_and_hits_integer_target():
    contents = {"H": 50, "C": 20, "N": 5, "O": 5}
    standard = mixtures.scale_to_density("atoms", contents, 19.0, 0.040)
    assert standard["target_atoms"] == round(0.040 * 19.0 ** 3)
    assert standard["result_atoms"] == standard["target_atoms"]
    assert all(amount >= 1 for amount in standard["contents"].values())
    assert standard["contents"]["H"] / standard["contents"]["C"] == pytest.approx(
        2.5, rel=0.03
    )
    dilute = mixtures.scale_to_density("atoms", contents, 19.0, 0.010)
    assert dilute["result_atoms"] == dilute["target_atoms"]
    assert all(amount >= 1 for amount in dilute["contents"].values())


def test_density_scaling_warns_when_all_species_cannot_be_retained():
    with pytest.raises(ValueError, match="retaining all"):
        mixtures.scale_to_density(
            "molecules", {"CH4": 1, "NH3": 1, "H2O": 1}, 2.0, 0.01
        )


def test_structured_dialog_result_and_raw_reparse(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtWidgets
    from lab import MixtureDialog

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = MixtureDialog(
        name="Structured test", kind="atoms",
        contents={"C": 80, "H": 200, "N": 20, "O": 30},
        box_size=19,
    )
    assert dialog.result() == (
        "Structured test", "atoms", {"C": 80, "H": 200, "N": 20, "O": 30}
    )
    dialog._raw_dirty = True
    dialog.raw_contents.setPlainText("C 5\nH 10\nC 2")
    assert dialog.result() == ("Structured test", "atoms", {"C": 7, "H": 10})
    dialog.reject()
    dialog.deleteLater()
    application.processEvents()


def test_unsupported_legacy_entry_remains_visible_but_cannot_be_saved(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtWidgets
    from lab import MixtureDialog

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = MixtureDialog(
        name="Old definition", kind="molecules", contents={"LegacyX": 3}
    )
    assert dialog.row_contents() == {"LegacyX": 3}
    assert "unsupported legacy entry" in dialog.rows[0].species.currentText()
    with pytest.raises(ValueError, match="not a supported"):
        dialog.result()
    dialog.reject()
    dialog.deleteLater()
    application.processEvents()


def test_amount_slider_and_density_controls_stay_synchronised(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtWidgets
    from lab import MixtureDialog

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = MixtureDialog(
        name="Density test", kind="atoms",
        contents={"H": 50, "C": 20, "N": 5, "O": 5}, box_size=19,
    )
    first = dialog.rows[0]
    first.amount_slider.setValue(123)
    assert first.amount.value() == 123
    first.amount.setValue(800)
    assert first.amount_slider.value() == 800
    assert first.amount_slider.maximum() >= 800

    dialog.target_density.setValue(0.0335)
    assert dialog.density_preset.currentText() == "Custom"
    target = round(0.0335 * 19.0 ** 3)
    dialog.apply_target_density()
    assert mixtures.atom_count("atoms", dialog.row_contents()) == target
    assert dialog.target_density.value() == pytest.approx(0.0335)
    assert all(row.amount_slider.value() == row.amount.value() for row in dialog.rows)
    dialog.reject()
    dialog.deleteLater()
    application.processEvents()


def test_open_builder_tracks_parent_run_tab_box_size(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtWidgets
    from lab import Choice, MixtureDialog

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    parent = QtWidgets.QWidget()
    parent.box_size = Choice([19, 21], 19, 1)
    dialog = MixtureDialog(
        parent, name="Linked box", kind="atoms", contents={"H": 20},
        box_size=19,
    )
    assert dialog.actual_box_size() == 19
    parent.box_size.setValue(21)
    application.processEvents()
    assert dialog.actual_box_size() == 21
    preview = mixtures.scale_to_density("atoms", {"H": 20}, 21, 0.040)
    assert f"{preview['target_atoms']:,}" in dialog.density_preview.text()
    dialog.reject()
    dialog.deleteLater()
    parent.deleteLater()
    application.processEvents()
