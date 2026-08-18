import json

import numpy as np

import molecule_library
import qm_structure_validator as qsv


def _stored_water(root):
    root.mkdir(parents=True, exist_ok=True)
    symbols = np.asarray(["O", "H", "H"], dtype="U2")
    positions = np.asarray([
        [0.000000, 0.000000, 0.000000],
        [0.957200, 0.000000, 0.000000],
        [-0.239987, 0.927297, 0.000000],
    ], dtype=np.float32)
    bonds = np.asarray([[0, 1], [0, 2]], dtype=np.int32)
    np.savez_compressed(
        root / "SP_000001.npz", symbols=symbols, positions=positions,
        bonds=bonds, source_atom_ids=np.arange(3, dtype=np.uint32),
        source_slots=np.arange(3, dtype=np.int32),
    )
    metadata = {
        "id": "SP_000001", "formula": "H2O", "atoms": 3,
        "heavy_atoms": 1, "payload": "SP_000001.npz",
        "source": {"recording": "run.npz", "seed": 7, "frame": 12},
    }
    (root / "SP_000001.json").write_text(json.dumps(metadata), encoding="utf-8")
    return positions.astype(np.float64), bonds


class FakeRunner:
    def run(self, symbols, coordinates, charge, multiplicity, method, basis, reference):
        assert symbols == ["O", "H", "H"]
        assert charge == 0
        assert multiplicity == 1
        moved = np.asarray(coordinates, dtype=np.float64).copy()
        moved[1] = moved[0] + 0.96 * (moved[1] - moved[0]) / np.linalg.norm(moved[1] - moved[0])
        return {
            "single_point_energy_hartree": -76.0,
            "gradient_hartree_per_bohr": np.full((3, 3), 0.001),
            "optimised_energy_hartree": -76.01,
            "optimised_coordinates_A": moved,
            "optimisation_metadata": {"OPTIMIZATION ITERATIONS": 4.0},
            "psi4_version": "test",
        }


class FailingRunner:
    def run(self, *args, **kwargs):
        raise RuntimeError("synthetic Psi4 failure")


class OptimisationFailingRunner:
    def run(self, *args, **kwargs):
        return {
            "single_point_energy_hartree": -76.0,
            "gradient_hartree_per_bohr": np.full((3, 3), 0.001),
            "optimisation_converged": False,
            "optimisation_error": "ConvergenceError: iteration limit",
            "psi4_version": "test",
        }


def test_geometry_input_preserves_coordinates_and_requires_state():
    coordinates = np.asarray([[0.123456789012, -1.0, 2.5]])
    text = qsv.build_psi4_geometry(["H"], coordinates, 0, 2)
    assert "0 2" in text
    assert "0.123456789012" in text
    assert "no_reorient" in text and "no_com" in text and "symmetry c1" in text
    np.testing.assert_array_equal(coordinates, [[0.123456789012, -1.0, 2.5]])
    try:
        qsv.build_psi4_geometry(["H"], coordinates, None, None)
    except ValueError as exc:
        assert "not inferred" in str(exc)
    else:
        raise AssertionError("missing electronic state was accepted")


def test_validation_persists_and_does_not_overwrite_library_geometry(tmp_path):
    molecule_root = tmp_path / "molecules"
    validation_root = tmp_path / "validations"
    original, _ = _stored_water(molecule_root)
    record = qsv.run_validation(
        "SP_000001", 0, 1, molecule_root=molecule_root,
        root=validation_root, runner=FakeRunner(),
    )
    assert record["status"] == "complete"
    assert record["single_point"]["max_force_eV_per_A"] > 0
    assert record["optimisation"]["relaxation_energy_eV"] < 0
    reloaded = qsv.load_validation("SP_000001", record["id"], validation_root)
    assert reloaded["id"] == record["id"]
    assert qsv.list_validations("SP_000001", validation_root)[0]["status"] == "complete"
    payload = qsv.load_geometries(reloaded, validation_root)
    np.testing.assert_allclose(payload["original_coordinates_A"], original)
    stored = molecule_library.load_molecule("SP_000001", root=molecule_root)
    np.testing.assert_allclose(stored["positions"], original)


def test_failed_job_is_persisted(tmp_path):
    molecule_root = tmp_path / "molecules"
    validation_root = tmp_path / "validations"
    _stored_water(molecule_root)
    record = qsv.run_validation(
        "SP_000001", 0, 1, molecule_root=molecule_root,
        root=validation_root, runner=FailingRunner(),
    )
    assert record["status"] == "failed"
    assert "synthetic Psi4 failure" in record["error"]
    persisted = qsv.load_validation("SP_000001", record["id"], validation_root)
    assert persisted["status"] == "failed"


def test_optimisation_failure_keeps_exact_geometry_result(tmp_path):
    molecule_root = tmp_path / "molecules"
    validation_root = tmp_path / "validations"
    _stored_water(molecule_root)
    record = qsv.run_validation(
        "SP_000001", 0, 1, molecule_root=molecule_root,
        root=validation_root, runner=OptimisationFailingRunner(),
    )
    assert record["status"] == "failed"
    assert record["single_point"]["energy_hartree"] == -76.0
    assert record["optimisation"]["converged"] is False
    payload = qsv.load_geometries(record, validation_root)
    assert "qm_forces_eV_per_A" in payload
    assert "optimised_coordinates_A" not in payload


def test_alignment_handles_rotation_translation_and_equivalent_hydrogens():
    half_angle = np.deg2rad(104.5 / 2.0)
    original = np.asarray([
        [0.0, 0.0, 0.0],
        [0.96 * np.sin(half_angle), 0.96 * np.cos(half_angle), 0.0],
        [-0.96 * np.sin(half_angle), 0.96 * np.cos(half_angle), 0.0],
    ])
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = original @ rotation + np.asarray([4.0, -3.0, 2.0])
    transformed[[1, 2]] = transformed[[2, 1]]
    result = qsv.compare_structures(
        ["O", "H", "H"], original, np.asarray([[0, 1], [0, 2]]), transformed
    )
    assert result["all_atom_rmsd_A"] < 1e-10
    assert result["connectivity_preserved"] is True
    assert result["fragmented"] is False


def test_connectivity_change_and_fragmentation_are_reported():
    original = np.asarray([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    separated = np.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    result = qsv.compare_structures(
        ["H", "H"], original, np.asarray([[0, 1]]), separated
    )
    assert result["connectivity_preserved"] is False
    assert result["connectivity_changed"] is True
    assert result["fragmented"] is True
