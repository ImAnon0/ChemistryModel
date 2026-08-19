import json
from pathlib import Path

import numpy as np

import molecule_scanner
from chemistry_manager import CandidateState, ManagerStore
from chemistry_manager.cli import build_parser
import chemistry_manager.reaction_producer as producer
from chemistry_manager.trust import MoleculeTrust, trust_level, trusted_molecules


def write_molecule(root, molecule_id, *, trust_status=None):
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / f"{molecule_id}.npz",
        symbols=np.asarray(["H", "H"], dtype="U2"),
        positions=np.asarray([[0, 0, 0], [.74, 0, 0]], dtype=np.float32),
        bonds=np.asarray([[0, 1]], dtype=np.int32),
        source_atom_ids=np.asarray([0, 1], dtype=np.uint32),
        source_slots=np.asarray([0, 1], dtype=np.int32),
    )
    metadata = {
        "id": molecule_id,
        "formula": "H2",
        "atoms": 2,
        "heavy_atoms": 0,
        "graph_fingerprint": molecule_id + "_fingerprint",
        "payload": f"{molecule_id}.npz",
    }
    if trust_status is not None:
        metadata["trust_status"] = trust_status
    (root / f"{molecule_id}.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return metadata


def write_qm(qm_root, molecule_id, status="complete", preserved=True):
    directory = qm_root / molecule_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "QV_test.json").write_text(json.dumps({
        "id": "QV_test",
        "molecule_id": molecule_id,
        "status": status,
        "comparison": {
            "connectivity_preserved": preserved,
            "fragmented": False,
            "rearranged": not preserved,
        },
    }), encoding="utf-8")


def test_trusted_pool_requires_explicit_or_successful_qm_validation(tmp_path):
    molecules = tmp_path / "molecules"
    qm = tmp_path / "qm"
    write_molecule(molecules, "SP_000001")
    write_molecule(molecules, "SP_000002")
    write_molecule(molecules, "SP_000003", trust_status="CM_VALIDATED")
    write_qm(qm, "SP_000001")
    write_qm(qm, "SP_000002", status="failed")

    found = trusted_molecules(molecules, qm_root=qm)
    assert [row["id"] for row in found] == ["SP_000001", "SP_000003"]
    assert trust_level(found[0], qm_root=qm) == MoleculeTrust.QM_VALIDATED


def test_generation_is_reproducible_and_uses_atom_fallback(tmp_path):
    molecules = tmp_path / "empty_molecules"
    one, pool = producer.generate_experiment_specs(
        9, 1234, molecule_root=molecules, qm_root=tmp_path / "qm"
    )
    two, _ = producer.generate_experiment_specs(
        9, 1234, molecule_root=molecules, qm_root=tmp_path / "qm"
    )

    assert one == two
    assert pool["molecules"] == []
    assert {row["category"] for row in one} == {"atom_atom"}
    assert {row["collision_class"] for row in one} == {
        "direct", "glancing", "near_miss"
    }
    assert {row["speed_class"] for row in one} == {
        "low", "moderate", "high"
    }
    assert any(row["impact_parameter_A"] == 0 for row in one)
    assert any(row["impact_fraction"] > 1 for row in one)


def test_producer_cli_defaults_to_optimised_valence_and_allows_overrides():
    parser = build_parser()
    assert parser.parse_args(["produce_reactions"]).physics == "optimised-valence"
    assert parser.parse_args([
        "produce_reactions", "--physics", "standard",
    ]).physics == "standard"
    assert parser.parse_args([
        "produce_reactions", "--physics", "high_fidelity",
    ]).physics == "high_fidelity"


def test_generation_covers_all_reactant_orderings_with_trusted_molecule(tmp_path):
    molecules = tmp_path / "molecules"
    qm = tmp_path / "qm"
    write_molecule(molecules, "SP_000001")
    write_qm(qm, "SP_000001")

    specs, _ = producer.generate_experiment_specs(
        8, 55, molecule_root=molecules, qm_root=qm
    )
    assert {row["category"] for row in specs} == {
        "atom_atom", "atom_molecule", "molecule_atom", "molecule_molecule"
    }


def test_teacher_frame_selection_keeps_sparse_close_and_event_frames():
    collector = producer.TeacherFrameCollector(
        ordinary_interval_fs=10.0, event_window_fs=1.0
    )
    info = {"first_count": 1}
    for time_fs, distance in [(0, 4.0), (1, 3.0), (2, .8), (3, .7), (20, 4.0)]:
        collector({
            "seed": 1, "symbols": ["H", "H"],
            "positions_A": np.asarray([[0, 0, 0], [distance, 0, 0]], dtype=float),
            "forces_eV_per_A": np.zeros((2, 3)),
            "potential_energy_eV": -1.0,
            "kinetic_energy_eV": .1,
            "temperature_K": 250.0,
            "time_fs": float(time_fs), "box_A": 12.0,
            "collision_info": info,
        })
    selected = collector.selected()
    reasons = {reason for row in selected for reason in row["selection_reasons"]}
    assert "pre_contact" in reasons
    assert "close_contact" in reasons
    assert "bond_change_window" in reasons
    assert "final" in reasons


def test_production_persists_shard_resumes_and_queues_new_event(
    tmp_path, monkeypatch
):
    molecules = tmp_path / "molecules"
    molecules.mkdir()
    output = tmp_path / "teacher"
    store = ManagerStore(tmp_path / "state.json")
    spec = {
        "id": "EXP_fixed", "number": 0, "category": "atom_atom",
        "reactant_a": "atom:H", "reactant_b": "atom:H",
        "simulation_seed": 7, "master_seed": 1, "profile": "balanced",
        "collision_class": "direct", "speed_class": "low",
        "approach_factor": .8, "impact_parameter_A": 0.0,
        "impact_scale_A": 1.0, "impact_fraction": 0.0,
        "start_gap_A": 2.5, "temperature_K": 250.0, "box_A": 12.0,
        "sampling_mode": "targeted_random", "impact_target": "hydrogen",
        "target_atom_a": 0,
    }
    monkeypatch.setattr(
        producer, "generate_experiment_specs",
        lambda *args, **kwargs: ([spec], {
            "atoms": list(producer.ELEMENTAL_REACTANTS), "molecules": [],
            "trusted_molecule_records": [],
        }),
    )

    calls = []

    class FakeRecorder:
        def save(self, path):
            Path(path).write_bytes(b"recording")

    class FakeSimulation:
        physics_model_name = "fake_full_cm"
        physics_model_revision = 9

    info = {
        "first_count": 1, "second_count": 1,
        "axis": np.asarray([1.0, 0.0, 0.0]),
        "impact_parameter_A": 0.0,
    }

    def fake_run(first, second, seeds, options, frame_observer=None):
        calls.append((spec["id"], options.physics))
        for time_fs, distance in [(0.0, 3.0), (1.0, .7)]:
            frame_observer({
                "seed": seeds[0], "symbols": ["H", "H"],
                "positions_A": np.asarray([[0, 0, 0], [distance, 0, 0]]),
                "forces_eV_per_A": np.ones((2, 3)),
                "potential_energy_eV": -2.0, "kinetic_energy_eV": .2,
                "temperature_K": 250.0, "time_fs": time_fs,
                "box_A": 12.0, "collision_info": info,
            })
        return [FakeRecorder()], FakeSimulation(), .1, False, [
            {"thermal_rms_speed": .01, "relative_speed": .008}
        ], [{}], [info]

    monkeypatch.setattr(producer.characterisation, "run_group", fake_run)
    monkeypatch.setattr(producer.characterisation, "summarise", lambda *a, **k: {
        "characterisation_outcome": "joined", "physics_model": "fake_full_cm",
        "seed": 7, "finished": True,
    })

    event = {
        "event_id": "teacher_event", "recording": None,
        "seed": 7, "reactants": [{"formula": "H", "count": 2}],
        "products": [{"formula": "H2", "count": 1}],
        "formed_bonds": [{"atom_ids": [0, 1]}], "broken_bonds": [],
    }

    def fake_scan(runs_root, library_root):
        log = Path(library_root) / "formation_events.jsonl"
        if not log.exists():
            payload = dict(event)
            payload["recording"] = str(
                Path(runs_root) / "recordings" / "EXP_fixed.npz"
            )
            log.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return {"scanned": 1, "formation_events": 1}

    monkeypatch.setattr(molecule_scanner, "scan_recordings", fake_scan)

    first = producer.run_production(
        store, count=1, duration_ps=.001, master_seed=1,
        output_root=output, molecule_root=molecules, device="cpu",
    )
    second = producer.run_production(
        store, count=1, duration_ps=.001, master_seed=1,
        output_root=output, molecule_root=molecules, device="cpu",
    )
    standard = producer.run_production(
        store, count=1, duration_ps=.001, master_seed=1,
        output_root=output, molecule_root=molecules, device="cpu",
        physics="standard",
    )

    assert calls == [
        ("EXP_fixed", "optimised-valence"),
        ("EXP_fixed", "standard"),
    ]
    assert first["completed_now"] == 1
    assert second["completed_now"] == 0
    assert standard["production_id"] != first["production_id"]
    assert first["new_candidates_queued"] == 1
    assert len(store.candidates(CandidateState.WAITING_CHARACTERISATION)) == 1
    shard = Path(first["root"]) / "shards" / "EXP_fixed.npz"
    with np.load(shard, allow_pickle=False) as data:
        assert data["positions_A"].shape == (2, 2, 3)
        assert data["forces_eV_per_A"].shape == (2, 2, 3)
    manifest = json.loads(
        (Path(first["root"]) / "production.json").read_text(encoding="utf-8")
    )
    assert manifest["physics"] == "optimised-valence"
    assert manifest["physics_model"] == (
        "reactive_v7_factorisable_valence_optimised_experimental"
    )
    assert manifest["physics_model_revision"] == 1
    experiment = json.loads(
        (Path(first["root"]) / "experiments" / "EXP_fixed.json").read_text(
            encoding="utf-8"
        )
    )
    assert experiment["physics_model"] == "fake_full_cm"
    assert experiment["physics_model_revision"] == 9
