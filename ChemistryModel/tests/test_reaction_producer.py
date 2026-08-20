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
    families = {row["experiment_family"] for row in one}
    assert families <= {"pair", "microcell"}
    assert all(row["category"] == "atom_atom" for row in one if row["experiment_family"] == "pair")
    assert all(3 <= row["object_count"] <= 8 for row in one if row["experiment_family"] == "microcell")


def test_producer_cli_defaults_to_optimised_valence_and_allows_overrides():
    parser = build_parser()
    assert parser.parse_args(["produce_reactions"]).master_seed is None
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
        40, 55, molecule_root=molecules, qm_root=qm
    )
    pair_specs = [row for row in specs if row["experiment_family"] == "pair"]
    assert {row["category"] for row in pair_specs} == {
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


def test_novelty_weight_never_zero_and_penalises_repeated_region():
    candidate = {
        "reactant_a": "atom:H", "reactant_b": "SP_1",
        "impact_target": "oxygen", "target_atom_a": 0,
        "collision_class": "glancing", "speed_class": "moderate",
        "approach_factor": 1.5, "impact_fraction": .5,
        "temperature_K": 500.0,
    }
    unused = producer.novelty_weight(candidate, [])
    repeated = producer.novelty_weight(candidate, [dict(candidate) for _ in range(8)])
    assert unused > repeated > 0.0


def test_generation_updates_novelty_within_same_invocation(tmp_path, monkeypatch):
    molecules = tmp_path / "empty_molecules"

    # Force a tiny controlled candidate universe so the test checks that the
    # selected first spec becomes history for the second slot.
    original_weight = producer.novelty_weight
    seen_history_lengths = []

    def recording_weight(candidate, history):
        seen_history_lengths.append(len(history))
        return original_weight(candidate, history)

    monkeypatch.setattr(producer, "novelty_weight", recording_weight)
    monkeypatch.setattr(
        producer, "_choose_experiment_family", lambda generator: "pair"
    )
    specs, _ = producer.generate_experiment_specs(
        2, 123, molecule_root=molecules, qm_root=tmp_path / "qm"
    )
    assert len(specs) == 2
    assert min(seen_history_lengths) == 0
    assert max(seen_history_lengths) >= 1



def test_microcell_novelty_penalises_repeated_composition():
    candidate = {
        "experiment_family": "microcell",
        "reactants": ["atom:H", "atom:H", "atom:O"],
        "cluster_radius_A": 3.5, "minimum_gap_A": 2.0,
        "inward_factor": .8, "temperature_K": 500.0,
    }
    unused = producer.microcell_novelty_weight(candidate, [])
    repeated = producer.microcell_novelty_weight(candidate, [dict(candidate) for _ in range(8)])
    assert unused > repeated > 0.0


def test_prepare_microcell_box_is_safe(tmp_path):
    reactants = [
        producer._load_reactant("atom:H", tmp_path),
        producer._load_reactant("atom:O", tmp_path),
        producer._load_reactant("atom:H", tmp_path),
    ]
    symbols, positions, info = producer.characterisation.prepare_microcell_box(
        reactants, 14.0, 7, cluster_radius_A=3.5, minimum_gap_A=1.7
    )
    assert symbols == ["H", "O", "H"]
    assert positions.shape == (3, 3)
    assert info["experiment_family"] == "microcell"
    assert info["object_count"] == 3
    assert info["safe_initial_separation"] is True



def test_production_is_fresh_daily_attempt_and_failures_do_not_abort(
    tmp_path, monkeypatch
):
    molecules = tmp_path / "molecules"
    molecules.mkdir()
    output = tmp_path / "teacher"
    store = ManagerStore(tmp_path / "state.json")

    base_spec = {
        "number": 0, "category": "atom_atom",
        "reactant_a": "atom:H", "reactant_b": "atom:H",
        "simulation_seed": 7, "master_seed": 1, "profile": "balanced",
        "collision_class": "direct", "speed_class": "low",
        "approach_factor": .8, "impact_parameter_A": 0.0,
        "impact_scale_A": 1.0, "impact_fraction": 0.0,
        "start_gap_A": 2.5, "temperature_K": 250.0, "box_A": 12.0,
        "sampling_mode": "targeted_random", "impact_target": "hydrogen",
        "target_atom_a": 0,
    }

    generation_calls = []

    def fake_generate(count, master_seed, **kwargs):
        generation_calls.append(master_seed)
        invocation_id = kwargs.get("invocation_id")
        specs = []
        for number in range(count):
            spec = dict(base_spec)
            spec["number"] = number
            spec["id"] = f"EXP_{invocation_id}_{number}"
            spec["master_seed"] = master_seed
            spec["experiment_family"] = "pair"
            specs.append(spec)
        return specs, {
            "atoms": list(producer.ELEMENTAL_REACTANTS), "molecules": [],
            "trusted_molecule_records": [],
        }

    monkeypatch.setattr(producer, "generate_experiment_specs", fake_generate)
    monkeypatch.setattr(producer, "_local_day_string", lambda now=None: "2026-08-19")

    random_seeds = iter([101, 202])
    monkeypatch.setattr(producer, "_fresh_master_seed", lambda: next(random_seeds))
    nonces = iter(["aaaa", "bbbb"])
    monkeypatch.setattr(producer.secrets, "token_hex", lambda n: next(nonces))

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

    attempt = {"count": 0}

    def fake_run(first, second, seeds, options, frame_observer=None):
        attempt["count"] += 1
        if attempt["count"] == 2:
            raise RuntimeError("intentional test failure")
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
    monkeypatch.setattr(
        molecule_scanner, "scan_recordings",
        lambda runs_root, library_root: {"scanned": 1, "formation_events": 0},
    )
    monkeypatch.setattr(
        molecule_scanner,
        "record_controlled_final_event",
        lambda *args, **kwargs: {
            "status": "no reaction",
            "event": None,
            "created_species": [],
        },
    )
    monkeypatch.setattr(producer, "read_events", lambda path: ([], []))

    first = producer.run_production(
        store, count=3, duration_ps=.001,
        output_root=output, molecule_root=molecules, device="cpu",
    )
    second = producer.run_production(
        store, count=1, duration_ps=.001,
        output_root=output, molecule_root=molecules, device="cpu",
    )

    assert generation_calls == [101, 202]
    assert first["requested"] == 3
    assert first["completed_now"] == 2
    assert first["failed_now"] == 1
    assert second["completed_now"] == 1
    assert second["invocation_id"] != first["invocation_id"]
    assert Path(first["root"]) == output / "2026-08-19"
    assert Path(second["root"]) == output / "2026-08-19"

    summary = producer.production_summary(output, "2026-08-19")
    assert summary["invocations"] == 2
    assert summary["attempted"] == 4
    assert summary["completed"] == 3
    assert summary["failed"] == 1


def test_explicit_seed_is_preserved_but_invocations_are_still_independent(
    tmp_path, monkeypatch
):
    molecules = tmp_path / "molecules"
    molecules.mkdir()
    output = tmp_path / "teacher"
    store = ManagerStore(tmp_path / "state.json")

    monkeypatch.setattr(producer, "_local_day_string", lambda now=None: "2026-08-19")
    nonces = iter(["cccc", "dddd"])
    monkeypatch.setattr(producer.secrets, "token_hex", lambda n: next(nonces))

    generated = []
    def fake_generate(count, master_seed, **kwargs):
        generated.append((master_seed, kwargs["invocation_id"]))
        return [], {
            "atoms": list(producer.ELEMENTAL_REACTANTS), "molecules": [],
            "trusted_molecule_records": [],
        }
    monkeypatch.setattr(producer, "generate_experiment_specs", fake_generate)
    monkeypatch.setattr(
        molecule_scanner, "scan_recordings",
        lambda runs_root, library_root: {"scanned": 0, "formation_events": 0},
    )
    monkeypatch.setattr(producer, "read_events", lambda path: ([], []))

    # count must remain >= 1; fake generation intentionally returns no specs.
    first = producer.run_production(
        store, count=1, master_seed=77, duration_ps=.001,
        output_root=output, molecule_root=molecules, device="cpu",
    )
    second = producer.run_production(
        store, count=1, master_seed=77, duration_ps=.001,
        output_root=output, molecule_root=molecules, device="cpu",
    )

    assert [row[0] for row in generated] == [77, 77]
    assert generated[0][1] != generated[1][1]
    assert first["master_seed_source"] == "explicit"
    assert second["master_seed_source"] == "explicit"



def test_events_for_experiment_only_returns_matching_recording():
    events = [
        {"event_id": "a", "recording": "x/EXP_one.npz"},
        {"event_id": "b", "recording": "x/EXP_two.npz"},
    ]
    found = producer._events_for_experiment(events, "EXP_two")
    assert [row["event_id"] for row in found] == ["b"]


def test_full_cm_router_goes_straight_to_qm(tmp_path, monkeypatch):
    from chemistry_manager.discovery import route_full_cm_events_to_qm

    molecules = tmp_path / "molecules"
    molecules.mkdir()
    store = ManagerStore(tmp_path / "state.json")
    reaction = {
        "event_id": "full_cm_event",
        "recording": "teacher_data/2026-08-19/recordings/EXP_one.npz",
        "products": [{"id": "SP_new", "formula": "CH"}],
        "reactants": [{"formula": "C"}, {"formula": "H"}],
        "formed_bonds": [],
        "broken_bonds": [],
    }
    monkeypatch.setattr(
        "chemistry_manager.trust.trusted_molecules",
        lambda *args, **kwargs: [],
    )

    result = route_full_cm_events_to_qm(
        store, [reaction], molecules,
        production_id="INV_test",
        teacher_layout="live_production",
    )

    assert result["queued"] == 1
    waiting = store.candidates(CandidateState.WAITING_QM)
    assert len(waiting) == 1
    assert len(store.candidates()) == 1
    assert waiting[0]["state"] == CandidateState.WAITING_QM.value


def test_controlled_final_event_routing_bypasses_characterisation_queue(
    tmp_path, monkeypatch
):
    from chemistry_manager.discovery import route_full_cm_events_to_qm

    molecules = tmp_path / "molecules"
    molecules.mkdir()
    store = ManagerStore(tmp_path / "state.json")
    reaction = {
        "event_id": "controlled_final",
        "event_kind": "controlled_final_state",
        "recording": "teacher_data/2026-08-20/recordings/EXP_one.npz",
        "products": [{"id": "SP_new", "formula": "H2"}],
        "reactants": [{"formula": "H", "count": 2}],
        "formed_bonds": [{"atom_ids": [0, 1], "symbols": ["H", "H"]}],
        "broken_bonds": [],
    }
    monkeypatch.setattr(
        "chemistry_manager.trust.trusted_molecules",
        lambda *args, **kwargs: [],
    )

    result = route_full_cm_events_to_qm(
        store,
        [reaction],
        molecules,
        production_id="INV_test",
        teacher_layout="live_production",
    )

    assert result["queued"] == 1
    waiting = store.candidates(CandidateState.WAITING_QM)
    assert len(waiting) == 1
    assert len(store.candidates()) == 1
    assert waiting[0]["state"] == CandidateState.WAITING_QM.value


def test_producer_source_isolates_postprocessing_from_md_failure():
    source = Path(producer.__file__).read_text(encoding="utf-8")
    assert "record_controlled_final_event" in source
    assert "postprocess_warning" in source
    assert "cannot retroactively fail MD" in source
