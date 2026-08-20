import json

import molecule_scanner

from chemistry_manager import CandidateState, ManagerStore
from chemistry_manager.cli import main
from chemistry_manager.discovery import discover, ingest_teacher_data
from chemistry_manager.qm import process_qm_queue


def event(event_id="abc123", seed=7):
    return {
        "event_id": event_id,
        "recording": "runs/example/run_s0007.npz",
        "batch": "example",
        "seed": seed,
        "mixture": "test mixture",
        "time_fs": 125.0,
        "temperature_K": 300.0,
        "box_A": 20.0,
        "reactants": [{"formula": "H", "count": 2}],
        "products": [{"formula": "H2", "count": 1, "id": "SP_000001"}],
        "formed_bonds": [{"atom_ids": [1, 2], "symbols": ["H", "H"]}],
        "broken_bonds": [],
    }


def test_store_persists_candidates_and_deduplicates_event_ids(tmp_path):
    path = tmp_path / "manager" / "state.json"
    first = ManagerStore(path)

    assert first.add_discovery_event(event(), tmp_path / "events.jsonl")
    assert not first.add_discovery_event(event(), tmp_path / "events.jsonl")

    second = ManagerStore(path)
    waiting = second.candidates(CandidateState.WAITING_QM)
    assert len(waiting) == 1
    assert waiting[0]["id"] == "EVENT_abc123"
    assert waiting[0]["provenance"]["seed"] == 7
    assert waiting[0]["products"][0]["id"] == "SP_000001"


def test_store_transitions_follow_the_v1_state_machine(tmp_path):
    store = ManagerStore(tmp_path / "state.json")
    store.add_discovery_event(event(), tmp_path / "events.jsonl")

    store.transition("EVENT_abc123", CandidateState.QM_VALIDATED)

    counts = store.counts()
    assert counts[CandidateState.QM_VALIDATED] == 1

    try:
        store.transition("EVENT_abc123", CandidateState.WAITING_QM)
    except ValueError as problem:
        assert "invalid candidate transition" in str(problem)
    else:
        raise AssertionError("terminal candidates must not move backwards")


def test_legacy_wait_states_are_persistently_migrated_to_qm(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "format_version": 1,
        "candidates": {
            "EVENT_old_full": {
                "id": "EVENT_old_full",
                "state": "WAITING_FULL_CM",
                "products": [{"id": "SP_old_full"}],
            },
            "EVENT_old_char": {
                "id": "EVENT_old_char",
                "state": "WAITING_CHARACTERISATION",
                "products": [{"id": "SP_old_char"}],
            },
        },
    }), encoding="utf-8")

    store = ManagerStore(path)
    result = store.migrate_legacy_wait_states()

    assert result["migrated"] == 2
    waiting = store.candidates(CandidateState.WAITING_QM)
    assert len(waiting) == 2

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert {
        row["state"] for row in persisted["candidates"].values()
    } == {"WAITING_QM"}
    assert persisted["candidates"]["EVENT_old_char"]["products"][0]["id"] == "SP_old_char"


def test_corrupt_store_is_reported_without_being_overwritten(tmp_path):
    path = tmp_path / "state.json"
    original = "{not valid json\n"
    path.write_text(original, encoding="utf-8")

    try:
        ManagerStore(path).counts()
    except ValueError as problem:
        assert "not valid JSON" in str(problem)
    else:
        raise AssertionError("a corrupt manager store must not be accepted")

    assert path.read_text(encoding="utf-8") == original


def test_status_is_clean_for_a_new_store(tmp_path, capsys):
    code = main([
        "--state-file", str(tmp_path / "state.json"),
        "status", "--molecule-root", str(tmp_path / "molecules"),
    ])
    output = capsys.readouterr().out

    assert code == 0
    assert "Chemistry Manager" in output
    assert "Waiting for QM validation:" in output
    assert "QM validated:" in output
    assert output.count("0") >= 4
    assert not (tmp_path / "state.json").exists()


def test_discover_reuses_scanner_event_log_and_is_repeatable(
    tmp_path, monkeypatch
):
    molecule_root = tmp_path / "molecules"
    molecule_root.mkdir()
    event_log = molecule_root / "formation_events.jsonl"
    event_log.write_text(json.dumps(event()) + "\n", encoding="utf-8")
    calls = []

    def fake_scan_recordings(runs_root, library_root, progress=None):
        calls.append((runs_root, library_root, progress))
        return {
            "scanned": 0,
            "unchanged": 1,
            "formation_events": 0,
        }

    monkeypatch.setattr(molecule_scanner, "scan_recordings", fake_scan_recordings)
    store = ManagerStore(tmp_path / "state.json")

    first = discover(store, tmp_path / "runs", molecule_root)
    second = discover(store, tmp_path / "runs", molecule_root)

    assert len(calls) == 2
    assert first["queued"] == 1
    assert second["queued"] == 0
    assert second["already_known"] == 1


def test_empty_qm_command_returns_cleanly(tmp_path, capsys):
    state = str(tmp_path / "state.json")

    assert main(["--state-file", state, "qm"]) == 0
    assert capsys.readouterr().out.strip() == "Nothing waiting for QM validation."


def test_discover_name_is_reserved_for_future_surrogate(tmp_path, capsys):
    state = str(tmp_path / "state.json")
    assert main(["--state-file", state, "discover"]) == 0
    assert "reserved" in capsys.readouterr().out


def test_store_can_route_full_cm_teacher_event_directly_to_qm(tmp_path):
    store = ManagerStore(tmp_path / "state.json")
    store.add_discovery_events(
        [event("teacher_qm")],
        tmp_path / "events.jsonl",
        initial_state=CandidateState.WAITING_QM,
        source_kind="full_cm_teacher_event",
        source_extra={"production_id": "PROD_test"},
    )
    waiting = store.candidates(CandidateState.WAITING_QM)
    assert len(waiting) == 1
    assert waiting[0]["source"]["kind"] == "full_cm_teacher_event"
    assert waiting[0]["source"]["production_id"] == "PROD_test"


def test_ingest_teacher_data_registers_shard_and_routes_untrusted_product_to_qm(
    tmp_path, monkeypatch
):
    teacher_root = tmp_path / "teacher_data"
    production_root = teacher_root / "PROD_test"
    experiments = production_root / "experiments"
    recordings = production_root / "recordings"
    shards = production_root / "shards"
    molecules = tmp_path / "molecules"
    for path in (experiments, recordings, shards, molecules):
        path.mkdir(parents=True, exist_ok=True)

    (production_root / "production.json").write_text(json.dumps({
        "id": "PROD_test",
        "physics": "optimised-valence",
        "physics_model": "teacher",
        "physics_model_revision": 1,
        "physics_source_sha256": "abc",
        "chemistrymodel_git_revision": "git",
    }), encoding="utf-8")
    (shards / "EXP_test.npz").write_bytes(b"shard")
    (experiments / "EXP_test.json").write_text(json.dumps({
        "status": "complete",
        "experiment_id": "EXP_test",
        "shard": "shards/EXP_test.npz",
        "teacher_frames": 7,
        "physics_model": "teacher",
        "physics_model_revision": 1,
        "physics_source_sha256": "abc",
        "chemistrymodel_git_revision": "git",
    }), encoding="utf-8")

    event_log = molecules / "formation_events.jsonl"
    event_log.write_text(json.dumps({
        **event("teacher_event"),
        "recording": str(recordings / "EXP_test.npz"),
        "products": [{"formula": "CH", "count": 1, "id": "SP_new"}],
    }) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        molecule_scanner, "scan_recordings",
        lambda runs_root, library_root: {
            "scanned": 0, "unchanged": 1, "formation_events": 0
        },
    )
    monkeypatch.setattr(
        "chemistry_manager.trust.trusted_molecules",
        lambda *args, **kwargs: [],
    )

    store = ManagerStore(tmp_path / "manager" / "state.json")
    result = ingest_teacher_data(
        store, teacher_root, molecules, state_file=store.path
    )

    assert result["productions_added"] == 1
    assert result["experiments_added"] == 1
    assert result["teacher_frames_added"] == 7
    assert result["queued_for_qm"] == 1
    assert len(store.candidates(CandidateState.WAITING_QM)) == 1


def test_ingest_teacher_data_does_not_queue_already_trusted_product(
    tmp_path, monkeypatch
):
    teacher_root = tmp_path / "teacher_data"
    production_root = teacher_root / "PROD_test"
    experiments = production_root / "experiments"
    recordings = production_root / "recordings"
    shards = production_root / "shards"
    molecules = tmp_path / "molecules"
    for path in (experiments, recordings, shards, molecules):
        path.mkdir(parents=True, exist_ok=True)

    (production_root / "production.json").write_text(json.dumps({
        "id": "PROD_test",
        "physics": "optimised-valence",
    }), encoding="utf-8")
    (shards / "EXP_test.npz").write_bytes(b"shard")
    (experiments / "EXP_test.json").write_text(json.dumps({
        "status": "complete",
        "experiment_id": "EXP_test",
        "shard": "shards/EXP_test.npz",
        "teacher_frames": 3,
    }), encoding="utf-8")

    (molecules / "formation_events.jsonl").write_text(json.dumps({
        **event("teacher_known"),
        "recording": str(recordings / "EXP_test.npz"),
        "products": [{"formula": "H2", "count": 1, "id": "SP_known"}],
    }) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        molecule_scanner, "scan_recordings",
        lambda runs_root, library_root: {
            "scanned": 0, "unchanged": 1, "formation_events": 0
        },
    )
    monkeypatch.setattr(
        "chemistry_manager.trust.trusted_molecules",
        lambda *args, **kwargs: [{"id": "SP_known"}],
    )

    store = ManagerStore(tmp_path / "manager" / "state.json")
    result = ingest_teacher_data(
        store, teacher_root, molecules, state_file=store.path
    )

    assert result["queued_for_qm"] == 0
    assert result["already_trusted_events"] == 1
    assert store.counts()[CandidateState.WAITING_QM] == 0


def test_generic_discovery_queues_directly_to_qm(tmp_path, monkeypatch):
    molecule_root = tmp_path / "molecules"
    molecule_root.mkdir()
    event_log = molecule_root / "formation_events.jsonl"
    event_log.write_text(json.dumps(event("generic_qm")) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        molecule_scanner,
        "scan_recordings",
        lambda *args, **kwargs: {
            "scanned": 0,
            "unchanged": 1,
            "formation_events": 0,
        },
    )

    store = ManagerStore(tmp_path / "state.json")
    result = discover(store, tmp_path / "runs", molecule_root)

    assert result["queued"] == 1
    waiting = store.candidates(CandidateState.WAITING_QM)
    assert len(waiting) == 1
    assert waiting[0]["source"]["kind"] == "full_cm_formation_event"


def _write_manager_molecule(root, molecule_id, symbols=("H", "H")):
    import numpy as np
    root.mkdir(parents=True, exist_ok=True)
    count = len(symbols)
    positions = np.zeros((count, 3), dtype=np.float32)
    if count > 1:
        positions[:, 0] = np.arange(count, dtype=np.float32) * 0.74
    bonds = (
        np.asarray([[i, i + 1] for i in range(count - 1)], dtype=np.int32)
        if count > 1 else np.empty((0, 2), dtype=np.int32)
    )
    np.savez_compressed(
        root / f"{molecule_id}.npz",
        symbols=np.asarray(symbols, dtype="U2"),
        positions=positions,
        bonds=bonds,
        source_atom_ids=np.arange(count, dtype=np.uint32),
        source_slots=np.arange(count, dtype=np.int32),
    )
    (root / f"{molecule_id}.json").write_text(json.dumps({
        "id": molecule_id,
        "formula": "test",
        "atoms": count,
        "heavy_atoms": sum(symbol != "H" for symbol in symbols),
        "graph_fingerprint": molecule_id + "_fp",
        "payload": f"{molecule_id}.npz",
    }), encoding="utf-8")


def test_qm_queue_validates_full_cm_product_and_persists_electronic_state(
    tmp_path
):
    molecules = tmp_path / "molecules"
    qm_root = tmp_path / "qm"
    _write_manager_molecule(molecules, "SP_000001")

    store = ManagerStore(tmp_path / "state.json")
    store.add_discovery_events(
        [event("qm_run")],
        tmp_path / "events.jsonl",
        initial_state=CandidateState.WAITING_QM,
        source_kind="full_cm_teacher_event",
    )

    def fake_worker(molecule_id, **kwargs):
        assert molecule_id == "SP_000001"
        return {
            "id": "QV_fake",
            "molecule_id": molecule_id,
            "status": "complete",
            "charge": 0,
            "multiplicity": 1,
            "comparison": {
                "connectivity_preserved": True,
                "connectivity_changed": False,
                "fragmented": False,
                "rearranged": False,
            },
            "electronic_state_selection": {
                "charge": 0,
                "source": "neutral_chno_electron_parity_qm_screen",
                "selected_multiplicity": 1,
            },
        }

    result = process_qm_queue(
        store,
        molecule_root=molecules,
        qm_root=qm_root,
        worker=fake_worker,
    )

    assert result["validated"] == 1
    assert len(store.candidates(CandidateState.QM_VALIDATED)) == 1
    metadata = json.loads(
        (molecules / "SP_000001.json").read_text(encoding="utf-8")
    )
    assert metadata["trust_status"] == "QM_VALIDATED"
    assert metadata["electronic_state"]["multiplicity"] == 1


def test_qm_queue_rejects_connectivity_changed_product(tmp_path):
    molecules = tmp_path / "molecules"
    qm_root = tmp_path / "qm"
    _write_manager_molecule(molecules, "SP_000001")

    store = ManagerStore(tmp_path / "state.json")
    store.add_discovery_events(
        [event("qm_reject")],
        tmp_path / "events.jsonl",
        initial_state=CandidateState.WAITING_QM,
        source_kind="full_cm_teacher_event",
    )

    def fake_worker(molecule_id, **kwargs):
        return {
            "id": "QV_reject",
            "molecule_id": molecule_id,
            "status": "complete",
            "charge": 0,
            "multiplicity": 1,
            "comparison": {
                "connectivity_preserved": False,
                "connectivity_changed": True,
                "fragmented": False,
                "rearranged": True,
            },
            "electronic_state_selection": {
                "charge": 0,
                "source": "neutral_chno_electron_parity_qm_screen",
                "selected_multiplicity": 1,
            },
        }

    result = process_qm_queue(
        store,
        molecule_root=molecules,
        qm_root=qm_root,
        worker=fake_worker,
    )

    assert result["rejected"] == 1
    assert len(store.candidates(CandidateState.QM_REJECTED)) == 1
    metadata = json.loads(
        (molecules / "SP_000001.json").read_text(encoding="utf-8")
    )
    assert metadata["trust_status"] == "REJECTED"


def test_qm_queue_worker_failure_leaves_candidate_waiting(tmp_path):
    molecules = tmp_path / "molecules"
    qm_root = tmp_path / "qm"
    _write_manager_molecule(molecules, "SP_000001")

    store = ManagerStore(tmp_path / "state.json")
    store.add_discovery_events(
        [event("qm_fail")],
        tmp_path / "events.jsonl",
        initial_state=CandidateState.WAITING_QM,
        source_kind="full_cm_teacher_event",
    )

    def fake_worker(*args, **kwargs):
        raise RuntimeError("synthetic worker failure")

    result = process_qm_queue(
        store,
        molecule_root=molecules,
        qm_root=qm_root,
        worker=fake_worker,
    )

    assert result["still_waiting"] == 1
    assert len(store.candidates(CandidateState.WAITING_QM)) == 1


def test_daily_teacher_ingest_routes_untrusted_product_directly_to_qm(
    tmp_path, monkeypatch
):
    teacher_root = tmp_path / "teacher_data"
    day = teacher_root / "2026-08-19"
    invocations = day / "invocations"
    experiments = day / "experiments"
    recordings = day / "recordings"
    shards = day / "shards"
    molecules = tmp_path / "molecules"
    for path in (invocations, experiments, recordings, shards, molecules):
        path.mkdir(parents=True, exist_ok=True)

    invocation_id = "INV_20260819_test"
    (invocations / f"{invocation_id}.json").write_text(json.dumps({
        "id": invocation_id,
        "physics": "optimised-valence",
        "physics_model": "teacher",
        "physics_model_revision": 7,
        "physics_source_sha256": "source",
        "chemistrymodel_git_revision": "git",
    }), encoding="utf-8")

    (shards / "EXP_daily.npz").write_bytes(b"shard")
    (experiments / "EXP_daily.json").write_text(json.dumps({
        "status": "complete",
        "experiment_id": "EXP_daily",
        "invocation_id": invocation_id,
        "production_id": invocation_id,
        "shard": "shards/EXP_daily.npz",
        "teacher_frames": 11,
        "physics_model": "teacher",
        "physics_model_revision": 7,
    }), encoding="utf-8")

    (molecules / "formation_events.jsonl").write_text(json.dumps({
        **event("daily_new"),
        "recording": str(recordings / "EXP_daily.npz"),
        "products": [{"formula": "CH", "count": 1, "id": "SP_daily_new"}],
    }) + "\n", encoding="utf-8")

    scan_calls = []
    monkeypatch.setattr(
        molecule_scanner, "scan_recordings",
        lambda runs_root, library_root: (
            scan_calls.append((runs_root, library_root))
            or {"scanned": 0, "unchanged": 1, "formation_events": 0}
        ),
    )
    monkeypatch.setattr(
        "chemistry_manager.trust.trusted_molecules",
        lambda *args, **kwargs: [],
    )

    store = ManagerStore(tmp_path / "manager" / "state.json")
    result = ingest_teacher_data(
        store, teacher_root, molecules, state_file=store.path
    )

    assert result["productions_found"] == 1
    assert result["productions_added"] == 1
    assert result["experiments_added"] == 1
    assert result["teacher_frames_added"] == 11
    assert result["events_seen"] == 1
    assert result["queued_for_qm"] == 1
    assert result["already_trusted_events"] == 0
    assert len(scan_calls) == 1

    waiting = store.candidates(CandidateState.WAITING_QM)
    assert len(waiting) == 1
    assert waiting[0]["products"][0]["id"] == "SP_daily_new"
    assert waiting[0]["source"]["production_id"] == invocation_id
    assert waiting[0]["source"]["teacher_layout"] == "daily"


def test_daily_teacher_ingest_is_repeatable_and_does_not_double_register(
    tmp_path, monkeypatch
):
    teacher_root = tmp_path / "teacher_data"
    day = teacher_root / "2026-08-19"
    invocations = day / "invocations"
    experiments = day / "experiments"
    recordings = day / "recordings"
    shards = day / "shards"
    molecules = tmp_path / "molecules"
    for path in (invocations, experiments, recordings, shards, molecules):
        path.mkdir(parents=True, exist_ok=True)

    invocation_id = "INV_repeat"
    (invocations / f"{invocation_id}.json").write_text(
        json.dumps({"id": invocation_id, "physics": "optimised-valence"}),
        encoding="utf-8",
    )
    (shards / "EXP_repeat.npz").write_bytes(b"shard")
    (experiments / "EXP_repeat.json").write_text(json.dumps({
        "status": "complete",
        "experiment_id": "EXP_repeat",
        "invocation_id": invocation_id,
        "shard": "shards/EXP_repeat.npz",
        "teacher_frames": 4,
    }), encoding="utf-8")
    (molecules / "formation_events.jsonl").write_text(json.dumps({
        **event("daily_repeat"),
        "recording": str(recordings / "EXP_repeat.npz"),
        "products": [{"formula": "CH", "count": 1, "id": "SP_repeat"}],
    }) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        molecule_scanner, "scan_recordings",
        lambda runs_root, library_root: {
            "scanned": 0, "unchanged": 1, "formation_events": 0
        },
    )
    monkeypatch.setattr(
        "chemistry_manager.trust.trusted_molecules",
        lambda *args, **kwargs: [],
    )

    store = ManagerStore(tmp_path / "manager" / "state.json")
    first = ingest_teacher_data(
        store, teacher_root, molecules, state_file=store.path
    )
    second = ingest_teacher_data(
        store, teacher_root, molecules, state_file=store.path
    )

    assert first["productions_added"] == 1
    assert first["experiments_added"] == 1
    assert first["queued_for_qm"] == 1

    assert second["productions_added"] == 0
    assert second["experiments_added"] == 0
    assert second["teacher_frames_added"] == 0
    assert second["queued_for_qm"] == 0
    assert second["duplicate_candidates"] == 1
    assert len(store.candidates(CandidateState.WAITING_QM)) == 1


def test_daily_teacher_ingest_scans_shared_day_folder_only_once(
    tmp_path, monkeypatch
):
    teacher_root = tmp_path / "teacher_data"
    day = teacher_root / "2026-08-19"
    invocations = day / "invocations"
    experiments = day / "experiments"
    recordings = day / "recordings"
    shards = day / "shards"
    molecules = tmp_path / "molecules"
    for path in (invocations, experiments, recordings, shards, molecules):
        path.mkdir(parents=True, exist_ok=True)

    for number in (1, 2):
        invocation_id = f"INV_{number}"
        experiment_id = f"EXP_{number}"
        (invocations / f"{invocation_id}.json").write_text(
            json.dumps({"id": invocation_id, "physics": "optimised-valence"}),
            encoding="utf-8",
        )
        (shards / f"{experiment_id}.npz").write_bytes(b"shard")
        (experiments / f"{experiment_id}.json").write_text(json.dumps({
            "status": "complete",
            "experiment_id": experiment_id,
            "invocation_id": invocation_id,
            "shard": f"shards/{experiment_id}.npz",
            "teacher_frames": number,
        }), encoding="utf-8")

    (molecules / "formation_events.jsonl").write_text("", encoding="utf-8")

    scan_calls = []
    monkeypatch.setattr(
        molecule_scanner, "scan_recordings",
        lambda runs_root, library_root: (
            scan_calls.append((runs_root, library_root))
            or {"scanned": 0, "unchanged": 2, "formation_events": 0}
        ),
    )
    monkeypatch.setattr(
        "chemistry_manager.trust.trusted_molecules",
        lambda *args, **kwargs: [],
    )

    store = ManagerStore(tmp_path / "manager" / "state.json")
    result = ingest_teacher_data(
        store, teacher_root, molecules, state_file=store.path
    )

    assert result["productions_found"] == 2
    assert result["productions_added"] == 2
    assert result["experiments_added"] == 2
    assert result["teacher_frames_added"] == 3
    assert len(scan_calls) == 1


def test_teacher_ingest_preserves_legacy_prod_layout_alongside_daily(
    tmp_path, monkeypatch
):
    teacher_root = tmp_path / "teacher_data"
    molecules = tmp_path / "molecules"
    molecules.mkdir(parents=True)

    legacy = teacher_root / "PROD_legacy"
    for path in (legacy / "experiments", legacy / "recordings", legacy / "shards"):
        path.mkdir(parents=True, exist_ok=True)
    (legacy / "production.json").write_text(
        json.dumps({"id": "PROD_legacy", "physics": "optimised-valence"}),
        encoding="utf-8",
    )
    (legacy / "shards" / "EXP_legacy.npz").write_bytes(b"shard")
    (legacy / "experiments" / "EXP_legacy.json").write_text(json.dumps({
        "status": "complete",
        "experiment_id": "EXP_legacy",
        "shard": "shards/EXP_legacy.npz",
        "teacher_frames": 2,
    }), encoding="utf-8")

    day = teacher_root / "2026-08-19"
    for path in (day / "invocations", day / "experiments", day / "recordings", day / "shards"):
        path.mkdir(parents=True, exist_ok=True)
    (day / "invocations" / "INV_daily.json").write_text(
        json.dumps({"id": "INV_daily", "physics": "optimised-valence"}),
        encoding="utf-8",
    )
    (day / "shards" / "EXP_daily.npz").write_bytes(b"shard")
    (day / "experiments" / "EXP_daily.json").write_text(json.dumps({
        "status": "complete",
        "experiment_id": "EXP_daily",
        "invocation_id": "INV_daily",
        "shard": "shards/EXP_daily.npz",
        "teacher_frames": 3,
    }), encoding="utf-8")

    (molecules / "formation_events.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        molecule_scanner, "scan_recordings",
        lambda runs_root, library_root: {
            "scanned": 0, "unchanged": 0, "formation_events": 0
        },
    )
    monkeypatch.setattr(
        "chemistry_manager.trust.trusted_molecules",
        lambda *args, **kwargs: [],
    )

    store = ManagerStore(tmp_path / "manager" / "state.json")
    result = ingest_teacher_data(
        store, teacher_root, molecules, state_file=store.path
    )

    assert result["productions_found"] == 2
    assert result["productions_added"] == 2
    assert result["experiments_added"] == 2
    assert result["teacher_frames_added"] == 5



def test_status_reports_idle_unvalidated_molecule(tmp_path, capsys):
    molecules = tmp_path / "molecules"
    _write_manager_molecule(molecules, "SP_idle")
    state = tmp_path / "state.json"

    assert main([
        "--state-file", str(state),
        "status", "--molecule-root", str(molecules),
    ]) == 0
    output = capsys.readouterr().out
    assert "Unvalidated, not queued:" in output
    assert "SP_idle" in output


def test_store_can_find_unique_product_ids_by_state(tmp_path):
    store = ManagerStore(tmp_path / "state.json")
    store.add_discovery_events(
        [event("one"), event("two")],
        tmp_path / "events.jsonl",
        initial_state=CandidateState.WAITING_QM,
        source_kind="full_cm_teacher_event",
    )
    assert store.product_ids(CandidateState.WAITING_QM) == ["SP_000001"]
    assert len(
        store.candidates_for_product(
            "SP_000001", CandidateState.WAITING_QM
        )
    ) == 2
