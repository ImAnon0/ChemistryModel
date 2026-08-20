import json

import molecule_scanner

from chemistry_manager import CandidateState, ManagerStore
from chemistry_manager.cli import main
from chemistry_manager.discovery import discover, ingest_teacher_data
from chemistry_manager.qm import process_qm_queue
from chemistry_manager import reaction_producer


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


def test_promote_existing_refreshes_provenance_without_reporting_new_queue(
    tmp_path,
):
    store = ManagerStore(tmp_path / "state.json")
    log = tmp_path / "events.jsonl"
    first = store.add_discovery_events([event("refresh")], log)
    second = store.add_discovery_events(
        [event("refresh")],
        log,
        source_kind="full_cm_teacher_event",
        source_extra={"production_id": "INV_refresh"},
        promote_existing=True,
    )

    assert first["added"] == 1
    assert second["added"] == 0
    assert second["duplicates"] == 1
    assert second["refreshed"] == 1

    candidate = store.candidates(CandidateState.WAITING_QM)[0]
    assert candidate["source"]["kind"] == "full_cm_teacher_event"
    assert candidate["source"]["production_id"] == "INV_refresh"


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
    assert second["refreshed_candidates"] == 1
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


def _planner_history_row(
    experiment_id="EXP_parent",
    *,
    reacted=True,
    stable=True,
    new_products=("SP_new",),
    trust="QM_VALIDATED",
    depth=0,
):
    outcome = {
        "reacted": bool(reacted),
        "stable": bool(stable),
        "reaction_event_count": 1 if reacted else 0,
        "new_product_species": list(new_products),
        "product_species": list(new_products),
        "current_product_trust": {
            product_id: trust for product_id in new_products
        },
        "postprocess_status": "complete",
    }
    row = {
        "id": experiment_id,
        "experiment_family": "microcell",
        "reactants": ["atom:H", "atom:C", "atom:O"],
        "temperature_K": 500.0,
        "cluster_radius_A": 4.0,
        "minimum_gap_A": 2.0,
        "inward_factor": 0.8,
        "atom_density_per_A3": 0.03,
        "experiment_outcome": outcome,
    }
    if depth:
        row["refinement_depth"] = depth
        row["refinement"] = {
            "depth": depth,
            "root_experiment_id": "EXP_root",
            "parent_experiment_id": "EXP_root",
            "mutation": {"type": "test", "to": depth},
        }
    return row


def test_refinement_parent_prefers_qm_validated_success_and_rejects_dead_runs():
    qm_parent = _planner_history_row(trust="QM_VALIDATED")
    unvalidated_parent = _planner_history_row(
        experiment_id="EXP_unvalidated", trust="UNVALIDATED"
    )
    rejected_parent = _planner_history_row(
        experiment_id="EXP_rejected", trust="REJECTED"
    )
    dead_parent = _planner_history_row(
        experiment_id="EXP_dead", reacted=False
    )
    unstable_parent = _planner_history_row(
        experiment_id="EXP_unstable", stable=False
    )

    assert (
        reaction_producer._refinement_parent_score(qm_parent)
        > reaction_producer._refinement_parent_score(unvalidated_parent)
        > 0.0
    )
    assert reaction_producer._refinement_parent_score(rejected_parent) == 0.0
    assert reaction_producer._refinement_parent_score(dead_parent) == 0.0
    assert reaction_producer._refinement_parent_score(unstable_parent) == 0.0


def test_refinement_parent_depth_and_child_budget_are_bounded():
    parent = _planner_history_row()
    history = [parent]

    for number in range(
        reaction_producer.REFINEMENT_MAX_CHILDREN_BY_PARENT_DEPTH[0]
    ):
        history.append({
            **_planner_history_row(
                experiment_id=f"EXP_child_{number}", depth=1
            ),
            "refinement": {
                "depth": 1,
                "root_experiment_id": parent["id"],
                "parent_experiment_id": parent["id"],
                "mutation": {"type": "temperature_K", "to": number},
            },
        })

    eligible = reaction_producer._eligible_refinement_parents(
        history,
        allowed_ids={"atom:H", "atom:C", "atom:O"},
    )
    assert all(row["spec"]["id"] != parent["id"] for row in eligible)

    depth_two = _planner_history_row(
        experiment_id="EXP_depth_two", depth=2
    )
    assert reaction_producer._refinement_parent_score(depth_two) == 0.0


def test_refinement_candidates_change_one_dimension_and_record_lineage(
    monkeypatch,
):
    parent = _planner_history_row()
    pool = {
        "atoms": ["atom:H", "atom:C", "atom:N", "atom:O"],
        "molecules": [],
    }

    def fake_load(reactant_id, molecule_root):
        symbol = reactant_id.split(":", 1)[1]
        return {
            "id": reactant_id,
            "formula": symbol,
            "symbols": [symbol],
            "positions": [[0.0, 0.0, 0.0]],
            "bonds": [],
        }

    monkeypatch.setattr(
        reaction_producer, "_load_reactant", fake_load
    )
    generator = reaction_producer.np.random.default_rng(123)
    candidates = reaction_producer._build_refinement_candidates(
        parent,
        0,
        generator,
        pool,
        "molecules",
        "balanced",
        {},
        [parent],
    )

    assert candidates
    assert all(
        candidate["refinement"]["parent_experiment_id"] == parent["id"]
        for candidate in candidates
    )
    assert all(candidate["refinement_depth"] == 1 for candidate in candidates)
    assert all(3 <= candidate["object_count"] <= 8 for candidate in candidates)

    temperature_children = [
        candidate for candidate in candidates
        if candidate["refinement"]["mutation"]["type"] == "temperature_K"
    ]
    assert temperature_children
    for candidate in temperature_children:
        assert candidate["reactants"] == parent["reactants"]
        assert candidate["cluster_radius_A"] == parent["cluster_radius_A"]
        assert candidate["minimum_gap_A"] == parent["minimum_gap_A"]
        assert candidate["inward_factor"] == parent["inward_factor"]
        assert candidate["temperature_K"] != parent["temperature_K"]


def test_refinement_reuses_no_completed_parent_mutation(monkeypatch):
    parent = _planner_history_row()
    used_child = {
        **_planner_history_row(
            experiment_id="EXP_used_child", depth=1
        ),
        "refinement": {
            "depth": 1,
            "root_experiment_id": parent["id"],
            "parent_experiment_id": parent["id"],
            "mutation": {
                "type": "temperature_K",
                "from": 500.0,
                "to": 250.0,
            },
        },
    }

    def fake_load(reactant_id, molecule_root):
        symbol = reactant_id.split(":", 1)[1]
        return {
            "id": reactant_id,
            "formula": symbol,
            "symbols": [symbol],
            "positions": [[0.0, 0.0, 0.0]],
            "bonds": [],
        }

    monkeypatch.setattr(
        reaction_producer, "_load_reactant", fake_load
    )
    generator = reaction_producer.np.random.default_rng(456)
    candidates = reaction_producer._build_refinement_candidates(
        parent,
        0,
        generator,
        {
            "atoms": ["atom:H", "atom:C", "atom:N", "atom:O"],
            "molecules": [],
        },
        "molecules",
        "balanced",
        {},
        [parent, used_child],
    )

    mutation_keys = {
        reaction_producer._refinement_mutation_key(
            candidate["refinement"]
        )
        for candidate in candidates
    }
    used_key = reaction_producer._refinement_mutation_key(
        used_child["refinement"]
    )
    assert used_key not in mutation_keys


def _fake_atomic_reactant(reactant_id):
    symbol = str(reactant_id).split(":", 1)[-1]
    return {
        "id": str(reactant_id),
        "formula": symbol,
        "symbols": [symbol],
        "positions": [[0.0, 0.0, 0.0]],
        "bonds": [],
    }


def test_wild_pair_collision_speed_reaches_beyond_normal_profile(
    monkeypatch,
):
    monkeypatch.setattr(
        reaction_producer, "_load_reactant",
        lambda reactant_id, molecule_root: _fake_atomic_reactant(reactant_id),
    )
    generator = reaction_producer.np.random.default_rng(99)
    candidate = reaction_producer._build_wild_pair_candidate(
        0,
        "atom_atom",
        "direct",
        "high",
        generator,
        {"atoms": ["atom:H", "atom:O"], "molecules": []},
        "molecules",
        "balanced",
        {},
        wild_dimension="collision_speed",
    )

    assert candidate["wild_exploration"]["dimension"] == "collision_speed"
    assert candidate["speed_class"] == "wild"
    assert (
        reaction_producer.WILD_PAIR_APPROACH_FACTOR_RANGE[0]
        <= candidate["approach_factor"]
        <= reaction_producer.WILD_PAIR_APPROACH_FACTOR_RANGE[1]
    )
    assert candidate["approach_factor"] > max(
        high for low, high in
        reaction_producer.SPEED_RANGES["balanced"].values()
    )


def test_wild_microreactor_modes_keep_valid_object_bounds_and_provenance(
    monkeypatch,
):
    monkeypatch.setattr(
        reaction_producer, "_load_reactant",
        lambda reactant_id, molecule_root: _fake_atomic_reactant(reactant_id),
    )
    pool = {
        "atoms": ["atom:H", "atom:C", "atom:N", "atom:O"],
        "molecules": [],
    }

    for index, dimension in enumerate(
        reaction_producer.WILD_MICROCELL_DIMENSIONS
    ):
        generator = reaction_producer.np.random.default_rng(500 + index)
        candidate = reaction_producer._build_wild_microcell_candidate(
            0,
            generator,
            pool,
            "molecules",
            "balanced",
            {},
            wild_dimension=dimension,
        )
        assert candidate["category"] == "microreactor_wild"
        assert candidate["wild_exploration"]["dimension"] == dimension
        assert 3 <= candidate["object_count"] <= 8
        assert candidate["cluster_radius_A"] > 0.0
        assert candidate["atom_density_per_A3"] > 0.0

        if dimension == "temperature":
            low, high = reaction_producer.WILD_TEMPERATURE_RANGE_K
            assert low <= candidate["temperature_K"] <= high
        elif dimension == "inward_factor":
            value = candidate["inward_factor"]
            assert (
                reaction_producer.WILD_MICROCELL_INWARD_LOW_RANGE[0]
                <= value
                <= reaction_producer.WILD_MICROCELL_INWARD_LOW_RANGE[1]
                or
                reaction_producer.WILD_MICROCELL_INWARD_HIGH_RANGE[0]
                <= value
                <= reaction_producer.WILD_MICROCELL_INWARD_HIGH_RANGE[1]
            )
        elif dimension == "minimum_gap":
            value = candidate["minimum_gap_A"]
            assert (
                reaction_producer.WILD_MICROCELL_GAP_TIGHT_RANGE_A[0]
                <= value
                <= reaction_producer.WILD_MICROCELL_GAP_TIGHT_RANGE_A[1]
                or
                reaction_producer.WILD_MICROCELL_GAP_LOOSE_RANGE_A[0]
                <= value
                <= reaction_producer.WILD_MICROCELL_GAP_LOOSE_RANGE_A[1]
            )
        elif dimension == "object_count":
            assert candidate["object_count"] in (
                reaction_producer.WILD_MICROCELL_OBJECT_COUNTS
            )


def test_planner_can_force_wild_microreactor_without_changing_family_weights(
    monkeypatch,
):
    monkeypatch.setattr(
        reaction_producer,
        "allowed_reactants",
        lambda *args, **kwargs: {
            "atoms": ["atom:H", "atom:C", "atom:N", "atom:O"],
            "molecules": [],
            "trusted_molecule_records": [],
        },
    )
    monkeypatch.setattr(
        reaction_producer, "_load_reactant",
        lambda reactant_id, molecule_root: _fake_atomic_reactant(reactant_id),
    )
    monkeypatch.setattr(
        reaction_producer,
        "_choose_experiment_family",
        lambda generator: "microcell",
    )

    specs, _ = reaction_producer.generate_experiment_specs(
        4,
        12345,
        molecule_root="molecules",
        history=[],
        wild_probability=1.0,
    )

    assert len(specs) == 4
    assert all(spec["experiment_family"] == "microcell" for spec in specs)
    assert all(spec["planner"]["mode"] == "wild_exploration" for spec in specs)
    assert all(
        spec["planner"]["wild_family"] == "microcell" for spec in specs
    )
    assert all("wild_exploration" in spec for spec in specs)


def test_pair_novelty_can_be_tested_without_rng_family_assumption(
    monkeypatch,
):
    monkeypatch.setattr(
        reaction_producer,
        "allowed_reactants",
        lambda *args, **kwargs: {
            "atoms": ["atom:H", "atom:C", "atom:N", "atom:O"],
            "molecules": [],
            "trusted_molecule_records": [],
        },
    )
    monkeypatch.setattr(
        reaction_producer, "_load_reactant",
        lambda reactant_id, molecule_root: _fake_atomic_reactant(reactant_id),
    )
    monkeypatch.setattr(
        reaction_producer,
        "_choose_experiment_family",
        lambda generator: "pair",
    )

    specs, _ = reaction_producer.generate_experiment_specs(
        3,
        77,
        molecule_root="molecules",
        history=[],
        wild_probability=0.0,
    )

    assert len(specs) == 3
    assert all(spec["experiment_family"] == "pair" for spec in specs)
    assert all(spec["planner"]["mode"] == "pair_novelty" for spec in specs)


def test_normal_microreactor_object_count_contract_is_three_to_eight(
    monkeypatch,
):
    monkeypatch.setattr(
        reaction_producer, "_load_reactant",
        lambda reactant_id, molecule_root: _fake_atomic_reactant(reactant_id),
    )
    generator = reaction_producer.np.random.default_rng(2026)
    pool = {
        "atoms": ["atom:H", "atom:C", "atom:N", "atom:O"],
        "molecules": [],
    }
    counts = {
        reaction_producer._build_microcell_candidate(
            number,
            generator,
            pool,
            "molecules",
            "balanced",
            {},
        )["object_count"]
        for number in range(64)
    }

    assert counts
    assert min(counts) >= 3
    assert max(counts) <= 8

