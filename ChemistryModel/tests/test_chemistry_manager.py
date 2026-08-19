import json

import molecule_scanner

from chemistry_manager import CandidateState, ManagerStore
from chemistry_manager.cli import main
from chemistry_manager.discovery import discover


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
    waiting = second.candidates(CandidateState.WAITING_FULL_CM)
    assert len(waiting) == 1
    assert waiting[0]["id"] == "EVENT_abc123"
    assert waiting[0]["provenance"]["seed"] == 7
    assert waiting[0]["products"][0]["id"] == "SP_000001"


def test_store_transitions_follow_the_v1_state_machine(tmp_path):
    store = ManagerStore(tmp_path / "state.json")
    store.add_discovery_event(event(), tmp_path / "events.jsonl")

    store.transition("EVENT_abc123", CandidateState.WAITING_QM)
    store.transition("EVENT_abc123", CandidateState.QM_VALIDATED)

    counts = store.counts()
    assert counts[CandidateState.WAITING_FULL_CM] == 0
    assert counts[CandidateState.QM_VALIDATED] == 1

    try:
        store.transition("EVENT_abc123", CandidateState.WAITING_QM)
    except ValueError as problem:
        assert "invalid candidate transition" in str(problem)
    else:
        raise AssertionError("terminal candidates must not move backwards")


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
    code = main(["--state-file", str(tmp_path / "state.json"), "status"])
    output = capsys.readouterr().out

    assert code == 0
    assert "Chemistry Manager" in output
    assert "Waiting for full-CM validation:" in output
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


def test_empty_processing_commands_return_cleanly(tmp_path, capsys):
    state = str(tmp_path / "state.json")

    assert main(["--state-file", state, "validate"]) == 0
    assert (
        capsys.readouterr().out.strip()
        == "Nothing waiting for full-ChemistryModel validation."
    )
    assert main(["--state-file", state, "qm"]) == 0
    assert capsys.readouterr().out.strip() == "Nothing waiting for QM validation."
