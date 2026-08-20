import json
from pathlib import Path

import pytest

from chemistry_manager import CandidateState, ManagerStore
import chemistry_manager.manager_runner as runner
from chemistry_manager.cli import build_parser


DAY = "2026-08-20"


def _event(name="one"):
    return {
        "event_id": name,
        "products": [{"id": f"SP_{name}", "formula": "H2"}],
        "reactants": [{"formula": "H", "count": 2}],
        "formed_bonds": [],
        "broken_bonds": [],
    }


def _fake_producer_factory(statuses=None, fail_call=None, knowledge=None):
    calls = []
    statuses = list(statuses or [])
    knowledge = knowledge if knowledge is not None else {}

    def fake_producer(store, **kwargs):
        number = len(calls) + 1
        if fail_call == number and not knowledge.get("failure_raised"):
            knowledge["failure_raised"] = True
            raise RuntimeError("orchestrator interruption")

        status = statuses[number - 1] if number <= len(statuses) else "complete"
        family = str(knowledge.get("family", "microcell"))
        spec = {
            "id": f"EXP_{kwargs['invocation_id']}",
            "category": "microreactor",
            "experiment_family": family,
            "reactant_a": "3 objects",
            "reactant_b": "H + C + O",
            "planner": {
                "mode": (
                    "microreactor_outcome_aware"
                    if knowledge.get("previous_complete")
                    else "microreactor_exploration"
                )
            },
        }
        calls.append({
            "count": kwargs["count"],
            "seed": kwargs["master_seed"],
            "invocation_id": kwargs["invocation_id"],
            "pair_offset": kwargs["planner_pair_offset"],
            "wild_probability": kwargs["wild_probability"],
            "knowledge_before": dict(knowledge),
            "spec": dict(spec),
        })
        kwargs["progress"](1, 1, spec)

        root = Path(kwargs["output_root"]) / DAY
        (root / "invocations").mkdir(parents=True, exist_ok=True)
        (root / "experiments").mkdir(parents=True, exist_ok=True)
        invocation = {
            "id": kwargs["invocation_id"],
            "status": "complete",
            "manager_run_id": kwargs["manager_run_id"],
            "specifications": [spec],
        }
        (root / "invocations" / f"{kwargs['invocation_id']}.json").write_text(
            json.dumps(invocation), encoding="utf-8"
        )
        record = {
            "status": status,
            "experiment_id": spec["id"],
            "experiment_family": family,
            "specification": spec,
        }
        if status == "complete":
            record["run_entry"] = {
                "characterisation_outcome": "no reaction",
                "picoseconds": kwargs["duration_ps"],
            }
            record["experiment_outcome"] = {
                "reaction_event_count": 0,
                "product_results": [],
            }
            item = {
                "experiment_id": spec["id"],
                "family": family,
                "outcome": "no reaction",
                "reaction_events": 0,
                "products": [],
                "actual_picoseconds": kwargs["duration_ps"],
                "requested_picoseconds": kwargs["duration_ps"],
                "termination_reason": "duration_complete",
            }
            knowledge["previous_complete"] = True
        else:
            record["error"] = "synthetic MD failure"
            item = {
                "experiment_id": spec["id"],
                "family": family,
                "outcome": "failed",
                "reaction_events": 0,
                "products": [],
                "error": record["error"],
                "requested_picoseconds": kwargs["duration_ps"],
                "termination_reason": "failed",
            }
        (root / "experiments" / f"{spec['id']}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
        kwargs["result_observer"](item)
        return {
            "invocation_id": kwargs["invocation_id"],
            "experiment_ids": [spec["id"]],
            "completed_now": int(status == "complete"),
            "failed_now": int(status == "failed"),
        }

    return fake_producer, calls


@pytest.fixture
def isolated_runner(monkeypatch):
    monkeypatch.setattr(runner, "_local_day_string", lambda: DAY)
    monkeypatch.setattr(runner, "trusted_molecules", lambda *a, **k: [])


def test_autonomous_plans_one_at_a_time_and_next_plan_sees_history(
    tmp_path, isolated_runner,
):
    fake, calls = _fake_producer_factory()
    planned = []
    receipt = runner.run_manager(
        ManagerStore(tmp_path / "state.json"),
        count=3, duration_ps=.01, qm_every=5,
        output_root=tmp_path / "teacher", molecule_root=tmp_path / "molecules",
        device="cpu", producer=fake,
        progress=lambda number, total, spec: planned.append(spec["planner"]["mode"]),
    )

    assert [row["count"] for row in calls] == [1, 1, 1]
    assert [row["wild_probability"] for row in calls] == [0.08, 0.08, 0.08]
    assert len({row["seed"] for row in calls}) == 3
    assert planned == [
        "microreactor_exploration",
        "microreactor_outcome_aware",
        "microreactor_outcome_aware",
    ]
    assert receipt["completed_experiments"] == 3


def test_autonomous_cli_defaults_and_resume_surface():
    parser = build_parser()
    options = parser.parse_args([
        "run", "--count", "100", "--duration", "2.0",
        "--qm-every", "5", "--device", "cuda",
    ])
    assert options.command == "run"
    assert options.count == 100
    assert options.duration == 2.0
    assert options.qm_every == 5
    assert options.device == "cuda"
    assert options.wild_probability is None

    explicit = parser.parse_args([
        "run", "--count", "10", "--wild-probability", "0.2"
    ])
    assert explicit.wild_probability == 0.2
    assert parser.parse_args(["run", "--resume", "RUN_test"]).resume == "RUN_test"


def test_sequential_pair_planning_preserves_pair_category_offset(
    tmp_path, isolated_runner,
):
    fake, calls = _fake_producer_factory(knowledge={"family": "pair"})
    runner.run_manager(
        ManagerStore(tmp_path / "state.json"),
        count=3, duration_ps=.01, qm_every=5,
        output_root=tmp_path / "teacher", molecule_root=tmp_path / "molecules",
        device="cpu", producer=fake,
    )
    assert [row["pair_offset"] for row in calls] == [0, 1, 2]


def test_periodic_empty_checkpoints_skip_qm_and_final_check_still_occurs(
    tmp_path, isolated_runner,
):
    fake, _ = _fake_producer_factory()
    qm_calls = []
    checkpoints = []

    def should_not_run(*args, **kwargs):
        qm_calls.append(True)
        raise AssertionError("empty QM queue should skip the processor")

    receipt = runner.run_manager(
        ManagerStore(tmp_path / "state.json"),
        count=5, duration_ps=.01, qm_every=2,
        output_root=tmp_path / "teacher", molecule_root=tmp_path / "molecules",
        device="cpu", producer=fake, qm_processor=should_not_run,
        qm_observer=checkpoints.append,
    )

    assert not qm_calls
    assert [row["kind"] for row in checkpoints] == [
        "periodic", "periodic", "final",
    ]
    assert [row["after_completed_experiments"] for row in checkpoints] == [2, 4, 5]
    assert receipt["status"] == "complete"


@pytest.mark.parametrize("decision", ["validated", "rejected"])
def test_qm_decision_is_visible_before_the_very_next_plan(
    tmp_path, isolated_runner, decision,
):
    store = ManagerStore(tmp_path / "state.json")
    store.add_discovery_event(
        _event(decision), tmp_path / "events.jsonl",
        initial_state=CandidateState.WAITING_QM,
    )
    knowledge = {}
    fake, calls = _fake_producer_factory(knowledge=knowledge)

    def fake_qm(store, **kwargs):
        final = (
            CandidateState.QM_VALIDATED
            if decision == "validated" else CandidateState.QM_REJECTED
        )
        for candidate in store.candidates(CandidateState.WAITING_QM):
            store.record_qm_result(
                candidate["id"], {"status": decision}, final_state=final
            )
        knowledge["qm_decision"] = decision
        return {
            "candidates_seen": 1,
            "validated": int(decision == "validated"),
            "rejected": int(decision == "rejected"),
            "still_waiting": 0,
            "molecules_validated": int(decision == "validated"),
            "molecules_rejected": int(decision == "rejected"),
            "molecules_reused": 0,
            "errors": [],
        }

    runner.run_manager(
        store, count=2, duration_ps=.01, qm_every=1,
        output_root=tmp_path / "teacher", molecule_root=tmp_path / "molecules",
        device="cpu", producer=fake, qm_processor=fake_qm,
    )

    assert calls[0]["knowledge_before"].get("qm_decision") is None
    assert calls[1]["knowledge_before"]["qm_decision"] == decision


def test_failed_md_does_not_abort_budget_or_advance_completed_qm_counter(
    tmp_path, isolated_runner,
):
    fake, calls = _fake_producer_factory(["failed", "complete", "complete"])
    checkpoints = []
    receipt = runner.run_manager(
        ManagerStore(tmp_path / "state.json"),
        count=3, duration_ps=.01, qm_every=2,
        output_root=tmp_path / "teacher", molecule_root=tmp_path / "molecules",
        device="cpu", producer=fake, qm_observer=checkpoints.append,
    )

    assert len(calls) == 3
    assert receipt["completed_experiments"] == 2
    assert receipt["failed_experiments"] == 1
    assert checkpoints[0]["kind"] == "periodic"
    assert checkpoints[0]["after_attempted_experiments"] == 3
    assert checkpoints[0]["after_completed_experiments"] == 2


def test_receipt_is_atomic_and_resume_does_not_repeat_completed_experiment(
    tmp_path, isolated_runner,
):
    knowledge = {}
    interrupted, first_calls = _fake_producer_factory(
        fail_call=2, knowledge=knowledge
    )
    store = ManagerStore(tmp_path / "state.json")
    output = tmp_path / "teacher"

    with pytest.raises(RuntimeError, match="orchestrator interruption"):
        runner.run_manager(
            store, count=3, duration_ps=.01, qm_every=5,
            output_root=output, molecule_root=tmp_path / "molecules",
            device="cpu", master_seed=77, producer=interrupted,
        )

    receipts = list(output.glob("*/manager_runs/RUN_*.json"))
    assert len(receipts) == 1
    partial = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert partial["completed_experiments"] == 1
    assert partial["status"] == "interrupted"
    assert not receipts[0].with_name(receipts[0].name + ".tmp").exists()

    resumed, resumed_calls = _fake_producer_factory(knowledge=knowledge)
    final = runner.run_manager(
        store, output_root=output, resume=partial["run_id"], producer=resumed,
    )

    assert len(first_calls) == 1
    assert len(resumed_calls) == 2
    assert final["completed_experiments"] == 3
    assert len(final["experiment_ids"]) == 3
    assert len(set(final["experiment_ids"])) == 3
    assert final["status"] == "complete"


def test_resume_rejects_changed_scientific_or_runtime_settings(
    tmp_path, isolated_runner,
):
    fake, _ = _fake_producer_factory(fail_call=1)
    store = ManagerStore(tmp_path / "state.json")
    output = tmp_path / "teacher"
    with pytest.raises(RuntimeError):
        runner.run_manager(
            store, count=2, duration_ps=.02, qm_every=1,
            output_root=output, molecule_root=tmp_path / "molecules",
            device="cpu", producer=fake,
        )
    receipt = json.loads(next(output.glob("*/manager_runs/RUN_*.json")).read_text())

    with pytest.raises(ValueError, match="cannot change duration_ps"):
        runner.run_manager(
            store, output_root=output, resume=receipt["run_id"],
            duration_ps=.03, producer=fake,
        )

    with pytest.raises(ValueError, match="cannot change wild_probability"):
        runner.run_manager(
            store, output_root=output, resume=receipt["run_id"],
            wild_probability=.2, producer=fake,
        )
