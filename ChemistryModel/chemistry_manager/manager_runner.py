"""Resumable sequential orchestration for autonomous Chemistry Manager runs."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from datetime import datetime
from pathlib import Path

import qm_structure_validator

from .discovery import ingest_teacher_data
from .qm import process_qm_queue
from .reaction_producer import (
    WILD_EXPLORATION_PROBABILITY,
    _fresh_master_seed,
    _local_day_string,
    run_production,
)
from .state import CandidateState
from .trust import trusted_molecules


FORMAT_VERSION = 1

# Windows can briefly deny an atomic rename while Defender, an indexer, an
# editor, or another reader has the freshly-written file open. Keep atomic
# replacement semantics, but tolerate those transient sharing/access locks.
_ATOMIC_REPLACE_RETRY_DELAYS_S = (
    0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.00,
)


def _replace_with_retry(source, destination):
    """Atomically replace destination, retrying transient Windows file locks."""
    source = Path(source)
    destination = Path(destination)

    for attempt in range(len(_ATOMIC_REPLACE_RETRY_DELAYS_S) + 1):
        try:
            os.replace(source, destination)
            return
        except OSError as problem:
            winerror = getattr(problem, "winerror", None)
            retryable = (
                isinstance(problem, PermissionError)
                or winerror in (5, 32, 33)
            )
            if (
                not retryable
                or attempt >= len(_ATOMIC_REPLACE_RETRY_DELAYS_S)
            ):
                raise
            time.sleep(_ATOMIC_REPLACE_RETRY_DELAYS_S[attempt])


def _now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _replace_with_retry(temporary, path)


def _planning_seed(master_seed, sequence):
    digest = hashlib.sha256(
        f"{int(master_seed)}:{int(sequence)}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _new_run_id():
    return (
        "RUN_"
        + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        + "_"
        + secrets.token_hex(6)
    )


def _child_invocation_id(run_id, sequence):
    return f"INV_{str(run_id)[4:]}_{int(sequence):05d}"


def _find_receipt(output_root, run_id):
    candidate = Path(str(run_id))
    if candidate.is_file():
        return candidate
    matches = list(
        Path(output_root).glob(f"*/manager_runs/{str(run_id)}.json")
    )
    if not matches:
        raise ValueError(f"autonomous manager run not found: {run_id}")
    if len(matches) > 1:
        raise ValueError(f"multiple receipts found for manager run: {run_id}")
    return matches[0]


def _load_receipt(path):
    try:
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as problem:
        raise ValueError(f"manager run receipt is not valid JSON: {path}") from problem
    if not isinstance(receipt, dict):
        raise ValueError("manager run receipt root must be an object")
    if receipt.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            "unsupported manager run format: "
            f"{receipt.get('format_version')!r}"
        )
    return receipt


def _experiment_record(day_root, experiment_id):
    path = Path(day_root) / "experiments" / f"{experiment_id}.json"
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as problem:
        raise ValueError(f"experiment record is not valid JSON: {path}") from problem
    return record


def _active_experiment(receipt, receipt_path):
    active = receipt.get("active_step")
    if not isinstance(active, dict):
        return None
    day_root = Path(receipt_path).parent.parent
    invocation_path = (
        day_root / "invocations" / f"{active['invocation_id']}.json"
    )
    if not invocation_path.is_file():
        return None
    try:
        invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as problem:
        raise ValueError(
            f"child invocation is not valid JSON: {invocation_path}"
        ) from problem
    specifications = invocation.get("specifications") or []
    experiment_id = active.get("experiment_id")
    if experiment_id is None and specifications:
        experiment_id = specifications[0].get("id")
    if not experiment_id:
        return None
    record = _experiment_record(day_root, experiment_id)
    if record is None or record.get("status") not in ("complete", "failed"):
        return None
    return record


def _observer_item_from_record(record, requested_ps):
    if record.get("status") == "failed":
        return {
            "experiment_id": record.get("experiment_id"),
            "family": record.get("experiment_family", "unknown"),
            "outcome": "failed",
            "reaction_events": 0,
            "products": [],
            "error": record.get("error", "unknown error"),
            "requested_picoseconds": float(requested_ps),
            "termination_reason": "failed",
        }
    entry = record.get("run_entry") or {}
    outcome = record.get("experiment_outcome") or {}
    termination = entry.get("termination") or {}
    actual = float(entry.get("picoseconds", 0.0))
    reason = str(
        entry.get("termination_reason")
        or termination.get("reason")
        or (
            "duration_complete"
            if actual + 1e-9 >= float(requested_ps)
            else "ended_early"
        )
    )
    return {
        "experiment_id": record.get("experiment_id"),
        "family": record.get("experiment_family", "unknown"),
        "outcome": entry.get("characterisation_outcome", "unknown"),
        "reaction_events": int(outcome.get("reaction_event_count", 0)),
        "products": list(outcome.get("product_results") or []),
        "actual_picoseconds": actual,
        "requested_picoseconds": float(requested_ps),
        "termination_reason": reason,
    }


def _record_attempt(receipt, receipt_path, result, observer_item):
    completed = int(result.get("completed_now", 0))
    failed = int(result.get("failed_now", 0))
    if completed + failed != 1:
        raise ValueError(
            "one-experiment production returned an invalid attempt count: "
            f"completed={completed}, failed={failed}"
        )
    experiment_ids = [str(value) for value in result.get("experiment_ids", [])]
    if len(experiment_ids) != 1:
        raise ValueError("one-experiment production did not report one experiment ID")

    experiment_id = experiment_ids[0]
    active = dict(receipt.get("active_step") or {})
    receipt["attempted_experiments"] += 1
    receipt["completed_experiments"] += completed
    receipt["failed_experiments"] += failed
    if completed:
        receipt["experiments_since_last_qm"] += 1
    receipt["experiment_ids"].append(experiment_id)
    receipt["invocation_ids"].append(str(result["invocation_id"]))
    if failed:
        receipt["failed_experiment_ids"].append(experiment_id)
    receipt["planning_seeds"].append(int(active["planning_seed"]))
    receipt["experiment_results"].append(dict(observer_item or {}))
    receipt["active_step"] = None
    receipt["updated_time"] = _now_iso()
    _atomic_json(receipt_path, receipt)


def _recover_active_step(receipt, receipt_path, store):
    record = _active_experiment(receipt, receipt_path)
    if record is None:
        return None

    # A complete experiment file is authoritative. If a crash happened before
    # its controlled-event handoff completed, the established ingest recovery
    # path safely scans/routes it without repeating MD.
    if record.get("status") == "complete" and not isinstance(
        record.get("experiment_outcome"), dict
    ):
        ingest_teacher_data(
            store,
            receipt["output_root"],
            receipt["molecule_root"],
            qm_root=receipt.get("qm_root"),
            state_file=store.path,
        )

    result = {
        "invocation_id": receipt["active_step"]["invocation_id"],
        "experiment_ids": [record["experiment_id"]],
        "completed_now": int(record.get("status") == "complete"),
        "failed_now": int(record.get("status") == "failed"),
    }
    item = _observer_item_from_record(record, receipt["duration_ps"])
    _record_attempt(receipt, receipt_path, result, item)
    return item


def _qm_checkpoint(
    receipt, receipt_path, store, *, kind, qm_processor,
    qm_observer=None,
):
    waiting = store.candidates(CandidateState.WAITING_QM)
    if waiting:
        summary = qm_processor(
            store,
            molecule_root=receipt["molecule_root"],
            qm_root=receipt.get("qm_root"),
            method=receipt["qm_method"],
            basis=receipt["qm_basis"],
            threads=receipt["qm_threads"],
            memory=receipt["qm_memory"],
        )
    else:
        summary = {
            "candidates_seen": 0,
            "validated": 0,
            "rejected": 0,
            "still_waiting": 0,
            "molecules_validated": 0,
            "molecules_rejected": 0,
            "molecules_reused": 0,
            "errors": [],
        }

    checkpoint = {
        "kind": str(kind),
        "time": _now_iso(),
        "after_attempted_experiments": receipt["attempted_experiments"],
        "after_completed_experiments": receipt["completed_experiments"],
        "waiting_before": len(waiting),
        **summary,
        "trusted_reactants": 4 + len(trusted_molecules(
            receipt["molecule_root"], qm_root=receipt.get("qm_root")
        )),
    }
    receipt["qm_checkpoints"].append(checkpoint)
    receipt["qm_validated_count"] += int(summary.get("validated", 0))
    receipt["qm_rejected_count"] += int(summary.get("rejected", 0))
    receipt["experiments_since_last_qm"] = 0
    receipt["updated_time"] = _now_iso()
    _atomic_json(receipt_path, receipt)
    if qm_observer:
        qm_observer(dict(checkpoint))
    return checkpoint


def _resolved_setting(value, default):
    return default if value is None else value


def run_manager(
    store, *, count=None, duration_ps=None, qm_every=None, device=None,
    master_seed=None, profile=None, output_root="teacher_data",
    molecule_root=None, qm_root=None, physics=None,
    ordinary_interval_fs=None, event_window_fs=None,
    diagnostic_sample_fs=None, capture_every=None,
    wild_probability=None,
    qm_method=None, qm_basis=None, qm_threads=None, qm_memory=None,
    resume=None, progress=None, result_observer=None, qm_observer=None,
    producer=run_production, qm_processor=process_qm_queue,
):
    """Run one planned/persisted MD experiment at a time with QM checkpoints."""
    if resume:
        receipt_path = _find_receipt(output_root, resume)
        receipt = _load_receipt(receipt_path)
        if Path(store.path).resolve() != Path(receipt["state_file"]).resolve():
            raise ValueError(
                "resume must use the original manager state file: "
                f"{receipt['state_file']}"
            )
        # Receipts created before broad-envelope exploration existed had no
        # wild channel. Treat them as 0.0 so resuming preserves their original
        # planner behavior instead of silently introducing new experiments.
        receipt.setdefault("wild_probability", 0.0)
        supplied = {
            "requested_experiments": count,
            "duration_ps": duration_ps,
            "qm_every": qm_every,
            "device": device,
            "master_seed": master_seed,
            "profile": profile,
            "molecule_root": molecule_root,
            "qm_root": qm_root,
            "physics": physics,
            "ordinary_interval_fs": ordinary_interval_fs,
            "event_window_fs": event_window_fs,
            "diagnostic_sample_fs": diagnostic_sample_fs,
            "capture_every": capture_every,
            "wild_probability": wild_probability,
            "qm_method": qm_method,
            "qm_basis": qm_basis,
            "qm_threads": qm_threads,
            "qm_memory": qm_memory,
        }
        for key, value in supplied.items():
            if value is not None and value != receipt.get(key):
                raise ValueError(
                    f"resume cannot change {key}: original "
                    f"{receipt.get(key)!r}, requested {value!r}"
                )
        if receipt.get("status") == "complete":
            return receipt
        receipt["status"] = "running"
        receipt["updated_time"] = _now_iso()
        _atomic_json(receipt_path, receipt)
    else:
        requested = int(_resolved_setting(count, 100))
        duration = float(_resolved_setting(duration_ps, 2.0))
        interval = int(_resolved_setting(qm_every, 5))
        wild = float(_resolved_setting(
            wild_probability, WILD_EXPLORATION_PROBABILITY
        ))
        if requested < 1:
            raise ValueError("experiment count must be at least one")
        if duration <= 0:
            raise ValueError("duration must be greater than zero")
        if interval < 1:
            raise ValueError("qm-every must be at least one")
        if not 0.0 <= wild <= 1.0:
            raise ValueError(
                "wild exploration probability must be between 0 and 1"
            )
        used_seed = _fresh_master_seed() if master_seed is None else int(master_seed)
        run_id = _new_run_id()
        day = _local_day_string()
        receipt_path = (
            Path(output_root) / day / "manager_runs" / f"{run_id}.json"
        )
        if device is None:
            import torch
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = str(device)
        receipt = {
            "format_version": FORMAT_VERSION,
            "run_id": run_id,
            "status": "running",
            "requested_experiments": requested,
            "attempted_experiments": 0,
            "completed_experiments": 0,
            "failed_experiments": 0,
            "duration_ps": duration,
            "qm_every": interval,
            "experiments_since_last_qm": 0,
            "qm_checkpoints": [],
            "qm_validated_count": 0,
            "qm_rejected_count": 0,
            "master_seed": used_seed,
            "master_seed_source": "random" if master_seed is None else "explicit",
            "planning_seeds": [],
            "device": resolved_device,
            "profile": str(_resolved_setting(profile, "balanced")),
            "physics": str(_resolved_setting(physics, "optimised-valence")),
            "output_root": str(output_root),
            "receipt_path": str(receipt_path),
            "molecule_root": str(_resolved_setting(molecule_root, "molecules")),
            "qm_root": None if qm_root is None else str(qm_root),
            "state_file": str(store.path),
            "ordinary_interval_fs": float(_resolved_setting(ordinary_interval_fs, 10.0)),
            "event_window_fs": float(_resolved_setting(event_window_fs, 5.0)),
            "diagnostic_sample_fs": float(_resolved_setting(diagnostic_sample_fs, 1.0)),
            "capture_every": int(_resolved_setting(capture_every, 4)),
            "wild_probability": wild,
            "qm_method": str(_resolved_setting(qm_method, qm_structure_validator.DEFAULT_METHOD)),
            "qm_basis": str(_resolved_setting(qm_basis, qm_structure_validator.DEFAULT_BASIS)),
            "qm_threads": int(_resolved_setting(qm_threads, 8)),
            "qm_memory": str(_resolved_setting(qm_memory, "4 GB")),
            "started_time": _now_iso(),
            "updated_time": _now_iso(),
            "completed_time": None,
            "experiment_ids": [],
            "failed_experiment_ids": [],
            "invocation_ids": [],
            "experiment_results": [],
            "active_step": None,
        }
        _atomic_json(receipt_path, receipt)

    try:
        recovered = _recover_active_step(receipt, receipt_path, store)
        if recovered is not None and result_observer:
            result_observer(dict(recovered))

        if (
            receipt["attempted_experiments"]
            < receipt["requested_experiments"]
            and receipt["experiments_since_last_qm"]
            >= receipt["qm_every"]
        ):
            _qm_checkpoint(
                receipt, receipt_path, store,
                kind="periodic", qm_processor=qm_processor,
                qm_observer=qm_observer,
            )

        while receipt["attempted_experiments"] < receipt["requested_experiments"]:
            sequence = receipt["attempted_experiments"] + 1
            planning_seed = _planning_seed(receipt["master_seed"], sequence)
            child_id = _child_invocation_id(receipt["run_id"], sequence)
            receipt["active_step"] = {
                "sequence": sequence,
                "planning_seed": planning_seed,
                "invocation_id": child_id,
                "started_time": _now_iso(),
            }
            receipt["updated_time"] = _now_iso()
            _atomic_json(receipt_path, receipt)

            observed = []

            def child_progress(_number, _total, spec):
                if progress:
                    progress(sequence, receipt["requested_experiments"], spec)

            def child_result(item):
                observed.append(dict(item))
                if result_observer:
                    result_observer(dict(item))

            result = producer(
                store,
                count=1,
                duration_ps=receipt["duration_ps"],
                master_seed=planning_seed,
                profile=receipt["profile"],
                output_root=receipt["output_root"],
                molecule_root=receipt["molecule_root"],
                qm_root=receipt.get("qm_root"),
                device=receipt.get("device"),
                ordinary_interval_fs=receipt["ordinary_interval_fs"],
                event_window_fs=receipt["event_window_fs"],
                diagnostic_sample_fs=receipt["diagnostic_sample_fs"],
                capture_every=receipt["capture_every"],
                physics=receipt["physics"],
                progress=child_progress,
                result_observer=child_result,
                invocation_id=child_id,
                manager_run_id=receipt["run_id"],
                planner_pair_offset=sum(
                    str(row.get("family")) == "pair"
                    for row in receipt["experiment_results"]
                ),
                wild_probability=receipt["wild_probability"],
            )
            item = observed[-1] if observed else {}
            _record_attempt(receipt, receipt_path, result, item)

            if (
                receipt["experiments_since_last_qm"]
                >= receipt["qm_every"]
            ):
                _qm_checkpoint(
                    receipt, receipt_path, store,
                    kind="periodic", qm_processor=qm_processor,
                    qm_observer=qm_observer,
                )

        latest_checkpoint = (
            receipt["qm_checkpoints"][-1]
            if receipt["qm_checkpoints"] else {}
        )
        if not (
            latest_checkpoint.get("kind") == "final"
            and latest_checkpoint.get("after_attempted_experiments")
            == receipt["attempted_experiments"]
        ):
            _qm_checkpoint(
                receipt, receipt_path, store,
                kind="final", qm_processor=qm_processor,
                qm_observer=qm_observer,
            )
        receipt["status"] = "complete"
        receipt["completed_time"] = _now_iso()
        receipt["updated_time"] = receipt["completed_time"]
        receipt["active_step"] = None
        _atomic_json(receipt_path, receipt)
        return receipt
    except Exception as problem:
        receipt["status"] = "interrupted"
        receipt["last_error"] = f"{type(problem).__name__}: {problem}"
        receipt["updated_time"] = _now_iso()
        _atomic_json(receipt_path, receipt)
        raise
