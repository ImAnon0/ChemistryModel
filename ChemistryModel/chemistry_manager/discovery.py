"""Handoff from the existing molecule scanner into the manager queue."""

import json
import os
from pathlib import Path

import molecule_library
import molecule_scanner


def event_log_path(molecule_root):
    return Path(molecule_root) / molecule_library.EVENT_LOG


def read_events(path):
    path = Path(path)
    if not path.is_file():
        return [], []

    events = []
    errors = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as problem:
                errors.append(f"line {line_number}: {problem.msg}")
                continue
            if not isinstance(event, dict) or not event.get("event_id"):
                errors.append(f"line {line_number}: missing event_id")
                continue
            events.append(event)
    return events, errors


def discover(store, runs_root, molecule_root, *, scan=True, progress=None):
    """Run the established scanner, then queue its stable event identities."""

    scan_summary = None
    if scan:
        scan_summary = molecule_scanner.scan_recordings(
            runs_root=str(runs_root),
            library_root=str(molecule_root),
            progress=progress,
        )

    path = event_log_path(molecule_root)
    events, errors = read_events(path)
    from .state import CandidateState

    imported = store.add_discovery_events(
        events,
        path,
        initial_state=CandidateState.WAITING_QM,
        source_kind="full_cm_formation_event",
    )

    return {
        "scan": scan_summary,
        "events_read": len(events),
        "queued": imported["added"],
        "already_known": imported["duplicates"],
        "errors": errors,
        "event_log": str(path),
    }


def _load_json(path):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as problem:
        raise ValueError(f"invalid teacher-data JSON: {path}: {problem}") from problem


def _teacher_registry_path(state_file):
    state_path = Path(state_file)
    return state_path.with_name("teacher_data.json")


def _load_teacher_registry(path):
    path = Path(path)
    if not path.exists():
        return {"format_version": 1, "productions": {}, "experiments": {}}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as problem:
        raise ValueError(f"teacher-data registry is not valid JSON: {path}") from problem
    if not isinstance(document, dict):
        raise ValueError("teacher-data registry root must be a JSON object")
    if document.get("format_version") != 1:
        raise ValueError(
            f"unsupported teacher-data registry format: {document.get('format_version')!r}"
        )
    if not isinstance(document.get("productions"), dict):
        raise ValueError("teacher-data registry productions must be a JSON object")
    if not isinstance(document.get("experiments"), dict):
        raise ValueError("teacher-data registry experiments must be a JSON object")
    return document


def _save_teacher_registry(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _event_belongs_to_records(event, records_dir):
    recording = event.get("recording")
    if not recording:
        return False

    try:
        recording_key = os.path.normcase(
            os.path.normpath(
                os.path.relpath(os.path.abspath(recording), os.getcwd())
            )
        )
        records_key = os.path.normcase(
            os.path.normpath(
                os.path.relpath(os.path.abspath(records_dir), os.getcwd())
            )
        )
    except ValueError:
        recording_key = os.path.normcase(
            os.path.normpath(os.path.abspath(recording))
        )
        records_key = os.path.normcase(
            os.path.normpath(os.path.abspath(records_dir))
        )

    return os.path.dirname(recording_key) == records_key


def _event_has_untrusted_product(event, trusted_ids):
    products = event.get("products") or []
    product_ids = {
        str(product.get("id"))
        for product in products
        if isinstance(product, dict) and product.get("id")
    }
    if not product_ids:
        # Scanner should normally assign species IDs for multi-atom products.
        # If it cannot, keep the event eligible rather than silently dropping it.
        return True
    return any(product_id not in trusted_ids for product_id in product_ids)



def _legacy_teacher_sources(teacher_root):
    """Yield legacy PROD_* teacher sources."""
    teacher_root = Path(teacher_root)
    if not teacher_root.is_dir():
        return

    for root in sorted(
        path for path in teacher_root.glob("PROD_*") if path.is_dir()
    ):
        try:
            production = _load_json(root / "production.json")
            production_id = str(production["id"])
        except (KeyError, ValueError) as problem:
            yield {
                "kind": "invalid",
                "root": root,
                "error": str(problem),
            }
            continue

        yield {
            "kind": "legacy",
            "root": root,
            "records_dir": root / "recordings",
            "experiments_dir": root / "experiments",
            "production_id": production_id,
            "production": production,
            "experiment_ids": None,
        }


def _daily_teacher_sources(teacher_root):
    """Yield each daily-layout invocation as one logical production source."""
    teacher_root = Path(teacher_root)
    if not teacher_root.is_dir():
        return

    for day_root in sorted(path for path in teacher_root.iterdir() if path.is_dir()):
        invocations_dir = day_root / "invocations"
        experiments_dir = day_root / "experiments"
        records_dir = day_root / "recordings"

        if not invocations_dir.is_dir() or not experiments_dir.is_dir():
            continue

        experiment_records = {}
        for experiment_path in sorted(experiments_dir.glob("EXP_*.json")):
            try:
                experiment = _load_json(experiment_path)
            except ValueError:
                continue
            if experiment.get("status") != "complete":
                continue
            experiment_id = experiment.get("experiment_id")
            if experiment_id:
                experiment_records[str(experiment_id)] = (
                    experiment_path, experiment
                )

        for invocation_path in sorted(invocations_dir.glob("INV_*.json")):
            try:
                invocation = _load_json(invocation_path)
                invocation_id = str(invocation["id"])
            except (KeyError, ValueError) as problem:
                yield {
                    "kind": "invalid",
                    "root": day_root,
                    "error": f"{invocation_path}: {problem}",
                }
                continue

            ids = {
                experiment_id
                for experiment_id, (_, experiment) in experiment_records.items()
                if str(
                    experiment.get("invocation_id")
                    or experiment.get("production_id")
                    or ""
                ) == invocation_id
            }

            yield {
                "kind": "daily",
                "root": day_root,
                "records_dir": records_dir,
                "experiments_dir": experiments_dir,
                "production_id": invocation_id,
                "production": invocation,
                "experiment_ids": ids,
                "experiment_records": experiment_records,
            }


def _event_experiment_id(event):
    recording = event.get("recording")
    if not recording:
        return None
    try:
        return Path(recording).stem
    except (OSError, TypeError):
        return None


def _event_belongs_to_source(event, source):
    if not _event_belongs_to_records(event, source["records_dir"]):
        return False

    allowed = source.get("experiment_ids")
    if allowed is None:
        return True

    experiment_id = _event_experiment_id(event)
    return experiment_id in allowed


def _register_teacher_experiments(
    source, registry, invalid,
):
    """Register complete shards for one legacy production or daily invocation."""
    added = 0
    frames = 0
    root = source["root"]
    production_id = source["production_id"]

    if source["kind"] == "daily":
        candidates = [
            source["experiment_records"][experiment_id]
            for experiment_id in sorted(source["experiment_ids"])
            if experiment_id in source["experiment_records"]
        ]
    else:
        directory = source["experiments_dir"]
        candidates = []
        if directory.is_dir():
            for experiment_path in sorted(directory.glob("EXP_*.json")):
                try:
                    experiment = _load_json(experiment_path)
                except ValueError as problem:
                    invalid.append(f"{experiment_path}: {problem}")
                    continue
                candidates.append((experiment_path, experiment))

    for experiment_path, experiment in candidates:
        try:
            if experiment.get("status") != "complete":
                continue
            experiment_id = str(experiment["experiment_id"])
            shard = root / str(experiment["shard"])
            if not shard.is_file():
                raise ValueError(f"missing shard: {shard}")
            teacher_frames = int(experiment.get("teacher_frames", 0))
        except (KeyError, ValueError) as problem:
            invalid.append(f"{experiment_path}: {problem}")
            continue

        if experiment_id in registry["experiments"]:
            continue

        registry["experiments"][experiment_id] = {
            "id": experiment_id,
            "production_id": production_id,
            "experiment": str(experiment_path),
            "shard": str(shard),
            "teacher_frames": teacher_frames,
            "physics_model": experiment.get("physics_model"),
            "physics_model_revision": experiment.get(
                "physics_model_revision"
            ),
            "physics_source_sha256": experiment.get(
                "physics_source_sha256"
            ),
            "chemistrymodel_git_revision": experiment.get(
                "chemistrymodel_git_revision"
            ),
        }
        added += 1
        frames += teacher_frames

    return added, frames


def route_full_cm_events_to_qm(
    store, events, molecule_root, *, qm_root=None,
    production_id=None, teacher_root=None, teacher_layout="live_production",
):
    """Route already-full-CM reaction events straight to QM.

    Optimised-Valence / H-state production has already passed the full-CM
    stage, so untrusted products go directly to QM.
    """
    from .trust import trusted_molecules
    from .state import CandidateState

    trusted_ids = {
        str(row["id"])
        for row in trusted_molecules(molecule_root, qm_root=qm_root)
        if row.get("id")
    }
    eligible = [
        event for event in events
        if _event_has_untrusted_product(event, trusted_ids)
    ]

    source_extra = {"teacher_layout": str(teacher_layout)}
    if production_id is not None:
        source_extra["production_id"] = str(production_id)
    if teacher_root is not None:
        source_extra["teacher_root"] = str(teacher_root)

    imported = store.add_discovery_events(
        eligible,
        event_log_path(molecule_root),
        initial_state=CandidateState.WAITING_QM,
        source_kind="full_cm_teacher_event",
        source_extra=source_extra,
        promote_existing=True,
    )
    return {
        "events_seen": len(events),
        "eligible_events": len(eligible),
        "already_trusted_events": len(events) - len(eligible),
        "queued": imported["added"],
        "duplicates": imported["duplicates"],
        "refreshed": imported.get("refreshed", 0),
        "trusted_ids": trusted_ids,
    }


def ingest_teacher_data(
    store, teacher_root, molecule_root, *,
    qm_root=None, state_file=None, scan=True,
):
    """Register full-CM teacher data and route untrusted products to QM.

    Supports both the legacy PROD_* layout and the current calendar-day layout.
    In the daily layout each invocation is a logical production, while
    recordings/experiments remain shared within that date folder.
    """
    from .trust import trusted_molecules
    from .state import CandidateState

    teacher_root = Path(teacher_root)
    registry_path = _teacher_registry_path(
        state_file if state_file is not None else store.path
    )
    registry = _load_teacher_registry(registry_path)

    productions_found = 0
    productions_added = 0
    experiments_added = 0
    frames_added = 0
    invalid = []
    queued = 0
    duplicates = 0
    refreshed = 0
    events_seen = 0
    already_trusted_events = 0

    trusted_ids = {
        str(row["id"])
        for row in trusted_molecules(molecule_root, qm_root=qm_root)
    }

    sources = list(_legacy_teacher_sources(teacher_root) or ())
    sources.extend(list(_daily_teacher_sources(teacher_root) or ()))

    valid_sources = []
    scanned_roots = set()

    for source in sources:
        if source.get("kind") == "invalid":
            invalid.append(f"{source.get('root')}: {source.get('error')}")
            continue

        productions_found += 1
        root = source["root"]

        # A daily folder can contain many invocations, but needs scanning only
        # once. Legacy PROD_* roots remain one scan each.
        root_key = os.path.normcase(os.path.abspath(root))
        if scan and root_key not in scanned_roots:
            try:
                molecule_scanner.scan_recordings(
                    runs_root=str(root), library_root=str(molecule_root)
                )
            except Exception as problem:
                invalid.append(f"{root}: scanner failed: {problem}")
                continue
            scanned_roots.add(root_key)

        production_id = source["production_id"]
        production = source["production"]

        if production_id not in registry["productions"]:
            registry["productions"][production_id] = {
                "id": production_id,
                "root": str(root),
                "layout": source["kind"],
                "physics": production.get("physics"),
                "physics_model": production.get("physics_model"),
                "physics_model_revision": production.get(
                    "physics_model_revision"
                ),
                "physics_source_sha256": production.get(
                    "physics_source_sha256"
                ),
                "chemistrymodel_git_revision": production.get(
                    "chemistrymodel_git_revision"
                ),
            }
            productions_added += 1

        added, frames = _register_teacher_experiments(
            source, registry, invalid
        )
        experiments_added += added
        frames_added += frames
        valid_sources.append(source)

    # Refresh only once after all required scans.
    all_events, event_errors = read_events(event_log_path(molecule_root))

    for source in valid_sources:
        production_events = [
            event for event in all_events
            if _event_belongs_to_source(event, source)
        ]
        events_seen += len(production_events)

        routed = route_full_cm_events_to_qm(
            store,
            production_events,
            molecule_root,
            qm_root=qm_root,
            production_id=source["production_id"],
            teacher_root=source["root"],
            teacher_layout=source["kind"],
        )
        already_trusted_events += routed["already_trusted_events"]
        queued += routed["queued"]
        duplicates += routed["duplicates"]
        refreshed += routed.get("refreshed", 0)

    if productions_added or experiments_added:
        _save_teacher_registry(registry_path, registry)

    return {
        "teacher_root": str(teacher_root),
        "registry": str(registry_path),
        "productions_found": productions_found,
        "productions_added": productions_added,
        "experiments_added": experiments_added,
        "teacher_frames_added": frames_added,
        "events_seen": events_seen,
        "queued_for_qm": queued,
        "duplicate_candidates": duplicates,
        "refreshed_candidates": refreshed,
        "already_trusted_events": already_trusted_events,
        "invalid": invalid,
        "event_log_errors": event_errors,
    }

