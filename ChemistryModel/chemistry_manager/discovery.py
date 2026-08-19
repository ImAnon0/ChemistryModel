"""Handoff from the existing molecule scanner into the manager queue."""

import json
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
    imported = store.add_discovery_events(events, path)

    return {
        "scan": scan_summary,
        "events_read": len(events),
        "queued": imported["added"],
        "already_known": imported["duplicates"],
        "errors": errors,
        "event_log": str(path),
    }
