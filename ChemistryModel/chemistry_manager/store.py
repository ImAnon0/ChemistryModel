"""Inspectable, atomic JSON persistence for Chemistry Manager."""

import copy
import json
import os
from pathlib import Path

from .state import CandidateState, coerce_state, require_transition


FORMAT_VERSION = 1


def _empty_document():
    return {
        "format_version": FORMAT_VERSION,
        "candidates": {},
    }


class ManagerStore:
    """A thin state overlay; simulation and scanner files stay authoritative."""

    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        if not self.path.exists():
            return _empty_document()

        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as problem:
            raise ValueError(
                f"manager state is not valid JSON: {self.path}"
            ) from problem

        if not isinstance(document, dict):
            raise ValueError("manager state root must be a JSON object")
        if document.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                "unsupported manager state format: "
                f"{document.get('format_version')!r}"
            )
        candidates = document.get("candidates")
        if not isinstance(candidates, dict):
            raise ValueError("manager state candidates must be a JSON object")

        for candidate_id, candidate in candidates.items():
            if not isinstance(candidate, dict):
                raise ValueError(f"candidate {candidate_id!r} must be an object")
            if candidate.get("id") != candidate_id:
                raise ValueError(f"candidate {candidate_id!r} has mismatched identity")
            candidate["state"] = coerce_state(candidate.get("state")).value

        return document

    def save(self, document):
        document = copy.deepcopy(document)
        document["format_version"] = FORMAT_VERSION
        document.setdefault("candidates", {})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def candidates(self, state=None):
        rows = list(self.load()["candidates"].values())
        if state is not None:
            wanted = coerce_state(state).value
            rows = [row for row in rows if row["state"] == wanted]
        return sorted(rows, key=lambda row: row["id"])

    def counts(self):
        counts = {state: 0 for state in CandidateState}
        for row in self.load()["candidates"].values():
            counts[coerce_state(row["state"])] += 1
        return counts

    def legacy_characterisation_candidates(self):
        """Return old generic formation events still waiting for characterisation."""
        rows = []
        for candidate in self.load()["candidates"].values():
            if (
                coerce_state(candidate["state"])
                == CandidateState.WAITING_CHARACTERISATION
                and (candidate.get("source") or {}).get("kind") == "formation_event"
            ):
                rows.append(copy.deepcopy(candidate))
        return sorted(rows, key=lambda row: row["id"])

    def remove_legacy_characterisation_candidates(self):
        """Delete only legacy generic formation events waiting for characterisation."""
        document = self.load()
        removed = []
        for candidate_id, candidate in list(document["candidates"].items()):
            if (
                coerce_state(candidate["state"])
                == CandidateState.WAITING_CHARACTERISATION
                and (candidate.get("source") or {}).get("kind") == "formation_event"
            ):
                removed.append(candidate_id)
                del document["candidates"][candidate_id]

        if removed:
            self.save(document)

        return {
            "removed": len(removed),
            "candidate_ids": sorted(removed),
        }

    def add_discovery_event(
        self, event, event_log, *,
        initial_state=CandidateState.WAITING_CHARACTERISATION,
        source_kind="formation_event",
        source_extra=None,
        promote_existing=False,
    ):
        result = self.add_discovery_events(
            [event], event_log,
            initial_state=initial_state,
            source_kind=source_kind,
            source_extra=source_extra,
            promote_existing=promote_existing,
        )
        return result["added"] == 1

    def add_discovery_events(
        self, events, event_log, *,
        initial_state=CandidateState.WAITING_CHARACTERISATION,
        source_kind="formation_event",
        source_extra=None,
        promote_existing=False,
    ):
        document = self.load()
        initial_state = coerce_state(initial_state)
        source_extra = dict(source_extra or {})
        added = 0
        duplicates = 0
        for event in events:
            event_id = str(event.get("event_id", "")).strip()
            if not event_id:
                raise ValueError("discovery event has no event_id")

            candidate_id = f"EVENT_{event_id}"
            if candidate_id in document["candidates"]:
                duplicates += 1
                if promote_existing:
                    candidate = document["candidates"][candidate_id]
                    current = coerce_state(candidate["state"])
                    if (
                        current == CandidateState.WAITING_CHARACTERISATION
                        and initial_state == CandidateState.WAITING_QM
                    ):
                        candidate["state"] = require_transition(
                            current, CandidateState.WAITING_QM
                        ).value
                        source = candidate.setdefault("source", {})
                        source["kind"] = str(source_kind)
                        source["event_id"] = event_id
                        source["event_log"] = os.path.normpath(str(event_log))
                        source.update(copy.deepcopy(source_extra))
                        added += 1
                continue

            source = {
                "kind": str(source_kind),
                "event_id": event_id,
                "event_log": os.path.normpath(str(event_log)),
            }
            source.update(copy.deepcopy(source_extra))

            document["candidates"][candidate_id] = {
                "id": candidate_id,
                "state": initial_state.value,
                "source": source,
                "provenance": {
                    key: event.get(key)
                    for key in (
                        "recording", "batch", "seed", "mixture", "time_fs",
                        "previous_frame_time_fs", "temperature_K", "box_A",
                    )
                },
                "reactants": copy.deepcopy(event.get("reactants", [])),
                "products": copy.deepcopy(event.get("products", [])),
                "formed_bonds": copy.deepcopy(event.get("formed_bonds", [])),
                "broken_bonds": copy.deepcopy(event.get("broken_bonds", [])),
            }
            added += 1

        if added:
            self.save(document)
        return {"added": added, "duplicates": duplicates}

    def record_qm_result(self, candidate_id, payload, final_state=None):
        """Persist QM provenance and optionally make one valid state transition."""
        document = self.load()
        try:
            candidate = document["candidates"][str(candidate_id)]
        except KeyError as problem:
            raise KeyError(f"unknown candidate: {candidate_id}") from problem

        candidate["qm"] = copy.deepcopy(payload)
        if final_state is not None:
            following = require_transition(candidate["state"], final_state)
            candidate["state"] = following.value

        self.save(document)
        return copy.deepcopy(candidate)

    def transition(self, candidate_id, state):
        document = self.load()
        try:
            candidate = document["candidates"][str(candidate_id)]
        except KeyError as problem:
            raise KeyError(f"unknown candidate: {candidate_id}") from problem

        following = require_transition(candidate["state"], state)
        candidate["state"] = following.value
        self.save(document)
        return copy.deepcopy(candidate)
