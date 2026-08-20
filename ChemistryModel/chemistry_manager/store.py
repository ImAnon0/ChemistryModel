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

    def migrate_legacy_wait_states(self):
        """Persist removed legacy wait states as WAITING_QM.

        Candidate identity, products, source and provenance are left untouched.
        This is safe to call repeatedly.
        """
        if not self.path.exists():
            return {"migrated": 0, "candidate_ids": []}

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as problem:
            raise ValueError(
                f"manager state is not valid JSON: {self.path}"
            ) from problem

        candidates = raw.get("candidates", {})
        migrated = []
        for candidate_id, candidate in candidates.items():
            if not isinstance(candidate, dict):
                continue
            state = str(candidate.get("state"))
            if state in ("WAITING_FULL_CM", "WAITING_CHARACTERISATION"):
                candidate["state"] = CandidateState.WAITING_QM.value
                migrated.append(str(candidate_id))

        if migrated:
            self.save(raw)

        return {
            "migrated": len(migrated),
            "candidate_ids": sorted(migrated),
        }

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

    def add_discovery_event(
        self, event, event_log, *,
        initial_state=CandidateState.WAITING_QM,
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
        initial_state=CandidateState.WAITING_QM,
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
                    candidate["state"] = coerce_state(candidate["state"]).value
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

    def product_ids(self, state=None):
        """Unique molecule IDs referenced by candidates, optionally in one state."""
        found = set()
        for candidate in self.candidates(state):
            for product in candidate.get("products") or []:
                if isinstance(product, dict) and product.get("id"):
                    found.add(str(product["id"]))
        return sorted(found)

    def candidates_for_product(self, molecule_id, state=None):
        """Candidates that reference one product molecule."""
        molecule_id = str(molecule_id)
        rows = []
        for candidate in self.candidates(state):
            product_ids = {
                str(product.get("id"))
                for product in candidate.get("products") or []
                if isinstance(product, dict) and product.get("id")
            }
            if molecule_id in product_ids:
                rows.append(candidate)
        return rows

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
