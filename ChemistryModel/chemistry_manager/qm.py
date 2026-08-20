"""QM queue execution for Chemistry Manager."""

from pathlib import Path

import molecule_library
import qm_structure_validator

from .state import CandidateState
from .trust import MoleculeTrust, trust_level


def _product_ids(candidate):
    found = []
    for product in candidate.get("products") or []:
        if not isinstance(product, dict):
            continue
        molecule_id = product.get("id")
        if molecule_id and molecule_id not in found:
            found.append(str(molecule_id))
    return found


def _successful(record):
    if record.get("status") != "complete":
        return False
    comparison = record.get("comparison") or {}
    return bool(
        comparison.get("connectivity_preserved") is True
        and comparison.get("fragmented") is not True
        and comparison.get("rearranged") is not True
    )


def _chemically_rejected(record):
    if record.get("status") != "complete":
        return False
    comparison = record.get("comparison") or {}
    return not _successful(record) and (
        comparison.get("connectivity_changed") is True
        or comparison.get("fragmented") is True
        or comparison.get("rearranged") is True
    )


def _electronic_state(record):
    selection = record.get("electronic_state_selection") or {}
    multiplicity = selection.get("selected_multiplicity", record.get("multiplicity"))
    return {
        "charge": int(record.get("charge", 0)),
        "multiplicity": int(multiplicity),
        "source": selection.get("source", "qm_validation"),
        "validation_id": record.get("id"),
    }


def process_qm_queue(
    store, *, molecule_root="molecules", qm_root=None,
    method=qm_structure_validator.DEFAULT_METHOD,
    basis=qm_structure_validator.DEFAULT_BASIS,
    threads=8, memory="4 GB", limit=None,
    worker=qm_structure_validator.run_validation_in_worker,
    progress=None, molecule_progress=None,
):
    waiting = store.candidates(CandidateState.WAITING_QM)
    if limit is not None:
        waiting = waiting[:max(0, int(limit))]

    validation_root = (
        Path(qm_root)
        if qm_root is not None
        else Path(molecule_root) / "qm_validations"
    )

    summary = {
        "candidates_seen": len(waiting),
        "validated": 0,
        "rejected": 0,
        "still_waiting": 0,
        "molecules_validated": 0,
        "molecules_rejected": 0,
        "molecules_reused": 0,
        "errors": [],
    }
    cache = {}

    for number, candidate in enumerate(waiting, start=1):
        candidate_id = candidate["id"]
        products = _product_ids(candidate)
        if progress:
            progress(number, len(waiting), candidate, products)

        if not products:
            summary["still_waiting"] += 1
            summary["errors"].append(
                f"{candidate_id}: no product molecule IDs were recorded"
            )
            store.record_qm_result(candidate_id, {
                "status": "blocked",
                "reason": "no product molecule IDs were recorded",
            })
            continue

        molecule_results = []
        candidate_rejected = False
        candidate_blocked = False

        for molecule_id in products:
            if molecule_id in cache:
                result = cache[molecule_id]
                summary["molecules_reused"] += 1
            else:
                try:
                    molecule = molecule_library.load_molecule(
                        molecule_id, root=molecule_root
                    )
                except Exception as problem:
                    result = {
                        "molecule_id": molecule_id,
                        "outcome": "blocked",
                        "error": f"{type(problem).__name__}: {problem}",
                    }
                    cache[molecule_id] = result
                    molecule_results.append(result)
                    candidate_blocked = True
                    continue

                level = trust_level(molecule, qm_root=validation_root)
                if level == MoleculeTrust.QM_VALIDATED:
                    validations = qm_structure_validator.list_validations(
                        molecule_id, root=validation_root
                    )
                    successful = next(
                        (record for record in validations if _successful(record)),
                        None,
                    )
                    result = {
                        "molecule_id": molecule_id,
                        "outcome": "validated",
                        "reused": True,
                        "validation_id": (
                            successful.get("id") if successful else None
                        ),
                    }
                    summary["molecules_reused"] += 1
                elif level == MoleculeTrust.REJECTED:
                    result = {
                        "molecule_id": molecule_id,
                        "outcome": "rejected",
                        "reused": True,
                    }
                    summary["molecules_reused"] += 1
                else:
                    try:
                        record = worker(
                            molecule_id,
                            method=method,
                            basis=basis,
                            root=validation_root,
                            molecule_root=molecule_root,
                            threads=threads,
                            memory=memory,
                        )
                    except Exception as problem:
                        result = {
                            "molecule_id": molecule_id,
                            "outcome": "blocked",
                            "error": f"{type(problem).__name__}: {problem}",
                        }
                    else:
                        if _successful(record):
                            state = _electronic_state(record)
                            molecule_library.update_validation_metadata(
                                molecule_id,
                                trust_status=MoleculeTrust.QM_VALIDATED.value,
                                electronic_state=state,
                                root=molecule_root,
                            )
                            result = {
                                "molecule_id": molecule_id,
                                "outcome": "validated",
                                "validation_id": record.get("id"),
                                "electronic_state": state,
                            }
                            summary["molecules_validated"] += 1
                        elif _chemically_rejected(record):
                            state = _electronic_state(record)
                            molecule_library.update_validation_metadata(
                                molecule_id,
                                trust_status=MoleculeTrust.REJECTED.value,
                                electronic_state=state,
                                root=molecule_root,
                            )
                            result = {
                                "molecule_id": molecule_id,
                                "outcome": "rejected",
                                "validation_id": record.get("id"),
                                "electronic_state": state,
                            }
                            summary["molecules_rejected"] += 1
                        else:
                            result = {
                                "molecule_id": molecule_id,
                                "outcome": "blocked",
                                "validation_id": record.get("id"),
                                "error": record.get("error", "QM validation failed"),
                            }
                cache[molecule_id] = result

            molecule_results.append(result)
            if molecule_progress:
                molecule_progress(number, len(waiting), candidate, dict(result))
            if result["outcome"] == "rejected":
                candidate_rejected = True
            elif result["outcome"] == "blocked":
                candidate_blocked = True

        payload = {
            "status": (
                "rejected" if candidate_rejected
                else "blocked" if candidate_blocked
                else "validated"
            ),
            "method": method,
            "basis": basis,
            "validation_root": str(validation_root),
            "molecules": molecule_results,
        }

        if candidate_rejected:
            store.record_qm_result(
                candidate_id, payload,
                final_state=CandidateState.QM_REJECTED,
            )
            summary["rejected"] += 1
        elif candidate_blocked:
            store.record_qm_result(candidate_id, payload)
            summary["still_waiting"] += 1
            summary["errors"].extend(
                f"{candidate_id}/{row['molecule_id']}: {row.get('error', 'blocked')}"
                for row in molecule_results
                if row["outcome"] == "blocked"
            )
        else:
            store.record_qm_result(
                candidate_id, payload,
                final_state=CandidateState.QM_VALIDATED,
            )
            summary["validated"] += 1

    return summary
