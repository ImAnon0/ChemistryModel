"""Candidate states for the direct full-CM -> QM workflow."""

from enum import Enum


class CandidateState(str, Enum):
    WAITING_QM = "WAITING_QM"
    QM_VALIDATED = "QM_VALIDATED"
    QM_REJECTED = "QM_REJECTED"


TRANSITIONS = {
    CandidateState.WAITING_QM: {
        CandidateState.QM_VALIDATED,
        CandidateState.QM_REJECTED,
    },
    CandidateState.QM_VALIDATED: set(),
    CandidateState.QM_REJECTED: set(),
}


def coerce_state(value):
    """Return a validated CandidateState, migrating removed legacy wait states."""
    if isinstance(value, CandidateState):
        return value

    text = str(value)
    if text in ("WAITING_FULL_CM", "WAITING_CHARACTERISATION"):
        return CandidateState.WAITING_QM

    return CandidateState(text)


def require_transition(previous, following):
    previous = coerce_state(previous)
    following = coerce_state(following)
    if following not in TRANSITIONS[previous]:
        raise ValueError(
            f"invalid candidate transition: {previous.value} -> {following.value}"
        )
    return following
