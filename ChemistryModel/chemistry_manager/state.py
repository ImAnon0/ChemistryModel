"""Candidate states and the deliberately small v1 transition graph."""

from enum import Enum


class CandidateState(str, Enum):
    WAITING_CHARACTERISATION = "WAITING_CHARACTERISATION"
    WAITING_QM = "WAITING_QM"
    QM_VALIDATED = "QM_VALIDATED"
    QM_REJECTED = "QM_REJECTED"


TRANSITIONS = {
    CandidateState.WAITING_CHARACTERISATION: {CandidateState.WAITING_QM},
    CandidateState.WAITING_QM: {
        CandidateState.QM_VALIDATED,
        CandidateState.QM_REJECTED,
    },
    CandidateState.QM_VALIDATED: set(),
    CandidateState.QM_REJECTED: set(),
}


def coerce_state(value):
    """Return a validated CandidateState without accepting unknown strings."""

    if isinstance(value, CandidateState):
        return value
    if str(value) == "WAITING_FULL_CM":
        return CandidateState.WAITING_CHARACTERISATION
    return CandidateState(str(value))


def require_transition(previous, following):
    previous = coerce_state(previous)
    following = coerce_state(following)
    if following not in TRANSITIONS[previous]:
        raise ValueError(
            f"invalid candidate transition: {previous.value} -> {following.value}"
        )
    return following
