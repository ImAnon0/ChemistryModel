"""Research-only electronic-observable validation framework."""

from .prototype import (
    CandidateComparison,
    ElectronicObservableRecord,
    compare_candidate_rows,
)

__all__ = [
    "CandidateComparison",
    "ElectronicObservableRecord",
    "compare_candidate_rows",
]
