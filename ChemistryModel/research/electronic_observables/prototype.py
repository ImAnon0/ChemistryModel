"""Data contracts and metrics for electronic-observable research.

This is intentionally not a force-field prototype.  It defines the evidence a
future electronic model must return so that it can be compared without being
coupled to production simulation code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ElectronicObservableRecord:
    geometry_id: str
    family: str
    reference_geometry_id: str
    energy_eV: float
    force_eV_per_angstrom: tuple[tuple[float, float, float], ...] = ()
    dipole_debye: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class CandidateComparison:
    geometry_id: str
    family: str
    baseline_residual_eV: float
    candidate_residual_eV: float
    absolute_improvement_eV: float
    classification: str


def relative_energies(records: Iterable[ElectronicObservableRecord]):
    rows = {row.geometry_id: row for row in records}
    relative = {}
    for row in rows.values():
        if row.reference_geometry_id not in rows:
            raise ValueError(
                f"{row.geometry_id}: missing reference {row.reference_geometry_id}"
            )
        relative[row.geometry_id] = (
            row.energy_eV - rows[row.reference_geometry_id].energy_eV
        )
    return relative


def vector_rmse(predicted, reference):
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if predicted.shape != reference.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} != {reference.shape}")
    return float(np.sqrt(np.mean((predicted - reference) ** 2)))


def compare_candidate_rows(qm_records, baseline_records, candidate_records, atol=1e-10):
    """Compare relative-energy residuals geometry by geometry.

    Absolute error reduction is used instead of signed mean reduction so an
    apparent improvement caused only by cancellation is not rewarded.
    """

    qm = {row.geometry_id: row for row in qm_records}
    baseline = {row.geometry_id: row for row in baseline_records}
    candidate = {row.geometry_id: row for row in candidate_records}
    ids = set(qm) & set(baseline) & set(candidate)
    if ids != set(qm):
        raise ValueError("baseline/candidate do not cover every QM geometry")

    qm_rel = relative_energies(qm.values())
    baseline_rel = relative_energies(baseline.values())
    candidate_rel = relative_energies(candidate.values())
    comparisons = []
    for geometry_id in sorted(ids):
        base_residual = baseline_rel[geometry_id] - qm_rel[geometry_id]
        new_residual = candidate_rel[geometry_id] - qm_rel[geometry_id]
        improvement = abs(base_residual) - abs(new_residual)
        if improvement > atol:
            classification = "improved"
        elif improvement < -atol:
            classification = "regressed"
        else:
            classification = "unchanged"
        comparisons.append(CandidateComparison(
            geometry_id=geometry_id,
            family=qm[geometry_id].family,
            baseline_residual_eV=base_residual,
            candidate_residual_eV=new_residual,
            absolute_improvement_eV=improvement,
            classification=classification,
        ))
    return comparisons


def cancellation_warning(comparisons):
    """Flag signed-mean gains not supported by absolute-error gains."""

    comparisons = list(comparisons)
    base = np.asarray([row.baseline_residual_eV for row in comparisons])
    candidate = np.asarray([row.candidate_residual_eV for row in comparisons])
    signed_gain = abs(base.mean()) - abs(candidate.mean())
    mae_gain = np.abs(base).mean() - np.abs(candidate).mean()
    return bool(signed_gain > 0.0 and mae_gain <= 0.0)
