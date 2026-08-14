"""Standalone float64 QTPIE/QEq reference implementation.

This module is deliberately disconnected from :mod:`reactive`.  It implements
the neutral atom-space convention of Chen, Hundertmark and Martinez, QSCP-XIII
(2008), equations 22 and 28--30, using their primitive-Gaussian Coulomb
integral (equation 33) and Table 1 exponents.  It is a validation reference,
not an MD force implementation.
"""

from dataclasses import dataclass
from math import erf, pi, sqrt

import numpy as np


BOHR_PER_ANGSTROM = 1.8897261254578281
HARTREE_TO_EV = 27.211386245988
E_ANGSTROM_TO_DEBYE = 4.803204712570263


@dataclass(frozen=True)
class ElementParameter:
    electronegativity_ev: float
    hardness_ev: float
    gaussian_exponent_bohr2: float


# chi and eta are the original QEq values reused by QTPIE.  eta is the full
# diagonal hardness J_ii.  alpha values are Table 1 of Chen et al. (2008).
QTPIE_PARAMETERS = {
    "H": ElementParameter(4.528, 13.890, 0.5434),
    "C": ElementParameter(5.343, 10.126, 0.2069),
    "N": ElementParameter(6.899, 11.760, 0.2214),
    "O": ElementParameter(8.741, 13.364, 0.2240),
}


@dataclass(frozen=True)
class ChargeSolution:
    method: str
    charges: np.ndarray
    effective_electronegativity_ev: np.ndarray
    chemical_potential_ev: float
    energy_ev: float
    self_energy_ev: float
    pair_energy_ev: float
    charge_error_e: float
    residual_inf: float
    condition_number: float


def _inputs(elements, positions_angstrom, total_charge):
    elements = tuple(elements)
    positions = np.asarray(positions_angstrom, dtype=np.float64)
    if positions.shape != (len(elements), 3) or not np.all(np.isfinite(positions)):
        raise ValueError("positions must be a finite float64 (N, 3) array")
    if not elements:
        raise ValueError("at least one atom is required")
    missing = sorted(set(elements) - set(QTPIE_PARAMETERS))
    if missing:
        raise KeyError(f"no convention-locked parameters for: {missing}")
    if float(total_charge) != 0.0:
        raise ValueError(
            "this reference is intentionally scoped to neutral systems; "
            "a global KKT constraint does not establish charged-fragment correctness"
        )
    return elements, positions


def gaussian_overlap_matrix(elements, positions_angstrom):
    """Normalized primitive s-Gaussian orbital overlaps S_ij."""
    elements, positions = _inputs(elements, positions_angstrom, 0.0)
    alpha = np.array(
        [QTPIE_PARAMETERS[e].gaussian_exponent_bohr2 for e in elements]
    )
    delta = (positions[:, None, :] - positions[None, :, :]) * BOHR_PER_ANGSTROM
    r2 = np.einsum("ijk,ijk->ij", delta, delta)
    ai, aj = alpha[:, None], alpha[None, :]
    prefactor = (2.0 * np.sqrt(ai * aj) / (ai + aj)) ** 1.5
    overlap = prefactor * np.exp(-(ai * aj / (ai + aj)) * r2)
    np.fill_diagonal(overlap, 1.0)
    return overlap


def hardness_matrix(elements, positions_angstrom):
    """Gaussian-screened Coulomb matrix J in eV, with J_ii = eta_i."""
    elements, positions = _inputs(elements, positions_angstrom, 0.0)
    alpha = np.array(
        [QTPIE_PARAMETERS[e].gaussian_exponent_bohr2 for e in elements]
    )
    delta = (positions[:, None, :] - positions[None, :, :]) * BOHR_PER_ANGSTROM
    r = np.linalg.norm(delta, axis=2)
    ai, aj = alpha[:, None], alpha[None, :]
    beta = 2.0 * ai * aj / (ai + aj)
    matrix = np.empty_like(r)
    for i in range(len(elements)):
        for j in range(len(elements)):
            if i == j:
                matrix[i, j] = QTPIE_PARAMETERS[elements[i]].hardness_ev
            elif r[i, j] == 0.0:
                matrix[i, j] = HARTREE_TO_EV * 2.0 * sqrt(beta[i, j] / pi)
            else:
                matrix[i, j] = HARTREE_TO_EV * erf(sqrt(beta[i, j]) * r[i, j]) / r[i, j]
    return matrix


def effective_electronegativity(elements, positions_angstrom):
    """Chen et al. equation 30 (including S_ii in each row normalization)."""
    elements, positions = _inputs(elements, positions_angstrom, 0.0)
    chi = np.array([QTPIE_PARAMETERS[e].electronegativity_ev for e in elements])
    overlap = gaussian_overlap_matrix(elements, positions)
    return np.sum(overlap * (chi[:, None] - chi[None, :]), axis=1) / np.sum(
        overlap, axis=1
    )


def solve_charges(elements, positions_angstrom, method="qtpie", total_charge=0.0):
    """Solve the neutral QTPIE or comparator-QEq dense KKT system in float64."""
    elements, positions = _inputs(elements, positions_angstrom, total_charge)
    method = method.lower()
    if method == "qtpie":
        chi = effective_electronegativity(elements, positions)
    elif method == "qeq":
        chi = np.array(
            [QTPIE_PARAMETERS[e].electronegativity_ev for e in elements],
            dtype=np.float64,
        )
    else:
        raise ValueError("method must be 'qtpie' or 'qeq'")

    hardness = hardness_matrix(elements, positions)
    count = len(elements)
    kkt = np.zeros((count + 1, count + 1), dtype=np.float64)
    kkt[:count, :count] = hardness
    kkt[:count, count] = 1.0
    kkt[count, :count] = 1.0
    rhs = np.concatenate((-chi, [0.0]))
    condition = float(np.linalg.cond(kkt))
    if not np.isfinite(condition):
        raise np.linalg.LinAlgError("non-finite KKT condition number")
    solved = np.linalg.solve(kkt, rhs)
    residual = float(np.linalg.norm(kkt @ solved - rhs, ord=np.inf))
    charges = solved[:-1]
    charge_error = float(np.sum(charges))
    if not np.all(np.isfinite(charges)) or residual > 1e-9 or abs(charge_error) > 1e-10:
        raise np.linalg.LinAlgError(
            f"charge solve failed: residual={residual:.3e}, charge error={charge_error:.3e}"
        )
    diagonal = np.diag(hardness)
    self_energy = float(chi @ charges + 0.5 * np.sum(diagonal * charges**2))
    pair_energy = float(0.5 * charges @ (hardness - np.diag(diagonal)) @ charges)
    return ChargeSolution(
        method=method,
        charges=charges,
        effective_electronegativity_ev=chi,
        chemical_potential_ev=float(solved[-1]),
        energy_ev=self_energy + pair_energy,
        self_energy_ev=self_energy,
        pair_energy_ev=pair_energy,
        charge_error_e=charge_error,
        residual_inf=residual,
        condition_number=condition,
    )


def dipole_debye(charges, positions_angstrom):
    charges = np.asarray(charges, dtype=np.float64)
    positions = np.asarray(positions_angstrom, dtype=np.float64)
    vector = np.sum(charges[:, None] * positions, axis=0) * E_ANGSTROM_TO_DEBYE
    return vector, float(np.linalg.norm(vector))
