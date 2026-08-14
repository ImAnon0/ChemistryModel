"""Historical Chen/Martinez QTPIE(-H) reference in atom space.

This reproduces the 2007 model: QEq electronegativities, full diagonal
hardnesses and normalized ns Slater orbitals; Rosen's analytic overlap and
two-centre Coulomb integrals; pair attenuation k_ij S_ij with k_ij = 1.
The hydrogen charge-dependent QEq correction was explicitly omitted in the
published QTPIE calculations, hence the conventional QTPIE(-H) label.
"""

from dataclasses import dataclass
from math import exp, factorial, sqrt

import numpy as np

from electrostatics import (
    BOHR_PER_ANGSTROM,
    HARTREE_TO_EV,
    QTPIE_PARAMETERS,
    ChargeSolution,
    _inputs,
)


@dataclass(frozen=True)
class SlaterParameter:
    principal_n: int
    zeta_bohr_inverse: float


# Rappe/Goddard QEq ns orbitals as tabulated with the original QTPIE work.
SLATER_PARAMETERS = {
    "H": SlaterParameter(1, 1.0698),
    "C": SlaterParameter(2, 0.8563),
    "N": SlaterParameter(2, 0.9089),
    "O": SlaterParameter(2, 0.9745),
}


def _rosen_a(n, value):
    terms = sum(value**nu / factorial(nu) for nu in range(n + 1))
    return factorial(n) * exp(-value) * terms / value ** (n + 1)


def _rosen_b(n, value):
    if abs(value) < 1e-10:
        return (1.0 - (-1.0) ** (n + 1)) / (n + 1.0)
    total = 0.0
    for nu in range(n + 1):
        total += (exp(value) * (-value) ** nu - exp(-value) * value**nu) / factorial(nu)
    return factorial(n) * total / value ** (n + 1)


def _rosen_d(m, n, p):
    total = 0.0
    for k in range(max(p - m, 0), min(n, p) + 1):
        term = (
            factorial(m) * factorial(n)
            / (factorial(p-k) * factorial(m-p+k) * factorial(n-k) * factorial(k))
        )
        total += (-1.0) ** k * term
    return total


def sto_overlap(zeta_a, zeta_b, n_a, n_b, distance_bohr):
    """Rosen normalized ns-STO overlap integral."""
    r = float(distance_bohr)
    if r <= 0.0:
        return 1.0 if zeta_a == zeta_b and n_a == n_b else np.nan
    if zeta_a == zeta_b:
        factor = (zeta_a*r) ** (n_a+n_b+1) / sqrt(factorial(2*n_a)*factorial(2*n_b))
        total = sum(
            _rosen_d(n_a, n_b, 2*q) / (2*q+1) * _rosen_a(n_a+n_b-2*q, zeta_a*r)
            for q in range((n_a+n_b)//2 + 1)
        )
    else:
        factor = 0.5 * (zeta_a*r) ** (n_a+0.5) * (zeta_b*r) ** (n_b+0.5) / sqrt(
            factorial(2*n_a)*factorial(2*n_b)
        )
        total = sum(
            _rosen_d(n_a, n_b, q)
            * _rosen_b(q, 0.5*r*(zeta_a-zeta_b))
            * _rosen_a(n_a+n_b-q, 0.5*r*(zeta_a+zeta_b))
            for q in range(n_a+n_b+1)
        )
    return factor * total


def sto_coulomb(zeta_a, zeta_b, n_a, n_b, distance_bohr):
    """Rosen K2 two-centre Coulomb integral over normalized ns STOs."""
    r = float(distance_bohr)
    if r <= 0.0:
        raise ValueError("two-centre STO Coulomb integral requires R > 0")
    a, b, m, n = zeta_a, zeta_b, n_a, n_b
    x = 2.0*a*r
    one_electron = 1.0/r + x**(2*m)/(factorial(2*m)*r) * (
        (x-2*m)*_rosen_a(2*m-1, x) - exp(-x)
    )
    two_electron = 0.0
    if a == b:
        factor1 = -a*(a*r)**(2*m)/(n*factorial(2*m))
        for nu in range(2*n):
            factor2 = (2*n-nu)/factorial(nu)*(a*r)**nu
            k2 = sum(
                _rosen_d(2*m-1, nu, 2*p)/(2*p+1)
                * _rosen_a(2*m+nu-1-2*p, x)
                for p in range(m+(nu-1)//2+1)
            )
            two_electron += k2*factor2
        two_electron *= factor1
    else:
        factor1 = -a*(a*r)**(2*m)/(2*n*factorial(2*m))
        for nu in range(2*n):
            k2 = sum(
                _rosen_d(2*m-1, nu, p)
                * _rosen_b(p, r*(a-b))
                * _rosen_a(2*m+nu-1-p, r*(a+b))
                for p in range(2*m+nu)
            )
            two_electron += k2*(2*n-nu)/factorial(nu)*(b*r)**nu
        two_electron *= factor1
    return one_electron + two_electron


def historical_matrices(elements, positions_angstrom):
    elements, positions = _inputs(elements, positions_angstrom, 0.0)
    count = len(elements)
    overlap = np.eye(count, dtype=np.float64)
    hardness = np.diag([QTPIE_PARAMETERS[e].hardness_ev for e in elements])
    for i in range(count):
        pa = SLATER_PARAMETERS[elements[i]]
        for j in range(i):
            pb = SLATER_PARAMETERS[elements[j]]
            r = np.linalg.norm(positions[i]-positions[j])*BOHR_PER_ANGSTROM
            overlap[i,j] = overlap[j,i] = sto_overlap(
                pa.zeta_bohr_inverse, pb.zeta_bohr_inverse,
                pa.principal_n, pb.principal_n, r,
            )
            hardness[i,j] = hardness[j,i] = HARTREE_TO_EV*sto_coulomb(
                pa.zeta_bohr_inverse, pb.zeta_bohr_inverse,
                pa.principal_n, pb.principal_n, r,
            )
    return overlap, hardness


def solve_historical(elements, positions_angstrom, method="qtpie"):
    """Solve historical QTPIE(-H), or matching Slater QEq(-H), neutrally."""
    elements, positions = _inputs(elements, positions_angstrom, 0.0)
    overlap, hardness = historical_matrices(elements, positions)
    bare = np.array([QTPIE_PARAMETERS[e].electronegativity_ev for e in elements])
    method = method.lower()
    if method == "qtpie":
        # Exact atom-space mapping of original f_ij=S_ij, k_ij=1 bond model.
        chi = np.sum(overlap*(bare[:,None]-bare[None,:]), axis=1)/len(elements)
    elif method == "qeq":
        chi = bare
    else:
        raise ValueError("method must be 'qtpie' or 'qeq'")
    n = len(elements)
    kkt = np.zeros((n+1,n+1)); kkt[:n,:n] = hardness
    kkt[:n,n] = 1.0; kkt[n,:n] = 1.0
    rhs = np.r_[-chi, 0.0]
    solved = np.linalg.solve(kkt, rhs)
    residual = float(np.linalg.norm(kkt@solved-rhs, ord=np.inf))
    q = solved[:-1]
    diagonal = np.diag(hardness)
    self_energy = float(chi@q + 0.5*np.sum(diagonal*q*q))
    pair_energy = float(0.5*q@(hardness-np.diag(diagonal))@q)
    return ChargeSolution(
        method=f"historical_{method}", charges=q,
        effective_electronegativity_ev=chi,
        chemical_potential_ev=float(solved[-1]),
        energy_ev=self_energy+pair_energy, self_energy_ev=self_energy,
        pair_energy_ev=pair_energy, charge_error_e=float(np.sum(q)),
        residual_inf=residual, condition_number=float(np.linalg.cond(kkt)),
    )


def projected_hardness_eigenvalues(hardness):
    """Eigenvalues restricted to sum(q)=0, the physically allowed subspace."""
    n = hardness.shape[0]
    if n == 1:
        return np.empty(0)
    basis = np.vstack((np.eye(n-1), -np.ones(n-1)))
    basis, _ = np.linalg.qr(basis)
    return np.linalg.eigvalsh(basis.T@hardness@basis)
