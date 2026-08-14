"""Reproducible compatibility comparison for isolated QTPIE references."""

import json
from math import erf, sqrt

import numpy as np

from electrostatics import (
    BOHR_PER_ANGSTROM, HARTREE_TO_EV, QTPIE_PARAMETERS,
    dipole_debye, effective_electronegativity, hardness_matrix, solve_charges,
)
from qtpie_historical import (
    historical_matrices, projected_hardness_eigenvalues, solve_historical,
)
from qtpie_validation import geometries


def gaussian_corrected_hardness(elements, positions):
    """Chen Gaussian/Open Babel Coulomb convention, kept experimental."""
    positions = np.asarray(positions, dtype=np.float64)
    alpha = np.array([QTPIE_PARAMETERS[e].gaussian_exponent_bohr2 for e in elements])
    distance = np.linalg.norm(
        (positions[:,None,:]-positions[None,:,:])*BOHR_PER_ANGSTROM, axis=2
    )
    matrix = np.diag([QTPIE_PARAMETERS[e].hardness_ev for e in elements])
    for i in range(len(elements)):
        for j in range(i):
            p = sqrt(alpha[i]*alpha[j]/(alpha[i]+alpha[j]))
            value = HARTREE_TO_EV*erf(p*distance[i,j])/distance[i,j]
            matrix[i,j] = matrix[j,i] = value
    return matrix


def solve_with_matrix(elements, positions, hardness, method="qtpie"):
    if method == "qtpie":
        chi = effective_electronegativity(elements, positions)
    else:
        chi = np.array([QTPIE_PARAMETERS[e].electronegativity_ev for e in elements])
    n = len(elements)
    kkt = np.zeros((n+1,n+1)); kkt[:n,:n] = hardness
    kkt[:n,n] = 1; kkt[n,:n] = 1
    rhs = np.r_[-chi,0.0]; x = np.linalg.solve(kkt,rhs)
    return {
        "charges_e": x[:-1].tolist(),
        "dipole_D": dipole_debye(x[:-1],positions)[1],
        "charge_residual_e": float(np.sum(x[:-1])),
        "linear_residual_inf": float(np.linalg.norm(kkt@x-rhs,ord=np.inf)),
        "condition_number": float(np.linalg.cond(kkt)),
        "effective_chi_eV": chi.tolist(),
        "projected_eigenvalues_eV": projected_hardness_eigenvalues(hardness).tolist(),
    }


def historical_metrics(elements, positions, method):
    solved = solve_historical(elements, positions, method)
    hardness = historical_matrices(elements, positions)[1]
    return {
        "charges_e": solved.charges.tolist(),
        "dipole_D": dipole_debye(solved.charges,positions)[1],
        "charge_residual_e": solved.charge_error_e,
        "linear_residual_inf": solved.residual_inf,
        "condition_number": solved.condition_number,
        "effective_chi_eV": solved.effective_electronegativity_ev.tolist(),
        "projected_eigenvalues_eV": projected_hardness_eigenvalues(hardness).tolist(),
    }


def comparison_report():
    systems = geometries()
    systems["CO"] = (["C","O"], np.array([[0,0,0],[1.1282,0,0]]))
    report = {"spectra": {}, "h2o_nh3": {}, "fragment_scan": []}
    for name,(elements,positions) in systems.items():
        current = hardness_matrix(elements,positions)
        historical = historical_matrices(elements,positions)[1]
        corrected = gaussian_corrected_hardness(elements,positions)
        report["spectra"][name] = {
            "A_current_gaussian": solve_with_matrix(elements,positions,current),
            "B_original_slater": historical_metrics(elements,positions,"qtpie"),
            "C_matching_slater_qeq": historical_metrics(elements,positions,"qeq"),
            "diagnostic_corrected_gaussian": solve_with_matrix(elements,positions,corrected),
        }
    for name in ("H2O","NH3"):
        report["h2o_nh3"][name] = report["spectra"][name]
    for distance in (1.5,2,3,4,5,6,8,10,12,16,20):
        positions = np.array([[0,0,0],[distance,0,0]],float)
        historical = solve_historical(["H","O"],positions,"qtpie")
        qeq = solve_historical(["H","O"],positions,"qeq")
        report["fragment_scan"].append({
            "distance_A": distance,
            "original_qtpie_H_charge_e": float(historical.charges[0]),
            "matching_qeq_H_charge_e": float(qeq.charges[0]),
        })
    return report


if __name__ == "__main__":
    print(json.dumps(comparison_report(),indent=2))
