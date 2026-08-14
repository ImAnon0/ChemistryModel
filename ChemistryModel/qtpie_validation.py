"""Print the standalone neutral QTPIE reference validation report."""

import json
import numpy as np

from electrostatics import QTPIE_PARAMETERS, dipole_debye, solve_charges


def geometries():
    water_angle = np.deg2rad(104.52)
    water = np.array([[0,0,0], [0.9572,0,0], [0.9572*np.cos(water_angle),0.9572*np.sin(water_angle),0]])
    tetra = np.array([[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]], float)/np.sqrt(3)
    methane = np.vstack((np.zeros(3), 1.087*tetra))
    nh = 1.012
    ammonia = np.array([[0,0,0], [0.9377,0,-0.380], [-0.46885,0.8121,-0.380], [-0.46885,-0.8121,-0.380]])
    formaldehyde = np.array([[0,0,0], [1.208,0,0], [-0.589,0.935,0], [-0.589,-0.935,0]])
    return {
        "H2": (["H","H"], np.array([[-0.37072,0,0],[0.37072,0,0]])),
        "CH4": (["C"]+["H"]*4, methane),
        "H2O": (["O","H","H"], water),
        "NH3": (["N","H","H","H"], ammonia),
        "CH2O": (["C","O","H","H"], formaldehyde),
        "CO2": (["O","C","O"], np.array([[-1.16,0,0],[0,0,0],[1.16,0,0]])),
    }


def report():
    output = {"molecules": {}, "fragment_scan": [], "parameters": {}}
    for symbol, p in QTPIE_PARAMETERS.items():
        output["parameters"][symbol] = {
            "chi_eV": p.electronegativity_ev, "eta_eV": p.hardness_ev,
            "alpha_bohr^-2": p.gaussian_exponent_bohr2,
        }
    for name, (elements, positions) in geometries().items():
        result = solve_charges(elements, positions)
        output["molecules"][name] = {
            "charges_e": result.charges.tolist(),
            "dipole_D": dipole_debye(result.charges, positions)[1],
            "condition": result.condition_number,
            "residual_inf": result.residual_inf,
        }
    for distance in (1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20):
        positions = np.array([[0,0,0], [distance,0,0]], dtype=float)
        row = {"distance_A": distance}
        for method in ("qtpie", "qeq"):
            solved = solve_charges(["H", "O"], positions, method)
            row[f"{method}_fragment_charge_e"] = float(solved.charges[0])
            row[f"{method}_condition"] = solved.condition_number
            row[f"{method}_residual_inf"] = solved.residual_inf
        output["fragment_scan"].append(row)
    return output


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
