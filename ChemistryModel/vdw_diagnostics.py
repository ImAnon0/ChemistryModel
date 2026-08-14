"""Generate the standalone vdW milestone's reproducible numeric report."""

from __future__ import annotations

import json

import numpy as np

import reactive as R
import vdw_reference as V


REPRESENTATIVE_PAIRS = (
    ("H", "H"), ("C", "H"), ("O", "H"), ("C", "C"),
    ("C", "O"), ("N", "N"), ("O", "O"),
)


def central_force(function, distance, step=1e-6):
    return -(function(distance + step) - function(distance - step)) / (2 * step)


def extrema_indices(values):
    minima = np.where(
        (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])
    )[0] + 1
    maxima = np.where(
        (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
    )[0] + 1
    return minima, maxima


def raw_pair_report(model):
    report = {}
    for first, second in V.unique_pairs():
        minimum, depth = V.pair_parameters(first, second)
        entry = {
            "minimum_A": minimum,
            "depth_eV": depth,
            "depth_kcal_mol": depth / V.KCAL_PER_MOL_TO_EV,
            "energy_at_outer_match_eV": float(V.RAW_MODELS[model][0](
                V.OUTER_MATCH_RATIO * minimum, first, second
            )),
            "energy_near_zero_eV": float(V.RAW_MODELS[model][0](
                1e-9, first, second
            )),
        }
        if model == "airebo_m":
            entry["alpha_A^-1"] = V.airebo_m_parameters(first, second)[2]
        elif model == "reaxff":
            radius, _, alpha, gamma, power = V.reaxff_shielded_parameters(
                first, second
            )
            entry.update({"r_vdw_A": radius, "alpha": alpha,
                          "gamma_w_A^-1": gamma, "p_vdw1": power})
        report[f"{first}-{second}"] = entry
    return report


def continuity_report(model):
    worst_absolute = (0.0, None)
    worst_relative = (0.0, None)
    boundary_jumps = {"energy_eV": 0.0, "force_eV_per_A": 0.0}
    for first, second in V.unique_pairs():
        inner, outer = V.reactive_interval(first, second)
        distances = np.concatenate((
            np.linspace(inner + 1e-4, outer - 1e-4, 300),
            np.linspace(2.5, 6.9, 100),
            np.linspace(7.001, 8.499, 300),
        ))
        for distance in distances:
            analytic = float(V.suppressed_vdw_force(
                distance, first, second, model=model
            ))
            numeric = float(central_force(
                lambda r: V.suppressed_vdw_energy(
                    r, first, second, model=model
                ),
                distance,
            ))
            absolute = abs(analytic - numeric)
            relative = absolute / max(abs(analytic), abs(numeric), 1e-8)
            if absolute > worst_absolute[0]:
                worst_absolute = (absolute, (first, second, distance, analytic, numeric))
            if relative > worst_relative[0]:
                worst_relative = (relative, (first, second, distance, analytic, numeric))

        for boundary in (inner, outer, V.CUTOFF_ON, V.CUTOFF):
            delta = 1e-9
            energy_jump = abs(float(
                V.suppressed_vdw_energy(
                    boundary + delta, first, second, model=model
                ) - V.suppressed_vdw_energy(
                    boundary - delta, first, second, model=model
                )
            ))
            force_jump = abs(float(
                V.suppressed_vdw_force(
                    boundary + delta, first, second, model=model
                ) - V.suppressed_vdw_force(
                    boundary - delta, first, second, model=model
                )
            ))
            boundary_jumps["energy_eV"] = max(
                boundary_jumps["energy_eV"], energy_jump
            )
            boundary_jumps["force_eV_per_A"] = max(
                boundary_jumps["force_eV_per_A"], force_jump
            )
    return {
        "worst_force_absolute_eV_per_A": worst_absolute[0],
        "worst_force_absolute_case": worst_absolute[1],
        "worst_force_relative": worst_relative[0],
        "worst_force_relative_case": worst_relative[1],
        "limiting_boundary_differences_at_1e-9_A": boundary_jumps,
    }


def reactive_pair_energy(distance, first, second):
    types = np.array([R.ELEMENT_INDEX[first], R.ELEMENT_INDEX[second]])
    positions = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    return float(R.potential_energy(positions, types))


def combined_report(model):
    report = {}
    distances = np.linspace(0.45, 6.5, 24001)
    for first, second in REPRESENTATIVE_PAIRS:
        reactive = np.array([
            reactive_pair_energy(distance, first, second)
            for distance in distances
        ])
        vdw = V.suppressed_vdw_energy(
            distances, first, second, model=model
        )
        total = reactive + vdw
        reactive_min_index = int(np.argmin(reactive))
        chemical_window = distances < V.reactive_interval(first, second)[1]
        combined_chemical_index = int(np.where(chemical_window)[0][
            np.argmin(total[chemical_window])
        ])
        minima, maxima = extrema_indices(total)
        inner, outer = V.reactive_interval(first, second)
        transition = (distances >= inner) & (distances <= outer)
        transition_force = np.abs(np.gradient(total, distances))
        nonbonded_minima = [
            int(index) for index in minima if distances[index] > outer
        ]
        report[f"{first}-{second}"] = {
            "reactive_minimum_A": float(distances[reactive_min_index]),
            "reactive_minimum_eV": float(reactive[reactive_min_index]),
            "combined_chemical_minimum_A": float(distances[combined_chemical_index]),
            "combined_chemical_minimum_eV": float(total[combined_chemical_index]),
            "chemical_minimum_shift_A": float(
                distances[combined_chemical_index] - distances[reactive_min_index]
            ),
            "transition_max_energy_eV": float(np.max(total[transition])),
            "transition_max_abs_force_eV_per_A": float(
                np.max(transition_force[transition])
            ),
            "transition_local_minima": [
                float(distances[index]) for index in minima
                if inner < distances[index] < outer
            ],
            "transition_local_maxima": [
                float(distances[index]) for index in maxima
                if inner < distances[index] < outer
            ],
            "nonbonded_minima": [
                {"distance_A": float(distances[index]),
                 "energy_eV": float(total[index])}
                for index in nonbonded_minima
            ],
        }
    return report


def molecule_geometries():
    h2 = (("H", "H"), np.array([[-0.37072, 0.0, 0.0], [0.37072, 0.0, 0.0]]))

    tetra = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float) / np.sqrt(3.0)
    ch4 = (("C", "H", "H", "H", "H"), np.vstack((
        np.zeros(3), 1.086 * tetra
    )))

    angle = np.deg2rad(104.5)
    h2o = (("O", "H", "H"), np.array([
        [0.0, 0.0, 0.0],
        [0.96 * np.cos(angle / 2), 0.96 * np.sin(angle / 2), 0.0],
        [0.96 * np.cos(angle / 2), -0.96 * np.sin(angle / 2), 0.0],
    ]))

    cos_pair = np.cos(np.deg2rad(107.0))
    cos_polar = np.sqrt((cos_pair + 0.5) / 1.5)
    sin_polar = np.sqrt(1.0 - cos_polar ** 2)
    nh_vectors = np.array([
        [sin_polar * np.cos(phi), sin_polar * np.sin(phi), cos_polar]
        for phi in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)
    ])
    nh3 = (("N", "H", "H", "H"), np.vstack((
        np.zeros(3), 1.0109 * nh_vectors
    )))
    return {"H2": h2, "CH4": ch4, "H2O": h2o, "NH3": nh3}


def molecular_report(model):
    geometries = molecule_geometries()
    cases = (
        ("H2", "H2"), ("CH4", "CH4"), ("H2O", "H2O"),
        ("NH3", "NH3"), ("CH4", "H2O"), ("NH3", "H2O"),
    )
    separations = np.linspace(2.0, 10.0, 3201)
    report = {}
    for first_name, second_name in cases:
        first_symbols, first_positions = geometries[first_name]
        second_symbols, second_base = geometries[second_name]
        energies = []
        forces = []
        for separation in separations:
            second_positions = second_base + np.array([separation, 0.0, 0.0])
            energy, force = V.pairwise_molecular_vdw(
                first_symbols, first_positions, second_symbols, second_positions,
                model=model,
            )
            energies.append(energy)
            forces.append(force[0])
        energies = np.asarray(energies)
        forces = np.asarray(forces)
        minimum = int(np.argmin(energies))
        report[f"{first_name}...{second_name}"] = {
            "minimum_centre_separation_A": float(separations[minimum]),
            "minimum_energy_eV": float(energies[minimum]),
            "energy_at_2A_eV": float(energies[0]),
            "force_at_2A_eV_per_A": float(forces[0]),
            "energy_at_10A_eV": float(energies[-1]),
            "force_at_10A_eV_per_A": float(forces[-1]),
            "finite": bool(np.all(np.isfinite(energies)) and np.all(np.isfinite(forces))),
        }
    return report


def build_report():
    report = {
        "parameter_mapping": {
            "outer_minimum_and_depth": "audited UFF values",
            "outer_tail_match": f"UFF energy at {V.OUTER_MATCH_RATIO} r_min",
            "reaxff_p_vdw1": V.REAXFF_P_VDW1,
            "reaxff_diagnostic_core_fraction": V.REAXFF_CORE_FRACTION,
        },
        "cutoff_A": V.CUTOFF,
        "minimum_compatible_box_A": 2.0 * V.CUTOFF,
        "models": {},
    }
    for model in V.RAW_MODELS:
        report["models"][model] = {
            "raw_pairs": raw_pair_report(model),
            "continuity": continuity_report(model),
            "combined_pairs": combined_report(model),
            "molecular_scans": molecular_report(model),
        }
    return report


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
