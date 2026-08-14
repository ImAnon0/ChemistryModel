"""Audit base pair repulsion and Tang-Toennies dispersion-only totals."""

from __future__ import annotations

import json

import numpy as np

import dispersion_reference as D
import reactive as R
import vdw_reference as V


def reactive_pair_energy(distance, first, second):
    types = np.array([R.ELEMENT_INDEX[first], R.ELEMENT_INDEX[second]])
    positions = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    return float(R.potential_energy(positions, types))


def extrema(values):
    minima = np.flatnonzero(
        (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])
    ) + 1
    maxima = np.flatnonzero(
        (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
    ) + 1
    return minima, maxima


def build_report(convention="morse_proxy"):
    report = {}
    for first, second in V.unique_pairs():
        equilibrium = V.REACTIVE_SINGLE_BOND_LENGTH[
            V.canonical_pair(first, second)
        ]
        distances = np.linspace(max(0.08, 0.15 * equilibrium), 8.5, 30001)
        reactive = np.array([
            reactive_pair_energy(distance, first, second)
            for distance in distances
        ])
        reactive_force = -np.gradient(reactive, distances)
        dispersion = D.dispersion_energy(
            distances, first, second, convention=convention
        )
        total = reactive + dispersion
        minima, maxima = extrema(total)
        reactive_minimum = int(np.argmin(reactive))
        total_minimum = int(np.argmin(total))
        inner_zero_candidates = np.flatnonzero(
            (reactive[:-1] > 0.0) & (reactive[1:] <= 0.0)
        )
        inner_zero = (
            float(distances[inner_zero_candidates[0]])
            if len(inner_zero_candidates) else None
        )
        _, reactive_outer = V.reactive_interval(first, second)
        outer_minima = [index for index in minima if distances[index] > reactive_outer]
        outer_maxima = [index for index in maxima if distances[index] > reactive_outer]
        probe = int(np.argmin(np.abs(distances - 0.5 * equilibrium)))
        report[f"{first}-{second}"] = {
            "reactive": {
                "chemical_minimum_A": float(distances[reactive_minimum]),
                "chemical_minimum_eV": float(reactive[reactive_minimum]),
                "inner_zero_crossing_A": inner_zero,
                "wall_probe_A": float(distances[probe]),
                "wall_probe_energy_eV": float(reactive[probe]),
                "wall_probe_force_eV_per_A": float(reactive_force[probe]),
                "energy_at_shortest_probe_eV": float(reactive[0]),
                "force_at_shortest_probe_eV_per_A": float(reactive_force[0]),
                "attraction_exactly_zero_from_A": reactive_outer,
                "collapse_possible": bool(
                    reactive[0] <= reactive[reactive_minimum]
                    or reactive_force[0] <= 0.0
                ),
            },
            "dispersion": {
                "C6_eV_A6": D.c6_coefficient(first, second),
                "TT_b_A^-1": D.damping_exponent(
                    first, second, convention
                ),
            },
            "combined": {
                "chemical_minimum_A": float(distances[total_minimum]),
                "chemical_minimum_shift_A": float(
                    distances[total_minimum] - distances[reactive_minimum]
                ),
                "chemical_minimum_energy_shift_eV": float(
                    total[total_minimum] - reactive[reactive_minimum]
                ),
                "outer_minima": [
                    {"distance_A": float(distances[index]),
                     "energy_eV": float(total[index])}
                    for index in outer_minima
                ],
                "outer_maxima": [float(distances[index]) for index in outer_maxima],
            },
        }
    return report


def molecular_report(convention="morse_proxy"):
    """Fixed-orientation dimer interaction scans; no geometry fitting."""
    from vdw_diagnostics import molecule_geometries

    geometries = molecule_geometries()
    cases = (("H2", "H2"), ("CH4", "CH4"), ("H2O", "H2O"),
             ("NH3", "NH3"), ("CH4", "H2O"), ("NH3", "H2O"))
    separations = np.linspace(1.5, 10.0, 1201)
    report = {}
    for first_name, second_name in cases:
        first_symbols, first_positions = geometries[first_name]
        second_symbols, second_base = geometries[second_name]
        first_types = np.array([R.ELEMENT_INDEX[x] for x in first_symbols])
        second_types = np.array([R.ELEMENT_INDEX[x] for x in second_symbols])
        first_internal = R.potential_energy(first_positions, first_types)
        second_internal = R.potential_energy(second_base, second_types)
        reactive_values = []
        dispersion_values = []
        for separation in separations:
            second_positions = second_base + np.array([separation, 0.0, 0.0])
            positions = np.vstack((first_positions, second_positions))
            types = np.concatenate((first_types, second_types))
            reactive_values.append(
                R.potential_energy(positions, types)
                - first_internal - second_internal
            )
            dispersion = 0.0
            for i, first in enumerate(first_symbols):
                for j, second in enumerate(second_symbols):
                    distance = np.linalg.norm(
                        second_positions[j] - first_positions[i]
                    )
                    dispersion += D.dispersion_energy(
                        distance, first, second, convention=convention
                    )
            dispersion_values.append(dispersion)
        reactive_values = np.asarray(reactive_values)
        total = reactive_values + np.asarray(dispersion_values)
        reactive_minima, reactive_maxima = extrema(reactive_values)
        total_minima, total_maxima = extrema(total)
        report[f"{first_name}...{second_name}"] = {
            "reactive_minima": [
                {"separation_A": float(separations[i]),
                 "energy_eV": float(reactive_values[i])}
                for i in reactive_minima
            ],
            "combined_minima": [
                {"separation_A": float(separations[i]),
                 "energy_eV": float(total[i])}
                for i in total_minima
            ],
            "reactive_maxima_A": [float(separations[i]) for i in reactive_maxima],
            "combined_maxima_A": [float(separations[i]) for i in total_maxima],
        }
    return report


if __name__ == "__main__":
    print(json.dumps({"pairs": build_report(), "dimers": molecular_report()}, indent=2))
