"""Numerical comparison of standalone vdW energy-partition architectures."""

from __future__ import annotations

import json

import numpy as np

import reactive as R
import vdw_partition as P
import vdw_reference as V


PAIRS = (("H", "H"), ("C", "H"), ("O", "H"), ("C", "C"),
         ("C", "O"), ("N", "N"), ("O", "O"))


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


def force_error(first, second, architecture):
    inner, outer = V.reactive_interval(first, second)
    probes = np.concatenate((
        np.linspace(inner + 1e-3, outer - 1e-3, 25),
        np.linspace(2.6, 6.8, 15), np.linspace(7.01, 8.49, 15),
    ))
    worst = 0.0
    for distance in probes:
        step = 1e-6
        _, analytic = P.partition_energy_force(
            distance, first, second, architecture
        )
        plus = P.partition_energy_force(
            distance + step, first, second, architecture
        )[0]
        minus = P.partition_energy_force(
            distance - step, first, second, architecture
        )[0]
        numeric = -(plus - minus) / (2 * step)
        worst = max(worst, abs(float(analytic - numeric)))
    return worst


def build_report():
    distances = np.linspace(0.45, 6.5, 24001)
    report = {architecture: {} for architecture in P.ARCHITECTURES}
    for first, second in PAIRS:
        reactive = np.array([
            reactive_pair_energy(distance, first, second)
            for distance in distances
        ])
        reactive_minimum = int(np.argmin(reactive))
        inner, outer = V.reactive_interval(first, second)
        transition = (distances >= inner) & (distances <= outer)
        for architecture in P.ARCHITECTURES:
            vdw, _ = P.partition_energy_force(
                distances, first, second, architecture
            )
            total = reactive + vdw
            minima, maxima = extrema(total)
            chemical = distances <= outer
            chemical_index = np.flatnonzero(chemical)[
                np.argmin(total[chemical])
            ]
            outer_minima = [index for index in minima if distances[index] > outer]
            transition_maxima = [
                index for index in maxima if inner < distances[index] < outer
            ]
            report[architecture][f"{first}-{second}"] = {
                "maximum_added_energy_in_transition_eV": float(
                    np.max(vdw[transition])
                ),
                "maximum_total_energy_in_transition_eV": float(
                    np.max(total[transition])
                ),
                "transition_local_maxima_A": [
                    float(distances[index]) for index in transition_maxima
                ],
                "chemical_minimum_shift_A": float(
                    distances[chemical_index] - distances[reactive_minimum]
                ),
                "chemical_minimum_energy_shift_eV": float(
                    total[chemical_index] - reactive[reactive_minimum]
                ),
                "outer_minima": [
                    {"distance_A": float(distances[index]),
                     "energy_eV": float(total[index])}
                    for index in outer_minima
                ],
                "maximum_force_derivative_disagreement_eV_per_A": force_error(
                    first, second, architecture
                ),
            }
    return report


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
