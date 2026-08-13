"""Fast, observational diagnostics for bond-parameter calibration.

Nothing in this module is imported by the simulation engine.  It derives
spectroscopic quantities from the engine's existing tables and runs short,
controlled probes so candidate constants can be compared with the untouched
baseline.
"""

from __future__ import annotations

import math
import time

import numpy as np

import build_box
import reactive as R
from reactive_torch import ReactiveSimulation


EV_PER_ANGSTROM2_TO_N_PER_M = 16.02176634
ATOMIC_MASS_TO_KG = 1.66053906892e-27
LIGHT_CM_PER_SECOND = 2.99792458e10
WAVENUMBER_TO_EV = 1.2398419843320026e-4

H2_REFERENCE = {
    "re_angstrom": 0.74144,
    "omega_e_cm-1": 4401.21,
    "omega_exe_cm-1": 121.33,
    "D0_cm-1": 36118.06962,
}


def morse_curvature(depth_ev, width_inverse_angstrom):
    """Return d2V/dr2 at re in eV/A^2 for D(e^-2ax - 2e^-ax)."""
    return 2.0 * float(depth_ev) * float(width_inverse_angstrom) ** 2


def harmonic_wavenumber(depth_ev, width_inverse_angstrom,
                        first_mass_amu, second_mass_amu):
    reduced_amu = first_mass_amu * second_mass_amu / (
        first_mass_amu + second_mass_amu
    )
    curvature_si = (
        morse_curvature(depth_ev, width_inverse_angstrom)
        * EV_PER_ANGSTROM2_TO_N_PER_M
    )
    angular_frequency = math.sqrt(
        curvature_si / (reduced_amu * ATOMIC_MASS_TO_KG)
    )
    return angular_frequency / (2.0 * math.pi * LIGHT_CM_PER_SECOND)


def width_for_frequency(depth_ev, wavenumber_cm, first_mass_amu,
                        second_mass_amu):
    reduced_kg = (
        first_mass_amu * second_mass_amu
        / (first_mass_amu + second_mass_amu)
        * ATOMIC_MASS_TO_KG
    )
    angular = 2.0 * math.pi * LIGHT_CM_PER_SECOND * wavenumber_cm
    curvature_ev_a2 = (
        reduced_kg * angular ** 2 / EV_PER_ANGSTROM2_TO_N_PER_M
    )
    return math.sqrt(curvature_ev_a2 / (2.0 * depth_ev))


def spectroscopic_h2_depths(reference=H2_REFERENCE):
    omega = reference["omega_e_cm-1"]
    anharmonicity = reference["omega_exe_cm-1"]
    # G(0) = omega_e/2 - omega_exe/4 for the usual diatomic expansion.
    zpe_cm = 0.5 * omega - 0.25 * anharmonicity
    de_from_d0_cm = reference["D0_cm-1"] + zpe_cm
    # Exact only for an ideal Morse oscillator; report as a diagnostic.
    ideal_morse_de_cm = omega ** 2 / (4.0 * anharmonicity)
    return {
        "zpe_cm-1": zpe_cm,
        "De_from_D0_eV": de_from_d0_cm * WAVENUMBER_TO_EV,
        "ideal_Morse_De_eV": ideal_morse_de_cm * WAVENUMBER_TO_EV,
    }


def h2_curve(samples=401):
    distances = np.linspace(0.35, 3.0, samples)
    types = R.types_from_symbols(["H", "H"])
    energies = np.array([
        R.potential_energy(np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]]), types)
        for r in distances
    ])
    minimum = int(np.argmin(energies))
    tail = float(energies[-1])
    derivative = np.diff(energies)
    post_minimum_falling_steps = int(
        np.count_nonzero(derivative[minimum:] < -1e-8)
    )
    i = R.ELEMENT_INDEX["H"]
    depth = float(R.BOND_DEPTH[i, i])
    width = float(R.BOND_WIDTH[i, i])
    return {
        "table_re_A": float(R.BOND_LENGTH[i, i]),
        "sampled_re_A": float(distances[minimum]),
        "table_depth_eV": depth,
        "sampled_well_eV": float(tail - energies[minimum]),
        "width_inv_A": width,
        "curvature_eV_A2": morse_curvature(depth, width),
        "harmonic_cm-1": harmonic_wavenumber(
            depth, width, R.MASS["H"], R.MASS["H"]
        ),
        "energy_at_0.35A_eV": float(energies[0]),
        "energy_at_3A_eV": tail,
        "post_minimum_falling_steps": post_minimum_falling_steps,
    }


def pair_local_diagnostic(first, second):
    """Report the raw table pair as a local two-body Morse oscillator."""
    first_index = R.ELEMENT_INDEX[first]
    second_index = R.ELEMENT_INDEX[second]
    depth = float(R.BOND_DEPTH[first_index, second_index])
    width = float(R.BOND_WIDTH[first_index, second_index])
    return {
        "re_A": float(R.BOND_LENGTH[first_index, second_index]),
        "depth_eV": depth,
        "width_inv_A": width,
        "curvature_eV_A2": morse_curvature(depth, width),
        "local_harmonic_cm-1": harmonic_wavenumber(
            depth, width, R.MASS[first], R.MASS[second]
        ),
    }


def methane_ch_coordinate(samples=401):
    """Stretch one methane C-H coordinate with the other atoms held fixed."""
    symbols, equilibrium = build_box.BUILDERS["CH4"]()
    direction = equilibrium[-1] / np.linalg.norm(equilibrium[-1])
    distances = np.linspace(0.55, 3.0, samples)
    types = R.types_from_symbols(symbols)
    energies = []
    for distance in distances:
        positions = equilibrium.copy()
        positions[-1] = direction * distance
        energies.append(R.potential_energy(positions, types))
    energies = np.asarray(energies)
    minimum = int(np.argmin(energies))
    derivative = np.diff(energies)
    return {
        "sampled_minimum_A": float(distances[minimum]),
        "dissociation_coordinate_eV": float(energies[-1] - energies[minimum]),
        "short_range_energy_eV": float(energies[0] - energies[-1]),
        "capture_region_falling_steps": int(np.count_nonzero(
            derivative[minimum:] < -1e-8
        )),
        "table": pair_local_diagnostic("C", "H"),
    }


def molecule_nve(name, steps=400, temperature=100.0):
    symbols, positions = build_box.BUILDERS[name]()
    positions = np.asarray(positions) + 5.0
    simulation = ReactiveSimulation(
        symbols, positions, 12.0, target_temperature=temperature,
        friction=0.0, device="cpu", random_seed=19,
    )
    simulation.thermostat_is_on = False
    start = float(simulation.potential_energy + simulation.kinetic_energy)
    initial = simulation.positions_numpy.copy()
    simulation.step(steps)
    end = float(simulation.potential_energy + simulation.kinetic_energy)
    displacement = simulation.positions_numpy - initial
    displacement -= simulation.box_size * np.round(displacement / simulation.box_size)
    return {
        "energy_start_eV": start,
        "energy_end_eV": end,
        "drift_eV": end - start,
        "max_displacement_A": float(np.linalg.norm(displacement, axis=1).max()),
        "capped_steps": int(simulation.capped_steps),
    }


def runtime_probe(repeats=5, steps=100):
    symbols, positions = build_box.BUILDERS["H2"]()
    timings = []
    for seed in range(repeats):
        simulation = ReactiveSimulation(
            symbols, positions + 5.0, 12.0, target_temperature=100.0,
            device="cpu", random_seed=seed,
        )
        started = time.perf_counter()
        simulation.step(steps)
        timings.append(time.perf_counter() - started)
    return {"median_seconds": float(np.median(timings)), "steps": steps}


def baseline_report():
    return {
        "h2_reference": dict(H2_REFERENCE),
        "h2_reference_derived": spectroscopic_h2_depths(),
        "h2_curve": h2_curve(),
        "molecules": {
            name: molecule_nve(name) for name in ("H2", "CH4", "NH3", "H2O")
        },
        "runtime": runtime_probe(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(baseline_report(), indent=2, sort_keys=True))
