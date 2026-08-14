"""Standalone NumPy reference for the proposed ChemistryModel vdW layer.

This module is deliberately not imported by the production simulators.

The element parameters and geometric combining rules are from the Universal
Force Field (UFF): A. K. Rappe et al., JACS 114, 10024-10035 (1992),
DOI 10.1021/ja00051a040.  ``x`` is the pair-potential minimum in angstrom and
``D`` is its well depth.  Energies returned here are eV.

The chemical-contact suppression and long-range quintic switch are the
ChemistryModel-specific design recorded in docs/vdw_design.md.  They are not
part of UFF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


KCAL_PER_MOL_TO_EV = 0.0433641153087705
ELEMENTS = ("H", "C", "N", "O")


@dataclass(frozen=True)
class UFFVdwParameter:
    minimum_angstrom: float
    depth_kcal_per_mol: float

    @property
    def depth_ev(self) -> float:
        return self.depth_kcal_per_mol * KCAL_PER_MOL_TO_EV


UFF_VDW = {
    "H": UFFVdwParameter(2.886, 0.044),
    "C": UFFVdwParameter(3.851, 0.105),
    "N": UFFVdwParameter(3.660, 0.069),
    "O": UFFVdwParameter(3.500, 0.060),
}


# Existing accepted single-bond lengths are copied intentionally rather than
# imported from reactive.py: this reference must remain isolated and cannot
# silently change production state.  They define only the proposed suppression
# interval (1.25 re to 1.60 re), not a discrete bond classification.
REACTIVE_SINGLE_BOND_LENGTH = {
    ("H", "H"): 0.74144,
    ("H", "C"): 1.086,
    ("H", "N"): 1.0109,
    ("H", "O"): 0.960,
    ("C", "C"): 1.525,
    ("C", "N"): 1.470,
    ("C", "O"): 1.427,
    ("N", "N"): 1.446,
    ("N", "O"): 1.453,
    ("O", "O"): 1.475,
}

CUTOFF_ON = 7.0
CUTOFF = 8.5
REAXFF_P_VDW1 = 1.69
REAXFF_CORE_FRACTION = 0.5
OUTER_MATCH_RATIO = 1.5


def canonical_pair(first: str, second: str) -> tuple[str, str]:
    if first not in UFF_VDW or second not in UFF_VDW:
        raise KeyError(f"unsupported vdW elements: {first!r}, {second!r}")
    return tuple(sorted((first, second), key=ELEMENTS.index))


def unique_pairs() -> Iterable[tuple[str, str]]:
    for index, first in enumerate(ELEMENTS):
        for second in ELEMENTS[index:]:
            yield first, second


def pair_parameters(first: str, second: str) -> tuple[float, float]:
    """Return UFF ``(x_ij [A], D_ij [eV])`` using geometric mixing."""
    first_value = UFF_VDW[first]
    second_value = UFF_VDW[second]
    minimum = np.sqrt(
        first_value.minimum_angstrom * second_value.minimum_angstrom
    )
    depth = np.sqrt(first_value.depth_ev * second_value.depth_ev)
    return float(minimum), float(depth)


def _distance_array(distance):
    values = np.asarray(distance, dtype=float)
    if np.any(values <= 0.0):
        raise ValueError("vdW distance must be strictly positive")
    return values


def raw_uff_energy(distance, first: str, second: str):
    """Unswitched UFF 12-6 energy in eV."""
    distance = _distance_array(distance)
    minimum, depth = pair_parameters(first, second)
    ratio6 = (minimum / distance) ** 6
    return depth * (ratio6 ** 2 - 2.0 * ratio6)


def raw_uff_force(distance, first: str, second: str):
    """Radial force ``-dE/dr`` in eV/A; positive means repulsive."""
    distance = _distance_array(distance)
    minimum, depth = pair_parameters(first, second)
    ratio6 = (minimum / distance) ** 6
    return 12.0 * depth * (ratio6 ** 2 - ratio6) / distance


def _morse_outer_match_exponent(first: str, second: str):
    """Return Morse ``a`` [1/A] matching UFF at 1.5 times its minimum."""
    minimum, depth = pair_parameters(first, second)
    outer = OUTER_MATCH_RATIO * minimum
    normalized_target = float(raw_uff_energy(outer, first, second) / depth)
    # z^2 - 2z = target; take 0 < z < 1 on the attractive outer branch.
    z = 1.0 - np.sqrt(1.0 + normalized_target)
    return float(-np.log(z) / (outer - minimum))


def airebo_m_parameters(first: str, second: str):
    """Diagnostic ``(r_eq, epsilon, alpha)`` matched to the UFF outer well.

    The functional form is published AIREBO-M. These H/C/N/O parameters are
    diagnostic mappings, not the fitted AIREBO-M hydrocarbon parameter set.
    """
    minimum, depth = pair_parameters(first, second)
    return minimum, depth, _morse_outer_match_exponent(first, second)


def raw_airebo_m_energy(distance, first: str, second: str):
    distance = _distance_array(distance)
    equilibrium, depth, alpha = airebo_m_parameters(first, second)
    exponential = np.exp(-alpha * (distance - equilibrium))
    return depth * (exponential ** 2 - 2.0 * exponential)


def raw_airebo_m_force(distance, first: str, second: str):
    distance = _distance_array(distance)
    equilibrium, depth, alpha = airebo_m_parameters(first, second)
    exponential = np.exp(-alpha * (distance - equilibrium))
    return 2.0 * alpha * depth * (exponential ** 2 - exponential)


def reaxff_shielded_parameters(first: str, second: str):
    """Return diagnostic ReaxFF ``(r_vdw, D, alpha, gamma_w, p)``.

    ReaxFF defines rho=(r^p+gamma_w^-p)^(1/p).  The core length is fixed here
    to half the audited UFF minimum for every pair; it is not copied from a
    ReaxFF force-field file. ``r_vdw`` is chosen so the actual-r minimum stays
    at the UFF value, and ``alpha`` matches the UFF outer attraction at 1.5x.
    """
    minimum, depth = pair_parameters(first, second)
    p = REAXFF_P_VDW1
    core = REAXFF_CORE_FRACTION * minimum

    def rho(radius):
        return (radius ** p + core ** p) ** (1.0 / p)

    r_vdw = rho(minimum)
    outer = OUTER_MATCH_RATIO * minimum
    normalized_target = float(raw_uff_energy(outer, first, second) / depth)
    z = 1.0 - np.sqrt(1.0 + normalized_target)
    half_alpha_over_radius = -np.log(z) / (rho(outer) - r_vdw)
    alpha = 2.0 * r_vdw * half_alpha_over_radius
    gamma_w = 1.0 / core
    return float(r_vdw), depth, float(alpha), float(gamma_w), p


def _reaxff_shielded_distance(distance, gamma_w, p):
    return (distance ** p + (1.0 / gamma_w) ** p) ** (1.0 / p)


def raw_reaxff_energy(distance, first: str, second: str):
    distance = _distance_array(distance)
    r_vdw, depth, alpha, gamma_w, p = reaxff_shielded_parameters(first, second)
    shielded = _reaxff_shielded_distance(distance, gamma_w, p)
    exponential = np.exp(0.5 * alpha * (1.0 - shielded / r_vdw))
    return depth * (exponential ** 2 - 2.0 * exponential)


def raw_reaxff_force(distance, first: str, second: str):
    distance = _distance_array(distance)
    r_vdw, depth, alpha, gamma_w, p = reaxff_shielded_parameters(first, second)
    shielded = _reaxff_shielded_distance(distance, gamma_w, p)
    exponential = np.exp(0.5 * alpha * (1.0 - shielded / r_vdw))
    shield_derivative = (distance / shielded) ** (p - 1.0)
    morse_alpha = 0.5 * alpha / r_vdw
    return (
        2.0 * morse_alpha * depth
        * (exponential ** 2 - exponential) * shield_derivative
    )


RAW_MODELS = {
    "uff": (raw_uff_energy, raw_uff_force),
    "airebo_m": (raw_airebo_m_energy, raw_airebo_m_force),
    "reaxff": (raw_reaxff_energy, raw_reaxff_force),
}


def cutoff_switch(distance, cutoff_on=CUTOFF_ON, cutoff=CUTOFF):
    distance = _distance_array(distance)
    if not 0.0 < cutoff_on < cutoff:
        raise ValueError("cutoff requires 0 < cutoff_on < cutoff")
    t = np.clip((distance - cutoff_on) / (cutoff - cutoff_on), 0.0, 1.0)
    value = 1.0 - 10.0 * t ** 3 + 15.0 * t ** 4 - 6.0 * t ** 5
    return np.where(distance >= cutoff, 0.0, np.where(distance <= cutoff_on, 1.0, value))


def cutoff_switch_derivative(distance, cutoff_on=CUTOFF_ON, cutoff=CUTOFF):
    distance = _distance_array(distance)
    t = (distance - cutoff_on) / (cutoff - cutoff_on)
    inside = (t > 0.0) & (t < 1.0)
    derivative_t = -30.0 * t ** 2 * (1.0 - t) ** 2
    return np.where(inside, derivative_t / (cutoff - cutoff_on), 0.0)


def switched_uff_energy(distance, first: str, second: str):
    distance = _distance_array(distance)
    return cutoff_switch(distance) * raw_uff_energy(distance, first, second)


def switched_uff_force(distance, first: str, second: str):
    distance = _distance_array(distance)
    raw_energy = raw_uff_energy(distance, first, second)
    raw_force = raw_uff_force(distance, first, second)
    switch = cutoff_switch(distance)
    switch_derivative = cutoff_switch_derivative(distance)
    return switch * raw_force - switch_derivative * raw_energy


def reactive_interval(first: str, second: str) -> tuple[float, float]:
    pair = canonical_pair(first, second)
    equilibrium = REACTIVE_SINGLE_BOND_LENGTH[pair]
    return 1.25 * equilibrium, 1.60 * equilibrium


def reactive_contact_taper(distance, first: str, second: str):
    """Minimal standalone reproduction of the production cosine taper."""
    distance = _distance_array(distance)
    inner, outer = reactive_interval(first, second)
    fraction = np.clip((distance - inner) / (outer - inner), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * fraction))


def reactive_contact_taper_derivative(distance, first: str, second: str):
    distance = _distance_array(distance)
    inner, outer = reactive_interval(first, second)
    fraction = (distance - inner) / (outer - inner)
    inside = (fraction > 0.0) & (fraction < 1.0)
    derivative = (
        -0.5 * np.pi * np.sin(np.pi * fraction) / (outer - inner)
    )
    return np.where(inside, derivative, 0.0)


def suppression_weight(distance, first: str, second: str):
    taper = reactive_contact_taper(distance, first, second)
    return 1.0 - 3.0 * taper ** 2 + 2.0 * taper ** 3


def suppression_weight_derivative(distance, first: str, second: str):
    taper = reactive_contact_taper(distance, first, second)
    taper_derivative = reactive_contact_taper_derivative(
        distance, first, second
    )
    return 6.0 * taper * (taper - 1.0) * taper_derivative


def suppressed_vdw_components(
    distance, first: str, second: str, model: str = "uff"
):
    """Return raw energy, suppression, cutoff switch, energy, and force.

    The radial force is the exact analytic derivative of the complete
    suppressed and switched energy.
    """
    distance = _distance_array(distance)
    try:
        energy_function, force_function = RAW_MODELS[model]
    except KeyError as error:
        raise ValueError(f"unknown vdW reference model: {model!r}") from error
    raw_energy = energy_function(distance, first, second)
    raw_force = force_function(distance, first, second)
    suppression = suppression_weight(distance, first, second)
    suppression_derivative = suppression_weight_derivative(
        distance, first, second
    )
    switch = cutoff_switch(distance)
    switch_derivative = cutoff_switch_derivative(distance)
    energy = suppression * switch * raw_energy
    force = (
        suppression * switch * raw_force
        - (suppression_derivative * switch
           + suppression * switch_derivative) * raw_energy
    )
    return {
        "raw_energy": raw_energy,
        "suppression": suppression,
        "cutoff_switch": switch,
        "energy": energy,
        "force": force,
    }


def suppressed_vdw_energy(
    distance, first: str, second: str, enabled=True, model: str = "uff"
):
    if not enabled:
        values = np.asarray(distance, dtype=float)
        return np.zeros_like(values)
    return suppressed_vdw_components(
        distance, first, second, model=model
    )["energy"]


def suppressed_vdw_force(
    distance, first: str, second: str, enabled=True, model: str = "uff"
):
    if not enabled:
        values = np.asarray(distance, dtype=float)
        return np.zeros_like(values)
    return suppressed_vdw_components(
        distance, first, second, model=model
    )["force"]


def pairwise_molecular_vdw(
    first_symbols,
    first_positions,
    second_symbols,
    second_positions,
    model="uff",
):
    """Return cross-molecule vdW energy and force on the second molecule.

    Intramolecular terms are deliberately excluded. Positions are ordinary
    Cartesian coordinates with no production PBC implementation.
    """
    first_positions = np.asarray(first_positions, dtype=float)
    second_positions = np.asarray(second_positions, dtype=float)
    total_energy = 0.0
    force_on_second = np.zeros(3, dtype=float)
    for i, first in enumerate(first_symbols):
        for j, second in enumerate(second_symbols):
            displacement = second_positions[j] - first_positions[i]
            distance = float(np.linalg.norm(displacement))
            if distance <= 0.0:
                raise ValueError("molecular scan contains coincident atoms")
            values = suppressed_vdw_components(
                distance, first, second, model=model
            )
            total_energy += float(values["energy"])
            force_on_second += float(values["force"]) * displacement / distance
    return total_energy, force_on_second
