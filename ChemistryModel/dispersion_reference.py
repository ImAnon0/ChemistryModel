"""Standalone Tang-Toennies C6 dispersion diagnostic for ChemistryModel."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammainc

import reactive as R
import vdw_reference as V


HARTREE_EV = 27.211386245981
BOHR_ANGSTROM = 0.529177210903
NIST_IONIZATION_ENERGY_EV = {
    "H": 13.598434599702,
    "C": 11.2602880,
    "N": 14.53413,
    "O": 13.618055,
}


def c6_coefficient(first: str, second: str) -> float:
    """UFF-derived C6 [eV A^6] from D[(x/r)^12 - 2(x/r)^6]."""
    minimum, depth = V.pair_parameters(first, second)
    return 2.0 * depth * minimum ** 6


def morse_proxy_exponent(first: str, second: str) -> float:
    """Use the existing single-bond Morse repulsive exponent, b=2a [A^-1]."""
    i = R.ELEMENT_INDEX[first]
    j = R.ELEMENT_INDEX[second]
    return 2.0 * float(R.BOND_WIDTH[i, j])


def atomic_ip_density_exponent(element: str) -> float:
    """Free-atom density exponent B=2 sqrt(2 IP), returned in A^-1."""
    ionization_hartree = NIST_IONIZATION_ENERGY_EV[element] / HARTREE_EV
    exponent_bohr_inverse = 2.0 * np.sqrt(2.0 * ionization_hartree)
    return float(exponent_bohr_inverse / BOHR_ANGSTROM)


def born_mayer_ip_exponent(first: str, second: str) -> float:
    """Published Born-Mayer-IP unlike-pair combination rule, in A^-1."""
    first_b = atomic_ip_density_exponent(first)
    second_b = atomic_ip_density_exponent(second)
    return float(
        (first_b + second_b) * first_b * second_b
        / (first_b ** 2 + second_b ** 2)
    )


def damping_exponent(first: str, second: str, convention: str) -> float:
    if convention == "morse_proxy":
        return morse_proxy_exponent(first, second)
    if convention == "born_mayer_ip":
        return born_mayer_ip_exponent(first, second)
    raise ValueError(f"unknown damping convention: {convention!r}")


def tang_toennies_f6(x):
    """f_6(x)=1-exp(-x) sum(k=0..6) x^k/k!."""
    x = np.asarray(x, dtype=float)
    if np.any(x < 0.0):
        raise ValueError("Tang-Toennies argument must be non-negative")
    return gammainc(7, x)


def tang_toennies_f6_derivative(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-x) * x ** 6 / math.factorial(6)


def dispersion_energy(
    distance, first: str, second: str, convention="morse_proxy"
):
    distance = np.asarray(distance, dtype=float)
    if np.any(distance <= 0.0):
        raise ValueError("dispersion distance must be positive")
    c6 = c6_coefficient(first, second)
    x = damping_exponent(first, second, convention) * distance
    return -c6 * tang_toennies_f6(x) / distance ** 6


def dispersion_force(
    distance, first: str, second: str, convention="morse_proxy"
):
    """Radial force -dE/dr [eV/A], negative when attractive."""
    distance = np.asarray(distance, dtype=float)
    if np.any(distance <= 0.0):
        raise ValueError("dispersion distance must be positive")
    c6 = c6_coefficient(first, second)
    exponent = damping_exponent(first, second, convention)
    x = exponent * distance
    damping = tang_toennies_f6(x)
    damping_derivative = tang_toennies_f6_derivative(x)
    return c6 * (
        exponent * damping_derivative / distance ** 6
        - 6.0 * damping / distance ** 7
    )
