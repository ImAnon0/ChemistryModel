"""
Continuous-environment SAPT exchange-repulsion reference.

This is the bridge between the frozen static reference model and the
eventual reactive-MD implementation.

Unlike nonbonded_reference.py, this module does NOT require integer
bond orders or a hard neighbour list.

Instead it receives a symmetric continuous covalent-weight matrix:

    0 <= w_ij <= 1

Stable bond:
    w_ij = 1

Absent bond:
    w_ij = 0

Reactive transition:
    0 < w_ij < 1

The production ChemistryModel implementation will eventually provide
these weights from its reactive state.

This module deliberately does not infer covalent weights from every
close interatomic distance, because a close nonbonded SAPT probe must
not automatically become part of the molecule's covalent environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from nonbonded_reference import (
    ELEMENT_PARAMETERS,
    PAULING_ELECTRONEGATIVITY,
    LAMBDA_H,
    G2,
    G3,
    H1,
    ZETA,
)


EPS = 1.0e-12

# Single-bond equilibrium distances used only for the continuous
# multiple-bond/compression descriptor.
SINGLE_BOND_RE = {
    ("H", "H"): 0.74144,
    ("C", "H"): 1.086,
    ("H", "C"): 1.086,
    ("N", "H"): 1.0109,
    ("H", "N"): 1.0109,
    ("O", "H"): 0.960,
    ("H", "O"): 0.960,
    ("C", "C"): 1.525,
    ("C", "N"): 1.470,
    ("N", "C"): 1.470,
    ("C", "O"): 1.427,
    ("O", "C"): 1.427,
    ("N", "N"): 1.446,
    ("N", "O"): 1.453,
    ("O", "N"): 1.453,
    ("O", "O"): 1.475,
}


# Roughly 15% compression of the fitted single-bond equilibrium
# distance represents fully developed multiple-bond character.
#
# This is NOT fitted to a reaction barrier.
MULTIPLE_COMPRESSION_FRACTION = 0.15

# Numerical geometry-support scale for the local-plane tensor.
#
# The normalized plane direction is not meaningful when all bonded
# axes are nearly collinear. For the simplest two-axis case,
# trace(N) = sin(theta)^2, so 0.01 corresponds to full activation
# once the axes are separated by roughly 5.7 degrees.
#
# This is a smooth regularization scale, not a SAPT fit parameter.
PLANE_TRACE_FULL_SCALE = 0.01


# ============================================================
# DATA
# ============================================================


@dataclass
class ContinuousFragment:
    symbols: list[str]
    positions: np.ndarray
    bond_weights: np.ndarray

    def __post_init__(self):

        self.symbols = list(
            self.symbols
        )

        self.positions = np.asarray(
            self.positions,
            dtype=float,
        )

        self.bond_weights = np.asarray(
            self.bond_weights,
            dtype=float,
        )

        atom_count = len(
            self.symbols
        )

        if self.positions.shape != (
            atom_count,
            3,
        ):
            raise ValueError(
                "positions must have shape "
                f"({atom_count}, 3)"
            )

        if self.bond_weights.shape != (
            atom_count,
            atom_count,
        ):
            raise ValueError(
                "bond_weights must have shape "
                f"({atom_count}, {atom_count})"
            )

        if not np.allclose(
            self.bond_weights,
            self.bond_weights.T,
            atol=1e-12,
        ):
            raise ValueError(
                "bond_weights must be symmetric"
            )

        if np.any(
            self.bond_weights < -EPS
        ) or np.any(
            self.bond_weights > 1.0 + EPS
        ):
            raise ValueError(
                "bond weights must lie in [0, 1]"
            )

        if not np.allclose(
            np.diag(self.bond_weights),
            0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "bond_weights diagonal must be zero"
            )

        for symbol in self.symbols:
            if symbol not in ELEMENT_PARAMETERS:
                raise ValueError(
                    f"Unsupported element: {symbol}"
                )


# ============================================================
# SMALL SMOOTH BUILDING BLOCKS
# ============================================================


def _unit(vector):

    vector = np.asarray(
        vector,
        dtype=float,
    )

    norm = float(
        np.linalg.norm(vector)
    )

    if norm < EPS:
        raise ValueError(
            "Cannot normalize zero vector"
        )

    return vector / norm


def smootherstep01(x):
    """
    C2 smooth transition:

        x <= 0 -> 0
        x >= 1 -> 1

    Between them:

        6x^5 - 15x^4 + 10x^3

    First and second derivatives vanish at both boundaries.
    """

    x = float(x)

    if x <= 0.0:
        return 0.0

    if x >= 1.0:
        return 1.0

    return (
        x*x*x
        * (
            x * (
                x*6.0 - 15.0
            )
            + 10.0
        )
    )


def p1(x):
    return x


def p2(x):
    return 0.5 * (
        3.0*x*x - 1.0
    )


def p3(x):
    return 0.5 * (
        5.0*x*x*x - 3.0*x
    )


# ============================================================
# WEIGHT HELPERS
# ============================================================


def atom_weights(
    fragment,
    atom_index,
):

    return np.asarray(
        fragment.bond_weights[
            atom_index
        ],
        dtype=float,
    )


def coordination_weight(
    fragment,
    atom_index,
):

    return float(
        np.sum(
            atom_weights(
                fragment,
                atom_index,
            )
        )
    )


def presence_gate(
    fragment,
    atom_index,
):
    """
    Exactly:
        0 for no bond weight
        1 once total covalent occupancy reaches one full bond

    Smoothly interpolates between them.
    """

    return smootherstep01(
        coordination_weight(
            fragment,
            atom_index,
        )
    )


def multibond_gate(
    fragment,
    atom_index,
):
    """
    Smooth measure of whether at least two covalent contacts exist.

    Pair mass:

        sum_(a<b) w_a w_b

    One full neighbour:
        pair mass = 0

    Two full neighbours:
        pair mass = 1

    Three or more:
        pair mass >= 1

    Therefore stable two/three/four-coordinate atoms reproduce
    the original multibond descriptor exactly.
    """

    weights = atom_weights(
        fragment,
        atom_index,
    )

    total = float(
        np.sum(weights)
    )

    squares = float(
        np.sum(
            weights*weights
        )
    )

    pair_mass = 0.5 * (
        total*total - squares
    )

    return smootherstep01(
        pair_mass
    )


# ============================================================
# WEIGHTED LOCAL DIRECTIONS
# ============================================================


def _bond_axis(
    fragment,
    atom_index,
    neighbour_index,
):

    return _unit(
        fragment.positions[
            neighbour_index
        ]
        - fragment.positions[
            atom_index
        ]
    )


# ============================================================
# P2 AMPLITUDE ANISOTROPY
# ============================================================


def amplitude_q2(
    fragment,
    atom_index,
    direction,
):

    weights = atom_weights(
        fragment,
        atom_index,
    )

    total_weight = float(
        np.sum(weights)
    )

    if total_weight < EPS:
        return 0.0

    rhat = _unit(direction)

    weighted = 0.0

    for neighbour, weight in enumerate(
        weights
    ):

        if weight <= 0.0:
            continue

        axis = _bond_axis(
            fragment,
            atom_index,
            neighbour,
        )

        cosine = float(
            np.dot(
                axis,
                rhat,
            )
        )

        weighted += (
            weight
            * p2(cosine)
        )

    average = (
        weighted
        / (
            total_weight + EPS
        )
    )

    return (
        presence_gate(
            fragment,
            atom_index,
        )
        * average
    )


# ============================================================
# MULTIBOND GEOMETRIC MOMENTS
# ============================================================


def geometric_moments(
    fragment,
    atom_index,
    direction,
):

    weights = atom_weights(
        fragment,
        atom_index,
    )

    total_weight = float(
        np.sum(weights)
    )

    if total_weight < EPS:
        return 0.0, 0.0

    gate = multibond_gate(
        fragment,
        atom_index,
    )

    if gate <= 0.0:
        return 0.0, 0.0

    rhat = _unit(direction)

    m2 = 0.0
    m3 = 0.0

    for neighbour, weight in enumerate(
        weights
    ):

        if weight <= 0.0:
            continue

        axis = _bond_axis(
            fragment,
            atom_index,
            neighbour,
        )

        cosine = float(
            np.dot(
                axis,
                rhat,
            )
        )

        m2 += (
            weight
            * p2(cosine)
        )

        m3 += (
            weight
            * p3(cosine)
        )

    denominator = (
        total_weight + EPS
    )

    return (
        gate * m2 / denominator,
        gate * m3 / denominator,
    )


# ============================================================
# CHEMICAL FRONT/BACK ASYMMETRY
# ============================================================


def chemical_n1(
    fragment,
    atom_index,
    direction,
):

    weights = atom_weights(
        fragment,
        atom_index,
    )

    total_weight = float(
        np.sum(weights)
    )

    if total_weight < EPS:
        return 0.0

    gate = multibond_gate(
        fragment,
        atom_index,
    )

    if gate <= 0.0:
        return 0.0

    weighted_chi = 0.0

    for neighbour, weight in enumerate(
        weights
    ):

        if weight <= 0.0:
            continue

        weighted_chi += (
            weight
            * PAULING_ELECTRONEGATIVITY[
                fragment.symbols[
                    neighbour
                ]
            ]
        )

    mean_chi = (
        weighted_chi
        / (
            total_weight + EPS
        )
    )

    rhat = _unit(direction)

    total = 0.0

    for neighbour, weight in enumerate(
        weights
    ):

        if weight <= 0.0:
            continue

        axis = _bond_axis(
            fragment,
            atom_index,
            neighbour,
        )

        cosine = float(
            np.dot(
                axis,
                rhat,
            )
        )

        neighbour_chi = (
            PAULING_ELECTRONEGATIVITY[
                fragment.symbols[
                    neighbour
                ]
            ]
        )

        total += (
            weight
            * (
                neighbour_chi
                - mean_chi
            )
            * p1(cosine)
        )

    return (
        gate
        * total
        / (
            total_weight + EPS
        )
    )


# ============================================================
# HYDROGEN ENVIRONMENT RESPONSE
# ============================================================


def effective_A(
    fragment,
    atom_index,
):

    symbol = fragment.symbols[
        atom_index
    ]

    base = ELEMENT_PARAMETERS[
        symbol
    ].A

    if symbol != "H":
        return base

    weights = atom_weights(
        fragment,
        atom_index,
    )

    total_weight = float(
        np.sum(weights)
    )

    if total_weight < EPS:
        return base

    weighted_delta_chi = 0.0

    for neighbour, weight in enumerate(
        weights
    ):

        if weight <= 0.0:
            continue

        neighbour_symbol = (
            fragment.symbols[
                neighbour
            ]
        )

        delta_chi = (
            PAULING_ELECTRONEGATIVITY[
                neighbour_symbol
            ]
            - PAULING_ELECTRONEGATIVITY[
                "H"
            ]
        )

        weighted_delta_chi += (
            weight
            * delta_chi
        )

    mean_delta_chi = (
        weighted_delta_chi
        / (
            total_weight + EPS
        )
    )

    environment = (
        presence_gate(
            fragment,
            atom_index,
        )
        * mean_delta_chi
    )

    return (
        base
        * math.exp(
            -LAMBDA_H
            * environment
        )
    )


# ============================================================
# CONTINUOUS MULTIPLE-BOND CHARACTER
# ============================================================


def multiple_character(
    fragment,
    atom_index,
    neighbour_index,
):
    """
    Continuous multiple-bond proxy derived from bond compression.

    At or beyond the fitted single-bond equilibrium:
        0

    At >=15% compression:
        1

    Smooth C2 interpolation between.

    It is also multiplied by the continuous covalent occupancy.
    """

    weight = float(
        fragment.bond_weights[
            atom_index,
            neighbour_index,
        ]
    )

    if weight <= 0.0:
        return 0.0

    symbol_a = fragment.symbols[
        atom_index
    ]

    symbol_b = fragment.symbols[
        neighbour_index
    ]

    try:
        re = SINGLE_BOND_RE[
            (
                symbol_a,
                symbol_b,
            )
        ]

    except KeyError as exc:
        raise ValueError(
            "No single-bond equilibrium "
            f"for {symbol_a}-{symbol_b}"
        ) from exc

    distance = float(
        np.linalg.norm(
            fragment.positions[
                neighbour_index
            ]
            - fragment.positions[
                atom_index
            ]
        )
    )

    fractional_compression = (
        re - distance
    ) / re

    normalized = (
        fractional_compression
        / MULTIPLE_COMPRESSION_FRACTION
    )

    return (
        weight
        * smootherstep01(
            normalized
        )
    )


def polar_multiple_bond_strength(
    fragment,
    atom_index,
):

    symbol_i = fragment.symbols[
        atom_index
    ]

    chi_i = (
        PAULING_ELECTRONEGATIVITY[
            symbol_i
        ]
    )

    strength = 0.0

    weights = atom_weights(
        fragment,
        atom_index,
    )

    for neighbour, weight in enumerate(
        weights
    ):

        if weight <= 0.0:
            continue

        neighbour_symbol = (
            fragment.symbols[
                neighbour
            ]
        )

        polarity = max(
            PAULING_ELECTRONEGATIVITY[
                neighbour_symbol
            ]
            - chi_i,
            0.0,
        )

        if polarity <= 0.0:
            continue

        strength += (
            multiple_character(
                fragment,
                atom_index,
                neighbour,
            )
            * polarity
        )

    return strength


# ============================================================
# WEIGHTED LOCAL PLANE TENSOR
# ============================================================


def perpendicular_projector_value(
    fragment,
    atom_index,
    direction,
):

    weights = atom_weights(
        fragment,
        atom_index,
    )

    axes = []

    for neighbour, weight in enumerate(
        weights
    ):

        if weight <= 0.0:
            continue

        axes.append(
            (
                weight,
                _bond_axis(
                    fragment,
                    atom_index,
                    neighbour,
                ),
            )
        )

    if len(axes) < 2:
        return 0.0

    tensor = np.zeros(
        (3, 3),
        dtype=float,
    )

    for a in range(
        len(axes)
    ):

        weight_a, axis_a = (
            axes[a]
        )

        for b in range(
            a + 1,
            len(axes),
        ):

            weight_b, axis_b = (
                axes[b]
            )

            normal = np.cross(
                axis_a,
                axis_b,
            )

            tensor += (
                weight_a
                * weight_b
                * np.outer(
                    normal,
                    normal,
                )
            )

    trace = float(
        np.trace(tensor)
    )

    rhat = _unit(direction)

    numerator = float(
        rhat
        @ tensor
        @ rhat
    )

    directional = (
        numerator
        / (
            trace + EPS
        )
    )

    support = multibond_gate(
        fragment,
        atom_index,
    )

    # The normalized tensor direction becomes ill-conditioned as the
    # bonded geometry approaches collinearity: numerator and trace both
    # vanish, while their ratio can jump from 0 to nearly 1 over an
    # arbitrarily tiny angle. Fade the plane response out according to
    # the actual geometric support of the plane.
    geometry_support = smootherstep01(
        trace / PLANE_TRACE_FULL_SCALE
    )

    return (
        support
        * geometry_support
        * directional
    )


def polar_pi_correction(
    fragment,
    atom_index,
    direction,
    *,
    zeta=ZETA,
):

    strength = (
        polar_multiple_bond_strength(
            fragment,
            atom_index,
        )
    )

    perpendicular = (
        perpendicular_projector_value(
            fragment,
            atom_index,
            direction,
        )
    )

    return (
        zeta
        * strength
        * perpendicular
    )


# ============================================================
# EFFECTIVE B
# ============================================================


def effective_B(
    fragment,
    atom_index,
    direction,
    *,
    zeta=ZETA,
):

    symbol = fragment.symbols[
        atom_index
    ]

    m2, m3 = geometric_moments(
        fragment,
        atom_index,
        direction,
    )

    n1 = chemical_n1(
        fragment,
        atom_index,
        direction,
    )

    correction = (
        G2*m2
        + G3*m3
        + H1*n1
        + polar_pi_correction(
            fragment,
            atom_index,
            direction,
            zeta=zeta,
        )
    )

    return (
        ELEMENT_PARAMETERS[
            symbol
        ].B
        * math.exp(
            correction
        )
    )


# ============================================================
# CROSS-FRAGMENT REPULSION
# ============================================================


def fragment_repulsion_energy(
    fragment_a,
    fragment_b,
    *,
    zeta=ZETA,
):

    total = 0.0

    for atom_a, symbol_a in enumerate(
        fragment_a.symbols
    ):

        A_a = effective_A(
            fragment_a,
            atom_a,
        )

        for atom_b, symbol_b in enumerate(
            fragment_b.symbols
        ):

            A_b = effective_A(
                fragment_b,
                atom_b,
            )

            delta = (
                fragment_b.positions[
                    atom_b
                ]
                - fragment_a.positions[
                    atom_a
                ]
            )

            distance = float(
                np.linalg.norm(delta)
            )

            if distance < EPS:
                raise ValueError(
                    "Cross-fragment atoms occupy "
                    "the same position"
                )

            rhat = (
                delta
                / distance
            )

            B_a = effective_B(
                fragment_a,
                atom_a,
                rhat,
                zeta=zeta,
            )

            B_b = effective_B(
                fragment_b,
                atom_b,
                -rhat,
                zeta=zeta,
            )

            beta = math.sqrt(
                B_a * B_b
            )

            x = (
                beta
                * distance
            )

            radial = (
                A_a
                * A_b
                * (
                    1.0
                    + x
                    + x*x/3.0
                )
                * math.exp(-x)
            )

            q_a = amplitude_q2(
                fragment_a,
                atom_a,
                rhat,
            )

            q_b = amplitude_q2(
                fragment_b,
                atom_b,
                -rhat,
            )

            angular = math.exp(
                ELEMENT_PARAMETERS[
                    symbol_a
                ].k*q_a
                + ELEMENT_PARAMETERS[
                    symbol_b
                ].k*q_b
            )

            total += (
                radial
                * angular
            )

    return float(total)


# ============================================================
# STATIC -> CONTINUOUS REGRESSION HELPER
# ============================================================


def weights_from_bonds(
    atom_count,
    bonds,
):
    """
    Convenience helper for validation only.

    Produces exact 0/1 occupancies from a static connectivity list.
    Production reactive MD must provide continuous weights instead.
    """

    matrix = np.zeros(
        (
            atom_count,
            atom_count,
        ),
        dtype=float,
    )

    for bond in bonds:

        if hasattr(bond, "i"):
            a = int(bond.i)
            b = int(bond.j)
        else:
            a = int(bond[0])
            b = int(bond[1])

        matrix[a, b] = 1.0
        matrix[b, a] = 1.0

    return matrix


__all__ = [
    "ContinuousFragment",
    "SINGLE_BOND_RE",
    "MULTIPLE_COMPRESSION_FRACTION",
    "PLANE_TRACE_FULL_SCALE",
    "smootherstep01",
    "presence_gate",
    "multibond_gate",
    "amplitude_q2",
    "geometric_moments",
    "chemical_n1",
    "effective_A",
    "multiple_character",
    "polar_multiple_bond_strength",
    "perpendicular_projector_value",
    "polar_pi_correction",
    "effective_B",
    "fragment_repulsion_energy",
    "weights_from_bonds",
]
