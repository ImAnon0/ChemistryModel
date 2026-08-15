"""
Static SAPT-derived exchange-repulsion reference model.

This module intentionally uses explicit molecular connectivity and bond orders.
It is a RESEARCH REFERENCE, not yet the production reactive-MD implementation.

The production version must later replace hard connectivity/bond orders with
smooth continuous environment descriptors.

Energy model
------------

For a cross-fragment atom pair i,j:

    V_ij =
        A_i_eff A_j_eff
        (1 + x + x^2/3) exp(-x)
        exp(k_i q2_i + k_j q2_j)

where:

    x = r sqrt(B_i_eff B_j_eff)

Environment-dependent B:

    B_i_eff = B_i exp(
        g2 m2_i
        + g3 m3_i
        + h1 n1_i
        + zeta s_i p_i
    )

Supported fitted terms:

* P2 single-axis amplitude anisotropy
* hydrogen neighbour response
* multibond m2/m3 directional decay
* n1 chemical front/back asymmetry
* polar-multiple-bond perpendicular decay

The parameters below are frozen from the SAPT development programme and
must not be casually refitted to reaction barriers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import math
import numpy as np


# ============================================================
# FROZEN PARAMETERS
# ============================================================


@dataclass(frozen=True)
class ElementParameters:
    A: float
    B: float
    k: float


ELEMENT_PARAMETERS = {
    "H": ElementParameters(
        A=3.95924,
        B=4.20796,
        k=-0.27107,
    ),
    "C": ElementParameters(
        A=14.24570,
        B=3.75862,
        k=+0.33993,
    ),
    "N": ElementParameters(
        A=18.09959,
        B=4.24701,
        k=+0.12138,
    ),
    "O": ElementParameters(
        A=20.93635,
        B=4.73944,
        k=+0.03677,
    ),
}


PAULING_ELECTRONEGATIVITY = {
    "H": 2.20,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
}


# Hydrogen amplitude response.
LAMBDA_H = 0.547542

# Multibond directional decay.
G2 = +0.230153
G3 = -0.126666

# Neighbour-identity front/back asymmetry.
H1 = -0.19440

# Polar multiple-bond perpendicular decay.
ZETA = +0.271733629


EPS = 1.0e-12


# ============================================================
# TYPES
# ============================================================


@dataclass(frozen=True)
class Bond:
    i: int
    j: int
    order: float = 1.0


@dataclass
class Fragment:
    symbols: Sequence[str]
    positions: np.ndarray
    bonds: Sequence[Bond]

    def __post_init__(self):
        self.positions = np.asarray(
            self.positions,
            dtype=float,
        )

        if self.positions.shape != (
            len(self.symbols),
            3,
        ):
            raise ValueError(
                "positions must have shape "
                f"({len(self.symbols)}, 3), "
                f"got {self.positions.shape}"
            )

        for symbol in self.symbols:
            if symbol not in ELEMENT_PARAMETERS:
                raise ValueError(
                    f"Unsupported element: {symbol}"
                )


# ============================================================
# GEOMETRY
# ============================================================


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(
        vector,
        dtype=float,
    )

    norm = float(
        np.linalg.norm(vector)
    )

    if norm < EPS:
        raise ValueError(
            "Cannot normalize zero-length vector."
        )

    return vector / norm


def p1(x: float) -> float:
    return x


def p2(x: float) -> float:
    return 0.5 * (
        3.0*x*x - 1.0
    )


def p3(x: float) -> float:
    return 0.5 * (
        5.0*x*x*x - 3.0*x
    )


# ============================================================
# CONNECTIVITY
# ============================================================


def _adjacency(
    fragment: Fragment,
) -> list[list[int]]:

    result = [
        []
        for _ in fragment.symbols
    ]

    for bond in fragment.bonds:

        result[bond.i].append(
            bond.j
        )

        result[bond.j].append(
            bond.i
        )

    return result


def _bond_orders(
    fragment: Fragment,
) -> list[dict[int, float]]:

    result = [
        {}
        for _ in fragment.symbols
    ]

    for bond in fragment.bonds:

        result[bond.i][bond.j] = (
            float(bond.order)
        )

        result[bond.j][bond.i] = (
            float(bond.order)
        )

    return result


# ============================================================
# ORIGINAL P2 AMPLITUDE ANISOTROPY
# ============================================================


def amplitude_q2(
    fragment: Fragment,
    adjacency: Sequence[Sequence[int]],
    atom_index: int,
    direction: np.ndarray,
) -> float:

    neighbours = adjacency[
        atom_index
    ]

    if not neighbours:
        return 0.0

    position = fragment.positions[
        atom_index
    ]

    rhat = _unit(direction)

    total = 0.0

    for neighbour in neighbours:

        axis = _unit(
            fragment.positions[neighbour]
            - position
        )

        cosine = float(
            np.dot(
                axis,
                rhat,
            )
        )

        total += p2(cosine)

    return (
        total
        / len(neighbours)
    )


# ============================================================
# MULTIBOND GEOMETRIC MOMENTS
# ============================================================


def geometric_moments(
    fragment: Fragment,
    adjacency: Sequence[Sequence[int]],
    atom_index: int,
    direction: np.ndarray,
) -> tuple[float, float]:

    neighbours = adjacency[
        atom_index
    ]

    # Preserve the original diatomic/single-axis calibration.
    if len(neighbours) <= 1:
        return 0.0, 0.0

    position = fragment.positions[
        atom_index
    ]

    rhat = _unit(direction)

    m2 = 0.0
    m3 = 0.0

    for neighbour in neighbours:

        axis = _unit(
            fragment.positions[neighbour]
            - position
        )

        cosine = float(
            np.dot(
                axis,
                rhat,
            )
        )

        m2 += p2(cosine)
        m3 += p3(cosine)

    count = float(
        len(neighbours)
    )

    return (
        m2 / count,
        m3 / count,
    )


# ============================================================
# CHEMICAL FRONT/BACK ASYMMETRY
# ============================================================


def chemical_n1(
    fragment: Fragment,
    adjacency: Sequence[Sequence[int]],
    atom_index: int,
    direction: np.ndarray,
) -> float:

    neighbours = adjacency[
        atom_index
    ]

    if len(neighbours) <= 1:
        return 0.0

    neighbour_chi = np.asarray(
        [
            PAULING_ELECTRONEGATIVITY[
                fragment.symbols[neighbour]
            ]
            for neighbour in neighbours
        ],
        dtype=float,
    )

    mean_chi = float(
        np.mean(neighbour_chi)
    )

    position = fragment.positions[
        atom_index
    ]

    rhat = _unit(direction)

    total = 0.0

    for neighbour, chi in zip(
        neighbours,
        neighbour_chi,
    ):

        axis = _unit(
            fragment.positions[neighbour]
            - position
        )

        cosine = float(
            np.dot(
                axis,
                rhat,
            )
        )

        total += (
            chi - mean_chi
        ) * p1(cosine)

    return (
        total
        / len(neighbours)
    )


# ============================================================
# HYDROGEN ENVIRONMENT RESPONSE
# ============================================================


def effective_A(
    fragment: Fragment,
    adjacency: Sequence[Sequence[int]],
    atom_index: int,
) -> float:

    symbol = fragment.symbols[
        atom_index
    ]

    base = ELEMENT_PARAMETERS[
        symbol
    ].A

    if symbol != "H":
        return base

    neighbours = adjacency[
        atom_index
    ]

    if len(neighbours) != 1:
        return base

    neighbour_symbol = (
        fragment.symbols[
            neighbours[0]
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

    return (
        base
        * math.exp(
            -LAMBDA_H
            * delta_chi
        )
    )


# ============================================================
# POLAR MULTIPLE-BOND DESCRIPTOR
# ============================================================


def polar_multiple_bond_strength(
    fragment: Fragment,
    adjacency: Sequence[Sequence[int]],
    orders: Sequence[dict[int, float]],
    atom_index: int,
) -> float:
    """
    Detect local multiple bonding toward a more electronegative
    neighbour.

    Examples:

        C=C   -> 0
        C-O   -> 0 for a single bond
        C=O   -> positive
    """

    symbol = fragment.symbols[
        atom_index
    ]

    chi_i = (
        PAULING_ELECTRONEGATIVITY[
            symbol
        ]
    )

    strength = 0.0

    for neighbour in adjacency[
        atom_index
    ]:

        order = orders[
            atom_index
        ].get(
            neighbour,
            1.0,
        )

        multiple_character = max(
            order - 1.0,
            0.0,
        )

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

        strength += (
            multiple_character
            * polarity
        )

    return strength


def perpendicular_projector_value(
    fragment: Fragment,
    adjacency: Sequence[Sequence[int]],
    atom_index: int,
    direction: np.ndarray,
) -> float:
    """
    Rotation- and permutation-invariant measure of how strongly
    an interaction points perpendicular to the local bonded plane.

    N = sum_(a<b) (u_a x u_b)(u_a x u_b)^T

    p = rhat^T N rhat / trace(N)

    For an exactly planar bonded environment this reduces to
    (n_hat dot r_hat)^2 without choosing an arbitrary pair of
    bonds to define the normal.
    """

    neighbours = adjacency[
        atom_index
    ]

    if len(neighbours) < 2:
        return 0.0

    position = fragment.positions[
        atom_index
    ]

    axes = [
        _unit(
            fragment.positions[neighbour]
            - position
        )
        for neighbour in neighbours
    ]

    tensor = np.zeros(
        (3, 3),
        dtype=float,
    )

    for a in range(
        len(axes)
    ):
        for b in range(
            a + 1,
            len(axes),
        ):

            normal = np.cross(
                axes[a],
                axes[b],
            )

            tensor += np.outer(
                normal,
                normal,
            )

    trace = float(
        np.trace(tensor)
    )

    if trace < EPS:
        return 0.0

    rhat = _unit(direction)

    value = float(
        rhat
        @ tensor
        @ rhat
        / trace
    )

    # Numerical noise only.
    return min(
        max(value, 0.0),
        1.0,
    )


def polar_pi_correction(
    fragment: Fragment,
    adjacency: Sequence[Sequence[int]],
    orders: Sequence[dict[int, float]],
    atom_index: int,
    direction: np.ndarray,
    *,
    zeta: float = ZETA,
) -> float:

    strength = (
        polar_multiple_bond_strength(
            fragment,
            adjacency,
            orders,
            atom_index,
        )
    )

    if strength <= 0.0:
        return 0.0

    perpendicular = (
        perpendicular_projector_value(
            fragment,
            adjacency,
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
    fragment: Fragment,
    adjacency: Sequence[Sequence[int]],
    orders: Sequence[dict[int, float]],
    atom_index: int,
    direction: np.ndarray,
    *,
    zeta: float = ZETA,
) -> float:

    symbol = fragment.symbols[
        atom_index
    ]

    m2, m3 = geometric_moments(
        fragment,
        adjacency,
        atom_index,
        direction,
    )

    n1 = chemical_n1(
        fragment,
        adjacency,
        atom_index,
        direction,
    )

    correction = (
        G2 * m2
        + G3 * m3
        + H1 * n1
        + polar_pi_correction(
            fragment,
            adjacency,
            orders,
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
# PAIR ENERGY
# ============================================================


def pair_repulsion_energy(
    fragment_a: Fragment,
    fragment_b: Fragment,
    atom_a: int,
    atom_b: int,
    *,
    zeta: float = ZETA,
) -> float:

    adjacency_a = _adjacency(
        fragment_a
    )

    adjacency_b = _adjacency(
        fragment_b
    )

    orders_a = _bond_orders(
        fragment_a
    )

    orders_b = _bond_orders(
        fragment_b
    )

    position_a = (
        fragment_a.positions[
            atom_a
        ]
    )

    position_b = (
        fragment_b.positions[
            atom_b
        ]
    )

    delta = (
        position_b
        - position_a
    )

    distance = float(
        np.linalg.norm(delta)
    )

    if distance < EPS:
        raise ValueError(
            "Atoms occupy the same position."
        )

    rhat = delta / distance

    symbol_a = fragment_a.symbols[
        atom_a
    ]

    symbol_b = fragment_b.symbols[
        atom_b
    ]

    A_a = effective_A(
        fragment_a,
        adjacency_a,
        atom_a,
    )

    A_b = effective_A(
        fragment_b,
        adjacency_b,
        atom_b,
    )

    B_a = effective_B(
        fragment_a,
        adjacency_a,
        orders_a,
        atom_a,
        rhat,
        zeta=zeta,
    )

    B_b = effective_B(
        fragment_b,
        adjacency_b,
        orders_b,
        atom_b,
        -rhat,
        zeta=zeta,
    )

    beta = math.sqrt(
        B_a * B_b
    )

    x = beta * distance

    slater = (
        A_a
        * A_b
        * (
            1.0
            + x
            + x*x / 3.0
        )
        * math.exp(-x)
    )

    q_a = amplitude_q2(
        fragment_a,
        adjacency_a,
        atom_a,
        rhat,
    )

    q_b = amplitude_q2(
        fragment_b,
        adjacency_b,
        atom_b,
        -rhat,
    )

    angular = math.exp(
        ELEMENT_PARAMETERS[
            symbol_a
        ].k * q_a
        + ELEMENT_PARAMETERS[
            symbol_b
        ].k * q_b
    )

    return (
        slater
        * angular
    )


# ============================================================
# CROSS-FRAGMENT ENERGY
# ============================================================


def fragment_repulsion_energy(
    fragment_a: Fragment,
    fragment_b: Fragment,
    *,
    zeta: float = ZETA,
) -> float:
    """
    Total exchange-repulsion energy between two fragments.

    Internal fragment energies are deliberately excluded.
    """

    adjacency_a = _adjacency(
        fragment_a
    )

    adjacency_b = _adjacency(
        fragment_b
    )

    orders_a = _bond_orders(
        fragment_a
    )

    orders_b = _bond_orders(
        fragment_b
    )

    total = 0.0

    for atom_a, symbol_a in enumerate(
        fragment_a.symbols
    ):

        A_a = effective_A(
            fragment_a,
            adjacency_a,
            atom_a,
        )

        for atom_b, symbol_b in enumerate(
            fragment_b.symbols
        ):

            A_b = effective_A(
                fragment_b,
                adjacency_b,
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
                    "the same position."
                )

            rhat = (
                delta
                / distance
            )

            B_a = effective_B(
                fragment_a,
                adjacency_a,
                orders_a,
                atom_a,
                rhat,
                zeta=zeta,
            )

            B_b = effective_B(
                fragment_b,
                adjacency_b,
                orders_b,
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

            slater = (
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
                adjacency_a,
                atom_a,
                rhat,
            )

            q_b = amplitude_q2(
                fragment_b,
                adjacency_b,
                atom_b,
                -rhat,
            )

            angular = math.exp(
                ELEMENT_PARAMETERS[
                    symbol_a
                ].k * q_a
                + ELEMENT_PARAMETERS[
                    symbol_b
                ].k * q_b
            )

            total += (
                slater
                * angular
            )

    return float(total)


# ============================================================
# DEBUG / PROVENANCE
# ============================================================


def parameter_summary() -> dict:
    """
    Machine-readable frozen parameter provenance.
    """

    return {
        "elements": {
            element: {
                "A": values.A,
                "B_A^-1": values.B,
                "k": values.k,
            }
            for element, values
            in ELEMENT_PARAMETERS.items()
        },
        "lambda_H": LAMBDA_H,
        "g2": G2,
        "g3": G3,
        "h1": H1,
        "zeta": ZETA,
        "reference_quantity":
            "SAPT0/jun-cc-pVDZ SAPT EXCH10 ENERGY",
        "status":
            "static research reference; "
            "not production reactive MD",
    }


__all__ = [
    "Bond",
    "Fragment",
    "ElementParameters",
    "ELEMENT_PARAMETERS",
    "PAULING_ELECTRONEGATIVITY",
    "LAMBDA_H",
    "G2",
    "G3",
    "H1",
    "ZETA",
    "amplitude_q2",
    "geometric_moments",
    "chemical_n1",
    "polar_multiple_bond_strength",
    "perpendicular_projector_value",
    "polar_pi_correction",
    "effective_A",
    "effective_B",
    "pair_repulsion_energy",
    "fragment_repulsion_energy",
    "parameter_summary",
]
