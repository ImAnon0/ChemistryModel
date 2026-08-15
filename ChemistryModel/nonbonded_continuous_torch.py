"""
Torch/autograd implementation of the continuous SAPT-derived
exchange-repulsion model.

This module mirrors nonbonded_continuous.py, but all geometry and
continuous bond-weight operations remain inside Torch so energies are
differentiable with respect to positions and bond weights.

It is still a research bridge, not production ChemistryModel wiring.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nonbonded_reference import (
    ELEMENT_PARAMETERS,
    PAULING_ELECTRONEGATIVITY,
    LAMBDA_H,
    G2,
    G3,
    H1,
    ZETA,
)

from nonbonded_continuous import (
    SINGLE_BOND_RE,
    MULTIPLE_COMPRESSION_FRACTION,
    PLANE_TRACE_FULL_SCALE,
)


EPS = 1.0e-12


@dataclass
class ContinuousTorchFragment:
    symbols: list[str]
    positions: torch.Tensor
    bond_weights: torch.Tensor

    def __post_init__(self):
        self.symbols = list(self.symbols)

        if not torch.is_tensor(self.positions):
            self.positions = torch.as_tensor(
                self.positions,
                dtype=torch.float64,
            )

        if not self.positions.dtype.is_floating_point:
            self.positions = self.positions.to(
                dtype=torch.float64,
            )

        if not torch.is_tensor(self.bond_weights):
            self.bond_weights = torch.as_tensor(
                self.bond_weights,
                dtype=self.positions.dtype,
                device=self.positions.device,
            )
        else:
            self.bond_weights = self.bond_weights.to(
                dtype=self.positions.dtype,
                device=self.positions.device,
            )

        atom_count = len(self.symbols)

        if tuple(self.positions.shape) != (
            atom_count,
            3,
        ):
            raise ValueError(
                "positions must have shape "
                f"({atom_count}, 3), got "
                f"{tuple(self.positions.shape)}"
            )

        if tuple(self.bond_weights.shape) != (
            atom_count,
            atom_count,
        ):
            raise ValueError(
                "bond_weights must have shape "
                f"({atom_count}, {atom_count}), got "
                f"{tuple(self.bond_weights.shape)}"
            )

        detached = self.bond_weights.detach()

        if not torch.allclose(
            detached,
            detached.T,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "bond_weights must be symmetric"
            )

        if bool(
            torch.any(detached < -EPS)
        ) or bool(
            torch.any(
                detached > 1.0 + EPS
            )
        ):
            raise ValueError(
                "bond weights must lie in [0, 1]"
            )

        if not torch.allclose(
            torch.diagonal(detached),
            torch.zeros(
                atom_count,
                dtype=detached.dtype,
                device=detached.device,
            ),
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "bond_weights diagonal must be zero"
            )

        for symbol in self.symbols:
            if symbol not in ELEMENT_PARAMETERS:
                raise ValueError(
                    f"Unsupported element: {symbol}"
                )


def _constant(
    value: float,
    like: torch.Tensor,
) -> torch.Tensor:
    return like.new_tensor(
        float(value)
    )


def _unit(
    vector: torch.Tensor,
) -> torch.Tensor:
    norm = torch.linalg.vector_norm(
        vector
    )

    return (
        vector
        / torch.clamp(
            norm,
            min=EPS,
        )
    )


def _axes(
    fragment: ContinuousTorchFragment,
    atom_index: int,
) -> torch.Tensor:
    delta = (
        fragment.positions
        - fragment.positions[
            atom_index
        ]
    )

    norms = torch.linalg.vector_norm(
        delta,
        dim=1,
        keepdim=True,
    )

    return (
        delta
        / torch.clamp(
            norms,
            min=EPS,
        )
    )


def smootherstep01(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    C2 smooth gate:
        x <= 0 -> 0
        x >= 1 -> 1

    Inside:
        6x^5 - 15x^4 + 10x^3

    torch.where keeps the endpoint branches differentiable with
    zero slope outside the active interval.
    """

    zero = torch.zeros_like(x)
    one = torch.ones_like(x)

    polynomial = (
        x*x*x
        * (
            x * (
                x*6.0 - 15.0
            )
            + 10.0
        )
    )

    return torch.where(
        x <= 0.0,
        zero,
        torch.where(
            x >= 1.0,
            one,
            polynomial,
        ),
    )


def p1(
    x: torch.Tensor,
) -> torch.Tensor:
    return x


def p2(
    x: torch.Tensor,
) -> torch.Tensor:
    return (
        0.5
        * (
            3.0*x*x - 1.0
        )
    )


def p3(
    x: torch.Tensor,
) -> torch.Tensor:
    return (
        0.5
        * (
            5.0*x*x*x
            - 3.0*x
        )
    )


def atom_weights(
    fragment: ContinuousTorchFragment,
    atom_index: int,
) -> torch.Tensor:
    return fragment.bond_weights[
        atom_index
    ]


def coordination_weight(
    fragment: ContinuousTorchFragment,
    atom_index: int,
) -> torch.Tensor:
    return torch.sum(
        atom_weights(
            fragment,
            atom_index,
        )
    )


def presence_gate(
    fragment: ContinuousTorchFragment,
    atom_index: int,
) -> torch.Tensor:
    return smootherstep01(
        coordination_weight(
            fragment,
            atom_index,
        )
    )


def multibond_gate(
    fragment: ContinuousTorchFragment,
    atom_index: int,
) -> torch.Tensor:
    weights = atom_weights(
        fragment,
        atom_index,
    )

    total = torch.sum(weights)
    squares = torch.sum(
        weights*weights
    )

    pair_mass = (
        0.5
        * (
            total*total
            - squares
        )
    )

    return smootherstep01(
        pair_mass
    )


def amplitude_q2(
    fragment: ContinuousTorchFragment,
    atom_index: int,
    direction: torch.Tensor,
) -> torch.Tensor:
    weights = atom_weights(
        fragment,
        atom_index,
    )

    total_weight = torch.sum(
        weights
    )

    rhat = _unit(direction)

    cosine = (
        _axes(
            fragment,
            atom_index,
        )
        @ rhat
    )

    weighted = torch.sum(
        weights
        * p2(cosine)
    )

    average = (
        weighted
        / (
            total_weight
            + EPS
        )
    )

    return (
        presence_gate(
            fragment,
            atom_index,
        )
        * average
    )


def geometric_moments(
    fragment: ContinuousTorchFragment,
    atom_index: int,
    direction: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    weights = atom_weights(
        fragment,
        atom_index,
    )

    total_weight = torch.sum(
        weights
    )

    rhat = _unit(direction)

    cosine = (
        _axes(
            fragment,
            atom_index,
        )
        @ rhat
    )

    gate = multibond_gate(
        fragment,
        atom_index,
    )

    denominator = (
        total_weight
        + EPS
    )

    m2 = (
        gate
        * torch.sum(
            weights
            * p2(cosine)
        )
        / denominator
    )

    m3 = (
        gate
        * torch.sum(
            weights
            * p3(cosine)
        )
        / denominator
    )

    return m2, m3


def _chi_vector(
    fragment: ContinuousTorchFragment,
) -> torch.Tensor:
    return fragment.positions.new_tensor(
        [
            PAULING_ELECTRONEGATIVITY[
                symbol
            ]
            for symbol
            in fragment.symbols
        ]
    )


def chemical_n1(
    fragment: ContinuousTorchFragment,
    atom_index: int,
    direction: torch.Tensor,
) -> torch.Tensor:
    weights = atom_weights(
        fragment,
        atom_index,
    )

    total_weight = torch.sum(
        weights
    )

    chis = _chi_vector(
        fragment
    )

    mean_chi = (
        torch.sum(
            weights
            * chis
        )
        / (
            total_weight
            + EPS
        )
    )

    rhat = _unit(direction)

    cosine = (
        _axes(
            fragment,
            atom_index,
        )
        @ rhat
    )

    total = torch.sum(
        weights
        * (
            chis
            - mean_chi
        )
        * p1(cosine)
    )

    return (
        multibond_gate(
            fragment,
            atom_index,
        )
        * total
        / (
            total_weight
            + EPS
        )
    )


def effective_A(
    fragment: ContinuousTorchFragment,
    atom_index: int,
) -> torch.Tensor:
    symbol = fragment.symbols[
        atom_index
    ]

    base = _constant(
        ELEMENT_PARAMETERS[
            symbol
        ].A,
        fragment.positions,
    )

    if symbol != "H":
        return base

    weights = atom_weights(
        fragment,
        atom_index,
    )

    total_weight = torch.sum(
        weights
    )

    delta_chi = (
        _chi_vector(fragment)
        - PAULING_ELECTRONEGATIVITY[
            "H"
        ]
    )

    mean_delta_chi = (
        torch.sum(
            weights
            * delta_chi
        )
        / (
            total_weight
            + EPS
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
        * torch.exp(
            -LAMBDA_H
            * environment
        )
    )


def multiple_character(
    fragment: ContinuousTorchFragment,
    atom_index: int,
    neighbour_index: int,
) -> torch.Tensor:
    weight = fragment.bond_weights[
        atom_index,
        neighbour_index,
    ]

    symbol_a = fragment.symbols[
        atom_index
    ]

    symbol_b = fragment.symbols[
        neighbour_index
    ]

    try:
        re_value = SINGLE_BOND_RE[
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

    distance = (
        torch.linalg.vector_norm(
            fragment.positions[
                neighbour_index
            ]
            - fragment.positions[
                atom_index
            ]
        )
    )

    re = _constant(
        re_value,
        fragment.positions,
    )

    normalized = (
        (re - distance)
        / re
        / MULTIPLE_COMPRESSION_FRACTION
    )

    return (
        weight
        * smootherstep01(
            normalized
        )
    )


def polar_multiple_bond_strength(
    fragment: ContinuousTorchFragment,
    atom_index: int,
) -> torch.Tensor:
    symbol_i = fragment.symbols[
        atom_index
    ]

    chi_i = (
        PAULING_ELECTRONEGATIVITY[
            symbol_i
        ]
    )

    strength = (
        fragment.positions.sum()
        * 0.0
    )

    for neighbour, neighbour_symbol in enumerate(
        fragment.symbols
    ):
        if neighbour == atom_index:
            continue

        polarity = max(
            PAULING_ELECTRONEGATIVITY[
                neighbour_symbol
            ]
            - chi_i,
            0.0,
        )

        if polarity == 0.0:
            continue

        strength = (
            strength
            + multiple_character(
                fragment,
                atom_index,
                neighbour,
            )
            * polarity
        )

    return strength


def perpendicular_projector_value(
    fragment: ContinuousTorchFragment,
    atom_index: int,
    direction: torch.Tensor,
) -> torch.Tensor:
    weights = atom_weights(
        fragment,
        atom_index,
    )

    axes = _axes(
        fragment,
        atom_index,
    )

    tensor = torch.zeros(
        (3, 3),
        dtype=fragment.positions.dtype,
        device=fragment.positions.device,
    )

    atom_count = len(
        fragment.symbols
    )

    for a in range(
        atom_count
    ):
        if a == atom_index:
            continue

        for b in range(
            a + 1,
            atom_count,
        ):
            if b == atom_index:
                continue

            normal = torch.linalg.cross(
                axes[a],
                axes[b],
                dim=0,
            )

            tensor = (
                tensor
                + weights[a]
                * weights[b]
                * torch.outer(
                    normal,
                    normal,
                )
            )

    trace = torch.trace(
        tensor
    )

    rhat = _unit(
        direction
    )

    numerator = (
        rhat
        @ tensor
        @ rhat
    )

    directional = (
        numerator
        / (
            trace
            + EPS
        )
    )

    support = multibond_gate(
        fragment,
        atom_index,
    )

    geometry_support = (
        smootherstep01(
            trace
            / PLANE_TRACE_FULL_SCALE
        )
    )

    return (
        support
        * geometry_support
        * directional
    )


def polar_pi_correction(
    fragment: ContinuousTorchFragment,
    atom_index: int,
    direction: torch.Tensor,
    *,
    zeta: float = ZETA,
) -> torch.Tensor:
    return (
        zeta
        * polar_multiple_bond_strength(
            fragment,
            atom_index,
        )
        * perpendicular_projector_value(
            fragment,
            atom_index,
            direction,
        )
    )


def effective_B(
    fragment: ContinuousTorchFragment,
    atom_index: int,
    direction: torch.Tensor,
    *,
    zeta: float = ZETA,
) -> torch.Tensor:
    symbol = fragment.symbols[
        atom_index
    ]

    base = _constant(
        ELEMENT_PARAMETERS[
            symbol
        ].B,
        fragment.positions,
    )

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
        base
        * torch.exp(
            correction
        )
    )


def fragment_repulsion_energy(
    fragment_a: ContinuousTorchFragment,
    fragment_b: ContinuousTorchFragment,
    *,
    zeta: float = ZETA,
) -> torch.Tensor:
    if (
        fragment_a.positions.device
        != fragment_b.positions.device
    ):
        raise ValueError(
            "Fragments must be on the same Torch device"
        )

    if (
        fragment_a.positions.dtype
        != fragment_b.positions.dtype
    ):
        raise ValueError(
            "Fragments must use the same Torch dtype"
        )

    total = (
        fragment_a.positions.sum()
        * 0.0
        + fragment_b.positions.sum()
        * 0.0
    )

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

            distance = (
                torch.linalg.vector_norm(
                    delta
                )
            )

            rhat = (
                delta
                / torch.clamp(
                    distance,
                    min=EPS,
                )
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

            beta = torch.sqrt(
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
                * torch.exp(
                    -x
                )
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

            angular = torch.exp(
                ELEMENT_PARAMETERS[
                    symbol_a
                ].k*q_a
                + ELEMENT_PARAMETERS[
                    symbol_b
                ].k*q_b
            )

            total = (
                total
                + radial
                * angular
            )

    return total


def weights_from_bonds(
    atom_count: int,
    bonds,
    *,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> torch.Tensor:
    """
    Validation helper only.

    Production reactive MD should provide its own continuous
    covalent-weight matrix.
    """

    matrix = torch.zeros(
        (
            atom_count,
            atom_count,
        ),
        dtype=dtype,
        device=device,
    )

    for bond in bonds:
        if hasattr(bond, "i"):
            a = int(
                bond.i
            )
            b = int(
                bond.j
            )
        else:
            a = int(
                bond[0]
            )
            b = int(
                bond[1]
            )

        matrix[a, b] = 1.0
        matrix[b, a] = 1.0

    return matrix


__all__ = [
    "ContinuousTorchFragment",
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
