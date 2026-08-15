"""
Four-system SAPT H-state decomposition at the coupling-requirement saddles.

Diagnostic only. No parameters are fitted or modified.

Input:
    sapt_coupling_requirement_diagnostic.csv

That CSV is produced by:
    sapt_coupling_requirement_diagnostic.py

For formaldehyde and methane, the script uses the required eta and its saved
frozen-surface saddle. For water, which has an acceptable eta interval, it uses
the boundary eta closest to the H3 anchor. H3 itself is evaluated at the
current H3-calibrated eta and symmetric saddle radius.

For each case it reports:
    - eta
    - reactant-like and product-like diabatic diagonals
    - signed and absolute diabatic gap
    - coupling V
    - coupling prefactor K = V / eta
    - mixing lowering relative to the lower diagonal
    - covalent contribution for each state
    - total SAPT wall contribution for each state
    - SAPT wall on the unoccupied competing transfer edge in each state

The purpose is to distinguish:
    1. a coupling-law problem (K / required V changes systematically), from
    2. a state-diagonal problem (large asymmetric wall / gap shifts that eta is
       merely compensating for).

Usage:
    py sapt_state_coupling_decomposition.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

import reactive as R
import h_state_torch as hs
import hf_surface_scan as scan
import nonbonded_continuous_torch as nb

from batched_torch import BatchedReactiveSimulation
from sapt_h_state_torch import (
    SAPT_H_STATE_MIXING,
    SaptHStateBatchedSimulation,
    _descriptor_weights_for_state,
    _sapt_pair_energy,
)


DEFAULT_CSV = (
    "sapt_coupling_requirement_diagnostic.csv"
)

DTYPE = torch.float64
DEVICE = "cpu"

CORRECT_WATER_FROZEN = np.array(
    [
        0.960,
        0.960,
        75.53,
        75.53,
    ],
    dtype=float,
)


def load_requirement_rows(path):
    rows = {}

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for row in reader:
            if row.get(
                "status"
            ) != "ok":
                continue

            rows[
                row[
                    "system"
                ]
            ] = row

    return rows


def as_float(
    row,
    key,
):
    value = row.get(
        key,
        ""
    )

    if value is None or value == "":
        return np.nan

    return float(
        value
    )


def choose_eta_and_saddle(
    system,
    row,
):
    low_eta = as_float(
        row,
        "required_eta_low",
    )

    high_eta = as_float(
        row,
        "required_eta_high",
    )

    if not np.isfinite(
        low_eta
    ):
        raise RuntimeError(
            f"{system}: missing required eta"
        )

    if not np.isfinite(
        high_eta
    ):
        high_eta = low_eta

    if math.isclose(
        low_eta,
        high_eta,
        abs_tol=1.0e-12,
    ):
        eta = low_eta
    else:
        # Water has a reference window. Use the acceptable eta boundary nearest
        # the H3 calibration so we do not artificially move farther from the
        # original anchor than required.
        choices = [
            low_eta,
            high_eta,
        ]

        eta = min(
            choices,
            key=lambda value: abs(
                value
                - SAPT_H_STATE_MIXING
            ),
        )

    donor = as_float(
        row,
        "saddle_donor_A",
    )

    transfer = as_float(
        row,
        "saddle_transfer_A",
    )

    if not (
        np.isfinite(
            donor
        )
        and np.isfinite(
            transfer
        )
    ):
        raise RuntimeError(
            f"{system}: missing saved saddle geometry"
        )

    return (
        float(
            eta
        ),
        float(
            donor
        ),
        float(
            transfer
        ),
    )


def frozen_spectators(
    system,
):
    if system == "water":
        return (
            CORRECT_WATER_FROZEN
            .copy()
        )

    return np.asarray(
        scan.SYSTEMS[
            system
        ][
            "frozen"
        ],
        dtype=float,
    ).copy()


def build_simulation(
    symbols,
    raw_positions,
    *,
    eta,
):
    return SaptHStateBatchedSimulation(
        boxes=[
            (
                symbols,
                np.asarray(
                    raw_positions,
                    dtype=float,
                )
                + scan.CENTRE,
            )
        ],
        box_size=scan.BOX,
        random_seed=0,
        relax_on_start=False,
        device=DEVICE,
        dtype=DTYPE,
        h_state_mixing=float(
            eta
        ),
    )


def prepare_intermediates(
    sim,
):
    positions = (
        sim.positions
        .detach()
        .requires_grad_(
            True
        )
    )

    BatchedReactiveSimulation.energy_per_atom(
        sim,
        positions,
    )

    cached = getattr(
        sim,
        "_reactive_intermediates",
        None,
    )

    if cached is None:
        raise RuntimeError(
            "reactive intermediates unavailable"
        )

    values = cached[
        1
    ]

    neighbours_numpy = (
        values[
            "neighbours"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    active_numpy = (
        (
            values[
                "taper"
            ]
            .detach()
            .cpu()
            .numpy()
            > 1.0e-12
        )
        & sim.neighbour_mask
        .detach()
        .cpu()
        .numpy()
    )

    (
        edge_atoms,
        edge_rows,
        edge_slots,
    ) = sim._active_edges_for_box(
        0,
        values,
        neighbours_numpy,
        active_numpy,
    )

    return (
        positions,
        values,
        edge_atoms,
        edge_rows,
        edge_slots,
    )


def distance(
    positions,
    first,
    second,
):
    delta = (
        positions[
            first
        ]
        - positions[
            second
        ]
    )

    return float(
        torch.linalg.norm(
            delta
        )
        .detach()
        .cpu()
    )


def identify_molecular_transfer_edges(
    edge_atoms,
    symbols,
    positions,
    *,
    donor_target,
    transfer_target,
):
    """
    Choose the H-H candidate closest to the saved forming H-H coordinate, then
    choose a heavy-H candidate sharing one H with it and closest to the saved
    donor coordinate.

    This works for C-H, O-H and N-H abstractions without hard-coding atom ids.
    """

    hh_candidates = []

    for index, (
        first,
        second,
    ) in enumerate(
        edge_atoms
    ):
        if (
            symbols[
                first
            ] == "H"
            and symbols[
                second
            ] == "H"
        ):
            hh_candidates.append(
                (
                    abs(
                        distance(
                            positions,
                            first,
                            second,
                        )
                        - transfer_target
                    ),
                    index,
                )
            )

    if not hh_candidates:
        raise RuntimeError(
            "no H-H transfer candidate found"
        )

    hh_index = min(
        hh_candidates
    )[
        1
    ]

    hh_atoms = set(
        edge_atoms[
            hh_index
        ]
    )

    donor_candidates = []

    for index, (
        first,
        second,
    ) in enumerate(
        edge_atoms
    ):
        first_symbol = symbols[
            first
        ]

        second_symbol = symbols[
            second
        ]

        is_heavy_h = (
            (
                first_symbol == "H"
                and second_symbol != "H"
            )
            or (
                second_symbol == "H"
                and first_symbol != "H"
            )
        )

        if not is_heavy_h:
            continue

        h_atom = (
            first
            if first_symbol == "H"
            else second
        )

        if h_atom not in hh_atoms:
            continue

        donor_candidates.append(
            (
                abs(
                    distance(
                        positions,
                        first,
                        second,
                    )
                    - donor_target
                ),
                index,
            )
        )

    if not donor_candidates:
        raise RuntimeError(
            "no donor heavy-H candidate sharing the transferred H"
        )

    donor_index = min(
        donor_candidates
    )[
        1
    ]

    return (
        donor_index,
        hh_index,
    )


def identify_h3_edges(
    edge_atoms,
):
    if len(
        edge_atoms
    ) != 2:
        raise RuntimeError(
            f"expected exactly 2 active H3 edges, got {len(edge_atoms)}"
        )

    return (
        0,
        1,
    )


def classify_states(
    states,
    donor_edge,
    product_edge,
):
    reactant_state = None
    product_state = None

    for index, state in enumerate(
        states
    ):
        occupied = set(
            state
        )

        if (
            donor_edge in occupied
            and product_edge not in occupied
        ):
            reactant_state = index

        if (
            product_edge in occupied
            and donor_edge not in occupied
        ):
            product_state = index

    if (
        reactant_state is None
        or product_state is None
    ):
        raise RuntimeError(
            "could not identify reactant/product diabatic states"
        )

    return (
        reactant_state,
        product_state,
    )


def symbol_lookup():
    return {
        int(
            index
        ): symbol
        for symbol, index
        in R.ELEMENT_INDEX.items()
    }


def build_state_matrix(
    sim,
    positions,
    values,
    edge_atoms,
    edge_rows,
    edge_slots,
    *,
    eta,
):
    taper = values[
        "taper"
    ]

    pair_depth = values[
        "pair_depth"
    ]

    pair_width = values[
        "pair_width"
    ]

    shift = values[
        "shift"
    ]

    repulsive = values[
        "repulsive"
    ]

    zero = taper.sum() * 0.0

    edge_tapers = []
    edge_depths = []
    edge_repulsive = []
    edge_attractive = []

    for row, slot in zip(
        edge_rows,
        edge_slots,
    ):
        contact = taper[
            row,
            slot,
        ]

        depth = pair_depth[
            row,
            slot,
        ]

        attractive = (
            2.0
            * depth
            * torch.exp(
                -pair_width[
                    row,
                    slot,
                ]
                * shift[
                    row,
                    slot,
                ]
            )
        )

        edge_tapers.append(
            contact
        )

        edge_depths.append(
            depth
        )

        edge_repulsive.append(
            contact
            * repulsive[
                row,
                slot,
            ]
        )

        edge_attractive.append(
            contact
            * attractive
        )

    states = hs._maximal_states(
        edge_atoms,
        sim.types_numpy,
    )

    local_symbols = [
        symbol_lookup()[
            int(
                sim.types_numpy[
                    atom
                ]
            )
        ]
        for atom in range(
            sim.per_box
        )
    ]

    local_positions = positions[
        :sim.per_box
    ]

    diagonals = []
    covalent_parts = []
    wall_parts = []
    wall_breakdowns = []

    for state in states:
        weights = (
            _descriptor_weights_for_state(
                box=0,
                per_box=sim.per_box,
                types_numpy=sim.types_numpy,
                neighbours=values[
                    "neighbours"
                ],
                neighbour_mask=(
                    sim.neighbour_mask
                ),
                taper=taper,
                edge_atoms=edge_atoms,
                edge_tapers=edge_tapers,
                state=state,
            )
        )

        fragment = (
            nb.ContinuousTorchFragment(
                symbols=local_symbols,
                positions=local_positions,
                bond_weights=weights,
            )
        )

        if state:
            covalent = torch.stack(
                [
                    edge_repulsive[
                        index
                    ]
                    - edge_attractive[
                        index
                    ]
                    for index in state
                ]
            ).sum()
        else:
            covalent = zero

        occupied = set(
            state
        )

        wall = zero
        breakdown = {}

        for edge_index, (
            first,
            second,
        ) in enumerate(
            edge_atoms
        ):
            if edge_index in occupied:
                continue

            contribution = (
                edge_tapers[
                    edge_index
                ]
                * _sapt_pair_energy(
                    fragment,
                    first,
                    second,
                )
            )

            wall = (
                wall
                + contribution
            )

            breakdown[
                edge_index
            ] = contribution

        diagonals.append(
            covalent
            + wall
        )

        covalent_parts.append(
            covalent
        )

        wall_parts.append(
            wall
        )

        wall_breakdowns.append(
            breakdown
        )

    diagonal = torch.stack(
        diagonals
    )

    weighted_degree = [
        zero
        for _ in states
    ]

    transitions = {}

    for first in range(
        len(states)
    ):
        for second in range(
            first + 1,
            len(states),
        ):
            transition = (
                hs._single_h_transfer(
                    states[
                        first
                    ],
                    states[
                        second
                    ],
                    edge_atoms,
                    sim.types_numpy,
                )
            )

            if transition is None:
                continue

            (
                old_index,
                new_index,
                _,
            ) = transition

            overlap = (
                hs._contact_overlap(
                    edge_tapers[
                        old_index
                    ],
                    edge_tapers[
                        new_index
                    ],
                )
            )

            transitions[
                (
                    first,
                    second,
                )
            ] = (
                old_index,
                new_index,
                overlap,
            )

            weighted_degree[
                first
            ] = (
                weighted_degree[
                    first
                ]
                + overlap * overlap
            )

            weighted_degree[
                second
            ] = (
                weighted_degree[
                    second
                ]
                + overlap * overlap
            )

    normalisation = torch.stack(
        [
            hs._crowding_normalisation(
                value
            )
            for value in weighted_degree
        ]
    )

    couplings = {}

    for (
        first,
        second,
    ), (
        old_index,
        new_index,
        overlap,
    ) in transitions.items():
        depth_scale = torch.sqrt(
            torch.clamp(
                edge_depths[
                    old_index
                ]
                * edge_depths[
                    new_index
                ],
                min=1.0e-12,
            )
        )

        denominator = torch.sqrt(
            torch.clamp(
                normalisation[
                    first
                ]
                * normalisation[
                    second
                ],
                min=1.0e-12,
            )
        )

        coupling = (
            float(
                eta
            )
            * depth_scale
            * overlap
            / denominator
        )

        couplings[
            (
                first,
                second,
            )
        ] = coupling

    rows = []

    for first in range(
        len(states)
    ):
        row = []

        for second in range(
            len(states)
        ):
            if first == second:
                value = diagonal[
                    first
                ]
            else:
                key = (
                    min(
                        first,
                        second,
                    ),
                    max(
                        first,
                        second,
                    ),
                )

                value = (
                    -couplings[
                        key
                    ]
                    if key in couplings
                    else zero
                )

            row.append(
                value
            )

        rows.append(
            torch.stack(
                row
            )
        )

    hamiltonian = torch.stack(
        rows
    )

    eigenvalues = torch.linalg.eigvalsh(
        hamiltonian
    )

    return {
        "states": states,
        "diagonal": diagonal,
        "covalent": torch.stack(
            covalent_parts
        ),
        "wall": torch.stack(
            wall_parts
        ),
        "wall_breakdowns": wall_breakdowns,
        "couplings": couplings,
        "eigenvalues": eigenvalues,
    }


def scalar(
    value,
):
    if torch.is_tensor(
        value
    ):
        return float(
            value
            .detach()
            .cpu()
        )

    return float(
        value
    )


def coupling_between(
    matrix,
    first,
    second,
):
    key = (
        min(
            first,
            second,
        ),
        max(
            first,
            second,
        ),
    )

    if key not in matrix[
        "couplings"
    ]:
        raise RuntimeError(
            "reactant/product states are not directly coupled"
        )

    return abs(
        scalar(
            matrix[
                "couplings"
            ][
                key
            ]
        )
    )


def edge_label(
    edge_atoms,
    symbols,
    edge_index,
):
    first, second = edge_atoms[
        edge_index
    ]

    return (
        f"{symbols[first]}{first}-"
        f"{symbols[second]}{second}"
    )


def state_label(
    state,
    edge_atoms,
    symbols,
):
    return (
        "{"
        + ", ".join(
            edge_label(
                edge_atoms,
                symbols,
                edge_index,
            )
            for edge_index in state
        )
        + "}"
    )


def decompose(
    label,
    symbols,
    raw_positions,
    *,
    eta,
    donor_target=None,
    transfer_target=None,
    h3=False,
):
    sim = build_simulation(
        symbols,
        raw_positions,
        eta=eta,
    )

    (
        positions,
        values,
        edge_atoms,
        edge_rows,
        edge_slots,
    ) = prepare_intermediates(
        sim
    )

    if h3:
        donor_edge, product_edge = (
            identify_h3_edges(
                edge_atoms
            )
        )
    else:
        (
            donor_edge,
            product_edge,
        ) = identify_molecular_transfer_edges(
            edge_atoms,
            symbols,
            positions,
            donor_target=float(
                donor_target
            ),
            transfer_target=float(
                transfer_target
            ),
        )

    matrix = build_state_matrix(
        sim,
        positions,
        values,
        edge_atoms,
        edge_rows,
        edge_slots,
        eta=eta,
    )

    (
        reactant_state,
        product_state,
    ) = classify_states(
        matrix[
            "states"
        ],
        donor_edge,
        product_edge,
    )

    reactant_diag = scalar(
        matrix[
            "diagonal"
        ][
            reactant_state
        ]
    )

    product_diag = scalar(
        matrix[
            "diagonal"
        ][
            product_state
        ]
    )

    coupling = coupling_between(
        matrix,
        reactant_state,
        product_state,
    )

    lowest = scalar(
        matrix[
            "eigenvalues"
        ][
            0
        ]
    )

    lower_diagonal = min(
        reactant_diag,
        product_diag,
    )

    reactant_breakdown = matrix[
        "wall_breakdowns"
    ][
        reactant_state
    ]

    product_breakdown = matrix[
        "wall_breakdowns"
    ][
        product_state
    ]

    reactant_competing_wall = scalar(
        reactant_breakdown.get(
            product_edge,
            torch.tensor(
                0.0,
                dtype=DTYPE,
            ),
        )
    )

    product_competing_wall = scalar(
        product_breakdown.get(
            donor_edge,
            torch.tensor(
                0.0,
                dtype=DTYPE,
            ),
        )
    )

    signed_gap = (
        product_diag
        - reactant_diag
    )

    return {
        "label": label,
        "eta": float(
            eta
        ),
        "donor_edge": (
            edge_label(
                edge_atoms,
                symbols,
                donor_edge,
            )
        ),
        "product_edge": (
            edge_label(
                edge_atoms,
                symbols,
                product_edge,
            )
        ),
        "reactant_state_label": (
            state_label(
                matrix[
                    "states"
                ][
                    reactant_state
                ],
                edge_atoms,
                symbols,
            )
        ),
        "product_state_label": (
            state_label(
                matrix[
                    "states"
                ][
                    product_state
                ],
                edge_atoms,
                symbols,
            )
        ),
        "reactant_diag": reactant_diag,
        "product_diag": product_diag,
        "signed_gap": signed_gap,
        "abs_gap": abs(
            signed_gap
        ),
        "coupling": coupling,
        "coupling_prefactor": (
            coupling
            / float(
                eta
            )
            if eta != 0.0
            else np.nan
        ),
        "mixing_lowering": (
            lowest
            - lower_diagonal
        ),
        "reactant_covalent": scalar(
            matrix[
                "covalent"
            ][
                reactant_state
            ]
        ),
        "product_covalent": scalar(
            matrix[
                "covalent"
            ][
                product_state
            ]
        ),
        "reactant_wall": scalar(
            matrix[
                "wall"
            ][
                reactant_state
            ]
        ),
        "product_wall": scalar(
            matrix[
                "wall"
            ][
                product_state
            ]
        ),
        "reactant_competing_wall": (
            reactant_competing_wall
        ),
        "product_competing_wall": (
            product_competing_wall
        ),
        "lowest_component": lowest,
    }


def h3_case():
    hydrogen = int(
        R.ELEMENT_INDEX[
            "H"
        ]
    )

    re = float(
        R.BOND_LENGTH[
            hydrogen,
            hydrogen,
        ]
    )

    # Current calibrated symmetric minimum from the coupling-requirement
    # diagnostic.
    radius = 0.935

    symbols = [
        "H",
        "H",
        "H",
    ]

    positions = np.array(
        [
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                radius,
                0.0,
                0.0,
            ],
            [
                2.0 * radius,
                0.0,
                0.0,
            ],
        ],
        dtype=float,
    )

    return decompose(
        "H3",
        symbols,
        positions,
        eta=SAPT_H_STATE_MIXING,
        h3=True,
    )


def molecular_case(
    system,
    row,
):
    eta, donor, transfer = (
        choose_eta_and_saddle(
            system,
            row,
        )
    )

    scan.apply_system(
        system
    )

    spectators = frozen_spectators(
        system
    )

    geometry_builder = scan.SYSTEMS[
        system
    ][
        "geometry"
    ]

    symbols, positions = (
        geometry_builder(
            donor,
            transfer,
            spectators,
        )
    )

    result = decompose(
        system,
        symbols,
        positions,
        eta=eta,
        donor_target=donor,
        transfer_target=transfer,
        h3=False,
    )

    result[
        "donor_distance"
    ] = donor

    result[
        "transfer_distance"
    ] = transfer

    return result


def show_case(
    result,
):
    title = result[
        "label"
    ].upper()

    print(
        title
    )

    print(
        "=" * len(
            title
        )
    )

    print(
        f"eta                       {result['eta']:.9f}"
    )

    if (
        "donor_distance"
        in result
    ):
        print(
            f"saved saddle donor        "
            f"{result['donor_distance']:.5f} A"
        )

        print(
            f"saved saddle transfer     "
            f"{result['transfer_distance']:.5f} A"
        )

    print(
        f"reactant edge             {result['donor_edge']}"
    )

    print(
        f"product edge              {result['product_edge']}"
    )

    print(
        f"reactant state            {result['reactant_state_label']}"
    )

    print(
        f"product state             {result['product_state_label']}"
    )

    print(
        f"reactant diagonal         "
        f"{result['reactant_diag']:+.9f} eV"
    )

    print(
        f"product diagonal          "
        f"{result['product_diag']:+.9f} eV"
    )

    print(
        f"signed gap P-R            "
        f"{result['signed_gap']:+.9f} eV"
    )

    print(
        f"|gap|                     "
        f"{result['abs_gap']:.9f} eV"
    )

    print(
        f"coupling V                "
        f"{result['coupling']:.9f} eV"
    )

    print(
        f"coupling prefactor V/eta  "
        f"{result['coupling_prefactor']:.9f} eV"
    )

    print(
        f"mixing lowering           "
        f"{result['mixing_lowering']:+.9f} eV"
    )

    print(
        f"reactant covalent         "
        f"{result['reactant_covalent']:+.9f} eV"
    )

    print(
        f"reactant SAPT wall        "
        f"{result['reactant_wall']:+.9f} eV"
    )

    print(
        f"reactant competing wall   "
        f"{result['reactant_competing_wall']:+.9f} eV"
    )

    print(
        f"product covalent          "
        f"{result['product_covalent']:+.9f} eV"
    )

    print(
        f"product SAPT wall         "
        f"{result['product_wall']:+.9f} eV"
    )

    print(
        f"product competing wall    "
        f"{result['product_competing_wall']:+.9f} eV"
    )

    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--requirements",
        default=DEFAULT_CSV,
    )

    args = parser.parse_args()

    requirement_rows = (
        load_requirement_rows(
            args.requirements
        )
    )

    required_systems = (
        "formaldehyde",
        "water",
        "methane",
    )

    missing = [
        system
        for system in required_systems
        if system not in requirement_rows
    ]

    if missing:
        raise RuntimeError(
            "missing successful rows in requirements CSV: "
            + ", ".join(
                missing
            )
        )

    results = [
        h3_case(),
    ]

    for system in required_systems:
        results.append(
            molecular_case(
                system,
                requirement_rows[
                    system
                ],
            )
        )

    print(
        "SAPT STATE / COUPLING DECOMPOSITION"
    )

    print(
        "==================================="
    )

    print(
        f"requirements source: {args.requirements}"
    )

    print()

    for result in results:
        show_case(
            result
        )

    print(
        "COMPACT COMPARISON"
    )

    print(
        "=================="
    )

    print(
        "system           eta      |gap|       V      V/eta   "
        "mix lower   R wall   P wall"
    )

    for result in results:
        print(
            f"{result['label']:>12}  "
            f"{result['eta']:8.4f}  "
            f"{result['abs_gap']:8.3f}  "
            f"{result['coupling']:7.3f}  "
            f"{result['coupling_prefactor']:7.3f}  "
            f"{result['mixing_lowering']:9.3f}  "
            f"{result['reactant_wall']:7.3f}  "
            f"{result['product_wall']:7.3f}"
        )

    print()

    print(
        "Read this table as follows:"
    )

    print(
        "  - similar V/eta but different required eta means the current "
        "geometric/depth prefactor is not explaining the reaction dependence;"
    )

    print(
        "  - large asymmetric R/P walls and large gaps indicate eta is "
        "compensating for state-diagonal placement;"
    )

    print(
        "  - similar required actual V across systems would instead point "
        "toward a coupling-normalisation problem."
    )


if __name__ == "__main__":
    main()
