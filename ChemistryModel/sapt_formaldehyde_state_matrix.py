"""
Inspect the actual diabatic H-state matrix at the saved relaxed
formaldehyde saddle.

No parameter is fitted or modified.

This diagnostic answers the question left by
sapt_formaldehyde_crossing_diagnostic.py:

    Why does SAPT + eta=0.534590721 lower the formaldehyde saddle
    much less than the old common-core H-state model?

It prints, for the exact saved saddle geometry:
    - enumerated H-valence states
    - old common-core state diagonals
    - SAPT-wall state diagonals
    - SAPT covalent and wall pieces per state
    - pairwise couplings
    - diabatic gaps
    - lowest eigenvalues

Usage:
    py sapt_formaldehyde_state_matrix.py

or:
    py sapt_formaldehyde_state_matrix.py --npz <relaxed scan npz>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import reactive as R
import hf_surface_scan as scan
import h_state_torch as hs
import nonbonded_continuous_torch as nb

from batched_torch import BatchedReactiveSimulation
from sapt_h_state_torch import (
    SAPT_H_STATE_MIXING,
    SaptHStateBatchedSimulation,
    _descriptor_weights_for_state,
    _sapt_pair_energy,
)


DEFAULT_NPZ = (
    "sapt_full_formaldehyde_relaxed_d0p04000_t0p04000.npz"
)

DTYPE = torch.float64
DEVICE = "cpu"


def load_saddle(path):
    data = np.load(
        Path(path),
        allow_pickle=False,
    )

    cell = tuple(
        int(value)
        for value in data[
            "saddle_cell"
        ]
    )

    donor = float(
        data[
            "donor_lengths"
        ][
            cell[0]
        ]
    )

    transfer = float(
        data[
            "transfer_lengths"
        ][
            cell[1]
        ]
    )

    spectators = np.asarray(
        data[
            "spectators"
        ][
            cell
        ],
        dtype=float,
    )

    symbols, positions = (
        scan.formaldehyde_geometry(
            donor,
            transfer,
            spectators,
        )
    )

    return (
        donor,
        transfer,
        spectators,
        symbols,
        positions,
    )


def build_simulation(
    symbols,
    positions,
):
    return SaptHStateBatchedSimulation(
        boxes=[
            (
                symbols,
                positions + scan.CENTRE,
            )
        ],
        box_size=scan.BOX,
        random_seed=0,
        relax_on_start=False,
        device=DEVICE,
        dtype=DTYPE,
        h_state_mixing=SAPT_H_STATE_MIXING,
    )


def prepare_intermediates(sim):
    """
    Evaluate only the ordinary reactive base method so its shared pair
    intermediates remain available instead of being cleared by the H-state
    adapter's finally block.
    """

    positions = (
        sim.positions
        .detach()
        .requires_grad_(True)
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
            "reactive intermediates were not exposed"
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
            > 1e-12
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


def state_label(
    state,
    edge_atoms,
    symbols,
):
    pieces = []

    for edge_index in state:
        a, b = edge_atoms[
            edge_index
        ]

        pieces.append(
            f"{symbols[a]}{a}-{symbols[b]}{b}"
        )

    if not pieces:
        return "{}"

    return (
        "{"
        + ", ".join(
            pieces
        )
        + "}"
    )


def old_matrix(
    sim,
    values,
    edge_atoms,
    edge_rows,
    edge_slots,
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

    zero = taper.sum() * 0.0

    common_core = torch.stack(
        edge_repulsive
    ).sum()

    diagonals = []

    for state in states:
        if state:
            attraction = torch.stack(
                [
                    edge_attractive[index]
                    for index in state
                ]
            ).sum()
        else:
            attraction = zero

        diagonals.append(
            common_core
            - attraction
        )

    diagonal = torch.stack(
        diagonals
    )

    return build_hamiltonian(
        sim,
        states,
        edge_atoms,
        edge_tapers,
        edge_depths,
        diagonal,
        mixing=float(
            hs.H_STATE_MIXING
        ),
    )


def sapt_matrix(
    sim,
    positions,
    values,
    edge_atoms,
    edge_rows,
    edge_slots,
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

    start = 0
    stop = sim.per_box

    local_positions = positions[
        start:stop
    ]

    symbol_for = {
        int(index): symbol
        for symbol, index
        in R.ELEMENT_INDEX.items()
    }

    local_symbols = [
        symbol_for[
            int(
                sim.types_numpy[
                    atom
                ]
            )
        ]
        for atom in range(
            start,
            stop,
        )
    ]

    diagonals = []
    covalent_parts = []
    wall_parts = []

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
                    edge_repulsive[index]
                    - edge_attractive[index]
                    for index in state
                ]
            ).sum()
        else:
            covalent = zero

        occupied = set(
            state
        )

        wall = zero

        for edge_index, (
            global_a,
            global_b,
        ) in enumerate(
            edge_atoms
        ):
            if edge_index in occupied:
                continue

            wall = (
                wall
                + edge_tapers[
                    edge_index
                ]
                * _sapt_pair_energy(
                    fragment,
                    global_a - start,
                    global_b - start,
                )
            )

        covalent_parts.append(
            covalent
        )

        wall_parts.append(
            wall
        )

        diagonals.append(
            covalent
            + wall
        )

    diagonal = torch.stack(
        diagonals
    )

    result = build_hamiltonian(
        sim,
        states,
        edge_atoms,
        edge_tapers,
        edge_depths,
        diagonal,
        mixing=(
            SAPT_H_STATE_MIXING
        ),
    )

    result[
        "covalent"
    ] = torch.stack(
        covalent_parts
    )

    result[
        "wall"
    ] = torch.stack(
        wall_parts
    )

    return result


def build_hamiltonian(
    sim,
    states,
    edge_atoms,
    edge_tapers,
    edge_depths,
    diagonal,
    *,
    mixing,
):
    zero = diagonal.sum() * 0.0

    weighted_degree = [
        zero
        for _ in states
    ]

    transitions = {}

    for first in range(
        len(
            states
        )
    ):
        for second in range(
            first + 1,
            len(
                states
            ),
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
                + overlap
                * overlap
            )

            weighted_degree[
                second
            ] = (
                weighted_degree[
                    second
                ]
                + overlap
                * overlap
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

    rows = []

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
                min=1e-12,
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
                min=1e-12,
            )
        )

        coupling = (
            float(
                mixing
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

    for first in range(
        len(
            states
        )
    ):
        row = []

        for second in range(
            len(
                states
            )
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
                    if key
                    in couplings
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
        "couplings": couplings,
        "hamiltonian": hamiltonian,
        "eigenvalues": eigenvalues,
    }


def print_matrix_report(
    title,
    result,
    edge_atoms,
    symbols,
):
    print(
        title
    )

    print(
        "="
        * len(
            title
        )
    )

    for index, state in enumerate(
        result[
            "states"
        ]
    ):
        label = state_label(
            state,
            edge_atoms,
            symbols,
        )

        line = (
            f"state {index}: "
            f"{label:<30} "
            f"diagonal "
            f"{float(result['diagonal'][index]):+.9f} eV"
        )

        if "covalent" in result:
            line += (
                f"  covalent "
                f"{float(result['covalent'][index]):+.9f}"
                f"  wall "
                f"{float(result['wall'][index]):+.9f}"
            )

        print(
            line
        )

    print()

    if len(
        result[
            "diagonal"
        ]
    ) > 1:
        values = (
            result[
                "diagonal"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        print(
            "diabatic diagonal span: "
            f"{float(values.max()-values.min()):.9f} eV"
        )

    for (
        first,
        second,
    ), coupling in result[
        "couplings"
    ].items():
        print(
            f"coupling {first}<->{second}: "
            f"{float(coupling):.9f} eV"
        )

    print()

    print(
        "Hamiltonian / eV:"
    )

    print(
        np.array2string(
            result[
                "hamiltonian"
            ]
            .detach()
            .cpu()
            .numpy(),
            precision=9,
            suppress_small=False,
        )
    )

    print()

    print(
        "eigenvalues / eV:"
    )

    print(
        np.array2string(
            result[
                "eigenvalues"
            ]
            .detach()
            .cpu()
            .numpy(),
            precision=9,
        )
    )

    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--npz",
        default=DEFAULT_NPZ,
    )

    args = parser.parse_args()

    (
        donor,
        transfer,
        spectators,
        symbols,
        positions,
    ) = load_saddle(
        args.npz
    )

    scan.apply_system(
        "formaldehyde"
    )

    sim = build_simulation(
        symbols,
        positions,
    )

    (
        torch_positions,
        values,
        edge_atoms,
        edge_rows,
        edge_slots,
    ) = prepare_intermediates(
        sim
    )

    print(
        "SAPT FORMALDEHYDE STATE MATRIX"
    )

    print(
        "=============================="
    )

    print(
        f"source: {args.npz}"
    )

    print(
        f"saddle donor C-H:   {donor:.6f} A"
    )

    print(
        f"saddle forming H-H: {transfer:.6f} A"
    )

    print(
        "spectators: "
        + np.array2string(
            spectators,
            precision=6,
        )
    )

    print()

    print(
        "candidate H edges:"
    )

    for index, (
        first,
        second,
    ) in enumerate(
        edge_atoms
    ):
        print(
            f"  {index}: "
            f"{symbols[first]}{first}-{symbols[second]}{second}"
        )

    print()

    old = old_matrix(
        sim,
        values,
        edge_atoms,
        edge_rows,
        edge_slots,
    )

    sapt = sapt_matrix(
        sim,
        torch_positions,
        values,
        edge_atoms,
        edge_rows,
        edge_slots,
    )

    print_matrix_report(
        f"OLD COMMON-CORE  eta={hs.H_STATE_MIXING:.6f}",
        old,
        edge_atoms,
        symbols,
    )

    print_matrix_report(
        f"SAPT WALL  eta={SAPT_H_STATE_MIXING:.9f}",
        sapt,
        edge_atoms,
        symbols,
    )

    if (
        len(
            sapt[
                "diagonal"
            ]
        )
        == 2
    ):
        gap = abs(
            float(
                sapt[
                    "diagonal"
                ][0]
                - sapt[
                    "diagonal"
                ][1]
            )
        )

        coupling = abs(
            float(
                next(
                    iter(
                        sapt[
                            "couplings"
                        ].values()
                    )
                )
            )
        )

        print(
            "KEY RATIO"
        )

        print(
            "========="
        )

        print(
            f"|diabatic gap| = {gap:.9f} eV"
        )

        print(
            f"|coupling|      = {coupling:.9f} eV"
        )

        print(
            f"gap/coupling    = {gap/max(coupling, 1e-15):.6f}"
        )

        print()

        print(
            "A large gap at the geometric saddle means the two SAPT"
        )

        print(
            "diabatic states are not crossing where the molecular"
        )

        print(
            "transition state sits. In that case increasing one global"
        )

        print(
            "mixing constant would hide the state-alignment problem and"
        )

        print(
            "would no longer be an H3-only calibration."
        )


if __name__ == "__main__":
    main()
