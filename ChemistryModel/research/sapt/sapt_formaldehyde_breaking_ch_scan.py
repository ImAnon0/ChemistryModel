"""
Formaldehyde breaking-C-H diabatic diagnostic.

This is deliberately NOT a fit.

Starting from the relaxed SAPT formaldehyde saddle saved by
sapt_full_formaldehyde_scan.py, vary only the breaking donor C-H distance while
holding the forming H-H distance and spectator coordinates fixed.

For each geometry, report:
    - reactant-like and product-like SAPT diabatic diagonals
    - covalent and total SAPT-wall pieces for each state
    - the reactant state's unoccupied H-H wall
    - the product state's unoccupied breaking-C-H wall
    - diabatic gap
    - off-diagonal coupling
    - lowest H-state eigenvalue
    - full ChemistryModel + SAPT/H-state energy

This lets us see exactly where the product-like state is lifted by treating the
still-close breaking C-H pair as a pure unoccupied SAPT contact.

No parameter is modified or fitted.

Default source:
    sapt_full_formaldehyde_relaxed_d0p04000_t0p04000.npz

Usage:
    py sapt_formaldehyde_breaking_ch_scan.py

Finer scan:
    py sapt_formaldehyde_breaking_ch_scan.py --step 0.01

Custom range:
    py sapt_formaldehyde_breaking_ch_scan.py --min 1.05 --max 2.00 --step 0.02
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import reactive as R
import hf_surface_scan as scan
import nonbonded_continuous_torch as nb

from batched_torch import BatchedReactiveSimulation
from sapt_h_state_torch import (
    SAPT_H_STATE_MIXING,
    SaptHStateBatchedSimulation,
    _descriptor_weights_for_state,
    _sapt_pair_energy,
)

# Reuse the already-tested state-matrix diagnostic helpers rather than
# duplicating its H-state Hamiltonian reconstruction.
from sapt_formaldehyde_state_matrix import (
    build_simulation,
    prepare_intermediates,
    sapt_matrix,
)


DEFAULT_NPZ = (
    "sapt_full_formaldehyde_relaxed_d0p04000_t0p04000.npz"
)

DEFAULT_MIN = 1.05
DEFAULT_MAX = 2.00
DEFAULT_STEP = 0.02


def load_reference_saddle(path):
    data = np.load(
        Path(path),
        allow_pickle=False,
    )

    cell = tuple(
        int(value)
        for value in data["saddle_cell"]
    )

    donor = float(
        data["donor_lengths"][cell[0]]
    )

    transfer = float(
        data["transfer_lengths"][cell[1]]
    )

    spectators = np.asarray(
        data["spectators"][cell],
        dtype=float,
    )

    return donor, transfer, spectators


def edge_key(pair):
    return frozenset(
        int(value)
        for value in pair
    )


def classify_transfer_edges(
    edge_atoms,
    symbols,
):
    """
    Identify:
      forming H-H edge
      donor C-H edge sharing the transferred H with that H-H edge
      spectator C-H edge(s)

    Returns None for an edge that is outside the current active candidate set.
    """

    hh_index = None

    for index, (first, second) in enumerate(
        edge_atoms
    ):
        if (
            symbols[first] == "H"
            and symbols[second] == "H"
        ):
            hh_index = index
            break

    if hh_index is None:
        return {
            "hh": None,
            "donor_ch": None,
            "spectator_ch": [],
            "transferred_h": None,
        }

    hh_atoms = set(
        edge_atoms[hh_index]
    )

    ch_edges = []

    for index, (first, second) in enumerate(
        edge_atoms
    ):
        pair_symbols = {
            symbols[first],
            symbols[second],
        }

        if pair_symbols == {
            "C",
            "H",
        }:
            ch_edges.append(
                (
                    index,
                    first,
                    second,
                )
            )

    donor_ch = None
    transferred_h = None
    spectator_ch = []

    for index, first, second in ch_edges:
        h_atom = (
            first
            if symbols[first] == "H"
            else second
        )

        if h_atom in hh_atoms:
            donor_ch = index
            transferred_h = h_atom
        else:
            spectator_ch.append(
                index
            )

    return {
        "hh": hh_index,
        "donor_ch": donor_ch,
        "spectator_ch": spectator_ch,
        "transferred_h": transferred_h,
    }


def classify_states(
    states,
    transfer_edges,
):
    donor_edge = transfer_edges[
        "donor_ch"
    ]

    hh_edge = transfer_edges[
        "hh"
    ]

    reactant_state = None
    product_state = None

    for index, state in enumerate(
        states
    ):
        occupied = set(
            state
        )

        if (
            donor_edge is not None
            and donor_edge in occupied
            and (
                hh_edge is None
                or hh_edge not in occupied
            )
        ):
            reactant_state = index

        if (
            hh_edge is not None
            and hh_edge in occupied
            and (
                donor_edge is None
                or donor_edge not in occupied
            )
        ):
            product_state = index

    return (
        reactant_state,
        product_state,
    )


def edge_tapers_and_depths(
    values,
    edge_rows,
    edge_slots,
):
    tapers = []
    depths = []

    for row, slot in zip(
        edge_rows,
        edge_slots,
    ):
        tapers.append(
            values["taper"][row, slot]
        )

        depths.append(
            values["pair_depth"][row, slot]
        )

    return tapers, depths


def wall_breakdown_for_state(
    sim,
    positions,
    values,
    edge_atoms,
    edge_rows,
    edge_slots,
    state,
):
    """
    Return each unoccupied candidate edge's SAPT-wall contribution for one
    diabatic state.
    """

    taper = values[
        "taper"
    ]

    edge_tapers, _ = (
        edge_tapers_and_depths(
            values,
            edge_rows,
            edge_slots,
        )
    )

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
            sim.per_box
        )
    ]

    fragment = (
        nb.ContinuousTorchFragment(
            symbols=local_symbols,
            positions=positions[
                :sim.per_box
            ],
            bond_weights=weights,
        )
    )

    occupied = set(
        state
    )

    contributions = {}

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

        contributions[
            edge_index
        ] = float(
            contribution
            .detach()
            .cpu()
        )

    return contributions


def coupling_between(
    matrix,
    first,
    second,
):
    if (
        first is None
        or second is None
    ):
        return np.nan

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

    value = matrix[
        "couplings"
    ].get(
        key
    )

    if value is None:
        return np.nan

    return abs(
        float(
            value
            .detach()
            .cpu()
        )
    )


def scalar_from_state(
    tensor,
    state_index,
):
    if state_index is None:
        return np.nan

    return float(
        tensor[
            state_index
        ]
        .detach()
        .cpu()
    )


def wall_for_edge(
    breakdown,
    edge_index,
):
    if edge_index is None:
        return np.nan

    return float(
        breakdown.get(
            edge_index,
            0.0,
        )
    )


def full_energy_for_geometry(
    symbols,
    positions,
):
    sim = SaptHStateBatchedSimulation(
        boxes=[
            (
                symbols,
                positions + scan.CENTRE,
            )
        ],
        box_size=scan.BOX,
        random_seed=0,
        relax_on_start=False,
        device="cpu",
        dtype=torch.float64,
        h_state_mixing=(
            SAPT_H_STATE_MIXING
        ),
    )

    return float(
        sim.potential_per_box[
            0
        ]
    )


def evaluate_point(
    donor,
    transfer,
    spectators,
):
    symbols, raw_positions = (
        scan.formaldehyde_geometry(
            donor,
            transfer,
            spectators,
        )
    )

    full_energy = (
        full_energy_for_geometry(
            symbols,
            raw_positions,
        )
    )

    sim = build_simulation(
        symbols,
        raw_positions,
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

    matrix = sapt_matrix(
        sim,
        positions,
        values,
        edge_atoms,
        edge_rows,
        edge_slots,
    )

    transfer_edges = (
        classify_transfer_edges(
            edge_atoms,
            symbols,
        )
    )

    (
        reactant_state,
        product_state,
    ) = classify_states(
        matrix[
            "states"
        ],
        transfer_edges,
    )

    reactant_breakdown = (
        {}
        if reactant_state is None
        else wall_breakdown_for_state(
            sim,
            positions,
            values,
            edge_atoms,
            edge_rows,
            edge_slots,
            matrix[
                "states"
            ][
                reactant_state
            ],
        )
    )

    product_breakdown = (
        {}
        if product_state is None
        else wall_breakdown_for_state(
            sim,
            positions,
            values,
            edge_atoms,
            edge_rows,
            edge_slots,
            matrix[
                "states"
            ][
                product_state
            ],
        )
    )

    reactant_diag = (
        scalar_from_state(
            matrix[
                "diagonal"
            ],
            reactant_state,
        )
    )

    product_diag = (
        scalar_from_state(
            matrix[
                "diagonal"
            ],
            product_state,
        )
    )

    gap_signed = (
        product_diag
        - reactant_diag
        if (
            np.isfinite(
                product_diag
            )
            and np.isfinite(
                reactant_diag
            )
        )
        else np.nan
    )

    coupling = coupling_between(
        matrix,
        reactant_state,
        product_state,
    )

    eigenvalues = (
        matrix[
            "eigenvalues"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    lowest_component = float(
        eigenvalues[
            0
        ]
    )

    if (
        np.isfinite(
            reactant_diag
        )
        and np.isfinite(
            product_diag
        )
    ):
        mixing_lowering = (
            lowest_component
            - min(
                reactant_diag,
                product_diag,
            )
        )
    else:
        mixing_lowering = np.nan

    return {
        "donor_ch_A": donor,
        "forming_hh_A": transfer,
        "reactant_diag_eV": reactant_diag,
        "product_diag_eV": product_diag,
        "signed_gap_product_minus_reactant_eV": (
            gap_signed
        ),
        "abs_gap_eV": (
            abs(
                gap_signed
            )
            if np.isfinite(
                gap_signed
            )
            else np.nan
        ),
        "coupling_eV": coupling,
        "reactant_covalent_eV": (
            scalar_from_state(
                matrix[
                    "covalent"
                ],
                reactant_state,
            )
        ),
        "reactant_wall_total_eV": (
            scalar_from_state(
                matrix[
                    "wall"
                ],
                reactant_state,
            )
        ),
        "reactant_unoccupied_hh_wall_eV": (
            wall_for_edge(
                reactant_breakdown,
                transfer_edges[
                    "hh"
                ],
            )
        ),
        "product_covalent_eV": (
            scalar_from_state(
                matrix[
                    "covalent"
                ],
                product_state,
            )
        ),
        "product_wall_total_eV": (
            scalar_from_state(
                matrix[
                    "wall"
                ],
                product_state,
            )
        ),
        "product_unoccupied_breaking_ch_wall_eV": (
            wall_for_edge(
                product_breakdown,
                transfer_edges[
                    "donor_ch"
                ],
            )
        ),
        "lowest_h_state_component_eV": (
            lowest_component
        ),
        "mixing_lowering_from_lower_diagonal_eV": (
            mixing_lowering
        ),
        "full_energy_eV": full_energy,
        "state_count": len(
            matrix[
                "states"
            ]
        ),
        "candidate_edge_count": len(
            edge_atoms
        ),
        "donor_edge_active": int(
            transfer_edges[
                "donor_ch"
            ]
            is not None
        ),
    }


def write_csv(
    path,
    rows,
):
    fieldnames = list(
        rows[
            0
        ].keys()
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def save_npz(
    path,
    rows,
    spectators,
):
    columns = {}

    for name in rows[
        0
    ]:
        columns[
            name
        ] = np.asarray(
            [
                row[
                    name
                ]
                for row in rows
            ]
        )

    np.savez_compressed(
        path,
        spectators=np.asarray(
            spectators,
            dtype=float,
        ),
        mixing=np.asarray(
            SAPT_H_STATE_MIXING,
            dtype=float,
        ),
        **columns,
    )


def save_plot(
    path,
    rows,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            f"plot skipped: {exc}"
        )
        return

    donor = np.asarray(
        [
            row[
                "donor_ch_A"
            ]
            for row in rows
        ]
    )

    reactant = np.asarray(
        [
            row[
                "reactant_diag_eV"
            ]
            for row in rows
        ]
    )

    product = np.asarray(
        [
            row[
                "product_diag_eV"
            ]
            for row in rows
        ]
    )

    full = np.asarray(
        [
            row[
                "full_energy_eV"
            ]
            for row in rows
        ]
    )

    ch_wall = np.asarray(
        [
            row[
                "product_unoccupied_breaking_ch_wall_eV"
            ]
            for row in rows
        ]
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            8,
            8,
        ),
        sharex=True,
    )

    axes[
        0
    ].plot(
        donor,
        reactant,
        label="reactant-like diagonal",
    )

    axes[
        0
    ].plot(
        donor,
        product,
        label="product-like diagonal",
    )

    axes[
        0
    ].plot(
        donor,
        full,
        label="full mixed energy",
    )

    axes[
        0
    ].set_ylabel(
        "Energy / eV"
    )

    axes[
        0
    ].legend()

    axes[
        1
    ].plot(
        donor,
        ch_wall,
        label="product-state unoccupied C-H SAPT wall",
    )

    axes[
        1
    ].set_xlabel(
        "Breaking donor C-H / A"
    )

    axes[
        1
    ].set_ylabel(
        "SAPT wall / eV"
    )

    axes[
        1
    ].legend()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=160,
    )

    plt.close(
        fig
    )


def print_summary(
    rows,
    reference_donor,
):
    print()

    print(
        "SUMMARY"
    )

    print(
        "======="
    )

    valid = [
        row
        for row in rows
        if np.isfinite(
            row[
                "abs_gap_eV"
            ]
        )
    ]

    if valid:
        closest = min(
            valid,
            key=lambda row: (
                row[
                    "abs_gap_eV"
                ]
            ),
        )

        print(
            "smallest diabatic gap:"
        )

        print(
            f"  C-H       {closest['donor_ch_A']:.4f} A"
        )

        print(
            f"  gap       {closest['abs_gap_eV']:.6f} eV"
        )

        print(
            f"  coupling  {closest['coupling_eV']:.6f} eV"
        )

        print(
            "  product-state unoccupied C-H wall "
            f"{closest['product_unoccupied_breaking_ch_wall_eV']:.6f} eV"
        )

    nearest = min(
        rows,
        key=lambda row: abs(
            row[
                "donor_ch_A"
            ]
            - reference_donor
        ),
    )

    print()

    print(
        "saved-saddle neighbourhood:"
    )

    print(
        f"  C-H       {nearest['donor_ch_A']:.4f} A"
    )

    print(
        f"  reactant diagonal {nearest['reactant_diag_eV']:+.6f} eV"
    )

    print(
        f"  product diagonal  {nearest['product_diag_eV']:+.6f} eV"
    )

    print(
        f"  gap               {nearest['abs_gap_eV']:.6f} eV"
    )

    print(
        f"  coupling          {nearest['coupling_eV']:.6f} eV"
    )

    print(
        "  product-state unoccupied C-H wall "
        f"{nearest['product_unoccupied_breaking_ch_wall_eV']:.6f} eV"
    )

    active = [
        row
        for row in rows
        if row[
            "donor_edge_active"
        ]
    ]

    inactive = [
        row
        for row in rows
        if not row[
            "donor_edge_active"
        ]
    ]

    if (
        active
        and inactive
    ):
        print()

        print(
            "reactive candidate cutoff transition:"
        )

        print(
            f"  last active donor edge  {active[-1]['donor_ch_A']:.4f} A"
        )

        print(
            f"  first inactive edge     {inactive[0]['donor_ch_A']:.4f} A"
        )

        print(
            "  beyond this point the current H-state machinery no longer "
            "contains the breaking C-H edge as a candidate."
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--npz",
        default=DEFAULT_NPZ,
    )

    parser.add_argument(
        "--min",
        dest="minimum",
        type=float,
        default=DEFAULT_MIN,
    )

    parser.add_argument(
        "--max",
        dest="maximum",
        type=float,
        default=DEFAULT_MAX,
    )

    parser.add_argument(
        "--step",
        type=float,
        default=DEFAULT_STEP,
    )

    parser.add_argument(
        "--transfer",
        type=float,
        default=None,
        help=(
            "forming H-H distance; default is the saved relaxed saddle value"
        ),
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
    )

    args = parser.parse_args()

    if args.step <= 0:
        raise ValueError(
            "--step must be positive"
        )

    if args.maximum < args.minimum:
        raise ValueError(
            "--max must be >= --min"
        )

    scan.apply_system(
        "formaldehyde"
    )

    (
        reference_donor,
        reference_transfer,
        spectators,
    ) = load_reference_saddle(
        args.npz
    )

    transfer = (
        reference_transfer
        if args.transfer is None
        else float(
            args.transfer
        )
    )

    donor_values = np.arange(
        args.minimum,
        args.maximum
        + 0.5
        * args.step,
        args.step,
    )

    print(
        "SAPT FORMALDEHYDE BREAKING C-H SCAN"
    )

    print(
        "==================================="
    )

    print(
        f"source saddle: {args.npz}"
    )

    print(
        f"saved saddle donor C-H: {reference_donor:.6f} A"
    )

    print(
        f"fixed forming H-H:      {transfer:.6f} A"
    )

    print(
        "fixed spectators: "
        + np.array2string(
            spectators,
            precision=6,
        )
    )

    print(
        f"H-state mixing:         {SAPT_H_STATE_MIXING:.9f}"
    )

    print(
        f"scan: {donor_values[0]:.3f} to {donor_values[-1]:.3f} A "
        f"in {args.step:.3f} A steps "
        f"({len(donor_values)} points)"
    )

    print()

    rows = []

    for index, donor in enumerate(
        donor_values,
        start=1,
    ):
        row = evaluate_point(
            float(
                donor
            ),
            transfer,
            spectators,
        )

        rows.append(
            row
        )

        gap = row[
            "abs_gap_eV"
        ]

        coupling = row[
            "coupling_eV"
        ]

        ch_wall = row[
            "product_unoccupied_breaking_ch_wall_eV"
        ]

        def fmt(value):
            return (
                f"{value:8.4f}"
                if np.isfinite(
                    value
                )
                else "     nan"
            )

        print(
            f"[{index:02d}/{len(donor_values):02d}] "
            f"C-H {donor:5.3f}  "
            f"gap {fmt(gap)}  "
            f"V {fmt(coupling)}  "
            f"product C-H wall {fmt(ch_wall)}  "
            f"full E {row['full_energy_eV']:+.6f}"
        )

    stem = (
        "sapt_formaldehyde_breaking_ch_scan"
    )

    csv_path = (
        Path(
            stem + ".csv"
        )
    )

    npz_path = (
        Path(
            stem + ".npz"
        )
    )

    plot_path = (
        Path(
            stem + ".png"
        )
    )

    write_csv(
        csv_path,
        rows,
    )

    save_npz(
        npz_path,
        rows,
        spectators,
    )

    if not args.no_plot:
        save_plot(
            plot_path,
            rows,
        )

    print_summary(
        rows,
        reference_donor,
    )

    print()

    print(
        f"saved CSV:  {csv_path}"
    )

    print(
        f"saved NPZ:  {npz_path}"
    )

    if not args.no_plot:
        print(
            f"saved plot: {plot_path}"
        )


if __name__ == "__main__":
    main()
