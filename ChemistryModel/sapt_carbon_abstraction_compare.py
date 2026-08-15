"""
Compare methane vs formaldehyde diabatic state placement under the SAME
C-H -> H-H SAPT H-state architecture.

Diagnostic only. No parameters are fitted or modified.

For each system, at the current frozen SAPT model with the same global
H-state mixing eta = SAPT_H_STATE_MIXING, this script:

  1. rebuilds the frozen 2D surface using hf_surface_scan geometry/basin logic;
  2. reconstructs the four-neighbour minimax path;
  3. evaluates OLD common-core and current SAPT diabatic diagonals at every
     path cell where the intended C-H -> H-H state pair exists;
  4. reports at the actual SAPT barrier:
       - reactant-like old vs SAPT diagonal shift
       - product-like old vs SAPT diagonal shift
       - old and SAPT diabatic gaps
       - SAPT competing H-H and breaking C-H walls
       - coupling and V/eta
  5. reports the closest SAPT diabatic approach on the actual path.

The decisive question is:

    Does methane show the same several-eV product-state lift as formaldehyde?

If yes, the problem is generic to C-H -> H-H state construction.
If no, formaldehyde's carbonyl/radical environment is the special failure.

Requires the previously created:
    sapt_formaldehyde_state_matrix.py
    sapt_state_coupling_decomposition.py
    sapt_coupling_requirement_diagnostic.py

Usage:
    py sapt_carbon_abstraction_compare.py
"""

from __future__ import annotations

import argparse
import heapq
from pathlib import Path

import numpy as np
import torch

import hf_surface_scan as scan

from sapt_h_state_torch import (
    SAPT_H_STATE_MIXING,
)

from sapt_coupling_requirement_diagnostic import (
    evaluate_raw_boxes,
    inclusive_axis,
)

from sapt_state_coupling_decomposition import (
    frozen_spectators,
    build_simulation,
    prepare_intermediates,
    identify_molecular_transfer_edges,
    classify_states,
)

from sapt_formaldehyde_state_matrix import (
    old_matrix,
    sapt_matrix,
)


DEFAULT_SYSTEMS = (
    "formaldehyde",
    "methane",
)

DEFAULT_DONOR_STEP = 0.04
DEFAULT_TRANSFER_STEP = 0.04
DEFAULT_BATCH_SIZE = 512


def neighbours4(i, j, rows, cols):
    if i > 0:
        yield i - 1, j

    if i + 1 < rows:
        yield i + 1, j

    if j > 0:
        yield i, j - 1

    if j + 1 < cols:
        yield i, j + 1


def minimax_path(grid, start, goal):
    rows, cols = grid.shape

    costs = np.full(
        (rows, cols),
        np.inf,
        dtype=float,
    )

    previous = {}

    start = tuple(
        int(value)
        for value in start
    )

    goal = tuple(
        int(value)
        for value in goal
    )

    costs[start] = float(
        grid[start]
    )

    queue = [
        (
            costs[start],
            start,
        )
    ]

    visited = set()

    while queue:
        cost, cell = heapq.heappop(
            queue
        )

        if cell in visited:
            continue

        visited.add(
            cell
        )

        if cell == goal:
            break

        i, j = cell

        for nxt in neighbours4(
            i,
            j,
            rows,
            cols,
        ):
            new_cost = max(
                cost,
                float(
                    grid[nxt]
                ),
            )

            if new_cost < costs[nxt]:
                costs[nxt] = new_cost
                previous[nxt] = cell

                heapq.heappush(
                    queue,
                    (
                        new_cost,
                        nxt,
                    ),
                )

    if not np.isfinite(
        costs[goal]
    ):
        raise RuntimeError(
            "no minimax path found"
        )

    path = [
        goal
    ]

    while path[-1] != start:
        current = path[-1]

        if current not in previous:
            raise RuntimeError(
                "failed to reconstruct minimax path"
            )

        path.append(
            previous[current]
        )

    path.reverse()

    return float(
        costs[goal]
    ), path


def build_surface(
    system,
    *,
    donor_step,
    transfer_step,
    batch_size,
):
    probe = scan.apply_system(
        system
    )

    donor_low, donor_high, _ = (
        probe[
            "donor"
        ]
    )

    transfer_low, transfer_high, _ = (
        probe[
            "transfer"
        ]
    )

    donors = inclusive_axis(
        donor_low,
        donor_high,
        donor_step,
    )

    transfers = inclusive_axis(
        transfer_low,
        transfer_high,
        transfer_step,
    )

    spectators = frozen_spectators(
        system
    )

    geometry_builder = scan.SYSTEMS[
        system
    ][
        "geometry"
    ]

    raw_boxes = []

    for donor in donors:
        for transfer in transfers:
            symbols, positions = geometry_builder(
                float(
                    donor
                ),
                float(
                    transfer
                ),
                spectators,
            )

            raw_boxes.append(
                (
                    symbols,
                    positions,
                )
            )

    energies = evaluate_raw_boxes(
        raw_boxes,
        mixing=SAPT_H_STATE_MIXING,
        device="cpu",
        dtype=torch.float64,
        batch_size=batch_size,
    )

    grid = energies.reshape(
        len(
            donors
        ),
        len(
            transfers
        ),
    )

    reactant, product = scan.basin_seeds(
        grid,
        donors,
        transfers,
    )

    if reactant is None:
        raise RuntimeError(
            f"{system}: no basin seeds"
        )

    minimax_energy, path = minimax_path(
        grid,
        reactant,
        product,
    )

    return {
        "grid": grid,
        "donors": donors,
        "transfers": transfers,
        "spectators": spectators,
        "geometry_builder": geometry_builder,
        "reactant": reactant,
        "product": product,
        "path": path,
        "minimax_energy": minimax_energy,
    }


def scalar(value):
    return float(
        value
        .detach()
        .cpu()
    )


def coupling_between(
    result,
    reactant_state,
    product_state,
):
    key = (
        min(
            reactant_state,
            product_state,
        ),
        max(
            reactant_state,
            product_state,
        ),
    )

    coupling = result[
        "couplings"
    ].get(
        key
    )

    if coupling is None:
        return np.nan

    return abs(
        scalar(
            coupling
        )
    )


def sapt_competing_walls(
    sim,
    positions,
    values,
    edge_atoms,
    edge_rows,
    edge_slots,
    sapt_result,
    reactant_state,
    product_state,
    donor_edge,
    hh_edge,
):
    """
    Reconstruct per-edge SAPT wall contributions for the two intended states.
    """

    import reactive as R
    import nonbonded_continuous_torch as nb

    from sapt_h_state_torch import (
        _descriptor_weights_for_state,
        _sapt_pair_energy,
    )

    taper = values[
        "taper"
    ]

    edge_tapers = [
        taper[
            row,
            slot,
        ]
        for row, slot
        in zip(
            edge_rows,
            edge_slots,
        )
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
            sim.per_box
        )
    ]

    fragment_position = positions[
        :sim.per_box
    ]

    def wall_for_state(
        state_index,
        target_edge,
    ):
        state = sapt_result[
            "states"
        ][
            state_index
        ]

        weights = _descriptor_weights_for_state(
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

        fragment = nb.ContinuousTorchFragment(
            symbols=local_symbols,
            positions=fragment_position,
            bond_weights=weights,
        )

        first, second = edge_atoms[
            target_edge
        ]

        contribution = (
            edge_tapers[
                target_edge
            ]
            * _sapt_pair_energy(
                fragment,
                first,
                second,
            )
        )

        return scalar(
            contribution
        )

    reactant_hh_wall = wall_for_state(
        reactant_state,
        hh_edge,
    )

    product_ch_wall = wall_for_state(
        product_state,
        donor_edge,
    )

    return (
        reactant_hh_wall,
        product_ch_wall,
    )


def evaluate_cell(
    system,
    surface,
    cell,
):
    i, j = cell

    donor = float(
        surface[
            "donors"
        ][i]
    )

    transfer = float(
        surface[
            "transfers"
        ][j]
    )

    symbols, raw_positions = surface[
        "geometry_builder"
    ](
        donor,
        transfer,
        surface[
            "spectators"
        ],
    )

    sim = build_simulation(
        symbols,
        raw_positions,
        eta=SAPT_H_STATE_MIXING,
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

    try:
        donor_edge, hh_edge = (
            identify_molecular_transfer_edges(
                edge_atoms,
                symbols,
                positions,
                donor_target=donor,
                transfer_target=transfer,
            )
        )
    except RuntimeError:
        return None

    old = old_matrix(
        sim,
        values,
        edge_atoms,
        edge_rows,
        edge_slots,
    )

    sapt = sapt_matrix(
        sim,
        positions,
        values,
        edge_atoms,
        edge_rows,
        edge_slots,
    )

    try:
        (
            reactant_state,
            product_state,
        ) = classify_states(
            sapt[
                "states"
            ],
            donor_edge,
            hh_edge,
        )
    except RuntimeError:
        return None

    old_reactant = scalar(
        old[
            "diagonal"
        ][
            reactant_state
        ]
    )

    old_product = scalar(
        old[
            "diagonal"
        ][
            product_state
        ]
    )

    sapt_reactant = scalar(
        sapt[
            "diagonal"
        ][
            reactant_state
        ]
    )

    sapt_product = scalar(
        sapt[
            "diagonal"
        ][
            product_state
        ]
    )

    (
        reactant_hh_wall,
        product_ch_wall,
    ) = sapt_competing_walls(
        sim,
        positions,
        values,
        edge_atoms,
        edge_rows,
        edge_slots,
        sapt,
        reactant_state,
        product_state,
        donor_edge,
        hh_edge,
    )

    coupling = coupling_between(
        sapt,
        reactant_state,
        product_state,
    )

    return {
        "cell": cell,
        "donor": donor,
        "transfer": transfer,
        "old_reactant": old_reactant,
        "old_product": old_product,
        "sapt_reactant": sapt_reactant,
        "sapt_product": sapt_product,
        "reactant_shift": (
            sapt_reactant
            - old_reactant
        ),
        "product_shift": (
            sapt_product
            - old_product
        ),
        "old_gap": abs(
            old_product
            - old_reactant
        ),
        "sapt_gap": abs(
            sapt_product
            - sapt_reactant
        ),
        "coupling": coupling,
        "coupling_prefactor": (
            coupling
            / SAPT_H_STATE_MIXING
            if np.isfinite(
                coupling
            )
            else np.nan
        ),
        "reactant_hh_wall": (
            reactant_hh_wall
        ),
        "product_ch_wall": (
            product_ch_wall
        ),
    }


def analyse_system(
    system,
    *,
    donor_step,
    transfer_step,
    batch_size,
):
    surface = build_surface(
        system,
        donor_step=donor_step,
        transfer_step=transfer_step,
        batch_size=batch_size,
    )

    grid = surface[
        "grid"
    ]

    reactant_energy = float(
        grid[
            surface[
                "reactant"
            ]
        ]
    )

    path = surface[
        "path"
    ]

    path_energies = [
        float(
            grid[
                cell
            ]
        )
        for cell in path
    ]

    barrier_index = int(
        np.argmax(
            path_energies
        )
    )

    barrier_cell = path[
        barrier_index
    ]

    decomposed = []

    for path_index, cell in enumerate(
        path
    ):
        result = evaluate_cell(
            system,
            surface,
            cell,
        )

        if result is None:
            continue

        result[
            "path_index"
        ] = path_index

        result[
            "relative_energy"
        ] = (
            float(
                grid[
                    cell
                ]
            )
            - reactant_energy
        )

        decomposed.append(
            result
        )

    if not decomposed:
        raise RuntimeError(
            f"{system}: no decomposable C-H -> H-H path cells"
        )

    barrier_result = next(
        (
            result
            for result in decomposed
            if result[
                "cell"
            ] == barrier_cell
        ),
        None,
    )

    closest_gap = min(
        decomposed,
        key=lambda result: (
            result[
                "sapt_gap"
            ]
        ),
    )

    max_product_lift = max(
        decomposed,
        key=lambda result: (
            result[
                "product_shift"
            ]
        ),
    )

    return {
        "system": system,
        "surface": surface,
        "barrier_index": barrier_index,
        "barrier_cell": barrier_cell,
        "barrier_relative": (
            path_energies[
                barrier_index
            ]
            - reactant_energy
        ),
        "barrier_result": barrier_result,
        "closest_gap": closest_gap,
        "max_product_lift": max_product_lift,
    }


def show_point(
    title,
    result,
):
    print(
        title
    )

    print(
        "-" * len(
            title
        )
    )

    print(
        f"path index               {result['path_index']}"
    )

    print(
        f"C-H / H-H               "
        f"{result['donor']:.5f} / {result['transfer']:.5f} A"
    )

    print(
        f"path energy              {result['relative_energy']:+.6f} eV"
    )

    print(
        f"old reactant diagonal    {result['old_reactant']:+.6f} eV"
    )

    print(
        f"SAPT reactant diagonal   {result['sapt_reactant']:+.6f} eV"
    )

    print(
        f"reactant shift           {result['reactant_shift']:+.6f} eV"
    )

    print(
        f"old product diagonal     {result['old_product']:+.6f} eV"
    )

    print(
        f"SAPT product diagonal    {result['sapt_product']:+.6f} eV"
    )

    print(
        f"product shift            {result['product_shift']:+.6f} eV"
    )

    print(
        f"old |gap|                {result['old_gap']:.6f} eV"
    )

    print(
        f"SAPT |gap|               {result['sapt_gap']:.6f} eV"
    )

    print(
        f"coupling V               {result['coupling']:.6f} eV"
    )

    print(
        f"V/eta                    {result['coupling_prefactor']:.6f} eV"
    )

    print(
        f"reactant unoccupied H-H  {result['reactant_hh_wall']:.6f} eV"
    )

    print(
        f"product unoccupied C-H   {result['product_ch_wall']:.6f} eV"
    )

    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--donor-step",
        type=float,
        default=DEFAULT_DONOR_STEP,
    )

    parser.add_argument(
        "--transfer-step",
        type=float,
        default=DEFAULT_TRANSFER_STEP,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    args = parser.parse_args()

    print(
        "SAPT C-H -> H-H DIABATIC COMPARISON"
    )

    print(
        "===================================="
    )

    print(
        f"same eta for both systems: {SAPT_H_STATE_MIXING:.9f}"
    )

    print(
        f"frozen grid step: "
        f"{args.donor_step:.3f}/{args.transfer_step:.3f} A"
    )

    print()

    analyses = []

    for system in DEFAULT_SYSTEMS:
        analysis = analyse_system(
            system,
            donor_step=args.donor_step,
            transfer_step=args.transfer_step,
            batch_size=args.batch_size,
        )

        analyses.append(
            analysis
        )

        title = system.upper()

        print(
            title
        )

        print(
            "=" * len(
                title
            )
        )

        print(
            f"SAPT minimax barrier     {analysis['barrier_relative']:.6f} eV"
        )

        if analysis[
            "barrier_result"
        ] is None:
            print(
                "actual barrier cell is not decomposable as the intended "
                "C-H -> H-H pair"
            )

            print()
        else:
            show_point(
                "ACTUAL SAPT BARRIER",
                analysis[
                    "barrier_result"
                ],
            )

        show_point(
            "CLOSEST SAPT DIABATIC APPROACH",
            analysis[
                "closest_gap"
            ],
        )

        show_point(
            "LARGEST PRODUCT-STATE LIFT ON PATH",
            analysis[
                "max_product_lift"
            ],
        )

    print(
        "COMPACT BARRIER COMPARISON"
    )

    print(
        "=========================="
    )

    print(
        "system          barrier   dReact   dProduct   oldGap   "
        "saptGap      V    H-Hwall   C-Hwall"
    )

    for analysis in analyses:
        result = analysis[
            "barrier_result"
        ]

        if result is None:
            print(
                f"{analysis['system']:>12}  "
                f"{analysis['barrier_relative']:7.3f}  "
                f"{'---':>7}  "
                f"{'---':>9}  "
                f"{'---':>7}  "
                f"{'---':>8}  "
                f"{'---':>6}  "
                f"{'---':>8}  "
                f"{'---':>8}"
            )

            continue

        print(
            f"{analysis['system']:>12}  "
            f"{analysis['barrier_relative']:7.3f}  "
            f"{result['reactant_shift']:+7.3f}  "
            f"{result['product_shift']:+9.3f}  "
            f"{result['old_gap']:7.3f}  "
            f"{result['sapt_gap']:8.3f}  "
            f"{result['coupling']:6.3f}  "
            f"{result['reactant_hh_wall']:8.3f}  "
            f"{result['product_ch_wall']:8.3f}"
        )

    print()

    print(
        "Decision rule:"
    )

    print(
        "  If methane also gets a multi-eV positive product shift with a "
        "large unoccupied C-H wall, the failure is generic to C-H -> H-H."
    )

    print(
        "  If methane's product shift is modest while formaldehyde's is huge, "
        "the carbonyl/radical environment is the special state-diagonal "
        "failure."
    )


if __name__ == "__main__":
    main()
