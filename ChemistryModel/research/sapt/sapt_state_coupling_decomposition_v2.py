"""
Robust four-system SAPT state/coupling decomposition.

This is a replacement follow-up for sapt_state_coupling_decomposition.py.

Why v2 exists
-------------
The first version assumed the fitted barrier cell always contains both the
breaking heavy-H candidate and the forming H-H candidate. That is false for
water at the eta boundary nearest H3: the saved barrier has forming H-H
~1.30 A, beyond the current H-H candidate cutoff (~1.186 A), so there is no
H-H state edge to decompose there.

That is scientifically useful, not an error to hide.

This version reconstructs each fitted frozen minimax path. It reports:
    1. whether the actual barrier cell has a directly coupled two-state
       transfer pair;
    2. if not, the nearest path cell that DOES contain both transfer states;
    3. the full diabatic/coupling decomposition at that active cell.

No physics or fitted parameters are changed.

Requires:
    sapt_coupling_requirement_diagnostic.py
    sapt_state_coupling_decomposition.py
    sapt_coupling_requirement_diagnostic.csv

Usage:
    py sapt_state_coupling_decomposition_v2.py
"""

from __future__ import annotations

import argparse
import heapq
import math

import numpy as np

import hf_surface_scan as scan

from sapt_h_state_torch import SAPT_H_STATE_MIXING

from sapt_coupling_requirement_diagnostic import (
    evaluate_raw_boxes,
    inclusive_axis,
)

from sapt_state_coupling_decomposition import (
    load_requirement_rows,
    choose_eta_and_saddle,
    frozen_spectators,
    decompose,
    h3_case,
)


DEFAULT_CSV = "sapt_coupling_requirement_diagnostic.csv"

DONOR_STEP = 0.04
TRANSFER_STEP = 0.04
DEVICE = "cpu"
BATCH_SIZE = 512


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
        int(x)
        for x in start
    )
    goal = tuple(
        int(x)
        for x in goal
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
    eta,
    *,
    donor_step,
    transfer_step,
    device,
    batch_size,
):
    probe = scan.apply_system(
        system
    )

    donor_low, donor_high, _ = probe[
        "donor"
    ]

    transfer_low, transfer_high, _ = probe[
        "transfer"
    ]

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
                float(donor),
                float(transfer),
                spectators,
            )

            raw_boxes.append(
                (
                    symbols,
                    positions,
                )
            )

    # Keep float64, matching the diagnostic that produced the eta requirements.
    import torch

    energies = evaluate_raw_boxes(
        raw_boxes,
        mixing=eta,
        device=device,
        dtype=torch.float64,
        batch_size=batch_size,
    )

    grid = energies.reshape(
        len(donors),
        len(transfers),
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


def try_decompose_cell(
    system,
    surface,
    cell,
    eta,
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

    symbols, positions = surface[
        "geometry_builder"
    ](
        donor,
        transfer,
        surface[
            "spectators"
        ],
    )

    try:
        result = decompose(
            system,
            symbols,
            positions,
            eta=eta,
            donor_target=donor,
            transfer_target=transfer,
            h3=False,
        )
    except RuntimeError as exc:
        text = str(
            exc
        )

        if (
            "no H-H transfer candidate found"
            in text
            or "no donor heavy-H candidate"
            in text
            or "could not identify reactant/product"
            in text
            or "not directly coupled"
            in text
        ):
            return None, text

        raise

    result[
        "donor_distance"
    ] = donor

    result[
        "transfer_distance"
    ] = transfer

    return result, None


def fitted_eta_for_system(
    system,
    row,
):
    low_eta = float(
        row[
            "required_eta_low"
        ]
    )

    high_eta = float(
        row[
            "required_eta_high"
        ]
    )

    if math.isclose(
        low_eta,
        high_eta,
        abs_tol=1.0e-12,
    ):
        return low_eta

    # For a reference interval use the acceptable boundary nearest H3.
    return min(
        (
            low_eta,
            high_eta,
        ),
        key=lambda value: abs(
            value
            - SAPT_H_STATE_MIXING
        ),
    )


def analyse_system(
    system,
    row,
    *,
    donor_step,
    transfer_step,
    device,
    batch_size,
):
    eta = fitted_eta_for_system(
        system,
        row,
    )

    surface = build_surface(
        system,
        eta,
        donor_step=donor_step,
        transfer_step=transfer_step,
        device=device,
        batch_size=batch_size,
    )

    path = surface[
        "path"
    ]

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

    barrier_result, barrier_failure = (
        try_decompose_cell(
            system,
            surface,
            barrier_cell,
            eta,
        )
    )

    active_candidates = []

    for path_index, cell in enumerate(
        path
    ):
        result, failure = try_decompose_cell(
            system,
            surface,
            cell,
            eta,
        )

        if result is None:
            continue

        energy = float(
            grid[
                cell
            ]
        )

        active_candidates.append(
            (
                abs(
                    path_index
                    - barrier_index
                ),
                abs(
                    energy
                    - path_energies[
                        barrier_index
                    ]
                ),
                path_index,
                cell,
                result,
            )
        )

    if not active_candidates:
        raise RuntimeError(
            f"{system}: no directly coupled transfer-state cell exists "
            "anywhere on the minimax path"
        )

    active_candidates.sort(
        key=lambda item: (
            item[
                0
            ],
            item[
                1
            ],
        )
    )

    (
        _,
        _,
        active_index,
        active_cell,
        active_result,
    ) = active_candidates[
        0
    ]

    active_energy = float(
        grid[
            active_cell
        ]
    )

    barrier_energy = float(
        grid[
            barrier_cell
        ]
    )

    return {
        "system": system,
        "eta": eta,
        "surface": surface,
        "barrier_index": barrier_index,
        "barrier_cell": barrier_cell,
        "barrier_energy": barrier_energy,
        "barrier_relative": (
            barrier_energy
            - reactant_energy
        ),
        "barrier_result": barrier_result,
        "barrier_failure": barrier_failure,
        "active_index": active_index,
        "active_cell": active_cell,
        "active_energy": active_energy,
        "active_relative": (
            active_energy
            - reactant_energy
        ),
        "active_result": active_result,
    }


def show_decomposition(
    result,
):
    print(
        f"reactant diagonal         {result['reactant_diag']:+.9f} eV"
    )
    print(
        f"product diagonal          {result['product_diag']:+.9f} eV"
    )
    print(
        f"signed gap P-R            {result['signed_gap']:+.9f} eV"
    )
    print(
        f"|gap|                     {result['abs_gap']:.9f} eV"
    )
    print(
        f"coupling V                {result['coupling']:.9f} eV"
    )
    print(
        f"coupling prefactor V/eta  {result['coupling_prefactor']:.9f} eV"
    )
    print(
        f"mixing lowering           {result['mixing_lowering']:+.9f} eV"
    )
    print(
        f"reactant covalent         {result['reactant_covalent']:+.9f} eV"
    )
    print(
        f"reactant SAPT wall        {result['reactant_wall']:+.9f} eV"
    )
    print(
        f"reactant competing wall   {result['reactant_competing_wall']:+.9f} eV"
    )
    print(
        f"product covalent          {result['product_covalent']:+.9f} eV"
    )
    print(
        f"product SAPT wall         {result['product_wall']:+.9f} eV"
    )
    print(
        f"product competing wall    {result['product_competing_wall']:+.9f} eV"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--requirements",
        default=DEFAULT_CSV,
    )

    parser.add_argument(
        "--donor-step",
        type=float,
        default=DONOR_STEP,
    )

    parser.add_argument(
        "--transfer-step",
        type=float,
        default=TRANSFER_STEP,
    )

    parser.add_argument(
        "--device",
        default=DEVICE,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    args = parser.parse_args()

    requirement_rows = load_requirement_rows(
        args.requirements
    )

    systems = (
        "formaldehyde",
        "water",
        "methane",
    )

    missing = [
        system
        for system in systems
        if system not in requirement_rows
    ]

    if missing:
        raise RuntimeError(
            "missing successful rows in requirements CSV: "
            + ", ".join(
                missing
            )
        )

    print(
        "SAPT STATE / COUPLING DECOMPOSITION V2"
    )
    print(
        "======================================"
    )
    print(
        "Barrier cells without an active H-H transfer edge are reported "
        "explicitly rather than treated as errors."
    )
    print()

    h3 = h3_case()

    print(
        "H3"
    )
    print(
        "=="
    )
    print(
        f"eta                       {h3['eta']:.9f}"
    )
    show_decomposition(
        h3
    )
    print()

    analyses = []

    for system in systems:
        analysis = analyse_system(
            system,
            requirement_rows[
                system
            ],
            donor_step=args.donor_step,
            transfer_step=args.transfer_step,
            device=args.device,
            batch_size=args.batch_size,
        )

        analyses.append(
            analysis
        )

        surface = analysis[
            "surface"
        ]

        bi, bj = analysis[
            "barrier_cell"
        ]

        ai, aj = analysis[
            "active_cell"
        ]

        barrier_donor = float(
            surface[
                "donors"
            ][bi]
        )
        barrier_transfer = float(
            surface[
                "transfers"
            ][bj]
        )

        active_donor = float(
            surface[
                "donors"
            ][ai]
        )
        active_transfer = float(
            surface[
                "transfers"
            ][aj]
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
            f"eta                       {analysis['eta']:.9f}"
        )
        print(
            f"path barrier              {analysis['barrier_relative']:.6f} eV"
        )
        print(
            f"barrier geometry          {barrier_donor:.5f} / "
            f"{barrier_transfer:.5f} A"
        )

        if analysis[
            "barrier_result"
        ] is None:
            print(
                "barrier direct coupling   NO"
            )
            print(
                f"reason                    {analysis['barrier_failure']}"
            )
            print(
                "This means eta does not act through a direct two-state "
                "H-transfer coupling at the barrier cell itself."
            )
        else:
            print(
                "barrier direct coupling   YES"
            )

        print()
        print(
            "nearest directly-coupled path cell"
        )
        print(
            "----------------------------------"
        )
        print(
            f"path index                {analysis['active_index']} "
            f"(barrier index {analysis['barrier_index']})"
        )
        print(
            f"geometry                  {active_donor:.5f} / "
            f"{active_transfer:.5f} A"
        )
        print(
            f"path energy                {analysis['active_relative']:.6f} eV"
        )
        print(
            f"energy vs barrier          "
            f"{analysis['active_relative']-analysis['barrier_relative']:+.6f} eV"
        )

        show_decomposition(
            analysis[
                "active_result"
            ]
        )

        print()

    print(
        "COMPACT COMPARISON"
    )
    print(
        "=================="
    )
    print(
        "system           eta    direct@barrier   |gap|       V      V/eta  "
        "mix lower   Rwall   Pwall"
    )

    print(
        f"{'H3':>12}  "
        f"{h3['eta']:7.4f}  "
        f"{'yes':>14}  "
        f"{h3['abs_gap']:7.3f}  "
        f"{h3['coupling']:7.3f}  "
        f"{h3['coupling_prefactor']:7.3f}  "
        f"{h3['mixing_lowering']:9.3f}  "
        f"{h3['reactant_wall']:6.3f}  "
        f"{h3['product_wall']:6.3f}"
    )

    for analysis in analyses:
        result = analysis[
            "active_result"
        ]

        direct = (
            "yes"
            if analysis[
                "barrier_result"
            ] is not None
            else "no"
        )

        print(
            f"{analysis['system']:>12}  "
            f"{analysis['eta']:7.4f}  "
            f"{direct:>14}  "
            f"{result['abs_gap']:7.3f}  "
            f"{result['coupling']:7.3f}  "
            f"{result['coupling_prefactor']:7.3f}  "
            f"{result['mixing_lowering']:9.3f}  "
            f"{result['reactant_wall']:6.3f}  "
            f"{result['product_wall']:6.3f}"
        )

    print()
    print(
        "Important: for any row with direct@barrier = no, the listed "
        "decomposition is the nearest directly-coupled cell on the same "
        "minimax path, not the barrier cell."
    )


if __name__ == "__main__":
    main()
