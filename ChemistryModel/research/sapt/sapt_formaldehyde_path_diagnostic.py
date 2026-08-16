"""
Diagnose the actual minimax path on the saved relaxed formaldehyde SAPT surface.

This is a diagnostic only. It does not fit or modify parameters.

It loads the saved 2D relaxed formaldehyde surface, reconstructs a four-neighbour
minimax path from the saved reactant basin to product basin, then evaluates the
SAPT H-state decomposition at every cell on that path.

This answers the key question left by the one-dimensional breaking-C-H scan:
does the actual minimum-barrier path encounter a diabatic crossing, or does it
remain strongly biased toward one diabatic state through the adiabatic saddle?

Requires the previously created:
    sapt_formaldehyde_breaking_ch_scan.py

Default source:
    sapt_full_formaldehyde_relaxed_d0p04000_t0p04000.npz

Usage:
    py sapt_formaldehyde_path_diagnostic.py
"""

from __future__ import annotations

import argparse
import csv
import heapq
from pathlib import Path

import numpy as np

from sapt_formaldehyde_breaking_ch_scan import evaluate_point


DEFAULT_NPZ = (
    "sapt_full_formaldehyde_relaxed_d0p04000_t0p04000.npz"
)


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
    """
    Four-neighbour Dijkstra where path cost is the maximum grid energy visited.
    Returns (cost, path).
    """

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
            "No four-neighbour minimax path found."
        )

    path = [
        goal
    ]

    while path[-1] != start:
        current = path[-1]

        if current not in previous:
            raise RuntimeError(
                "Failed to reconstruct minimax path."
            )

        path.append(
            previous[
                current
            ]
        )

    path.reverse()

    return float(
        costs[goal]
    ), path


def load_surface(path):
    data = np.load(
        Path(path),
        allow_pickle=False,
    )

    required = (
        "grid",
        "spectators",
        "donor_lengths",
        "transfer_lengths",
        "reactant",
        "product",
        "saddle_cell",
    )

    missing = [
        key
        for key in required
        if key not in data
    ]

    if missing:
        raise RuntimeError(
            "Missing NPZ fields: "
            + ", ".join(
                missing
            )
        )

    return {
        key: np.asarray(
            data[key]
        )
        for key in required
    }


def fmt(value):
    if not np.isfinite(
        value
    ):
        return "nan"

    return f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--npz",
        default=DEFAULT_NPZ,
    )

    parser.add_argument(
        "--csv",
        default=(
            "sapt_formaldehyde_path_diagnostic.csv"
        ),
    )

    args = parser.parse_args()

    surface = load_surface(
        args.npz
    )

    grid = surface[
        "grid"
    ]

    donors = surface[
        "donor_lengths"
    ]

    transfers = surface[
        "transfer_lengths"
    ]

    spectators = surface[
        "spectators"
    ]

    reactant = tuple(
        int(x)
        for x in surface[
            "reactant"
        ]
    )

    product = tuple(
        int(x)
        for x in surface[
            "product"
        ]
    )

    saved_saddle = tuple(
        int(x)
        for x in surface[
            "saddle_cell"
        ]
    )

    minimax_energy, path = minimax_path(
        grid,
        reactant,
        product,
    )

    reactant_energy = float(
        grid[
            reactant
        ]
    )

    rows = []

    print(
        "SAPT FORMALDEHYDE MINIMAX-PATH DIAGNOSTIC"
    )

    print(
        "========================================="
    )

    print(
        f"source: {args.npz}"
    )

    print(
        f"path cells: {len(path)}"
    )

    print(
        f"reactant energy: {reactant_energy:+.9f} eV"
    )

    print(
        f"minimax energy:  {minimax_energy:+.9f} eV"
    )

    print(
        f"minimax barrier: {minimax_energy-reactant_energy:.9f} eV"
    )

    print()

    for path_index, cell in enumerate(
        path
    ):
        i, j = cell

        donor = float(
            donors[
                i
            ]
        )

        transfer = float(
            transfers[
                j
            ]
        )

        point_spectators = np.asarray(
            spectators[
                cell
            ],
            dtype=float,
        )

        diag = evaluate_point(
            donor,
            transfer,
            point_spectators,
        )

        grid_energy = float(
            grid[
                cell
            ]
        )

        row = {
            "path_index": path_index,
            "grid_i": i,
            "grid_j": j,
            "donor_ch_A": donor,
            "forming_hh_A": transfer,
            "grid_energy_eV": grid_energy,
            "relative_to_reactant_eV": (
                grid_energy
                - reactant_energy
            ),
            **diag,
        }

        rows.append(
            row
        )

    barrier_row = max(
        rows,
        key=lambda row: (
            row[
                "grid_energy_eV"
            ]
        ),
    )

    finite_gap_rows = [
        row
        for row in rows
        if np.isfinite(
            row[
                "abs_gap_eV"
            ]
        )
    ]

    closest_gap_row = (
        min(
            finite_gap_rows,
            key=lambda row: (
                row[
                    "abs_gap_eV"
                ]
            ),
        )
        if finite_gap_rows
        else None
    )

    print(
        "PATH BARRIER CELL"
    )

    print(
        "================="
    )

    print(
        f"index      {barrier_row['path_index']}"
    )

    print(
        f"C-H        {barrier_row['donor_ch_A']:.5f} A"
    )

    print(
        f"H-H        {barrier_row['forming_hh_A']:.5f} A"
    )

    print(
        f"barrier    {barrier_row['relative_to_reactant_eV']:.6f} eV"
    )

    print(
        f"gap        {fmt(barrier_row['abs_gap_eV'])} eV"
    )

    print(
        f"coupling   {fmt(barrier_row['coupling_eV'])} eV"
    )

    print(
        "product unoccupied C-H wall "
        f"{fmt(barrier_row['product_unoccupied_breaking_ch_wall_eV'])} eV"
    )

    print(
        "reactant unoccupied H-H wall "
        f"{fmt(barrier_row['reactant_unoccupied_hh_wall_eV'])} eV"
    )

    print()

    if closest_gap_row is not None:
        print(
            "CLOSEST DIABATIC APPROACH ON ACTUAL PATH"
        )

        print(
            "========================================"
        )

        print(
            f"index      {closest_gap_row['path_index']}"
        )

        print(
            f"C-H        {closest_gap_row['donor_ch_A']:.5f} A"
        )

        print(
            f"H-H        {closest_gap_row['forming_hh_A']:.5f} A"
        )

        print(
            f"path energy {closest_gap_row['relative_to_reactant_eV']:.6f} eV"
        )

        print(
            f"gap        {closest_gap_row['abs_gap_eV']:.6f} eV"
        )

        print(
            f"coupling   {closest_gap_row['coupling_eV']:.6f} eV"
        )

        print(
            "product unoccupied C-H wall "
            f"{closest_gap_row['product_unoccupied_breaking_ch_wall_eV']:.6f} eV"
        )

        print()

        separation = (
            closest_gap_row[
                "relative_to_reactant_eV"
            ]
            - barrier_row[
                "relative_to_reactant_eV"
            ]
        )

        print(
            "closest-gap point minus barrier energy: "
            f"{separation:+.6f} eV"
        )

    print()

    print(
        "SAVED SADDLE CHECK"
    )

    print(
        "=================="
    )

    print(
        f"saved saddle cell: {saved_saddle}"
    )

    saved_matches = [
        row
        for row in rows
        if (
            row[
                "grid_i"
            ],
            row[
                "grid_j"
            ],
        ) == saved_saddle
    ]

    print(
        "saved saddle lies on reconstructed path: "
        f"{bool(saved_matches)}"
    )

    print()

    print(
        "SELECTED PATH POINTS"
    )

    print(
        "===================="
    )

    interesting = {
        0,
        len(
            rows
        )
        - 1,
        barrier_row[
            "path_index"
        ],
    }

    if closest_gap_row is not None:
        interesting.add(
            closest_gap_row[
                "path_index"
            ]
        )

    # Also include a few neighbours around the barrier.
    for offset in (
        -2,
        -1,
        1,
        2,
    ):
        index = (
            barrier_row[
                "path_index"
            ]
            + offset
        )

        if 0 <= index < len(
            rows
        ):
            interesting.add(
                index
            )

    print(
        " idx    C-H     H-H    relE       gap       V      C-H wall"
    )

    for index in sorted(
        interesting
    ):
        row = rows[
            index
        ]

        print(
            f"{index:4d}  "
            f"{row['donor_ch_A']:6.3f}  "
            f"{row['forming_hh_A']:6.3f}  "
            f"{row['relative_to_reactant_eV']:+8.4f}  "
            f"{row['abs_gap_eV']:8.4f}  "
            f"{row['coupling_eV']:7.4f}  "
            f"{row['product_unoccupied_breaking_ch_wall_eV']:9.4f}"
        )

    fieldnames = list(
        rows[
            0
        ].keys()
    )

    with open(
        args.csv,
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

    print()

    print(
        f"saved CSV: {args.csv}"
    )


if __name__ == "__main__":
    main()
