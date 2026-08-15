
"""
Local high-resolution refinement of the full ChemistryModel + SAPT-wall
water transfer saddle.

Requires:
    sapt_full_water_scan.py
    sapt_h_state_torch.py
    sapt_full_water_frozen.npz

The global NPZ should be the converged 0.01 A frozen scan. This script
reconstructs that global minimax path, replaces only the saddle-region
segment with a much finer local minimax path, and reports the refined
barrier relative to the original global reactant.

Frozen:
    py sapt_water_saddle_refine.py

Relaxed local correction:
    py sapt_water_saddle_refine.py --relax --step 0.005

No SAPT or state-coupling parameters are changed.
"""

from __future__ import annotations

import argparse
import heapq
import time
from pathlib import Path

import numpy as np
import torch

import hf_surface_scan as scan
import sapt_full_water_scan as full


DEFAULT_GLOBAL = "sapt_full_water_frozen.npz"

# The transfer is symmetric, so refine both sides of the crossing.
# The earlier one-sided box left the mirrored product-side staircase on
# the coarse global grid and therefore could not control the final barrier.
DEFAULT_DONOR_MIN = 1.04
DEFAULT_DONOR_MAX = 1.32
DEFAULT_TRANSFER_MIN = 1.04
DEFAULT_TRANSFER_MAX = 1.32


NEIGHBOURS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    
)


def inclusive_axis(start, stop, step):
    count = int(
        round(
            (stop - start)
            / step
        )
    )

    values = (
        start
        + np.arange(
            count + 1,
            dtype=float,
        )
        * step
    )

    if values[-1] < stop - 1.0e-10:
        values = np.append(
            values,
            stop,
        )

    return values


def minimax_path(grid, start, goal):
    """
    Dijkstra variant whose path cost is the maximum cell energy visited.
    Returns the minimum possible maximum and one corresponding path.
    """

    rows, cols = grid.shape

    cost = np.full(
        grid.shape,
        np.inf,
        dtype=float,
    )

    previous = np.full(
        (rows, cols, 2),
        -1,
        dtype=int,
    )

    cost[start] = float(
        grid[start]
    )

    queue = [
        (
            float(grid[start]),
            int(start[0]),
            int(start[1]),
        )
    ]

    visited = np.zeros(
        grid.shape,
        dtype=bool,
    )

    while queue:
        current_cost, i, j = (
            heapq.heappop(
                queue
            )
        )

        if visited[i, j]:
            continue

        visited[i, j] = True

        if (
            i == goal[0]
            and j == goal[1]
        ):
            break

        for di, dj in NEIGHBOURS:
            ni = i + di
            nj = j + dj

            if (
                ni < 0
                or nj < 0
                or ni >= rows
                or nj >= cols
            ):
                continue

            candidate = max(
                current_cost,
                float(
                    grid[ni, nj]
                ),
            )

            if candidate < cost[ni, nj]:
                cost[ni, nj] = candidate

                previous[ni, nj] = (
                    i,
                    j,
                )

                heapq.heappush(
                    queue,
                    (
                        candidate,
                        ni,
                        nj,
                    ),
                )

    if not np.isfinite(
        cost[goal]
    ):
        raise RuntimeError(
            "No minimax path found."
        )

    path = []

    cell = (
        int(goal[0]),
        int(goal[1]),
    )

    while True:
        path.append(
            cell
        )

        if cell == tuple(start):
            break

        pi, pj = previous[
            cell
        ]

        if pi < 0:
            raise RuntimeError(
                "Broken predecessor chain."
            )

        cell = (
            int(pi),
            int(pj),
        )

    path.reverse()

    return (
        float(
            cost[goal]
        ),
        path,
    )


def nearest_index(values, target):
    return int(
        np.argmin(
            np.abs(
                values - target
            )
        )
    )


def load_global(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing global surface: {path}\n"
            "Run the 0.01 A frozen full-water scan first."
        )

    data = np.load(
        path,
        allow_pickle=False,
    )

    required = (
        "grid",
        "donor_lengths",
        "transfer_lengths",
        "reactant",
        "product",
        "saddle_cell",
    )

    missing = [
        name
        for name in required
        if name not in data
    ]

    if missing:
        raise RuntimeError(
            "Global NPZ is missing: "
            + ", ".join(
                missing
            )
        )

    result = {
        name: np.asarray(
            data[name]
        )
        for name in required
    }

    result["reactant"] = tuple(
        int(x)
        for x in result[
            "reactant"
        ]
    )

    result["product"] = tuple(
        int(x)
        for x in result[
            "product"
        ]
    )

    result["saddle_cell"] = tuple(
        int(x)
        for x in result[
            "saddle_cell"
        ]
    )

    return result


def path_segment_in_box(
    global_path,
    donor_lengths,
    transfer_lengths,
    donor_min,
    donor_max,
    transfer_min,
    transfer_max,
):
    inside = []

    for path_index, (
        i,
        j,
    ) in enumerate(
        global_path
    ):
        donor = float(
            donor_lengths[i]
        )

        transfer = float(
            transfer_lengths[j]
        )

        if (
            donor_min - 1.0e-12
            <= donor
            <= donor_max + 1.0e-12
            and transfer_min - 1.0e-12
            <= transfer
            <= transfer_max + 1.0e-12
        ):
            inside.append(
                (
                    path_index,
                    i,
                    j,
                )
            )

    if not inside:
        raise RuntimeError(
            "The global minimax path never enters "
            "the requested refinement box."
        )

    first = inside[0]
    last = inside[-1]

    return first, last


def configure(
    device,
    fast,
):
    full.install_water_geometry_fix()

    scan.SCAN_DEVICE = (
        device
    )

    scan.SCAN_DTYPE = (
        torch.float32
        if fast
        else torch.float64
    )


def evaluate_local_grid(
    donor_lengths,
    transfer_lengths,
    *,
    relax,
):
    original_build = (
        scan.build
    )

    scan.build = (
        full.build_sapt
    )

    try:
        sim = full.build_sapt()

        if relax:
            grid, spectators = scan.surface(
                sim,
                donor_lengths,
                transfer_lengths,
                relax=True,
                progress=True,
                progress_label=(
                    "SAPT local relax"
                ),
                record_spectators=True,
                gradient_based=True,
            )
        else:
            grid = scan.surface(
                sim,
                donor_lengths,
                transfer_lengths,
                relax=False,
                progress=True,
                progress_label=(
                    "SAPT local"
                ),
                physics="sapt",
                build_kwargs={},
                gradient_based=True,
            )

            spectators = np.broadcast_to(
                full.CORRECT_WATER_FROZEN,
                (
                    len(donor_lengths),
                    len(
                        transfer_lengths
                    ),
                    4,
                ),
            ).copy()

    finally:
        scan.build = (
            original_build
        )

    return (
        np.asarray(
            grid,
            dtype=float,
        ),
        np.asarray(
            spectators,
            dtype=float,
        ),
    )


def evaluate_single_relaxed(
    donor,
    transfer,
):
    grid, spectators = (
        evaluate_local_grid(
            np.asarray(
                [donor],
                dtype=float,
            ),
            np.asarray(
                [transfer],
                dtype=float,
            ),
            relax=True,
        )
    )

    return (
        float(
            grid[0, 0]
        ),
        np.asarray(
            spectators[
                0,
                0,
            ],
            dtype=float,
        ),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--global-npz",
        default=DEFAULT_GLOBAL,
    )

    parser.add_argument(
        "--step",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--donor-min",
        type=float,
        default=DEFAULT_DONOR_MIN,
    )

    parser.add_argument(
        "--donor-max",
        type=float,
        default=DEFAULT_DONOR_MAX,
    )

    parser.add_argument(
        "--transfer-min",
        type=float,
        default=DEFAULT_TRANSFER_MIN,
    )

    parser.add_argument(
        "--transfer-max",
        type=float,
        default=DEFAULT_TRANSFER_MAX,
    )

    parser.add_argument(
        "--relax",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
    )

    args = parser.parse_args()

    # 0.005 A is the default for both modes. It is fine enough to remove
    # the coarse-grid staircase while keeping relaxed local scans practical.
    step = (
        args.step
        if args.step is not None
        else 0.005
    )

    configure(
        args.device,
        args.fast,
    )

    global_data = load_global(
        args.global_npz
    )

    global_grid = (
        global_data[
            "grid"
        ]
    )

    global_donor = (
        global_data[
            "donor_lengths"
        ]
    )

    global_transfer = (
        global_data[
            "transfer_lengths"
        ]
    )

    reactant = (
        global_data[
            "reactant"
        ]
    )

    product = (
        global_data[
            "product"
        ]
    )

    global_cost, global_path = (
        minimax_path(
            global_grid,
            reactant,
            product,
        )
    )

    reactant_energy = float(
        global_grid[
            reactant
        ]
    )

    global_barrier = (
        global_cost
        - reactant_energy
    )

    first, last = path_segment_in_box(
        global_path,
        global_donor,
        global_transfer,
        args.donor_min,
        args.donor_max,
        args.transfer_min,
        args.transfer_max,
    )

    first_path_index, fi, fj = (
        first
    )

    last_path_index, li, lj = (
        last
    )

    entry_coord = (
        float(
            global_donor[fi]
        ),
        float(
            global_transfer[fj]
        ),
    )

    exit_coord = (
        float(
            global_donor[li]
        ),
        float(
            global_transfer[lj]
        ),
    )

    outside_energies = []

    for index, cell in enumerate(
        global_path
    ):
        if (
            index < first_path_index
            or index > last_path_index
        ):
            outside_energies.append(
                float(
                    global_grid[
                        cell
                    ]
                )
            )

    outside_max = (
        max(
            outside_energies
        )
        if outside_energies
        else -np.inf
    )

    donor_lengths = inclusive_axis(
        args.donor_min,
        args.donor_max,
        step,
    )

    transfer_lengths = inclusive_axis(
        args.transfer_min,
        args.transfer_max,
        step,
    )

    local_start = (
        nearest_index(
            donor_lengths,
            entry_coord[0],
        ),
        nearest_index(
            transfer_lengths,
            entry_coord[1],
        ),
    )

    local_goal = (
        nearest_index(
            donor_lengths,
            exit_coord[0],
        ),
        nearest_index(
            transfer_lengths,
            exit_coord[1],
        ),
    )

    print(
        "SAPT WATER SADDLE REFINEMENT"
    )
    print(
        "============================"
    )

    print(
        f"mode: "
        f"{'relaxed' if args.relax else 'frozen'}"
    )

    print(
        f"device: {scan.SCAN_DEVICE}"
    )

    print(
        f"dtype: {scan.SCAN_DTYPE}"
    )

    print(
        f"global surface: "
        f"{args.global_npz}"
    )

    global_step_d = (
        float(
            np.median(
                np.diff(
                    global_donor
                )
            )
        )
    )

    global_step_t = (
        float(
            np.median(
                np.diff(
                    global_transfer
                )
            )
        )
    )

    print(
        f"global steps: "
        f"{global_step_d:.5f} / "
        f"{global_step_t:.5f} A"
    )

    print(
        f"global barrier: "
        f"{global_barrier:.6f} eV"
    )

    print(
        "refinement box:"
        f" donor {args.donor_min:.4f}–"
        f"{args.donor_max:.4f} A,"
        f" transfer {args.transfer_min:.4f}–"
        f"{args.transfer_max:.4f} A"
    )

    print(
        f"local step: {step:.5f} A"
    )

    print(
        f"local cells: "
        f"{len(donor_lengths)} x "
        f"{len(transfer_lengths)} = "
        f"{len(donor_lengths) * len(transfer_lengths)}"
    )

    print(
        "global path entry:"
        f" donor={entry_coord[0]:.4f},"
        f" transfer={entry_coord[1]:.4f} A"
    )

    print(
        "global path exit: "
        f" donor={exit_coord[0]:.4f},"
        f" transfer={exit_coord[1]:.4f} A"
    )

    print()

    started = (
        time.monotonic()
    )

    local_grid, spectators = (
        evaluate_local_grid(
            donor_lengths,
            transfer_lengths,
            relax=args.relax,
        )
    )

    local_cost, local_path = (
        minimax_path(
            local_grid,
            local_start,
            local_goal,
        )
    )

    local_path_energies = [
        float(
            local_grid[
                cell
            ]
        )
        for cell in local_path
    ]

    local_saddle_index = int(
        np.argmax(
            local_path_energies
        )
    )

    local_saddle_cell = (
        local_path[
            local_saddle_index
        ]
    )

    local_saddle_energy = float(
        local_grid[
            local_saddle_cell
        ]
    )

    hybrid_saddle_energy = max(
        outside_max,
        local_cost,
    )

    if args.relax:
        reactant_donor = float(
            global_donor[
                reactant[0]
            ]
        )

        reactant_transfer = float(
            global_transfer[
                reactant[1]
            ]
        )

        relaxed_reactant_energy, (
            relaxed_reactant_spectators
        ) = evaluate_single_relaxed(
            reactant_donor,
            reactant_transfer,
        )

        barrier = (
            hybrid_saddle_energy
            - relaxed_reactant_energy
        )
    else:
        relaxed_reactant_energy = None
        relaxed_reactant_spectators = None

        barrier = (
            hybrid_saddle_energy
            - reactant_energy
        )

    elapsed = (
        time.monotonic()
        - started
    )

    saddle_donor = float(
        donor_lengths[
            local_saddle_cell[0]
        ]
    )

    saddle_transfer = float(
        transfer_lengths[
            local_saddle_cell[1]
        ]
    )

    saddle_spectators = (
        spectators[
            local_saddle_cell
        ]
    )

    print()
    print(
        "REFINED RESULT"
    )
    print(
        "=============="
    )

    print(
        f"local minimax maximum: "
        f"{local_cost:.9f} eV"
    )

    print(
        f"outside-path maximum:  "
        f"{outside_max:.9f} eV"
    )

    print(
        f"hybrid saddle energy:  "
        f"{hybrid_saddle_energy:.9f} eV"
    )

    print()

    print(
        f"saddle donor:    "
        f"{saddle_donor:.5f} A"
    )

    print(
        f"saddle transfer: "
        f"{saddle_transfer:.5f} A"
    )

    print(
        f"saddle O-O:      "
        f"{saddle_donor + saddle_transfer:.5f} A"
    )

    print(
        "saddle spectators: "
        + np.array2string(
            np.asarray(
                saddle_spectators,
                dtype=float,
            ),
            precision=6,
        )
    )

    print()

    if args.relax:
        print(
            "reactant reference:"
            f" donor={reactant_donor:.5f},"
            f" transfer={reactant_transfer:.5f} A"
        )

        print(
            "reactant spectators: "
            + np.array2string(
                relaxed_reactant_spectators,
                precision=6,
            )
        )

        print(
            f"relaxed reactant energy: "
            f"{relaxed_reactant_energy:.9f} eV"
        )
    else:
        print(
            f"global reactant energy: "
            f"{reactant_energy:.9f} eV"
        )

    print(
        f"refined barrier: "
        f"{barrier:.9f} eV"
    )

    print(
        f"elapsed: {elapsed:.1f} s"
    )

    boundary_warning = (
        local_saddle_cell[0]
        in (
            0,
            len(
                donor_lengths
            ) - 1,
        )
        or local_saddle_cell[1]
        in (
            0,
            len(
                transfer_lengths
            ) - 1,
        )
    )

    if boundary_warning:
        print()
        print(
            "WARNING: refined saddle lies on the "
            "local box boundary. Expand the box "
            "before treating this value as converged."
        )

    filename = (
        "sapt_water_saddle_"
        + (
            "relaxed"
            if args.relax
            else "frozen"
        )
        + f"_{step:.5f}.npz"
    )

    np.savez_compressed(
        filename,
        grid=local_grid,
        spectators=spectators,
        donor_lengths=donor_lengths,
        transfer_lengths=transfer_lengths,
        local_start=np.asarray(
            local_start
        ),
        local_goal=np.asarray(
            local_goal
        ),
        local_saddle_cell=np.asarray(
            local_saddle_cell
        ),
        global_reactant=np.asarray(
            reactant
        ),
        global_product=np.asarray(
            product
        ),
        global_barrier=np.asarray(
            global_barrier
        ),
        refined_barrier=np.asarray(
            barrier
        ),
    )

    print(
        f"saved: {filename}"
    )


if __name__ == "__main__":
    main()
