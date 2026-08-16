"""
Inspect H-state topology at the fitted barrier and along the fitted minimax path.

This diagnostic exists because the previous decomposition found that water has
no *expected* directly coupled donor-H / H-H transfer pair anywhere on its
minimax path.

Rather than crashing, this script asks what the H-state machinery is actually
doing.

For formaldehyde, water and methane it:
  - rebuilds the frozen surface at the diagnostic eta from
    sapt_coupling_requirement_diagnostic.csv;
  - reconstructs the minimax path;
  - inventories every active H candidate edge at the barrier;
  - prints every nonzero H-state coupling at the barrier;
  - independently identifies the geometric donor heavy-H pair and forming H-H
    pair from all atoms, even if they are NOT active H-state candidates;
  - reports whether those expected pairs are active;
  - searches the whole minimax path for cells where both expected pairs are
    simultaneously active and directly coupled;
  - continues to all systems even when no such cell exists.

No parameter is changed.

Usage:
    py sapt_barrier_topology_diagnostic.py
"""

from __future__ import annotations

import argparse
import heapq
import math

import numpy as np
import torch

import reactive as R
import hf_surface_scan as scan

from sapt_h_state_torch import SAPT_H_STATE_MIXING

from sapt_coupling_requirement_diagnostic import (
    evaluate_raw_boxes,
    inclusive_axis,
)

from sapt_state_coupling_decomposition import (
    load_requirement_rows,
    frozen_spectators,
    build_simulation,
    prepare_intermediates,
    build_state_matrix,
    classify_states,
    state_label,
)


DEFAULT_REQUIREMENTS = (
    "sapt_coupling_requirement_diagnostic.csv"
)

DONOR_STEP = 0.04
TRANSFER_STEP = 0.04
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
            "no minimax path"
        )

    path = [
        goal
    ]

    while path[-1] != start:
        path.append(
            previous[
                path[-1]
            ]
        )

    path.reverse()

    return float(
        costs[goal]
    ), path


def fitted_eta(row):
    low = float(
        row[
            "required_eta_low"
        ]
    )

    high = float(
        row[
            "required_eta_high"
        ]
    )

    if math.isclose(
        low,
        high,
        abs_tol=1.0e-12,
    ):
        return low

    return min(
        (
            low,
            high,
        ),
        key=lambda value: abs(
            value
            - SAPT_H_STATE_MIXING
        ),
    )


def build_surface(
    system,
    eta,
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

    boxes = []

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

            boxes.append(
                (
                    symbols,
                    positions,
                )
            )

    energies = evaluate_raw_boxes(
        boxes,
        mixing=eta,
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

    _, path = minimax_path(
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
    }


def pair_distance(
    positions,
    first,
    second,
):
    return float(
        np.linalg.norm(
            np.asarray(
                positions[
                    first
                ],
                dtype=float,
            )
            - np.asarray(
                positions[
                    second
                ],
                dtype=float,
            )
        )
    )


def identify_expected_pairs(
    symbols,
    positions,
    donor_target,
    transfer_target,
):
    """
    Identify the geometric forming H-H pair from ALL H-H atom pairs, then the
    donor heavy-H pair sharing one of those H atoms from ALL heavy-H pairs.

    This deliberately does not depend on H-state candidate activation.
    """

    hh_candidates = []

    for first in range(
        len(
            symbols
        )
    ):
        for second in range(
            first + 1,
            len(
                symbols
            ),
        ):
            if not (
                symbols[
                    first
                ] == "H"
                and symbols[
                    second
                ] == "H"
            ):
                continue

            distance = pair_distance(
                positions,
                first,
                second,
            )

            hh_candidates.append(
                (
                    abs(
                        distance
                        - transfer_target
                    ),
                    first,
                    second,
                    distance,
                )
            )

    if not hh_candidates:
        raise RuntimeError(
            "geometry has no H-H atom pair"
        )

    _, hh_first, hh_second, hh_distance = min(
        hh_candidates
    )

    hh_atoms = {
        hh_first,
        hh_second,
    }

    donor_candidates = []

    for first in range(
        len(
            symbols
        )
    ):
        for second in range(
            first + 1,
            len(
                symbols
            ),
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

            distance = pair_distance(
                positions,
                first,
                second,
            )

            donor_candidates.append(
                (
                    abs(
                        distance
                        - donor_target
                    ),
                    first,
                    second,
                    distance,
                )
            )

    if not donor_candidates:
        raise RuntimeError(
            "could not identify geometric donor heavy-H pair"
        )

    _, donor_first, donor_second, donor_distance = min(
        donor_candidates
    )

    return {
        "donor_pair": frozenset(
            (
                donor_first,
                donor_second,
            )
        ),
        "donor_atoms": (
            donor_first,
            donor_second,
        ),
        "donor_distance": donor_distance,
        "hh_pair": frozenset(
            (
                hh_first,
                hh_second,
            )
        ),
        "hh_atoms": (
            hh_first,
            hh_second,
        ),
        "hh_distance": hh_distance,
    }


def inventory_cell(
    system,
    eta,
    donor,
    transfer,
    spectators,
):
    geometry_builder = scan.SYSTEMS[
        system
    ][
        "geometry"
    ]

    symbols, raw_positions = geometry_builder(
        donor,
        transfer,
        spectators,
    )

    expected = identify_expected_pairs(
        symbols,
        raw_positions,
        donor,
        transfer,
    )

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

    matrix = build_state_matrix(
        sim,
        positions,
        values,
        edge_atoms,
        edge_rows,
        edge_slots,
        eta=eta,
    )

    active_lookup = {
        frozenset(
            pair
        ): index
        for index, pair
        in enumerate(
            edge_atoms
        )
    }

    donor_edge = active_lookup.get(
        expected[
            "donor_pair"
        ]
    )

    hh_edge = active_lookup.get(
        expected[
            "hh_pair"
        ]
    )

    direct = False
    reactant_state = None
    product_state = None

    if (
        donor_edge is not None
        and hh_edge is not None
    ):
        try:
            (
                reactant_state,
                product_state,
            ) = classify_states(
                matrix[
                    "states"
                ],
                donor_edge,
                hh_edge,
            )

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

            direct = (
                key
                in matrix[
                    "couplings"
                ]
            )
        except RuntimeError:
            direct = False

    edges = []

    taper = values[
        "taper"
    ]

    for index, (
        first,
        second,
    ) in enumerate(
        edge_atoms
    ):
        row = edge_rows[
            index
        ]

        slot = edge_slots[
            index
        ]

        edges.append(
            {
                "index": index,
                "first": first,
                "second": second,
                "label": (
                    f"{symbols[first]}{first}-"
                    f"{symbols[second]}{second}"
                ),
                "distance": pair_distance(
                    raw_positions,
                    first,
                    second,
                ),
                "taper": float(
                    taper[
                        row,
                        slot,
                    ]
                    .detach()
                    .cpu()
                ),
            }
        )

    couplings = []

    for (
        first_state,
        second_state,
    ), value in matrix[
        "couplings"
    ].items():
        couplings.append(
            {
                "first_state": first_state,
                "second_state": second_state,
                "value": abs(
                    float(
                        value
                        .detach()
                        .cpu()
                    )
                ),
                "first_label": state_label(
                    matrix[
                        "states"
                    ][
                        first_state
                    ],
                    edge_atoms,
                    symbols,
                ),
                "second_label": state_label(
                    matrix[
                        "states"
                    ][
                        second_state
                    ],
                    edge_atoms,
                    symbols,
                ),
            }
        )

    couplings.sort(
        key=lambda item: (
            item[
                "value"
            ]
        ),
        reverse=True,
    )

    return {
        "symbols": symbols,
        "raw_positions": raw_positions,
        "expected": expected,
        "edge_atoms": edge_atoms,
        "edges": edges,
        "matrix": matrix,
        "donor_edge": donor_edge,
        "hh_edge": hh_edge,
        "direct": direct,
        "reactant_state": reactant_state,
        "product_state": product_state,
        "couplings": couplings,
    }


def print_barrier_inventory(
    system,
    eta,
    donor,
    transfer,
    relative_energy,
    inventory,
):
    print(
        f"eta                       {eta:.9f}"
    )

    print(
        f"barrier                   {relative_energy:.6f} eV"
    )

    print(
        f"geometry donor/forming    "
        f"{donor:.5f} / {transfer:.5f} A"
    )

    expected = inventory[
        "expected"
    ]

    donor_first, donor_second = expected[
        "donor_atoms"
    ]

    hh_first, hh_second = expected[
        "hh_atoms"
    ]

    symbols = inventory[
        "symbols"
    ]

    print(
        "expected donor pair       "
        f"{symbols[donor_first]}{donor_first}-"
        f"{symbols[donor_second]}{donor_second}  "
        f"{expected['donor_distance']:.5f} A  "
        f"active={inventory['donor_edge'] is not None}"
    )

    print(
        "expected forming pair     "
        f"{symbols[hh_first]}{hh_first}-"
        f"{symbols[hh_second]}{hh_second}  "
        f"{expected['hh_distance']:.5f} A  "
        f"active={inventory['hh_edge'] is not None}"
    )

    print(
        f"expected direct transfer  {inventory['direct']}"
    )

    print()

    print(
        "active H-state candidate edges:"
    )

    if not inventory[
        "edges"
    ]:
        print(
            "  none"
        )

    for edge in inventory[
        "edges"
    ]:
        print(
            f"  {edge['index']:2d}: "
            f"{edge['label']:<12} "
            f"r={edge['distance']:.5f} A  "
            f"taper={edge['taper']:.6f}"
        )

    print()

    print(
        "nonzero state couplings:"
    )

    if not inventory[
        "couplings"
    ]:
        print(
            "  none"
        )

    for coupling in inventory[
        "couplings"
    ]:
        print(
            f"  V={coupling['value']:.6f} eV  "
            f"{coupling['first_label']}  <->  "
            f"{coupling['second_label']}"
        )

    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--requirements",
        default=DEFAULT_REQUIREMENTS,
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

    print(
        "SAPT BARRIER H-STATE TOPOLOGY DIAGNOSTIC"
    )

    print(
        "========================================"
    )

    print(
        "This checks what transfer states/couplings actually exist at each "
        "fitted barrier and along its minimax path."
    )

    print()

    for system in systems:
        if system not in requirement_rows:
            print(
                f"{system}: no successful requirement row"
            )

            print()

            continue

        eta = fitted_eta(
            requirement_rows[
                system
            ]
        )

        surface = build_surface(
            system,
            eta,
            donor_step=args.donor_step,
            transfer_step=args.transfer_step,
            batch_size=args.batch_size,
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

        i, j = barrier_cell

        barrier_donor = float(
            surface[
                "donors"
            ][i]
        )

        barrier_transfer = float(
            surface[
                "transfers"
            ][j]
        )

        barrier_relative = (
            path_energies[
                barrier_index
            ]
            - reactant_energy
        )

        barrier_inventory = inventory_cell(
            system,
            eta,
            barrier_donor,
            barrier_transfer,
            surface[
                "spectators"
            ],
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

        print_barrier_inventory(
            system,
            eta,
            barrier_donor,
            barrier_transfer,
            barrier_relative,
            barrier_inventory,
        )

        direct_cells = []

        donor_active_cells = []

        hh_active_cells = []

        for path_index, cell in enumerate(
            path
        ):
            pi, pj = cell

            donor = float(
                surface[
                    "donors"
                ][pi]
            )

            transfer = float(
                surface[
                    "transfers"
                ][pj]
            )

            inventory = inventory_cell(
                system,
                eta,
                donor,
                transfer,
                surface[
                    "spectators"
                ],
            )

            if inventory[
                "donor_edge"
            ] is not None:
                donor_active_cells.append(
                    (
                        path_index,
                        donor,
                        transfer,
                    )
                )

            if inventory[
                "hh_edge"
            ] is not None:
                hh_active_cells.append(
                    (
                        path_index,
                        donor,
                        transfer,
                    )
                )

            if inventory[
                "direct"
            ]:
                direct_cells.append(
                    (
                        path_index,
                        donor,
                        transfer,
                        float(
                            grid[
                                cell
                            ]
                        )
                        - reactant_energy,
                    )
                )

        print(
            "path topology:"
        )

        print(
            f"  cells                     {len(path)}"
        )

        print(
            f"  expected donor active     {len(donor_active_cells)}"
        )

        print(
            f"  expected H-H active       {len(hh_active_cells)}"
        )

        print(
            f"  expected direct transfer  {len(direct_cells)}"
        )

        if donor_active_cells:
            print(
                "  donor-active path span    "
                f"{donor_active_cells[0][0]}.."
                f"{donor_active_cells[-1][0]}"
            )

        if hh_active_cells:
            print(
                "  H-H-active path span      "
                f"{hh_active_cells[0][0]}.."
                f"{hh_active_cells[-1][0]}"
            )

        if direct_cells:
            nearest = min(
                direct_cells,
                key=lambda item: abs(
                    item[
                        0
                    ]
                    - barrier_index
                ),
            )

            print(
                "  nearest direct cell       "
                f"index {nearest[0]}, "
                f"{nearest[1]:.3f}/{nearest[2]:.3f} A, "
                f"path E {nearest[3]:+.6f} eV"
            )
        else:
            print(
                "  nearest direct cell       NONE ON PATH"
            )

            if (
                donor_active_cells
                and hh_active_cells
            ):
                donor_last = (
                    donor_active_cells[
                        -1
                    ]
                )

                hh_first = (
                    hh_active_cells[
                        0
                    ]
                )

                print(
                    "  donor last active        "
                    f"index {donor_last[0]}, "
                    f"{donor_last[1]:.3f}/{donor_last[2]:.3f} A"
                )

                print(
                    "  H-H first active         "
                    f"index {hh_first[0]}, "
                    f"{hh_first[1]:.3f}/{hh_first[2]:.3f} A"
                )

                if (
                    donor_last[
                        0
                    ]
                    < hh_first[
                        0
                    ]
                ):
                    print(
                        "  topology                  DEAD ZONE: donor candidate "
                        "dies before forming H-H candidate appears"
                    )
                else:
                    print(
                        "  topology                  candidate spans overlap, "
                        "but state enumeration/coupling still gives no expected "
                        "direct transfer"
                    )

        print()


if __name__ == "__main__":
    main()
