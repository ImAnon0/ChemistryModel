"""
Barrier ablation screen for the state-aware base-valence experiment.

Compares:
    current SAPT
    bond_order only
    softening only
    heavy overcoord only
    angles/lone-pairs only
    all state-aware terms together

No parameter is fitted. All use the same frozen H3 mixing value.

Usage:
    py sapt_state_aware_ablation_screen.py
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import reactive as R
import hf_surface_scan as scan

from sapt_h_state_torch import (
    SAPT_H_STATE_MIXING,
    SaptHStateBatchedSimulation,
)

from sapt_state_aware_valence_torch import (
    StateAwareValenceSaptHStateBatchedSimulation,
)

from sapt_state_aware_ablation_torch import (
    StateAwareAblationSaptHStateBatchedSimulation,
)


CORRECT_WATER_FROZEN = np.array(
    [
        0.960,
        0.960,
        75.53,
        75.53,
    ],
    dtype=float,
)


def inclusive_axis(
    start,
    stop,
    step,
):
    return np.arange(
        start,
        stop
        + 0.5
        * step,
        step,
        dtype=float,
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


def build_model(
    kind,
    boxes,
):
    common = dict(
        boxes=boxes,
        box_size=scan.BOX,
        random_seed=0,
        relax_on_start=False,
        device="cpu",
        dtype=torch.float64,
        h_state_mixing=(
            SAPT_H_STATE_MIXING
        ),
    )

    if kind == "current":
        return SaptHStateBatchedSimulation(
            **common
        )

    if kind == "all":
        return StateAwareValenceSaptHStateBatchedSimulation(
            **common
        )

    return StateAwareAblationSaptHStateBatchedSimulation(
        **common,
        ablation_mode=kind,
    )


def evaluate_boxes(
    kind,
    raw_boxes,
    *,
    batch_size,
):
    energies = []

    for start in range(
        0,
        len(
            raw_boxes
        ),
        batch_size,
    ):
        chunk = raw_boxes[
            start:
            start
            + batch_size
        ]

        boxes = [
            (
                symbols,
                np.asarray(
                    positions,
                    dtype=float,
                )
                + scan.CENTRE,
            )
            for symbols, positions
            in chunk
        ]

        simulation = build_model(
            kind,
            boxes,
        )

        raw = simulation.potential_per_box

        if torch.is_tensor(
            raw
        ):
            values = (
                raw
                .detach()
                .cpu()
                .numpy()
                .astype(
                    float
                )
            )
        else:
            values = np.asarray(
                raw,
                dtype=float,
            )

        energies.extend(
            values.tolist()
        )

    return np.asarray(
        energies,
        dtype=float,
    )


def h3_barrier(
    kind,
    *,
    batch_size,
):
    h = int(
        R.ELEMENT_INDEX[
            "H"
        ]
    )

    re = float(
        R.BOND_LENGTH[
            h,
            h,
        ]
    )

    symbols = [
        "H",
        "H",
        "H",
    ]

    reactant = np.array(
        [
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                re,
                0.0,
                0.0,
            ],
            [
                re + 3.0,
                0.0,
                0.0,
            ],
        ],
        dtype=float,
    )

    radii = np.linspace(
        0.80,
        1.08,
        141,
    )

    boxes = [
        (
            symbols,
            reactant,
        )
    ]

    for radius in radii:
        boxes.append(
            (
                symbols,
                np.array(
                    [
                        [
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            float(
                                radius
                            ),
                            0.0,
                            0.0,
                        ],
                        [
                            2.0
                            * float(
                                radius
                            ),
                            0.0,
                            0.0,
                        ],
                    ],
                    dtype=float,
                ),
            )
        )

    energies = evaluate_boxes(
        kind,
        boxes,
        batch_size=batch_size,
    )

    index = int(
        np.argmin(
            energies[
                1:
            ]
        )
    )

    return {
        "barrier": float(
            energies[
                1
                + index
            ]
            - energies[
                0
            ]
        ),
        "radius": float(
            radii[
                index
            ]
        ),
    }


def measure_system(
    kind,
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

    energies = evaluate_boxes(
        kind,
        boxes,
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
            f"{system}/{kind}: no basins"
        )

    (
        saddle_cell,
        saddle_energy,
    ) = scan.flood_saddle(
        grid,
        reactant,
        product,
    )

    if saddle_cell is None:
        raise RuntimeError(
            f"{system}/{kind}: no route"
        )

    reactant_energy = float(
        grid[
            reactant
        ]
    )

    product_energy = float(
        grid[
            product
        ]
    )

    return {
        "barrier": float(
            saddle_energy
            - reactant_energy
        ),
        "reaction": float(
            product_energy
            - reactant_energy
        ),
        "saddle_donor": float(
            donors[
                saddle_cell[
                    0
                ]
            ]
        ),
        "saddle_transfer": float(
            transfers[
                saddle_cell[
                    1
                ]
            ]
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--donor-step",
        type=float,
        default=0.04,
    )

    parser.add_argument(
        "--transfer-step",
        type=float,
        default=0.04,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    kinds = (
        "current",
        "bond_order",
        "softening",
        "overcoord",
        "angles",
        "all",
    )

    systems = (
        "formaldehyde",
        "water",
        "methane",
    )

    print(
        "STATE-AWARE VALENCE COMPONENT ABLATION"
    )

    print(
        "======================================"
    )

    print(
        f"mixing frozen at {SAPT_H_STATE_MIXING:.9f}"
    )

    print(
        f"grid step {args.donor_step:.3f}/"
        f"{args.transfer_step:.3f} A"
    )

    print()

    h3 = {}

    for kind in kinds:
        h3[
            kind
        ] = h3_barrier(
            kind,
            batch_size=args.batch_size,
        )

    print(
        "H3 CONTROL"
    )

    print(
        "----------"
    )

    baseline_h3 = h3[
        "current"
    ][
        "barrier"
    ]

    for kind in kinds:
        result = h3[
            kind
        ]

        print(
            f"{kind:>12}: "
            f"{result['barrier']:.6f} eV "
            f"({result['barrier']-baseline_h3:+.6f}), "
            f"r={result['radius']:.5f} A"
        )

    print()

    results = {}

    for system in systems:
        results[
            system
        ] = {}

        print(
            f"evaluating {system}..."
        )

        for kind in kinds:
            results[
                system
            ][
                kind
            ] = measure_system(
                kind,
                system,
                donor_step=args.donor_step,
                transfer_step=args.transfer_step,
                batch_size=args.batch_size,
            )

    print()

    print(
        "BARRIER ABLATION"
    )

    print(
        "----------------"
    )

    print(
        "system/mode        barrier      delta       saddle"
    )

    for system in systems:
        baseline = results[
            system
        ][
            "current"
        ][
            "barrier"
        ]

        print()

        print(
            system.upper()
        )

        for kind in kinds:
            result = results[
                system
            ][
                kind
            ]

            print(
                f"  {kind:>10}  "
                f"{result['barrier']:9.6f}  "
                f"{result['barrier']-baseline:+9.6f}  "
                f"{result['saddle_donor']:.3f}/"
                f"{result['saddle_transfer']:.3f} A"
            )

    print()

    print(
        "COMPACT DELTAS FROM CURRENT SAPT"
    )

    print(
        "-------------------------------"
    )

    print(
        f"{'mode':>12}  "
        f"{'formaldehyde':>14}  "
        f"{'water':>10}  "
        f"{'methane':>10}"
    )

    for kind in kinds[
        1:
    ]:
        values = []

        for system in systems:
            baseline = results[
                system
            ][
                "current"
            ][
                "barrier"
            ]

            changed = results[
                system
            ][
                kind
            ][
                "barrier"
            ]

            values.append(
                changed
                - baseline
            )

        print(
            f"{kind:>12}  "
            f"{values[0]:+14.6f}  "
            f"{values[1]:+10.6f}  "
            f"{values[2]:+10.6f}"
        )

    print()

    print(
        "REACTION-ENERGY DRIFT"
    )

    print(
        "---------------------"
    )

    print(
        "These should be near zero if the ablation is confined to the "
        "transfer region."
    )

    for system in systems:
        baseline = results[
            system
        ][
            "current"
        ][
            "reaction"
        ]

        print(
            f"{system}:"
        )

        for kind in kinds[
            1:
        ]:
            drift = (
                results[
                    system
                ][
                    kind
                ][
                    "reaction"
                ]
                - baseline
            )

            print(
                f"  {kind:>10}: {drift:+.9f} eV"
            )

    print()

    print(
        "Decision:"
    )

    print(
        "  A large positive/negative delta in one isolated mode identifies "
        "which early-code mechanism was responsible for the all-at-once "
        "barrier shift."
    )

    print(
        "  The individual deltas need not sum exactly to the 'all' delta "
        "because bond order, softening and angles are nonlinear and coupled."
    )


if __name__ == "__main__":
    main()
