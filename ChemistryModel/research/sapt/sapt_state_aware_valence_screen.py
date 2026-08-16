"""
Screen the state-aware base-valence SAPT H-state experiment.

No parameters are fitted. The same H3-anchored mixing
SAPT_H_STATE_MIXING = 0.534590721 is used for both the current adapter and the
state-aware experiment.

The screen compares frozen 2D minimax barriers for:
    formaldehyde
    water
    methane

and verifies that H3 remains anchored.

Usage:
    py sapt_state_aware_valence_screen.py

Finer:
    py sapt_state_aware_valence_screen.py --donor-step 0.02 --transfer-step 0.02
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


def evaluate_boxes(
    model,
    raw_boxes,
    *,
    batch_size,
):
    values = []

    for start in range(
        0,
        len(
            raw_boxes
        ),
        batch_size,
    ):
        chunk = raw_boxes[
            start:
            start + batch_size
        ]

        prepared = [
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

        sim = model(
            boxes=prepared,
            box_size=scan.BOX,
            random_seed=0,
            relax_on_start=False,
            device="cpu",
            dtype=torch.float64,
            h_state_mixing=(
                SAPT_H_STATE_MIXING
            ),
        )

        raw = sim.potential_per_box

        if torch.is_tensor(
            raw
        ):
            chunk_values = (
                raw
                .detach()
                .cpu()
                .numpy()
                .astype(
                    float
                )
            )
        else:
            chunk_values = np.asarray(
                raw,
                dtype=float,
            )

        values.extend(
            chunk_values.tolist()
        )

    return np.asarray(
        values,
        dtype=float,
    )


def h3_barrier(
    model,
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
        model,
        boxes,
        batch_size=batch_size,
    )

    seam = energies[
        1:
    ]

    index = int(
        np.argmin(
            seam
        )
    )

    return (
        float(
            seam[
                index
            ]
            - energies[
                0
            ]
        ),
        float(
            radii[
                index
            ]
        ),
    )


def measure_system(
    model,
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
        model,
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
            f"{system}: no basins"
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
            f"{system}: no minimax route"
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

    models = (
        (
            "current SAPT",
            SaptHStateBatchedSimulation,
        ),
        (
            "state-aware",
            StateAwareValenceSaptHStateBatchedSimulation,
        ),
    )

    print(
        "STATE-AWARE BASE-VALENCE SAPT SCREEN"
    )

    print(
        "===================================="
    )

    print(
        f"mixing frozen at {SAPT_H_STATE_MIXING:.9f}"
    )

    print(
        f"grid step {args.donor_step:.3f}/"
        f"{args.transfer_step:.3f} A"
    )

    print()

    print(
        "H3 CONTROL"
    )

    print(
        "----------"
    )

    for label, model in models:
        barrier, radius = h3_barrier(
            model,
            batch_size=args.batch_size,
        )

        print(
            f"{label:>14}: "
            f"{barrier:.6f} eV at r={radius:.5f} A"
        )

    print()

    results = {}

    for system in (
        "formaldehyde",
        "water",
        "methane",
    ):
        results[
            system
        ] = {}

        for label, model in models:
            results[
                system
            ][
                label
            ] = measure_system(
                model,
                system,
                donor_step=args.donor_step,
                transfer_step=args.transfer_step,
                batch_size=args.batch_size,
            )

    print(
        "FROZEN BARRIERS"
    )

    print(
        "---------------"
    )

    print(
        "system          current   state-aware     change       "
        "current saddle       state-aware saddle"
    )

    for system in (
        "formaldehyde",
        "water",
        "methane",
    ):
        current = results[
            system
        ][
            "current SAPT"
        ]

        aware = results[
            system
        ][
            "state-aware"
        ]

        print(
            f"{system:>12}  "
            f"{current['barrier']:8.3f}  "
            f"{aware['barrier']:11.3f}  "
            f"{aware['barrier']-current['barrier']:+9.3f}  "
            f"{current['saddle_donor']:.3f}/"
            f"{current['saddle_transfer']:.3f} A       "
            f"{aware['saddle_donor']:.3f}/"
            f"{aware['saddle_transfer']:.3f} A"
        )

    print()

    print(
        "REACTION ENERGIES"
    )

    print(
        "-----------------"
    )

    for system in (
        "formaldehyde",
        "water",
        "methane",
    ):
        current = results[
            system
        ][
            "current SAPT"
        ]

        aware = results[
            system
        ][
            "state-aware"
        ]

        print(
            f"{system:>12}: "
            f"current {current['reaction']:+.6f} eV, "
            f"state-aware {aware['reaction']:+.6f} eV, "
            f"change {aware['reaction']-current['reaction']:+.6f} eV"
        )

    print()

    print(
        "Interpretation:"
    )

    print(
        "  If H3 stays fixed while carbon-abstraction barriers move strongly, "
        "the missed state->base-valence connection was materially affecting "
        "the barrier architecture."
    )

    print(
        "  If settled reactant/product reaction energies barely move but the "
        "crossing barriers do, that is especially strong evidence that this "
        "was transition-state bookkeeping rather than a new fitted energy "
        "term."
    )


if __name__ == "__main__":
    main()
