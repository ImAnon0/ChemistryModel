"""
Full ChemistryModel formaldehyde H-abstraction scan using the SAPT-wall
hydrogen-state adapter.

Reaction:
    H + CH2O -> H2 + HCO

This is a diagnostic only. It does not modify production reactive.py or
reactive_torch.py, and it does not fit any parameter.

The SAPT exchange model is frozen. The hydrogen-state coupling is frozen at
the H3-reanchored value:

    SAPT_H_STATE_MIXING = 0.534590721

Unlike the water holdout, formaldehyde exercises the heavy-atom environment
inside the SAPT descriptor: the persistent C=O covalent environment remains
visible while the transferring H changes diabatic partner.

Examples
--------
Frozen first pass:
    py sapt_full_formaldehyde_scan.py

Higher-resolution frozen scan:
    py sapt_full_formaldehyde_scan.py --donor-step 0.01 --transfer-step 0.01

Relaxed spectator geometry:
    py sapt_full_formaldehyde_scan.py --relax
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import hf_surface_scan as scan
from sapt_h_state_torch import (
    SAPT_H_STATE_MIXING,
    SaptHStateBatchedSimulation,
)


SYSTEM_NAME = "formaldehyde"


def configure_system():
    """Select the existing formaldehyde geometry and basin definitions."""

    scan.apply_system(SYSTEM_NAME)


def build_sapt(
    physics="sapt",
    *,
    boxes=None,
    **_ignored,
):
    """Scanner-compatible constructor for the frozen SAPT/H-state model."""

    configure_system()

    if boxes is None:
        symbols, positions = scan.active_geometry(
            *scan.SYSTEM_START[SYSTEM_NAME]
        )

        prepared = [
            (
                symbols,
                positions + scan.CENTRE,
            )
        ]
    else:
        prepared = boxes

    return SaptHStateBatchedSimulation(
        boxes=prepared,
        box_size=scan.BOX,
        random_seed=0,
        relax_on_start=False,
        device=scan.SCAN_DEVICE,
        dtype=scan.SCAN_DTYPE,
        h_state_mixing=SAPT_H_STATE_MIXING,
    )


def scan_axes(
    donor_step,
    transfer_step,
):
    """Use the same formaldehyde window already defined by hf_surface_scan."""

    probe = scan.SYSTEM_PROBES[
        SYSTEM_NAME
    ]

    donor_min, donor_max, _ = probe[
        "donor"
    ]

    transfer_min, transfer_max, _ = probe[
        "transfer"
    ]

    donor_lengths = np.arange(
        donor_min,
        donor_max + 0.5 * donor_step,
        donor_step,
    )

    transfer_lengths = np.arange(
        transfer_min,
        transfer_max + 0.5 * transfer_step,
        transfer_step,
    )

    return (
        donor_lengths,
        transfer_lengths,
    )


def measure_grid(
    *,
    relax,
    donor_step,
    transfer_step,
):
    configure_system()

    donor_lengths, transfer_lengths = scan_axes(
        donor_step,
        transfer_step,
    )

    # Reuse hf_surface_scan's grid/minimisation machinery without changing
    # that module on disk.
    original_build = scan.build
    scan.build = build_sapt

    try:
        sim = build_sapt()

        started = time.monotonic()

        if relax:
            grid, spectators = scan.surface(
                sim,
                donor_lengths,
                transfer_lengths,
                relax=True,
                progress=True,
                progress_label="CH2O SAPT",
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
                progress_label="CH2O SAPT",
                physics="sapt",
                build_kwargs={},
                gradient_based=True,
            )

            frozen = np.asarray(
                scan.SYSTEMS[
                    SYSTEM_NAME
                ][
                    "frozen"
                ],
                dtype=float,
            )

            spectators = np.broadcast_to(
                frozen,
                (
                    len(donor_lengths),
                    len(transfer_lengths),
                    len(frozen),
                ),
            ).copy()

        elapsed = (
            time.monotonic()
            - started
        )

    finally:
        scan.build = original_build

    reactant, product = scan.basin_seeds(
        grid,
        donor_lengths,
        transfer_lengths,
    )

    if reactant is None:
        raise RuntimeError(
            "Could not identify formaldehyde reactant/product basins. "
            "The scan window may need widening."
        )

    saddle_cell, saddle_energy = scan.flood_saddle(
        grid,
        reactant,
        product,
    )

    if saddle_cell is None:
        raise RuntimeError(
            "No connected formaldehyde path between the two basins."
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

    barrier = float(
        saddle_energy
        - reactant_energy
    )

    reaction = float(
        product_energy
        - reactant_energy
    )

    saddle_donor = float(
        donor_lengths[
            saddle_cell[0]
        ]
    )

    saddle_transfer = float(
        transfer_lengths[
            saddle_cell[1]
        ]
    )

    return {
        "grid": np.asarray(
            grid,
            dtype=float,
        ),
        "spectators": np.asarray(
            spectators,
            dtype=float,
        ),
        "donor_lengths": donor_lengths,
        "transfer_lengths": transfer_lengths,
        "reactant": reactant,
        "product": product,
        "saddle_cell": saddle_cell,
        "reactant_energy": reactant_energy,
        "product_energy": product_energy,
        "saddle_energy": float(
            saddle_energy
        ),
        "barrier": barrier,
        "reaction": reaction,
        "saddle_donor": saddle_donor,
        "saddle_transfer": saddle_transfer,
        "reactant_donor": float(
            donor_lengths[
                reactant[0]
            ]
        ),
        "reactant_transfer": float(
            transfer_lengths[
                reactant[1]
            ]
        ),
        "product_donor": float(
            donor_lengths[
                product[0]
            ]
        ),
        "product_transfer": float(
            transfer_lengths[
                product[1]
            ]
        ),
        "saddle_spectators": np.asarray(
            spectators[
                saddle_cell
            ],
            dtype=float,
        ),
        "reactant_spectators": np.asarray(
            spectators[
                reactant
            ],
            dtype=float,
        ),
        "product_spectators": np.asarray(
            spectators[
                product
            ],
            dtype=float,
        ),
        "elapsed": elapsed,
    }


def save_result(
    result,
    *,
    relax,
    donor_step,
    transfer_step,
):
    label = (
        "relaxed"
        if relax
        else "frozen"
    )

    step_label = (
        f"d{donor_step:.5f}_t{transfer_step:.5f}"
        .replace(
            ".",
            "p",
        )
    )

    filename = (
        f"sapt_full_formaldehyde_{label}_{step_label}.npz"
    )

    np.savez_compressed(
        filename,
        grid=result[
            "grid"
        ],
        spectators=result[
            "spectators"
        ],
        donor_lengths=result[
            "donor_lengths"
        ],
        transfer_lengths=result[
            "transfer_lengths"
        ],
        reactant=np.asarray(
            result[
                "reactant"
            ],
            dtype=int,
        ),
        product=np.asarray(
            result[
                "product"
            ],
            dtype=int,
        ),
        saddle_cell=np.asarray(
            result[
                "saddle_cell"
            ],
            dtype=int,
        ),
        barrier=np.asarray(
            result[
                "barrier"
            ],
            dtype=float,
        ),
        reaction=np.asarray(
            result[
                "reaction"
            ],
            dtype=float,
        ),
        mixing=np.asarray(
            SAPT_H_STATE_MIXING,
            dtype=float,
        ),
        donor_step=np.asarray(
            donor_step,
            dtype=float,
        ),
        transfer_step=np.asarray(
            transfer_step,
            dtype=float,
        ),
        relaxed=np.asarray(
            relax,
            dtype=bool,
        ),
    )

    return filename


def reference_barrier_text():
    reference = scan.REFERENCE_BARRIERS.get(
        SYSTEM_NAME
    )

    if reference is None:
        return None

    low, high = (
        float(
            reference[0]
        ),
        float(
            reference[1]
        ),
    )

    if abs(
        high - low
    ) < 1.0e-12:
        return (
            f"{low:.3f} eV"
        )

    return (
        f"{low:.3f}-{high:.3f} eV"
    )


def report(
    result,
    *,
    relax,
):
    label = (
        "RELAXED"
        if relax
        else "FROZEN"
    )

    print()
    print(
        f"FULL FORMALDEHYDE SURFACE - {label}"
    )

    print(
        "="
        * (
            30
            + len(
                label
            )
        )
    )

    print(
        "reaction: H + CH2O -> H2 + HCO"
    )

    print(
        "physics: full ChemistryModel base + "
        "SAPT-wall H-state + heavy-atom SAPT environment"
    )

    print(
        f"H-state mixing: "
        f"{SAPT_H_STATE_MIXING:.9f}"
    )

    print(
        "SAPT parameters: frozen"
    )

    print(
        "formaldehyde parameters: not fitted by this scan"
    )

    print()

    print(
        "reactant:"
        f" donor C-H={result['reactant_donor']:.3f} A"
        f" forming H-H={result['reactant_transfer']:.3f} A"
        f" energy={result['reactant_energy']:.9f} eV"
    )

    print(
        "product: "
        f" donor C-H={result['product_donor']:.3f} A"
        f" forming H-H={result['product_transfer']:.3f} A"
        f" energy={result['product_energy']:.9f} eV"
    )

    print(
        f"reaction energy: "
        f"{result['reaction']:+.6f} eV"
    )

    reference_reaction = getattr(
        scan,
        "REFERENCE_REACTION",
        None,
    )

    if reference_reaction is not None:
        error = (
            result[
                "reaction"
            ]
            - float(
                reference_reaction
            )
        )

        print(
            f"project reaction reference: "
            f"{float(reference_reaction):+.3f} eV"
            f"  error={error:+.6f} eV"
        )

    print()

    print(
        "SADDLE"
    )

    print(
        f"  donor C-H   "
        f"{result['saddle_donor']:.3f} A"
    )

    print(
        f"  forming H-H "
        f"{result['saddle_transfer']:.3f} A"
    )

    print(
        f"  energy      "
        f"{result['saddle_energy']:.9f} eV"
    )

    print(
        f"  barrier     "
        f"{result['barrier']:.6f} eV"
    )

    print(
        "  spectators  "
        + np.array2string(
            result[
                "saddle_spectators"
            ],
            precision=5,
        )
    )

    reference_text = (
        reference_barrier_text()
    )

    if reference_text is not None:
        print()
        print(
            "project formaldehyde barrier reference: "
            + reference_text
        )

        low, high = scan.REFERENCE_BARRIERS[
            SYSTEM_NAME
        ]

        if (
            low
            <= result[
                "barrier"
            ]
            <= high
        ):
            print(
                "RESULT: barrier is inside the current project reference."
            )
        else:
            if result[
                "barrier"
            ] < low:
                miss = (
                    result[
                        "barrier"
                    ]
                    - low
                )
            else:
                miss = (
                    result[
                        "barrier"
                    ]
                    - high
                )

            print(
                "RESULT: barrier is outside the current project reference "
                f"by {miss:+.6f} eV."
            )

    print()

    print(
        "saddle spectator order:"
    )

    for name, value in zip(
        scan.SPECTATOR_NAMES[
            SYSTEM_NAME
        ],
        result[
            "saddle_spectators"
        ],
    ):
        print(
            f"  {name:<24} "
            f"{float(value):.5f}"
        )

    print()

    print(
        f"elapsed: "
        f"{result['elapsed']:.1f} s"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--relax",
        action="store_true",
        help=(
            "relax formaldehyde spectator coordinates at every grid cell"
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="use float32 instead of float64",
    )

    parser.add_argument(
        "--donor-step",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--transfer-step",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    configure_system()

    scan.SCAN_DEVICE = (
        args.device
    )

    scan.SCAN_DTYPE = (
        torch.float32
        if args.fast
        else torch.float64
    )

    if args.donor_step is None:
        donor_step = (
            0.04
            if args.relax
            else 0.02
        )
    else:
        donor_step = (
            args.donor_step
        )

    if args.transfer_step is None:
        transfer_step = (
            0.04
            if args.relax
            else 0.02
        )
    else:
        transfer_step = (
            args.transfer_step
        )

    donor_lengths, transfer_lengths = scan_axes(
        donor_step,
        transfer_step,
    )

    print(
        "SAPT FULL FORMALDEHYDE DIAGNOSTIC"
    )

    print(
        "================================"
    )

    print(
        f"mode: "
        f"{'relaxed' if args.relax else 'frozen'}"
    )

    print(
        f"grid: "
        f"{len(donor_lengths)} x {len(transfer_lengths)}"
        f" = {len(donor_lengths) * len(transfer_lengths)} cells"
    )

    print(
        f"grid step: donor {donor_step:.3f} A, "
        f"transfer {transfer_step:.3f} A"
    )

    print(
        f"device: "
        f"{scan.SCAN_DEVICE}"
    )

    print(
        f"dtype: "
        f"{scan.SCAN_DTYPE}"
    )

    print(
        "frozen spectators: "
        f"{np.asarray(scan.SYSTEMS[SYSTEM_NAME]['frozen']).tolist()}"
    )

    print(
        f"H-state mixing: "
        f"{SAPT_H_STATE_MIXING:.9f}"
    )

    print(
        "no parameter is fitted by this scan"
    )

    result = measure_grid(
        relax=args.relax,
        donor_step=donor_step,
        transfer_step=transfer_step,
    )

    report(
        result,
        relax=args.relax,
    )

    filename = save_result(
        result,
        relax=args.relax,
        donor_step=donor_step,
        transfer_step=transfer_step,
    )

    print(
        f"saved grid: "
        f"{filename}"
    )


if __name__ == "__main__":
    main()
