
"""
Full ChemistryModel water transfer scan using the SAPT-wall H-state adapter.

This is a diagnostic only. It does not modify production reactive_torch.py.

First run:
    py sapt_full_water_scan.py

Optional relaxed run after the frozen result is inspected:
    py sapt_full_water_scan.py --relax

The water spectator angles are corrected to 75.53 degrees from the
transfer axis, corresponding to the physical 104.47 degree H-O-H angle.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import hf_surface_scan as scan
from sapt_h_state_torch import (
    SaptHStateBatchedSimulation,
    SAPT_H_STATE_MIXING,
)


CORRECT_WATER_FROZEN = np.array(
    [0.960, 0.960, 75.53, 75.53],
    dtype=float,
)


def install_water_geometry_fix():
    # hf_surface_scan's comments already establish that 75.53 degrees
    # from the transfer axis corresponds to H-O-H = 104.47 degrees.
    # Update both the module global used by water_geometry() and the
    # SYSTEMS registry used by active_frozen()/relaxation.
    scan.WATER_FROZEN = CORRECT_WATER_FROZEN.copy()
    scan.SYSTEMS["water"]["frozen"] = CORRECT_WATER_FROZEN.copy()

    scan.apply_system("water")


def build_sapt(
    physics="sapt",
    *,
    boxes=None,
    **_ignored,
):
    """
    Scanner-compatible builder.

    Frozen scans can therefore keep using hf_surface_scan's batched grid
    evaluator instead of evaluating thousands of cells one by one.
    """

    if boxes is None:
        symbols, positions = scan.active_geometry(
            *scan.SYSTEM_START["water"]
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


def measure_grid(
    *,
    relax,
    donor_step,
    transfer_step,
):
    donor_lengths = np.arange(
        0.90,
        1.80 + 0.5*donor_step,
        donor_step,
    )

    transfer_lengths = np.arange(
        0.90,
        1.80 + 0.5*transfer_step,
        transfer_step,
    )

    # Monkeypatch only inside this diagnostic process.
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
                progress_label="water SAPT",
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
                progress_label="water SAPT",
                physics="sapt",
                build_kwargs={},
                gradient_based=True,
            )

            spectators = np.broadcast_to(
                CORRECT_WATER_FROZEN,
                (
                    len(donor_lengths),
                    len(transfer_lengths),
                    4,
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
            "Could not identify water reactant/product basins. "
            "The scan window may need widening."
        )

    saddle_cell, saddle_energy = scan.flood_saddle(
        grid,
        reactant,
        product,
    )

    if saddle_cell is None:
        raise RuntimeError(
            "No connected path between water basins."
        )

    reactant_energy = float(
        grid[reactant]
    )

    product_energy = float(
        grid[product]
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

    result = {
        "grid": grid,
        "spectators": spectators,
        "donor_lengths": donor_lengths,
        "transfer_lengths": transfer_lengths,
        "reactant": reactant,
        "product": product,
        "saddle_cell": saddle_cell,
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
        "elapsed": elapsed,
    }

    return result


def save_result(
    result,
    *,
    relax,
):
    label = (
        "relaxed"
        if relax
        else "frozen"
    )

    filename = (
        f"sapt_full_water_{label}.npz"
    )

    np.savez_compressed(
        filename,
        grid=result["grid"],
        spectators=result["spectators"],
        donor_lengths=result["donor_lengths"],
        transfer_lengths=result["transfer_lengths"],
        reactant=np.asarray(
            result["reactant"]
        ),
        product=np.asarray(
            result["product"]
        ),
        saddle_cell=np.asarray(
            result["saddle_cell"]
        ),
    )

    return filename


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
        f"FULL WATER SURFACE — {label}"
    )
    print(
        "=" * (
            21 + len(label)
        )
    )

    print(
        "physics: full ChemistryModel base + "
        "SAPT-wall H-state"
    )

    print(
        f"H-state mixing: "
        f"{SAPT_H_STATE_MIXING:.9f}"
    )

    print(
        "SAPT parameters: frozen"
    )

    print(
        "water parameters: not fitted"
    )

    print()

    print(
        "reactant:"
        f" donor={result['reactant_donor']:.3f} A"
        f" transfer={result['reactant_transfer']:.3f} A"
    )

    print(
        "product: "
        f" donor={result['product_donor']:.3f} A"
        f" transfer={result['product_transfer']:.3f} A"
    )

    print(
        f"reaction energy: "
        f"{result['reaction']:+.6f} eV"
    )

    print()

    print(
        "SADDLE"
    )

    print(
        f"  donor     "
        f"{result['saddle_donor']:.3f} A"
    )

    print(
        f"  transfer  "
        f"{result['saddle_transfer']:.3f} A"
    )

    print(
        f"  O-O       "
        f"{result['saddle_donor'] + result['saddle_transfer']:.3f} A"
    )

    print(
        f"  barrier   "
        f"{result['barrier']:.6f} eV"
    )

    print(
        "  spectators "
        + np.array2string(
            result[
                "saddle_spectators"
            ],
            precision=5,
        )
    )

    print()

    print(
        "reference interval used by the project:"
        " 0.364–0.525 eV"
    )

    print(
        "radial-only comparison:"
        " rerun sapt_transfer_diagnostic.py with the matching"
        " four-neighbour minimax before quoting a value"
    )

    print()

    if (
        0.364
        <= result["barrier"]
        <= 0.525
    ):
        print(
            "RESULT: full-potential barrier remains "
            "inside the comparison interval."
        )
    else:
        print(
            "RESULT: full-potential terms move the "
            "barrier outside the comparison interval."
        )

    print(
        f"elapsed: {result['elapsed']:.1f} s"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--relax",
        action="store_true",
        help=(
            "relax water spectator bond lengths/angles "
            "at every grid cell"
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

    install_water_geometry_fix()

    scan.SCAN_DEVICE = args.device
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

    print(
        "SAPT FULL WATER DIAGNOSTIC"
    )

    print(
        "=========================="
    )

    print(
        f"mode: "
        f"{'relaxed' if args.relax else 'frozen'}"
    )

    print(
        f"grid step: donor {donor_step:.3f} A, "
        f"transfer {transfer_step:.3f} A"
    )

    print(
        f"device: {scan.SCAN_DEVICE}"
    )

    print(
        f"dtype: {scan.SCAN_DTYPE}"
    )

    print(
        "corrected frozen spectators: "
        f"{CORRECT_WATER_FROZEN.tolist()}"
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
    )

    print(
        f"saved grid: {filename}"
    )


if __name__ == "__main__":
    main()
