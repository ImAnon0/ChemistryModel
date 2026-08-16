"""
Diagnostic: how much H-state coupling does each reaction require?

This is NOT a proposed parameter fit.

The SAPT wall, Morse tables, environment softening and all other physics are
kept fixed. For each reaction, only the global H-state mixing scalar eta is
temporarily varied to answer:

    "What eta would this reaction need to reproduce its reference barrier?"

The purpose is to test whether the H3-calibrated eta transfers across
environments, and whether reactions sharing a donor pair (for example two C-H
abstractions) demand similar effective coupling.

Default systems:
    formaldehyde
    water
    methane

The frozen 2D barrier scan uses the same geometry builders, basins and minimax
logic as hf_surface_scan.py.

Water has a reference interval rather than one exact value, so this script
reports the eta interval that puts its frozen barrier inside that range.

Usage:
    py sapt_coupling_requirement_diagnostic.py

Finer surfaces:
    py sapt_coupling_requirement_diagnostic.py --donor-step 0.02 --transfer-step 0.02

Wider eta bracket:
    py sapt_coupling_requirement_diagnostic.py --eta-max 2.0
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

import reactive as R
import hf_surface_scan as scan

from sapt_h_state_torch import (
    SAPT_H_STATE_MIXING,
    SaptHStateBatchedSimulation,
)


H3_TARGET_BARRIER = 0.420

DEFAULT_SYSTEMS = (
    "formaldehyde",
    "water",
    "methane",
)

# Keep the corrected water convention used by the recent SAPT scans.
CORRECT_WATER_FROZEN = np.array(
    [
        0.960,
        0.960,
        75.53,
        75.53,
    ],
    dtype=float,
)


def parse_systems(text):
    values = tuple(
        part.strip()
        for part in text.split(",")
        if part.strip()
    )

    if not values:
        raise ValueError(
            "expected at least one system"
        )

    unknown = [
        name
        for name in values
        if name not in scan.SYSTEMS
    ]

    if unknown:
        raise ValueError(
            "unknown systems: "
            + ", ".join(unknown)
        )

    return values


def inclusive_axis(start, stop, step):
    return np.arange(
        start,
        stop + 0.5 * step,
        step,
        dtype=float,
    )


def numpy_energies(simulation):
    values = simulation.potential_per_box

    if torch.is_tensor(values):
        return (
            values
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )

    return np.asarray(
        values,
        dtype=float,
    )


def evaluate_raw_boxes(
    raw_boxes,
    *,
    mixing,
    device,
    dtype,
    batch_size,
):
    energies = []

    for start in range(
        0,
        len(raw_boxes),
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

        simulation = (
            SaptHStateBatchedSimulation(
                boxes=prepared,
                box_size=scan.BOX,
                random_seed=0,
                relax_on_start=False,
                device=device,
                dtype=dtype,
                h_state_mixing=float(mixing),
            )
        )

        energies.extend(
            numpy_energies(
                simulation
            ).tolist()
        )

    return np.asarray(
        energies,
        dtype=float,
    )


def frozen_spectators(system):
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


def frozen_barrier(
    system,
    eta,
    *,
    donor_step,
    transfer_step,
    device,
    dtype,
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

    (
        transfer_low,
        transfer_high,
        _,
    ) = probe[
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

    energies = evaluate_raw_boxes(
        raw_boxes,
        mixing=eta,
        device=device,
        dtype=dtype,
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
            f"{system}: no reactant/product basins"
        )

    saddle_cell, saddle_energy = (
        scan.flood_saddle(
            grid,
            reactant,
            product,
        )
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
        "reactant_donor": float(
            donors[
                reactant[
                    0
                ]
            ]
        ),
        "reactant_transfer": float(
            transfers[
                reactant[
                    1
                ]
            ]
        ),
    }


def h3_barrier(
    eta,
    *,
    r_min,
    r_max,
    points,
    device,
    dtype,
    batch_size,
):
    hydrogen = int(
        R.ELEMENT_INDEX[
            "H"
        ]
    )

    re = float(
        R.BOND_LENGTH[
            hydrogen,
            hydrogen,
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
        r_min,
        r_max,
        points,
    )

    raw_boxes = [
        (
            symbols,
            reactant,
        )
    ]

    for radius in radii:
        raw_boxes.append(
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
                            float(radius),
                            0.0,
                            0.0,
                        ],
                        [
                            2.0 * float(radius),
                            0.0,
                            0.0,
                        ],
                    ],
                    dtype=float,
                ),
            )
        )

    energies = evaluate_raw_boxes(
        raw_boxes,
        mixing=eta,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )

    reactant_energy = float(
        energies[
            0
        ]
    )

    seam = energies[
        1:
    ]

    index = int(
        np.argmin(
            seam
        )
    )

    return {
        "barrier": float(
            seam[
                index
            ]
            - reactant_energy
        ),
        "radius": float(
            radii[
                index
            ]
        ),
    }


def reference_range(system):
    reference = getattr(
        scan,
        "REFERENCE_BARRIERS",
        {},
    ).get(
        system
    )

    if reference is None:
        raise RuntimeError(
            f"{system}: no REFERENCE_BARRIERS entry"
        )

    low = float(
        reference[
            0
        ]
    )

    high = float(
        reference[
            1
        ]
    )

    return low, high


class BarrierCache:
    def __init__(
        self,
        *,
        donor_step,
        transfer_step,
        device,
        dtype,
        batch_size,
    ):
        self.donor_step = donor_step
        self.transfer_step = transfer_step
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.cache = {}

    def get(
        self,
        system,
        eta,
    ):
        key = (
            system,
            round(
                float(eta),
                12,
            ),
        )

        if key not in self.cache:
            result = frozen_barrier(
                system,
                float(eta),
                donor_step=(
                    self.donor_step
                ),
                transfer_step=(
                    self.transfer_step
                ),
                device=self.device,
                dtype=self.dtype,
                batch_size=self.batch_size,
            )

            self.cache[
                key
            ] = result

            print(
                f"    {system:>12}  eta={float(eta):.7f}  "
                f"barrier={result['barrier']:.6f} eV  "
                f"saddle={result['saddle_donor']:.3f}/"
                f"{result['saddle_transfer']:.3f} A"
            )

        return self.cache[
            key
        ]


def bracket_target(
    evaluator,
    target,
    *,
    eta_min,
    eta_max,
    samples,
):
    """
    Find an adjacent sampled eta interval where barrier-target changes sign.

    This does not assume globally monotonic behaviour.
    """

    etas = np.linspace(
        eta_min,
        eta_max,
        samples,
    )

    values = []

    for eta in etas:
        barrier = evaluator(
            float(eta)
        )[
            "barrier"
        ]

        residual = (
            barrier
            - target
        )

        values.append(
            (
                float(eta),
                residual,
            )
        )

        if abs(
            residual
        ) < 1.0e-12:
            return (
                float(eta),
                float(eta),
            )

    for (
        eta_a,
        residual_a,
    ), (
        eta_b,
        residual_b,
    ) in zip(
        values[
            :-1
        ],
        values[
            1:
        ],
    ):
        if (
            residual_a == 0.0
            or residual_b == 0.0
            or residual_a
            * residual_b
            < 0.0
        ):
            return (
                eta_a,
                eta_b,
            )

    return None


def solve_target_eta(
    evaluator,
    target,
    *,
    eta_min,
    eta_max,
    bracket_samples,
    tolerance,
    max_iterations,
):
    bracket = bracket_target(
        evaluator,
        target,
        eta_min=eta_min,
        eta_max=eta_max,
        samples=bracket_samples,
    )

    if bracket is None:
        return None

    left, right = bracket

    if left == right:
        return (
            left,
            evaluator(
                left
            ),
        )

    f_left = (
        evaluator(
            left
        )[
            "barrier"
        ]
        - target
    )

    for _ in range(
        max_iterations
    ):
        middle = (
            0.5
            * (
                left
                + right
            )
        )

        result = evaluator(
            middle
        )

        f_middle = (
            result[
                "barrier"
            ]
            - target
        )

        if abs(
            f_middle
        ) <= tolerance:
            return (
                middle,
                result,
            )

        if (
            f_left
            * f_middle
            <= 0.0
        ):
            right = middle
        else:
            left = middle
            f_left = f_middle

    eta = (
        0.5
        * (
            left
            + right
        )
    )

    return (
        eta,
        evaluator(
            eta
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--systems",
        default=",".join(
            DEFAULT_SYSTEMS
        ),
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
        "--eta-min",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--eta-max",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--bracket-samples",
        type=int,
        default=7,
        help=(
            "coarse eta samples used only to find a target crossing"
        ),
    )

    parser.add_argument(
        "--eta-tolerance",
        type=float,
        default=0.002,
        help=(
            "barrier tolerance in eV for the eta diagnostic"
        ),
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--h3-r-min",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--h3-r-max",
        type=float,
        default=1.30,
    )

    parser.add_argument(
        "--h3-points",
        type=int,
        default=261,
    )

    parser.add_argument(
        "--csv",
        default=(
            "sapt_coupling_requirement_diagnostic.csv"
        ),
    )

    args = parser.parse_args()

    systems = parse_systems(
        args.systems
    )

    if args.eta_max <= args.eta_min:
        raise ValueError(
            "--eta-max must be greater than --eta-min"
        )

    if args.bracket_samples < 2:
        raise ValueError(
            "--bracket-samples must be >= 2"
        )

    dtype = (
        torch.float32
        if args.fast
        else torch.float64
    )

    cache = BarrierCache(
        donor_step=args.donor_step,
        transfer_step=args.transfer_step,
        device=args.device,
        dtype=dtype,
        batch_size=args.batch_size,
    )

    print(
        "SAPT COUPLING-REQUIREMENT DIAGNOSTIC"
    )

    print(
        "===================================="
    )

    print(
        "No wall, Morse, environment or bridge parameters are fitted."
    )

    print(
        "Only eta is temporarily varied per reaction."
    )

    print(
        f"surface step: {args.donor_step:.3f}/"
        f"{args.transfer_step:.3f} A"
    )

    print(
        f"eta search: {args.eta_min:.3f} to "
        f"{args.eta_max:.3f}"
    )

    print()

    h3_current = h3_barrier(
        SAPT_H_STATE_MIXING,
        r_min=args.h3_r_min,
        r_max=args.h3_r_max,
        points=args.h3_points,
        device=args.device,
        dtype=dtype,
        batch_size=args.batch_size,
    )

    print(
        "H3 ANCHOR"
    )

    print(
        "---------"
    )

    print(
        f"current eta       {SAPT_H_STATE_MIXING:.9f}"
    )

    print(
        f"H3 barrier        {h3_current['barrier']:.6f} eV"
    )

    print(
        f"H3 target         {H3_TARGET_BARRIER:.6f} eV"
    )

    print(
        f"symmetric r       {h3_current['radius']:.5f} A"
    )

    print()

    rows = []

    started = time.monotonic()

    for system in systems:
        low, high = reference_range(
            system
        )

        print(
            f"{system.upper()}"
        )

        print(
            "-" * len(
                system
            )
        )

        baseline = cache.get(
            system,
            SAPT_H_STATE_MIXING,
        )

        evaluator = lambda eta, _system=system: cache.get(
            _system,
            eta,
        )

        if math.isclose(
            low,
            high,
            abs_tol=1.0e-12,
        ):
            target = low

            solved = solve_target_eta(
                evaluator,
                target,
                eta_min=args.eta_min,
                eta_max=args.eta_max,
                bracket_samples=(
                    args.bracket_samples
                ),
                tolerance=(
                    args.eta_tolerance
                ),
                max_iterations=(
                    args.max_iterations
                ),
            )

            if solved is None:
                print(
                    f"  no eta in search range brackets "
                    f"target {target:.6f} eV"
                )

                rows.append(
                    {
                        "system": system,
                        "reference_low_eV": low,
                        "reference_high_eV": high,
                        "baseline_eta": (
                            SAPT_H_STATE_MIXING
                        ),
                        "baseline_barrier_eV": (
                            baseline[
                                "barrier"
                            ]
                        ),
                        "status": (
                            "target not bracketed"
                        ),
                    }
                )
            else:
                eta, result = solved

                print(
                    f"  required eta      {eta:.7f}"
                )

                print(
                    f"  matched barrier   {result['barrier']:.6f} eV"
                )

                print(
                    f"  eta/H3 eta        "
                    f"{eta / SAPT_H_STATE_MIXING:.4f}"
                )

                rows.append(
                    {
                        "system": system,
                        "reference_low_eV": low,
                        "reference_high_eV": high,
                        "baseline_eta": (
                            SAPT_H_STATE_MIXING
                        ),
                        "baseline_barrier_eV": (
                            baseline[
                                "barrier"
                            ]
                        ),
                        "required_eta_low": eta,
                        "required_eta_high": eta,
                        "matched_barrier_low_eV": (
                            result[
                                "barrier"
                            ]
                        ),
                        "matched_barrier_high_eV": (
                            result[
                                "barrier"
                            ]
                        ),
                        "eta_over_h3_low": (
                            eta
                            / SAPT_H_STATE_MIXING
                        ),
                        "eta_over_h3_high": (
                            eta
                            / SAPT_H_STATE_MIXING
                        ),
                        "saddle_donor_A": (
                            result[
                                "saddle_donor"
                            ]
                        ),
                        "saddle_transfer_A": (
                            result[
                                "saddle_transfer"
                            ]
                        ),
                        "status": "ok",
                    }
                )

        else:
            # For a reference window, find eta at each barrier boundary.
            # Depending on barrier direction, the numerical eta order can be
            # reversed; sort afterwards.
            solved_low_barrier = solve_target_eta(
                evaluator,
                low,
                eta_min=args.eta_min,
                eta_max=args.eta_max,
                bracket_samples=(
                    args.bracket_samples
                ),
                tolerance=(
                    args.eta_tolerance
                ),
                max_iterations=(
                    args.max_iterations
                ),
            )

            solved_high_barrier = solve_target_eta(
                evaluator,
                high,
                eta_min=args.eta_min,
                eta_max=args.eta_max,
                bracket_samples=(
                    args.bracket_samples
                ),
                tolerance=(
                    args.eta_tolerance
                ),
                max_iterations=(
                    args.max_iterations
                ),
            )

            if (
                solved_low_barrier is None
                or solved_high_barrier is None
            ):
                print(
                    f"  could not bracket both reference "
                    f"boundaries {low:.6f} to {high:.6f} eV"
                )

                rows.append(
                    {
                        "system": system,
                        "reference_low_eV": low,
                        "reference_high_eV": high,
                        "baseline_eta": (
                            SAPT_H_STATE_MIXING
                        ),
                        "baseline_barrier_eV": (
                            baseline[
                                "barrier"
                            ]
                        ),
                        "status": (
                            "window not fully bracketed"
                        ),
                    }
                )
            else:
                eta_a, result_a = (
                    solved_low_barrier
                )

                eta_b, result_b = (
                    solved_high_barrier
                )

                ordered = sorted(
                    [
                        (
                            eta_a,
                            result_a,
                        ),
                        (
                            eta_b,
                            result_b,
                        ),
                    ],
                    key=lambda item: item[
                        0
                    ],
                )

                (
                    eta_low,
                    result_low_eta,
                ) = ordered[
                    0
                ]

                (
                    eta_high,
                    result_high_eta,
                ) = ordered[
                    1
                ]

                print(
                    f"  acceptable eta     "
                    f"{eta_low:.7f} to {eta_high:.7f}"
                )

                print(
                    f"  eta/H3 eta         "
                    f"{eta_low / SAPT_H_STATE_MIXING:.4f} "
                    f"to "
                    f"{eta_high / SAPT_H_STATE_MIXING:.4f}"
                )

                print(
                    f"  boundary barriers  "
                    f"{result_low_eta['barrier']:.6f}, "
                    f"{result_high_eta['barrier']:.6f} eV"
                )

                rows.append(
                    {
                        "system": system,
                        "reference_low_eV": low,
                        "reference_high_eV": high,
                        "baseline_eta": (
                            SAPT_H_STATE_MIXING
                        ),
                        "baseline_barrier_eV": (
                            baseline[
                                "barrier"
                            ]
                        ),
                        "required_eta_low": eta_low,
                        "required_eta_high": eta_high,
                        "matched_barrier_low_eV": (
                            result_low_eta[
                                "barrier"
                            ]
                        ),
                        "matched_barrier_high_eV": (
                            result_high_eta[
                                "barrier"
                            ]
                        ),
                        "eta_over_h3_low": (
                            eta_low
                            / SAPT_H_STATE_MIXING
                        ),
                        "eta_over_h3_high": (
                            eta_high
                            / SAPT_H_STATE_MIXING
                        ),
                        "saddle_donor_A": (
                            result_low_eta[
                                "saddle_donor"
                            ]
                        ),
                        "saddle_transfer_A": (
                            result_low_eta[
                                "saddle_transfer"
                            ]
                        ),
                        "status": "ok",
                    }
                )

        print()

    fields = [
        "system",
        "reference_low_eV",
        "reference_high_eV",
        "baseline_eta",
        "baseline_barrier_eV",
        "required_eta_low",
        "required_eta_high",
        "matched_barrier_low_eV",
        "matched_barrier_high_eV",
        "eta_over_h3_low",
        "eta_over_h3_high",
        "saddle_donor_A",
        "saddle_transfer_A",
        "status",
    ]

    with open(
        args.csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(
                        field,
                        ""
                    )
                    for field in fields
                }
            )

    elapsed = (
        time.monotonic()
        - started
    )

    print(
        "SUMMARY"
    )

    print(
        "======="
    )

    print(
        f"{'system':>14}  "
        f"{'baseline':>10}  "
        f"{'required eta':>24}  "
        f"{'eta/H3':>18}"
    )

    for row in rows:
        status = row[
            "status"
        ]

        if status != "ok":
            eta_text = status
            ratio_text = "---"
        else:
            low_eta = row[
                "required_eta_low"
            ]

            high_eta = row[
                "required_eta_high"
            ]

            low_ratio = row[
                "eta_over_h3_low"
            ]

            high_ratio = row[
                "eta_over_h3_high"
            ]

            if math.isclose(
                low_eta,
                high_eta,
                abs_tol=1.0e-10,
            ):
                eta_text = (
                    f"{low_eta:.6f}"
                )

                ratio_text = (
                    f"{low_ratio:.3f}"
                )
            else:
                eta_text = (
                    f"{low_eta:.6f}.."
                    f"{high_eta:.6f}"
                )

                ratio_text = (
                    f"{low_ratio:.3f}.."
                    f"{high_ratio:.3f}"
                )

        print(
            f"{row['system']:>14}  "
            f"{row['baseline_barrier_eV']:10.3f}  "
            f"{eta_text:>24}  "
            f"{ratio_text:>18}"
        )

    print()

    print(
        f"saved: {args.csv}"
    )

    print(
        f"elapsed: {elapsed:.1f} s"
    )

    print()

    print(
        "Interpretation:"
    )

    print(
        "  Similar required eta values for chemically related reactions "
        "support a pair- or descriptor-dependent coupling law."
    )

    print(
        "  Very different eta values imply stronger environment dependence."
    )

    print(
        "  This script is diagnostic only: do not promote these fitted eta "
        "values directly into production parameters."
    )


if __name__ == "__main__":
    main()
