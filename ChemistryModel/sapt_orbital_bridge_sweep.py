"""
Screen a generic short-range orbital bridge across H-transfer systems.

For every bridge strength this script first re-anchors ONLY the H-state
mixing scalar to the H + H2 -> H2 + H barrier target of 0.420 eV. The SAPT
wall parameters, Morse tables, environment softening and bridge functional
form are not refitted.

It then measures coarse frozen full-system minimax barriers for:
    formaldehyde
    water
    methane

The published/project comparison ranges are read from hf_surface_scan.py
rather than copied here.

This is intentionally a screening pass. If one region looks promising, rerun
that small region with finer grid steps and then do relaxed surfaces.

Examples
--------
Default screening:
    py sapt_orbital_bridge_sweep.py

Narrow around a promising region:
    py sapt_orbital_bridge_sweep.py --strengths 0.06,0.08,0.10,0.12,0.14

Finer transfer coordinate:
    py sapt_orbital_bridge_sweep.py --transfer-step 0.01

Include ammonia as an extra holdout:
    py sapt_orbital_bridge_sweep.py --systems formaldehyde,water,methane,ammonia
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

import reactive as R
import hf_surface_scan as scan

from sapt_h_state_torch import (
    SAPT_H_STATE_MIXING,
)

from sapt_orbital_bridge_torch import (
    OrbitalBridgeSaptHStateBatchedSimulation,
)


H3_TARGET_BARRIER = 0.420

DEFAULT_STRENGTHS = (
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
)

DEFAULT_SYSTEMS = (
    "formaldehyde",
    "water",
    "methane",
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


def parse_csv_floats(
    text,
):
    values = tuple(
        float(
            part.strip()
        )
        for part in text.split(
            ","
        )
        if part.strip()
    )

    if not values:
        raise ValueError(
            "expected at least one numeric value"
        )

    return values


def parse_csv_strings(
    text,
):
    values = tuple(
        part.strip()
        for part in text.split(
            ","
        )
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
            + ", ".join(
                unknown
            )
        )

    return values


def numpy_energies(
    simulation,
):
    values = simulation.potential_per_box

    if torch.is_tensor(
        values
    ):
        return (
            values
            .detach()
            .cpu()
            .numpy()
            .astype(
                float
            )
        )

    return np.asarray(
        values,
        dtype=float,
    )


def evaluate_raw_boxes(
    raw_boxes,
    *,
    mixing,
    strength,
    device,
    dtype,
    batch_size,
):
    """
    Evaluate arbitrary same-size boxes in chunks.

    ``raw_boxes`` contain unshifted molecular coordinates. This helper adds the
    scanner centre exactly once before constructing each batched simulation.
    """

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
            OrbitalBridgeSaptHStateBatchedSimulation(
                boxes=prepared,
                box_size=scan.BOX,
                random_seed=0,
                relax_on_start=False,
                device=device,
                dtype=dtype,
                h_state_mixing=float(
                    mixing
                ),
                orbital_bridge_strength=float(
                    strength
                ),
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


def h3_geometries(
    *,
    r_min,
    r_max,
    points,
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

    symmetric = [
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
        )
        for radius in radii
    ]

    raw_boxes = [
        (
            symbols,
            reactant,
        )
    ] + [
        (
            symbols,
            geometry,
        )
        for geometry in symmetric
    ]

    return (
        radii,
        raw_boxes,
    )


def h3_barrier(
    mixing,
    strength,
    *,
    device,
    dtype,
    batch_size,
    r_min,
    r_max,
    points,
):
    radii, boxes = h3_geometries(
        r_min=r_min,
        r_max=r_max,
        points=points,
    )

    energies = evaluate_raw_boxes(
        boxes,
        mixing=mixing,
        strength=strength,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )

    reactant = float(
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

    saddle = float(
        seam[
            index
        ]
    )

    return {
        "barrier": (
            saddle
            - reactant
        ),
        "radius": float(
            radii[
                index
            ]
        ),
        "reactant": reactant,
        "saddle": saddle,
    }


def fit_h3_mixing(
    strength,
    *,
    device,
    dtype,
    batch_size,
    r_min,
    r_max,
    points,
    low=0.0,
    high=1.5,
    tolerance=1.0e-6,
    max_iterations=32,
):
    """
    Re-anchor only the state-coupling scalar for this bridge strength.

    Returns None when the 0.420 eV target cannot be bracketed with a
    non-negative mixing in [low, high]. That is useful information: it means
    the bridge alone has already lowered H3 too far.
    """

    cache = {}

    def result(
        mixing,
    ):
        key = float(
            mixing
        )

        if key not in cache:
            cache[
                key
            ] = h3_barrier(
                key,
                strength,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                r_min=r_min,
                r_max=r_max,
                points=points,
            )

        return cache[
            key
        ]

    def residual(
        mixing,
    ):
        return (
            result(
                mixing
            )[
                "barrier"
            ]
            - H3_TARGET_BARRIER
        )

    f_low = residual(
        low
    )

    f_high = residual(
        high
    )

    if abs(
        f_low
    ) <= tolerance:
        return (
            float(
                low
            ),
            result(
                low
            ),
        )

    if abs(
        f_high
    ) <= tolerance:
        return (
            float(
                high
            ),
            result(
                high
            ),
        )

    if (
        f_low
        * f_high
        > 0.0
    ):
        return None

    left = float(
        low
    )

    right = float(
        high
    )

    for _ in range(
        max_iterations
    ):
        middle = 0.5 * (
            left
            + right
        )

        f_middle = residual(
            middle
        )

        if abs(
            f_middle
        ) <= tolerance:
            return (
                middle,
                result(
                    middle
                ),
            )

        if (
            f_low
            * f_middle
            <= 0.0
        ):
            right = middle
            f_high = f_middle
        else:
            left = middle
            f_low = f_middle

    fitted = 0.5 * (
        left
        + right
    )

    return (
        fitted,
        result(
            fitted
        ),
    )


def inclusive_axis(
    start,
    stop,
    step,
):
    values = np.arange(
        start,
        stop
        + 0.5
        * step,
        step,
        dtype=float,
    )

    return values


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


def measure_frozen_system(
    system,
    *,
    mixing,
    strength,
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

    donor_lengths = inclusive_axis(
        donor_low,
        donor_high,
        donor_step,
    )

    transfer_lengths = inclusive_axis(
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

    for donor in donor_lengths:
        for transfer in transfer_lengths:
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
        mixing=mixing,
        strength=strength,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )

    grid = energies.reshape(
        len(
            donor_lengths
        ),
        len(
            transfer_lengths
        ),
    )

    reactant, product = scan.basin_seeds(
        grid,
        donor_lengths,
        transfer_lengths,
    )

    if reactant is None:
        return {
            "status": (
                "no basins"
            ),
            "grid_shape": grid.shape,
        }

    (
        saddle_cell,
        saddle_energy,
    ) = scan.flood_saddle(
        grid,
        reactant,
        product,
    )

    if saddle_cell is None:
        return {
            "status": (
                "no route"
            ),
            "grid_shape": grid.shape,
        }

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

    reference = getattr(
        scan,
        "REFERENCE_BARRIERS",
        {},
    ).get(
        system
    )

    if reference is None:
        error = np.nan
        ref_low = np.nan
        ref_high = np.nan
    else:
        ref_low = float(
            reference[
                0
            ]
        )

        ref_high = float(
            reference[
                1
            ]
        )

        if barrier < ref_low:
            error = (
                barrier
                - ref_low
            )
        elif barrier > ref_high:
            error = (
                barrier
                - ref_high
            )
        else:
            error = 0.0

    return {
        "status": "ok",
        "grid_shape": grid.shape,
        "barrier": barrier,
        "reaction": reaction,
        "saddle_donor": float(
            donor_lengths[
                saddle_cell[
                    0
                ]
            ]
        ),
        "saddle_transfer": float(
            transfer_lengths[
                saddle_cell[
                    1
                ]
            ]
        ),
        "reactant_donor": float(
            donor_lengths[
                reactant[
                    0
                ]
            ]
        ),
        "reactant_transfer": float(
            transfer_lengths[
                reactant[
                    1
                ]
            ]
        ),
        "reference_low": ref_low,
        "reference_high": ref_high,
        "reference_error": float(
            error
        ),
    }


def write_csv(
    path,
    rows,
):
    fields = [
        "bridge_strength",
        "h3_mixing",
        "h3_barrier_eV",
        "h3_symmetric_r_A",
        "system",
        "status",
        "barrier_eV",
        "reference_low_eV",
        "reference_high_eV",
        "reference_error_eV",
        "reaction_eV",
        "saddle_donor_A",
        "saddle_transfer_A",
        "reactant_donor_A",
        "reactant_transfer_A",
        "donor_step_A",
        "transfer_step_A",
    ]

    with open(
        path,
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


def fmt_barrier(
    result,
):
    if (
        result is None
        or result.get(
            "status"
        )
        != "ok"
    ):
        return "   ---   "

    return (
        f"{result['barrier']:7.3f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--strengths",
        default=",".join(
            f"{value:g}"
            for value
            in DEFAULT_STRENGTHS
        ),
        help=(
            "comma-separated dimensionless bridge strengths"
        ),
    )

    parser.add_argument(
        "--systems",
        default=",".join(
            DEFAULT_SYSTEMS
        ),
        help=(
            "comma-separated systems registered in hf_surface_scan.py"
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
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="use float32; default is float64",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--h3-points",
        type=int,
        default=261,
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
        "--mixing-max",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--csv",
        default="sapt_orbital_bridge_sweep.csv",
    )

    args = parser.parse_args()

    if args.donor_step <= 0.0:
        raise ValueError(
            "--donor-step must be positive"
        )

    if args.transfer_step <= 0.0:
        raise ValueError(
            "--transfer-step must be positive"
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    if args.h3_points < 3:
        raise ValueError(
            "--h3-points must be >= 3"
        )

    strengths = parse_csv_floats(
        args.strengths
    )

    systems = parse_csv_strings(
        args.systems
    )

    dtype = (
        torch.float32
        if args.fast
        else torch.float64
    )

    print(
        "SAPT ORBITAL-BRIDGE SCREEN"
    )

    print(
        "=========================="
    )

    print(
        "SAPT wall: frozen"
    )

    print(
        "bridge form: generic transfer-overlap x existing Morse attraction"
    )

    print(
        f"H3 target: {H3_TARGET_BARRIER:.3f} eV"
    )

    print(
        "every bridge strength gets its own H3-only mixing re-anchor"
    )

    print(
        f"systems: {', '.join(systems)}"
    )

    print(
        f"frozen grid step: donor {args.donor_step:.3f} A, "
        f"transfer {args.transfer_step:.3f} A"
    )

    print(
        f"device/dtype: {args.device} / {dtype}"
    )

    print()

    current_h3 = h3_barrier(
        SAPT_H_STATE_MIXING,
        0.0,
        device=args.device,
        dtype=dtype,
        batch_size=args.batch_size,
        r_min=args.h3_r_min,
        r_max=args.h3_r_max,
        points=args.h3_points,
    )

    print(
        "CURRENT ADAPTER SANITY CHECK"
    )

    print(
        "----------------------------"
    )

    print(
        f"lambda=0, current mixing {SAPT_H_STATE_MIXING:.9f}: "
        f"full-adapter H3 barrier {current_h3['barrier']:.6f} eV "
        f"at r={current_h3['radius']:.5f} A"
    )

    print(
        "If this is not very close to 0.420 eV, the integrated adapter and "
        "the earlier standalone H3 calibration are not exactly the same "
        "calibration problem; the lambda=0 fitted row below makes that "
        "difference explicit rather than hiding it."
    )

    print()

    header = (
        f"{'lambda':>7}"
        f"{'mixing':>12}"
        f"{'H3':>9}"
        + "".join(
            f"{name:>14}"
            for name in systems
        )
        + f"{'worst |err|':>14}"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    rows = []

    started = time.monotonic()

    for strength in strengths:
        fitted = fit_h3_mixing(
            strength,
            device=args.device,
            dtype=dtype,
            batch_size=args.batch_size,
            r_min=args.h3_r_min,
            r_max=args.h3_r_max,
            points=args.h3_points,
            low=0.0,
            high=args.mixing_max,
        )

        if fitted is None:
            print(
                f"{strength:7.3f}"
                f"{'no H3 fit':>12}"
            )

            rows.append(
                {
                    "bridge_strength": strength,
                    "status": (
                        "H3 target not bracketed"
                    ),
                    "donor_step_A": args.donor_step,
                    "transfer_step_A": args.transfer_step,
                }
            )

            continue

        mixing, h3 = fitted

        results = {}

        for system in systems:
            results[
                system
            ] = measure_frozen_system(
                system,
                mixing=mixing,
                strength=strength,
                donor_step=args.donor_step,
                transfer_step=args.transfer_step,
                device=args.device,
                dtype=dtype,
                batch_size=args.batch_size,
            )

        errors = [
            abs(
                result[
                    "reference_error"
                ]
            )
            for result
            in results.values()
            if (
                result.get(
                    "status"
                )
                == "ok"
                and np.isfinite(
                    result[
                        "reference_error"
                    ]
                )
            )
        ]

        worst = (
            max(
                errors
            )
            if errors
            else np.nan
        )

        line = (
            f"{strength:7.3f}"
            f"{mixing:12.6f}"
            f"{h3['barrier']:9.3f}"
            + "".join(
                f"{fmt_barrier(results[name]):>14}"
                for name in systems
            )
            + (
                f"{worst:14.3f}"
                if np.isfinite(
                    worst
                )
                else f"{'---':>14}"
            )
        )

        print(
            line
        )

        for system, result in results.items():
            row = {
                "bridge_strength": strength,
                "h3_mixing": mixing,
                "h3_barrier_eV": h3[
                    "barrier"
                ],
                "h3_symmetric_r_A": h3[
                    "radius"
                ],
                "system": system,
                "status": result.get(
                    "status",
                    "unknown",
                ),
                "donor_step_A": args.donor_step,
                "transfer_step_A": args.transfer_step,
            }

            if result.get(
                "status"
            ) == "ok":
                row.update(
                    {
                        "barrier_eV": result[
                            "barrier"
                        ],
                        "reference_low_eV": result[
                            "reference_low"
                        ],
                        "reference_high_eV": result[
                            "reference_high"
                        ],
                        "reference_error_eV": result[
                            "reference_error"
                        ],
                        "reaction_eV": result[
                            "reaction"
                        ],
                        "saddle_donor_A": result[
                            "saddle_donor"
                        ],
                        "saddle_transfer_A": result[
                            "saddle_transfer"
                        ],
                        "reactant_donor_A": result[
                            "reactant_donor"
                        ],
                        "reactant_transfer_A": result[
                            "reactant_transfer"
                        ],
                    }
                )

            rows.append(
                row
            )

    csv_path = Path(
        args.csv
    )

    write_csv(
        csv_path,
        rows,
    )

    elapsed = (
        time.monotonic()
        - started
    )

    print()

    print(
        "REFERENCE RANGES USED"
    )

    print(
        "---------------------"
    )

    for system in systems:
        reference = getattr(
            scan,
            "REFERENCE_BARRIERS",
            {},
        ).get(
            system
        )

        if reference is None:
            print(
                f"{system:>14}: none in scanner"
            )
        else:
            low, high = reference

            if low == high:
                text = (
                    f"{float(low):.3f} eV"
                )
            else:
                text = (
                    f"{float(low):.3f} to "
                    f"{float(high):.3f} eV"
                )

            print(
                f"{system:>14}: {text}"
            )

    print()

    print(
        f"saved: {csv_path}"
    )

    print(
        f"elapsed: {elapsed:.1f} s"
    )

    print()

    print(
        "Interpretation: a promising lambda is one where the H3-reanchored "
        "model improves formaldehyde while water and methane remain sensible. "
        "Do not choose a value from one barrier alone. Re-run only the "
        "promising region at finer resolution, then relax the spectator "
        "coordinates before considering the architecture successful."
    )


if __name__ == "__main__":
    main()
