"""
Diagnostic: use the measured FULL-SAPT / EXCH10 ratio on unoccupied heavy-H
state edges, without fitting to any reaction barrier.

This is deliberately a narrow hypothesis test.

Data source:
    research_data/sapt/sapt_component_probe.csv

Ratio curves are derived only from bond-axis molecular-probe rows:
    C-H : CH4 + CH2O, approach == CH_bond
    O-H : H2O,       approach == OH_bond
    N-H : NH3,       approach == NH_bond

At each stored distance:
    scale(r) = mean[ SAPT_TOTAL / SAPT_EXCH10 ]

The scale is linearly interpolated in distance and clamped outside the sampled
range. H-H is left EXACTLY unchanged, so the H3 anchor/coupling construction is
not retuned.

Nothing is fitted to formaldehyde, methane, water, or H3 barriers.

The current frozen SAPT H-state model is compared against this temporary
"full-SAPT-ratio" wall for:
    H3
    formaldehyde
    water
    methane

This is NOT production physics. It tests whether the component omission found
by the Psi4 probe is large enough, in the correct direction, to explain the
barrier problem.

Usage:
    python sapt_total_ratio_wall_screen.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

import reactive as R
import hf_surface_scan as scan
import sapt_h_state_torch as sht

from sapt_coupling_requirement_diagnostic import (
    evaluate_raw_boxes,
    inclusive_axis,
)


DEFAULT_COMPONENT_CSV = Path(
    "research_data/sapt/sapt_component_probe.csv"
)

DEFAULT_DONOR_STEP = 0.04
DEFAULT_TRANSFER_STEP = 0.04
DEFAULT_BATCH_SIZE = 512

CORRECT_WATER_FROZEN = np.array(
    [
        0.960,
        0.960,
        75.53,
        75.53,
    ],
    dtype=float,
)


def _as_float(
    row,
    field,
):
    return float(
        row[
            field
        ]
    )


def load_ratio_curves(
    path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"component probe CSV not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = [
            row
            for row in csv.DictReader(
                handle
            )
            if row.get(
                "status",
                ""
            ).strip().lower()
            == "ok"
        ]

    if not rows:
        raise RuntimeError(
            f"no successful component rows found in {path}"
        )

    selectors = {
        "C-H": lambda row: (
            row[
                "system"
            ]
            in {
                "CH4",
                "CH2O",
            }
            and row[
                "approach"
            ]
            == "CH_bond"
        ),
        "O-H": lambda row: (
            row[
                "system"
            ]
            == "H2O"
            and row[
                "approach"
            ]
            == "OH_bond"
        ),
        "N-H": lambda row: (
            row[
                "system"
            ]
            == "NH3"
            and row[
                "approach"
            ]
            == "NH_bond"
        ),
    }

    curves = {}

    for pair_name, selector in selectors.items():
        grouped = defaultdict(
            list
        )

        for row in rows:
            if not selector(
                row
            ):
                continue

            exchange = _as_float(
                row,
                "rerun_exch10_eV",
            )

            total = _as_float(
                row,
                "sapt_total_eV",
            )

            if abs(
                exchange
            ) < 1.0e-12:
                continue

            distance = _as_float(
                row,
                "contact_distance_A",
            )

            grouped[
                distance
            ].append(
                total
                / exchange
            )

        if not grouped:
            raise RuntimeError(
                f"no ratio data found for {pair_name}"
            )

        distances = sorted(
            grouped
        )

        ratios = [
            float(
                np.mean(
                    grouped[
                        distance
                    ]
                )
            )
            for distance in distances
        ]

        curves[
            pair_name
        ] = (
            np.asarray(
                distances,
                dtype=float,
            ),
            np.asarray(
                ratios,
                dtype=float,
            ),
        )

    return curves


def print_curves(
    curves,
):
    print(
        "DATA-DERIVED BOND-AXIS TOTAL/EXCH10 RATIOS"
    )
    print(
        "=========================================="
    )

    for pair_name in (
        "C-H",
        "O-H",
        "N-H",
    ):
        distances, ratios = curves[
            pair_name
        ]

        values = "  ".join(
            f"{distance:.2f}A:{ratio:.4f}"
            for distance, ratio
            in zip(
                distances,
                ratios,
            )
        )

        print(
            f"{pair_name:>3}  {values}"
        )

    print(
        "H-H  unchanged: 1.0000"
    )
    print()


def fragment_symbol(
    fragment,
    index,
):
    symbols = getattr(
        fragment,
        "symbols",
        None,
    )

    if symbols is None:
        symbols = getattr(
            fragment,
            "_symbols",
            None,
        )

    if symbols is None:
        raise RuntimeError(
            "ContinuousTorchFragment does not expose symbols; "
            "cannot apply pair-specific diagnostic scaling"
        )

    return str(
        symbols[
            index
        ]
    )


def interpolate_torch(
    distance,
    xs_numpy,
    ys_numpy,
):
    xs = torch.as_tensor(
        xs_numpy,
        dtype=distance.dtype,
        device=distance.device,
    )

    ys = torch.as_tensor(
        ys_numpy,
        dtype=distance.dtype,
        device=distance.device,
    )

    result = ys[
        0
    ]

    result = torch.where(
        distance
        <= xs[
            0
        ],
        ys[
            0
        ],
        result,
    )

    for index in range(
        len(
            xs
        )
        - 1
    ):
        left_x = xs[
            index
        ]

        right_x = xs[
            index
            + 1
        ]

        left_y = ys[
            index
        ]

        right_y = ys[
            index
            + 1
        ]

        fraction = (
            distance
            - left_x
        ) / (
            right_x
            - left_x
        )

        segment = (
            left_y
            + fraction
            * (
                right_y
                - left_y
            )
        )

        inside = (
            (distance >= left_x)
            & (distance <= right_x)
        )

        result = torch.where(
            inside,
            segment,
            result,
        )

    result = torch.where(
        distance
        >= xs[
            -1
        ],
        ys[
            -1
        ],
        result,
    )

    return result


@contextmanager
def full_sapt_ratio_wall(
    curves,
):
    original = sht._sapt_pair_energy

    def scaled_pair_energy(
        fragment,
        first,
        second,
    ):
        raw = original(
            fragment,
            first,
            second,
        )

        first_symbol = fragment_symbol(
            fragment,
            first,
        )

        second_symbol = fragment_symbol(
            fragment,
            second,
        )

        symbols = {
            first_symbol,
            second_symbol,
        }

        # The component dataset does not supply an appropriate open-shell
        # full-SAPT H-H replacement. Keep H3/H-H exactly on the frozen model.
        if symbols == {
            "H",
        }:
            return raw

        if "H" not in symbols:
            return raw

        heavy = (
            second_symbol
            if first_symbol == "H"
            else first_symbol
        )

        pair_name = (
            f"{heavy}-H"
        )

        if pair_name not in curves:
            return raw

        positions = fragment.positions

        distance = torch.linalg.vector_norm(
            positions[
                first
            ]
            - positions[
                second
            ]
        )

        xs, ys = curves[
            pair_name
        ]

        scale = interpolate_torch(
            distance,
            xs,
            ys,
        )

        return (
            raw
            * scale
        )

    sht._sapt_pair_energy = (
        scaled_pair_energy
    )

    try:
        yield
    finally:
        sht._sapt_pair_energy = (
            original
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


def evaluate_surface(
    system,
    *,
    donor_step,
    transfer_step,
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
        mixing=sht.SAPT_H_STATE_MIXING,
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
            f"{system}: no minimax saddle"
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


def h3_barrier(
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

    energies = evaluate_raw_boxes(
        boxes,
        mixing=sht.SAPT_H_STATE_MIXING,
        device="cpu",
        dtype=torch.float64,
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

    return {
        "barrier": float(
            seam[
                index
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--components",
        type=Path,
        default=DEFAULT_COMPONENT_CSV,
    )

    parser.add_argument(
        "--donor-step",
        type=float,
        default=DEFAULT_DONOR_STEP,
    )

    parser.add_argument(
        "--transfer-step",
        type=float,
        default=DEFAULT_TRANSFER_STEP,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    args = parser.parse_args()

    curves = load_ratio_curves(
        args.components
    )

    print(
        "FULL-SAPT / EXCH10 HEAVY-H WALL SCREEN"
    )
    print(
        "======================================="
    )
    print(
        "No reaction barrier is used to fit this correction."
    )
    print(
        "Only bond-axis SAPT TOTAL/EXCH10 ratios from the component CSV are used."
    )
    print()

    print_curves(
        curves
    )

    current_h3 = h3_barrier(
        batch_size=args.batch_size,
    )

    current = {}

    for system in (
        "formaldehyde",
        "water",
        "methane",
    ):
        current[
            system
        ] = evaluate_surface(
            system,
            donor_step=args.donor_step,
            transfer_step=args.transfer_step,
            batch_size=args.batch_size,
        )

    with full_sapt_ratio_wall(
        curves
    ):
        corrected_h3 = h3_barrier(
            batch_size=args.batch_size,
        )

        corrected = {}

        for system in (
            "formaldehyde",
            "water",
            "methane",
        ):
            corrected[
                system
            ] = evaluate_surface(
                system,
                donor_step=args.donor_step,
                transfer_step=args.transfer_step,
                batch_size=args.batch_size,
            )

    print(
        "H3 CONTROL"
    )
    print(
        "----------"
    )
    print(
        f"current       {current_h3['barrier']:.6f} eV "
        f"at r={current_h3['radius']:.5f} A"
    )
    print(
        f"ratio wall    {corrected_h3['barrier']:.6f} eV "
        f"at r={corrected_h3['radius']:.5f} A"
    )
    print(
        f"change        "
        f"{corrected_h3['barrier']-current_h3['barrier']:+.6f} eV"
    )
    print()

    print(
        "FROZEN BARRIER SCREEN"
    )
    print(
        "---------------------"
    )
    print(
        "system          current   ratio-wall     change       "
        "current saddle       ratio-wall saddle"
    )

    for system in (
        "formaldehyde",
        "water",
        "methane",
    ):
        before = current[
            system
        ]

        after = corrected[
            system
        ]

        print(
            f"{system:>12}  "
            f"{before['barrier']:8.3f}  "
            f"{after['barrier']:10.3f}  "
            f"{after['barrier']-before['barrier']:+9.3f}  "
            f"{before['saddle_donor']:.3f}/"
            f"{before['saddle_transfer']:.3f} A       "
            f"{after['saddle_donor']:.3f}/"
            f"{after['saddle_transfer']:.3f} A"
        )

    print()
    print(
        "REACTION-ENERGY DRIFT"
    )
    print(
        "---------------------"
    )

    for system in (
        "formaldehyde",
        "water",
        "methane",
    ):
        before = current[
            system
        ]

        after = corrected[
            system
        ]

        print(
            f"{system:>12}: "
            f"current {before['reaction']:+.6f} eV, "
            f"ratio-wall {after['reaction']:+.6f} eV, "
            f"change {after['reaction']-before['reaction']:+.6f} eV"
        )

    print()
    print(
        "Interpretation:"
    )
    print(
        "  A large downward movement of the C-H abstraction barriers, with H3 "
        "unchanged, supports the missing-full-SAPT hypothesis."
    )
    print(
        "  Overshoot would still be informative: it would mean the omitted "
        "SAPT components matter strongly, but a simple TOTAL/EXCH10 scaling is "
        "too crude for the diabatic wall."
    )
    print(
        "  Little movement would rule out the simple wall explanation despite "
        "the large component offsets in the probe dataset."
    )


if __name__ == "__main__":
    main()
