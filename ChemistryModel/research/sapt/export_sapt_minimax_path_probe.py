"""
Export five representative points from the CURRENT frozen minimax paths for
formaldehyde and methane, together with the ChemistryModel diabatic diagonals.

Run from the normal/base ChemistryModel environment:

    py export_sapt_minimax_path_probe.py

Requires the committed diagnostics:
    sapt_carbon_abstraction_compare.py

Output:
    research_data/sapt/sapt_minimax_path_probe.json

The five points are chosen from the actual current SAPT-H-state minimax path:
    reactant_side
    early_climb
    barrier
    closest_gap
    product_side

No reference barrier is used and no parameter is fitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sapt_carbon_abstraction_compare import (
    analyse_system,
    evaluate_cell,
)


OUTPUT = Path(
    "research_data/sapt/sapt_minimax_path_probe.json"
)

SYSTEMS = (
    "formaldehyde",
    "methane",
)

# Atom ordering comes directly from hf_surface_scan geometry builders.
FRAGMENTS = {
    "formaldehyde": {
        "reactant": {
            "fragment_a": [0, 1, 2, 3],  # CH2O
            "fragment_b": [4],           # H
            "charge_a": 0,
            "mult_a": 1,
            "charge_b": 0,
            "mult_b": 2,
        },
        "product": {
            "fragment_a": [0, 1, 3],     # HCO
            "fragment_b": [2, 4],        # H2
            "charge_a": 0,
            "mult_a": 2,
            "charge_b": 0,
            "mult_b": 1,
        },
    },
    "methane": {
        "reactant": {
            "fragment_a": [0, 1, 2, 3, 4],  # CH4
            "fragment_b": [5],              # H
            "charge_a": 0,
            "mult_a": 1,
            "charge_b": 0,
            "mult_b": 2,
        },
        "product": {
            "fragment_a": [0, 2, 3, 4],     # CH3
            "fragment_b": [1, 5],           # H2
            "charge_a": 0,
            "mult_a": 2,
            "charge_b": 0,
            "mult_b": 1,
        },
    },
}


def nearest_unique(
    rows,
    target_index,
    used,
    *,
    predicate=None,
):
    candidates = [
        row
        for row in rows
        if row["path_index"] not in used
        and (
            predicate is None
            or predicate(row)
        )
    ]

    if not candidates:
        candidates = [
            row
            for row in rows
            if row["path_index"] not in used
        ]

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda row: abs(
            row["path_index"]
            - target_index
        ),
    )


def select_points(
    rows,
    barrier_index,
):
    if len(rows) < 5:
        raise RuntimeError(
            f"Need at least 5 decomposable path points; got {len(rows)}"
        )

    rows = sorted(
        rows,
        key=lambda row: row[
            "path_index"
        ],
    )

    used = set()
    selected = []

    def add(label, row):
        if row is None:
            return
        if row["path_index"] in used:
            return
        used.add(
            row["path_index"]
        )
        selected.append(
            (
                label,
                row,
            )
        )

    add(
        "reactant_side",
        rows[0],
    )

    early_target = int(
        round(
            0.5
            * (
                rows[0]["path_index"]
                + barrier_index
            )
        )
    )

    add(
        "early_climb",
        nearest_unique(
            rows,
            early_target,
            used,
            predicate=lambda row: (
                row["path_index"]
                < barrier_index
            ),
        ),
    )

    add(
        "barrier",
        nearest_unique(
            rows,
            barrier_index,
            used,
        ),
    )

    gap_candidates = sorted(
        (
            row
            for row in rows
            if row["path_index"] not in used
        ),
        key=lambda row: abs(
            row[
                "sapt_product"
            ]
            - row[
                "sapt_reactant"
            ]
        ),
    )

    add(
        "closest_gap",
        gap_candidates[0]
        if gap_candidates
        else None,
    )

    add(
        "product_side",
        rows[-1]
        if rows[-1]["path_index"] not in used
        else nearest_unique(
            rows,
            rows[-1]["path_index"],
            used,
            predicate=lambda row: (
                row["path_index"]
                > barrier_index
            ),
        ),
    )

    # Extremely defensive fallback: fill to five unique points with points
    # nearest the barrier, while preserving the diagnostic nature.
    if len(selected) < 5:
        remaining = sorted(
            (
                row
                for row in rows
                if row["path_index"] not in used
            ),
            key=lambda row: abs(
                row["path_index"]
                - barrier_index
            ),
        )

        for row in remaining:
            add(
                f"extra_{len(selected)+1}",
                row,
            )
            if len(selected) == 5:
                break

    if len(selected) != 5:
        raise RuntimeError(
            f"Could not choose five unique path points; got {len(selected)}"
        )

    return selected


def main():
    output = {
        "schema_version": 1,
        "description": (
            "Five representative cells from the current frozen minimax path, "
            "with ChemistryModel SAPT-H-state diabatic diagonals and exact "
            "geometries for open-shell SAPT fragment calculations."
        ),
        "systems": {},
    }

    print(
        "EXPORT SAPT MINIMAX-PATH OPEN-SHELL PROBE"
    )
    print(
        "========================================="
    )
    print()

    for system in SYSTEMS:
        analysis = analyse_system(
            system,
            donor_step=0.04,
            transfer_step=0.04,
            batch_size=512,
        )

        surface = analysis[
            "surface"
        ]

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

        decomposed = []

        for path_index, cell in enumerate(
            surface[
                "path"
            ]
        ):
            result = evaluate_cell(
                system,
                surface,
                cell,
            )

            if result is None:
                continue

            result = dict(
                result
            )

            result[
                "path_index"
            ] = int(
                path_index
            )

            result[
                "relative_energy"
            ] = (
                float(
                    grid[
                        cell
                    ]
                )
                - reactant_energy
            )

            decomposed.append(
                result
            )

        selected = select_points(
            decomposed,
            int(
                analysis[
                    "barrier_index"
                ]
            ),
        )

        system_out = {
            "barrier_index": int(
                analysis[
                    "barrier_index"
                ]
            ),
            "barrier_relative_eV": float(
                analysis[
                    "barrier_relative"
                ]
            ),
            "points": [],
        }

        print(
            system.upper()
        )
        print(
            "-" * len(
                system
            )
        )

        for label, result in selected:
            donor = float(
                result[
                    "donor"
                ]
            )

            transfer = float(
                result[
                    "transfer"
                ]
            )

            symbols, positions = surface[
                "geometry_builder"
            ](
                donor,
                transfer,
                surface[
                    "spectators"
                ],
            )

            model_reactant = float(
                result[
                    "sapt_reactant"
                ]
            )

            model_product = float(
                result[
                    "sapt_product"
                ]
            )

            point = {
                "label": label,
                "path_index": int(
                    result[
                        "path_index"
                    ]
                ),
                "donor_A": donor,
                "transfer_A": transfer,
                "relative_path_energy_eV": float(
                    result[
                        "relative_energy"
                    ]
                ),
                "model_reactant_diagonal_eV": model_reactant,
                "model_product_diagonal_eV": model_product,
                "model_signed_gap_product_minus_reactant_eV": (
                    model_product
                    - model_reactant
                ),
                "model_abs_gap_eV": abs(
                    model_product
                    - model_reactant
                ),
                "symbols": list(
                    symbols
                ),
                "positions_A": np.asarray(
                    positions,
                    dtype=float,
                ).tolist(),
                "states": FRAGMENTS[
                    system
                ],
            }

            system_out[
                "points"
            ].append(
                point
            )

            print(
                f"  {label:<13} "
                f"idx={point['path_index']:3d}  "
                f"C-H/H-H={donor:.3f}/{transfer:.3f} A  "
                f"path={point['relative_path_energy_eV']:+.4f} eV  "
                f"model gap(P-R)="
                f"{point['model_signed_gap_product_minus_reactant_eV']:+.4f} eV"
            )

        output[
            "systems"
        ][
            system
        ] = system_out

        print()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
