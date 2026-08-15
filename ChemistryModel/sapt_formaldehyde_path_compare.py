"""
Compare OLD common-core and SAPT diabatic H-state energies along the actual
relaxed formaldehyde minimax path.

Diagnostic only: no fitting and no parameter changes.

This isolates the state-specific energy shift introduced by replacing the old
common-core unoccupied-contact treatment with the SAPT wall.

Requires the previously created:
    sapt_formaldehyde_state_matrix.py
    sapt_formaldehyde_breaking_ch_scan.py
    sapt_formaldehyde_path_diagnostic.py

Usage:
    py sapt_formaldehyde_path_compare.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import hf_surface_scan as scan

from sapt_formaldehyde_state_matrix import (
    build_simulation,
    prepare_intermediates,
    old_matrix,
    sapt_matrix,
)
from sapt_formaldehyde_breaking_ch_scan import (
    classify_transfer_edges,
    classify_states,
)
from sapt_formaldehyde_path_diagnostic import (
    load_surface,
    minimax_path,
)


DEFAULT_NPZ = (
    "sapt_full_formaldehyde_relaxed_d0p04000_t0p04000.npz"
)


def scalar(tensor, index):
    if index is None:
        return np.nan

    return float(
        tensor[index]
        .detach()
        .cpu()
    )


def evaluate_cell(
    donor,
    transfer,
    spectators,
):
    symbols, positions = scan.formaldehyde_geometry(
        donor,
        transfer,
        spectators,
    )

    sim = build_simulation(
        symbols,
        positions,
    )

    (
        torch_positions,
        values,
        edge_atoms,
        edge_rows,
        edge_slots,
    ) = prepare_intermediates(
        sim
    )

    old = old_matrix(
        sim,
        values,
        edge_atoms,
        edge_rows,
        edge_slots,
    )

    sapt = sapt_matrix(
        sim,
        torch_positions,
        values,
        edge_atoms,
        edge_rows,
        edge_slots,
    )

    transfer_edges = classify_transfer_edges(
        edge_atoms,
        symbols,
    )

    old_reactant, old_product = classify_states(
        old["states"],
        transfer_edges,
    )

    sapt_reactant, sapt_product = classify_states(
        sapt["states"],
        transfer_edges,
    )

    # Candidate topology should be the same for both reconstructions.
    if (
        old_reactant != sapt_reactant
        or old_product != sapt_product
    ):
        raise RuntimeError(
            "Old and SAPT state classifications disagree."
        )

    reactant = sapt_reactant
    product = sapt_product

    old_r = scalar(
        old["diagonal"],
        reactant,
    )
    old_p = scalar(
        old["diagonal"],
        product,
    )

    sapt_r = scalar(
        sapt["diagonal"],
        reactant,
    )
    sapt_p = scalar(
        sapt["diagonal"],
        product,
    )

    old_ground = float(
        old["eigenvalues"][0]
        .detach()
        .cpu()
    )

    sapt_ground = float(
        sapt["eigenvalues"][0]
        .detach()
        .cpu()
    )

    return {
        "old_reactant_diag_eV": old_r,
        "old_product_diag_eV": old_p,
        "sapt_reactant_diag_eV": sapt_r,
        "sapt_product_diag_eV": sapt_p,
        "reactant_diag_shift_eV": (
            sapt_r - old_r
            if np.isfinite(old_r)
            and np.isfinite(sapt_r)
            else np.nan
        ),
        "product_diag_shift_eV": (
            sapt_p - old_p
            if np.isfinite(old_p)
            and np.isfinite(sapt_p)
            else np.nan
        ),
        "old_ground_component_eV": old_ground,
        "sapt_ground_component_eV": sapt_ground,
        "ground_component_shift_eV": (
            sapt_ground - old_ground
        ),
        "sapt_gap_eV": (
            abs(sapt_p - sapt_r)
            if np.isfinite(sapt_p)
            and np.isfinite(sapt_r)
            else np.nan
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--npz",
        default=DEFAULT_NPZ,
    )

    parser.add_argument(
        "--csv",
        default="sapt_formaldehyde_path_compare.csv",
    )

    args = parser.parse_args()

    scan.apply_system(
        "formaldehyde"
    )

    surface = load_surface(
        args.npz
    )

    grid = surface["grid"]
    donors = surface["donor_lengths"]
    transfers = surface["transfer_lengths"]
    spectators = surface["spectators"]

    reactant = tuple(
        int(x)
        for x in surface["reactant"]
    )
    product = tuple(
        int(x)
        for x in surface["product"]
    )

    _, path = minimax_path(
        grid,
        reactant,
        product,
    )

    reactant_energy = float(
        grid[reactant]
    )

    rows = []

    for path_index, cell in enumerate(path):
        i, j = cell

        donor = float(
            donors[i]
        )
        transfer = float(
            transfers[j]
        )
        point_spectators = np.asarray(
            spectators[cell],
            dtype=float,
        )

        comparison = evaluate_cell(
            donor,
            transfer,
            point_spectators,
        )

        rows.append(
            {
                "path_index": path_index,
                "grid_i": i,
                "grid_j": j,
                "donor_ch_A": donor,
                "forming_hh_A": transfer,
                "path_energy_eV": float(
                    grid[cell]
                ),
                "relative_path_energy_eV": (
                    float(grid[cell])
                    - reactant_energy
                ),
                **comparison,
            }
        )

    finite = [
        row
        for row in rows
        if np.isfinite(
            row["product_diag_shift_eV"]
        )
    ]

    if not finite:
        raise RuntimeError(
            "No path cells contained both transfer diabatic states."
        )

    barrier = max(
        finite,
        key=lambda row: row[
            "relative_path_energy_eV"
        ],
    )

    max_product_shift = max(
        finite,
        key=lambda row: row[
            "product_diag_shift_eV"
        ],
    )

    closest_gap = min(
        finite,
        key=lambda row: row[
            "sapt_gap_eV"
        ],
    )

    print(
        "FORMALDEHYDE OLD-vs-SAPT PATH COMPARISON"
    )
    print(
        "======================================="
    )
    print(
        f"source: {args.npz}"
    )
    print(
        f"path cells: {len(path)}"
    )
    print()

    def show(title, row):
        print(title)
        print(
            "=" * len(title)
        )
        print(
            f"path index             {row['path_index']}"
        )
        print(
            f"C-H / H-H              "
            f"{row['donor_ch_A']:.5f} / "
            f"{row['forming_hh_A']:.5f} A"
        )
        print(
            f"path energy             "
            f"{row['relative_path_energy_eV']:+.6f} eV"
        )
        print(
            f"old reactant diagonal   "
            f"{row['old_reactant_diag_eV']:+.6f} eV"
        )
        print(
            f"SAPT reactant diagonal  "
            f"{row['sapt_reactant_diag_eV']:+.6f} eV"
        )
        print(
            f"reactant shift          "
            f"{row['reactant_diag_shift_eV']:+.6f} eV"
        )
        print(
            f"old product diagonal    "
            f"{row['old_product_diag_eV']:+.6f} eV"
        )
        print(
            f"SAPT product diagonal   "
            f"{row['sapt_product_diag_eV']:+.6f} eV"
        )
        print(
            f"product shift           "
            f"{row['product_diag_shift_eV']:+.6f} eV"
        )
        print(
            f"old mixed component     "
            f"{row['old_ground_component_eV']:+.6f} eV"
        )
        print(
            f"SAPT mixed component    "
            f"{row['sapt_ground_component_eV']:+.6f} eV"
        )
        print(
            f"mixed-component shift   "
            f"{row['ground_component_shift_eV']:+.6f} eV"
        )
        print(
            f"SAPT diabatic gap       "
            f"{row['sapt_gap_eV']:.6f} eV"
        )
        print()

    show(
        "ACTUAL PATH BARRIER",
        barrier,
    )

    show(
        "CLOSEST SAPT DIABATIC APPROACH",
        closest_gap,
    )

    show(
        "LARGEST PRODUCT-STATE LIFT",
        max_product_shift,
    )

    print(
        "SELECTED PATH PROFILE"
    )
    print(
        "====================="
    )
    print(
        " idx   C-H    H-H    relE   "
        "dReact   dProd   dMixed   gap"
    )

    selected = set()

    for row in (
        barrier,
        closest_gap,
        max_product_shift,
    ):
        selected.add(
            int(row["path_index"])
        )

    for offset in (
        -3,
        -2,
        -1,
        1,
        2,
        3,
    ):
        index = int(
            barrier["path_index"]
        ) + offset

        if 0 <= index < len(rows):
            selected.add(
                index
            )

    for index in sorted(selected):
        row = rows[index]

        print(
            f"{index:4d} "
            f"{row['donor_ch_A']:6.3f} "
            f"{row['forming_hh_A']:6.3f} "
            f"{row['relative_path_energy_eV']:+7.3f} "
            f"{row['reactant_diag_shift_eV']:+8.3f} "
            f"{row['product_diag_shift_eV']:+8.3f} "
            f"{row['ground_component_shift_eV']:+8.3f} "
            f"{row['sapt_gap_eV']:7.3f}"
        )

    with open(
        args.csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print(
        f"saved CSV: {args.csv}"
    )


if __name__ == "__main__":
    main()
