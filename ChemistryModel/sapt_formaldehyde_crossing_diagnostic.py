"""
Decompose the formaldehyde SAPT/H-state barrier at the exact geometries saved
by sapt_full_formaldehyde_scan.py.

This does not fit or modify any parameter. It asks a narrower question:

    Is the high formaldehyde barrier coming from the unmixed radial
    state energies, or from insufficient/incorrect state mixing?

For the saved reactant, saddle and product geometries it evaluates:

    base                  ordinary ChemistryModel
    old H-state, eta=0    old common-core radial decomposition, no mixing
    old H-state, legacy   old common-core + legacy H3 mixing
    SAPT H-state, eta=0   new SAPT wall radial decomposition, no mixing
    SAPT H-state, frozen  new SAPT wall + frozen H3-reanchored mixing

All models are evaluated at the SAME three geometries, so this is a
pointwise energy decomposition, not a separate saddle search for each model.

Usage:
    py sapt_formaldehyde_crossing_diagnostic.py

or:
    py sapt_formaldehyde_crossing_diagnostic.py --npz <file.npz>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import hf_surface_scan as scan
from batched_torch import BatchedReactiveSimulation
from h_state_reference import H_STATE_MIXING
from h_state_torch import HStateReferenceBatchedSimulation
from sapt_h_state_torch import (
    SAPT_H_STATE_MIXING,
    SaptHStateBatchedSimulation,
)


DEFAULT_NPZ = (
    "sapt_full_formaldehyde_relaxed_d0p04000_t0p04000.npz"
)

DTYPE = torch.float64
DEVICE = "cpu"


def load_scan(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing scan file: {path}"
        )

    data = np.load(
        path,
        allow_pickle=False,
    )

    required = (
        "grid",
        "spectators",
        "donor_lengths",
        "transfer_lengths",
        "reactant",
        "product",
        "saddle_cell",
    )

    missing = [
        name
        for name in required
        if name not in data
    ]

    if missing:
        raise RuntimeError(
            "NPZ is missing: "
            + ", ".join(missing)
        )

    result = {
        name: np.asarray(
            data[name]
        )
        for name in required
    }

    for name in (
        "reactant",
        "product",
        "saddle_cell",
    ):
        result[name] = tuple(
            int(value)
            for value in result[name]
        )

    return result


def geometry_for_cell(
    data,
    cell,
):
    i, j = cell

    donor = float(
        data[
            "donor_lengths"
        ][i]
    )

    transfer = float(
        data[
            "transfer_lengths"
        ][j]
    )

    spectators = np.asarray(
        data[
            "spectators"
        ][i, j],
        dtype=float,
    )

    symbols, positions = (
        scan.formaldehyde_geometry(
            donor,
            transfer,
            spectators,
        )
    )

    return {
        "donor": donor,
        "transfer": transfer,
        "spectators": spectators,
        "symbols": symbols,
        "positions": positions,
    }


def build_model(
    model,
    symbols,
    positions,
    *,
    mixing=None,
):
    boxes = [
        (
            symbols,
            positions + scan.CENTRE,
        )
    ]

    common = dict(
        boxes=boxes,
        box_size=scan.BOX,
        random_seed=0,
        relax_on_start=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    if model == "base":
        return BatchedReactiveSimulation(
            **common
        )

    if model == "old_zero":
        return HStateReferenceBatchedSimulation(
            **common,
            h_state_mixing=0.0,
        )

    if model == "old_legacy":
        return HStateReferenceBatchedSimulation(
            **common,
            h_state_mixing=H_STATE_MIXING,
        )

    if model == "sapt_zero":
        return SaptHStateBatchedSimulation(
            **common,
            h_state_mixing=0.0,
        )

    if model == "sapt_frozen":
        return SaptHStateBatchedSimulation(
            **common,
            h_state_mixing=SAPT_H_STATE_MIXING,
        )

    raise ValueError(
        f"unknown model: {model}"
    )


def energy_at(
    model,
    geometry,
):
    simulation = build_model(
        model,
        geometry[
            "symbols"
        ],
        geometry[
            "positions"
        ],
    )

    return float(
        simulation.potential_per_box[
            0
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--npz",
        default=DEFAULT_NPZ,
    )

    args = parser.parse_args()

    scan.apply_system(
        "formaldehyde"
    )

    data = load_scan(
        args.npz
    )

    points = {
        "reactant": geometry_for_cell(
            data,
            data[
                "reactant"
            ],
        ),
        "saddle": geometry_for_cell(
            data,
            data[
                "saddle_cell"
            ],
        ),
        "product": geometry_for_cell(
            data,
            data[
                "product"
            ],
        ),
    }

    models = (
        (
            "base",
            "base ChemistryModel",
        ),
        (
            "old_zero",
            "old H-state eta=0",
        ),
        (
            "old_legacy",
            f"old H-state eta={H_STATE_MIXING:.6f}",
        ),
        (
            "sapt_zero",
            "SAPT H-state eta=0",
        ),
        (
            "sapt_frozen",
            f"SAPT H-state eta={SAPT_H_STATE_MIXING:.9f}",
        ),
    )

    energies = {
        key: {}
        for key, _ in models
    }

    print(
        "SAPT FORMALDEHYDE CROSSING DIAGNOSTIC"
    )

    print(
        "====================================="
    )

    print(
        f"surface source: {args.npz}"
    )

    print(
        "all models evaluated at identical saved geometries"
    )

    print()

    for point_name, geometry in points.items():
        print(
            point_name.upper()
        )

        print(
            f"  donor C-H   {geometry['donor']:.5f} A"
        )

        print(
            f"  forming H-H {geometry['transfer']:.5f} A"
        )

        print(
            "  spectators  "
            + np.array2string(
                geometry[
                    "spectators"
                ],
                precision=6,
            )
        )

        for key, label in models:
            value = energy_at(
                key,
                geometry,
            )

            energies[
                key
            ][
                point_name
            ] = value

            print(
                f"  {label:<34} "
                f"{value:+.9f} eV"
            )

        print()

    print(
        "SAME-GEOMETRY BARRIER / REACTION"
    )

    print(
        "================================"
    )

    print(
        f"{'model':<34}"
        f"{'barrier':>14}"
        f"{'reaction':>14}"
    )

    for key, label in models:
        reactant = energies[
            key
        ][
            "reactant"
        ]

        saddle = energies[
            key
        ][
            "saddle"
        ]

        product = energies[
            key
        ][
            "product"
        ]

        print(
            f"{label:<34}"
            f"{saddle-reactant:+12.6f} eV"
            f"{product-reactant:+12.6f} eV"
        )

    print()

    old_mix_lowering_reactant = (
        energies[
            "old_legacy"
        ][
            "reactant"
        ]
        - energies[
            "old_zero"
        ][
            "reactant"
        ]
    )

    old_mix_lowering_saddle = (
        energies[
            "old_legacy"
        ][
            "saddle"
        ]
        - energies[
            "old_zero"
        ][
            "saddle"
        ]
    )

    sapt_mix_lowering_reactant = (
        energies[
            "sapt_frozen"
        ][
            "reactant"
        ]
        - energies[
            "sapt_zero"
        ][
            "reactant"
        ]
    )

    sapt_mix_lowering_saddle = (
        energies[
            "sapt_frozen"
        ][
            "saddle"
        ]
        - energies[
            "sapt_zero"
        ][
            "saddle"
        ]
    )

    print(
        "MIXING CONTRIBUTION"
    )

    print(
        "==================="
    )

    print(
        "negative means state mixing lowers the energy"
    )

    print()

    print(
        f"old radial, reactant: "
        f"{old_mix_lowering_reactant:+.6f} eV"
    )

    print(
        f"old radial, saddle:   "
        f"{old_mix_lowering_saddle:+.6f} eV"
    )

    print(
        f"old mixing effect on barrier: "
        f"{old_mix_lowering_saddle-old_mix_lowering_reactant:+.6f} eV"
    )

    print()

    print(
        f"SAPT radial, reactant: "
        f"{sapt_mix_lowering_reactant:+.6f} eV"
    )

    print(
        f"SAPT radial, saddle:   "
        f"{sapt_mix_lowering_saddle:+.6f} eV"
    )

    print(
        f"SAPT mixing effect on barrier: "
        f"{sapt_mix_lowering_saddle-sapt_mix_lowering_reactant:+.6f} eV"
    )

    print()

    radial_change_reactant = (
        energies[
            "sapt_zero"
        ][
            "reactant"
        ]
        - energies[
            "old_zero"
        ][
            "reactant"
        ]
    )

    radial_change_saddle = (
        energies[
            "sapt_zero"
        ][
            "saddle"
        ]
        - energies[
            "old_zero"
        ][
            "saddle"
        ]
    )

    print(
        "RADIAL-DECOMPOSITION CHANGE"
    )

    print(
        "==========================="
    )

    print(
        "SAPT eta=0 minus old-common-core eta=0"
    )

    print(
        f"reactant: {radial_change_reactant:+.6f} eV"
    )

    print(
        f"saddle:   {radial_change_saddle:+.6f} eV"
    )

    print(
        f"effect on same-geometry barrier: "
        f"{radial_change_saddle-radial_change_reactant:+.6f} eV"
    )

    print()

    print(
        "Interpretation:"
    )

    print(
        "  If SAPT eta=0 is already far too high at the saddle, the"
    )

    print(
        "  problem is primarily the diabatic radial wall / state energies."
    )

    print(
        "  If eta=0 is sensible but the frozen mixed result is too high,"
    )

    print(
        "  the H3-calibrated coupling does not transfer to this environment."
    )


if __name__ == "__main__":
    main()
