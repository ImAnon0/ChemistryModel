"""
Ablate the omitted SAPT components in the H-state heavy-H wall.

This follows the full-SAPT ratio screen, which showed:

    formaldehyde  0.750 -> 0.355 eV
    water         0.546 -> 0.117 eV
    methane       0.957 -> 0.338 eV

So the omitted SAPT physics clearly matters, but applying the entire SAPT
TOTAL/EXCH10 correction is too strong for water and methane.

This diagnostic asks which omitted component is responsible.

For each bond-axis heavy-H dataset:
    C-H : CH4 + CH2O, approach CH_bond
    O-H : H2O, approach OH_bond
    N-H : NH3, approach NH_bond

it builds distance-dependent wall scales for:

    elst       = (EXCH10 + ELST) / EXCH10
    ind        = (EXCH10 + IND) / EXCH10
    disp       = (EXCH10 + DISP) / EXCH10
    elst_ind   = (EXCH10 + ELST + IND) / EXCH10
    elst_disp  = (EXCH10 + ELST + DISP) / EXCH10
    ind_disp   = (EXCH10 + IND + DISP) / EXCH10
    full       = SAPT TOTAL / EXCH10

H-H is never changed, so H3 remains an untouched control.

No reaction barrier is used to fit any scale.

Requires:
    sapt_total_ratio_wall_screen.py
    research_data/sapt/sapt_component_probe.csv

Usage:
    py sapt_component_wall_ablation.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

import sapt_h_state_torch as sht
import sapt_total_ratio_wall_screen as base


DEFAULT_COMPONENT_CSV = Path(
    "research_data/sapt/sapt_component_probe.csv"
)

MODES = (
    "elst",
    "ind",
    "disp",
    "elst_ind",
    "elst_disp",
    "ind_disp",
    "full",
)

SELECTORS = {
    "C-H": lambda row: (
        row["system"] in {"CH4", "CH2O"}
        and row["approach"] == "CH_bond"
    ),
    "O-H": lambda row: (
        row["system"] == "H2O"
        and row["approach"] == "OH_bond"
    ),
    "N-H": lambda row: (
        row["system"] == "NH3"
        and row["approach"] == "NH_bond"
    ),
}


def load_rows(path):
    if not path.exists():
        raise FileNotFoundError(
            f"component CSV not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("status", "").strip().lower() == "ok"
        ]

    if not rows:
        raise RuntimeError(
            f"no successful component rows in {path}"
        )

    return rows


def scale_for_row(row, mode):
    exch = float(row["rerun_exch10_eV"])

    if abs(exch) < 1.0e-12:
        return 1.0

    elst = float(row["electrostatics_eV"])
    ind = float(row["induction_eV"])
    disp = float(row["dispersion_eV"])

    if mode == "elst":
        total = exch + elst
    elif mode == "ind":
        total = exch + ind
    elif mode == "disp":
        total = exch + disp
    elif mode == "elst_ind":
        total = exch + elst + ind
    elif mode == "elst_disp":
        total = exch + elst + disp
    elif mode == "ind_disp":
        total = exch + ind + disp
    elif mode == "full":
        total = float(row["sapt_total_eV"])
    else:
        raise ValueError(mode)

    return total / exch


def build_mode_curves(rows, mode):
    curves = {}

    for pair_name, selector in SELECTORS.items():
        grouped = defaultdict(list)

        for row in rows:
            if not selector(row):
                continue

            distance = float(
                row["contact_distance_A"]
            )

            grouped[distance].append(
                scale_for_row(
                    row,
                    mode,
                )
            )

        if not grouped:
            raise RuntimeError(
                f"{mode}: no data for {pair_name}"
            )

        distances = np.asarray(
            sorted(grouped),
            dtype=float,
        )

        ratios = np.asarray(
            [
                float(np.mean(grouped[d]))
                for d in distances
            ],
            dtype=float,
        )

        curves[pair_name] = (
            distances,
            ratios,
        )

    return curves


@contextmanager
def scaled_wall(curves):
    original = sht._sapt_pair_energy

    def patched(fragment, first, second):
        raw = original(
            fragment,
            first,
            second,
        )

        first_symbol = base.fragment_symbol(
            fragment,
            first,
        )

        second_symbol = base.fragment_symbol(
            fragment,
            second,
        )

        symbols = {
            first_symbol,
            second_symbol,
        }

        # Leave H-H exactly unchanged.
        if symbols == {"H"}:
            return raw

        if "H" not in symbols:
            return raw

        heavy = (
            second_symbol
            if first_symbol == "H"
            else first_symbol
        )

        pair_name = f"{heavy}-H"

        if pair_name not in curves:
            return raw

        distance = torch.linalg.vector_norm(
            fragment.positions[first]
            - fragment.positions[second]
        )

        xs, ys = curves[pair_name]

        scale = base.interpolate_torch(
            distance,
            xs,
            ys,
        )

        return raw * scale

    sht._sapt_pair_energy = patched

    try:
        yield
    finally:
        sht._sapt_pair_energy = original


def print_curves(mode, curves):
    print(mode.upper())
    print("-" * len(mode))

    for pair_name in (
        "C-H",
        "O-H",
        "N-H",
    ):
        xs, ys = curves[pair_name]

        text = "  ".join(
            f"{x:.2f}:{y:.4f}"
            for x, y in zip(xs, ys)
        )

        print(
            f"{pair_name:>3}  {text}"
        )

    print()


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
        default=512,
    )

    args = parser.parse_args()

    rows = load_rows(
        args.components
    )

    curves_by_mode = {
        mode: build_mode_curves(
            rows,
            mode,
        )
        for mode in MODES
    }

    print(
        "SAPT HEAVY-H WALL COMPONENT ABLATION"
    )
    print(
        "===================================="
    )
    print(
        "No reaction barrier is used to fit these corrections."
    )
    print(
        "H-H is unchanged in every mode."
    )
    print()

    for mode in MODES:
        print_curves(
            mode,
            curves_by_mode[mode],
        )

    systems = (
        "formaldehyde",
        "water",
        "methane",
    )

    current_h3 = base.h3_barrier(
        batch_size=args.batch_size,
    )

    current = {
        system: base.evaluate_surface(
            system,
            donor_step=args.donor_step,
            transfer_step=args.transfer_step,
            batch_size=args.batch_size,
        )
        for system in systems
    }

    results = {}

    for mode in MODES:
        with scaled_wall(
            curves_by_mode[mode]
        ):
            h3 = base.h3_barrier(
                batch_size=args.batch_size,
            )

            systems_result = {
                system: base.evaluate_surface(
                    system,
                    donor_step=args.donor_step,
                    transfer_step=args.transfer_step,
                    batch_size=args.batch_size,
                )
                for system in systems
            }

        results[mode] = {
            "h3": h3,
            "systems": systems_result,
        }

    print(
        "H3 CONTROL"
    )
    print(
        "----------"
    )
    print(
        f"{'current':>10}: "
        f"{current_h3['barrier']:.6f} eV"
    )

    for mode in MODES:
        value = results[
            mode
        ][
            "h3"
        ][
            "barrier"
        ]

        print(
            f"{mode:>10}: "
            f"{value:.6f} eV "
            f"({value-current_h3['barrier']:+.6f})"
        )

    print()
    print(
        "BARRIER ABLATION"
    )
    print(
        "----------------"
    )

    header = (
        f"{'mode':>10}  "
        f"{'formaldehyde':>14}  "
        f"{'water':>10}  "
        f"{'methane':>10}"
    )

    print(header)

    print(
        f"{'current':>10}  "
        f"{current['formaldehyde']['barrier']:14.6f}  "
        f"{current['water']['barrier']:10.6f}  "
        f"{current['methane']['barrier']:10.6f}"
    )

    for mode in MODES:
        values = results[
            mode
        ][
            "systems"
        ]

        print(
            f"{mode:>10}  "
            f"{values['formaldehyde']['barrier']:14.6f}  "
            f"{values['water']['barrier']:10.6f}  "
            f"{values['methane']['barrier']:10.6f}"
        )

    print()
    print(
        "DELTAS FROM CURRENT"
    )
    print(
        "-------------------"
    )
    print(header)

    for mode in MODES:
        values = results[
            mode
        ][
            "systems"
        ]

        print(
            f"{mode:>10}  "
            f"{values['formaldehyde']['barrier']-current['formaldehyde']['barrier']:+14.6f}  "
            f"{values['water']['barrier']-current['water']['barrier']:+10.6f}  "
            f"{values['methane']['barrier']-current['methane']['barrier']:+10.6f}"
        )

    print()
    print(
        "REACTION-ENERGY DRIFT"
    )
    print(
        "---------------------"
    )

    for mode in MODES:
        values = results[
            mode
        ][
            "systems"
        ]

        print(
            f"{mode:>10}: "
            f"CH2O {values['formaldehyde']['reaction']-current['formaldehyde']['reaction']:+.9f}, "
            f"H2O {values['water']['reaction']-current['water']['reaction']:+.9f}, "
            f"CH4 {values['methane']['reaction']-current['methane']['reaction']:+.9f} eV"
        )

    print()
    print(
        "Readout:"
    )
    print(
        "  If one component gives most of the useful carbon-barrier drop while "
        "leaving water much less disturbed, that component is the leading "
        "candidate for the missing wall physics."
    )
    print(
        "  If all three individual components overshoot similarly, the problem "
        "is not component identity but how closed-shell SAPT attraction is "
        "being mapped onto a reactive diabatic edge."
    )


if __name__ == "__main__":
    main()
