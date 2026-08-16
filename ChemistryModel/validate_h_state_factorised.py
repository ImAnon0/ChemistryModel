"""
Validate h_state_factorised_torch.py.

Required checks:
    1. Existing 106-point microscope remains unchanged vs historical H-state.
    2. Equal and unequal disconnected H3 systems are exactly additive.
    3. Two H3 competition networks merge/split smoothly through a vanishing
       H-H bridge.
    4. Autograd force agrees with finite difference across that boundary.

This is still a research validation.  NVE comes next only if these pass.

Run:
    py validate_h_state_factorised.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

import reactive as R

from h_state_torch import (
    HStateReferenceBatchedSimulation,
)
from h_state_factorised_torch import (
    FactorisedHStateBatchedSimulation,
)


GEOMETRIES = Path(
    "research_data/qm_residual/dense_scan_geometries.json"
)

BOX_SIZE = 40.0
DTYPE = torch.float64
DEVICE = "cpu"

LEGACY_ENERGY_TOL = 1.0e-9
LEGACY_FORCE_TOL = 1.0e-8

ADDITIVITY_ENERGY_TOL = 1.0e-10
ADDITIVITY_FORCE_TOL = 1.0e-9

MERGE_ENERGY_SPAN_TOL = 1.0e-4
MERGE_GRADIENT_TOL = 1.0e-3

FD_STEP_A = 1.0e-6


def hh_cutoffs():
    h = int(
        R.ELEMENT_INDEX["H"]
    )

    return (
        float(
            R.CUTOFF_INNER[h, h]
        ),
        float(
            R.CUTOFF_OUTER[h, h]
        ),
    )


def choose_spacings():
    inner, outer = hh_cutoffs()
    span = outer - inner

    a = max(
        inner + 0.25 * span,
        0.55 * outer,
    )

    b = max(
        inner + 0.62 * span,
        0.58 * outer,
    )

    a = min(
        a,
        outer - max(
            0.02,
            0.05 * span,
        ),
    )

    b = min(
        b,
        outer - max(
            0.01,
            0.025 * span,
        ),
    )

    if not (
        0.5 * outer < a < outer
        and 0.5 * outer < b < outer
    ):
        raise RuntimeError(
            "failed to choose H3 spacings"
        )

    if (
        2.0 * a <= outer
        or 2.0 * b <= outer
    ):
        raise RuntimeError(
            "H3 end-to-end contact would be active"
        )

    return a, b


def centred(
    coordinates,
    box_size=BOX_SIZE,
):
    positions = np.asarray(
        coordinates,
        dtype=float,
    )

    return (
        positions
        - positions.mean(axis=0)
        + box_size / 2.0
    )


def build(
    model_class,
    symbols,
    positions,
    box_size=BOX_SIZE,
):
    simulation = model_class(
        boxes=[
            (
                list(symbols),
                np.asarray(
                    positions,
                    dtype=float,
                ),
            )
        ],
        box_size=box_size,
        target_temperature=0.0,
        friction=0.0,
        device=DEVICE,
        dtype=DTYPE,
        random_seed=0,
        relax_on_start=False,
    )

    if hasattr(
        simulation,
        "thermostat_is_on",
    ):
        simulation.thermostat_is_on = False

    return simulation


def evaluate(
    model_class,
    symbols,
    positions,
    box_size=BOX_SIZE,
):
    simulation = build(
        model_class,
        symbols,
        positions,
        box_size=box_size,
    )

    energy = float(
        simulation.potential_per_box[0]
    )

    force = (
        simulation.forces
        .detach()
        .cpu()
        .numpy()
        .reshape(
            simulation.per_box,
            3,
        )
    )

    diagnostics = getattr(
        simulation,
        "_h_component_diagnostics",
        {},
    )

    return (
        energy,
        force,
        diagnostics,
    )


def legacy_microscope():
    payload = json.loads(
        GEOMETRIES.read_text(
            encoding="utf-8"
        )
    )

    geometries = payload[
        "geometries"
    ]

    max_energy = 0.0
    max_force = 0.0
    worst_energy = None
    worst_force = None

    by_system_energy = {}
    by_system_force = {}

    for index, geometry in enumerate(
        geometries,
        start=1,
    ):
        symbols = geometry["symbols"]
        positions = centred(
            geometry[
                "coordinates_angstrom"
            ]
        )

        old_e, old_f, _ = evaluate(
            HStateReferenceBatchedSimulation,
            symbols,
            positions,
        )

        new_e, new_f, _ = evaluate(
            FactorisedHStateBatchedSimulation,
            symbols,
            positions,
        )

        de = abs(
            new_e - old_e
        )

        df = float(
            np.max(
                np.abs(
                    new_f - old_f
                )
            )
        )

        system = geometry["system"]

        by_system_energy[system] = max(
            by_system_energy.get(
                system,
                0.0,
            ),
            de,
        )

        by_system_force[system] = max(
            by_system_force.get(
                system,
                0.0,
            ),
            df,
        )

        if de > max_energy:
            max_energy = de
            worst_energy = geometry[
                "geometry_id"
            ]

        if df > max_force:
            max_force = df
            worst_force = geometry[
                "geometry_id"
            ]

        if (
            index == 1
            or index % 20 == 0
            or index == len(geometries)
        ):
            print(
                f"  [{index:3d}/{len(geometries):3d}] "
                f"{geometry['geometry_id']:<38s} "
                f"|dE|={de:.3e} "
                f"|dF|max={df:.3e}"
            )

    return {
        "count": len(geometries),
        "max_energy": max_energy,
        "max_force": max_force,
        "worst_energy": worst_energy,
        "worst_force": worst_force,
        "by_system_energy": by_system_energy,
        "by_system_force": by_system_force,
        "passed": (
            max_energy
            <= LEGACY_ENERGY_TOL
            and max_force
            <= LEGACY_FORCE_TOL
        ),
    }


def h3(
    spacing,
    origin,
):
    origin = np.asarray(
        origin,
        dtype=float,
    )

    positions = np.array([
        [0.0, 0.0, 0.0],
        [spacing, 0.0, 0.0],
        [2.0 * spacing, 0.0, 0.0],
    ])

    return positions + origin


def additive_case(
    spacing_a,
    spacing_b,
):
    positions_a = h3(
        spacing_a,
        [8.0, 8.0, 8.0],
    )

    positions_b = h3(
        spacing_b,
        [8.0, 23.0, 8.0],
    )

    e_a, f_a, _ = evaluate(
        FactorisedHStateBatchedSimulation,
        ["H"] * 3,
        positions_a,
    )

    e_b, f_b, _ = evaluate(
        FactorisedHStateBatchedSimulation,
        ["H"] * 3,
        positions_b,
    )

    e_ab, f_ab, diag = evaluate(
        FactorisedHStateBatchedSimulation,
        ["H"] * 6,
        np.vstack([
            positions_a,
            positions_b,
        ]),
    )

    return {
        "e_a": e_a,
        "e_b": e_b,
        "e_ab": e_ab,
        "nonadditivity": (
            e_ab - e_a - e_b
        ),
        "force_a": float(
            np.max(
                np.abs(
                    f_ab[:3] - f_a
                )
            )
        ),
        "force_b": float(
            np.max(
                np.abs(
                    f_ab[3:] - f_b
                )
            )
        ),
        "diagnostics": diag,
    }


def merge_positions(
    gap,
    spacing_a,
    spacing_b,
):
    start = 8.0

    a = h3(
        spacing_a,
        [start, 12.0, 12.0],
    )

    b_start = (
        start
        + 2.0 * spacing_a
        + gap
    )

    b = h3(
        spacing_b,
        [b_start, 12.0, 12.0],
    )

    return np.vstack([a, b])


def gap_row(
    gap,
    spacing_a,
    spacing_b,
):
    energy, forces, diag = evaluate(
        FactorisedHStateBatchedSimulation,
        ["H"] * 6,
        merge_positions(
            gap,
            spacing_a,
            spacing_b,
        ),
    )

    counts = diag.get(
        "component_counts_per_box",
        ("",),
    )

    return {
        "gap": gap,
        "energy": energy,
        "gradient": float(
            -np.sum(
                forces[3:, 0]
            )
        ),
        "components": (
            counts[0]
            if counts
            else ""
        ),
    }


def locate_transition(
    spacing_a,
    spacing_b,
):
    _, outer = hh_cutoffs()

    low = outer - 1.0e-3
    high = outer + 1.0e-6

    if int(
        gap_row(
            low,
            spacing_a,
            spacing_b,
        )["components"]
    ) != 1:
        raise RuntimeError(
            "low side is not merged"
        )

    if int(
        gap_row(
            high,
            spacing_a,
            spacing_b,
        )["components"]
    ) != 2:
        raise RuntimeError(
            "high side is not split"
        )

    for _ in range(50):
        middle = (
            0.5 * (low + high)
        )

        components = int(
            gap_row(
                middle,
                spacing_a,
                spacing_b,
            )["components"]
        )

        if components == 1:
            low = middle
        else:
            high = middle

    return low, high


def merge_validation(
    spacing_a,
    spacing_b,
):
    low, high = locate_transition(
        spacing_a,
        spacing_b,
    )

    middle = (
        0.5 * (low + high)
    )

    offsets = (
        -5.0e-6,
        -2.0e-6,
        -1.0e-6,
        -5.0e-7,
        -1.0e-7,
        0.0,
        1.0e-7,
        5.0e-7,
        1.0e-6,
        2.0e-6,
        5.0e-6,
    )

    rows = [
        gap_row(
            middle + offset,
            spacing_a,
            spacing_b,
        )
        for offset in offsets
    ]

    fd = (
        gap_row(
            middle + FD_STEP_A,
            spacing_a,
            spacing_b,
        )["energy"]
        - gap_row(
            middle - FD_STEP_A,
            spacing_a,
            spacing_b,
        )["energy"]
    ) / (
        2.0 * FD_STEP_A
    )

    centre = gap_row(
        middle,
        spacing_a,
        spacing_b,
    )

    span = (
        max(
            row["energy"]
            for row in rows
        )
        - min(
            row["energy"]
            for row in rows
        )
    )

    gradient_error = abs(
        fd - centre["gradient"]
    )

    return {
        "low": low,
        "high": high,
        "middle": middle,
        "rows": rows,
        "fd": fd,
        "autograd": centre[
            "gradient"
        ],
        "span": span,
        "gradient_error": gradient_error,
        "passed": (
            span
            < MERGE_ENERGY_SPAN_TOL
            and gradient_error
            < MERGE_GRADIENT_TOL
        ),
    }


def print_additive(
    label,
    result,
):
    print(label)

    print(
        f"  E(A+B)-E(A)-E(B) : "
        f"{result['nonadditivity']:+.12e} eV"
    )

    print(
        f"  max |dF| A       : "
        f"{result['force_a']:.12e} eV/A"
    )

    print(
        f"  max |dF| B       : "
        f"{result['force_b']:.12e} eV/A"
    )

    print(
        f"  diagnostics      : "
        f"{result['diagnostics']}"
    )

    print()


def main():
    print(
        "FACTORISABLE H-STATE VALIDATION"
    )
    print()

    spacing_a, spacing_b = (
        choose_spacings()
    )

    print(
        "1. EXISTING 106-GEOMETRY MICROSCOPE"
    )

    legacy = legacy_microscope()

    print()

    print(
        f"  max |dE| = "
        f"{legacy['max_energy']:.12e} eV "
        f"at {legacy['worst_energy']}"
    )

    print(
        f"  max |dF| = "
        f"{legacy['max_force']:.12e} eV/A "
        f"at {legacy['worst_force']}"
    )

    for system in sorted(
        legacy[
            "by_system_energy"
        ]
    ):
        print(
            f"  {system:14s} "
            f"max|dE|="
            f"{legacy['by_system_energy'][system]:.3e}  "
            f"max|dF|="
            f"{legacy['by_system_force'][system]:.3e}"
        )

    print(
        "  legacy equivalence: "
        + (
            "PASS"
            if legacy["passed"]
            else "FAIL"
        )
    )

    print()
    print("2. DISCONNECTED SIZE CONSISTENCY")

    equal = additive_case(
        spacing_a,
        spacing_a,
    )

    unequal = additive_case(
        spacing_a,
        spacing_b,
    )

    print_additive(
        "EQUAL COMPONENTS",
        equal,
    )

    print_additive(
        "UNEQUAL COMPONENTS",
        unequal,
    )

    additive_pass = all(
        abs(
            result["nonadditivity"]
        ) <= ADDITIVITY_ENERGY_TOL
        and result["force_a"]
        <= ADDITIVITY_FORCE_TOL
        and result["force_b"]
        <= ADDITIVITY_FORCE_TOL
        for result in (
            equal,
            unequal,
        )
    )

    print(
        "  size consistency: "
        + (
            "PASS"
            if additive_pass
            else "FAIL"
        )
    )

    print()
    print("3. COMPONENT MERGE / SPLIT")

    merge = merge_validation(
        spacing_a,
        spacing_b,
    )

    print(
        f"  transition       : "
        f"{merge['low']:.12f} .. "
        f"{merge['high']:.12f} A"
    )

    print(
        f"  microscopic span : "
        f"{merge['span']:.12e} eV"
    )

    print(
        f"  FD dE/dgap       : "
        f"{merge['fd']:+.12e} eV/A"
    )

    print(
        f"  autograd dE/dgap : "
        f"{merge['autograd']:+.12e} eV/A"
    )

    print(
        f"  |FD-autograd|    : "
        f"{merge['gradient_error']:.12e} eV/A"
    )

    print()
    print(
        "  microscopic rows:"
    )

    for row in merge["rows"]:
        print(
            f"    {row['gap']:.12f} A  "
            f"components={row['components']}  "
            f"E={row['energy']:+.12f}  "
            f"dE/dgap={row['gradient']:+.6e}"
        )

    print(
        "  merge continuity: "
        + (
            "PASS"
            if merge["passed"]
            else "FAIL"
        )
    )

    print()
    print("FINAL")

    if (
        legacy["passed"]
        and additive_pass
        and merge["passed"]
    ):
        print(
            "  PASS - factorisable H-state preserves the existing "
            "microscope, is size-consistent, and is smooth through the "
            "tested component merge/split boundary."
        )
        print(
            "  Next: controlled NVE before any performance optimisation."
        )
        return

    print(
        "  FAIL - do not promote this formulation."
    )

    if not legacy["passed"]:
        print(
            "  Existing single-reaction H-state physics changed."
        )

    if not additive_pass:
        print(
            "  Disconnected components are still not additive."
        )

    if not merge["passed"]:
        print(
            "  The zero-bridge merge/split limit is still discontinuous."
        )

    raise SystemExit(1)


if __name__ == "__main__":
    main()
