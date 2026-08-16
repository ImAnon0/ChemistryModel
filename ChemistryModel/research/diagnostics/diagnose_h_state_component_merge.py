"""
Stress-test locality of h_state_component_torch.py.

Checks
------
1. UNEQUAL disconnected H3 competition networks
   Confirms exact size consistency was not an accident of two identical,
   symmetric components.

2. COMPONENT MERGE / SPLIT scan
   Two H3 competition networks approach collinearly.  A single H-H candidate
   edge between them appears at the H-H outer cutoff and changes the discrete
   component graph from two components to one.

   We record:
       - total energy
       - force conjugate to the inter-cluster gap
       - H-H bridge taper
       - component count
       - largest component edge count

   We then compare autograd force against a central finite-difference
   derivative of the energy, including microscopic probes around the discrete
   component transition.

This script changes no physics.

If the merge/split boundary is not continuous, DO NOT proceed to NVE yet:
fix the component-boundary Hamiltonian first.

Run:
    py diagnose_h_state_component_merge.py

Output:
    research_data/qm_residual/h_state_component_merge_scan.csv
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch

import reactive as R

from h_state_component_torch import (
    HStateComponentBatchedSimulation,
)


OUTPUT = Path(
    "research_data/qm_residual/h_state_component_merge_scan.csv"
)

BOX_SIZE = 40.0
DTYPE = torch.float64
DEVICE = "cpu"

# Dense visible scan around the H-H cutoff.
SCAN_POINTS = 241

# Tiny parameter-space finite difference used after locating the actual
# discrete component-count transition.
FD_STEP_A = 1.0e-6

ENERGY_ADDITIVITY_TOL_EV = 1.0e-10
FORCE_ADDITIVITY_TOL_EV_A = 1.0e-9


def hh_cutoffs():
    hydrogen = int(R.ELEMENT_INDEX["H"])

    return (
        float(
            R.CUTOFF_INNER[
                hydrogen,
                hydrogen,
            ]
        ),
        float(
            R.CUTOFF_OUTER[
                hydrogen,
                hydrogen,
            ]
        ),
    )


def taper_for_hh(distance):
    inner, outer = hh_cutoffs()

    if distance <= inner:
        return 1.0

    if distance >= outer:
        return 0.0

    fraction = (
        (distance - inner)
        / (outer - inner)
    )

    return 0.5 * (
        1.0
        + math.cos(
            math.pi * fraction
        )
    )


def choose_internal_spacings():
    """
    Choose two different H3 spacings robustly from the model's H-H cutoff.

    Both adjacent H-H contacts remain active, while the end-to-end H-H pair
    is outside the active cutoff so each isolated cluster is the intended
    two-edge H3 competition problem.
    """

    inner, outer = hh_cutoffs()
    span = outer - inner

    first = max(
        inner + 0.25 * span,
        0.55 * outer,
    )

    second = max(
        inner + 0.62 * span,
        0.58 * outer,
    )

    # Keep a little margin below the outer cutoff.
    first = min(
        first,
        outer - max(0.02, 0.05 * span),
    )

    second = min(
        second,
        outer - max(0.01, 0.025 * span),
    )

    if not (
        0.5 * outer < first < outer
        and 0.5 * outer < second < outer
    ):
        raise RuntimeError(
            "could not choose safe H3 spacings from current H-H cutoffs"
        )

    if 2.0 * first <= outer or 2.0 * second <= outer:
        raise RuntimeError(
            "chosen H3 spacing activates unwanted end-to-end H-H contact"
        )

    return first, second


def build(
    symbols,
    positions,
):
    simulation = HStateComponentBatchedSimulation(
        boxes=[
            (
                list(symbols),
                np.asarray(
                    positions,
                    dtype=float,
                ),
            )
        ],
        box_size=BOX_SIZE,
        time_step=0.25,
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
    symbols,
    positions,
):
    simulation = build(
        symbols,
        positions,
    )

    energy = float(
        simulation.potential_per_box[0]
    )

    forces = (
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
        forces,
        diagnostics,
    )


def h3_positions(
    spacing,
    origin,
):
    origin = np.asarray(
        origin,
        dtype=float,
    )

    return np.array([
        [0.0, 0.0, 0.0],
        [spacing, 0.0, 0.0],
        [2.0 * spacing, 0.0, 0.0],
    ]) + origin


def unequal_separability():
    spacing_a, spacing_b = (
        choose_internal_spacings()
    )

    origin_a = np.array([
        8.0,
        8.0,
        8.0,
    ])

    origin_b = np.array([
        8.0,
        23.0,
        8.0,
    ])

    positions_a = h3_positions(
        spacing_a,
        origin_a,
    )

    positions_b = h3_positions(
        spacing_b,
        origin_b,
    )

    symbols_single = ["H"] * 3

    energy_a, force_a, _ = evaluate(
        symbols_single,
        positions_a,
    )

    energy_b, force_b, _ = evaluate(
        symbols_single,
        positions_b,
    )

    combined_positions = np.vstack([
        positions_a,
        positions_b,
    ])

    energy_ab, force_ab, diagnostics = (
        evaluate(
            ["H"] * 6,
            combined_positions,
        )
    )

    return {
        "spacing_a": spacing_a,
        "spacing_b": spacing_b,
        "energy_a": energy_a,
        "energy_b": energy_b,
        "energy_ab": energy_ab,
        "nonadditivity": (
            energy_ab
            - energy_a
            - energy_b
        ),
        "force_error_a": float(
            np.max(
                np.abs(
                    force_ab[:3]
                    - force_a
                )
            )
        ),
        "force_error_b": float(
            np.max(
                np.abs(
                    force_ab[3:]
                    - force_b
                )
            )
        ),
        "diagnostics": diagnostics,
    }


def merge_geometry(
    gap,
    spacing_a,
    spacing_b,
):
    """
    Collinear arrangement:

        A0--A1--A2  <gap>  B0--B1--B2

    Around the H-H outer cutoff, A2--B0 is the only new cross-cluster
    H-containing contact.
    """

    start = 8.0

    positions_a = h3_positions(
        spacing_a,
        [start, 12.0, 12.0],
    )

    b_start = (
        start
        + 2.0 * spacing_a
        + gap
    )

    positions_b = h3_positions(
        spacing_b,
        [b_start, 12.0, 12.0],
    )

    return np.vstack([
        positions_a,
        positions_b,
    ])


def evaluate_gap(
    gap,
    spacing_a,
    spacing_b,
):
    positions = merge_geometry(
        gap,
        spacing_a,
        spacing_b,
    )

    energy, forces, diagnostics = (
        evaluate(
            ["H"] * 6,
            positions,
        )
    )

    # Parameter gap translates the entire B cluster in +x.
    #
    # dE/dgap = sum_B dE/dx = -sum_B F_x
    conjugate_force = float(
        -np.sum(
            forces[3:, 0]
        )
    )

    component_counts = diagnostics.get(
        "component_counts_per_box",
        ("",),
    )

    return {
        "gap_A": float(gap),
        "bridge_taper": taper_for_hh(
            float(gap)
        ),
        "energy_eV": energy,
        "dE_dgap_autograd_eV_per_A": (
            conjugate_force
        ),
        "component_count": (
            component_counts[0]
            if component_counts
            else ""
        ),
        "largest_component_edges": diagnostics.get(
            "largest_component_edges",
            "",
        ),
    }


def locate_component_transition(
    spacing_a,
    spacing_b,
):
    """
    Binary-search the two-component/one-component boundary.

    The active-edge threshold in h_state_torch.py is taper > 1e-12, so this
    boundary sits microscopically inside the formal radial outer cutoff.
    """

    _, outer = hh_cutoffs()

    low = outer - 1.0e-3
    high = outer + 1.0e-6

    low_row = evaluate_gap(
        low,
        spacing_a,
        spacing_b,
    )

    high_row = evaluate_gap(
        high,
        spacing_a,
        spacing_b,
    )

    if not (
        int(low_row["component_count"]) == 1
        and int(high_row["component_count"]) == 2
    ):
        raise RuntimeError(
            "merge geometry did not bracket a 1-component/2-component "
            "transition; inspect current H-H cutoffs/contact graph"
        )

    for _ in range(50):
        middle = 0.5 * (
            low + high
        )

        row = evaluate_gap(
            middle,
            spacing_a,
            spacing_b,
        )

        if int(
            row["component_count"]
        ) == 1:
            low = middle
        else:
            high = middle

    return low, high


def finite_difference_gradient(
    gap,
    spacing_a,
    spacing_b,
    step,
):
    ahead = evaluate_gap(
        gap + step,
        spacing_a,
        spacing_b,
    )

    behind = evaluate_gap(
        gap - step,
        spacing_a,
        spacing_b,
    )

    return (
        ahead["energy_eV"]
        - behind["energy_eV"]
    ) / (
        2.0 * step
    )


def merge_scan():
    spacing_a, spacing_b = (
        choose_internal_spacings()
    )

    inner, outer = hh_cutoffs()
    span = outer - inner

    half_width = max(
        0.04,
        0.12 * span,
    )

    gaps = np.linspace(
        outer - half_width,
        outer + half_width,
        SCAN_POINTS,
    )

    rows = [
        evaluate_gap(
            gap,
            spacing_a,
            spacing_b,
        )
        for gap in gaps
    ]

    for index in range(
        1,
        len(rows) - 1,
    ):
        left = rows[index - 1]
        right = rows[index + 1]

        fd = (
            right["energy_eV"]
            - left["energy_eV"]
        ) / (
            right["gap_A"]
            - left["gap_A"]
        )

        rows[index][
            "dE_dgap_finite_difference_eV_per_A"
        ] = fd

        rows[index][
            "gradient_error_eV_per_A"
        ] = (
            rows[index][
                "dE_dgap_autograd_eV_per_A"
            ]
            - fd
        )

    rows[0][
        "dE_dgap_finite_difference_eV_per_A"
    ] = ""

    rows[0][
        "gradient_error_eV_per_A"
    ] = ""

    rows[-1][
        "dE_dgap_finite_difference_eV_per_A"
    ] = ""

    rows[-1][
        "gradient_error_eV_per_A"
    ] = ""

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
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
        writer.writerows(rows)

    # Visible scan summaries.
    energy_steps = []

    force_steps = []

    for left, right in zip(
        rows,
        rows[1:],
    ):
        energy_steps.append(
            (
                abs(
                    right["energy_eV"]
                    - left["energy_eV"]
                ),
                left,
                right,
            )
        )

        force_steps.append(
            (
                abs(
                    right[
                        "dE_dgap_autograd_eV_per_A"
                    ]
                    - left[
                        "dE_dgap_autograd_eV_per_A"
                    ]
                ),
                left,
                right,
            )
        )

    max_energy_step = max(
        energy_steps,
        key=lambda item: item[0],
    )

    max_force_step = max(
        force_steps,
        key=lambda item: item[0],
    )

    finite_gradient_errors = [
        (
            abs(
                row[
                    "gradient_error_eV_per_A"
                ]
            ),
            row,
        )
        for row in rows
        if row[
            "gradient_error_eV_per_A"
        ] != ""
    ]

    max_gradient_error = max(
        finite_gradient_errors,
        key=lambda item: item[0],
    )

    transition_low, transition_high = (
        locate_component_transition(
            spacing_a,
            spacing_b,
        )
    )

    transition_mid = 0.5 * (
        transition_low
        + transition_high
    )

    # Microscopic probes on each side and across the discrete transition.
    micro_offsets = (
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

    micro_rows = [
        evaluate_gap(
            transition_mid + offset,
            spacing_a,
            spacing_b,
        )
        for offset in micro_offsets
    ]

    fd_at_transition = (
        finite_difference_gradient(
            transition_mid,
            spacing_a,
            spacing_b,
            FD_STEP_A,
        )
    )

    centre_row = evaluate_gap(
        transition_mid,
        spacing_a,
        spacing_b,
    )

    return {
        "rows": rows,
        "spacing_a": spacing_a,
        "spacing_b": spacing_b,
        "inner": inner,
        "outer": outer,
        "max_energy_step": max_energy_step,
        "max_force_step": max_force_step,
        "max_gradient_error": max_gradient_error,
        "transition_low": transition_low,
        "transition_high": transition_high,
        "transition_mid": transition_mid,
        "micro_rows": micro_rows,
        "fd_at_transition": fd_at_transition,
        "autograd_at_transition": centre_row[
            "dE_dgap_autograd_eV_per_A"
        ],
    }


def main():
    inner, outer = hh_cutoffs()

    print(
        "H-STATE LOCAL-COMPONENT MERGE DIAGNOSTIC"
    )

    print()

    print(
        f"H-H cutoff inner/outer : "
        f"{inner:.9f} / {outer:.9f} A"
    )

    print(
        f"device / dtype         : "
        f"{DEVICE} / {DTYPE}"
    )

    print()

    unequal = unequal_separability()

    print("UNEQUAL DISTANT COMPONENTS")

    print(
        f"  H3 spacings            : "
        f"{unequal['spacing_a']:.6f} A, "
        f"{unequal['spacing_b']:.6f} A"
    )

    print(
        f"  E(A)                   : "
        f"{unequal['energy_a']:+.12f} eV"
    )

    print(
        f"  E(B)                   : "
        f"{unequal['energy_b']:+.12f} eV"
    )

    print(
        f"  E(A+B)                 : "
        f"{unequal['energy_ab']:+.12f} eV"
    )

    print(
        f"  E(A+B)-E(A)-E(B)       : "
        f"{unequal['nonadditivity']:+.12e} eV"
    )

    print(
        f"  max |dF| A             : "
        f"{unequal['force_error_a']:.12e} eV/A"
    )

    print(
        f"  max |dF| B             : "
        f"{unequal['force_error_b']:.12e} eV/A"
    )

    print(
        f"  diagnostics            : "
        f"{unequal['diagnostics']}"
    )

    unequal_pass = (
        abs(
            unequal["nonadditivity"]
        ) <= ENERGY_ADDITIVITY_TOL_EV
        and unequal["force_error_a"]
        <= FORCE_ADDITIVITY_TOL_EV_A
        and unequal["force_error_b"]
        <= FORCE_ADDITIVITY_TOL_EV_A
    )

    print(
        "  result                 : "
        + (
            "PASS"
            if unequal_pass
            else "FAIL"
        )
    )

    print()
    print("COMPONENT MERGE / SPLIT SCAN")

    scan = merge_scan()

    print(
        f"  internal H3 spacings   : "
        f"{scan['spacing_a']:.6f} A, "
        f"{scan['spacing_b']:.6f} A"
    )

    print(
        f"  visible scan points    : "
        f"{len(scan['rows'])}"
    )

    print(
        f"  wrote                  : "
        f"{OUTPUT}"
    )

    print(
        f"  component transition   : "
        f"{scan['transition_low']:.12f} .. "
        f"{scan['transition_high']:.12f} A"
    )

    e_step, e_left, e_right = (
        scan["max_energy_step"]
    )

    print(
        f"  max adjacent |dE|      : "
        f"{e_step:.9e} eV "
        f"({e_left['gap_A']:.9f} -> "
        f"{e_right['gap_A']:.9f} A, "
        f"components "
        f"{e_left['component_count']} -> "
        f"{e_right['component_count']})"
    )

    f_step, f_left, f_right = (
        scan["max_force_step"]
    )

    print(
        f"  max adjacent |dForce|  : "
        f"{f_step:.9e} eV/A "
        f"({f_left['gap_A']:.9f} -> "
        f"{f_right['gap_A']:.9f} A)"
    )

    g_error, g_row = (
        scan["max_gradient_error"]
    )

    print(
        f"  max visible |autograd-FD| : "
        f"{g_error:.9e} eV/A "
        f"at gap {g_row['gap_A']:.9f} A"
    )

    print()

    print("MICROSCOPIC TRANSITION PROBE")

    print(
        "  gap_A              taper          components  "
        "energy_eV            dE/dgap_autograd"
    )

    for row in scan["micro_rows"]:
        print(
            f"  {row['gap_A']:.12f}  "
            f"{row['bridge_taper']:.6e}  "
            f"{str(row['component_count']):>10s}  "
            f"{row['energy_eV']:+.12f}  "
            f"{row['dE_dgap_autograd_eV_per_A']:+.9e}"
        )

    print()

    print(
        f"  central FD across transition "
        f"(h={FD_STEP_A:.1e} A) : "
        f"{scan['fd_at_transition']:+.9e} eV/A"
    )

    print(
        f"  autograd at transition midpoint        : "
        f"{scan['autograd_at_transition']:+.9e} eV/A"
    )

    transition_gradient_error = abs(
        scan["fd_at_transition"]
        - scan["autograd_at_transition"]
    )

    print(
        f"  |autograd-FD| across transition        : "
        f"{transition_gradient_error:.9e} eV/A"
    )

    print()
    print("VERDICT")

    if not unequal_pass:
        print(
            "  FAIL - local-component H-state is not size-consistent "
            "for unequal disconnected competition networks."
        )
        raise SystemExit(1)

    # Do not impose an arbitrary chemistry-scale force threshold here.
    # The key diagnostic is whether the microscopic two->one component
    # transition creates a macroscopic energy/gradient mismatch.
    #
    # 1e-4 eV in a 1e-6 A neighbourhood is already far too large for a
    # nominally vanishing new contact; likewise 1e-3 eV/A gradient mismatch
    # is unambiguously larger than float64/autograd noise in these tests.
    microscopic_energies = [
        row["energy_eV"]
        for row in scan["micro_rows"]
    ]

    microscopic_span = (
        max(microscopic_energies)
        - min(microscopic_energies)
    )

    continuity_pass = (
        microscopic_span < 1.0e-4
        and transition_gradient_error < 1.0e-3
    )

    print(
        f"  unequal separability : PASS"
    )

    print(
        "  merge continuity     : "
        + (
            "PASS"
            if continuity_pass
            else "FAIL"
        )
    )

    print(
        f"  microscopic energy span : "
        f"{microscopic_span:.9e} eV"
    )

    if not continuity_pass:
        print()
        print(
            "  The local-component factorisation fixes distant "
            "size consistency, but the discrete component merge/split "
            "boundary is not yet a smooth potential."
        )
        print(
            "  Do NOT promote this model or run NVE across component "
            "mergers yet. The state-mixing normalisation must be made "
            "factorisable in the zero-bridge limit."
        )
        raise SystemExit(2)

    print()
    print(
        "  Component locality is size-consistent and smooth through this "
        "merge/split probe. Proceed to the two-component NVE test."
    )


if __name__ == "__main__":
    main()
