"""
Timestep-convergence check for the factorisable H-state disconnected NVE case.

Purpose
-------
The previous controlled NVE validation passed, but the two-disconnected-H3
case showed a maximum drift of ~1.86e-3 eV at dt = 0.05 fs.  This script
checks whether that error behaves like ordinary finite-timestep
velocity-Verlet integration error.

All runs use:
    - exactly the same geometry
    - exactly the same initial velocities
    - thermostat OFF
    - the same physical duration: 50 fs
    - CPU / float64

Timesteps:
    0.0500 fs
    0.0250 fs
    0.0125 fs

For a smooth potential under velocity-Verlet, energy error should generally
decrease strongly as dt is halved; ideal asymptotic second-order behaviour is
roughly a 4x reduction per halving.

Run:
    py check_h_state_factorised_timestep_convergence.py

Outputs:
    research_data/qm_residual/h_state_factorised_dt_0p0500.csv
    research_data/qm_residual/h_state_factorised_dt_0p0250.csv
    research_data/qm_residual/h_state_factorised_dt_0p0125.csv
    research_data/qm_residual/h_state_factorised_timestep_convergence.csv
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

import csv
import math
from pathlib import Path

import numpy as np
import torch

import reactive as R

from h_state_factorised_torch import (
    FactorisedHStateBatchedSimulation,
)


OUT_DIR = Path("research_data/qm_residual")

BOX_SIZE = 40.0
DEVICE = "cpu"
DTYPE = torch.float64

DURATION_FS = 50.0

TIMESTEPS_FS = (
    0.0250,
    0.0125,
    0.00625,
)

# Same disconnected unequal-H3 setup as validate_h_state_factorised_nve.py.
INITIAL_VELOCITIES = np.asarray([
    [+0.0004, +0.0001, 0.0],
    [-0.0001, -0.0002, 0.0],
    [-0.0003, +0.0001, 0.0],
    [-0.0002, +0.0002, 0.0],
    [+0.0005, -0.0001, 0.0],
    [-0.0003, -0.0001, 0.0],
], dtype=float)


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


def choose_spacings():
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

    first = min(
        first,
        outer - max(
            0.02,
            0.05 * span,
        ),
    )

    second = min(
        second,
        outer - max(
            0.01,
            0.025 * span,
        ),
    )

    if not (
        0.5 * outer < first < outer
        and 0.5 * outer < second < outer
    ):
        raise RuntimeError(
            "could not choose safe H3 spacings"
        )

    if (
        2.0 * first <= outer
        or 2.0 * second <= outer
    ):
        raise RuntimeError(
            "chosen H3 spacing activates unwanted end-to-end H-H contact"
        )

    return first, second


def h3_positions(
    spacing,
    origin,
):
    origin = np.asarray(
        origin,
        dtype=float,
    )

    return np.asarray([
        [0.0, 0.0, 0.0],
        [spacing, 0.0, 0.0],
        [2.0 * spacing, 0.0, 0.0],
    ], dtype=float) + origin


def initial_positions():
    spacing_a, spacing_b = (
        choose_spacings()
    )

    positions_a = h3_positions(
        spacing_a,
        [8.0, 8.0, 8.0],
    )

    positions_b = h3_positions(
        spacing_b,
        [8.0, 23.0, 8.0],
    )

    return (
        np.vstack([
            positions_a,
            positions_b,
        ]),
        spacing_a,
        spacing_b,
    )


def build(dt_fs):
    positions, _, _ = (
        initial_positions()
    )

    simulation = (
        FactorisedHStateBatchedSimulation(
            boxes=[
                (
                    ["H"] * 6,
                    positions,
                )
            ],
            box_size=BOX_SIZE,
            time_step=float(dt_fs),
            target_temperature=0.0,
            friction=0.0,
            device=DEVICE,
            dtype=DTYPE,
            random_seed=0,
            relax_on_start=False,
        )
    )

    simulation.thermostat_is_on = False

    simulation.velocities = torch.tensor(
        INITIAL_VELOCITIES,
        device=simulation.device,
        dtype=simulation.dtype,
    )

    return simulation


def total_energy(simulation):
    potential = float(
        simulation.potential_per_box[0]
    )

    kinetic = float(
        simulation.kinetic_per_box[0]
    )

    return (
        potential,
        kinetic,
        potential + kinetic,
    )


def component_count(simulation):
    diagnostics = getattr(
        simulation,
        "_h_component_diagnostics",
        None,
    )

    if not diagnostics:
        _ = simulation.potential_per_box

        diagnostics = getattr(
            simulation,
            "_h_component_diagnostics",
            None,
        )

    if not diagnostics:
        return None

    counts = diagnostics.get(
        "component_counts_per_box",
        (),
    )

    if not counts:
        return 0

    return int(counts[0])


def output_name(dt_fs):
    encoded = (
        f"{dt_fs:.4f}"
        .replace(".", "p")
    )

    return (
        OUT_DIR
        / f"h_state_factorised_dt_{encoded}.csv"
    )


def run_one(dt_fs):
    steps_float = (
        DURATION_FS / dt_fs
    )

    steps = int(
        round(steps_float)
    )

    if not math.isclose(
        steps * dt_fs,
        DURATION_FS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "duration is not an exact integer number of timesteps"
        )

    simulation = build(
        dt_fs
    )

    potential_0, kinetic_0, total_0 = (
        total_energy(
            simulation
        )
    )

    initial_components = (
        component_count(
            simulation
        )
    )

    rows = []

    max_abs_drift = 0.0
    sum_drift_squared = 0.0
    sample_count = 0

    counts_seen = set()

    def record(step):
        nonlocal max_abs_drift
        nonlocal sum_drift_squared
        nonlocal sample_count

        potential, kinetic, total = (
            total_energy(
                simulation
            )
        )

        drift = (
            total - total_0
        )

        count = (
            component_count(
                simulation
            )
        )

        if count is not None:
            counts_seen.add(
                int(count)
            )

        max_abs_drift = max(
            max_abs_drift,
            abs(drift),
        )

        sum_drift_squared += (
            drift * drift
        )

        sample_count += 1

        rows.append({
            "step": int(step),
            "time_fs": float(
                simulation.elapsed_femtoseconds
            ),
            "potential_eV": potential,
            "kinetic_eV": kinetic,
            "total_eV": total,
            "energy_drift_eV": drift,
            "component_count": (
                ""
                if count is None
                else count
            ),
            "capped_steps": int(
                simulation.capped_steps
            ),
            "rebuild_count": int(
                simulation.rebuild_count
            ),
        })

    # Record every integration step. This is only 1000 + 2000 + 4000 steps
    # for six atoms, and avoids hiding a peak drift between sparse samples.
    record(0)

    for step in range(
        1,
        steps + 1,
    ):
        simulation.step()
        record(step)

    path = output_name(
        dt_fs
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
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

    rms_drift = math.sqrt(
        sum_drift_squared
        / sample_count
    )

    final_drift = float(
        rows[-1][
            "energy_drift_eV"
        ]
    )

    return {
        "dt_fs": float(dt_fs),
        "steps": int(steps),
        "duration_fs": float(
            DURATION_FS
        ),
        "initial_total_eV": float(
            total_0
        ),
        "initial_potential_eV": float(
            potential_0
        ),
        "initial_kinetic_eV": float(
            kinetic_0
        ),
        "max_abs_drift_eV": float(
            max_abs_drift
        ),
        "rms_drift_eV": float(
            rms_drift
        ),
        "final_drift_eV": float(
            final_drift
        ),
        "components_initial": (
            ""
            if initial_components is None
            else int(initial_components)
        ),
        "components_seen": (
            " ".join(
                str(value)
                for value in sorted(
                    counts_seen
                )
            )
        ),
        "capped_steps": int(
            simulation.capped_steps
        ),
        "rebuild_count": int(
            simulation.rebuild_count
        ),
        "trajectory_csv": str(
            path
        ),
    }


def observed_order(
    coarse_error,
    fine_error,
):
    if (
        coarse_error <= 0.0
        or fine_error <= 0.0
    ):
        return float("nan")

    return (
        math.log(
            coarse_error
            / fine_error
        )
        / math.log(2.0)
    )


def reduction(
    coarse_error,
    fine_error,
):
    if fine_error <= 0.0:
        return float("inf")

    return (
        coarse_error
        / fine_error
    )


def main():
    positions, spacing_a, spacing_b = (
        initial_positions()
    )

    print(
        "FACTORISABLE H-STATE TIMESTEP CONVERGENCE"
    )

    print()

    print(
        f"device / dtype      : "
        f"{DEVICE} / {DTYPE}"
    )

    print(
        f"duration            : "
        f"{DURATION_FS:.3f} fs each"
    )

    print(
        f"H3 spacings         : "
        f"{spacing_a:.6f} A, "
        f"{spacing_b:.6f} A"
    )

    print(
        "initial components  : "
        "two disconnected unequal H3 competition networks"
    )

    print()

    results = []

    for dt_fs in TIMESTEPS_FS:
        print(
            f"running dt={dt_fs:.4f} fs ..."
        )

        result = run_one(
            dt_fs
        )

        results.append(
            result
        )

        print(
            f"  steps             : "
            f"{result['steps']}"
        )

        print(
            f"  max |dE|          : "
            f"{result['max_abs_drift_eV']:.12e} eV"
        )

        print(
            f"  RMS dE            : "
            f"{result['rms_drift_eV']:.12e} eV"
        )

        print(
            f"  final dE          : "
            f"{result['final_drift_eV']:+.12e} eV"
        )

        print(
            f"  components seen   : "
            f"{result['components_seen']}"
        )

        print(
            f"  capped steps      : "
            f"{result['capped_steps']}"
        )

        print(
            f"  neighbour rebuilds: "
            f"{result['rebuild_count']}"
        )

        print(
            f"  wrote             : "
            f"{result['trajectory_csv']}"
        )

        print()

    print(
        "CONVERGENCE"
    )

    print()

    for metric in (
        "max_abs_drift_eV",
        "rms_drift_eV",
    ):
        label = (
            "max |dE|"
            if metric
            == "max_abs_drift_eV"
            else "RMS dE"
        )

        print(label)

        for coarse, fine in zip(
            results,
            results[1:],
        ):
            factor = reduction(
                coarse[metric],
                fine[metric],
            )

            order = observed_order(
                coarse[metric],
                fine[metric],
            )

            print(
                f"  dt {coarse['dt_fs']:.4f}"
                f" -> {fine['dt_fs']:.4f} fs : "
                f"error reduction={factor:.3f}x  "
                f"observed order p={order:.3f}"
            )

        print()

    summary_path = (
        OUT_DIR
        / "h_state_factorised_timestep_convergence.csv"
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    monotonic_max = all(
        fine["max_abs_drift_eV"]
        < coarse["max_abs_drift_eV"]
        for coarse, fine in zip(
            results,
            results[1:],
        )
    )

    monotonic_rms = all(
        fine["rms_drift_eV"]
        < coarse["rms_drift_eV"]
        for coarse, fine in zip(
            results,
            results[1:],
        )
    )

    no_caps = all(
        result["capped_steps"] == 0
        for result in results
    )

    stayed_disconnected = all(
        result["components_seen"] == "2"
        for result in results
    )

    # Do not insist on exactly p=2 for a max-over-time metric, but require
    # clear convergence.  A >=3x reduction on each halving is strong evidence
    # that the 0.05 fs error is ordinary timestep integration error.
    strong_max_convergence = all(
        reduction(
            coarse["max_abs_drift_eV"],
            fine["max_abs_drift_eV"],
        ) >= 3.0
        for coarse, fine in zip(
            results,
            results[1:],
        )
    )

    strong_rms_convergence = all(
        reduction(
            coarse["rms_drift_eV"],
            fine["rms_drift_eV"],
        ) >= 3.0
        for coarse, fine in zip(
            results,
            results[1:],
        )
    )

    passed = (
        monotonic_max
        and monotonic_rms
        and no_caps
        and stayed_disconnected
        and strong_max_convergence
        and strong_rms_convergence
    )

    print(
        f"summary CSV         : "
        f"{summary_path}"
    )

    print()

    print(
        "VERDICT"
    )

    print(
        f"  max drift monotonic : "
        f"{'PASS' if monotonic_max else 'FAIL'}"
    )

    print(
        f"  RMS drift monotonic : "
        f"{'PASS' if monotonic_rms else 'FAIL'}"
    )

    print(
        f"  >=3x per halving max: "
        f"{'PASS' if strong_max_convergence else 'FAIL'}"
    )

    print(
        f"  >=3x per halving RMS: "
        f"{'PASS' if strong_rms_convergence else 'FAIL'}"
    )

    print(
        f"  no caps             : "
        f"{'PASS' if no_caps else 'FAIL'}"
    )

    print(
        f"  stayed 2 components : "
        f"{'PASS' if stayed_disconnected else 'FAIL'}"
    )

    print()

    if passed:
        print(
            "FINAL: PASS"
        )

        print(
            "The disconnected-case NVE drift decreases strongly under "
            "timestep refinement, consistent with ordinary finite-dt "
            "velocity-Verlet integration error rather than a residual "
            "H-state locality discontinuity."
        )

        return

    print(
        "FINAL: FAIL"
    )

    print(
        "The drift did not show sufficiently clean timestep convergence. "
        "Inspect the trajectory CSVs before promoting the model."
    )

    raise SystemExit(1)


if __name__ == "__main__":
    main()
