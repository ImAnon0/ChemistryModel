"""
Validate cached H-topology execution against the current fully validated
batched-heavy engine.

Reference:
    BatchedHeavyValenceStateBatchedSimulation

Candidate:
    CachedHFastValenceStateBatchedSimulation

Checks
------
1. 106 QM microscope geometries on CPU and CUDA:
   exact/near-exact energy and force equivalence.

2. 8 x 330 repeated-water workload:
   energy and force equivalence.

3. Cache-hit equivalence:
   repeated force evaluations at unchanged topology.

4. Live H topology change:
   two H3 components merge 2 -> 1 while reference/candidate trajectories are
   compared step by step.

5. Explicit neighbour rebuild:
   rebuild both neighbour tables at the same fixed geometry and verify the
   candidate invalidates/reconstructs its cached directed representatives.

6. Throughput on 8 x 330 = 2640 atoms, CPU and CUDA.

No chemistry parameter or equation changes.

Run:
    py validate_cached_h_topology.py
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

import reactive as R

from valence_state_batched_membership_torch import (
    BatchedHeavyValenceStateBatchedSimulation,
)
from valence_state_cached_h_topology_torch import (
    CachedHFastValenceStateBatchedSimulation,
)

from validate_batched_heavy_valence import (
    load_payload,
    compatible_groups,
    build_geometry_group,
    make_repeated_water,
    build_repeated,
)


DTYPE = torch.float64

CPU_ENERGY_TOL = 1.0e-10
CPU_FORCE_TOL = 1.0e-9
CUDA_ENERGY_TOL = 2.0e-8
CUDA_FORCE_TOL = 2.0e-7

MERGE_DT_FS = 0.001
MERGE_STEPS = 200
MERGE_START_OUTSIDE_A = 0.0002
MERGE_RELATIVE_SPEED_A_PER_FS = 0.05

MERGE_ENERGY_TOL = 1.0e-10
MERGE_POSITION_TOL = 1.0e-10
MERGE_VELOCITY_TOL = 1.0e-10
MERGE_FORCE_TOL = 1.0e-9

WARMUP = 3
TIMED = 8


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def compare_geometry_set(
    groups,
    device,
):
    max_energy = 0.0
    max_force = 0.0
    worst_energy = None
    worst_force = None
    count = 0

    for group in groups:
        reference = (
            build_geometry_group(
                BatchedHeavyValenceStateBatchedSimulation,
                group,
                device,
            )
        )

        candidate = (
            build_geometry_group(
                CachedHFastValenceStateBatchedSimulation,
                group,
                device,
            )
        )

        e_reference = np.asarray(
            reference.potential_per_box,
            dtype=float,
        )

        e_candidate = np.asarray(
            candidate.potential_per_box,
            dtype=float,
        )

        f_reference = (
            reference.forces
            .detach()
            .reshape(
                reference.box_count,
                reference.per_box,
                3,
            )
            .cpu()
            .numpy()
        )

        f_candidate = (
            candidate.forces
            .detach()
            .reshape(
                candidate.box_count,
                candidate.per_box,
                3,
            )
            .cpu()
            .numpy()
        )

        for local_index, geometry in enumerate(group):
            de = abs(
                float(
                    e_candidate[local_index]
                    - e_reference[local_index]
                )
            )

            df = float(
                np.max(
                    np.abs(
                        f_candidate[local_index]
                        - f_reference[local_index]
                    )
                )
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

            count += 1

    return {
        "count": count,
        "max_energy": max_energy,
        "max_force": max_force,
        "worst_energy": worst_energy,
        "worst_force": worst_force,
    }


def compare_large(
    workload,
    device,
):
    reference = build_repeated(
        BatchedHeavyValenceStateBatchedSimulation,
        workload,
        device,
    )

    candidate = build_repeated(
        CachedHFastValenceStateBatchedSimulation,
        workload,
        device,
    )

    # Force one cache-hit evaluation after initial construction.
    reference.compute_forces()
    candidate.compute_forces()

    de = float(
        np.max(
            np.abs(
                np.asarray(
                    candidate.potential_per_box,
                    dtype=float,
                )
                - np.asarray(
                    reference.potential_per_box,
                    dtype=float,
                )
            )
        )
    )

    df = float(
        torch.max(
            torch.abs(
                candidate.forces
                - reference.forces
            )
        )
        .detach()
        .cpu()
    )

    return {
        "max_energy": de,
        "max_force": df,
        "diagnostics": getattr(
            candidate,
            "_h_component_diagnostics",
            {},
        ),
    }


def hh_cutoffs():
    hydrogen = int(
        R.ELEMENT_INDEX["H"]
    )

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


def choose_h3_spacings():
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
        outer
        - max(
            0.02,
            0.05 * span,
        ),
    )

    second = min(
        second,
        outer
        - max(
            0.01,
            0.025 * span,
        ),
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
    ]) + origin


def merge_initial_state():
    spacing_a, spacing_b = (
        choose_h3_spacings()
    )

    _, outer = hh_cutoffs()

    gap = (
        outer
        + MERGE_START_OUTSIDE_A
    )

    start = 8.0

    a = h3_positions(
        spacing_a,
        [start, 12.0, 12.0],
    )

    b_start = (
        start
        + 2.0 * spacing_a
        + gap
    )

    b = h3_positions(
        spacing_b,
        [b_start, 12.0, 12.0],
    )

    positions = np.vstack(
        [a, b]
    )

    velocities = np.zeros(
        (6, 3),
        dtype=float,
    )

    half = (
        0.5
        * MERGE_RELATIVE_SPEED_A_PER_FS
    )

    velocities[
        :3,
        0,
    ] = +half

    velocities[
        3:,
        0,
    ] = -half

    return (
        positions,
        velocities,
    )


def build_merge(
    model_class,
):
    positions, velocities = (
        merge_initial_state()
    )

    simulation = model_class(
        boxes=[
            (
                ["H"] * 6,
                positions,
            )
        ],
        box_size=40.0,
        time_step=MERGE_DT_FS,
        target_temperature=0.0,
        friction=0.0,
        device="cpu",
        dtype=DTYPE,
        random_seed=0,
        relax_on_start=False,
    )

    simulation.thermostat_is_on = False

    simulation.velocities = torch.tensor(
        velocities,
        device=simulation.device,
        dtype=simulation.dtype,
    )

    return simulation


def live_merge_equivalence():
    reference = build_merge(
        BatchedHeavyValenceStateBatchedSimulation
    )

    candidate = build_merge(
        CachedHFastValenceStateBatchedSimulation
    )

    max_energy = 0.0
    max_position = 0.0
    max_velocity = 0.0
    max_force = 0.0

    counts_seen = set()
    transitions = []
    previous_count = None

    for step in range(
        MERGE_STEPS + 1
    ):
        max_energy = max(
            max_energy,
            abs(
                (
                    candidate.potential_energy
                    + candidate.kinetic_energy
                )
                - (
                    reference.potential_energy
                    + reference.kinetic_energy
                )
            ),
        )

        max_position = max(
            max_position,
            float(
                torch.max(
                    torch.abs(
                        candidate.positions
                        - reference.positions
                    )
                )
                .detach()
                .cpu()
            ),
        )

        max_velocity = max(
            max_velocity,
            float(
                torch.max(
                    torch.abs(
                        candidate.velocities
                        - reference.velocities
                    )
                )
                .detach()
                .cpu()
            ),
        )

        max_force = max(
            max_force,
            float(
                torch.max(
                    torch.abs(
                        candidate.forces
                        - reference.forces
                    )
                )
                .detach()
                .cpu()
            ),
        )

        diagnostics = getattr(
            candidate,
            "_h_component_diagnostics",
            {},
        )

        values = diagnostics.get(
            "component_counts_per_box",
            (),
        )

        if values:
            count = int(
                values[0]
            )

            counts_seen.add(
                count
            )

            if (
                previous_count is not None
                and count
                != previous_count
            ):
                transitions.append(
                    (
                        step,
                        previous_count,
                        count,
                    )
                )

            previous_count = count

        if step == MERGE_STEPS:
            break

        reference.step()
        candidate.step()

    return {
        "max_energy": max_energy,
        "max_position": max_position,
        "max_velocity": max_velocity,
        "max_force": max_force,
        "counts_seen": tuple(
            sorted(
                counts_seen
            )
        ),
        "transitions": tuple(
            transitions
        ),
        "diagnostics": getattr(
            candidate,
            "_h_component_diagnostics",
            {},
        ),
    }


def explicit_rebuild_check(
    workload,
    device,
):
    reference = build_repeated(
        BatchedHeavyValenceStateBatchedSimulation,
        workload,
        device,
    )

    candidate = build_repeated(
        CachedHFastValenceStateBatchedSimulation,
        workload,
        device,
    )

    # Prime candidate cache at rebuild_count=1.
    candidate.compute_forces()

    old_candidate_rebuild = int(
        candidate.rebuild_count
    )

    reference.build_neighbours()
    candidate.build_neighbours()

    reference.compute_forces()
    candidate.compute_forces()

    de = float(
        np.max(
            np.abs(
                np.asarray(
                    candidate.potential_per_box,
                    dtype=float,
                )
                - np.asarray(
                    reference.potential_per_box,
                    dtype=float,
                )
            )
        )
    )

    df = float(
        torch.max(
            torch.abs(
                candidate.forces
                - reference.forces
            )
        )
        .detach()
        .cpu()
    )

    diagnostics = getattr(
        candidate,
        "_h_component_diagnostics",
        {},
    )

    return {
        "old_rebuild": old_candidate_rebuild,
        "new_rebuild": int(
            candidate.rebuild_count
        ),
        "max_energy": de,
        "max_force": df,
        "diagnostics": diagnostics,
    }


def benchmark_model(
    model,
):
    for _ in range(WARMUP):
        model.compute_forces()

    sync(
        model.device
    )

    samples = []

    for _ in range(TIMED):
        sync(
            model.device
        )

        started = time.perf_counter()

        model.compute_forces()

        sync(
            model.device
        )

        samples.append(
            time.perf_counter()
            - started
        )

    return {
        "mean_ms": (
            1000.0
            * statistics.mean(
                samples
            )
        ),
        "median_ms": (
            1000.0
            * statistics.median(
                samples
            )
        ),
        "min_ms": (
            1000.0
            * min(
                samples
            )
        ),
    }


def benchmark_pair(
    workload,
    device,
):
    results = {}

    for label, model_class in (
        (
            "reference",
            BatchedHeavyValenceStateBatchedSimulation,
        ),
        (
            "cached-H",
            CachedHFastValenceStateBatchedSimulation,
        ),
    ):
        model = build_repeated(
            model_class,
            workload,
            device,
        )

        timing = benchmark_model(
            model
        )

        results[label] = timing

        print(
            f"    {label:<10s}: "
            f"{timing['mean_ms']:.3f} ms "
            f"(median {timing['median_ms']:.3f}, "
            f"min {timing['min_ms']:.3f})"
        )

        if label == "cached-H":
            print(
                "      diagnostics: "
                + str(
                    getattr(
                        model,
                        "_h_component_diagnostics",
                        {},
                    )
                )
            )

    speedup = (
        results[
            "reference"
        ][
            "mean_ms"
        ]
        / results[
            "cached-H"
        ][
            "mean_ms"
        ]
    )

    print(
        f"    speedup   : "
        f"{speedup:.3f} x"
    )

    return speedup


def main():
    payload = load_payload()

    groups = compatible_groups(
        payload[
            "geometries"
        ]
    )

    workload = make_repeated_water(
        payload,
        copies_per_box=66,
    )

    print(
        "CACHED H-TOPOLOGY EXECUTION VALIDATION"
    )

    print(
        f"large workload: "
        f"8 x {workload['atoms_per_box']} atoms "
        f"= {workload['total_atoms']} total"
    )

    print()
    print(
        "LIVE 2->1 H COMPONENT MERGE (CPU)"
    )

    merge = live_merge_equivalence()

    print(
        f"  counts seen       : "
        f"{merge['counts_seen']}"
    )

    print(
        f"  transitions       : "
        f"{merge['transitions']}"
    )

    print(
        f"  max |dEtot|       : "
        f"{merge['max_energy']:.12e} eV"
    )

    print(
        f"  max |dPosition|   : "
        f"{merge['max_position']:.12e} A"
    )

    print(
        f"  max |dVelocity|   : "
        f"{merge['max_velocity']:.12e} A/fs"
    )

    print(
        f"  max |dForce|      : "
        f"{merge['max_force']:.12e} eV/A"
    )

    print(
        f"  final diagnostics : "
        f"{merge['diagnostics']}"
    )

    merge_pass = (
        1 in merge[
            "counts_seen"
        ]
        and 2 in merge[
            "counts_seen"
        ]
        and any(
            old == 2
            and new == 1
            for _, old, new
            in merge[
                "transitions"
            ]
        )
        and merge[
            "max_energy"
        ]
        <= MERGE_ENERGY_TOL
        and merge[
            "max_position"
        ]
        <= MERGE_POSITION_TOL
        and merge[
            "max_velocity"
        ]
        <= MERGE_VELOCITY_TOL
        and merge[
            "max_force"
        ]
        <= MERGE_FORCE_TOL
    )

    print(
        "  merge equivalence : "
        + (
            "PASS"
            if merge_pass
            else "FAIL"
        )
    )

    devices = [
        "cpu"
    ]

    if torch.cuda.is_available():
        devices.append(
            "cuda"
        )

    all_passed = merge_pass

    for device in devices:
        print()
        print(
            "=" * 80
        )
        print(
            f"DEVICE: {device}"
        )

        if device == "cuda":
            print(
                "GPU   : "
                + torch.cuda.get_device_name(
                    torch.cuda.current_device()
                )
            )

        print(
            "=" * 80
        )

        energy_tol = (
            CUDA_ENERGY_TOL
            if device == "cuda"
            else CPU_ENERGY_TOL
        )

        force_tol = (
            CUDA_FORCE_TOL
            if device == "cuda"
            else CPU_FORCE_TOL
        )

        print()
        print(
            "1. 106-GEOMETRY EQUIVALENCE"
        )

        geometry = compare_geometry_set(
            groups,
            device,
        )

        print(
            f"  geometries       : "
            f"{geometry['count']}"
        )

        print(
            f"  max |dE|         : "
            f"{geometry['max_energy']:.12e} eV "
            f"({geometry['worst_energy']})"
        )

        print(
            f"  max |dF|         : "
            f"{geometry['max_force']:.12e} eV/A "
            f"({geometry['worst_force']})"
        )

        print()
        print(
            "2. 8 x 330 CACHE-HIT EQUIVALENCE"
        )

        large = compare_large(
            workload,
            device,
        )

        print(
            f"  max |dE|         : "
            f"{large['max_energy']:.12e} eV"
        )

        print(
            f"  max |dF|         : "
            f"{large['max_force']:.12e} eV/A"
        )

        print(
            f"  diagnostics      : "
            f"{large['diagnostics']}"
        )

        print()
        print(
            "3. EXPLICIT NEIGHBOUR-REBUILD INVALIDATION"
        )

        rebuild = explicit_rebuild_check(
            workload,
            device,
        )

        print(
            f"  rebuild count    : "
            f"{rebuild['old_rebuild']} -> "
            f"{rebuild['new_rebuild']}"
        )

        print(
            f"  max |dE|         : "
            f"{rebuild['max_energy']:.12e} eV"
        )

        print(
            f"  max |dF|         : "
            f"{rebuild['max_force']:.12e} eV/A"
        )

        print(
            f"  diagnostics      : "
            f"{rebuild['diagnostics']}"
        )

        equivalence_pass = (
            geometry[
                "max_energy"
            ]
            <= energy_tol
            and geometry[
                "max_force"
            ]
            <= force_tol
            and large[
                "max_energy"
            ]
            <= energy_tol
            and large[
                "max_force"
            ]
            <= force_tol
            and rebuild[
                "max_energy"
            ]
            <= energy_tol
            and rebuild[
                "max_force"
            ]
            <= force_tol
            and rebuild[
                "new_rebuild"
            ]
            > rebuild[
                "old_rebuild"
            ]
        )

        print()
        print(
            "4. 8 x 330 THROUGHPUT"
        )

        benchmark_pair(
            workload,
            device,
        )

        print()
        print(
            "  equivalence : "
            + (
                "PASS"
                if equivalence_pass
                else "FAIL"
            )
        )

        all_passed = (
            all_passed
            and equivalence_pass
        )

    print()
    print(
        "=" * 80
    )

    if all_passed:
        print(
            "FINAL PASS - cached H topology reproduces the validated "
            "batched-heavy reference through static, cache-hit, live topology "
            "change and neighbour-rebuild cases."
        )

        print(
            "Use the throughput numbers to decide whether CPU/GPU topology "
            "bookkeeping is now small enough to move to the base/autograd path."
        )

        return

    print(
        "FINAL FAIL - do not use cached H-topology candidate."
    )

    raise SystemExit(1)


if __name__ == "__main__":
    main()
