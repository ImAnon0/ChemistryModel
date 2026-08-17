"""
Final integration validation for OptimisedValenceStateBatchedSimulation.

Reference:
    CachedHFastValenceStateBatchedSimulation
    with its default advanced-index gather backend.

Candidate:
    OptimisedValenceStateBatchedSimulation
    CPU  -> advanced indexing
    CUDA -> index_select

Checks:
1. Device/backend selection:
   explicit CPU, explicit CUDA when available, and device=None auto mode.

2. 106 QM microscope geometries:
   energy + every force component.

3. 8 x 330 repeated-water workload:
   energy + forces.

4. Matched water-competition NVE.

5. Live H component merge 2 -> 1.

6. Explicit neighbour rebuild invalidation.

7. Actual 8 x 330 MD-step throughput.

This validator changes no chemistry.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

import statistics
import time

import numpy as np
import torch

from valence_state_cached_h_topology_torch import (
    CachedHFastValenceStateBatchedSimulation,
)
from valence_state_optimised_torch import (
    OptimisedValenceStateBatchedSimulation,
)

from validate_batched_heavy_valence import (
    load_payload,
    compatible_groups,
    build_geometry_group,
    make_repeated_water,
    build_repeated,
    find_water_x,
    centred,
)

from validate_cached_h_topology import (
    live_merge_equivalence as _unused_reference_merge,
    explicit_rebuild_check as _unused_reference_rebuild,
    merge_initial_state,
    MERGE_DT_FS,
    MERGE_STEPS,
)


DTYPE = torch.float64

CPU_ENERGY_TOL = 1e-10
CPU_FORCE_TOL = 1e-9
CUDA_ENERGY_TOL = 2e-8
CUDA_FORCE_TOL = 2e-7

NVE_STEPS = 250
NVE_DT_FS = 0.25

NVE_ENERGY_TOL = 1e-9
NVE_POSITION_TOL = 1e-10
NVE_VELOCITY_TOL = 1e-10
NVE_FORCE_TOL = 1e-9

MERGE_ENERGY_TOL = 1e-10
MERGE_POSITION_TOL = 1e-10
MERGE_VELOCITY_TOL = 1e-10
MERGE_FORCE_TOL = 1e-9

STEP_WARMUP = 8
TIMED_STEPS = 120


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def backend_name(model):
    return getattr(
        model,
        "selected_neighbour_gather_backend",
        (
            "index_select"
            if getattr(
                model,
                "experimental_index_select_gather",
                False,
            )
            else "advanced_indexing"
        ),
    )


def selection_checks(workload):
    results = []

    cpu = build_repeated(
        OptimisedValenceStateBatchedSimulation,
        workload,
        "cpu",
    )

    results.append({
        "request": "cpu",
        "actual": cpu.device.type,
        "backend": backend_name(cpu),
        "expected_device": "cpu",
        "expected_backend": "advanced_indexing",
    })

    if torch.cuda.is_available():
        cuda = build_repeated(
            OptimisedValenceStateBatchedSimulation,
            workload,
            "cuda",
        )

        results.append({
            "request": "cuda",
            "actual": cuda.device.type,
            "backend": backend_name(cuda),
            "expected_device": "cuda",
            "expected_backend": "index_select",
        })

    # Construct a tiny workload through the same batched interface but with
    # device=None, matching ReactiveSimulation's automatic device convention.
    first_symbols, first_positions = workload["boxes"][0]

    auto = OptimisedValenceStateBatchedSimulation(
        boxes=[
            (
                list(first_symbols),
                np.asarray(first_positions, dtype=float),
            )
        ],
        box_size=float(workload["box_size"]),
        target_temperature=0.0,
        friction=0.0,
        device=None,
        dtype=DTYPE,
        random_seed=0,
        relax_on_start=False,
    )

    expected_auto = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    results.append({
        "request": "None",
        "actual": auto.device.type,
        "backend": backend_name(auto),
        "expected_device": expected_auto,
        "expected_backend": (
            "index_select"
            if expected_auto == "cuda"
            else "advanced_indexing"
        ),
    })

    passed = all(
        row["actual"] == row["expected_device"]
        and row["backend"] == row["expected_backend"]
        for row in results
    )

    return passed, results


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
        reference = build_geometry_group(
            CachedHFastValenceStateBatchedSimulation,
            group,
            device,
        )

        candidate = build_geometry_group(
            OptimisedValenceStateBatchedSimulation,
            group,
            device,
        )

        e_ref = np.asarray(
            reference.potential_per_box,
            dtype=float,
        )

        e_new = np.asarray(
            candidate.potential_per_box,
            dtype=float,
        )

        f_ref = (
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

        f_new = (
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
                    e_new[local_index]
                    - e_ref[local_index]
                )
            )

            df = float(
                np.max(
                    np.abs(
                        f_new[local_index]
                        - f_ref[local_index]
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
        CachedHFastValenceStateBatchedSimulation,
        workload,
        device,
    )

    candidate = build_repeated(
        OptimisedValenceStateBatchedSimulation,
        workload,
        device,
    )

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
        "backend": backend_name(candidate),
    }


def matched_nve(payload, device):
    geometry = find_water_x(
        payload,
        1.160,
    )

    symbols = list(
        geometry["symbols"]
    )

    positions = centred(
        geometry[
            "coordinates_angstrom"
        ],
        30.0,
    )

    common = dict(
        boxes=[
            (
                symbols,
                positions,
            )
        ],
        box_size=30.0,
        time_step=NVE_DT_FS,
        target_temperature=100.0,
        friction=0.0,
        device=device,
        dtype=DTYPE,
        random_seed=913,
        relax_on_start=False,
    )

    reference = (
        CachedHFastValenceStateBatchedSimulation(
            **common
        )
    )

    candidate = (
        OptimisedValenceStateBatchedSimulation(
            **common
        )
    )

    reference.thermostat_is_on = False
    candidate.thermostat_is_on = False

    candidate.velocities = (
        reference.velocities
        .detach()
        .clone()
    )

    max_energy = 0.0
    max_position = 0.0
    max_velocity = 0.0
    max_force = 0.0

    for step in range(
        NVE_STEPS + 1
    ):
        e_ref = (
            reference.potential_energy
            + reference.kinetic_energy
        )

        e_new = (
            candidate.potential_energy
            + candidate.kinetic_energy
        )

        max_energy = max(
            max_energy,
            abs(
                e_new - e_ref
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

        if step == NVE_STEPS:
            break

        reference.step()
        candidate.step()

    return {
        "max_energy": max_energy,
        "max_position": max_position,
        "max_velocity": max_velocity,
        "max_force": max_force,
        "caps_ref": int(
            reference.capped_steps
        ),
        "caps_new": int(
            candidate.capped_steps
        ),
        "backend": backend_name(candidate),
    }


def build_merge(
    model_class,
    device,
):
    positions, velocities = (
        merge_initial_state()
    )

    model = model_class(
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
        device=device,
        dtype=DTYPE,
        random_seed=0,
        relax_on_start=False,
    )

    model.thermostat_is_on = False

    model.velocities = torch.tensor(
        velocities,
        device=model.device,
        dtype=model.dtype,
    )

    return model


def live_merge(device):
    reference = build_merge(
        CachedHFastValenceStateBatchedSimulation,
        device,
    )

    candidate = build_merge(
        OptimisedValenceStateBatchedSimulation,
        device,
    )

    counts_seen = set()
    transitions = []
    previous_count = None

    max_energy = 0.0
    max_position = 0.0
    max_velocity = 0.0
    max_force = 0.0

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
                ).detach().cpu()
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
                ).detach().cpu()
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
                ).detach().cpu()
            ),
        )

        diagnostics = getattr(
            candidate,
            "_h_component_diagnostics",
            {},
        )

        component_counts = diagnostics.get(
            "component_counts_per_box",
            (),
        )

        if component_counts:
            count = int(
                component_counts[0]
            )

            counts_seen.add(
                count
            )

            if (
                previous_count is not None
                and previous_count != count
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
        "counts_seen": tuple(
            sorted(counts_seen)
        ),
        "transitions": tuple(
            transitions
        ),
        "max_energy": max_energy,
        "max_position": max_position,
        "max_velocity": max_velocity,
        "max_force": max_force,
        "backend": backend_name(candidate),
    }


def explicit_rebuild(
    workload,
    device,
):
    reference = build_repeated(
        CachedHFastValenceStateBatchedSimulation,
        workload,
        device,
    )

    candidate = build_repeated(
        OptimisedValenceStateBatchedSimulation,
        workload,
        device,
    )

    candidate.compute_forces()

    old_rebuild = int(
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
        ).detach().cpu()
    )

    return {
        "old_rebuild": old_rebuild,
        "new_rebuild": int(
            candidate.rebuild_count
        ),
        "max_energy": de,
        "max_force": df,
        "backend": backend_name(candidate),
    }


def benchmark_steps(
    model,
):
    model.thermostat_is_on = False

    for _ in range(
        STEP_WARMUP
    ):
        model.step()

    sync(model.device)

    samples = []

    for _ in range(
        TIMED_STEPS
    ):
        sync(model.device)

        started = (
            time.perf_counter()
        )

        model.step()

        sync(model.device)

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
    }


def throughput(
    workload,
    device,
):
    reference = build_repeated(
        CachedHFastValenceStateBatchedSimulation,
        workload,
        device,
    )

    candidate = build_repeated(
        OptimisedValenceStateBatchedSimulation,
        workload,
        device,
    )

    ref = benchmark_steps(
        reference
    )

    new = benchmark_steps(
        candidate
    )

    return {
        "reference": ref,
        "candidate": new,
        "speedup": (
            ref["median_ms"]
            / new["median_ms"]
        ),
        "backend": backend_name(candidate),
    }


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
        "FINAL OPTIMISED VALENCE-STATE INTEGRATION VALIDATION"
    )

    print()
    print(
        f"workload : 8 x {workload['atoms_per_box']} atoms "
        f"= {workload['total_atoms']} total"
    )

    print()
    print(
        "1. DEVICE-AWARE GATHER SELECTION"
    )

    selection_pass, rows = (
        selection_checks(
            workload
        )
    )

    for row in rows:
        print(
            f"  request={row['request']:<4s} "
            f"actual={row['actual']:<4s} "
            f"backend={row['backend']:<18s} "
            f"expected={row['expected_device']}/"
            f"{row['expected_backend']}"
        )

    print(
        "  selection : "
        + (
            "PASS"
            if selection_pass
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

    all_passed = selection_pass

    for device in devices:
        print()
        print(
            "=" * 84
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
            "=" * 84
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
            "2. 106-GEOMETRY EQUIVALENCE"
        )

        geometry = compare_geometry_set(
            groups,
            device,
        )

        print(
            f"  max |dE| : "
            f"{geometry['max_energy']:.12e} eV "
            f"({geometry['worst_energy']})"
        )

        print(
            f"  max |dF| : "
            f"{geometry['max_force']:.12e} eV/A "
            f"({geometry['worst_force']})"
        )

        print()
        print(
            "3. 8 x 330 STATIC EQUIVALENCE"
        )

        large = compare_large(
            workload,
            device,
        )

        print(
            f"  backend  : "
            f"{large['backend']}"
        )
        print(
            f"  max |dE| : "
            f"{large['max_energy']:.12e} eV"
        )
        print(
            f"  max |dF| : "
            f"{large['max_force']:.12e} eV/A"
        )

        print()
        print(
            "4. MATCHED WATER NVE"
        )

        nve = matched_nve(
            payload,
            device,
        )

        print(
            f"  backend        : "
            f"{nve['backend']}"
        )
        print(
            f"  max |dEtot|    : "
            f"{nve['max_energy']:.12e} eV"
        )
        print(
            f"  max |dPosition|: "
            f"{nve['max_position']:.12e} A"
        )
        print(
            f"  max |dVelocity|: "
            f"{nve['max_velocity']:.12e} A/fs"
        )
        print(
            f"  max |dForce|   : "
            f"{nve['max_force']:.12e} eV/A"
        )
        print(
            f"  caps ref/new   : "
            f"{nve['caps_ref']}/"
            f"{nve['caps_new']}"
        )

        print()
        print(
            "5. LIVE H COMPONENT MERGE"
        )

        merge = live_merge(
            device,
        )

        print(
            f"  backend      : "
            f"{merge['backend']}"
        )
        print(
            f"  counts seen  : "
            f"{merge['counts_seen']}"
        )
        print(
            f"  transitions  : "
            f"{merge['transitions']}"
        )
        print(
            f"  max |dEtot|  : "
            f"{merge['max_energy']:.12e} eV"
        )
        print(
            f"  max |dF|     : "
            f"{merge['max_force']:.12e} eV/A"
        )

        print()
        print(
            "6. EXPLICIT NEIGHBOUR REBUILD"
        )

        rebuild = explicit_rebuild(
            workload,
            device,
        )

        print(
            f"  backend       : "
            f"{rebuild['backend']}"
        )
        print(
            f"  rebuild count : "
            f"{rebuild['old_rebuild']} -> "
            f"{rebuild['new_rebuild']}"
        )
        print(
            f"  max |dE|      : "
            f"{rebuild['max_energy']:.12e} eV"
        )
        print(
            f"  max |dF|      : "
            f"{rebuild['max_force']:.12e} eV/A"
        )

        print()
        print(
            "7. ACTUAL 8 x 330 MD-STEP THROUGHPUT"
        )

        speed = throughput(
            workload,
            device,
        )

        print(
            f"  backend             : "
            f"{speed['backend']}"
        )
        print(
            f"  reference median    : "
            f"{speed['reference']['median_ms']:.3f} ms"
        )
        print(
            f"  optimised median    : "
            f"{speed['candidate']['median_ms']:.3f} ms"
        )
        print(
            f"  speedup             : "
            f"{speed['speedup']:.3f} x"
        )

        nve_pass = (
            nve[
                "max_energy"
            ]
            <= NVE_ENERGY_TOL
            and nve[
                "max_position"
            ]
            <= NVE_POSITION_TOL
            and nve[
                "max_velocity"
            ]
            <= NVE_VELOCITY_TOL
            and nve[
                "max_force"
            ]
            <= NVE_FORCE_TOL
            and nve[
                "caps_ref"
            ]
            == nve[
                "caps_new"
            ]
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

        device_pass = (
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
            and nve_pass
            and merge_pass
            and rebuild[
                "new_rebuild"
            ]
            > rebuild[
                "old_rebuild"
            ]
            and rebuild[
                "max_energy"
            ]
            <= energy_tol
            and rebuild[
                "max_force"
            ]
            <= force_tol
        )

        print()
        print(
            "  DEVICE RESULT : "
            + (
                "PASS"
                if device_pass
                else "FAIL"
            )
        )

        all_passed = (
            all_passed
            and device_pass
        )

    print()
    print(
        "=" * 84
    )

    if all_passed:
        print(
            "FINAL PASS - device-aware gather selection preserves the "
            "validated cached-H/batched-heavy physics and chooses the measured "
            "fast backend on each device."
        )
        print(
            "This is a suitable final performance candidate before wiring it "
            "into the normal simulation runner."
        )
        return

    print(
        "FINAL FAIL - do not promote the device-aware candidate."
    )

    raise SystemExit(1)


if __name__ == "__main__":
    main()
