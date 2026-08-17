"""
Validate and benchmark batched heavy-valence membership.

Reference:
    GroupedFactorisableValenceStateBatchedSimulation
        (already validated grouped H-state, original atom-by-atom heavy layer)

Candidate:
    BatchedHeavyValenceStateBatchedSimulation
        (same physics, heavy centres grouped by (N candidates, V capacity))

Checks:
1. All 106 QM microscope geometries, CPU and CUDA:
   per-box energy and every force component.

2. Repeated water-competition stress, CPU and CUDA:
   8 boxes x 24 copies = 960 atoms total.

3. Matched short NVE on water competition x=1.160 A:
   reference vs candidate.

4. Throughput:
   - 8 x 24 repeated water copies (120 atoms/box)
   - 8 x 66 repeated water copies (330 atoms/box; 2640 total atoms)

No chemistry equation or fitted parameter changes.

Run:
    py validate_batched_heavy_valence.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from valence_state_factorised_batched_torch import (
    GroupedFactorisableValenceStateBatchedSimulation,
)
from valence_state_batched_membership_torch import (
    BatchedHeavyValenceStateBatchedSimulation,
)


GEOMETRIES = Path(
    "research_data/qm_residual/dense_scan_geometries.json"
)

DTYPE = torch.float64
GEOMETRY_BOX_SIZE = 30.0
GROUP_SIZE = 16

CPU_ENERGY_TOL = 1.0e-10
CPU_FORCE_TOL = 1.0e-9
CUDA_ENERGY_TOL = 2.0e-8
CUDA_FORCE_TOL = 2.0e-7

NVE_STEPS = 250
NVE_DT_FS = 0.25
NVE_ENERGY_TOL = 1.0e-9
NVE_POSITION_TOL = 1.0e-10
NVE_VELOCITY_TOL = 1.0e-10
NVE_FORCE_TOL = 1.0e-9

WARMUP = 2
TIMED_SMALL = 7
TIMED_330 = 4


def sync(device):
    if (
        torch.device(device).type
        == "cuda"
    ):
        torch.cuda.synchronize()


def load_payload():
    return json.loads(
        GEOMETRIES.read_text(
            encoding="utf-8"
        )
    )


def centred(
    coordinates,
    box_size,
):
    positions = np.asarray(
        coordinates,
        dtype=float,
    )

    return (
        positions
        - positions.mean(
            axis=0
        )
        + box_size
        / 2.0
    )


def compatible_groups(
    geometries,
):
    buckets = defaultdict(
        list
    )

    for geometry in geometries:
        buckets[
            tuple(
                geometry[
                    "symbols"
                ]
            )
        ].append(
            geometry
        )

    groups = []

    for key in sorted(
        buckets,
        key=lambda value: (
            len(
                value
            ),
            value,
        ),
    ):
        rows = buckets[
            key
        ]

        for start in range(
            0,
            len(rows),
            GROUP_SIZE,
        ):
            groups.append(
                rows[
                    start:
                    start + GROUP_SIZE
                ]
            )

    return groups


def build_geometry_group(
    model_class,
    group,
    device,
):
    boxes = [
        (
            list(
                geometry[
                    "symbols"
                ]
            ),
            centred(
                geometry[
                    "coordinates_angstrom"
                ],
                GEOMETRY_BOX_SIZE,
            ),
        )
        for geometry in group
    ]

    return model_class(
        boxes=boxes,
        box_size=GEOMETRY_BOX_SIZE,
        target_temperature=100.0,
        friction=0.0,
        device=device,
        dtype=DTYPE,
        random_seed=0,
        relax_on_start=False,
    )


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
                GroupedFactorisableValenceStateBatchedSimulation,
                group,
                device,
            )
        )

        candidate = (
            build_geometry_group(
                BatchedHeavyValenceStateBatchedSimulation,
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

        for local_index, geometry in enumerate(
            group
        ):
            de = abs(
                float(
                    e_candidate[
                        local_index
                    ]
                    - e_reference[
                        local_index
                    ]
                )
            )

            df = float(
                np.max(
                    np.abs(
                        f_candidate[
                            local_index
                        ]
                        - f_reference[
                            local_index
                        ]
                    )
                )
            )

            if de > max_energy:
                max_energy = de
                worst_energy = (
                    geometry[
                        "geometry_id"
                    ]
                )

            if df > max_force:
                max_force = df
                worst_force = (
                    geometry[
                        "geometry_id"
                    ]
                )

            count += 1

    return {
        "count": count,
        "max_energy": max_energy,
        "max_force": max_force,
        "worst_energy": worst_energy,
        "worst_force": worst_force,
    }


def find_water_x(
    payload,
    x=1.160,
):
    matches = []

    for geometry in payload[
        "geometries"
    ]:
        if (
            geometry.get(
                "system"
            )
            != "water"
            or geometry.get(
                "sample_kind"
            )
            != "dense_transfer_scan"
        ):
            continue

        value = (
            geometry.get(
                "reaction_coordinate",
                {},
            ).get(
                "transfer_distance_angstrom"
            )
        )

        if value is None:
            continue

        matches.append(
            (
                abs(
                    float(value)
                    - x
                ),
                geometry,
            )
        )

    if not matches:
        raise RuntimeError(
            "No dense water transfer geometry"
        )

    matches.sort(
        key=lambda item: item[0]
    )

    error, geometry = matches[0]

    if error > 1e-6:
        raise RuntimeError(
            f"No water geometry at x={x}"
        )

    return geometry


def make_repeated_water(
    payload,
    copies_per_box,
    box_count=8,
):
    geometry = find_water_x(
        payload
    )

    unit_symbols = list(
        geometry[
            "symbols"
        ]
    )

    unit = np.asarray(
        geometry[
            "coordinates_angstrom"
        ],
        dtype=float,
    )

    unit -= unit.mean(
        axis=0
    )

    spacing = max(
        8.0,
        float(
            np.max(
                np.ptp(
                    unit,
                    axis=0,
                )
            )
        )
        + 6.0,
    )

    nx = int(
        math.ceil(
            copies_per_box
            ** (
                1.0 / 3.0
            )
        )
    )

    ny = nx
    nz = int(
        math.ceil(
            copies_per_box
            / (
                nx * ny
            )
        )
    )

    copies = []

    index = 0

    for iz in range(
        nz
    ):
        for iy in range(
            ny
        ):
            for ix in range(
                nx
            ):
                if (
                    index
                    >= copies_per_box
                ):
                    break

                centre = np.asarray([
                    spacing
                    * (
                        1.0 + ix
                    ),
                    spacing
                    * (
                        1.0 + iy
                    ),
                    spacing
                    * (
                        1.0 + iz
                    ),
                ])

                copies.append(
                    unit + centre
                )

                index += 1

    box_size = (
        spacing
        * (
            max(
                nx,
                ny,
                nz,
            )
            + 2.0
        )
    )

    positions = np.vstack(
        copies
    )

    symbols = (
        unit_symbols
        * copies_per_box
    )

    boxes = []

    for box_index in range(
        box_count
    ):
        shift = np.asarray([
            0.005
            * box_index,
            0.007
            * box_index,
            0.009
            * box_index,
        ])

        boxes.append(
            (
                list(
                    symbols
                ),
                (
                    positions
                    + shift
                )
                % box_size,
            )
        )

    return {
        "boxes": boxes,
        "box_size": box_size,
        "copies_per_box": copies_per_box,
        "atoms_per_box": len(
            symbols
        ),
        "total_atoms": (
            len(symbols)
            * box_count
        ),
        "spacing": spacing,
    }


def build_repeated(
    model_class,
    workload,
    device,
):
    return model_class(
        boxes=workload[
            "boxes"
        ],
        box_size=float(
            workload[
                "box_size"
            ]
        ),
        target_temperature=0.0,
        friction=0.0,
        device=device,
        dtype=DTYPE,
        random_seed=0,
        relax_on_start=False,
    )


def compare_repeated(
    workload,
    device,
):
    reference = build_repeated(
        GroupedFactorisableValenceStateBatchedSimulation,
        workload,
        device,
    )

    candidate = build_repeated(
        BatchedHeavyValenceStateBatchedSimulation,
        workload,
        device,
    )

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
            "_heavy_valence_diagnostics",
            {},
        ),
    }


def matched_water_nve(
    payload,
):
    geometry = find_water_x(
        payload
    )

    symbols = list(
        geometry[
            "symbols"
        ]
    )

    positions = centred(
        geometry[
            "coordinates_angstrom"
        ],
        GEOMETRY_BOX_SIZE,
    )

    reference = (
        GroupedFactorisableValenceStateBatchedSimulation(
            boxes=[
                (
                    symbols,
                    positions,
                )
            ],
            box_size=GEOMETRY_BOX_SIZE,
            time_step=NVE_DT_FS,
            target_temperature=100.0,
            friction=0.0,
            device="cpu",
            dtype=DTYPE,
            random_seed=913,
            relax_on_start=False,
        )
    )

    candidate = (
        BatchedHeavyValenceStateBatchedSimulation(
            boxes=[
                (
                    symbols,
                    positions,
                )
            ],
            box_size=GEOMETRY_BOX_SIZE,
            time_step=NVE_DT_FS,
            target_temperature=100.0,
            friction=0.0,
            device="cpu",
            dtype=DTYPE,
            random_seed=913,
            relax_on_start=False,
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
        e_reference = (
            reference.potential_energy
            + reference.kinetic_energy
        )

        e_candidate = (
            candidate.potential_energy
            + candidate.kinetic_energy
        )

        max_energy = max(
            max_energy,
            abs(
                e_candidate
                - e_reference
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
        "max_energy": (
            max_energy
        ),
        "max_position": (
            max_position
        ),
        "max_velocity": (
            max_velocity
        ),
        "max_force": (
            max_force
        ),
        "reference_caps": int(
            reference.capped_steps
        ),
        "candidate_caps": int(
            candidate.capped_steps
        ),
    }


def benchmark(
    model,
    timed,
):
    for _ in range(
        WARMUP
    ):
        model.compute_forces()

    sync(
        model.device
    )

    samples = []

    for _ in range(
        timed
    ):
        sync(
            model.device
        )

        start = (
            time.perf_counter()
        )

        model.compute_forces()

        sync(
            model.device
        )

        samples.append(
            time.perf_counter()
            - start
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


def benchmark_workload(
    workload,
    device,
    timed,
):
    results = {}

    for label, model_class in (
        (
            "reference",
            GroupedFactorisableValenceStateBatchedSimulation,
        ),
        (
            "batched-heavy",
            BatchedHeavyValenceStateBatchedSimulation,
        ),
    ):
        model = build_repeated(
            model_class,
            workload,
            device,
        )

        timing = benchmark(
            model,
            timed,
        )

        results[
            label
        ] = timing

        print(
            f"    {label:<13s}: "
            f"{timing['mean_ms']:.3f} ms "
            f"(median {timing['median_ms']:.3f}, "
            f"min {timing['min_ms']:.3f})"
        )

        if (
            label
            == "batched-heavy"
        ):
            print(
                "      heavy diagnostics: "
                + str(
                    getattr(
                        model,
                        "_heavy_valence_diagnostics",
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
            "batched-heavy"
        ][
            "mean_ms"
        ]
    )

    print(
        f"    speedup      : "
        f"{speedup:.3f} x"
    )

    return speedup


def run_device(
    device,
    groups,
    payload,
    small_workload,
    workload_330,
):
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

    geometry_result = (
        compare_geometry_set(
            groups,
            device,
        )
    )

    print(
        f"  geometries       : "
        f"{geometry_result['count']}"
    )

    print(
        f"  max |dE|         : "
        f"{geometry_result['max_energy']:.12e} eV "
        f"({geometry_result['worst_energy']})"
    )

    print(
        f"  max |dF|         : "
        f"{geometry_result['max_force']:.12e} eV/A "
        f"({geometry_result['worst_force']})"
    )

    print()
    print(
        "2. 960-ATOM REPEATED-WATER EQUIVALENCE"
    )

    stress = compare_repeated(
        small_workload,
        device,
    )

    print(
        f"  max |dE|         : "
        f"{stress['max_energy']:.12e} eV"
    )

    print(
        f"  max |dF|         : "
        f"{stress['max_force']:.12e} eV/A"
    )

    print(
        f"  diagnostics      : "
        f"{stress['diagnostics']}"
    )

    pass_equivalence = (
        geometry_result[
            "max_energy"
        ]
        <= energy_tol
        and geometry_result[
            "max_force"
        ]
        <= force_tol
        and stress[
            "max_energy"
        ]
        <= energy_tol
        and stress[
            "max_force"
        ]
        <= force_tol
    )

    print()
    print(
        "3. THROUGHPUT"
    )

    print(
        "  8 x 120 atoms "
        f"({small_workload['total_atoms']} total):"
    )

    small_speedup = (
        benchmark_workload(
            small_workload,
            device,
            TIMED_SMALL,
        )
    )

    print()
    print(
        "  8 x 330 atoms "
        f"({workload_330['total_atoms']} total):"
    )

    big_speedup = (
        benchmark_workload(
            workload_330,
            device,
            TIMED_330,
        )
    )

    print()
    print(
        "  equivalence : "
        + (
            "PASS"
            if pass_equivalence
            else "FAIL"
        )
    )

    return (
        pass_equivalence,
        small_speedup,
        big_speedup,
    )


def main():
    payload = load_payload()

    groups = compatible_groups(
        payload[
            "geometries"
        ]
    )

    small_workload = (
        make_repeated_water(
            payload,
            copies_per_box=24,
        )
    )

    workload_330 = (
        make_repeated_water(
            payload,
            copies_per_box=66,
        )
    )

    print(
        "BATCHED HEAVY-VALENCE MEMBERSHIP VALIDATION"
    )

    print(
        "reference : grouped factorisable H + atom-by-atom heavy membership"
    )

    print(
        "candidate : grouped factorisable H + batched heavy membership"
    )

    print()
    print(
        "workload geometry: repeated validated water-competition x=1.160 A"
    )

    print(
        f"small : {small_workload['atoms_per_box']} atoms/box, "
        f"{small_workload['total_atoms']} total"
    )

    print(
        f"large : {workload_330['atoms_per_box']} atoms/box, "
        f"{workload_330['total_atoms']} total"
    )

    print()
    print(
        "4. MATCHED WATER-COMPETITION NVE (CPU)"
    )

    nve = matched_water_nve(
        payload
    )

    print(
        f"  max |dEtot|       : "
        f"{nve['max_energy']:.12e} eV"
    )
    print(
        f"  max |dPosition|   : "
        f"{nve['max_position']:.12e} A"
    )
    print(
        f"  max |dVelocity|   : "
        f"{nve['max_velocity']:.12e} A/fs"
    )
    print(
        f"  max |dForce|      : "
        f"{nve['max_force']:.12e} eV/A"
    )
    print(
        f"  caps ref/candidate: "
        f"{nve['reference_caps']}/"
        f"{nve['candidate_caps']}"
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
            "reference_caps"
        ]
        == nve[
            "candidate_caps"
        ]
    )

    print(
        "  NVE equivalence   : "
        + (
            "PASS"
            if nve_pass
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

    all_passed = (
        nve_pass
    )

    for device in devices:
        passed, _, _ = (
            run_device(
                device,
                groups,
                payload,
                small_workload,
                workload_330,
            )
        )

        all_passed = (
            all_passed
            and passed
        )

    print()
    print(
        "=" * 80
    )

    if all_passed:
        print(
            "FINAL PASS - batched heavy-valence membership reproduces the "
            "validated reference within tolerance."
        )

        print(
            "Use the 8 x 330 timing to decide whether the next target is "
            "remaining H topology host-sync or the heavy angle/topology tensor "
            "work."
        )

        return

    print(
        "FINAL FAIL - do not use the batched heavy-membership candidate."
    )

    raise SystemExit(1)


if __name__ == "__main__":
    main()
