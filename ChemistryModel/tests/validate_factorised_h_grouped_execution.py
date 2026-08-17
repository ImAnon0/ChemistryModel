"""
Validate and benchmark grouped execution of the factorisable local H-state.

This is an execution-layer test.  No physics parameter changes.

Checks:
1. 106 QM microscope geometries:
   FactorisedHStateBatchedSimulation
        vs
   GroupedFactorisedHStateBatchedSimulation

   Compare per-box energy and every force component on CPU and CUDA.

2. Heavy-valence wrapper:
   FactorisableValenceStateBatchedSimulation
        vs
   GroupedFactorisableValenceStateBatchedSimulation

   Same 106 geometry comparison.

3. Multi-component stress:
   8 boxes, each containing many disconnected H3 competition components.
   This specifically exercises grouping of many local Hamiltonians and checks
   energy / force equivalence.

4. Throughput benchmark:
   - 8-box formaldehyde transfer microscope
   - 8-box multi-H3 component stress
   reference vs grouped execution on CPU and CUDA.

Run:
    py validate_factorised_h_grouped_execution.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

from collections import defaultdict
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import hf_surface_scan as scan

from h_state_factorised_torch import (
    FactorisedHStateBatchedSimulation,
)
from h_state_factorised_batched_torch import (
    GroupedFactorisedHStateBatchedSimulation,
)
from valence_state_factorised_torch import (
    FactorisableValenceStateBatchedSimulation,
)
from valence_state_factorised_batched_torch import (
    GroupedFactorisableValenceStateBatchedSimulation,
)


GEOMETRIES = Path(
    "research_data/qm_residual/dense_scan_geometries.json"
)

DTYPE = torch.float64
BOX_SIZE = 30.0
GROUP_SIZE = 16

CPU_ENERGY_TOL = 1.0e-10
CPU_FORCE_TOL = 1.0e-9

CUDA_ENERGY_TOL = 2.0e-8
CUDA_FORCE_TOL = 2.0e-7

STRESS_BOX_COUNT = 8
STRESS_COMPONENTS_PER_BOX = 24
STRESS_BOX_SIZE = 40.0

WARMUP = 3
TIMED = 10


def sync(device):
    if torch.device(
        device
    ).type == "cuda":
        torch.cuda.synchronize()


def mean_ms(samples):
    return (
        1000.0
        * statistics.mean(
            samples
        )
    )


def load_geometries():
    payload = json.loads(
        GEOMETRIES.read_text(
            encoding="utf-8"
        )
    )

    return payload[
        "geometries"
    ]


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
            len(
                rows
            ),
            GROUP_SIZE,
        ):
            groups.append(
                rows[
                    start:
                    start
                    + GROUP_SIZE
                ]
            )

    return groups


def build_group(
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
                ]
            ),
        )
        for geometry in group
    ]

    return model_class(
        boxes=boxes,
        box_size=BOX_SIZE,
        target_temperature=100.0,
        friction=0.0,
        device=device,
        dtype=DTYPE,
        random_seed=0,
        relax_on_start=False,
    )


def compare_geometry_set(
    reference_class,
    grouped_class,
    groups,
    device,
):
    max_energy = 0.0
    max_force = 0.0
    worst_energy = None
    worst_force = None
    count = 0

    for group in groups:
        reference = build_group(
            reference_class,
            group,
            device,
        )

        grouped = build_group(
            grouped_class,
            group,
            device,
        )

        e_reference = np.asarray(
            reference.potential_per_box,
            dtype=float,
        )

        e_grouped = np.asarray(
            grouped.potential_per_box,
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

        f_grouped = (
            grouped.forces
            .detach()
            .reshape(
                grouped.box_count,
                grouped.per_box,
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
                    e_grouped[
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
                        f_grouped[
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
        "worst_energy": (
            worst_energy
        ),
        "worst_force": (
            worst_force
        ),
    }


def h3_cluster(
    origin,
    spacing=0.90,
):
    origin = np.asarray(
        origin,
        dtype=float,
    )

    return np.asarray([
        [0.0, 0.0, 0.0],
        [spacing, 0.0, 0.0],
        [
            2.0 * spacing,
            0.0,
            0.0,
        ],
    ]) + origin


def stress_box():
    """
    24 disconnected H3 competition components = 72 atoms.

    Grid spacing 6 A keeps clusters far beyond H-H reactive cutoffs.
    """

    positions = []

    nx = 4
    ny = 3
    nz = 2

    if (
        nx
        * ny
        * nz
        != STRESS_COMPONENTS_PER_BOX
    ):
        raise RuntimeError(
            "stress grid does not match component count"
        )

    for iz in range(
        nz
    ):
        for iy in range(
            ny
        ):
            for ix in range(
                nx
            ):
                origin = np.asarray([
                    5.0
                    + 8.0
                    * ix,
                    5.0
                    + 10.0
                    * iy,
                    5.0
                    + 15.0
                    * iz,
                ])

                positions.append(
                    h3_cluster(
                        origin
                    )
                )

    combined = np.vstack(
        positions
    )

    return (
        ["H"]
        * len(
            combined
        ),
        combined,
    )


def build_stress_boxes():
    symbols, positions = (
        stress_box()
    )

    boxes = []

    for box in range(
        STRESS_BOX_COUNT
    ):
        # Same H topology with tiny box-specific translation so tensors are
        # not byte-for-byte clones.  Periodic wrapping keeps everything valid.
        shifted = (
            positions
            + np.asarray([
                0.01 * box,
                0.02 * box,
                0.03 * box,
            ])
        ) % STRESS_BOX_SIZE

        boxes.append(
            (
                list(
                    symbols
                ),
                shifted,
            )
        )

    return boxes


def build_stress(
    model_class,
    device,
):
    return model_class(
        boxes=build_stress_boxes(),
        box_size=STRESS_BOX_SIZE,
        target_temperature=0.0,
        friction=0.0,
        device=device,
        dtype=DTYPE,
        random_seed=0,
        relax_on_start=False,
    )


def compare_stress(
    device,
):
    reference = build_stress(
        FactorisedHStateBatchedSimulation,
        device,
    )

    grouped = build_stress(
        GroupedFactorisedHStateBatchedSimulation,
        device,
    )

    de = float(
        np.max(
            np.abs(
                np.asarray(
                    grouped.potential_per_box
                )
                - np.asarray(
                    reference.potential_per_box
                )
            )
        )
    )

    df = float(
        torch.max(
            torch.abs(
                grouped.forces
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
            grouped,
            "_h_component_diagnostics",
            {},
        ),
    }


def formaldehyde_boxes():
    symbols, positions = (
        scan.formaldehyde_geometry(
            donor_length=1.20,
            transfer_length=1.05,
        )
    )

    positions = np.asarray(
        positions,
        dtype=float,
    )

    positions = (
        positions
        - positions.mean(
            axis=0
        )
    )

    centre = np.full(
        3,
        6.0,
        dtype=float,
    )

    boxes = []

    for seed in range(
        8
    ):
        generator = (
            np.random.default_rng(
                4000
                + seed
            )
        )

        q = generator.normal(
            size=4
        )

        q /= np.linalg.norm(
            q
        )

        w, x, y, z = q

        rotation = np.asarray([
            [
                1
                - 2
                * (
                    y * y
                    + z * z
                ),
                2
                * (
                    x * y
                    - z * w
                ),
                2
                * (
                    x * z
                    + y * w
                ),
            ],
            [
                2
                * (
                    x * y
                    + z * w
                ),
                1
                - 2
                * (
                    x * x
                    + z * z
                ),
                2
                * (
                    y * z
                    - x * w
                ),
            ],
            [
                2
                * (
                    x * z
                    - y * w
                ),
                2
                * (
                    y * z
                    + x * w
                ),
                1
                - 2
                * (
                    x * x
                    + y * y
                ),
            ],
        ])

        boxes.append(
            (
                list(
                    symbols
                ),
                positions
                @ rotation.T
                + centre,
            )
        )

    return boxes


def build_formaldehyde(
    model_class,
    device,
):
    return model_class(
        boxes=formaldehyde_boxes(),
        box_size=12.0,
        target_temperature=250.0,
        friction=0.01,
        device=device,
        dtype=DTYPE,
        random_seed=0,
        relax_on_start=False,
    )


def benchmark_model(
    model,
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
        TIMED
    ):
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

    return mean_ms(
        samples
    )


def benchmark_pair(
    name,
    builder,
    device,
):
    reference = builder(
        FactorisedHStateBatchedSimulation,
        device,
    )

    grouped = builder(
        GroupedFactorisedHStateBatchedSimulation,
        device,
    )

    reference_ms = benchmark_model(
        reference
    )

    grouped_ms = benchmark_model(
        grouped
    )

    return {
        "name": name,
        "reference_ms": reference_ms,
        "grouped_ms": grouped_ms,
        "speedup": (
            reference_ms
            / grouped_ms
        ),
        "diagnostics": getattr(
            grouped,
            "_h_component_diagnostics",
            {},
        ),
    }


def run_device(
    device,
    groups,
):
    print()
    print(
        "=" * 78
    )

    print(
        f"DEVICE: {device}"
    )

    print(
        "=" * 78
    )

    if torch.device(
        device
    ).type == "cuda":
        print(
            "GPU: "
            + torch.cuda.get_device_name(
                torch.cuda.current_device()
            )
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
        "1. FACTORISABLE H-STATE - 106 GEOMETRIES"
    )

    h_result = (
        compare_geometry_set(
            FactorisedHStateBatchedSimulation,
            GroupedFactorisedHStateBatchedSimulation,
            groups,
            device,
        )
    )

    print(
        f"  geometries       : "
        f"{h_result['count']}"
    )

    print(
        f"  max |dE|         : "
        f"{h_result['max_energy']:.12e} eV "
        f"({h_result['worst_energy']})"
    )

    print(
        f"  max |dF|         : "
        f"{h_result['max_force']:.12e} eV/A "
        f"({h_result['worst_force']})"
    )

    print()
    print(
        "2. HEAVY-VALENCE WRAPPER - 106 GEOMETRIES"
    )

    v_result = (
        compare_geometry_set(
            FactorisableValenceStateBatchedSimulation,
            GroupedFactorisableValenceStateBatchedSimulation,
            groups,
            device,
        )
    )

    print(
        f"  geometries       : "
        f"{v_result['count']}"
    )

    print(
        f"  max |dE|         : "
        f"{v_result['max_energy']:.12e} eV "
        f"({v_result['worst_energy']})"
    )

    print(
        f"  max |dF|         : "
        f"{v_result['max_force']:.12e} eV/A "
        f"({v_result['worst_force']})"
    )

    print()
    print(
        "3. MULTI-COMPONENT STRESS EQUIVALENCE"
    )

    stress = compare_stress(
        device
    )

    print(
        f"  boxes            : "
        f"{STRESS_BOX_COUNT}"
    )

    print(
        f"  H3 components/box: "
        f"{STRESS_COMPONENTS_PER_BOX}"
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
        f"  grouped diagnostics: "
        f"{stress['diagnostics']}"
    )

    equivalence_pass = (
        h_result[
            "max_energy"
        ]
        <= energy_tol
        and h_result[
            "max_force"
        ]
        <= force_tol
        and v_result[
            "max_energy"
        ]
        <= energy_tol
        and v_result[
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
        "4. THROUGHPUT"
    )

    benchmark_results = []

    for name, builder in (
        (
            "8 x formaldehyde transfer",
            build_formaldehyde,
        ),
        (
            "8 x multi-H3 stress",
            build_stress,
        ),
    ):
        result = benchmark_pair(
            name,
            builder,
            device,
        )

        benchmark_results.append(
            result
        )

        print(
            f"  {name}:"
        )

        print(
            f"    reference factorisable : "
            f"{result['reference_ms']:.3f} ms/force"
        )

        print(
            f"    grouped factorisable   : "
            f"{result['grouped_ms']:.3f} ms/force"
        )

        print(
            f"    speedup                : "
            f"{result['speedup']:.3f} x"
        )

        print(
            f"    diagnostics            : "
            f"{result['diagnostics']}"
        )

    print()
    print(
        "RESULT: "
        + (
            "PASS"
            if equivalence_pass
            else "FAIL"
        )
    )

    return (
        equivalence_pass,
        benchmark_results,
    )


def main():
    geometries = (
        load_geometries()
    )

    groups = compatible_groups(
        geometries
    )

    devices = [
        "cpu"
    ]

    if torch.cuda.is_available():
        devices.append(
            "cuda"
        )

    all_passed = True

    for device in devices:
        passed, _ = run_device(
            device,
            groups,
        )

        all_passed = (
            all_passed
            and passed
        )

    print()
    print(
        "=" * 78
    )

    if all_passed:
        print(
            "FINAL PASS - grouped factorisable H-state reproduces the "
            "validated reference within tolerance."
        )

        print(
            "Performance numbers above determine whether the next target "
            "should be topology discovery / host synchronisation or the "
            "heavy-valence membership path."
        )

        return

    print(
        "FINAL FAIL - do not use the grouped execution candidate."
    )

    raise SystemExit(
        1
    )


if __name__ == "__main__":
    main()
