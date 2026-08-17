"""
Benchmark the existing experimental neighbour-gather backend.

No physics changes.

reactive_torch._gather_neighbours() already supports:
    advanced indexing:
        values[neighbours]

    experimental index_select:
        torch.index_select(values, 0, neighbours.reshape(-1)).reshape(...)

The base potential uses neighbour gathers for:
    positions
    spare
    totals
    commitment

Now that base autograd backward is the dominant CUDA cost, test which gather
roles benefit from index_select.

Candidates:
    default       : existing advanced indexing everywhere
    positions     : index_select only positions
    scalar        : index_select spare/totals/commitment
    all           : index_select every _gather_neighbours role

Checks:
1. Full corrected-engine static energy/force equivalence on all 106 geometries.
2. 8 x 330 repeated-water static equivalence.
3. Short matched NVE for the best CUDA candidate against default.
4. compute_forces and actual step throughput on CPU and CUDA.

Run:
    py validate_index_select_gather.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

from collections import defaultdict
import statistics
import time

import numpy as np
import torch

from valence_state_cached_h_topology_torch import (
    CachedHFastValenceStateBatchedSimulation,
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


DTYPE = torch.float64

ROLE_MODES = (
    ("default", False),
    ("positions", {"positions"}),
    ("scalar", {"spare", "totals", "commitment"}),
    ("all", True),
)

CPU_ENERGY_TOL = 1e-10
CPU_FORCE_TOL = 1e-9
CUDA_ENERGY_TOL = 2e-8
CUDA_FORCE_TOL = 2e-7

WARMUP = 3
TIMED_FORCE = 10
STEP_WARMUP = 8
TIMED_STEPS = 120

NVE_STEPS = 250
NVE_DT_FS = 0.25
NVE_ENERGY_TOL = 1e-9
NVE_POSITION_TOL = 1e-10
NVE_VELOCITY_TOL = 1e-10
NVE_FORCE_TOL = 1e-9


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


class GatherModeSimulation(CachedHFastValenceStateBatchedSimulation):
    gather_mode = False

    def __init__(self, *args, **kwargs):
        # Must be set before ReactiveSimulation.__init__ computes initial force.
        self.experimental_index_select_gather = self.gather_mode
        super().__init__(*args, **kwargs)


def make_mode_class(name, mode):
    return type(
        f"Gather_{name}_Simulation",
        (GatherModeSimulation,),
        {
            "gather_mode": mode,
            "physics_model_name": f"diagnostic_gather_{name}",
        },
    )


MODE_CLASSES = {
    name: make_mode_class(name, mode)
    for name, mode in ROLE_MODES
}


def compare_geometry_set(groups, candidate_class, device):
    reference_class = MODE_CLASSES["default"]

    max_energy = 0.0
    max_force = 0.0
    worst_energy = None
    worst_force = None
    count = 0

    for group in groups:
        reference = build_geometry_group(
            reference_class,
            group,
            device,
        )
        candidate = build_geometry_group(
            candidate_class,
            group,
            device,
        )

        e_ref = np.asarray(reference.potential_per_box, dtype=float)
        e_new = np.asarray(candidate.potential_per_box, dtype=float)

        f_ref = (
            reference.forces.detach()
            .reshape(reference.box_count, reference.per_box, 3)
            .cpu().numpy()
        )
        f_new = (
            candidate.forces.detach()
            .reshape(candidate.box_count, candidate.per_box, 3)
            .cpu().numpy()
        )

        for local, geometry in enumerate(group):
            de = abs(float(e_new[local] - e_ref[local]))
            df = float(np.max(np.abs(f_new[local] - f_ref[local])))

            if de > max_energy:
                max_energy = de
                worst_energy = geometry["geometry_id"]

            if df > max_force:
                max_force = df
                worst_force = geometry["geometry_id"]

            count += 1

    return {
        "count": count,
        "max_energy": max_energy,
        "max_force": max_force,
        "worst_energy": worst_energy,
        "worst_force": worst_force,
    }


def compare_large(workload, candidate_class, device):
    reference = build_repeated(
        MODE_CLASSES["default"],
        workload,
        device,
    )
    candidate = build_repeated(
        candidate_class,
        workload,
        device,
    )

    # Put both on normal cache-hit path.
    reference.compute_forces()
    candidate.compute_forces()

    de = float(
        np.max(
            np.abs(
                np.asarray(candidate.potential_per_box, dtype=float)
                - np.asarray(reference.potential_per_box, dtype=float)
            )
        )
    )

    df = float(
        torch.max(
            torch.abs(candidate.forces - reference.forces)
        ).detach().cpu()
    )

    return de, df


def benchmark_force(model):
    for _ in range(WARMUP):
        model.compute_forces()

    sync(model.device)

    samples = []

    for _ in range(TIMED_FORCE):
        sync(model.device)
        start = time.perf_counter()
        model.compute_forces()
        sync(model.device)
        samples.append(time.perf_counter() - start)

    return {
        "mean_ms": 1000.0 * statistics.mean(samples),
        "median_ms": 1000.0 * statistics.median(samples),
        "min_ms": 1000.0 * min(samples),
    }


def benchmark_steps(model):
    model.thermostat_is_on = False

    for _ in range(STEP_WARMUP):
        model.step()

    sync(model.device)

    start_rebuild = int(model.rebuild_count)
    samples = []

    for _ in range(TIMED_STEPS):
        sync(model.device)
        start = time.perf_counter()
        model.step()
        sync(model.device)
        samples.append(time.perf_counter() - start)

    return {
        "mean_ms": 1000.0 * statistics.mean(samples),
        "median_ms": 1000.0 * statistics.median(samples),
        "min_ms": 1000.0 * min(samples),
        "rebuilds": int(model.rebuild_count) - start_rebuild,
    }


def benchmark_modes(workload, device):
    rows = []

    for name, _ in ROLE_MODES:
        model = build_repeated(
            MODE_CLASSES[name],
            workload,
            device,
        )

        force = benchmark_force(model)

        # Use a fresh model for steps so the force microbenchmark does not
        # alter cache counters / timing state.
        step_model = build_repeated(
            MODE_CLASSES[name],
            workload,
            device,
        )
        steps = benchmark_steps(step_model)

        rows.append({
            "name": name,
            "force": force,
            "steps": steps,
        })

    return rows


def matched_nve(payload, candidate_name):
    geometry = find_water_x(payload, 1.160)
    symbols = list(geometry["symbols"])
    positions = centred(
        geometry["coordinates_angstrom"],
        30.0,
    )

    common = dict(
        boxes=[(symbols, positions)],
        box_size=30.0,
        time_step=NVE_DT_FS,
        target_temperature=100.0,
        friction=0.0,
        device="cpu",
        dtype=DTYPE,
        random_seed=913,
        relax_on_start=False,
    )

    reference = MODE_CLASSES["default"](**common)
    candidate = MODE_CLASSES[candidate_name](**common)

    reference.thermostat_is_on = False
    candidate.thermostat_is_on = False
    candidate.velocities = reference.velocities.detach().clone()

    max_energy = 0.0
    max_position = 0.0
    max_velocity = 0.0
    max_force = 0.0

    for step in range(NVE_STEPS + 1):
        e_ref = reference.potential_energy + reference.kinetic_energy
        e_new = candidate.potential_energy + candidate.kinetic_energy

        max_energy = max(max_energy, abs(e_new - e_ref))
        max_position = max(
            max_position,
            float(torch.max(torch.abs(candidate.positions - reference.positions)).detach().cpu()),
        )
        max_velocity = max(
            max_velocity,
            float(torch.max(torch.abs(candidate.velocities - reference.velocities)).detach().cpu()),
        )
        max_force = max(
            max_force,
            float(torch.max(torch.abs(candidate.forces - reference.forces)).detach().cpu()),
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
        "caps_ref": int(reference.capped_steps),
        "caps_new": int(candidate.capped_steps),
    }


def main():
    payload = load_payload()
    groups = compatible_groups(payload["geometries"])
    workload = make_repeated_water(
        payload,
        copies_per_box=66,
    )

    print("INDEX_SELECT NEIGHBOUR-GATHER VALIDATION")
    print()
    print(
        f"workload : 8 x {workload['atoms_per_box']} atoms "
        f"= {workload['total_atoms']} total"
    )
    print("modes    : default / positions / scalar / all")

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    equivalence = {}
    timings = {}

    for device in devices:
        print()
        print("=" * 84)
        print(f"DEVICE: {device}")
        if device == "cuda":
            print("GPU   : " + torch.cuda.get_device_name(torch.cuda.current_device()))
        print("=" * 84)

        energy_tol = CUDA_ENERGY_TOL if device == "cuda" else CPU_ENERGY_TOL
        force_tol = CUDA_FORCE_TOL if device == "cuda" else CPU_FORCE_TOL

        equivalence[device] = {}

        print()
        print("1. STATIC EQUIVALENCE")

        for name, _ in ROLE_MODES:
            if name == "default":
                continue

            geometry = compare_geometry_set(
                groups,
                MODE_CLASSES[name],
                device,
            )

            large_de, large_df = compare_large(
                workload,
                MODE_CLASSES[name],
                device,
            )

            passed = (
                geometry["max_energy"] <= energy_tol
                and geometry["max_force"] <= force_tol
                and large_de <= energy_tol
                and large_df <= force_tol
            )

            equivalence[device][name] = passed

            print(
                f"  {name:<10s} "
                f"106 dE={geometry['max_energy']:.3e} "
                f"dF={geometry['max_force']:.3e} | "
                f"2640 dE={large_de:.3e} dF={large_df:.3e} "
                f"=> {'PASS' if passed else 'FAIL'}"
            )

        print()
        print("2. THROUGHPUT")

        rows = benchmark_modes(
            workload,
            device,
        )

        timings[device] = rows

        default_force = rows[0]["force"]["median_ms"]
        default_step = rows[0]["steps"]["median_ms"]

        for row in rows:
            force = row["force"]
            steps = row["steps"]

            print(
                f"  {row['name']:<10s} "
                f"force median={force['median_ms']:8.3f} ms "
                f"mean={force['mean_ms']:8.3f} | "
                f"step median={steps['median_ms']:8.3f} ms "
                f"mean={steps['mean_ms']:8.3f} "
                f"rebuilds={steps['rebuilds']}"
            )

            if row["name"] != "default":
                print(
                    f"             "
                    f"force speedup={default_force / force['median_ms']:.3f} x  "
                    f"step speedup={default_step / steps['median_ms']:.3f} x"
                )

    # Choose the fastest *valid* CUDA mode by median actual MD step.
    selection_device = "cuda" if "cuda" in timings else "cpu"

    valid_names = [
        name
        for name, _ in ROLE_MODES
        if (
            name == "default"
            or equivalence.get(selection_device, {}).get(name, False)
        )
    ]

    selected = min(
        (
            row
            for row in timings[selection_device]
            if row["name"] in valid_names
        ),
        key=lambda row: row["steps"]["median_ms"],
    )

    selected_name = selected["name"]

    print()
    print("=" * 84)
    print(
        f"BEST VALID {selection_device.upper()} ACTUAL-STEP MODE: "
        f"{selected_name}"
    )
    print("=" * 84)

    if selected_name == "default":
        print(
            "No index_select mode beat the existing gather path. "
            "Do not change production gather execution."
        )
        return

    print()
    print("3. MATCHED WATER NVE FOR SELECTED MODE")

    nve = matched_nve(
        payload,
        selected_name,
    )

    print(f"  max |dEtot|     : {nve['max_energy']:.12e} eV")
    print(f"  max |dPosition| : {nve['max_position']:.12e} A")
    print(f"  max |dVelocity| : {nve['max_velocity']:.12e} A/fs")
    print(f"  max |dForce|    : {nve['max_force']:.12e} eV/A")
    print(f"  caps ref/new    : {nve['caps_ref']}/{nve['caps_new']}")

    nve_pass = (
        nve["max_energy"] <= NVE_ENERGY_TOL
        and nve["max_position"] <= NVE_POSITION_TOL
        and nve["max_velocity"] <= NVE_VELOCITY_TOL
        and nve["max_force"] <= NVE_FORCE_TOL
        and nve["caps_ref"] == nve["caps_new"]
    )

    print(
        "  NVE equivalence : "
        + ("PASS" if nve_pass else "FAIL")
    )

    print()
    if nve_pass:
        print(
            "FINAL: selected gather mode is a valid execution-only "
            "performance candidate."
        )
    else:
        print(
            "FINAL: selected gather mode failed NVE equivalence; "
            "do not use it."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
