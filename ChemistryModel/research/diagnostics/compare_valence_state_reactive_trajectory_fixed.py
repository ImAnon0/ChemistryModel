"""
Matched reactive trajectory comparison:
    H + CH2O -> H2 + HCO

Compares:
    HStateReferenceBatchedSimulation
    ValenceStateBatchedSimulation

Protocol
--------
This intentionally reuses the CURRENT characterisation collision machinery:
    - formaldehyde stored in the hf_surface_scan atom ordering
      C0, O1, donor-H2, spectator-H3
    - incoming partner H becomes atom 4
    - targeted collision aimed at donor-H2
    - 2.5 A requested start gap
    - 5 x thermal RMS directed relative COM approach speed
    - 250 K internal thermal distribution
    - 0.25 fs timestep
    - friction 0.01 (normal characterisation setting)
    - 5 ps duration
    - 48 seeds, grouped 8 at a time

Reaction criterion
------------------
Uses the same geometric taper threshold (0.35) as reactive_torch.bond_list():

    donor H2 -- incoming H4 bonded
    C0 -- donor H2 broken
    C0 -- spectator H3 retained
    C0 -- O1 retained

A hit is counted once this product topology persists continuously for 20 fs.
Final retention is also reported separately.

This is a matched MODEL COMPARISON, not an experimental rate calculation.

Run:
    py compare_valence_state_reactive_trajectory.py

Outputs:
    research_data/qm_residual/valence_state_reactive_trajectory.csv
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

import csv
import math
from pathlib import Path

import numpy as np
import torch

import reactive as R
import hf_surface_scan as scan

from characterisation_runner import (
    prepare_collision_box,
    apply_approach_velocities,
)
from h_state_torch import HStateReferenceBatchedSimulation
from valence_state_torch import ValenceStateBatchedSimulation


OUTPUT = Path(
    "research_data/qm_residual/valence_state_reactive_trajectory.csv"
)

BOX_SIZE = 12.0
TEMPERATURE_K = 250.0
TIME_STEP_FS = 0.25
FRICTION = 0.01
DURATION_PS = 1.0

APPROACH_FACTOR = 5.0
START_GAP_A = 2.5
SAMPLING_MODE = "targeted"
TARGET_ATOM_A = 2  # donor H in hf_surface_scan formaldehyde ordering

SEEDS = tuple(range(8))
GROUP_SIZE = 8

BOND_TAPER_THRESHOLD = 0.35
PERSISTENCE_FS = 20.0
SAMPLE_EVERY_FS = 1.0


def minimum_image(vector, box):
    return vector - box * np.round(vector / box)


def distance(positions, first, second, box):
    delta = np.asarray(positions[second] - positions[first], dtype=float)
    delta = minimum_image(delta, box)
    return float(np.linalg.norm(delta))


def taper_for(symbol_a, symbol_b, separation):
    i = int(R.ELEMENT_INDEX[symbol_a])
    j = int(R.ELEMENT_INDEX[symbol_b])

    inner = float(R.CUTOFF_INNER[i, j])
    outer = float(R.CUTOFF_OUTER[i, j])

    if separation <= inner:
        return 1.0

    if separation >= outer:
        return 0.0

    fraction = (separation - inner) / (outer - inner)

    return float(
        0.5 * (1.0 + math.cos(math.pi * fraction))
    )


def topology_snapshot(positions, box):
    """
    Atom order:
        0 C
        1 O
        2 donor H
        3 spectator H
        4 incoming H
    """

    pairs = {
        "CH_donor": (0, 2, "C", "H"),
        "HH_product": (2, 4, "H", "H"),
        "CH_spectator": (0, 3, "C", "H"),
        "CO": (0, 1, "C", "O"),
    }

    values = {}

    for name, (first, second, symbol_a, symbol_b) in pairs.items():
        r = distance(
            positions,
            first,
            second,
            box,
        )

        values[f"{name}_distance_A"] = r
        values[f"{name}_taper"] = taper_for(
            symbol_a,
            symbol_b,
            r,
        )

    values["product_now"] = bool(
        values["HH_product_taper"] > BOND_TAPER_THRESHOLD
        and values["CH_donor_taper"] < BOND_TAPER_THRESHOLD
        and values["CH_spectator_taper"] > BOND_TAPER_THRESHOLD
        and values["CO_taper"] > BOND_TAPER_THRESHOLD
    )

    return values


def formaldehyde_payload():
    # Build the same formaldehyde geometry used by the current surface code.
    # The fifth atom produced by this helper is its scan-only incoming H;
    # remove it here because the collision runner supplies the partner H.
    symbols, positions = scan.formaldehyde_geometry(
        donor_length=1.09,
        transfer_length=3.0,
    )

    return {
        "id": "formaldehyde_reaction_probe",
        "formula": "CH2O",
        "symbols": list(symbols[:4]),
        "positions": np.asarray(
            positions[:4],
            dtype=float,
        ),
    }


def hydrogen_payload():
    return {
        "id": "atom:H",
        "formula": "H",
        "symbols": ["H"],
        "positions": np.zeros((1, 3), dtype=float),
    }


def prepare_group(seed_group):
    molecule = formaldehyde_payload()
    partner = hydrogen_payload()

    boxes = []
    infos = []

    for seed in seed_group:
        symbols, positions, info = prepare_collision_box(
            molecule,
            partner,
            BOX_SIZE,
            int(seed),
            START_GAP_A,
            "com",
            SAMPLING_MODE,
            TARGET_ATOM_A,
        )

        boxes.append(
            (
                list(symbols),
                np.asarray(positions, dtype=float),
            )
        )
        infos.append(info)

    return boxes, infos


def build_simulation(model_class, boxes, seed, *, relax_on_start):
    simulation = model_class(
        boxes=boxes,
        box_size=BOX_SIZE,
        time_step=TIME_STEP_FS,
        target_temperature=TEMPERATURE_K,
        friction=FRICTION,
        device="cuda",
        dtype=torch.float64,
        random_seed=int(seed),
        relax_on_start=bool(relax_on_start),
    )

    # Match the normal characterisation protocol explicitly.
    simulation.thermostat_is_on = True

    return simulation


def prepare_matched_models(seed_group):
    boxes, infos = prepare_group(seed_group)

    # The real characterisation runner constructs the simulation with its
    # normal startup relaxation enabled, then imposes the directed collision
    # velocity. Use H-state as the common preparation surface so the two
    # models begin from EXACTLY the same relaxed geometry rather than each
    # relaxing to a potentially different starting point.
    control = build_simulation(
        HStateReferenceBatchedSimulation,
        boxes,
        seed_group[0],
        relax_on_start=True,
    )

    collision_measure = apply_approach_velocities(
        control,
        infos,
        APPROACH_FACTOR,
    )

    valence = build_simulation(
        ValenceStateBatchedSimulation,
        boxes,
        seed_group[0],
        relax_on_start=False,
    )

    # Copy the already-relaxed H-state geometry into the experimental model,
    # rebuild all discrete neighbour state, and evaluate its own forces there.
    valence.positions = control.positions.detach().clone()
    valence.reference_positions = None
    valence.build_neighbours()
    valence.forces, valence._potential_energy = valence.compute_forces()

    # Exact same initial kinetic state. Both simulations also own independent
    # torch RNGs seeded identically, so Langevin noise is matched thereafter.
    valence.velocities = control.velocities.detach().clone()

    return control, valence, infos, collision_measure


def per_box_force_arrays(simulation):
    return (
        simulation.forces.detach()
        .cpu()
        .numpy()
        .reshape(
            int(simulation.box_count),
            int(simulation.per_box),
            3,
        )
    )


def initialise_trackers(
    model_name,
    seed_group,
    simulation,
    collision_measure,
):
    positions = simulation.positions_per_box

    forces = per_box_force_arrays(simulation)

    trackers = []

    for box, seed in enumerate(seed_group):
        snapshot = topology_snapshot(
            positions[box],
            simulation.box_size,
        )

        total_energy = (
            float(simulation.potential_per_box[box])
            + float(simulation.thermodynamics_per_box[0][box])
        )

        trackers.append({
            "seed": int(seed),
            "model": model_name,
            "hit": False,
            "reaction_time_fs": None,
            "continuous_product_fs": 0.0,
            "final_retained": False,
            "initial_total_eV": total_energy,
            "max_abs_total_change_eV": 0.0,
            "worst_force_jump_eV_per_A": 0.0,
            "previous_force": forces[box].copy(),
            "minimum_HH_A": snapshot["HH_product_distance_A"],
            "maximum_CH_donor_A": snapshot["CH_donor_distance_A"],
            "maximum_HH_taper": snapshot["HH_product_taper"],
            "minimum_CH_donor_taper": snapshot["CH_donor_taper"],
            "thermal_rms_speed": float(
                collision_measure[box]["thermal_rms_speed"]
            ),
            "relative_approach_speed": float(
                collision_measure[box]["relative_speed"]
            ),
            "caps_start": int(simulation.capped_steps),
            "last_snapshot": snapshot,
        })

    return trackers


def update_trackers(
    trackers,
    simulation,
    elapsed_sample_fs,
):
    positions = simulation.positions_per_box
    forces = per_box_force_arrays(simulation)

    potentials = simulation.potential_per_box
    kinetics, _ = simulation.thermodynamics_per_box

    for box, tracker in enumerate(trackers):
        snapshot = topology_snapshot(
            positions[box],
            simulation.box_size,
        )

        tracker["last_snapshot"] = snapshot

        tracker["minimum_HH_A"] = min(
            tracker["minimum_HH_A"],
            snapshot["HH_product_distance_A"],
        )

        tracker["maximum_CH_donor_A"] = max(
            tracker["maximum_CH_donor_A"],
            snapshot["CH_donor_distance_A"],
        )

        tracker["maximum_HH_taper"] = max(
            tracker["maximum_HH_taper"],
            snapshot["HH_product_taper"],
        )

        tracker["minimum_CH_donor_taper"] = min(
            tracker["minimum_CH_donor_taper"],
            snapshot["CH_donor_taper"],
        )

        total = (
            float(potentials[box])
            + float(kinetics[box])
        )

        tracker["max_abs_total_change_eV"] = max(
            tracker["max_abs_total_change_eV"],
            abs(total - tracker["initial_total_eV"]),
        )

        force_jump = float(
            np.max(
                np.abs(
                    forces[box]
                    - tracker["previous_force"]
                )
            )
        )

        tracker["worst_force_jump_eV_per_A"] = max(
            tracker["worst_force_jump_eV_per_A"],
            force_jump,
        )

        tracker["previous_force"] = forces[box].copy()

        if snapshot["product_now"]:
            tracker["continuous_product_fs"] += elapsed_sample_fs

            if (
                not tracker["hit"]
                and tracker["continuous_product_fs"] >= PERSISTENCE_FS
            ):
                tracker["hit"] = True
                tracker["reaction_time_fs"] = (
                    float(simulation.elapsed_femtoseconds)
                    - tracker["continuous_product_fs"]
                    + elapsed_sample_fs
                )
        else:
            tracker["continuous_product_fs"] = 0.0


def run_group_model(
    model_name,
    simulation,
    seed_group,
    collision_measure,
):
    trackers = initialise_trackers(
        model_name,
        seed_group,
        simulation,
        collision_measure,
    )

    total_steps = int(
        round(
            DURATION_PS * 1000.0 / TIME_STEP_FS
        )
    )

    sample_steps = max(
        1,
        int(round(SAMPLE_EVERY_FS / TIME_STEP_FS)),
    )

    steps_done = 0

    while steps_done < total_steps:
        this_chunk = min(
            sample_steps,
            total_steps - steps_done,
        )

        simulation.target_temperature = TEMPERATURE_K
        simulation.step(this_chunk)

        steps_done += this_chunk

        update_trackers(
            trackers,
            simulation,
            this_chunk * TIME_STEP_FS,
        )

        if not np.all(
            np.isfinite(
                np.asarray(
                    simulation.potential_per_box,
                    dtype=float,
                )
            )
        ):
            raise RuntimeError(
                f"{model_name} became non-finite "
                f"in seed group {seed_group}"
            )

    for tracker in trackers:
        tracker["final_retained"] = bool(
            tracker["last_snapshot"]["product_now"]
        )

        # capped_steps is global over the batched simulation. This tells us
        # whether any cap occurred in the group; group result is duplicated
        # into each row for easy auditing.
        tracker["group_capped_steps"] = int(
            simulation.capped_steps
        )

    return trackers


def summary(rows, model):
    selected = [
        row
        for row in rows
        if row["model"] == model
    ]

    hits = sum(
        bool(row["hit"])
        for row in selected
    )

    retained = sum(
        bool(row["final_retained"])
        for row in selected
    )

    reacting_seeds = [
        row["seed"]
        for row in selected
        if row["hit"]
    ]

    reaction_times = [
        row["reaction_time_fs"]
        for row in selected
        if row["reaction_time_fs"] is not None
    ]

    return {
        "model": model,
        "runs": len(selected),
        "hits": hits,
        "retained": retained,
        "reacting_seeds": reacting_seeds,
        "median_reaction_time_fs": (
            float(np.median(reaction_times))
            if reaction_times
            else float("nan")
        ),
        "max_energy_change_eV": max(
            row["max_abs_total_change_eV"]
            for row in selected
        ),
        "max_force_jump_eV_per_A": max(
            row["worst_force_jump_eV_per_A"]
            for row in selected
        ),
        "groups_with_caps": len({
            row["seed"] // GROUP_SIZE
            for row in selected
            if row["group_capped_steps"] > 0
        }),
    }


def write_csv(rows):
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialised = []

    for row in rows:
        serialised.append({
            "seed": row["seed"],
            "model": row["model"],
            "hit": int(row["hit"]),
            "reaction_time_fs": (
                ""
                if row["reaction_time_fs"] is None
                else row["reaction_time_fs"]
            ),
            "final_retained": int(row["final_retained"]),
            "minimum_HH_A": row["minimum_HH_A"],
            "maximum_CH_donor_A": row["maximum_CH_donor_A"],
            "maximum_HH_taper": row["maximum_HH_taper"],
            "minimum_CH_donor_taper": row["minimum_CH_donor_taper"],
            "max_abs_total_change_eV": row["max_abs_total_change_eV"],
            "worst_force_jump_eV_per_A": row["worst_force_jump_eV_per_A"],
            "group_capped_steps": row["group_capped_steps"],
            "thermal_rms_speed_A_per_fs": row["thermal_rms_speed"],
            "relative_approach_speed_A_per_fs": row["relative_approach_speed"],
            "final_HH_A": row["last_snapshot"]["HH_product_distance_A"],
            "final_CH_donor_A": row["last_snapshot"]["CH_donor_distance_A"],
            "final_CH_spectator_A": row["last_snapshot"]["CH_spectator_distance_A"],
            "final_CO_A": row["last_snapshot"]["CO_distance_A"],
        })

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(serialised[0].keys()),
        )
        writer.writeheader()
        writer.writerows(serialised)


def main():
    print("H + CH2O -> H2 + HCO REACTIVE TRAJECTORY COMPARISON")
    print()
    print(f"runs/model          : {len(SEEDS)}")
    print(f"group size          : {GROUP_SIZE}")
    print(f"temperature         : {TEMPERATURE_K:.1f} K")
    print(f"approach            : {APPROACH_FACTOR:.1f} x thermal RMS")
    print(f"start gap           : {START_GAP_A:.2f} A")
    print(f"sampling            : {SAMPLING_MODE}")
    print(f"target atom         : donor H #{TARGET_ATOM_A}")
    print(f"duration            : {DURATION_PS:.2f} ps")
    print(f"timestep            : {TIME_STEP_FS:.2f} fs")
    print(f"friction            : {FRICTION:.3f}")
    print("thermostat           : on (matched Langevin noise)")
    print("startup relaxation   : H-state relaxed geometry, copied to both models")
    print(
        f"product persistence : {PERSISTENCE_FS:.1f} fs "
        f"at taper threshold {BOND_TAPER_THRESHOLD:.2f}"
    )
    print("energy column        : thermostatted |dE_total|, NOT NVE drift")
    print()

    all_rows = []

    for group_start in range(
        0,
        len(SEEDS),
        GROUP_SIZE,
    ):
        seed_group = SEEDS[
            group_start:group_start + GROUP_SIZE
        ]

        print(
            f"group {seed_group[0]:02d}-{seed_group[-1]:02d}"
        )

        control, valence, infos, collision_measure = (
            prepare_matched_models(seed_group)
        )

        control_rows = run_group_model(
            "H-state",
            control,
            seed_group,
            collision_measure,
        )

        valence_rows = run_group_model(
            "valence-state",
            valence,
            seed_group,
            collision_measure,
        )

        all_rows.extend(control_rows)
        all_rows.extend(valence_rows)

        control_hits = sum(
            row["hit"] for row in control_rows
        )

        valence_hits = sum(
            row["hit"] for row in valence_rows
        )

        print(
            f"  hits H-state={control_hits}/{len(seed_group)}  "
            f"valence={valence_hits}/{len(seed_group)}"
        )

    write_csv(all_rows)

    print()
    print(f"wrote : {OUTPUT}")
    print()

    control = summary(
        all_rows,
        "H-state",
    )

    valence = summary(
        all_rows,
        "valence-state",
    )

    print("REACTION SUMMARY")
    print(
        f"{'model':15s} "
        f"{'hits':>8s} "
        f"{'retained':>10s} "
        f"{'median t/fs':>13s} "
        f"{'max |dE|/eV':>12s} "
        f"{'max dF/eV/A':>14s} "
        f"{'cap groups':>11s}"
    )

    for result in (
        control,
        valence,
    ):
        median = (
            "-"
            if not math.isfinite(
                result["median_reaction_time_fs"]
            )
            else f"{result['median_reaction_time_fs']:.1f}"
        )

        print(
            f"{result['model']:15s} "
            f"{result['hits']:3d}/{result['runs']:<4d} "
            f"{result['retained']:3d}/{result['runs']:<6d} "
            f"{median:>13s} "
            f"{result['max_energy_change_eV']:12.5f} "
            f"{result['max_force_jump_eV_per_A']:14.5f} "
            f"{result['groups_with_caps']:11d}"
        )

    print()
    print("REACTING SEEDS")
    print(
        "  H-state       : "
        + (
            ", ".join(
                str(seed)
                for seed in control["reacting_seeds"]
            )
            if control["reacting_seeds"]
            else "none"
        )
    )

    print(
        "  valence-state : "
        + (
            ", ".join(
                str(seed)
                for seed in valence["reacting_seeds"]
            )
            if valence["reacting_seeds"]
            else "none"
        )
    )

    shared = sorted(
        set(control["reacting_seeds"])
        & set(valence["reacting_seeds"])
    )

    only_control = sorted(
        set(control["reacting_seeds"])
        - set(valence["reacting_seeds"])
    )

    only_valence = sorted(
        set(valence["reacting_seeds"])
        - set(control["reacting_seeds"])
    )

    print()
    print("MATCHED-SEED OUTCOME")
    print(
        "  reacted in both      : "
        + (", ".join(map(str, shared)) if shared else "none")
    )
    print(
        "  H-state only         : "
        + (", ".join(map(str, only_control)) if only_control else "none")
    )
    print(
        "  valence-state only   : "
        + (", ".join(map(str, only_valence)) if only_valence else "none")
    )


if __name__ == "__main__":
    main()
