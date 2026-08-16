"""
Controlled NVE validation for the factorisable H-state model.

Prerequisite:
    validate_h_state_factorised.py must PASS.

Cases
-----
1. SINGLE-COMPONENT MATCHED TRAJECTORY
   Historical whole-box H-state vs factorisable H-state from exactly the same
   H3 geometry and velocity state. Since the 106-point microscope showed
   static equality, their short NVE trajectories should also remain matched.

2. TWO DISCONNECTED UNEQUAL COMPONENTS
   Two different H3 competition networks evolve in one box with the thermostat
   off. This checks ordinary NVE conservation after the locality correction.

3. DYNAMIC COMPONENT MERGE
   Two H3 competition networks start just outside the H-H component bridge
   cutoff and are given equal/opposite COM velocities so they cross from
   2 components -> 1 component during actual velocity-Verlet dynamics.
   This is the dynamic version of the static merge/split continuity test.

Metrics
-------
For each NVE trajectory:
    max |E(t)-E(0)|
    RMS energy drift
    final energy drift
    capped steps
    neighbour rebuilds

The merge case additionally records component-count transitions and bridge
distance.

No thermostat is used in any case.

Run:
    py validate_h_state_factorised_nve.py

Outputs:
    research_data/qm_residual/h_state_factorised_nve_single.csv
    research_data/qm_residual/h_state_factorised_nve_disconnected.csv
    research_data/qm_residual/h_state_factorised_nve_merge.csv
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch

import reactive as R

from h_state_torch import HStateReferenceBatchedSimulation
from h_state_factorised_torch import FactorisedHStateBatchedSimulation


OUT_DIR = Path("research_data/qm_residual")

BOX_SIZE = 40.0
DTYPE = torch.float64
DEVICE = "cpu"

# Smaller than the production 0.25 fs because this is an integrator/continuity
# microscope, not a throughput test.
SINGLE_DT_FS = 0.05
SINGLE_STEPS = 1000
SINGLE_SAMPLE_EVERY = 10

DISCONNECTED_DT_FS = 0.05
DISCONNECTED_STEPS = 1000
DISCONNECTED_SAMPLE_EVERY = 10

MERGE_DT_FS = 0.001
MERGE_STEPS = 200
MERGE_SAMPLE_EVERY = 1

# Start microscopically outside the H-H component cutoff and cross it quickly.
# The previous 0.006 A / 0.003 A/fs setup never merged because the internal
# H3 forces pushed the facing atoms apart faster than the imposed COM approach.
# This is intentionally a short boundary-crossing microscope, not a collision
# experiment.
MERGE_START_OUTSIDE_A = 0.0002
MERGE_RELATIVE_SPEED_A_PER_FS = 0.05

# Diagnostic tolerances. These are intentionally conservative compared with
# float64 noise; the point is to catch a topology/integrator pathology, not to
# demand mathematically exact symplectic conservation from finite dt.
NVE_MAX_DRIFT_TOL_EV = 2.0e-3
NVE_RMS_DRIFT_TOL_EV = 1.0e-3

MATCH_POSITION_TOL_A = 1.0e-8
MATCH_VELOCITY_TOL_A_PER_FS = 1.0e-9
MATCH_FORCE_TOL_EV_PER_A = 1.0e-7
MATCH_ENERGY_TOL_EV = 1.0e-9


def hh_cutoffs():
    h = int(R.ELEMENT_INDEX["H"])
    return (
        float(R.CUTOFF_INNER[h, h]),
        float(R.CUTOFF_OUTER[h, h]),
    )


def choose_spacings():
    inner, outer = hh_cutoffs()
    span = outer - inner

    a = max(inner + 0.25 * span, 0.55 * outer)
    b = max(inner + 0.62 * span, 0.58 * outer)

    a = min(a, outer - max(0.02, 0.05 * span))
    b = min(b, outer - max(0.01, 0.025 * span))

    if not (0.5 * outer < a < outer and 0.5 * outer < b < outer):
        raise RuntimeError("could not choose safe H3 spacings")

    if 2.0 * a <= outer or 2.0 * b <= outer:
        raise RuntimeError("chosen H3 spacing activates end-to-end H-H contact")

    return a, b


def h3_positions(spacing, origin):
    origin = np.asarray(origin, dtype=float)
    return np.asarray([
        [0.0, 0.0, 0.0],
        [spacing, 0.0, 0.0],
        [2.0 * spacing, 0.0, 0.0],
    ], dtype=float) + origin


def build(model_class, symbols, positions, dt_fs):
    sim = model_class(
        boxes=[(
            list(symbols),
            np.asarray(positions, dtype=float),
        )],
        box_size=BOX_SIZE,
        time_step=float(dt_fs),
        target_temperature=0.0,
        friction=0.0,
        device=DEVICE,
        dtype=DTYPE,
        random_seed=0,
        relax_on_start=False,
    )

    sim.thermostat_is_on = False
    return sim


def set_velocities(sim, velocities):
    sim.velocities = torch.tensor(
        np.asarray(velocities, dtype=float),
        device=sim.device,
        dtype=sim.dtype,
    )


def total_energy(sim):
    potential = float(sim.potential_per_box[0])
    kinetic = float(sim.kinetic_per_box[0])
    return potential, kinetic, potential + kinetic


def minimum_image(delta, box):
    delta = np.asarray(delta, dtype=float)
    return delta - box * np.round(delta / box)


def distance(positions, first, second, box):
    return float(np.linalg.norm(
        minimum_image(
            positions[second] - positions[first],
            box,
        )
    ))


def component_count(sim):
    diag = getattr(sim, "_h_component_diagnostics", None)

    if not diag:
        # Force/energy evaluation normally populates this. If a caller samples
        # before that happened, trigger the already-current potential path.
        _ = sim.potential_per_box
        diag = getattr(sim, "_h_component_diagnostics", None)

    if not diag:
        return None, None

    counts = diag.get("component_counts_per_box", ())
    count = int(counts[0]) if counts else 0
    largest = int(diag.get("largest_component_edges", 0))
    return count, largest


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def drift_summary(rows):
    drifts = np.asarray(
        [float(row["energy_drift_eV"]) for row in rows],
        dtype=float,
    )

    return {
        "max_abs_drift_eV": float(np.max(np.abs(drifts))),
        "rms_drift_eV": float(np.sqrt(np.mean(drifts * drifts))),
        "final_drift_eV": float(drifts[-1]),
    }


def sample_nve(sim, step_index, initial_energy, *, bridge_pair=None):
    potential, kinetic, total = total_energy(sim)
    positions = sim.positions_per_box[0]

    count, largest = component_count(sim)

    row = {
        "step": int(step_index),
        "time_fs": float(sim.elapsed_femtoseconds),
        "potential_eV": potential,
        "kinetic_eV": kinetic,
        "total_eV": total,
        "energy_drift_eV": total - initial_energy,
        "component_count": "" if count is None else count,
        "largest_component_edges": "" if largest is None else largest,
        "capped_steps": int(sim.capped_steps),
        "rebuild_count": int(sim.rebuild_count),
    }

    if bridge_pair is not None:
        row["bridge_distance_A"] = distance(
            positions,
            bridge_pair[0],
            bridge_pair[1],
            sim.box_size,
        )

    return row


# ---------------------------------------------------------------------------
# 1. Matched old/new single-component trajectory
# ---------------------------------------------------------------------------

def run_single_matched(spacing):
    symbols = ["H", "H", "H"]
    positions = h3_positions(
        spacing,
        [12.0, 12.0, 12.0],
    )

    # Small asymmetric internal velocity pattern with zero-ish net COM.
    velocities = np.asarray([
        [+0.0005, +0.0002, 0.0],
        [-0.0002, -0.0003, 0.0],
        [-0.0003, +0.0001, 0.0],
    ], dtype=float)

    old = build(
        HStateReferenceBatchedSimulation,
        symbols,
        positions,
        SINGLE_DT_FS,
    )

    new = build(
        FactorisedHStateBatchedSimulation,
        symbols,
        positions,
        SINGLE_DT_FS,
    )

    set_velocities(old, velocities)
    set_velocities(new, velocities)

    _, _, e0_old = total_energy(old)
    _, _, e0_new = total_energy(new)

    rows = []

    max_position_difference = 0.0
    max_velocity_difference = 0.0
    max_force_difference = 0.0
    max_energy_difference = 0.0

    def compare(step_index):
        nonlocal max_position_difference
        nonlocal max_velocity_difference
        nonlocal max_force_difference
        nonlocal max_energy_difference

        p_old = old.positions.detach().cpu().numpy()
        p_new = new.positions.detach().cpu().numpy()

        v_old = old.velocities.detach().cpu().numpy()
        v_new = new.velocities.detach().cpu().numpy()

        f_old = old.forces.detach().cpu().numpy()
        f_new = new.forces.detach().cpu().numpy()

        _, _, e_old = total_energy(old)
        _, _, e_new = total_energy(new)

        dp = float(np.max(np.abs(p_new - p_old)))
        dv = float(np.max(np.abs(v_new - v_old)))
        df = float(np.max(np.abs(f_new - f_old)))
        de = abs(e_new - e_old)

        max_position_difference = max(max_position_difference, dp)
        max_velocity_difference = max(max_velocity_difference, dv)
        max_force_difference = max(max_force_difference, df)
        max_energy_difference = max(max_energy_difference, de)

        rows.append({
            "step": int(step_index),
            "time_fs": float(old.elapsed_femtoseconds),
            "old_total_eV": e_old,
            "new_total_eV": e_new,
            "old_drift_eV": e_old - e0_old,
            "new_drift_eV": e_new - e0_new,
            "max_position_difference_A": dp,
            "max_velocity_difference_A_per_fs": dv,
            "max_force_difference_eV_per_A": df,
            "energy_difference_eV": e_new - e_old,
            "old_capped_steps": int(old.capped_steps),
            "new_capped_steps": int(new.capped_steps),
        })

    compare(0)

    for step in range(1, SINGLE_STEPS + 1):
        old.step()
        new.step()

        if (
            step % SINGLE_SAMPLE_EVERY == 0
            or step == SINGLE_STEPS
        ):
            compare(step)

    path = OUT_DIR / "h_state_factorised_nve_single.csv"
    write_csv(path, rows)

    old_drifts = np.asarray(
        [row["old_drift_eV"] for row in rows],
        dtype=float,
    )

    new_drifts = np.asarray(
        [row["new_drift_eV"] for row in rows],
        dtype=float,
    )

    return {
        "path": path,
        "max_position_difference_A": max_position_difference,
        "max_velocity_difference_A_per_fs": max_velocity_difference,
        "max_force_difference_eV_per_A": max_force_difference,
        "max_energy_difference_eV": max_energy_difference,
        "old_max_drift_eV": float(np.max(np.abs(old_drifts))),
        "new_max_drift_eV": float(np.max(np.abs(new_drifts))),
        "old_capped_steps": int(old.capped_steps),
        "new_capped_steps": int(new.capped_steps),
        "passed": (
            max_position_difference <= MATCH_POSITION_TOL_A
            and max_velocity_difference <= MATCH_VELOCITY_TOL_A_PER_FS
            and max_force_difference <= MATCH_FORCE_TOL_EV_PER_A
            and max_energy_difference <= MATCH_ENERGY_TOL_EV
            and old.capped_steps == 0
            and new.capped_steps == 0
        ),
    }


# ---------------------------------------------------------------------------
# 2. Two disconnected unequal components
# ---------------------------------------------------------------------------

def run_disconnected(spacing_a, spacing_b):
    positions_a = h3_positions(
        spacing_a,
        [8.0, 8.0, 8.0],
    )

    positions_b = h3_positions(
        spacing_b,
        [8.0, 23.0, 8.0],
    )

    positions = np.vstack([positions_a, positions_b])

    sim = build(
        FactorisedHStateBatchedSimulation,
        ["H"] * 6,
        positions,
        DISCONNECTED_DT_FS,
    )

    # Independent tiny internal motions. No COM approach.
    velocities = np.asarray([
        [+0.0004, +0.0001, 0.0],
        [-0.0001, -0.0002, 0.0],
        [-0.0003, +0.0001, 0.0],
        [-0.0002, +0.0002, 0.0],
        [+0.0005, -0.0001, 0.0],
        [-0.0003, -0.0001, 0.0],
    ], dtype=float)

    set_velocities(sim, velocities)

    _, _, initial = total_energy(sim)

    rows = [
        sample_nve(
            sim,
            0,
            initial,
        )
    ]

    component_counts = {
        int(rows[0]["component_count"])
    }

    for step in range(1, DISCONNECTED_STEPS + 1):
        sim.step()

        if (
            step % DISCONNECTED_SAMPLE_EVERY == 0
            or step == DISCONNECTED_STEPS
        ):
            row = sample_nve(
                sim,
                step,
                initial,
            )
            rows.append(row)

            if row["component_count"] != "":
                component_counts.add(
                    int(row["component_count"])
                )

    path = OUT_DIR / "h_state_factorised_nve_disconnected.csv"
    write_csv(path, rows)

    summary = drift_summary(rows)
    summary.update({
        "path": path,
        "component_counts": tuple(sorted(component_counts)),
        "capped_steps": int(sim.capped_steps),
        "rebuild_count": int(sim.rebuild_count),
    })

    summary["passed"] = (
        summary["max_abs_drift_eV"] <= NVE_MAX_DRIFT_TOL_EV
        and summary["rms_drift_eV"] <= NVE_RMS_DRIFT_TOL_EV
        and sim.capped_steps == 0
        and component_counts == {2}
    )

    return summary


# ---------------------------------------------------------------------------
# 3. Dynamic 2 -> 1 component merge
# ---------------------------------------------------------------------------

def run_merge(spacing_a, spacing_b):
    _, outer = hh_cutoffs()

    gap = outer + MERGE_START_OUTSIDE_A

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

    positions = np.vstack([positions_a, positions_b])

    sim = build(
        FactorisedHStateBatchedSimulation,
        ["H"] * 6,
        positions,
        MERGE_DT_FS,
    )

    # Equal/opposite COM translation. Relative bridge-closing speed is the
    # difference between the two cluster velocities.
    half = 0.5 * MERGE_RELATIVE_SPEED_A_PER_FS

    velocities = np.zeros((6, 3), dtype=float)
    velocities[:3, 0] = +half
    velocities[3:, 0] = -half

    set_velocities(sim, velocities)

    _, _, initial = total_energy(sim)

    rows = [
        sample_nve(
            sim,
            0,
            initial,
            bridge_pair=(2, 3),
        )
    ]

    counts_seen = [
        int(rows[0]["component_count"])
    ]

    transitions = []

    previous_count = counts_seen[-1]

    for step in range(1, MERGE_STEPS + 1):
        sim.step()

        count, _ = component_count(sim)

        if count is not None and count != previous_count:
            positions_now = sim.positions_per_box[0]
            bridge_now = distance(
                positions_now,
                2,
                3,
                sim.box_size,
            )

            transitions.append({
                "step": int(step),
                "time_fs": float(sim.elapsed_femtoseconds),
                "from": int(previous_count),
                "to": int(count),
                "bridge_distance_A": bridge_now,
            })

            previous_count = int(count)

        if count is not None:
            counts_seen.append(int(count))

        if (
            step % MERGE_SAMPLE_EVERY == 0
            or step == MERGE_STEPS
        ):
            rows.append(
                sample_nve(
                    sim,
                    step,
                    initial,
                    bridge_pair=(2, 3),
                )
            )

    path = OUT_DIR / "h_state_factorised_nve_merge.csv"
    write_csv(path, rows)

    summary = drift_summary(rows)
    summary.update({
        "path": path,
        "counts_seen": tuple(sorted(set(counts_seen))),
        "transitions": transitions,
        "capped_steps": int(sim.capped_steps),
        "rebuild_count": int(sim.rebuild_count),
        "initial_bridge_A": float(rows[0]["bridge_distance_A"]),
        "minimum_bridge_A": float(
            min(float(row["bridge_distance_A"]) for row in rows)
        ),
        "final_bridge_A": float(rows[-1]["bridge_distance_A"]),
    })

    observed_merge = any(
        transition["from"] == 2 and transition["to"] == 1
        for transition in transitions
    )

    summary["observed_merge"] = observed_merge

    summary["passed"] = (
        observed_merge
        and summary["max_abs_drift_eV"] <= NVE_MAX_DRIFT_TOL_EV
        and summary["rms_drift_eV"] <= NVE_RMS_DRIFT_TOL_EV
        and sim.capped_steps == 0
    )

    return summary


def print_drift(label, result):
    print(label)
    print(
        f"  max |dE|          : {result['max_abs_drift_eV']:.9e} eV"
    )
    print(
        f"  RMS dE            : {result['rms_drift_eV']:.9e} eV"
    )
    print(
        f"  final dE          : {result['final_drift_eV']:+.9e} eV"
    )
    print(
        f"  capped steps      : {result['capped_steps']}"
    )
    print(
        f"  neighbour rebuilds: {result['rebuild_count']}"
    )
    print(
        f"  wrote             : {result['path']}"
    )
    print(
        "  result            : "
        + ("PASS" if result["passed"] else "FAIL")
    )
    print()


def main():
    spacing_a, spacing_b = choose_spacings()

    print("FACTORISABLE H-STATE NVE VALIDATION")
    print()
    print(f"device / dtype      : {DEVICE} / {DTYPE}")
    print(
        f"H3 spacings         : {spacing_a:.6f} A, {spacing_b:.6f} A"
    )
    print()

    print("1. SINGLE-COMPONENT MATCHED DYNAMICS")

    single = run_single_matched(spacing_a)

    print(
        f"  max |dPosition|   : "
        f"{single['max_position_difference_A']:.12e} A"
    )
    print(
        f"  max |dVelocity|   : "
        f"{single['max_velocity_difference_A_per_fs']:.12e} A/fs"
    )
    print(
        f"  max |dForce|      : "
        f"{single['max_force_difference_eV_per_A']:.12e} eV/A"
    )
    print(
        f"  max |dEnergy|     : "
        f"{single['max_energy_difference_eV']:.12e} eV"
    )
    print(
        f"  old max |dE_NVE|  : "
        f"{single['old_max_drift_eV']:.9e} eV"
    )
    print(
        f"  new max |dE_NVE|  : "
        f"{single['new_max_drift_eV']:.9e} eV"
    )
    print(
        f"  caps old/new      : "
        f"{single['old_capped_steps']} / {single['new_capped_steps']}"
    )
    print(
        f"  wrote             : {single['path']}"
    )
    print(
        "  result            : "
        + ("PASS" if single["passed"] else "FAIL")
    )

    print()
    print("2. TWO DISCONNECTED UNEQUAL COMPONENTS")

    disconnected = run_disconnected(
        spacing_a,
        spacing_b,
    )

    print(
        f"  components seen   : {disconnected['component_counts']}"
    )
    print_drift("", disconnected)

    print("3. DYNAMIC COMPONENT MERGE")

    merge = run_merge(
        spacing_a,
        spacing_b,
    )

    print(
        f"  bridge initial/min/final : "
        f"{merge['initial_bridge_A']:.9f} / "
        f"{merge['minimum_bridge_A']:.9f} / "
        f"{merge['final_bridge_A']:.9f} A"
    )
    print(
        f"  component counts seen    : {merge['counts_seen']}"
    )
    print(
        f"  observed 2->1 merge      : "
        f"{'yes' if merge['observed_merge'] else 'NO'}"
    )

    if merge["transitions"]:
        print("  transitions:")
        for transition in merge["transitions"]:
            print(
                f"    step {transition['step']:5d}  "
                f"t={transition['time_fs']:.6f} fs  "
                f"{transition['from']}->{transition['to']}  "
                f"bridge={transition['bridge_distance_A']:.12f} A"
            )
    else:
        print("  transitions              : none")

    print_drift("", merge)

    print("FINAL")

    all_pass = (
        single["passed"]
        and disconnected["passed"]
        and merge["passed"]
    )

    if all_pass:
        print(
            "  PASS - factorisable H-state preserves matched single-component "
            "dynamics, conserves energy for disconnected components, and "
            "crosses a live 2->1 component merge without an NVE pathology."
        )
        print(
            "  Next: integrate this H-state formulation into the experimental "
            "heavy-valence engine, rerun its equivalence/QM/force/NVE suite, "
            "then return to performance engineering."
        )
        return

    print(
        "  FAIL - do not promote the factorisable H-state yet."
    )

    if not single["passed"]:
        print(
            "  matched single-component dynamics diverged from the historical "
            "reference."
        )

    if not disconnected["passed"]:
        print(
            "  disconnected-component NVE failed the drift/component-count gate."
        )

    if not merge["passed"]:
        if not merge["observed_merge"]:
            print(
                "  merge trajectory never actually crossed from two components "
                "to one; adjust the controlled approach before judging continuity."
            )
        else:
            print(
                "  live component merging produced excessive NVE drift or a "
                "movement cap."
            )

    raise SystemExit(1)


if __name__ == "__main__":
    main()
