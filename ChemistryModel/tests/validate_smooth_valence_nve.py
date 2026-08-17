"""
NVE validation for the differentiable smooth-valence topology prototype.

Compares, from matched initial positions and velocities:

    ordinary H-state
    smooth-valence H-state

Systems:
    1. intact water reactant reference
    2. water in the dense state-competition region (x = 1.160 A)
    3. H2O2 at its existing O-O reference geometry

For every run:
    - thermostat off
    - friction 0
    - CPU float64
    - same random seed / same copied initial velocities
    - current engine timestep unchanged
    - track total-energy drift every step
    - track force jumps and move-cap events
    - for the smooth model, track directed O-O memberships

This does NOT modify production physics.

Run:
    py validate_smooth_valence_nve.py

Output:
    research_data/qm_residual/smooth_valence_nve.csv
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from bond_calibration import hydrogen_peroxide_geometry
from h_state_torch import HStateReferenceBatchedSimulation
from probe_smooth_valence_forces import SmoothValenceForceSimulation


GEOMETRIES = Path("research_data/qm_residual/dense_scan_geometries.json")
OUTPUT = Path("research_data/qm_residual/smooth_valence_nve.csv")

BOX_SIZE = 30.0
STEPS = 1000
TEMPERATURE_K = 100.0
SEED = 913


class AuditSmoothSimulation(SmoothValenceForceSimulation):
    """Smooth prototype with detached read-only membership snapshots."""

    def _smooth_membership(self, values):
        membership = super()._smooth_membership(values)

        self._last_membership = membership.detach().cpu().clone()
        self._last_neighbours = values["neighbours"].detach().cpu().clone()
        self._last_mask = values["mask"].detach().cpu().clone()

        return membership


def load_payload():
    return json.loads(
        GEOMETRIES.read_text(encoding="utf-8")
    )


def find_reference(payload, system):
    matches = [
        geometry
        for geometry in payload["geometries"]
        if (
            geometry["system"] == system
            and geometry["sample_kind"] == "reactant_reference"
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {system} reactant reference, got {len(matches)}"
        )

    return matches[0]


def find_dense_water(payload, x_target):
    matches = [
        geometry
        for geometry in payload["geometries"]
        if (
            geometry["system"] == "water"
            and geometry["sample_kind"] == "dense_transfer_scan"
            and geometry.get("reaction_coordinate", {}).get(
                "transfer_distance_angstrom"
            ) is not None
            and abs(
                float(
                    geometry["reaction_coordinate"][
                        "transfer_distance_angstrom"
                    ]
                )
                - x_target
            ) < 1e-9
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one dense water x={x_target:.3f}, got {len(matches)}"
        )

    return matches[0]


def centre_in_box(coordinates):
    positions = np.asarray(coordinates, dtype=float).copy()

    centre = positions.mean(axis=0)

    return positions - centre + 0.5 * BOX_SIZE


def build_systems(payload):
    intact = find_reference(payload, "water")
    competition = find_dense_water(payload, 1.160)

    peroxide_symbols, peroxide_positions = hydrogen_peroxide_geometry(
        oo_distance=1.475
    )

    return [
        {
            "name": "water_intact",
            "symbols": list(intact["symbols"]),
            "positions": centre_in_box(
                intact["coordinates_angstrom"]
            ),
        },
        {
            "name": "water_competition_x1.160",
            "symbols": list(competition["symbols"]),
            "positions": centre_in_box(
                competition["coordinates_angstrom"]
            ),
        },
        {
            "name": "peroxide_OO_1.475",
            "symbols": list(peroxide_symbols),
            "positions": centre_in_box(peroxide_positions),
        },
    ]


def make_simulation(cls, symbols, positions):
    simulation = cls(
        boxes=[(
            symbols,
            np.asarray(positions, dtype=float),
        )],
        box_size=BOX_SIZE,
        target_temperature=TEMPERATURE_K,
        friction=0.0,
        device="cpu",
        dtype=torch.float64,
        random_seed=SEED,
        relax_on_start=False,
    )

    simulation.thermostat_is_on = False

    return simulation


def total_energy(simulation):
    return float(
        simulation.potential_energy
        + simulation.kinetic_energy
    )


def max_force(simulation):
    return float(
        torch.max(
            torch.abs(simulation.forces)
        ).detach().cpu()
    )


def oo_memberships(simulation, symbols):
    if not isinstance(simulation, AuditSmoothSimulation):
        return float("nan"), float("nan")

    oxygens = [
        index
        for index, symbol in enumerate(symbols)
        if symbol == "O"
    ]

    if len(oxygens) != 2:
        return float("nan"), float("nan")

    first, second = oxygens

    def directed(centre, other):
        neighbours = simulation._last_neighbours[centre]
        mask = simulation._last_mask[centre]
        membership = simulation._last_membership[centre]

        for slot in range(len(neighbours)):
            if (
                bool(mask[slot])
                and int(neighbours[slot]) == other
            ):
                return float(membership[slot])

        return 0.0

    return (
        directed(first, second),
        directed(second, first),
    )


def run_model(
    system_name,
    symbols,
    positions,
    model_name,
    simulation,
):
    rows = []

    initial_energy = total_energy(simulation)
    previous_force = simulation.forces.detach().cpu().numpy().copy()

    initial_caps = int(simulation.capped_steps)

    def record(step):
        energy = total_energy(simulation)
        oo_01, oo_10 = oo_memberships(
            simulation,
            symbols,
        )

        return {
            "system": system_name,
            "model": model_name,
            "step": step,
            "elapsed_fs": float(simulation.elapsed_femtoseconds),
            "potential_eV": float(simulation.potential_energy),
            "kinetic_eV": float(simulation.kinetic_energy),
            "total_eV": energy,
            "drift_eV": energy - initial_energy,
            "temperature_K": float(simulation.temperature),
            "max_force_eV_per_A": max_force(simulation),
            "force_jump_eV_per_A": 0.0,
            "capped_steps": int(simulation.capped_steps),
            "oo_membership_01": oo_01,
            "oo_membership_10": oo_10,
        }

    rows.append(record(0))

    worst_force_jump = 0.0

    for step in range(1, STEPS + 1):
        simulation.step(1)

        current_force = (
            simulation.forces.detach().cpu().numpy().copy()
        )

        force_jump = float(
            np.max(
                np.abs(
                    current_force - previous_force
                )
            )
        )

        previous_force = current_force

        row = record(step)
        row["force_jump_eV_per_A"] = force_jump

        rows.append(row)

        worst_force_jump = max(
            worst_force_jump,
            force_jump,
        )

        if not math.isfinite(row["total_eV"]):
            raise RuntimeError(
                f"{system_name}/{model_name} became non-finite "
                f"at step {step}"
            )

    drifts = [
        row["drift_eV"]
        for row in rows
    ]

    energies = [
        row["total_eV"]
        for row in rows
    ]

    oo_01 = [
        row["oo_membership_01"]
        for row in rows
        if math.isfinite(row["oo_membership_01"])
    ]

    oo_10 = [
        row["oo_membership_10"]
        for row in rows
        if math.isfinite(row["oo_membership_10"])
    ]

    summary = {
        "system": system_name,
        "model": model_name,
        "dt_fs": float(simulation.time_step),
        "steps": STEPS,
        "elapsed_fs": float(simulation.elapsed_femtoseconds),
        "energy_start_eV": energies[0],
        "energy_end_eV": energies[-1],
        "final_drift_eV": drifts[-1],
        "max_abs_drift_eV": max(abs(value) for value in drifts),
        "energy_span_eV": max(energies) - min(energies),
        "worst_force_jump_eV_per_A": worst_force_jump,
        "capped_steps_added": (
            int(simulation.capped_steps)
            - initial_caps
        ),
        "oo_membership_01_min": (
            min(oo_01) if oo_01 else float("nan")
        ),
        "oo_membership_01_max": (
            max(oo_01) if oo_01 else float("nan")
        ),
        "oo_membership_10_min": (
            min(oo_10) if oo_10 else float("nan")
        ),
        "oo_membership_10_max": (
            max(oo_10) if oo_10 else float("nan")
        ),
    }

    return rows, summary


def main():
    payload = load_payload()
    systems = build_systems(payload)

    print("SMOOTH-VALENCE NVE VALIDATION")
    print()
    print(f"steps        : {STEPS}")
    print(f"temperature  : {TEMPERATURE_K:.1f} K")
    print(f"seed         : {SEED}")
    print("thermostat   : off")
    print("friction     : 0")
    print("device       : CPU / float64")
    print()

    all_rows = []
    summaries = []

    for system in systems:
        print(f"{system['name']}")

        # Build the control first, then copy its exact velocities into the
        # smooth model so the two runs begin with identical kinetic state.
        control = make_simulation(
            HStateReferenceBatchedSimulation,
            system["symbols"],
            system["positions"],
        )

        smooth = make_simulation(
            AuditSmoothSimulation,
            system["symbols"],
            system["positions"],
        )

        smooth.velocities = control.velocities.detach().clone()

        # Velocities do not affect forces/potential, but refresh the kinetic
        # bookkeeping implicitly through the live property.
        control_rows, control_summary = run_model(
            system["name"],
            system["symbols"],
            system["positions"],
            "H-state",
            control,
        )

        smooth_rows, smooth_summary = run_model(
            system["name"],
            system["symbols"],
            system["positions"],
            "smooth",
            smooth,
        )

        all_rows.extend(control_rows)
        all_rows.extend(smooth_rows)

        summaries.append(control_summary)
        summaries.append(smooth_summary)

        for summary in (
            control_summary,
            smooth_summary,
        ):
            print(
                f"  {summary['model']:<8s} "
                f"dt={summary['dt_fs']:.6g} fs  "
                f"drift={summary['final_drift_eV']:+.6e} eV  "
                f"max|drift|={summary['max_abs_drift_eV']:.6e} eV  "
                f"span={summary['energy_span_eV']:.6e} eV  "
                f"force_jump={summary['worst_force_jump_eV_per_A']:.6e} eV/A  "
                f"caps={summary['capped_steps_added']}"
            )

            if summary["model"] == "smooth":
                print(
                    f"           O-O membership "
                    f"01={summary['oo_membership_01_min']:.5f}"
                    f"..{summary['oo_membership_01_max']:.5f}  "
                    f"10={summary['oo_membership_10_min']:.5f}"
                    f"..{summary['oo_membership_10_max']:.5f}"
                )

        print()

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
            fieldnames=list(all_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"wrote : {OUTPUT}")
    print()

    print("CONTROL COMPARISON")
    print(
        f"{'system':28s} "
        f"{'H max drift':>13s} "
        f"{'smooth max':>13s} "
        f"{'ratio':>9s} "
        f"{'H caps':>7s} "
        f"{'S caps':>7s}"
    )

    by_system = {}

    for summary in summaries:
        by_system.setdefault(
            summary["system"],
            {},
        )[summary["model"]] = summary

    for system_name, pair in by_system.items():
        control = pair["H-state"]
        smooth = pair["smooth"]

        denominator = max(
            control["max_abs_drift_eV"],
            1e-15,
        )

        ratio = (
            smooth["max_abs_drift_eV"]
            / denominator
        )

        print(
            f"{system_name:28s} "
            f"{control['max_abs_drift_eV']:13.6e} "
            f"{smooth['max_abs_drift_eV']:13.6e} "
            f"{ratio:9.3f} "
            f"{control['capped_steps_added']:7d} "
            f"{smooth['capped_steps_added']:7d}"
        )


if __name__ == "__main__":
    main()
