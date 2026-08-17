"""
Promotion-equivalence validation for valence_state_torch.py.

Compares the validated scratch prototype:
    probe_smooth_valence_forces.SmoothValenceForceSimulation

against the promoted experimental engine:
    valence_state_torch.ValenceStateBatchedSimulation

Checks:
    1. energy equivalence on all 106 QM microscope geometries
    2. force equivalence at representative water geometries
    3. short matched NVE trajectory equivalence for:
         - intact water
         - water state-competition geometry
         - H2O2

If this passes, promotion from scratch prototype -> engine module did not
change the validated physics.

Run:
    py validate_valence_state_promotion.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

import json
import math
from pathlib import Path

import numpy as np
import torch

from bond_calibration import hydrogen_peroxide_geometry
from probe_smooth_valence_forces import SmoothValenceForceSimulation
from valence_state_torch import ValenceStateBatchedSimulation


GEOMETRIES = Path("research_data/qm_residual/dense_scan_geometries.json")

BOX_SIZE = 30.0
TEMPERATURE_K = 100.0
SEED = 913

FORCE_POINTS = (1.080, 1.160, 1.320, 1.535)

NVE_STEPS = 250

ENERGY_TOL = 1.0e-10
FORCE_TOL = 1.0e-9
NVE_ENERGY_TOL = 1.0e-9
NVE_POSITION_TOL = 1.0e-10
NVE_VELOCITY_TOL = 1.0e-10


def load_payload():
    return json.loads(
        GEOMETRIES.read_text(encoding="utf-8")
    )


def build_simulation(cls, symbols, positions):
    simulation = cls(
        boxes=[(
            list(symbols),
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


def live_energy_force(simulation, coordinates):
    positions = torch.tensor(
        np.asarray(coordinates, dtype=float),
        dtype=torch.float64,
        device="cpu",
        requires_grad=True,
    )

    energy = simulation.energy_per_atom(
        positions
    ).sum()

    gradient = torch.autograd.grad(
        energy,
        positions,
        create_graph=False,
        retain_graph=False,
    )[0]

    return (
        float(energy.detach().cpu()),
        (-gradient).detach().cpu().numpy(),
    )


def dense_water(payload):
    rows = [
        geometry
        for geometry in payload["geometries"]
        if (
            geometry["system"] == "water"
            and geometry["sample_kind"] == "dense_transfer_scan"
            and geometry.get("reaction_coordinate", {}).get(
                "transfer_distance_angstrom"
            ) is not None
        )
    ]

    rows.sort(
        key=lambda geometry: float(
            geometry["reaction_coordinate"][
                "transfer_distance_angstrom"
            ]
        )
    )

    return rows


def interpolate_water(rows, x):
    for left, right in zip(rows, rows[1:]):
        x0 = float(
            left["reaction_coordinate"][
                "transfer_distance_angstrom"
            ]
        )

        x1 = float(
            right["reaction_coordinate"][
                "transfer_distance_angstrom"
            ]
        )

        if x0 - 1e-12 <= x <= x1 + 1e-12:
            fraction = (
                0.0
                if x1 == x0
                else (x - x0) / (x1 - x0)
            )

            p0 = np.asarray(
                left["coordinates_angstrom"],
                dtype=float,
            )

            p1 = np.asarray(
                right["coordinates_angstrom"],
                dtype=float,
            )

            return {
                "symbols": list(left["symbols"]),
                "positions": (
                    p0 + fraction * (p1 - p0)
                ),
            }

    raise ValueError(
        f"x={x:.6f} outside dense water path"
    )


def centre_in_box(coordinates):
    positions = np.asarray(
        coordinates,
        dtype=float,
    ).copy()

    centre = positions.mean(axis=0)

    return (
        positions
        - centre
        + 0.5 * BOX_SIZE
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


def find_dense_water_exact(payload, x):
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
                - x
            ) < 1e-9
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one water dense geometry at x={x}, got {len(matches)}"
        )

    return matches[0]


def promotion_energy_check(payload):
    worst = None

    for geometry in payload["geometries"]:
        symbols = geometry["symbols"]
        positions = geometry["coordinates_angstrom"]

        scratch = build_simulation(
            SmoothValenceForceSimulation,
            symbols,
            positions,
        )

        module = build_simulation(
            ValenceStateBatchedSimulation,
            symbols,
            positions,
        )

        scratch_energy = float(
            scratch.potential_per_box[0]
        )

        module_energy = float(
            module.potential_per_box[0]
        )

        difference = (
            module_energy - scratch_energy
        )

        candidate = (
            abs(difference),
            difference,
            geometry["geometry_id"],
            scratch_energy,
            module_energy,
        )

        if worst is None or candidate[0] > worst[0]:
            worst = candidate

    return worst


def promotion_force_check(payload):
    rows = dense_water(payload)

    results = []

    for x in FORCE_POINTS:
        geometry = interpolate_water(
            rows,
            x,
        )

        scratch = build_simulation(
            SmoothValenceForceSimulation,
            geometry["symbols"],
            geometry["positions"],
        )

        module = build_simulation(
            ValenceStateBatchedSimulation,
            geometry["symbols"],
            geometry["positions"],
        )

        scratch_energy, scratch_force = live_energy_force(
            scratch,
            geometry["positions"],
        )

        module_energy, module_force = live_energy_force(
            module,
            geometry["positions"],
        )

        results.append({
            "x": x,
            "energy_diff": module_energy - scratch_energy,
            "max_force_diff": float(
                np.max(
                    np.abs(
                        module_force - scratch_force
                    )
                )
            ),
            "rms_force_diff": float(
                np.sqrt(
                    np.mean(
                        (
                            module_force
                            - scratch_force
                        ) ** 2
                    )
                )
            ),
        })

    return results


def nve_systems(payload):
    water_reference = find_reference(
        payload,
        "water",
    )

    water_competition = find_dense_water_exact(
        payload,
        1.160,
    )

    peroxide_symbols, peroxide_positions = (
        hydrogen_peroxide_geometry(
            oo_distance=1.475
        )
    )

    return [
        {
            "name": "water_intact",
            "symbols": list(
                water_reference["symbols"]
            ),
            "positions": centre_in_box(
                water_reference[
                    "coordinates_angstrom"
                ]
            ),
        },
        {
            "name": "water_competition_x1.160",
            "symbols": list(
                water_competition["symbols"]
            ),
            "positions": centre_in_box(
                water_competition[
                    "coordinates_angstrom"
                ]
            ),
        },
        {
            "name": "peroxide_OO_1.475",
            "symbols": list(
                peroxide_symbols
            ),
            "positions": centre_in_box(
                peroxide_positions
            ),
        },
    ]


def promotion_nve_check(payload):
    results = []

    for system in nve_systems(payload):
        scratch = build_simulation(
            SmoothValenceForceSimulation,
            system["symbols"],
            system["positions"],
        )

        module = build_simulation(
            ValenceStateBatchedSimulation,
            system["symbols"],
            system["positions"],
        )

        # Force identical starting kinetic state.
        module.velocities = (
            scratch.velocities.detach().clone()
        )

        worst_total_energy_difference = 0.0
        worst_position_difference = 0.0
        worst_velocity_difference = 0.0
        worst_force_difference = 0.0

        for step in range(NVE_STEPS + 1):
            scratch_total = (
                scratch.potential_energy
                + scratch.kinetic_energy
            )

            module_total = (
                module.potential_energy
                + module.kinetic_energy
            )

            total_difference = abs(
                module_total
                - scratch_total
            )

            position_difference = float(
                torch.max(
                    torch.abs(
                        module.positions
                        - scratch.positions
                    )
                ).detach().cpu()
            )

            velocity_difference = float(
                torch.max(
                    torch.abs(
                        module.velocities
                        - scratch.velocities
                    )
                ).detach().cpu()
            )

            force_difference = float(
                torch.max(
                    torch.abs(
                        module.forces
                        - scratch.forces
                    )
                ).detach().cpu()
            )

            worst_total_energy_difference = max(
                worst_total_energy_difference,
                total_difference,
            )

            worst_position_difference = max(
                worst_position_difference,
                position_difference,
            )

            worst_velocity_difference = max(
                worst_velocity_difference,
                velocity_difference,
            )

            worst_force_difference = max(
                worst_force_difference,
                force_difference,
            )

            if step == NVE_STEPS:
                break

            scratch.step(1)
            module.step(1)

        results.append({
            "system": system["name"],
            "max_total_energy_diff": worst_total_energy_difference,
            "max_position_diff": worst_position_difference,
            "max_velocity_diff": worst_velocity_difference,
            "max_force_diff": worst_force_difference,
            "scratch_caps": int(
                scratch.capped_steps
            ),
            "module_caps": int(
                module.capped_steps
            ),
        })

    return results


def main():
    payload = load_payload()

    print("VALENCE-STATE MODULE PROMOTION VALIDATION")
    print()
    print(
        "scratch : "
        "probe_smooth_valence_forces.SmoothValenceForceSimulation"
    )
    print(
        "module  : "
        "valence_state_torch.ValenceStateBatchedSimulation"
    )
    print()

    print("1. ENERGY EQUIVALENCE — ALL QM GEOMETRIES")

    worst_energy = promotion_energy_check(
        payload
    )

    print(
        f"  geometries            : "
        f"{len(payload['geometries'])}"
    )

    print(
        f"  worst geometry        : "
        f"{worst_energy[2]}"
    )

    print(
        f"  scratch energy        : "
        f"{worst_energy[3]:+.12f} eV"
    )

    print(
        f"  module energy         : "
        f"{worst_energy[4]:+.12f} eV"
    )

    print(
        f"  max |difference|      : "
        f"{worst_energy[0]:.6e} eV"
    )

    print()

    print("2. FORCE EQUIVALENCE")

    force_results = promotion_force_check(
        payload
    )

    for row in force_results:
        print(
            f"  x={row['x']:.3f} A  "
            f"dE={row['energy_diff']:+.6e} eV  "
            f"max dF={row['max_force_diff']:.6e} eV/A  "
            f"RMS dF={row['rms_force_diff']:.6e} eV/A"
        )

    print()

    print("3. MATCHED NVE TRAJECTORY EQUIVALENCE")
    print(
        f"  steps per system      : {NVE_STEPS}"
    )

    nve_results = promotion_nve_check(
        payload
    )

    for row in nve_results:
        print(
            f"  {row['system']:<28s} "
            f"dEtot_max={row['max_total_energy_diff']:.6e} eV  "
            f"dpos_max={row['max_position_diff']:.6e} A  "
            f"dvel_max={row['max_velocity_diff']:.6e} A/fs  "
            f"dF_max={row['max_force_diff']:.6e} eV/A  "
            f"caps={row['scratch_caps']}/{row['module_caps']}"
        )

    max_force_difference = max(
        row["max_force_diff"]
        for row in force_results
    )

    max_nve_energy_difference = max(
        row["max_total_energy_diff"]
        for row in nve_results
    )

    max_nve_position_difference = max(
        row["max_position_diff"]
        for row in nve_results
    )

    max_nve_velocity_difference = max(
        row["max_velocity_diff"]
        for row in nve_results
    )

    passed = (
        worst_energy[0] <= ENERGY_TOL
        and max_force_difference <= FORCE_TOL
        and max_nve_energy_difference <= NVE_ENERGY_TOL
        and max_nve_position_difference <= NVE_POSITION_TOL
        and max_nve_velocity_difference <= NVE_VELOCITY_TOL
        and all(
            row["scratch_caps"]
            == row["module_caps"]
            for row in nve_results
        )
    )

    print()
    print("PROMOTION RESULT")

    if passed:
        print(
            "  PASS — promoted module reproduces the validated scratch "
            "physics within numerical tolerance."
        )
    else:
        print(
            "  FAIL — promotion changed the validated scratch behaviour."
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
