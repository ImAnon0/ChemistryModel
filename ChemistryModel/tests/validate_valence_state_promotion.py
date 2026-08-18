"""
Promotion-equivalence validation for valence_state_torch.py.

Compares the validated scratch prototype:
    probe_smooth_valence_forces.SmoothValenceForceSimulation

against the promoted experimental engine:
    valence_state_torch.ValenceStateBatchedSimulation

Checks:
    1. energy equivalence on all 106 QM microscope geometries
    2. force equivalence at representative water geometries
    3. shared-geometry dynamic equivalence along independently generated
       scratch and promoted-module NVE trajectories for:
         - intact water
         - water state-competition geometry
         - H2O2

The dynamic check deliberately does NOT require two independently integrated
trajectories to remain bit-for-bit identical. Tiny round-off-level force
differences can grow into trajectory separation in a competitive/chaotic
region even when the two implementations evaluate the same potential.

Instead, both implementations are re-evaluated on the exact same sampled
geometry. That tests the actual promotion invariant: same geometry -> same
energy and force.

If this passes, promotion from scratch prototype -> engine module did not
change the validated physics within numerical tolerance.

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
DYNAMIC_SAMPLE_INTERVAL = 25

ENERGY_TOL = 1.0e-10
FORCE_TOL = 1.0e-9
DYNAMIC_ENERGY_TOL = 1.0e-10
DYNAMIC_FORCE_TOL = 1.0e-9


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


def _evaluate_same_geometry(symbols, coordinates):
    """Evaluate scratch and promoted module on one identical geometry.

    Fresh evaluator instances are used deliberately. This prevents either
    driver's neighbour/cache history from influencing the comparison.
    """
    scratch = build_simulation(
        SmoothValenceForceSimulation,
        symbols,
        coordinates,
    )

    module = build_simulation(
        ValenceStateBatchedSimulation,
        symbols,
        coordinates,
    )

    scratch_energy, scratch_force = live_energy_force(
        scratch,
        coordinates,
    )

    module_energy, module_force = live_energy_force(
        module,
        coordinates,
    )

    force_difference = (
        module_force - scratch_force
    )

    return {
        "energy_diff": (
            module_energy - scratch_energy
        ),
        "max_force_diff": float(
            np.max(
                np.abs(
                    force_difference
                )
            )
        ),
        "rms_force_diff": float(
            np.sqrt(
                np.mean(
                    force_difference ** 2
                )
            )
        ),
    }


def promotion_dynamic_check(payload):
    """Compare both implementations on identical dynamically visited states.

    Each implementation is allowed to integrate its own NVE trajectory. At
    regular intervals we take that driver's current geometry and ask fresh
    scratch/module evaluators to compute energy and force on the SAME
    coordinates.

    This separates two questions that the old validator mixed together:

        implementation equivalence:
            same geometry -> same E/F

        trajectory identity:
            two independently integrated chaotic trajectories remain exactly
            coincident for hundreds of steps

    Only the first is a valid promotion requirement.
    """
    results = []

    for system in nve_systems(payload):
        scratch_driver = build_simulation(
            SmoothValenceForceSimulation,
            system["symbols"],
            system["positions"],
        )

        module_driver = build_simulation(
            ValenceStateBatchedSimulation,
            system["symbols"],
            system["positions"],
        )

        # Give both drivers the same initial kinetic state so the diagnostic
        # still begins from a matched physical condition. Their subsequent
        # coordinate separation is reported but is not a pass/fail criterion.
        module_driver.velocities = (
            scratch_driver.velocities.detach().clone()
        )

        worst_energy = {
            "abs": 0.0,
            "signed": 0.0,
            "driver": None,
            "step": 0,
        }

        worst_force = {
            "max": 0.0,
            "rms": 0.0,
            "driver": None,
            "step": 0,
        }

        samples = 0

        drivers = (
            ("scratch", scratch_driver),
            ("module", module_driver),
        )

        for step in range(NVE_STEPS + 1):
            if (
                step % DYNAMIC_SAMPLE_INTERVAL == 0
                or step == NVE_STEPS
            ):
                for driver_name, driver in drivers:
                    coordinates = (
                        driver.positions
                        .detach()
                        .cpu()
                        .numpy()
                        .copy()
                    )

                    comparison = _evaluate_same_geometry(
                        system["symbols"],
                        coordinates,
                    )

                    samples += 1

                    absolute_energy = abs(
                        comparison["energy_diff"]
                    )

                    if absolute_energy > worst_energy["abs"]:
                        worst_energy = {
                            "abs": absolute_energy,
                            "signed": comparison[
                                "energy_diff"
                            ],
                            "driver": driver_name,
                            "step": step,
                        }

                    if (
                        comparison["max_force_diff"]
                        > worst_force["max"]
                    ):
                        worst_force = {
                            "max": comparison[
                                "max_force_diff"
                            ],
                            "rms": comparison[
                                "rms_force_diff"
                            ],
                            "driver": driver_name,
                            "step": step,
                        }

            if step == NVE_STEPS:
                break

            scratch_driver.step(1)
            module_driver.step(1)

        final_position_difference = float(
            torch.max(
                torch.abs(
                    module_driver.positions
                    - scratch_driver.positions
                )
            ).detach().cpu()
        )

        final_velocity_difference = float(
            torch.max(
                torch.abs(
                    module_driver.velocities
                    - scratch_driver.velocities
                )
            ).detach().cpu()
        )

        results.append({
            "system": system["name"],
            "samples": samples,
            "max_same_geometry_energy_diff": worst_energy["abs"],
            "signed_energy_diff_at_worst": worst_energy["signed"],
            "energy_worst_driver": worst_energy["driver"],
            "energy_worst_step": worst_energy["step"],
            "max_same_geometry_force_diff": worst_force["max"],
            "rms_force_diff_at_worst": worst_force["rms"],
            "force_worst_driver": worst_force["driver"],
            "force_worst_step": worst_force["step"],
            "final_independent_position_diff": (
                final_position_difference
            ),
            "final_independent_velocity_diff": (
                final_velocity_difference
            ),
            "scratch_caps": int(
                scratch_driver.capped_steps
            ),
            "module_caps": int(
                module_driver.capped_steps
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

    print("3. SHARED-GEOMETRY DYNAMIC EQUIVALENCE")
    print(
        f"  trajectory steps      : {NVE_STEPS}"
    )
    print(
        f"  sample interval       : {DYNAMIC_SAMPLE_INTERVAL} steps"
    )
    print(
        "  criterion             : same geometry -> same energy/force"
    )
    print(
        "  trajectory separation : reported only; not a failure criterion"
    )

    dynamic_results = promotion_dynamic_check(
        payload
    )

    for row in dynamic_results:
        print(
            f"  {row['system']:<28s} "
            f"samples={row['samples']:>3d}  "
            f"dE_same_max={row['max_same_geometry_energy_diff']:.6e} eV  "
            f"dF_same_max={row['max_same_geometry_force_diff']:.6e} eV/A  "
            f"final dpos={row['final_independent_position_diff']:.6e} A  "
            f"caps={row['scratch_caps']}/{row['module_caps']}"
        )
        print(
            f"    worst dE: driver={row['energy_worst_driver']} "
            f"step={row['energy_worst_step']}  "
            f"signed={row['signed_energy_diff_at_worst']:+.6e} eV"
        )
        print(
            f"    worst dF: driver={row['force_worst_driver']} "
            f"step={row['force_worst_step']}  "
            f"RMS={row['rms_force_diff_at_worst']:.6e} eV/A"
        )

    max_force_difference = max(
        row["max_force_diff"]
        for row in force_results
    )

    max_dynamic_energy_difference = max(
        row["max_same_geometry_energy_diff"]
        for row in dynamic_results
    )

    max_dynamic_force_difference = max(
        row["max_same_geometry_force_diff"]
        for row in dynamic_results
    )

    passed = (
        worst_energy[0] <= ENERGY_TOL
        and max_force_difference <= FORCE_TOL
        and max_dynamic_energy_difference <= DYNAMIC_ENERGY_TOL
        and max_dynamic_force_difference <= DYNAMIC_FORCE_TOL
        and all(
            row["scratch_caps"]
            == row["module_caps"]
            for row in dynamic_results
        )
    )

    print()
    print("PROMOTION RESULT")

    if passed:
        print(
            "  PASS — promoted module reproduces the corrected scratch "
            "physics on static and dynamically visited shared geometries "
            "within numerical tolerance."
        )
    else:
        print(
            "  FAIL — promoted module differs from the corrected scratch "
            "implementation on at least one identical geometry."
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
