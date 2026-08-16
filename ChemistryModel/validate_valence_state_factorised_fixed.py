"""
Validate the factorisable-H heavy-valence integration.

Compares:

    historical:
        valence_state_torch.ValenceStateBatchedSimulation

    candidate:
        valence_state_factorised_torch.
        FactorisableValenceStateBatchedSimulation

The heavy-valence equations are intentionally unchanged.  On the existing
single-reaction/heavy-topology validation set, the two engines should therefore
match to floating-point precision.

The candidate must additionally inherit the corrected H-state behaviour:
    - disconnected H-transfer networks are exactly additive
    - the historical valence engine demonstrates the old whole-box error
    - a short live H-component merge matches the standalone factorisable
      H-state engine exactly

Checks
------
1. Energy equivalence on all 106 QM microscope geometries.
2. Force equivalence at the existing representative water points.
3. Matched NVE trajectory equivalence on:
       - intact water
       - water competition x=1.160 A
       - H2O2 O-O = 1.475 A
4. H-only size-consistency regression:
       - exact historical bug geometry: two distant symmetric H3 networks
         at 0.90 A spacing
       - unequal disconnected H3 control
5. Live H-only 2->1 component merge:
       factorisable valence vs standalone factorisable H-state.

This validates integration, not QM agreement itself.

Run:
    py validate_valence_state_factorised.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from bond_calibration import hydrogen_peroxide_geometry

from h_state_factorised_torch import (
    FactorisedHStateBatchedSimulation,
)

from valence_state_torch import (
    ValenceStateBatchedSimulation,
)

from valence_state_factorised_torch import (
    FactorisableValenceStateBatchedSimulation,
)


GEOMETRIES = Path(
    "research_data/qm_residual/dense_scan_geometries.json"
)

BOX_SIZE = 30.0
SEED = 913
TEMPERATURE_K = 100.0

FORCE_POINTS = (
    1.080,
    1.160,
    1.320,
    1.535,
)

NVE_STEPS = 250

ENERGY_TOL = 1.0e-10
FORCE_TOL = 1.0e-9
NVE_ENERGY_TOL = 1.0e-9
NVE_POSITION_TOL = 1.0e-10
NVE_VELOCITY_TOL = 1.0e-10
NVE_FORCE_TOL = 1.0e-9

SIZE_ENERGY_TOL = 1.0e-10
SIZE_FORCE_TOL = 1.0e-9

MERGE_DT_FS = 0.001
MERGE_STEPS = 200
MERGE_START_OUTSIDE_A = 0.0002
MERGE_RELATIVE_SPEED_A_PER_FS = 0.05

MERGE_ENERGY_MATCH_TOL = 1.0e-10
MERGE_POSITION_MATCH_TOL = 1.0e-10
MERGE_VELOCITY_MATCH_TOL = 1.0e-10
MERGE_FORCE_MATCH_TOL = 1.0e-9


def load_payload():
    return json.loads(
        GEOMETRIES.read_text(
            encoding="utf-8"
        )
    )


def centre_in_box(
    coordinates,
    box_size=BOX_SIZE,
):
    positions = np.asarray(
        coordinates,
        dtype=float,
    )

    return (
        positions
        - positions.mean(axis=0)
        + 0.5 * box_size
    )


def build(
    model_class,
    symbols,
    positions,
    *,
    box_size=BOX_SIZE,
    dt_fs=0.25,
    temperature=TEMPERATURE_K,
):
    simulation = model_class(
        boxes=[(
            list(symbols),
            np.asarray(
                positions,
                dtype=float,
            ),
        )],
        box_size=float(
            box_size
        ),
        time_step=float(
            dt_fs
        ),
        target_temperature=float(
            temperature
        ),
        friction=0.0,
        device="cpu",
        dtype=torch.float64,
        random_seed=SEED,
        relax_on_start=False,
    )

    simulation.thermostat_is_on = False

    return simulation


def live_energy_force(
    simulation,
    coordinates,
):
    positions = torch.tensor(
        np.asarray(
            coordinates,
            dtype=float,
        ),
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
        float(
            energy.detach().cpu()
        ),
        (
            -gradient
        ).detach().cpu().numpy(),
    )


def dense_water(payload):
    rows = [
        geometry
        for geometry in payload[
            "geometries"
        ]
        if (
            geometry["system"]
            == "water"
            and geometry["sample_kind"]
            == "dense_transfer_scan"
            and geometry.get(
                "reaction_coordinate",
                {},
            ).get(
                "transfer_distance_angstrom"
            )
            is not None
        )
    ]

    rows.sort(
        key=lambda geometry: float(
            geometry[
                "reaction_coordinate"
            ][
                "transfer_distance_angstrom"
            ]
        )
    )

    return rows


def interpolate_water(
    rows,
    x,
):
    for left, right in zip(
        rows,
        rows[1:],
    ):
        x0 = float(
            left[
                "reaction_coordinate"
            ][
                "transfer_distance_angstrom"
            ]
        )

        x1 = float(
            right[
                "reaction_coordinate"
            ][
                "transfer_distance_angstrom"
            ]
        )

        if (
            x0 - 1e-12
            <= x
            <= x1 + 1e-12
        ):
            fraction = (
                0.0
                if x1 == x0
                else (
                    (x - x0)
                    / (x1 - x0)
                )
            )

            p0 = np.asarray(
                left[
                    "coordinates_angstrom"
                ],
                dtype=float,
            )

            p1 = np.asarray(
                right[
                    "coordinates_angstrom"
                ],
                dtype=float,
            )

            return {
                "symbols": list(
                    left["symbols"]
                ),
                "positions": (
                    p0
                    + fraction
                    * (p1 - p0)
                ),
            }

    raise ValueError(
        f"x={x:.6f} outside dense water path"
    )


def find_reference(
    payload,
    system,
):
    matches = [
        geometry
        for geometry in payload[
            "geometries"
        ]
        if (
            geometry["system"]
            == system
            and geometry[
                "sample_kind"
            ]
            == "reactant_reference"
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {system} reactant reference, "
            f"got {len(matches)}"
        )

    return matches[0]


def find_dense_water_exact(
    payload,
    x,
):
    matches = [
        geometry
        for geometry in payload[
            "geometries"
        ]
        if (
            geometry["system"]
            == "water"
            and geometry[
                "sample_kind"
            ]
            == "dense_transfer_scan"
            and geometry.get(
                "reaction_coordinate",
                {},
            ).get(
                "transfer_distance_angstrom"
            )
            is not None
            and abs(
                float(
                    geometry[
                        "reaction_coordinate"
                    ][
                        "transfer_distance_angstrom"
                    ]
                )
                - x
            )
            < 1e-9
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one water geometry at x={x}, "
            f"got {len(matches)}"
        )

    return matches[0]


# ----------------------------------------------------------------------
# 1. Existing 106-geometry equivalence
# ----------------------------------------------------------------------

def energy_equivalence(
    payload,
):
    worst = None

    for geometry in payload[
        "geometries"
    ]:
        symbols = geometry[
            "symbols"
        ]

        positions = geometry[
            "coordinates_angstrom"
        ]

        historical = build(
            ValenceStateBatchedSimulation,
            symbols,
            positions,
        )

        candidate = build(
            FactorisableValenceStateBatchedSimulation,
            symbols,
            positions,
        )

        old_energy = float(
            historical.potential_per_box[
                0
            ]
        )

        new_energy = float(
            candidate.potential_per_box[
                0
            ]
        )

        difference = (
            new_energy
            - old_energy
        )

        row = (
            abs(difference),
            difference,
            geometry[
                "geometry_id"
            ],
            old_energy,
            new_energy,
        )

        if (
            worst is None
            or row[0] > worst[0]
        ):
            worst = row

    return worst


# ----------------------------------------------------------------------
# 2. Water force equivalence
# ----------------------------------------------------------------------

def force_equivalence(
    payload,
):
    water = dense_water(
        payload
    )

    rows = []

    for x in FORCE_POINTS:
        geometry = (
            interpolate_water(
                water,
                x,
            )
        )

        historical = build(
            ValenceStateBatchedSimulation,
            geometry["symbols"],
            geometry["positions"],
        )

        candidate = build(
            FactorisableValenceStateBatchedSimulation,
            geometry["symbols"],
            geometry["positions"],
        )

        old_energy, old_force = (
            live_energy_force(
                historical,
                geometry["positions"],
            )
        )

        new_energy, new_force = (
            live_energy_force(
                candidate,
                geometry["positions"],
            )
        )

        force_difference = (
            new_force - old_force
        )

        rows.append({
            "x": x,
            "energy_difference_eV": (
                new_energy
                - old_energy
            ),
            "max_force_difference_eV_per_A": float(
                np.max(
                    np.abs(
                        force_difference
                    )
                )
            ),
            "rms_force_difference_eV_per_A": float(
                np.sqrt(
                    np.mean(
                        force_difference
                        * force_difference
                    )
                )
            ),
        })

    return rows


# ----------------------------------------------------------------------
# 3. Existing NVE control equivalence
# ----------------------------------------------------------------------

def nve_systems(
    payload,
):
    water_reference = (
        find_reference(
            payload,
            "water",
        )
    )

    water_competition = (
        find_dense_water_exact(
            payload,
            1.160,
        )
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
                water_reference[
                    "symbols"
                ]
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
                water_competition[
                    "symbols"
                ]
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


def matched_nve_equivalence(
    payload,
):
    results = []

    for system in nve_systems(
        payload
    ):
        historical = build(
            ValenceStateBatchedSimulation,
            system["symbols"],
            system["positions"],
        )

        candidate = build(
            FactorisableValenceStateBatchedSimulation,
            system["symbols"],
            system["positions"],
        )

        candidate.velocities = (
            historical.velocities
            .detach()
            .clone()
        )

        worst_energy = 0.0
        worst_position = 0.0
        worst_velocity = 0.0
        worst_force = 0.0

        for step in range(
            NVE_STEPS + 1
        ):
            old_total = (
                historical.potential_energy
                + historical.kinetic_energy
            )

            new_total = (
                candidate.potential_energy
                + candidate.kinetic_energy
            )

            worst_energy = max(
                worst_energy,
                abs(
                    new_total
                    - old_total
                ),
            )

            worst_position = max(
                worst_position,
                float(
                    torch.max(
                        torch.abs(
                            candidate.positions
                            - historical.positions
                        )
                    )
                    .detach()
                    .cpu()
                ),
            )

            worst_velocity = max(
                worst_velocity,
                float(
                    torch.max(
                        torch.abs(
                            candidate.velocities
                            - historical.velocities
                        )
                    )
                    .detach()
                    .cpu()
                ),
            )

            worst_force = max(
                worst_force,
                float(
                    torch.max(
                        torch.abs(
                            candidate.forces
                            - historical.forces
                        )
                    )
                    .detach()
                    .cpu()
                ),
            )

            if step == NVE_STEPS:
                break

            historical.step(1)
            candidate.step(1)

        results.append({
            "system": system[
                "name"
            ],
            "max_total_energy_difference_eV": (
                worst_energy
            ),
            "max_position_difference_A": (
                worst_position
            ),
            "max_velocity_difference_A_per_fs": (
                worst_velocity
            ),
            "max_force_difference_eV_per_A": (
                worst_force
            ),
            "old_caps": int(
                historical.capped_steps
            ),
            "new_caps": int(
                candidate.capped_steps
            ),
        })

    return results


# ----------------------------------------------------------------------
# 4. H-only size consistency
# ----------------------------------------------------------------------

def hh_cutoffs():
    import reactive as R

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
    inner, outer = (
        hh_cutoffs()
    )

    span = (
        outer - inner
    )

    first = max(
        inner
        + 0.25 * span,
        0.55 * outer,
    )

    second = max(
        inner
        + 0.62 * span,
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
        [
            2.0 * spacing,
            0.0,
            0.0,
        ],
    ]) + origin


def evaluate_static(
    model_class,
    symbols,
    positions,
    *,
    box_size=40.0,
):
    simulation = build(
        model_class,
        symbols,
        positions,
        box_size=box_size,
        temperature=0.0,
    )

    energy = float(
        simulation.potential_per_box[
            0
        ]
    )

    force = (
        simulation.forces
        .detach()
        .cpu()
        .numpy()
        .reshape(
            simulation.per_box,
            3,
        )
    )

    return energy, force


def size_consistency_case(
    model_class,
    spacing_a,
    spacing_b,
):
    positions_a = h3_positions(
        spacing_a,
        [8.0, 8.0, 8.0],
    )

    positions_b = h3_positions(
        spacing_b,
        [8.0, 23.0, 8.0],
    )

    e_a, f_a = evaluate_static(
        model_class,
        ["H"] * 3,
        positions_a,
    )

    e_b, f_b = evaluate_static(
        model_class,
        ["H"] * 3,
        positions_b,
    )

    e_ab, f_ab = evaluate_static(
        model_class,
        ["H"] * 6,
        np.vstack([
            positions_a,
            positions_b,
        ]),
    )

    return {
        "spacing_a": float(spacing_a),
        "spacing_b": float(spacing_b),
        "energy_nonadditivity_eV": (
            e_ab - e_a - e_b
        ),
        "force_error_a_eV_per_A": float(
            np.max(
                np.abs(
                    f_ab[:3]
                    - f_a
                )
            )
        ),
        "force_error_b_eV_per_A": float(
            np.max(
                np.abs(
                    f_ab[3:]
                    - f_b
                )
            )
        ),
    }


def known_historical_bug_case(
    model_class,
):
    # Exact geometry from diagnose_h_state_separability.py, which originally
    # exposed the whole-box H-state size-consistency bug:
    #
    #     H --0.90 A-- H --0.90 A-- H
    #
    # duplicated 15 A away in the same box.
    return size_consistency_case(
        model_class,
        0.90,
        0.90,
    )


def unequal_size_consistency_case(
    model_class,
):
    spacing_a, spacing_b = (
        choose_h3_spacings()
    )

    return size_consistency_case(
        model_class,
        spacing_a,
        spacing_b,
    )


# ----------------------------------------------------------------------
# 5. Live H-only merge: wrapper vs standalone factorisable H-state
# ----------------------------------------------------------------------

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

    positions = np.vstack([
        a,
        b,
    ])

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

    return positions, velocities


def live_merge_equivalence():
    positions, velocities = (
        merge_initial_state()
    )

    standalone = build(
        FactorisedHStateBatchedSimulation,
        ["H"] * 6,
        positions,
        box_size=40.0,
        dt_fs=MERGE_DT_FS,
        temperature=0.0,
    )

    wrapped = build(
        FactorisableValenceStateBatchedSimulation,
        ["H"] * 6,
        positions,
        box_size=40.0,
        dt_fs=MERGE_DT_FS,
        temperature=0.0,
    )

    velocity_tensor = torch.tensor(
        velocities,
        dtype=torch.float64,
        device="cpu",
    )

    standalone.velocities = (
        velocity_tensor.clone()
    )

    wrapped.velocities = (
        velocity_tensor.clone()
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
        old_total = (
            standalone.potential_energy
            + standalone.kinetic_energy
        )

        new_total = (
            wrapped.potential_energy
            + wrapped.kinetic_energy
        )

        max_energy = max(
            max_energy,
            abs(
                new_total
                - old_total
            ),
        )

        max_position = max(
            max_position,
            float(
                torch.max(
                    torch.abs(
                        wrapped.positions
                        - standalone.positions
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
                        wrapped.velocities
                        - standalone.velocities
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
                        wrapped.forces
                        - standalone.forces
                    )
                )
                .detach()
                .cpu()
            ),
        )

        diagnostics = getattr(
            wrapped,
            "_h_component_diagnostics",
            None,
        )

        if diagnostics:
            values = diagnostics.get(
                "component_counts_per_box",
                (),
            )

            count = (
                int(values[0])
                if values
                else 0
            )

            counts_seen.add(
                count
            )

            if (
                previous_count
                is not None
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

        standalone.step()
        wrapped.step()

    return {
        "max_total_energy_difference_eV": (
            max_energy
        ),
        "max_position_difference_A": (
            max_position
        ),
        "max_velocity_difference_A_per_fs": (
            max_velocity
        ),
        "max_force_difference_eV_per_A": (
            max_force
        ),
        "component_counts_seen": tuple(
            sorted(
                counts_seen
            )
        ),
        "transitions": tuple(
            transitions
        ),
        "standalone_caps": int(
            standalone.capped_steps
        ),
        "wrapped_caps": int(
            wrapped.capped_steps
        ),
    }


def main():
    payload = (
        load_payload()
    )

    print(
        "FACTORISABLE-H VALENCE-STATE INTEGRATION VALIDATION"
    )

    print()

    print(
        "historical : "
        "valence_state_torch.ValenceStateBatchedSimulation"
    )

    print(
        "candidate  : "
        "valence_state_factorised_torch."
        "FactorisableValenceStateBatchedSimulation"
    )

    print()

    print(
        "1. ENERGY EQUIVALENCE - 106 QM MICROSCOPE GEOMETRIES"
    )

    worst = energy_equivalence(
        payload
    )

    print(
        f"  geometries       : "
        f"{len(payload['geometries'])}"
    )

    print(
        f"  worst geometry   : "
        f"{worst[2]}"
    )

    print(
        f"  old energy       : "
        f"{worst[3]:+.12f} eV"
    )

    print(
        f"  new energy       : "
        f"{worst[4]:+.12f} eV"
    )

    print(
        f"  max |difference| : "
        f"{worst[0]:.12e} eV"
    )

    print()

    print(
        "2. FORCE EQUIVALENCE - WATER CONTROL POINTS"
    )

    force_rows = (
        force_equivalence(
            payload
        )
    )

    for row in force_rows:
        print(
            f"  x={row['x']:.3f} A  "
            f"dE={row['energy_difference_eV']:+.3e} eV  "
            f"max|dF|="
            f"{row['max_force_difference_eV_per_A']:.3e} eV/A  "
            f"RMS="
            f"{row['rms_force_difference_eV_per_A']:.3e}"
        )

    print()

    print(
        "3. MATCHED NVE EQUIVALENCE - EXISTING HEAVY-VALENCE CONTROLS"
    )

    nve_rows = (
        matched_nve_equivalence(
            payload
        )
    )

    for row in nve_rows:
        print(
            f"  {row['system']:<28s} "
            f"dEtot={row['max_total_energy_difference_eV']:.3e}  "
            f"dpos={row['max_position_difference_A']:.3e}  "
            f"dvel={row['max_velocity_difference_A_per_fs']:.3e}  "
            f"dF={row['max_force_difference_eV_per_A']:.3e}  "
            f"caps={row['old_caps']}/{row['new_caps']}"
        )

    print()

    print(
        "4. H-ONLY SIZE CONSISTENCY REGRESSION"
    )

    old_known = (
        known_historical_bug_case(
            ValenceStateBatchedSimulation
        )
    )

    new_known = (
        known_historical_bug_case(
            FactorisableValenceStateBatchedSimulation
        )
    )

    new_unequal = (
        unequal_size_consistency_case(
            FactorisableValenceStateBatchedSimulation
        )
    )

    print(
        "  known historical bug geometry "
        "(two identical H3, spacing 0.900 A):"
    )

    print(
        "    historical valence:"
    )

    print(
        f"      E(A+B)-E(A)-E(B) : "
        f"{old_known['energy_nonadditivity_eV']:+.12e} eV"
    )

    print(
        f"      max |dF| A/B     : "
        f"{old_known['force_error_a_eV_per_A']:.3e} / "
        f"{old_known['force_error_b_eV_per_A']:.3e} eV/A"
    )

    print(
        "    factorisable valence:"
    )

    print(
        f"      E(A+B)-E(A)-E(B) : "
        f"{new_known['energy_nonadditivity_eV']:+.12e} eV"
    )

    print(
        f"      max |dF| A/B     : "
        f"{new_known['force_error_a_eV_per_A']:.3e} / "
        f"{new_known['force_error_b_eV_per_A']:.3e} eV/A"
    )

    print(
        "  unequal disconnected control:"
    )

    print(
        f"    spacings            : "
        f"{new_unequal['spacing_a']:.9f} / "
        f"{new_unequal['spacing_b']:.9f} A"
    )

    print(
        f"    factorisable E nonadditivity : "
        f"{new_unequal['energy_nonadditivity_eV']:+.12e} eV"
    )

    print(
        f"    factorisable max |dF| A/B    : "
        f"{new_unequal['force_error_a_eV_per_A']:.3e} / "
        f"{new_unequal['force_error_b_eV_per_A']:.3e} eV/A"
    )

    print()

    print(
        "5. LIVE H-COMPONENT MERGE - WRAPPER VS STANDALONE FACTORISABLE H"
    )

    merge = (
        live_merge_equivalence()
    )

    print(
        f"  component counts seen : "
        f"{merge['component_counts_seen']}"
    )

    print(
        f"  transitions           : "
        f"{merge['transitions']}"
    )

    print(
        f"  max |dEtot|           : "
        f"{merge['max_total_energy_difference_eV']:.12e} eV"
    )

    print(
        f"  max |dPosition|       : "
        f"{merge['max_position_difference_A']:.12e} A"
    )

    print(
        f"  max |dVelocity|       : "
        f"{merge['max_velocity_difference_A_per_fs']:.12e} A/fs"
    )

    print(
        f"  max |dForce|          : "
        f"{merge['max_force_difference_eV_per_A']:.12e} eV/A"
    )

    print(
        f"  caps standalone/wrapped : "
        f"{merge['standalone_caps']}/"
        f"{merge['wrapped_caps']}"
    )

    max_force_difference = max(
        row[
            "max_force_difference_eV_per_A"
        ]
        for row in force_rows
    )

    max_nve_energy = max(
        row[
            "max_total_energy_difference_eV"
        ]
        for row in nve_rows
    )

    max_nve_position = max(
        row[
            "max_position_difference_A"
        ]
        for row in nve_rows
    )

    max_nve_velocity = max(
        row[
            "max_velocity_difference_A_per_fs"
        ]
        for row in nve_rows
    )

    max_nve_force = max(
        row[
            "max_force_difference_eV_per_A"
        ]
        for row in nve_rows
    )

    existing_equivalence_pass = (
        worst[0]
        <= ENERGY_TOL
        and max_force_difference
        <= FORCE_TOL
        and max_nve_energy
        <= NVE_ENERGY_TOL
        and max_nve_position
        <= NVE_POSITION_TOL
        and max_nve_velocity
        <= NVE_VELOCITY_TOL
        and max_nve_force
        <= NVE_FORCE_TOL
        and all(
            row["old_caps"]
            == row["new_caps"]
            for row in nve_rows
        )
    )

    size_pass = all(
        (
            abs(
                result[
                    "energy_nonadditivity_eV"
                ]
            )
            <= SIZE_ENERGY_TOL
            and result[
                "force_error_a_eV_per_A"
            ]
            <= SIZE_FORCE_TOL
            and result[
                "force_error_b_eV_per_A"
            ]
            <= SIZE_FORCE_TOL
        )
        for result in (
            new_known,
            new_unequal,
        )
    )

    # Reproduce the exact symmetric 0.90 A geometry that originally exposed
    # the historical whole-box H-state size-consistency bug.
    old_still_exposes_bug = (
        abs(
            old_known[
                "energy_nonadditivity_eV"
            ]
        )
        > 1.0e-6
    )

    merge_pass = (
        1
        in merge[
            "component_counts_seen"
        ]
        and 2
        in merge[
            "component_counts_seen"
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
            "max_total_energy_difference_eV"
        ]
        <= MERGE_ENERGY_MATCH_TOL
        and merge[
            "max_position_difference_A"
        ]
        <= MERGE_POSITION_MATCH_TOL
        and merge[
            "max_velocity_difference_A_per_fs"
        ]
        <= MERGE_VELOCITY_MATCH_TOL
        and merge[
            "max_force_difference_eV_per_A"
        ]
        <= MERGE_FORCE_MATCH_TOL
        and merge[
            "standalone_caps"
        ]
        == merge[
            "wrapped_caps"
        ]
    )

    print()
    print(
        "FINAL"
    )

    print(
        "  existing heavy-valence behaviour preserved : "
        + (
            "PASS"
            if existing_equivalence_pass
            else "FAIL"
        )
    )

    print(
        "  candidate H size consistency               : "
        + (
            "PASS"
            if size_pass
            else "FAIL"
        )
    )

    print(
        "  historical size-consistency bug reproduced : "
        + (
            "PASS"
            if old_still_exposes_bug
            else "FAIL"
        )
    )

    print(
        "  live factorisable H merge survives wrapper : "
        + (
            "PASS"
            if merge_pass
            else "FAIL"
        )
    )

    if (
        existing_equivalence_pass
        and size_pass
        and old_still_exposes_bug
        and merge_pass
    ):
        print()
        print(
            "  PASS - the corrected factorisable H-state is integrated "
            "under the heavy-valence engine without changing the existing "
            "validated heavy-valence behaviour."
        )

        print(
            "  Next: run the broader valence QM/force/NVE checks if desired, "
            "then return to component-batched performance engineering."
        )

        return

    print()
    print(
        "  FAIL - do not promote this integration."
    )

    raise SystemExit(1)


if __name__ == "__main__":
    main()
