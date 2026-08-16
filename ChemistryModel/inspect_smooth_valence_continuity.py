"""
Dense continuity microscope for valence-state topology.

Purpose
-------
Interpolate very finely along the existing dense water transfer path and
compare:

    ordinary H-state
    hard top-V topology ablation
    smooth valence-state topology ablation

The goal is NOT yet force validation. It is to detect hidden energy or
membership discontinuities before wiring the smooth formulation into
autograd/production physics.

This script:
  - uses only existing dense QM-reference geometries
  - linearly interpolates between adjacent 0.02 A water scan points
  - samples at 0.001 A transfer-distance spacing
  - records energies
  - records O->O and strongest O->H memberships
  - reports worst energy first-difference and second-difference
  - reports worst membership jump
  - compares hard vs smooth switching behaviour

Run:
    py inspect_smooth_valence_continuity.py

Output:
    research_data/qm_residual/smooth_valence_continuity.csv
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

import reactive as R
import ablate_hard_valence_topology as HARD
import ablate_smooth_valence_topology as SMOOTH


GEOMETRIES = Path("research_data/qm_residual/dense_scan_geometries.json")
OUTPUT = Path("research_data/qm_residual/smooth_valence_continuity.csv")

STEP_A = 0.001


def dense_water_geometries(payload):
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
            geometry["reaction_coordinate"]["transfer_distance_angstrom"]
        )
    )

    if len(rows) < 2:
        raise RuntimeError("Need at least two dense water geometries")

    return rows


def evaluate_hard(geometry):
    simulation = HARD.CaptureHState(
        boxes=[(
            geometry["symbols"],
            geometry["coordinates_angstrom"],
        )],
        box_size=HARD.BOX_SIZE,
        target_temperature=0.0,
        friction=0.0,
        device="cpu",
        dtype=torch.float64,
        random_seed=0,
        relax_on_start=False,
    )

    normal = float(simulation.potential_per_box[0])

    parts = HARD.hard_topology_counterfactual(simulation)

    hard = (
        normal
        + parts["over_delta_eV"]
        + parts["angle_delta_eV"]
    )

    return simulation, normal, hard, parts


def evaluate_smooth(geometry):
    return SMOOTH.evaluate_geometry(geometry)


def pair_membership(
    symbols,
    counterfactual,
    centre_atom,
    neighbour_atom,
):
    for entry in counterfactual["selections"][centre_atom]:
        if entry["neighbour"] == neighbour_atom:
            return float(entry.get("membership", 1.0))

    return 0.0


def pair_taper(simulation, centre_atom, neighbour_atom):
    values = simulation._hard_valence_values
    neighbours = values["neighbours"][centre_atom]
    mask = values["mask"][centre_atom]

    for slot in range(len(neighbours)):
        if (
            bool(mask[slot])
            and int(neighbours[slot]) == neighbour_atom
        ):
            return float(values["taper"][centre_atom, slot])

    return 0.0


def oxygen_indices(symbols):
    return [
        index
        for index, symbol in enumerate(symbols)
        if symbol == "O"
    ]


def hydrogen_indices(symbols):
    return [
        index
        for index, symbol in enumerate(symbols)
        if symbol == "H"
    ]


def strongest_h_memberships(symbols, counterfactual, oxygen):
    hydrogen = set(hydrogen_indices(symbols))

    values = []

    for entry in counterfactual["selections"][oxygen]:
        neighbour = entry["neighbour"]

        if neighbour not in hydrogen:
            continue

        values.append(
            float(entry.get("membership", 1.0))
        )

    values.sort(reverse=True)

    while len(values) < 2:
        values.append(0.0)

    return values[0], values[1]


def interpolate_geometry(left, right, x):
    x0 = float(
        left["reaction_coordinate"]["transfer_distance_angstrom"]
    )

    x1 = float(
        right["reaction_coordinate"]["transfer_distance_angstrom"]
    )

    if not (x0 - 1e-12 <= x <= x1 + 1e-12):
        raise ValueError("Interpolation point outside segment")

    fraction = (x - x0) / (x1 - x0)

    left_positions = np.asarray(
        left["coordinates_angstrom"],
        dtype=float,
    )

    right_positions = np.asarray(
        right["coordinates_angstrom"],
        dtype=float,
    )

    if left["symbols"] != right["symbols"]:
        raise RuntimeError("Water dense scan atom ordering changed")

    positions = (
        left_positions
        + fraction * (right_positions - left_positions)
    )

    return {
        "geometry_id": f"interp_{x:.6f}",
        "system": "water",
        "sample_kind": "continuity_interpolation",
        "region": "continuity",
        "symbols": list(left["symbols"]),
        "coordinates_angstrom": positions.tolist(),
        "reaction_coordinate": {
            "transfer_distance_angstrom": float(x),
        },
    }


def sample_path(dense):
    samples = []

    for segment_index, (left, right) in enumerate(
        zip(dense, dense[1:])
    ):
        x0 = float(
            left["reaction_coordinate"]["transfer_distance_angstrom"]
        )

        x1 = float(
            right["reaction_coordinate"]["transfer_distance_angstrom"]
        )

        count = int(round((x1 - x0) / STEP_A))

        for local in range(count + 1):
            if segment_index > 0 and local == 0:
                continue

            x = x0 + local * STEP_A

            if x > x1:
                x = x1

            geometry = interpolate_geometry(left, right, x)

            hard_sim, normal, hard_energy, hard_parts = evaluate_hard(
                geometry
            )

            smooth_sim, _, smooth_energy, smooth_parts = evaluate_smooth(
                geometry
            )

            symbols = geometry["symbols"]
            oxygens = oxygen_indices(symbols)

            if len(oxygens) != 2:
                raise RuntimeError(
                    f"Expected two oxygens, got {oxygens}"
                )

            o0, o1 = oxygens

            hard_oo_01 = pair_membership(
                symbols,
                hard_parts,
                o0,
                o1,
            )

            hard_oo_10 = pair_membership(
                symbols,
                hard_parts,
                o1,
                o0,
            )

            smooth_oo_01 = pair_membership(
                symbols,
                smooth_parts,
                o0,
                o1,
            )

            smooth_oo_10 = pair_membership(
                symbols,
                smooth_parts,
                o1,
                o0,
            )

            hard_h0a, hard_h0b = strongest_h_memberships(
                symbols,
                hard_parts,
                o0,
            )

            hard_h1a, hard_h1b = strongest_h_memberships(
                symbols,
                hard_parts,
                o1,
            )

            smooth_h0a, smooth_h0b = strongest_h_memberships(
                symbols,
                smooth_parts,
                o0,
            )

            smooth_h1a, smooth_h1b = strongest_h_memberships(
                symbols,
                smooth_parts,
                o1,
            )

            samples.append({
                "x_A": x,
                "hstate_energy_eV": normal,
                "hard_energy_eV": hard_energy,
                "smooth_energy_eV": smooth_energy,
                "oo_taper_01": pair_taper(
                    smooth_sim,
                    o0,
                    o1,
                ),
                "oo_taper_10": pair_taper(
                    smooth_sim,
                    o1,
                    o0,
                ),
                "hard_oo_membership_01": hard_oo_01,
                "hard_oo_membership_10": hard_oo_10,
                "smooth_oo_membership_01": smooth_oo_01,
                "smooth_oo_membership_10": smooth_oo_10,
                "hard_o0_h1": hard_h0a,
                "hard_o0_h2": hard_h0b,
                "hard_o1_h1": hard_h1a,
                "hard_o1_h2": hard_h1b,
                "smooth_o0_h1": smooth_h0a,
                "smooth_o0_h2": smooth_h0b,
                "smooth_o1_h1": smooth_h1a,
                "smooth_o1_h2": smooth_h1b,
                "hard_over_delta_eV": hard_parts["over_delta_eV"],
                "hard_angle_delta_eV": hard_parts["angle_delta_eV"],
                "smooth_over_delta_eV": smooth_parts["over_delta_eV"],
                "smooth_angle_delta_eV": smooth_parts["angle_delta_eV"],
            })

    return samples


def first_differences(samples, column):
    values = []

    for left, right in zip(samples, samples[1:]):
        dx = right["x_A"] - left["x_A"]

        values.append({
            "x0": left["x_A"],
            "x1": right["x_A"],
            "delta": right[column] - left[column],
            "slope": (
                (right[column] - left[column]) / dx
                if dx != 0.0
                else float("nan")
            ),
        })

    return values


def second_differences(samples, column):
    values = []

    for left, middle, right in zip(
        samples,
        samples[1:],
        samples[2:],
    ):
        dx1 = middle["x_A"] - left["x_A"]
        dx2 = right["x_A"] - middle["x_A"]

        if abs(dx1 - dx2) > 1e-8:
            continue

        second = (
            right[column]
            - 2.0 * middle[column]
            + left[column]
        )

        values.append({
            "x": middle["x_A"],
            "second": second,
            "curvature_like": second / (dx1 * dx1),
        })

    return values


def worst_abs(rows, key):
    if not rows:
        return None

    return max(rows, key=lambda row: abs(row[key]))


def print_energy_audit(samples, column, label):
    first = first_differences(samples, column)
    second = second_differences(samples, column)

    worst_step = worst_abs(first, "delta")
    worst_slope = worst_abs(first, "slope")
    worst_second = worst_abs(second, "second")

    print(f"{label}")

    if worst_step is not None:
        print(
            f"  worst 0.001 A energy step : "
            f"{worst_step['x0']:.3f}->{worst_step['x1']:.3f} "
            f"{worst_step['delta']:+.8f} eV"
        )

    if worst_slope is not None:
        print(
            f"  largest |dE/dx| estimate  : "
            f"{worst_slope['x0']:.3f}->{worst_slope['x1']:.3f} "
            f"{worst_slope['slope']:+.6f} eV/A"
        )

    if worst_second is not None:
        print(
            f"  worst second difference   : "
            f"x={worst_second['x']:.3f} "
            f"{worst_second['second']:+.10f} eV"
        )


def print_membership_audit(samples, column, label):
    diffs = first_differences(samples, column)
    worst = worst_abs(diffs, "delta")

    values = [row[column] for row in samples]

    print(
        f"{label:<27s} "
        f"range={min(values):.6f}..{max(values):.6f}"
    )

    if worst is not None:
        print(
            f"  worst 0.001 A jump        : "
            f"{worst['x0']:.3f}->{worst['x1']:.3f} "
            f"{worst['delta']:+.8f}"
        )


def main():
    payload = json.loads(
        GEOMETRIES.read_text(encoding="utf-8")
    )

    dense = dense_water_geometries(payload)

    x_min = float(
        dense[0]["reaction_coordinate"]["transfer_distance_angstrom"]
    )

    x_max = float(
        dense[-1]["reaction_coordinate"]["transfer_distance_angstrom"]
    )

    print("SMOOTH VALENCE-STATE CONTINUITY MICROSCOPE")
    print()
    print(
        f"path               : dense water interpolation "
        f"{x_min:.3f} -> {x_max:.3f} A"
    )
    print(f"sample spacing     : {STEP_A:.3f} A")
    print("production forces  : NOT tested here")
    print()

    samples = sample_path(dense)

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
            fieldnames=list(samples[0].keys()),
        )
        writer.writeheader()
        writer.writerows(samples)

    print(f"samples            : {len(samples)}")
    print(f"wrote              : {OUTPUT}")
    print()

    print("ENERGY CONTINUITY")
    print_energy_audit(
        samples,
        "hstate_energy_eV",
        "H-state",
    )
    print_energy_audit(
        samples,
        "hard_energy_eV",
        "hard top-V",
    )
    print_energy_audit(
        samples,
        "smooth_energy_eV",
        "smooth states",
    )

    print()
    print("O-O MEMBERSHIP CONTINUITY")

    for column, label in (
        (
            "hard_oo_membership_01",
            "hard O0->O1",
        ),
        (
            "hard_oo_membership_10",
            "hard O1->O0",
        ),
        (
            "smooth_oo_membership_01",
            "smooth O0->O1",
        ),
        (
            "smooth_oo_membership_10",
            "smooth O1->O0",
        ),
    ):
        print_membership_audit(
            samples,
            column,
            label,
        )

    print()
    print("SMOOTH O-H MEMBERSHIP CONTINUITY")

    for column, label in (
        ("smooth_o0_h1", "smooth O0 strongest H"),
        ("smooth_o0_h2", "smooth O0 second H"),
        ("smooth_o1_h1", "smooth O1 strongest H"),
        ("smooth_o1_h2", "smooth O1 second H"),
    ):
        print_membership_audit(
            samples,
            column,
            label,
        )

    print()
    print("LOCAL AUDIT AROUND 1.08 A")

    local = [
        row
        for row in samples
        if 1.075 <= row["x_A"] <= 1.085
    ]

    for row in local:
        print(
            f"x={row['x_A']:.3f}  "
            f"H={row['hstate_energy_eV']:+.6f}  "
            f"hard={row['hard_energy_eV']:+.6f}  "
            f"smooth={row['smooth_energy_eV']:+.6f}  "
            f"OOhard=({row['hard_oo_membership_01']:.3f},"
            f"{row['hard_oo_membership_10']:.3f})  "
            f"OOsmooth=({row['smooth_oo_membership_01']:.5f},"
            f"{row['smooth_oo_membership_10']:.5f})"
        )


if __name__ == "__main__":
    main()
