"""
Bond-dissociation topology-occupancy microscope.

Purpose
-------
Measure how the existing Morse-overlap signal

    s = exp[-a (r - r0)]

changes as genuine bonds are stretched from their trusted reference geometry
toward the live pair cutoff, and compare that with the known nonbonded O...O
encounter contacts from the dense water QM scan.

This is diagnostic only. It does not modify ChemistryModel physics.

Run:
    py inspect_occupancy_dissociation.py

Output:
    research_data/qm_residual/occupancy_dissociation_scan.csv
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

import reactive as R
from bond_calibration import hydrogen_peroxide_geometry
from h_state_torch import HStateReferenceBatchedSimulation


GEOMETRIES = Path("research_data/qm_residual/dense_scan_geometries.json")
OUTPUT = Path("research_data/qm_residual/occupancy_dissociation_scan.csv")
BOX_SIZE = 30.0

# Sample the genuine bond from equilibrium through progressive extension.
# The last point is added separately at the live outer cutoff.
STRETCH_RATIOS = (
    1.00,
    1.05,
    1.10,
    1.15,
    1.20,
    1.25,
    1.30,
    1.40,
    1.50,
)


class CaptureHState(HStateReferenceBatchedSimulation):
    def _hydrogen_state_correction(self, positions, base_per_atom):
        cached = getattr(self, "_reactive_intermediates", None)

        if cached is None or cached[0] is not positions:
            raise RuntimeError("Missing reactive intermediates")

        values = cached[1]

        self._occupancy_values = {
            key: value.detach().cpu().clone()
            for key, value in values.items()
            if isinstance(value, torch.Tensor)
        }

        return super()._hydrogen_state_correction(
            positions,
            base_per_atom,
        )


def evaluate(symbols, coordinates):
    simulation = CaptureHState(
        boxes=[(symbols, coordinates)],
        box_size=BOX_SIZE,
        target_temperature=0.0,
        friction=0.0,
        device="cpu",
        dtype=torch.float64,
        random_seed=0,
        relax_on_start=False,
    )

    return simulation, simulation._occupancy_values


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


def active_pair_records(simulation, values, symbols, wanted_pair):
    target = tuple(sorted(wanted_pair))

    neighbours = values["neighbours"]
    mask = values["mask"]

    records = []

    for atom_i in range(len(symbols)):
        for slot in range(neighbours.shape[1]):
            if not bool(mask[atom_i, slot]):
                continue

            atom_j = int(neighbours[atom_i, slot])

            if atom_j <= atom_i or atom_j >= len(symbols):
                continue

            pair = tuple(sorted((symbols[atom_i], symbols[atom_j])))

            if pair != target:
                continue

            type_i = int(simulation.types_numpy[atom_i])
            type_j = int(simulation.types_numpy[atom_j])

            r = float(values["distances"][atom_i, slot])
            r0 = float(R.BOND_LENGTH[type_i, type_j])
            width = float(R.BOND_WIDTH[type_i, type_j])

            overlap = math.exp(-width * (r - r0))

            records.append({
                "atom_i": atom_i,
                "atom_j": atom_j,
                "r": r,
                "r0": r0,
                "width": width,
                "taper": float(values["taper"][atom_i, slot]),
                "order": float(values["order"][atom_i, slot]),
                "overlap": overlap,
                "inner": float(R.CUTOFF_INNER[type_i, type_j]),
                "outer": float(R.CUTOFF_OUTER[type_i, type_j]),
            })

    return records


def strongest_pair(simulation, values, symbols, wanted_pair):
    records = active_pair_records(
        simulation,
        values,
        symbols,
        wanted_pair,
    )

    if not records:
        raise RuntimeError(
            f"No active {wanted_pair} pair found in {symbols}"
        )

    return max(
        records,
        key=lambda row: (
            row["overlap"],
            row["taper"],
            -row["r"],
        ),
    )


def stretch_geometry(coordinates, atom_i, atom_j, target_distance):
    positions = np.asarray(coordinates, dtype=float).copy()

    vector = positions[atom_j] - positions[atom_i]
    length = float(np.linalg.norm(vector))

    if length < 1e-12:
        raise RuntimeError("Cannot stretch zero-length bond")

    direction = vector / length

    positions[atom_j] = (
        positions[atom_i]
        + direction * target_distance
    )

    return positions


def target_pair_snapshot(
    symbols,
    coordinates,
    atom_i,
    atom_j,
):
    simulation, values = evaluate(symbols, coordinates)

    neighbours = values["neighbours"][atom_i]
    mask = values["mask"][atom_i]

    for slot in range(len(neighbours)):
        if (
            bool(mask[slot])
            and int(neighbours[slot]) == atom_j
        ):
            type_i = int(simulation.types_numpy[atom_i])
            type_j = int(simulation.types_numpy[atom_j])

            r = float(values["distances"][atom_i, slot])
            r0 = float(R.BOND_LENGTH[type_i, type_j])
            width = float(R.BOND_WIDTH[type_i, type_j])

            return {
                "r": r,
                "r0": r0,
                "width": width,
                "dr": r - r0,
                "taper": float(values["taper"][atom_i, slot]),
                "order": float(values["order"][atom_i, slot]),
                "overlap": math.exp(-width * (r - r0)),
                "inner": float(R.CUTOFF_INNER[type_i, type_j]),
                "outer": float(R.CUTOFF_OUTER[type_i, type_j]),
            }

    # Once outside the neighbour/cutoff region the pair may disappear from
    # the neighbour table. Its radial topology quantities are then zero.
    type_i = int(R.ELEMENT_INDEX[symbols[atom_i]])
    type_j = int(R.ELEMENT_INDEX[symbols[atom_j]])

    r0 = float(R.BOND_LENGTH[type_i, type_j])
    width = float(R.BOND_WIDTH[type_i, type_j])
    r = float(
        np.linalg.norm(
            np.asarray(coordinates[atom_j])
            - np.asarray(coordinates[atom_i])
        )
    )

    return {
        "r": r,
        "r0": r0,
        "width": width,
        "dr": r - r0,
        "taper": 0.0,
        "order": 0.0,
        "overlap": math.exp(-width * (r - r0)),
        "inner": float(R.CUTOFF_INNER[type_i, type_j]),
        "outer": float(R.CUTOFF_OUTER[type_i, type_j]),
    }


def nonbonded_water_oo(payload):
    rows = []

    dense = [
        geometry
        for geometry in payload["geometries"]
        if (
            geometry["system"] == "water"
            and geometry["sample_kind"] == "dense_transfer_scan"
        )
    ]

    for geometry in dense:
        rc = geometry.get("reaction_coordinate", {})
        x = rc.get("transfer_distance_angstrom")

        if x is None:
            continue

        x = float(x)

        if not (1.08 <= x <= 1.16 + 1e-9):
            continue

        simulation, values = evaluate(
            geometry["symbols"],
            geometry["coordinates_angstrom"],
        )

        candidates = active_pair_records(
            simulation,
            values,
            geometry["symbols"],
            ("O", "O"),
        )

        if not candidates:
            continue

        selected = max(
            candidates,
            key=lambda row: (
                row["overlap"],
                row["taper"],
            ),
        )

        rows.append({
            "label": f"water O...O x={x:.3f}",
            **selected,
        })

    return rows


def distance_for_overlap(r0, width, overlap):
    return r0 - math.log(overlap) / width


def main():
    payload = json.loads(
        GEOMETRIES.read_text(encoding="utf-8")
    )

    # Genuine bond environments from the same QM microscope references.
    specs = [
        ("h3", ("H", "H"), "H3 H-H"),
        ("methane", ("C", "H"), "CH4 C-H"),
        ("formaldehyde", ("C", "H"), "CH2O C-H"),
        ("formaldehyde", ("C", "O"), "CH2O C-O"),
        ("water", ("O", "H"), "H2O O-H"),
    ]

    systems = []

    for system, pair_type, label in specs:
        geometry = find_reference(payload, system)

        simulation, values = evaluate(
            geometry["symbols"],
            geometry["coordinates_angstrom"],
        )

        selected = strongest_pair(
            simulation,
            values,
            geometry["symbols"],
            pair_type,
        )

        systems.append({
            "label": label,
            "symbols": list(geometry["symbols"]),
            "coordinates": np.asarray(
                geometry["coordinates_angstrom"],
                dtype=float,
            ),
            "atom_i": selected["atom_i"],
            "atom_j": selected["atom_j"],
        })

    # Same-pair O-O holdout from existing peroxide calibration geometry.
    peroxide_symbols, peroxide_coordinates = (
        hydrogen_peroxide_geometry(oo_distance=1.475)
    )

    systems.append({
        "label": "H2O2 O-O",
        "symbols": list(peroxide_symbols),
        "coordinates": np.asarray(
            peroxide_coordinates,
            dtype=float,
        ),
        "atom_i": 0,
        "atom_j": 1,
    })

    nonbonded = nonbonded_water_oo(payload)

    if not nonbonded:
        raise RuntimeError(
            "Could not recover the bad water O...O contacts"
        )

    nonbonded_ceiling = max(
        row["overlap"]
        for row in nonbonded
    )

    print()
    print("BOND-DISSOCIATION OCCUPANCY MICROSCOPE")
    print()
    print(
        f"bad water O...O overlap ceiling = "
        f"{nonbonded_ceiling:.6f}"
    )
    print()

    output_rows = []

    for system in systems:
        symbols = system["symbols"]
        coordinates = system["coordinates"]
        atom_i = system["atom_i"]
        atom_j = system["atom_j"]

        reference = target_pair_snapshot(
            symbols,
            coordinates,
            atom_i,
            atom_j,
        )

        r0 = reference["r0"]
        width = reference["width"]
        inner = reference["inner"]
        outer = reference["outer"]

        print(
            f"{system['label']}"
            f"  r0={r0:.4f} A"
            f"  a={width:.4f}"
            f"  inner={inner:.4f}"
            f"  outer={outer:.4f}"
        )

        sample_distances = []

        for ratio in STRETCH_RATIOS:
            distance = r0 * ratio
            if distance < outer - 1e-7:
                sample_distances.append(distance)

        # Add distances corresponding to useful overlap landmarks.
        for overlap_landmark in (0.8, 0.6, 0.4, 0.2, nonbonded_ceiling):
            distance = distance_for_overlap(
                r0,
                width,
                overlap_landmark,
            )

            if r0 <= distance <= outer + 1e-7:
                sample_distances.append(distance)

        sample_distances.extend([
            inner,
            max(inner, outer - 1e-4),
        ])

        sample_distances = sorted({
            round(float(distance), 8)
            for distance in sample_distances
            if distance >= r0 - 1e-8
        })

        for distance in sample_distances:
            stretched = stretch_geometry(
                coordinates,
                atom_i,
                atom_j,
                distance,
            )

            snapshot = target_pair_snapshot(
                symbols,
                stretched,
                atom_i,
                atom_j,
            )

            output_rows.append({
                "example": system["label"],
                "pair": (
                    f"{symbols[atom_i]}-{symbols[atom_j]}"
                ),
                "distance_A": snapshot["r"],
                "r0_A": snapshot["r0"],
                "width_per_A": snapshot["width"],
                "stretch_ratio": (
                    snapshot["r"] / snapshot["r0"]
                ),
                "dr_A": snapshot["dr"],
                "taper": snapshot["taper"],
                "order": snapshot["order"],
                "morse_overlap": snapshot["overlap"],
                "inner_cutoff_A": snapshot["inner"],
                "outer_cutoff_A": snapshot["outer"],
                "below_bad_water_ceiling": int(
                    snapshot["overlap"]
                    <= nonbonded_ceiling
                ),
            })

            marker = ""

            if abs(snapshot["overlap"] - 0.8) < 0.01:
                marker = "  ~s=0.8"
            elif abs(snapshot["overlap"] - 0.6) < 0.01:
                marker = "  ~s=0.6"
            elif abs(snapshot["overlap"] - 0.4) < 0.01:
                marker = "  ~s=0.4"
            elif abs(snapshot["overlap"] - 0.2) < 0.01:
                marker = "  ~s=0.2"
            elif abs(
                snapshot["overlap"]
                - nonbonded_ceiling
            ) < 0.005:
                marker = "  ~bad-water ceiling"

            print(
                f"  r={snapshot['r']:.4f}"
                f"  r/r0={snapshot['r']/snapshot['r0']:.3f}"
                f"  dr={snapshot['dr']:+.4f}"
                f"  taper={snapshot['taper']:.4f}"
                f"  overlap={snapshot['overlap']:.5f}"
                f"  order={snapshot['order']:.4f}"
                f"{marker}"
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
            fieldnames=list(output_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("NONBONDED WATER O...O BENCHMARK")
    for row in nonbonded:
        print(
            f"  {row['label']:<23s}"
            f" r={row['r']:.4f}"
            f" taper={row['taper']:.4f}"
            f" overlap={row['overlap']:.6f}"
            f" order={row['order']:.4f}"
        )

    print()
    print("OVERLAP LANDMARK DISTANCES")
    print(
        "(distance beyond each pair's live r0 where s reaches a value)"
    )

    for system in systems:
        symbols = system["symbols"]
        coordinates = system["coordinates"]
        atom_i = system["atom_i"]
        atom_j = system["atom_j"]

        reference = target_pair_snapshot(
            symbols,
            coordinates,
            atom_i,
            atom_j,
        )

        print(f"  {system['label']:<12s}", end="")

        for overlap in (
            0.8,
            0.6,
            0.4,
            0.2,
            nonbonded_ceiling,
        ):
            distance = distance_for_overlap(
                reference["r0"],
                reference["width"],
                overlap,
            )

            print(
                f"  s={overlap:.3f}:"
                f"{distance:.3f}A"
                f"({distance/reference['r0']:.2f}r0)",
                end="",
            )

        print()

    print()
    print(f"wrote : {OUTPUT}")


if __name__ == "__main__":
    main()
