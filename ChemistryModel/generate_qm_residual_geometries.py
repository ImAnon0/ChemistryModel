"""Generate the first QM-residual geometry dataset.

This file deliberately has no Torch or Psi4 dependency. It only creates
molecular geometries and metadata, then writes ordinary JSON. The same JSON
can therefore be consumed by:

    1. ChemistryModel/Torch in the normal environment
    2. Psi4 in the chem-sapt environment

The first experiment is intentionally small:
    - H + H2
    - H2O + OH proton transfer
    - H + CH2O
    - H + CH4

Methane is marked as a whole-system holdout. The first residual model should
not train on methane; it exists to test whether the correction transfers.

Energy convention
-----------------
Do NOT train directly on raw "QM total energy - ChemistryModel energy".
The two programs have unrelated absolute energy zeros. Later, the dataset
builder will use each system's reactant_reference geometry and construct:

    delta_relative(g) =
        [E_QM(g)   - E_QM(reference)]
      - [E_base(g) - E_base(reference)]

That removes the arbitrary per-composition energy offset while preserving the
shape of the potential-energy surface, reaction energies, and barriers.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


SCHEMA_VERSION = 1
DEFAULT_SEED = 20260815
DEFAULT_OUTPUT = Path("research_data/qm_residual/geometries.json")
GRID_SIZE = 8
JITTER_POINTS = 24


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count < 2:
        return [float(start)]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]


def vec_add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def vec_scale(v, scale):
    return [v[0] * scale, v[1] * scale, v[2] * scale]


def round_coords(coords, digits=10):
    return [[round(float(value), digits) for value in xyz] for xyz in coords]


def h3_geometry(donor_length: float, transfer_length: float, spectators=None):
    """H-H...H collinear abstraction geometry."""
    donor_h = [0.0, 0.0, 0.0]
    transfer_h = [0.0, 0.0, donor_length]
    incoming_h = [0.0, 0.0, donor_length + transfer_length]
    return ["H", "H", "H"], [donor_h, transfer_h, incoming_h]


def formaldehyde_geometry(donor_length: float, transfer_length: float, spectators=None):
    """H + CH2O -> H2 + HCO geometry matching hf_surface_scan.py."""
    if spectators is None:
        spectators = [1.20, 1.09, 122.0, 122.0]

    length_co, length_ch, angle_donor, angle_other = spectators
    donor_radians = math.radians(angle_donor)
    other_radians = math.radians(angle_other)

    donor_axis = [math.sin(donor_radians), math.cos(donor_radians), 0.0]
    other_axis = [-math.sin(other_radians), math.cos(other_radians), 0.0]

    carbon = [0.0, 0.0, 0.0]
    oxygen = [0.0, length_co, 0.0]
    donor_h = vec_scale(donor_axis, donor_length)
    other_h = vec_scale(other_axis, length_ch)
    incoming_h = vec_add(donor_h, vec_scale(donor_axis, transfer_length))

    return ["C", "O", "H", "H", "H"], [carbon, oxygen, donor_h, other_h, incoming_h]


def methane_geometry(donor_length: float, transfer_length: float, spectators=None):
    """H + CH4 -> H2 + CH3 geometry matching hf_surface_scan.py."""
    if spectators is None:
        spectators = [1.09, 1.09, 1.09, 109.47]

    first, second, third, angle = spectators
    radians = math.radians(angle)

    carbon = [0.0, 0.0, 0.0]
    donor_h = [0.0, 0.0, donor_length]
    incoming_h = [0.0, 0.0, donor_length + transfer_length]

    spectator_positions = []
    for index, length in enumerate((first, second, third)):
        turn = 2.0 * math.pi * index / 3.0
        spectator_positions.append([
            length * math.sin(radians) * math.cos(turn),
            length * math.sin(radians) * math.sin(turn),
            length * math.cos(radians),
        ])

    return ["C", "H", "H", "H", "H", "H"], [
        carbon, donor_h, *spectator_positions, incoming_h
    ]


def water_geometry(donor_length: float, transfer_length: float, spectators=None):
    """H2O + OH -> OH + H2O geometry matching hf_surface_scan.py."""
    if spectators is None:
        spectators = [0.96, 0.96, 104.5, 104.5]

    donor_oh, acceptor_oh, donor_angle, acceptor_angle = spectators
    donor_oxygen = [0.0, 0.0, 0.0]
    moving_h = [donor_length, 0.0, 0.0]
    acceptor_oxygen = [donor_length + transfer_length, 0.0, 0.0]

    def spoke(origin, length, angle, direction, tilt):
        radians = math.radians(angle)
        offset = [
            direction * math.cos(radians),
            math.sin(radians) * math.cos(tilt),
            math.sin(radians) * math.sin(tilt),
        ]
        return vec_add(origin, vec_scale(offset, length))

    donor_spoke = spoke(donor_oxygen, donor_oh, donor_angle, -1.0, math.pi / 2.0)
    acceptor_spoke = spoke(acceptor_oxygen, acceptor_oh, acceptor_angle, 1.0, math.pi / 2.0)

    return ["O", "H", "H", "O", "H"], [
        donor_oxygen,
        moving_h,
        donor_spoke,
        acceptor_oxygen,
        acceptor_spoke,
    ]


SYSTEMS = {
    "h3": {
        "description": "H + H2 -> H2 + H",
        "builder": h3_geometry,
        "split": "train",
        "charge": 0,
        "multiplicity": 2,
        "grid": {"donor": (0.70, 1.70), "transfer": (0.70, 1.70)},
        "reactant_reference": (0.74144, 5.00),
        "product_reference": (5.00, 0.74144),
        "spectators": None,
    },
    "water": {
        "description": "H2O + OH -> OH + H2O",
        "builder": water_geometry,
        "split": "train",
        "charge": 0,
        "multiplicity": 2,
        "grid": {"donor": (0.90, 1.80), "transfer": (0.90, 1.80)},
        "reactant_reference": (0.96, 5.00),
        "product_reference": (5.00, 0.96),
        "spectators": [0.96, 0.96, 104.5, 104.5],
    },
    "formaldehyde": {
        "description": "H + CH2O -> H2 + HCO",
        "builder": formaldehyde_geometry,
        "split": "train",
        "charge": 0,
        "multiplicity": 2,
        "grid": {"donor": (1.00, 1.90), "transfer": (0.65, 1.60)},
        "reactant_reference": (1.09, 5.00),
        "product_reference": (5.00, 0.74144),
        "spectators": [1.20, 1.09, 122.0, 122.0],
    },
    "methane": {
        "description": "H + CH4 -> H2 + CH3",
        "builder": methane_geometry,
        "split": "holdout",
        "charge": 0,
        "multiplicity": 2,
        "grid": {"donor": (1.00, 1.90), "transfer": (0.65, 1.60)},
        "reactant_reference": (1.09, 5.00),
        "product_reference": (5.00, 0.74144),
        "spectators": [1.09, 1.09, 1.09, 109.47],
    },
}


def classify_region(system: str, donor: float, transfer: float) -> str:
    """Loose labels for plots/diagnostics, not training targets."""
    if system == "water":
        if donor <= 1.10 and transfer >= 1.40:
            return "reactant_like"
        if donor >= 1.40 and transfer <= 1.10:
            return "product_like"
        if abs(donor - transfer) <= 0.18:
            return "transfer_region"
        return "off_path"

    equilibrium = 0.74144 if system == "h3" else 1.09
    if abs(donor - equilibrium) <= 0.16 and transfer >= 1.30:
        return "reactant_like"
    if donor >= 1.45 and transfer <= 0.92:
        return "product_like"
    if 0.85 <= transfer <= 1.25 and donor >= equilibrium:
        return "transfer_region"
    return "off_path"


def jitter_spectators(system: str, base, rng: random.Random):
    if base is None:
        return None

    values = list(base)
    if system == "formaldehyde":
        values[0] += rng.uniform(-0.05, 0.05)
        values[1] += rng.uniform(-0.04, 0.04)
        values[2] += rng.uniform(-5.0, 5.0)
        values[3] += rng.uniform(-5.0, 5.0)
    elif system == "methane":
        for index in range(3):
            values[index] += rng.uniform(-0.04, 0.04)
        values[3] += rng.uniform(-4.0, 4.0)
    elif system == "water":
        values[0] += rng.uniform(-0.035, 0.035)
        values[1] += rng.uniform(-0.035, 0.035)
        values[2] += rng.uniform(-4.0, 4.0)
        values[3] += rng.uniform(-4.0, 4.0)
    return values


def make_record(*, system_name, sample_kind, index, donor, transfer, spectators, region=None):
    config = SYSTEMS[system_name]
    symbols, coordinates = config["builder"](donor, transfer, spectators)
    if region is None:
        region = classify_region(system_name, donor, transfer)

    return {
        "geometry_id": f"{system_name}_{sample_kind}_{index:04d}",
        "system": system_name,
        "description": config["description"],
        "split": config["split"],
        "sample_kind": sample_kind,
        "region": region,
        "charge": config["charge"],
        "multiplicity": config["multiplicity"],
        "symbols": symbols,
        "coordinates_angstrom": round_coords(coordinates),
        "reaction_coordinate": {
            "donor_distance_angstrom": round(float(donor), 10),
            "transfer_distance_angstrom": round(float(transfer), 10),
        },
        "spectators": None if spectators is None else [round(float(v), 10) for v in spectators],
    }


def generate_records(seed: int) -> list[dict]:
    rng = random.Random(seed)
    records = []

    for system_name, config in SYSTEMS.items():
        donor, transfer = config["reactant_reference"]
        records.append(make_record(
            system_name=system_name,
            sample_kind="reactant_reference",
            index=0,
            donor=donor,
            transfer=transfer,
            spectators=config["spectators"],
            region="reactant_reference",
        ))

        donor, transfer = config["product_reference"]
        records.append(make_record(
            system_name=system_name,
            sample_kind="product_reference",
            index=0,
            donor=donor,
            transfer=transfer,
            spectators=config["spectators"],
            region="product_reference",
        ))

        donor_values = linspace(*config["grid"]["donor"], GRID_SIZE)
        transfer_values = linspace(*config["grid"]["transfer"], GRID_SIZE)
        grid_index = 0
        for donor in donor_values:
            for transfer in transfer_values:
                records.append(make_record(
                    system_name=system_name,
                    sample_kind="grid",
                    index=grid_index,
                    donor=donor,
                    transfer=transfer,
                    spectators=config["spectators"],
                ))
                grid_index += 1

        donor_low, donor_high = config["grid"]["donor"]
        transfer_low, transfer_high = config["grid"]["transfer"]
        for jitter_index in range(JITTER_POINTS):
            donor = rng.uniform(donor_low, donor_high)
            transfer = rng.uniform(transfer_low, transfer_high)
            spectators = jitter_spectators(system_name, config["spectators"], rng)
            records.append(make_record(
                system_name=system_name,
                sample_kind="jitter",
                index=jitter_index,
                donor=donor,
                transfer=transfer,
                spectators=spectators,
            ))

    return records


def validate_records(records: list[dict]):
    ids = [row["geometry_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate geometry_id values found")

    for row in records:
        symbols = row["symbols"]
        coords = row["coordinates_angstrom"]
        if len(symbols) != len(coords):
            raise ValueError(f"{row['geometry_id']}: symbol/coordinate count mismatch")
        for xyz in coords:
            if len(xyz) != 3:
                raise ValueError(f"{row['geometry_id']}: coordinate is not xyz")
            if not all(math.isfinite(float(value)) for value in xyz):
                raise ValueError(f"{row['geometry_id']}: non-finite coordinate")

    for system_name in SYSTEMS:
        system_rows = [r for r in records if r["system"] == system_name]
        reactant_refs = [r for r in system_rows if r["sample_kind"] == "reactant_reference"]
        product_refs = [r for r in system_rows if r["sample_kind"] == "product_reference"]
        if len(reactant_refs) != 1:
            raise ValueError(f"{system_name}: expected one reactant reference")
        if len(product_refs) != 1:
            raise ValueError(f"{system_name}: expected one product reference")


def build_payload(seed: int) -> dict:
    records = generate_records(seed)
    validate_records(records)

    counts = {}
    for system_name in SYSTEMS:
        rows = [row for row in records if row["system"] == system_name]
        counts[system_name] = {
            "total": len(rows),
            "train": sum(row["split"] == "train" for row in rows),
            "holdout": sum(row["split"] == "holdout" for row in rows),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "units": {"distance": "angstrom", "energy": "eV"},
        "energy_alignment": {
            "reference_sample_kind": "reactant_reference",
            "formula": "delta_relative = (E_qm - E_qm_reference) - (E_base - E_base_reference)",
        },
        "systems": {
            name: {
                "description": config["description"],
                "split": config["split"],
                "charge": config["charge"],
                "multiplicity": config["multiplicity"],
            }
            for name, config in SYSTEMS.items()
        },
        "counts": counts,
        "geometries": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    payload = build_payload(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    total = len(payload["geometries"])
    train = sum(row["split"] == "train" for row in payload["geometries"])
    holdout = total - train

    print(f"wrote {args.output}")
    print(f"total geometries : {total}")
    print(f"train geometries : {train}")
    print(f"holdout          : {holdout}")
    print()
    for name, counts in payload["counts"].items():
        split = payload["systems"][name]["split"]
        print(f"{name:12s} {counts['total']:4d}  split={split}")


if __name__ == "__main__":
    main()
