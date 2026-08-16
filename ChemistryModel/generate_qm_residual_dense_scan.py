"""Generate dense one-dimensional QM/base microscope scans.

Purpose
-------
The broad residual dataset showed large energy mismatches concentrated where
ChemistryModel moves through its reactive taper/overcoordination region.  This
script freezes the spectator geometry and scans only the transferring bond
distance in 0.02 A increments through the suspicious interval.

It intentionally reuses the exact geometry builders from:
    generate_qm_residual_geometries.py

Outputs:
    research_data/qm_residual/dense_scan_geometries.json

Representative slices were chosen from the grid rows that produced the largest
adjacent residual jumps:

    H3            donor = 0.842857 A
    water         donor = 1.028571 A
    formaldehyde  donor = 1.385714 A
    methane       donor = 1.385714 A

Each system also contains one normal reactant_reference row so later energy
alignment can use exactly the same convention as the broad residual dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import generate_qm_residual_geometries as G


DEFAULT_OUTPUT = Path(
    "research_data/qm_residual/dense_scan_geometries.json"
)
STEP = 0.02


SCAN_SPECS = {
    "h3": {
        "fixed_donor": 0.8428571428571429,
        "transfer_start": 0.84,
        "transfer_stop": 1.28,
        "source_jump": "h3_grid_0010 -> h3_grid_0011",
        "active_bond": "H-H",
        "taper_inner": 1.25 * 0.74144,
        "taper_outer": 1.60 * 0.74144,
    },
    "water": {
        "fixed_donor": 1.0285714285714285,
        "transfer_start": 1.08,
        "transfer_stop": 1.64,
        "source_jump": "water_grid_0010 -> water_grid_0011",
        "active_bond": "O-H",
        "taper_inner": 1.25 * 0.960,
        "taper_outer": 1.60 * 0.960,
    },
    "formaldehyde": {
        "fixed_donor": 1.3857142857142857,
        "transfer_start": 0.84,
        "transfer_stop": 1.28,
        "source_jump": (
            "formaldehyde_grid_0026 -> formaldehyde_grid_0027"
        ),
        "active_bond": "H-H",
        "taper_inner": 1.25 * 0.74144,
        "taper_outer": 1.60 * 0.74144,
    },
    "methane": {
        "fixed_donor": 1.3857142857142857,
        "transfer_start": 0.84,
        "transfer_stop": 1.28,
        "source_jump": "methane_grid_0026 -> methane_grid_0027",
        "active_bond": "H-H",
        "taper_inner": 1.25 * 0.74144,
        "taper_outer": 1.60 * 0.74144,
    },
}


def inclusive_range(start: float, stop: float, step: float):
    count = int(round((stop - start) / step))
    return [round(start + i * step, 10) for i in range(count + 1)]


def make_dense_row(
    *,
    system_name: str,
    index: int,
    donor: float,
    transfer: float,
):
    config = G.SYSTEMS[system_name]
    symbols, coordinates = config["builder"](
        donor,
        transfer,
        config["spectators"],
    )

    spec = SCAN_SPECS[system_name]

    return {
        "geometry_id": f"{system_name}_dense_transfer_{index:04d}",
        "system": system_name,
        "description": config["description"],
        "split": config["split"],
        "sample_kind": "dense_transfer_scan",
        "region": "transfer_microscope",
        "charge": config["charge"],
        "multiplicity": config["multiplicity"],
        "symbols": symbols,
        "coordinates_angstrom": G.round_coords(coordinates),
        "reaction_coordinate": {
            "donor_distance_angstrom": round(float(donor), 10),
            "transfer_distance_angstrom": round(float(transfer), 10),
        },
        "spectators": (
            None
            if config["spectators"] is None
            else [
                round(float(value), 10)
                for value in config["spectators"]
            ]
        ),
        "dense_scan": {
            "axis": "transfer",
            "fixed_donor_distance_angstrom": round(float(donor), 10),
            "active_bond": spec["active_bond"],
            "taper_inner_angstrom": round(spec["taper_inner"], 10),
            "taper_outer_angstrom": round(spec["taper_outer"], 10),
            "source_jump": spec["source_jump"],
        },
    }


def build_payload():
    rows = []

    for system_name, spec in SCAN_SPECS.items():
        config = G.SYSTEMS[system_name]

        # Standard reference row so the existing residual builder can align
        # this dense scan without any special energy-zero handling.
        donor, transfer = config["reactant_reference"]
        rows.append(
            G.make_record(
                system_name=system_name,
                sample_kind="reactant_reference",
                index=0,
                donor=donor,
                transfer=transfer,
                spectators=config["spectators"],
                region="reactant_reference",
            )
        )

        # Keep the same validation contract as the original dataset generator:
        # every system carries exactly one reactant and one product reference.
        donor, transfer = config["product_reference"]
        rows.append(
            G.make_record(
                system_name=system_name,
                sample_kind="product_reference",
                index=0,
                donor=donor,
                transfer=transfer,
                spectators=config["spectators"],
                region="product_reference",
            )
        )

        transfers = inclusive_range(
            spec["transfer_start"],
            spec["transfer_stop"],
            STEP,
        )

        for index, transfer in enumerate(transfers):
            rows.append(
                make_dense_row(
                    system_name=system_name,
                    index=index,
                    donor=spec["fixed_donor"],
                    transfer=transfer,
                )
            )

    G.validate_records(rows)

    return {
        "schema_version": G.SCHEMA_VERSION,
        "purpose": "dense reactive-taper microscope scan",
        "units": {
            "distance": "angstrom",
            "energy": "eV",
        },
        "energy_alignment": {
            "reference_sample_kind": "reactant_reference",
            "formula": (
                "delta_relative = "
                "(E_qm - E_qm_reference) - "
                "(E_base - E_base_reference)"
            ),
        },
        "scan_step_angstrom": STEP,
        "scan_specs": SCAN_SPECS,
        "geometries": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote : {args.output}")
    print(f"rows  : {len(payload['geometries'])}")
    print(f"step  : {STEP:.2f} A")
    print("")

    for system_name, spec in SCAN_SPECS.items():
        count = sum(
            row["system"] == system_name
            and row["sample_kind"] == "dense_transfer_scan"
            for row in payload["geometries"]
        )
        print(
            f"{system_name:14s} "
            f"scan={count:2d}  "
            f"donor={spec['fixed_donor']:.6f} A  "
            f"transfer={spec['transfer_start']:.2f}.."
            f"{spec['transfer_stop']:.2f} A  "
            f"taper=[{spec['taper_inner']:.4f}, "
            f"{spec['taper_outer']:.4f}] A"
        )


if __name__ == "__main__":
    main()
