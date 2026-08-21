"""Build the small, versioned electronic-observable geometry manifest.

The manifest contains coordinates only.  It can be consumed in both the
normal ChemistryModel environment and the separate Psi4 ``chem-sapt``
environment without importing production physics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_DENSE = Path("research_data/qm_residual/dense_scan_geometries.json")
DEFAULT_OUTPUT = Path("research_data/electronic_observables/manifest.json")


def _round(coordinates):
    return [[round(float(value), 10) for value in xyz] for xyz in coordinates]


def _row(
    geometry_id,
    family,
    role,
    symbols,
    coordinates,
    *,
    charge=0,
    multiplicity=1,
    group=None,
    reference=None,
    coordinate=None,
    purpose="",
):
    return {
        "geometry_id": geometry_id,
        "family": family,
        "role": role,
        "comparison_group": group or family,
        "reference_geometry_id": reference or geometry_id,
        "purpose": purpose,
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "symbols": list(symbols),
        "coordinates_angstrom": _round(coordinates),
        "scan_coordinate": coordinate or {},
    }


def _select_dense(rows, system, *, sample_kind=None, transfer=None):
    candidates = [row for row in rows if row["system"] == system]
    if sample_kind is not None:
        candidates = [
            row for row in candidates if row["sample_kind"] == sample_kind
        ]
    if transfer is not None:
        candidates = [
            row for row in candidates
            if row["sample_kind"] == "dense_transfer_scan"
        ]
        candidates.sort(
            key=lambda row: abs(
                float(row["reaction_coordinate"]["transfer_distance_angstrom"])
                - float(transfer)
            )
        )
    if not candidates:
        raise ValueError(f"no dense row for {system=}, {sample_kind=}, {transfer=}")
    return candidates[0]


def _copy_dense(source, geometry_id, family, role, group, reference, purpose):
    coordinate = dict(source.get("reaction_coordinate", {}))
    coordinate["source_geometry_id"] = source["geometry_id"]
    return _row(
        geometry_id,
        family,
        role,
        source["symbols"],
        source["coordinates_angstrom"],
        charge=source["charge"],
        multiplicity=source["multiplicity"],
        group=group,
        reference=reference,
        coordinate=coordinate,
        purpose=purpose,
    )


def _water(angle_degrees, distance=0.9572):
    half = math.radians(angle_degrees / 2.0)
    return ["O", "H", "H"], [
        [0.0, 0.0, 0.0],
        [distance * math.cos(half), distance * math.sin(half), 0.0],
        [distance * math.cos(half), -distance * math.sin(half), 0.0],
    ]


def _methane(distance=1.09):
    scale = distance / math.sqrt(3.0)
    return ["C", "H", "H", "H", "H"], [
        [0.0, 0.0, 0.0],
        [scale, scale, scale],
        [scale, -scale, -scale],
        [-scale, scale, -scale],
        [-scale, -scale, scale],
    ]


def _ethane(cc=1.525, ch=1.09):
    # Staggered D3d geometry, C-C axis along x.
    axial = ch / 3.0
    radial = ch * 2.0 * math.sqrt(2.0) / 3.0
    coords = [[-cc / 2.0, 0.0, 0.0], [cc / 2.0, 0.0, 0.0]]
    for index in range(3):
        phi = 2.0 * math.pi * index / 3.0
        coords.append([
            -cc / 2.0 - axial,
            radial * math.cos(phi),
            radial * math.sin(phi),
        ])
    for index in range(3):
        phi = 2.0 * math.pi * index / 3.0 + math.pi / 3.0
        coords.append([
            cc / 2.0 + axial,
            radial * math.cos(phi),
            radial * math.sin(phi),
        ])
    return ["C", "C", *(["H"] * 6)], coords


def _formaldehyde():
    co, ch, angle = 1.208, 1.116, math.radians(121.8)
    return ["C", "O", "H", "H"], [
        [0.0, 0.0, 0.0],
        [co, 0.0, 0.0],
        [ch * math.cos(angle), ch * math.sin(angle), 0.0],
        [ch * math.cos(angle), -ch * math.sin(angle), 0.0],
    ]


def build_manifest(dense_path=DEFAULT_DENSE):
    dense = json.loads(Path(dense_path).read_text(encoding="utf-8"))["geometries"]
    rows = []

    rows.append(_row(
        "h2_equilibrium", "h2", "characterisation", ["H", "H"],
        [[0, 0, -0.37072], [0, 0, 0.37072]],
        purpose="Symmetry, zero dipole, and accepted H2 reference.",
    ))

    for system, family, values, reference_id, purpose in [
        ("h3", "h3_transfer", [0.90, 1.06, 1.22], "h3_reactant",
         "Three-centre H electronic redistribution and false-binding gate."),
        ("formaldehyde", "h_formaldehyde_transfer", [0.90, 1.06, 1.22],
         "h_formaldehyde_reactant", "H abstraction from formaldehyde."),
        ("water", "water_transfer", [1.10, 1.30, 1.52], "water_transfer_reactant",
         "Proton transfer and the established water microscope."),
    ]:
        source = _select_dense(dense, system, sample_kind="reactant_reference")
        rows.append(_copy_dense(
            source, reference_id, family, "characterisation", family,
            reference_id, purpose + " Separated reactant reference.",
        ))
        for index, value in enumerate(values):
            source = _select_dense(dense, system, transfer=value)
            role = "validation" if index < 2 else "final_holdout"
            rows.append(_copy_dense(
                source, f"{family}_{value:.2f}".replace(".", "p"), family,
                role, family, reference_id, purpose,
            ))

    # H approaching H2: distance and orientation are not represented by one
    # radial bond-capacity number and therefore provide a clean direction test.
    h2 = [[0, 0, -0.37072], [0, 0, 0.37072]]
    h_h2_specs = [
        ("h_h2_linear_3p00", [0, 0, 3.37072], 3.0, "linear", "characterisation"),
        ("h_h2_linear_1p80", [0, 0, 2.17072], 1.8, "linear", "validation"),
        ("h_h2_linear_1p20", [0, 0, 1.57072], 1.2, "linear", "validation"),
        ("h_h2_perpendicular_1p50", [1.5, 0, 0], 1.5, "perpendicular", "final_holdout"),
    ]
    for geometry_id, incoming, distance, orientation, role in h_h2_specs:
        rows.append(_row(
            geometry_id, "h_h2_approach", role, ["H", "H", "H"],
            [*h2, incoming], multiplicity=2, group="h_h2_approach",
            reference="h_h2_linear_3p00",
            coordinate={"approach_distance_angstrom": distance, "orientation": orientation},
            purpose="Reject an artificial H-H2 complex and resolve orientation.",
        ))

    for angle, role in [(90.0, "validation"), (104.5, "characterisation"), (120.0, "final_holdout")]:
        symbols, coordinates = _water(angle)
        rows.append(_row(
            f"water_angle_{angle:.1f}".replace(".", "p"), "water_angle", role,
            symbols, coordinates, group="water_angle", reference="water_angle_104p5",
            coordinate={"hoh_angle_degrees": angle, "oh_distance_angstrom": 0.9572},
            purpose="Directional density response at fixed radial participation.",
        ))

    for distance, role in [(0.80, "validation"), (0.97, "characterisation"), (1.20, "final_holdout")]:
        rows.append(_row(
            f"oh_stretch_{distance:.2f}".replace(".", "p"), "oh_stretch", role,
            ["O", "H"], [[0, 0, 0], [distance, 0, 0]], multiplicity=2,
            group="oh_stretch", reference="oh_stretch_0p97",
            coordinate={"oh_distance_angstrom": distance},
            purpose="Open-shell bond stretching and continuous charge response.",
        ))

    water_symbols, water_coordinates = _water(104.5)
    probe_specs = [
        ("water_h_probe_lone_pair", [-1.70, 0, 0], "opposite_bisector", "validation"),
        ("water_h_probe_out_of_plane", [0, 0, 1.70], "out_of_plane", "final_holdout"),
        ("water_h_probe_h_side", [1.70, 0, 0], "hydrogen_side", "final_holdout"),
    ]
    for geometry_id, probe, orientation, role in probe_specs:
        rows.append(_row(
            geometry_id, "water_h_direction", role,
            [*water_symbols, "H"], [*water_coordinates, probe], multiplicity=2,
            group="water_h_direction", reference="water_h_probe_out_of_plane",
            coordinate={"probe_radius_angstrom": 1.70, "orientation": orientation},
            purpose="Directional donor/acceptor density response without fitting a water term.",
        ))

    equilibrium = [
        ("methane_equilibrium", "methane", *_methane(),
         "Tetrahedral capacity and zero-dipole symmetry."),
        ("ethane_equilibrium", "ethane", *_ethane(),
         "Single C-C bond and local carbon response."),
        ("formaldehyde_equilibrium", "formaldehyde", *_formaldehyde(),
         "Polar C=O multiple-bond reference."),
        ("n2_equilibrium", "n2", ["N", "N"], [[0, 0, -0.5488], [0, 0, 0.5488]],
         "Triple-bond reference with zero-dipole symmetry."),
    ]
    for geometry_id, family, symbols, coordinates, purpose in equilibrium:
        rows.append(_row(
            geometry_id, family, "final_holdout", symbols, coordinates,
            purpose=purpose,
        ))

    ids = [row["geometry_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate geometry IDs")
    known = set(ids)
    for row in rows:
        if row["reference_geometry_id"] not in known:
            raise AssertionError(
                f"{row['geometry_id']}: unknown reference {row['reference_geometry_id']}"
            )
        if len(row["symbols"]) != len(row["coordinates_angstrom"]):
            raise AssertionError(f"{row['geometry_id']}: atom count mismatch")

    return {
        "schema_version": 1,
        "purpose": "small electronic-observable characterisation and hold-out set",
        "fitting_policy": "No parameters are fitted in this phase.",
        "coordinate_policy": "Exact fixed Cartesian inputs; C1, no reorientation, no COM shift.",
        "roles": {
            "characterisation": "May be inspected while designing a future model.",
            "validation": "Used to reject formulations, not tune individual geometries.",
            "final_holdout": "Must remain unseen until a candidate is frozen.",
        },
        "geometries": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest(args.dense)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {args.output}")
    print(f"geometries: {len(payload['geometries'])}")
    for role in payload["roles"]:
        count = sum(row["role"] == role for row in payload["geometries"])
        print(f"  {role:16s}: {count}")


if __name__ == "__main__":
    main()
