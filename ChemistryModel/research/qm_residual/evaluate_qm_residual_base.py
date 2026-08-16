"""Evaluate the frozen QM-residual geometry set with ChemistryModel.

Run this in the normal/base Python environment where Torch and ChemistryModel
are available.

The script does NOT move, relax, or integrate any atoms.  Each geometry is
evaluated exactly as stored in:

    research_data/qm_residual/geometries.json

Outputs:
    research_data/qm_residual/base_results.csv
    research_data/qm_residual/base_results.meta.json

The CSV contains the base potential energy plus useful diagnostic components.
The later dataset builder will align energies to each system's
reactant_reference before forming the residual target, so ChemistryModel's
absolute energy zero does not need to match the QM program's zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import torch

from reactive_torch import ReactiveSimulation


DEFAULT_INPUT = Path("research_data/qm_residual/geometries.json")
DEFAULT_OUTPUT = Path("research_data/qm_residual/base_results.csv")
DEFAULT_METADATA = Path("research_data/qm_residual/base_results.meta.json")
DEFAULT_BOX_SIZE = 30.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if "geometries" not in payload:
        raise ValueError(f"{path} does not contain a 'geometries' array")

    geometries = payload["geometries"]
    if not isinstance(geometries, list) or not geometries:
        raise ValueError(f"{path}: 'geometries' must be a non-empty list")

    ids = [row.get("geometry_id") for row in geometries]
    if any(not value for value in ids):
        raise ValueError("every geometry needs a geometry_id")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate geometry_id values found")

    return payload


def evaluate_geometry(row: dict, *, device: str, box_size: float) -> dict:
    symbols = row["symbols"]
    coordinates = row["coordinates_angstrom"]

    if len(symbols) != len(coordinates):
        raise ValueError(
            f"{row['geometry_id']}: symbols/coordinates length mismatch"
        )

    # float64 + no relaxation gives a deterministic static energy evaluation.
    # target_temperature=0 means the constructor creates zero velocities; those
    # velocities do not affect potential energy, but keeping them zero makes
    # the static-evaluation intent explicit.
    simulation = ReactiveSimulation(
        symbols=symbols,
        positions=coordinates,
        box_size=box_size,
        target_temperature=0.0,
        friction=0.0,
        device=device,
        dtype=torch.float64,
        random_seed=0,
        relax_on_start=False,
    )

    energy = float(simulation.potential_energy)

    parts = getattr(simulation, "_energy_parts", None)
    if parts is None:
        raise RuntimeError(
            f"{row['geometry_id']}: reactive_torch did not expose "
            "_energy_parts"
        )

    bond = float(torch.sum(parts["bond"]).detach().cpu())
    over = float(torch.sum(parts["over"]).detach().cpu())
    angle = float(torch.sum(parts["angle"]).detach().cpu())
    component_sum = bond + over + angle
    component_error = component_sum - energy

    forces = simulation.forces.detach().cpu().to(torch.float64)
    force_magnitudes = torch.linalg.norm(forces, dim=1)
    force_rms = float(torch.sqrt(torch.mean(force_magnitudes ** 2)))
    force_max = float(torch.max(force_magnitudes))

    values = [
        energy,
        bond,
        over,
        angle,
        component_error,
        force_rms,
        force_max,
    ]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(
            f"{row['geometry_id']}: non-finite ChemistryModel result"
        )

    # In float64 this should be many orders tighter than this.  The loose
    # threshold is simply an audit that the diagnostic decomposition still
    # means what this script thinks it means if reactive_torch changes later.
    if abs(component_error) > 1e-8:
        raise RuntimeError(
            f"{row['geometry_id']}: energy component sum differs from total "
            f"by {component_error:+.3e} eV"
        )

    reaction_coordinate = row.get("reaction_coordinate", {})

    return {
        "geometry_id": row["geometry_id"],
        "system": row["system"],
        "split": row["split"],
        "sample_kind": row["sample_kind"],
        "region": row["region"],
        "donor_distance_angstrom": reaction_coordinate.get(
            "donor_distance_angstrom", ""
        ),
        "transfer_distance_angstrom": reaction_coordinate.get(
            "transfer_distance_angstrom", ""
        ),
        "base_energy_eV": energy,
        "base_bond_energy_eV": bond,
        "base_overcoord_energy_eV": over,
        "base_angle_energy_eV": angle,
        "component_sum_error_eV": component_error,
        "base_force_rms_eV_per_angstrom": force_rms,
        "base_force_max_eV_per_angstrom": force_max,
    }


FIELDNAMES = [
    "geometry_id",
    "system",
    "split",
    "sample_kind",
    "region",
    "donor_distance_angstrom",
    "transfer_distance_angstrom",
    "base_energy_eV",
    "base_bond_energy_eV",
    "base_overcoord_energy_eV",
    "base_angle_energy_eV",
    "component_sum_error_eV",
    "base_force_rms_eV_per_angstrom",
    "base_force_max_eV_per_angstrom",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    path: Path,
    *,
    input_path: Path,
    input_sha256: str,
    output_path: Path,
    device: str,
    box_size: float,
    rows: list[dict],
    requested_limit: int | None,
) -> None:
    systems = {}
    for row in rows:
        systems[row["system"]] = systems.get(row["system"], 0) + 1

    metadata = {
        "input": str(input_path),
        "input_sha256": input_sha256,
        "output": str(output_path),
        "evaluated_geometry_count": len(rows),
        "system_counts": systems,
        "requested_limit": requested_limit,
        "engine": "reactive_torch.ReactiveSimulation",
        "device": device,
        "dtype": "torch.float64",
        "box_size_angstrom": box_size,
        "relax_on_start": False,
        "target_temperature_K": 0.0,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"geometry JSON (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"result CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"metadata JSON (default: {DEFAULT_METADATA})",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Torch device; cpu is the reproducible default",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=DEFAULT_BOX_SIZE,
        help=f"static periodic box size in A (default: {DEFAULT_BOX_SIZE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N geometries for a smoke test",
    )
    args = parser.parse_args()

    if args.box_size <= 0:
        raise SystemExit("--box-size must be positive")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    payload = load_payload(args.input)
    all_geometries = payload["geometries"]
    geometries = (
        all_geometries
        if args.limit is None
        else all_geometries[: args.limit]
    )

    input_hash = sha256_file(args.input)

    print(f"input      : {args.input}")
    print(f"sha256     : {input_hash}")
    print(f"geometries : {len(geometries)}")
    print(f"device     : {args.device}")
    print(f"dtype      : float64")
    print(f"box        : {args.box_size:.1f} A")
    print("")

    results = []
    total = len(geometries)

    for index, row in enumerate(geometries, start=1):
        result = evaluate_geometry(
            row,
            device=args.device,
            box_size=args.box_size,
        )
        results.append(result)

        if index == 1 or index % 25 == 0 or index == total:
            print(
                f"[{index:4d}/{total:4d}] "
                f"{result['geometry_id']:<32s} "
                f"E={result['base_energy_eV']:+.8f} eV"
            )

    write_csv(args.output, results)
    write_metadata(
        args.metadata,
        input_path=args.input,
        input_sha256=input_hash,
        output_path=args.output,
        device=args.device,
        box_size=args.box_size,
        rows=results,
        requested_limit=args.limit,
    )

    maximum_component_error = max(
        abs(row["component_sum_error_eV"]) for row in results
    )
    maximum_force = max(
        row["base_force_max_eV_per_angstrom"] for row in results
    )

    print("")
    print(f"wrote      : {args.output}")
    print(f"metadata   : {args.metadata}")
    print(f"rows       : {len(results)}")
    print(
        "max component audit error : "
        f"{maximum_component_error:.3e} eV"
    )
    print(
        "largest static force      : "
        f"{maximum_force:.6f} eV/A"
    )


if __name__ == "__main__":
    main()
