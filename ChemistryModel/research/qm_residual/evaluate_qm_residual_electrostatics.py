"""Evaluate QM residual geometries with unified radial + electrostatics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from dataclasses import replace
from pathlib import Path

import torch

from chemistry_engine.registry import build
import chemistry_engine.unified_radial  # noqa: F401 - registers model builder
from research.unified_bond_capacity import UnifiedBondCapacityEnergyPrototype

DEFAULT_INPUT = Path("research_data/qm_residual/geometries.json")
DEFAULT_OUTPUT = Path(
    "research_data/qm_residual/electrostatics_results.csv"
)
DEFAULT_METADATA = Path(
    "research_data/qm_residual/electrostatics_results.meta.json"
)
DEFAULT_BOX_SIZE = 30.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))

    geometries = payload.get("geometries")
    if not isinstance(geometries, list) or not geometries:
        raise ValueError("missing geometries")

    return payload


def evaluate_geometry(
    row: dict,
    *,
    device: str,
    box_size: float,
) -> dict:

    symbols = row["symbols"]
    coordinates = torch.tensor(
        row["coordinates_angstrom"],
        dtype=torch.float64,
    )

    if len(symbols) != len(coordinates):
        raise ValueError(
            f"{row['geometry_id']}: symbols/coordinates mismatch"
        )

    # Same convention as unified_radial_equivalence.py
    coordinates = coordinates - coordinates.mean(dim=0)
    coordinates = coordinates + box_size / 2.0

    simulation = UnifiedBondCapacityEnergyPrototype(
        boxes=[
            (
                list(symbols),
                coordinates,
            )
        ],
        box_size=float(box_size),
        time_step=0.1,
        target_temperature=0.0,
        friction=0.0,
        device=device,
        dtype=torch.float64,
        random_seed=0,
        relax_on_start=False,
    )

    spec = replace(
        simulation.chemistry_physics_spec,
        enabled_extensions=("electrostatics",),
    )

    simulation.chemistry_physics_spec = spec

    simulation.chemistry_engine = build(
        spec.model_id,
        simulation,
        spec,
    )

    positions = simulation.positions.detach().requires_grad_(True)
    context = simulation.build_interaction_context(positions)
    result = simulation.chemistry_engine.energy(context)

    if result is None:
        raise RuntimeError(
            "missing ChemistryEngine EnergyResult"
        )

    if not hasattr(result, "components"):
        raise RuntimeError(
            "EnergyResult has no components"
        )

    parts = result.components

    enabled_total = parts["total"].sum()
    electrostatic_tensor = parts["electrostatics"].sum()
    electrostatic_gradient, = torch.autograd.grad(
        electrostatic_tensor,
        positions,
        retain_graph=True,
    )
    enabled_gradient, = torch.autograd.grad(enabled_total, positions)
    enabled_forces = -enabled_gradient.detach().cpu().to(torch.float64)
    electrostatic_forces = -electrostatic_gradient.detach().cpu().to(torch.float64)
    disabled_forces = enabled_forces - electrostatic_forces

    energy = float(enabled_total.detach().cpu())

    def component(name):
        value = parts.get(name)
        if value is None:
            return 0.0
        return float(
            torch.sum(value)
            .detach()
            .cpu()
        )

    bond = component("base_bond")
    over = component("base_overcoordination")
    angle = component("base_angle")
    electrostatics = component("electrostatics")

    component_sum = (
        bond
        + over
        + angle
        + component("capacity_correction")
        + component("topology_correction")
        + electrostatics
    )

    component_error = component_sum - energy

    magnitudes = torch.linalg.norm(enabled_forces, dim=1)

    force_rms = float(
        torch.sqrt(torch.mean(magnitudes ** 2))
    )
    force_max = float(torch.max(magnitudes))
    disabled_magnitudes = torch.linalg.norm(disabled_forces, dim=1)
    electrostatic_magnitudes = torch.linalg.norm(electrostatic_forces, dim=1)
    disabled_energy = energy - electrostatics

    values = [
        energy,
        bond,
        over,
        angle,
        electrostatics,
        component_error,
        force_rms,
        force_max,
    ]

    if not all(math.isfinite(v) for v in values):
        raise RuntimeError(
            f"{row['geometry_id']}: non finite result"
        )

    reaction_coordinate = row.get(
        "reaction_coordinate",
        {},
    )

    return {
        "geometry_id": row["geometry_id"],
        "system": row["system"],
        "split": row["split"],
        "sample_kind": row["sample_kind"],
        "region": row["region"],
        "donor_distance_angstrom": reaction_coordinate.get(
            "donor_distance_angstrom",
            "",
        ),
        "transfer_distance_angstrom": reaction_coordinate.get(
            "transfer_distance_angstrom",
            "",
        ),
        "base_energy_eV": energy,
        "unified_disabled_energy_eV": disabled_energy,
        "electrostatics_enabled_energy_eV": energy,
        "base_bond_energy_eV": bond,
        "base_overcoord_energy_eV": over,
        "base_angle_energy_eV": angle,
        "electrostatics_energy_eV": electrostatics,
        "component_sum_error_eV": component_error,
        "base_force_rms_eV_per_angstrom": force_rms,
        "base_force_max_eV_per_angstrom": force_max,
        "unified_disabled_force_max_eV_per_angstrom": float(
            torch.max(disabled_magnitudes)
        ),
        "electrostatics_force_max_eV_per_angstrom": float(
            torch.max(electrostatic_magnitudes)
        ),
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
    "unified_disabled_energy_eV",
    "electrostatics_enabled_energy_eV",
    "base_bond_energy_eV",
    "base_overcoord_energy_eV",
    "base_angle_energy_eV",
    "electrostatics_energy_eV",
    "component_sum_error_eV",
    "base_force_rms_eV_per_angstrom",
    "base_force_max_eV_per_angstrom",
    "unified_disabled_force_max_eV_per_angstrom",
    "electrostatics_force_max_eV_per_angstrom",
]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path, *, args, input_hash, rows):
    systems = {}
    for row in rows:
        systems[row["system"]] = systems.get(row["system"], 0) + 1
    metadata = {
        "input": str(args.input),
        "input_sha256": input_hash,
        "output": str(args.output),
        "evaluated_geometry_count": len(rows),
        "system_counts": systems,
        "requested_limit": args.limit,
        "engine": "unified_radial_v1 + opt-in electrostatics",
        "comparison": "same unified-radial engine with extension disabled/enabled",
        "device": args.device,
        "dtype": "torch.float64",
        "box_size_angstrom": args.box_size,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--box-size",
        type=float,
        default=DEFAULT_BOX_SIZE,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    payload = load_payload(args.input)

    geometries = payload["geometries"]
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        geometries = geometries[: args.limit]
    if args.box_size <= 0:
        raise SystemExit("--box-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")

    input_hash = sha256_file(args.input)

    print(f"input      : {args.input}")
    print(f"sha256     : {input_hash}")
    print(f"geometries : {len(geometries)}")
    print(f"device     : {args.device}")
    print()

    results = []

    for index, row in enumerate(
        geometries,
        start=1,
    ):
        result = evaluate_geometry(
            row,
            device=args.device,
            box_size=args.box_size,
        )

        results.append(result)

        if index == 1 or index % 25 == 0:
            print(
                f"[{index:4d}/{len(geometries):4d}] "
                f"{row['geometry_id']:<35s}"
            )

    write_csv(
        args.output,
        results,
    )
    write_metadata(
        args.metadata,
        args=args,
        input_hash=input_hash,
        rows=results,
    )

    print()
    print(f"wrote : {args.output}")
    print(f"meta  : {args.metadata}")
    print(f"rows  : {len(results)}")


if __name__ == "__main__":
    main()
