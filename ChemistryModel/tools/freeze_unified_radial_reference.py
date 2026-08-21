"""Create the immutable Stage 2A unified-radial scientific fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import build_box
import bond_calibration
from physics_provenance import physics_source_identity
from research.unified_bond_capacity import UnifiedBondCapacityEnergyPrototype
from unified_radial_equivalence import (
    CPU_TOLERANCES,
    CUDA_TOLERANCES,
    DEFAULT_FIXTURE,
    MODEL_ID,
    evaluate_cases,
)


def _case(name, symbols, positions, category):
    return {
        "name": name,
        "category": category,
        "symbols": list(symbols),
        "positions": np.asarray(positions, dtype=float).tolist(),
    }


def reference_cases():
    cases = [
        _case("h2_equilibrium", ["H", "H"], [[-0.37072, 0, 0], [0.37072, 0, 0]], "small_system"),
        _case("h3_symmetric", ["H", "H", "H"], [[-0.75, 0, 0], [0, 0, 0], [0.75, 0, 0]], "small_system"),
        _case("h_plus_h2", ["H", "H", "H"], [[0, 0, 0], [0.74144, 0, 0], [2.05, 0, 0]], "small_system"),
        _case("water", ["O", "H", "H"], [[0, 0, 0], [0.9572, 0, 0], [-0.2399872, 0.9266272, 0]], "stable_molecule"),
        _case("methane", ["C", "H", "H", "H", "H"], [[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]], "stable_molecule"),
        _case("formaldehyde", ["C", "O", "H", "H"], [[0, 0, 0], [1.21, 0, 0], [-0.55, 0.9526, 0], [-0.55, -0.9526, 0]], "stable_molecule"),
    ]
    for name, builder in (
        ("ethane", lambda: bond_calibration.ethane_geometry()),
        ("hydroxylamine", bond_calibration.hydroxylamine_geometry),
        ("hydrogen_peroxide", bond_calibration.hydrogen_peroxide_geometry),
        ("ammonia", build_box.BUILDERS["NH3"]),
    ):
        symbols, positions = builder()
        cases.append(_case(name, symbols, positions, "stable_molecule"))
    cases.append(_case(
        "nitrogen", ["N", "N"], [[-0.55, 0, 0], [0.55, 0, 0]],
        "stable_molecule",
    ))

    dense = json.loads((
        ROOT / "research_data/qm_residual/dense_scan_geometries.json"
    ).read_text(encoding="utf-8"))["geometries"]
    water = [row for row in dense if row["system"] == "water"]
    row = water[len(water) // 2]
    cases.append(_case(
        "water_transfer_midpoint", row["symbols"],
        row["coordinates_angstrom"], "water_transfer",
    ))

    grambow = json.loads((
        ROOT / "research_data/benchmark/grambow_endpoints.json"
    ).read_text(encoding="utf-8"))["geometries"]
    selected = {"rxn002775", "rxn006559", "rxn011804"}
    for row in grambow:
        if row["reaction_id"] not in selected:
            continue
        cases.append(_case(
            row["geometry_id"].replace("/", "_"), row["symbols"],
            row["coordinates_angstrom"], "grambow_representative",
        ))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--replace", action="store_true",
        help="explicitly replace an existing scientific fixture",
    )
    args = parser.parse_args()
    if args.output.exists() and not args.replace:
        raise SystemExit(
            f"refusing to overwrite frozen fixture {args.output}; use --replace"
        )
    cases = reference_cases()
    payload = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "purpose": "Stage 2A frozen scientific migration ruler",
        "source_identity": physics_source_identity(
            UnifiedBondCapacityEnergyPrototype
        ),
        "tolerances": {
            "cpu_float64": CPU_TOLERANCES,
            "cuda_float32": CUDA_TOLERANCES,
        },
        "cases": cases,
        "reference": {
            "single": evaluate_cases(
                UnifiedBondCapacityEnergyPrototype, cases, grouped=False
            ),
            "grouped": evaluate_cases(
                UnifiedBondCapacityEnergyPrototype, cases, grouped=True
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
