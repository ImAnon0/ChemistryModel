"""Strict scientific-equivalence ruler for unified radial v1.

This module observes the existing implementation.  It neither reimplements
nor rearranges any energy equation.  A future canonical engine can be passed
as a candidate and compared against the frozen reference on identical inputs.
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from physics_provenance import physics_source_identity
from research.unified_bond_capacity import (
    LegacyUnifiedRadialReference,
    UnifiedBondCapacityEnergyPrototype,
)


MODEL_ID = "unified_radial_v1"
DEFAULT_FIXTURE = Path("tests/fixtures/unified_radial_v1_reference.json")
CPU_TOLERANCES = {
    "energy_atol_eV": 1e-10,
    "force_atol_eV_per_angstrom": 1e-8,
    "state_atol": 1e-10,
}
CUDA_TOLERANCES = {
    "energy_atol_eV": 2e-5,
    "force_atol_eV_per_angstrom": 2e-4,
    "state_atol": 2e-5,
}


def _shifted_positions(coordinates, box_size):
    positions = np.asarray(coordinates, dtype=float)
    positions = positions - positions.mean(axis=0)
    return positions + 0.5 * float(box_size)


def build_simulation(
    simulation_class, cases, *, device, dtype, box_size,
    capture_equivalence_state=True,
):
    boxes = [
        (list(case["symbols"]), _shifted_positions(case["positions"], box_size))
        for case in cases
    ]
    return simulation_class(
        boxes=boxes,
        box_size=float(box_size),
        time_step=0.1,
        target_temperature=0.0,
        friction=0.0,
        device=device,
        dtype=dtype,
        random_seed=17,
        relax_on_start=False,
        capture_equivalence_state=capture_equivalence_state,
    )


def _normalised_box_diagnostic(diagnostic, start):
    result = {
        "h_factors": diagnostic["h_factors"],
        "heavy_factors": diagnostic["heavy_factors"],
        "largest_h_states": diagnostic["largest_h_states"],
        "capacity": diagnostic["capacity"],
        "usage": diagnostic["usage"],
        "lambda_eV": diagnostic["lambda_eV"],
        "solver": diagnostic["solver"],
        "unified_pair_eV": diagnostic["unified_pair_eV"],
        "base_pair_eV": diagnostic["base_pair_eV"],
        "removed_over_eV": diagnostic["removed_over_eV"],
        "heavy_bond_orders": [],
        "h_state_probabilities": diagnostic.get("h_state_probabilities", []),
        "h_state_bases": diagnostic.get("h_state_bases", []),
        "heavy_state_probabilities": diagnostic.get(
            "heavy_state_probabilities", []
        ),
    }
    for item in diagnostic["heavy_bond_orders"]:
        result["heavy_bond_orders"].append({
            **item,
            "atoms": [int(atom) - start for atom in item["atoms"]],
        })
    return result


def snapshot(simulation, cases):
    """Capture all frozen observables needed for migration equivalence."""

    if getattr(simulation, "model_id", None) != MODEL_ID:
        raise ValueError(
            f"candidate reports {getattr(simulation, 'model_id', None)!r}; "
            f"expected {MODEL_ID!r}"
        )
    parts = getattr(simulation, "_unified_equivalence_energy_parts", None)
    membership = getattr(simulation, "_unified_equivalence_membership", None)
    diagnostics = getattr(simulation, "_unified_diagnostics", None)
    if parts is None or membership is None or diagnostics is None:
        raise RuntimeError(
            "implementation did not expose the Stage 2A equivalence state"
        )

    count = simulation.per_box
    neighbours = simulation.neighbours.detach().cpu().numpy()
    mask = simulation.neighbour_mask.detach().cpu().numpy()
    membership_np = membership.detach().cpu().numpy()
    forces = simulation.forces.detach().cpu().numpy()
    per_atom = parts["total"].detach().cpu().numpy()
    result = {}
    for box, case in enumerate(cases):
        start = box * count
        stop = start + count
        directed_membership = []
        for atom in range(start, stop):
            for slot in range(mask.shape[1]):
                if not mask[atom, slot]:
                    continue
                directed_membership.append({
                    "first": atom - start,
                    "second": int(neighbours[atom, slot]) - start,
                    "membership": float(membership_np[atom, slot]),
                })
        component_totals = {
            name: float(values[start:stop].sum().detach().cpu())
            for name, values in parts.items()
        }
        component_per_atom = {
            name: values[start:stop].detach().cpu().tolist()
            for name, values in parts.items()
        }
        result[case["name"]] = {
            "total_energy_eV": float(per_atom[start:stop].sum()),
            "per_box_energy_eV": float(per_atom[start:stop].sum()),
            "per_atom_energy_eV": per_atom[start:stop].tolist(),
            "energy_components_eV": component_totals,
            "energy_components_per_atom_eV": component_per_atom,
            "forces_eV_per_angstrom": forces[start:stop].tolist(),
            "membership": directed_membership,
            "state": _normalised_box_diagnostic(
                diagnostics["boxes"][box], start
            ),
        }
    return result


def evaluate_cases(
    simulation_class,
    cases,
    *,
    device="cpu",
    dtype=torch.float64,
    box_size=40.0,
    grouped=False,
):
    """Evaluate cases singly or in same-size grouped batches."""

    if not grouped:
        merged = {}
        for case in cases:
            simulation = build_simulation(
                simulation_class, [case], device=device, dtype=dtype,
                box_size=box_size,
            )
            merged.update(snapshot(simulation, [case]))
        return merged

    by_size = defaultdict(list)
    for case in cases:
        by_size[len(case["symbols"])].append(case)
    merged = {}
    for same_size in by_size.values():
        simulation = build_simulation(
            simulation_class, same_size, device=device, dtype=dtype,
            box_size=box_size,
        )
        merged.update(snapshot(simulation, same_size))
    return merged


def _tolerance_for(path, tolerances):
    lowered = path.lower()
    if "force" in lowered:
        return tolerances["force_atol_eV_per_angstrom"]
    if "energy" in lowered or lowered.endswith("_ev"):
        return tolerances["energy_atol_eV"]
    return tolerances["state_atol"]


def compare_values(expected, actual, tolerances, path="root"):
    """Return human-readable strict structural/numerical differences."""

    differences = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected mapping, got {type(actual).__name__}"]
        if set(expected) != set(actual):
            differences.append(
                f"{path}: keys differ expected={sorted(expected)} "
                f"actual={sorted(actual)}"
            )
        for key in sorted(set(expected) & set(actual)):
            differences.extend(compare_values(
                expected[key], actual[key], tolerances, f"{path}.{key}"
            ))
        return differences
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return [
                f"{path}: list shape differs expected={len(expected)} "
                f"actual={len(actual) if isinstance(actual, list) else 'not-list'}"
            ]
        for index, (left, right) in enumerate(zip(expected, actual)):
            differences.extend(compare_values(
                left, right, tolerances, f"{path}[{index}]"
            ))
        return differences
    if isinstance(expected, bool) or isinstance(expected, str) or expected is None:
        return [] if expected == actual else [
            f"{path}: expected {expected!r}, got {actual!r}"
        ]
    if isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)):
            return [f"{path}: expected number, got {actual!r}"]
        tolerance = _tolerance_for(path, tolerances)
        if not np.isfinite(expected) or not np.isfinite(actual):
            return [] if expected == actual else [
                f"{path}: non-finite mismatch {expected!r} vs {actual!r}"
            ]
        difference = abs(float(expected) - float(actual))
        return [] if difference <= tolerance else [
            f"{path}: |delta|={difference:.3e} exceeds {tolerance:.3e} "
            f"({expected!r} vs {actual!r})"
        ]
    return [] if expected == actual else [
        f"{path}: expected {expected!r}, got {actual!r}"
    ]


def load_fixture(path=DEFAULT_FIXTURE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_implementation_to_fixture(
    simulation_class=UnifiedBondCapacityEnergyPrototype,
    *,
    fixture_path=DEFAULT_FIXTURE,
):
    fixture = load_fixture(fixture_path)
    cases = fixture["cases"]
    tolerances = fixture["tolerances"]["cpu_float64"]
    single = evaluate_cases(simulation_class, cases, grouped=False)
    grouped = evaluate_cases(simulation_class, cases, grouped=True)
    differences = compare_values(
        fixture["reference"]["single"], single, tolerances, "single"
    )
    differences.extend(compare_values(
        fixture["reference"]["grouped"], grouped, tolerances, "grouped"
    ))
    return differences


def compare_implementations(
    reference_class,
    candidate_class,
    cases,
    *,
    device="cpu",
    dtype=torch.float64,
    tolerances=CPU_TOLERANCES,
):
    """Compare two implementations on identical single/grouped executions."""

    differences = []
    for grouped, label in ((False, "single"), (True, "grouped")):
        reference = evaluate_cases(
            reference_class, cases, device=device, dtype=dtype,
            grouped=grouped,
        )
        candidate = evaluate_cases(
            candidate_class, cases, device=device, dtype=dtype,
            grouped=grouped,
        )
        differences.extend(compare_values(
            reference, candidate, tolerances, label,
        ))
    return differences


def _load_class(specification):
    module_name, separator, class_name = specification.partition(":")
    if not separator:
        raise ValueError("candidate must use module.path:ClassName")
    return getattr(importlib.import_module(module_name), class_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--candidate",
        default=(
            "research.unified_bond_capacity:"
            "UnifiedBondCapacityEnergyPrototype"
        ),
        help="implementation to compare, as module.path:ClassName",
    )
    parser.add_argument(
        "--cuda", action="store_true",
        help="also compare reference and candidate live on CUDA when available",
    )
    args = parser.parse_args()
    candidate = _load_class(args.candidate)
    fixture = load_fixture(args.fixture)
    differences = compare_implementation_to_fixture(
        candidate, fixture_path=args.fixture
    )
    differences.extend(compare_implementations(
        LegacyUnifiedRadialReference,
        candidate,
        fixture["cases"],
    ))
    if args.cuda:
        if not torch.cuda.is_available():
            print("CUDA comparison: skipped (CUDA unavailable)")
        else:
            cases = fixture["cases"]
            differences.extend(compare_implementations(
                LegacyUnifiedRadialReference,
                candidate,
                cases,
                device="cuda",
                dtype=torch.float32,
                tolerances=fixture["tolerances"]["cuda_float32"],
            ))
    if differences:
        print("UNIFIED RADIAL EQUIVALENCE: FAIL")
        for difference in differences[:100]:
            print(f"  {difference}")
        if len(differences) > 100:
            print(f"  ... {len(differences) - 100} more")
        raise SystemExit(1)
    identity = physics_source_identity(candidate)
    print("UNIFIED RADIAL EQUIVALENCE: PASS")
    print(f"  model id    : {MODEL_ID}")
    print(f"  cases       : {len(fixture['cases'])}")
    print(f"  source hash : {identity['sha256']}")


if __name__ == "__main__":
    main()
