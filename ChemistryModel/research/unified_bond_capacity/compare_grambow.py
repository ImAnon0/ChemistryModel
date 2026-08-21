"""Add the gated unified model to the frozen 200-reaction comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from research.benchmark.benchmark_reaction_barriers import group_by_reaction, statistics
from research.heavy_valence_state.compare_formulations import _comparison
from research.heavy_valence_state.compare_grambow import evaluate
from research.unified_bond_capacity import UnifiedBondCapacityEnergyPrototype


DEFAULT_INPUT = Path("research_data/benchmark/grambow_endpoints.json")
DEFAULT_BASELINE = Path(
    "research_data/benchmark/diagnostics/bond_free_energy_comparison.csv"
)
DEFAULT_OUTPUT = Path(
    "research_data/benchmark/diagnostics/unified_bond_capacity_comparison.csv"
)
DEFAULT_SUMMARY = Path(
    "research_data/benchmark/diagnostics/unified_bond_capacity_comparison.json"
)
COMPARATORS = ("production", "v0", "continuous_edge", "bond_free_energy")


def _distribution(values):
    result = statistics(list(values))
    absolute = np.abs(np.asarray(values, dtype=float))
    result["quantiles_absolute_eV"] = {
        str(percentile): float(np.quantile(absolute, percentile / 100.0))
        for percentile in (50, 75, 90, 95, 99, 100)
    }
    return result


def compare(input_path, baseline_path, box_size, device):
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    reactions = group_by_reaction(payload["geometries"])
    with baseline_path.open("r", newline="", encoding="utf-8") as handle:
        baseline = {row["reaction_id"]: row for row in csv.DictReader(handle)}
    rows = []
    failures = []
    for index, (reaction_id, endpoints) in enumerate(sorted(reactions.items())):
        if reaction_id not in baseline:
            failures.append(f"{reaction_id}: absent from frozen baseline")
            continue
        try:
            energies = {
                region: evaluate(
                    UnifiedBondCapacityEnergyPrototype,
                    endpoints[region], box_size, device,
                )
                for region in ("reactant", "transition_state", "product")
            }
        except Exception as problem:
            failures.append(f"{reaction_id}: {type(problem).__name__}: {problem}")
            continue
        row = {}
        for key, value in baseline[reaction_id].items():
            if key == "reaction_id":
                row[key] = value
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
        reference_barrier = float(row["reference_barrier_eV"])
        reference_reaction = float(row["reference_reaction_eV"])
        barrier = energies["transition_state"] - energies["reactant"]
        reaction = energies["product"] - energies["reactant"]
        row.update({
            "unified_radial_barrier_eV": barrier,
            "unified_radial_barrier_error_eV": barrier - reference_barrier,
            "unified_radial_reaction_eV": reaction,
            "unified_radial_reaction_error_eV": reaction - reference_reaction,
            "unified_radial_barrier_sign_agrees": int(
                (barrier > 0.0) == (reference_barrier > 0.0)
            ),
        })
        rows.append(row)
        if (index + 1) % 25 == 0:
            print(f"Grambow: {index + 1}/200", flush=True)

    summary = {"evaluated_reactions": len(rows), "failures": failures, "models": {}}
    for name in (*COMPARATORS, "unified_radial"):
        barrier = [row[f"{name}_barrier_error_eV"] for row in rows]
        reaction = [row[f"{name}_reaction_error_eV"] for row in rows]
        summary["models"][name] = {
            "barrier": _distribution(barrier),
            "reaction": _distribution(reaction),
            "barrier_sign_agreement_fraction": sum(
                row[f"{name}_barrier_sign_agrees"] for row in rows
            ) / len(rows),
        }
    summary["models"]["unified_radial"]["comparisons"] = {
        name: _comparison(rows, "unified_radial", name) for name in COMPARATORS
    }
    summary["models"]["unified_radial"]["worst_barriers"] = sorted(
        ({
            "reaction_id": row["reaction_id"],
            "error_eV": row["unified_radial_barrier_error_eV"],
            "production_error_eV": row["production_barrier_error_eV"],
            "v0_error_eV": row["v0_barrier_error_eV"],
            "bond_free_energy_error_eV": row["bond_free_energy_barrier_error_eV"],
        } for row in rows),
        key=lambda row: abs(row["error_eV"]), reverse=True,
    )[:10]
    summary["models"]["unified_radial"]["worst_reactions"] = sorted(
        ({
            "reaction_id": row["reaction_id"],
            "error_eV": row["unified_radial_reaction_error_eV"],
            "production_error_eV": row["production_reaction_error_eV"],
            "v0_error_eV": row["v0_reaction_error_eV"],
            "bond_free_energy_error_eV": row["bond_free_energy_reaction_error_eV"],
        } for row in rows),
        key=lambda row: abs(row["error_eV"]), reverse=True,
    )[:10]
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--box-size", type=float, default=30.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    rows, summary = compare(args.input, args.baseline, args.box_size, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
