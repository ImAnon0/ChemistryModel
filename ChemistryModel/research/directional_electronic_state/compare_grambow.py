"""Evaluate the no-fit directional P2 probe on frozen Grambow endpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from research.benchmark.benchmark_reaction_barriers import group_by_reaction, statistics
from research.directional_electronic_state import (
    StateConditionedP2CouplingPrototype,
)
from research.heavy_valence_state.compare_formulations import _comparison
from research.heavy_valence_state.compare_grambow import evaluate


DEFAULT_INPUT = Path("research_data/benchmark/grambow_endpoints.json")
DEFAULT_BASELINE = Path(
    "research_data/benchmark/diagnostics/unified_bond_capacity_comparison.csv"
)
DEFAULT_OUTPUT = Path(
    "research_data/benchmark/diagnostics/directional_electronic_grambow.csv"
)
DEFAULT_SUMMARY = Path(
    "research_data/benchmark/diagnostics/directional_electronic_grambow.json"
)
COMPARATORS = ("production", "unified_radial")
CANDIDATE = "state_conditioned_p2"


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
                    StateConditionedP2CouplingPrototype,
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
            f"{CANDIDATE}_barrier_eV": barrier,
            f"{CANDIDATE}_barrier_error_eV": barrier - reference_barrier,
            f"{CANDIDATE}_reaction_eV": reaction,
            f"{CANDIDATE}_reaction_error_eV": reaction - reference_reaction,
            f"{CANDIDATE}_barrier_sign_agrees": int(
                (barrier > 0.0) == (reference_barrier > 0.0)
            ),
        })
        rows.append(row)
        if (index + 1) % 25 == 0:
            print(f"Grambow: {index + 1}/200", flush=True)

    summary = {"evaluated_reactions": len(rows), "failures": failures, "models": {}}
    for name in (*COMPARATORS, CANDIDATE):
        barrier = [row[f"{name}_barrier_error_eV"] for row in rows]
        reaction = [row[f"{name}_reaction_error_eV"] for row in rows]
        summary["models"][name] = {
            "barrier": _distribution(barrier),
            "reaction": _distribution(reaction),
            "barrier_sign_agreement_fraction": sum(
                row[f"{name}_barrier_sign_agrees"] for row in rows
            ) / len(rows),
        }
    summary["models"][CANDIDATE]["comparisons"] = {
        name: _comparison(rows, CANDIDATE, name) for name in COMPARATORS
    }
    summary["models"][CANDIDATE]["worst_barriers"] = sorted(
        ({
            "reaction_id": row["reaction_id"],
            "error_eV": row[f"{CANDIDATE}_barrier_error_eV"],
            "unified_radial_error_eV": row["unified_radial_barrier_error_eV"],
        } for row in rows),
        key=lambda row: abs(row["error_eV"]), reverse=True,
    )[:10]
    summary["models"][CANDIDATE]["worst_reactions"] = sorted(
        ({
            "reaction_id": row["reaction_id"],
            "error_eV": row[f"{CANDIDATE}_reaction_error_eV"],
            "unified_radial_error_eV": row["unified_radial_reaction_error_eV"],
        } for row in rows),
        key=lambda row: abs(row["error_eV"]), reverse=True,
    )[:10]
    summary["models"][CANDIDATE]["changed_from_radial"] = {
        "barriers_above_1e-10_eV": sum(
            abs(row[f"{CANDIDATE}_barrier_eV"] - row["unified_radial_barrier_eV"])
            > 1e-10 for row in rows
        ),
        "reactions_above_1e-10_eV": sum(
            abs(row[f"{CANDIDATE}_reaction_eV"] - row["unified_radial_reaction_eV"])
            > 1e-10 for row in rows
        ),
        "maximum_barrier_change_eV": max(
            abs(row[f"{CANDIDATE}_barrier_eV"] - row["unified_radial_barrier_eV"])
            for row in rows
        ),
        "maximum_reaction_change_eV": max(
            abs(row[f"{CANDIDATE}_reaction_eV"] - row["unified_radial_reaction_eV"])
            for row in rows
        ),
    }
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
    if failures := summary["failures"]:
        raise SystemExit(f"{len(failures)} Grambow failures")


if __name__ == "__main__":
    main()
