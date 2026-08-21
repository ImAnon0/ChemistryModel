"""Frozen Grambow comparison for heavy-valence formulations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from research.benchmark.benchmark_reaction_barriers import group_by_reaction, statistics
from research.heavy_valence_state import (
    HeavyValenceStateEnergyPrototype,
    JointEdgeStateHeavyValencePrototype,
    LocalFreeEnergyHeavyValencePrototype,
)
from research.heavy_valence_continuous_edge import (
    ContinuousSharedEdgeHeavyValencePrototype,
)
from research.heavy_valence_bond_channels import (
    ContinuousBondFreeEnergyPrototype,
    SharedBondOrderChannelPrototype,
)
from research.heavy_valence_state.compare_grambow import evaluate
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


MODELS = {
    "production": OptimisedValenceStateBatchedSimulation,
    "v0": HeavyValenceStateEnergyPrototype,
    "free_energy": LocalFreeEnergyHeavyValencePrototype,
    "joint_edge": JointEdgeStateHeavyValencePrototype,
    "continuous_edge": ContinuousSharedEdgeHeavyValencePrototype,
    "bond_channels": SharedBondOrderChannelPrototype,
    "bond_free_energy": ContinuousBondFreeEnergyPrototype,
}
DEFAULT_INPUT = Path("research_data/benchmark/grambow_endpoints.json")
DEFAULT_OUTPUT = Path(
    "research_data/benchmark/diagnostics/heavy_valence_formulations.csv"
)
DEFAULT_SUMMARY = Path(
    "research_data/benchmark/diagnostics/heavy_valence_formulations.json"
)


def _distribution(errors):
    absolute = np.abs(np.asarray(errors, dtype=float))
    result = statistics(list(errors))
    result["quantiles_absolute_eV"] = {
        str(percentile): float(np.quantile(absolute, percentile / 100.0))
        for percentile in (50, 75, 90, 95, 99, 100)
    }
    return result


def _comparison(rows, model, reference):
    result = {}
    for metric in ("barrier", "reaction"):
        candidate = np.abs(np.asarray([
            row[f"{model}_{metric}_error_eV"] for row in rows
        ]))
        baseline = np.abs(np.asarray([
            row[f"{reference}_{metric}_error_eV"] for row in rows
        ]))
        result[metric] = {
            "improved": int(np.count_nonzero(candidate < baseline - 1e-12)),
            "worsened": int(np.count_nonzero(candidate > baseline + 1e-12)),
            "unchanged": int(np.count_nonzero(np.abs(candidate - baseline) <= 1e-12)),
            "new_over_5_eV": int(np.count_nonzero((candidate > 5.0) & (baseline <= 5.0))),
            "new_over_10_eV": int(np.count_nonzero((candidate > 10.0) & (baseline <= 10.0))),
            "old_over_10_eV_removed": int(np.count_nonzero((candidate <= 10.0) & (baseline > 10.0))),
        }
    return result


def compare(input_path, limit, box_size, device):
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    reactions = group_by_reaction(payload["geometries"])
    rows = []
    failures = []
    for index, (reaction_id, endpoints) in enumerate(sorted(reactions.items())):
        if limit is not None and index >= limit:
            break
        try:
            reference = {
                region: float(endpoints[region]["reference_energy_eV"])
                for region in ("reactant", "transition_state", "product")
            }
            energies = {
                name: {
                    region: evaluate(cls, endpoints[region], box_size, device)
                    for region in ("reactant", "transition_state", "product")
                }
                for name, cls in MODELS.items()
            }
        except Exception as problem:
            failures.append(f"{reaction_id}: {type(problem).__name__}: {problem}")
            continue
        reference_barrier = reference["transition_state"] - reference["reactant"]
        reference_reaction = reference["product"] - reference["reactant"]
        row = {
            "reaction_id": reaction_id,
            "atom_count": len(endpoints["reactant"]["symbols"]),
            "reference_barrier_eV": reference_barrier,
            "reference_reaction_eV": reference_reaction,
        }
        for name, model in energies.items():
            barrier = model["transition_state"] - model["reactant"]
            reaction = model["product"] - model["reactant"]
            row.update({
                f"{name}_barrier_eV": barrier,
                f"{name}_barrier_error_eV": barrier - reference_barrier,
                f"{name}_reaction_eV": reaction,
                f"{name}_reaction_error_eV": reaction - reference_reaction,
                f"{name}_barrier_sign_agrees": int(
                    (barrier > 0.0) == (reference_barrier > 0.0)
                ),
            })
        rows.append(row)
        if len(rows) == 1 or len(rows) % 25 == 0:
            print(f"[{len(rows):3d}] {reaction_id}", flush=True)

    summary = {"evaluated_reactions": len(rows), "failures": failures, "models": {}}
    for name in MODELS:
        barrier = [row[f"{name}_barrier_error_eV"] for row in rows]
        reaction = [row[f"{name}_reaction_error_eV"] for row in rows]
        summary["models"][name] = {
            "barrier": _distribution(barrier),
            "reaction": _distribution(reaction),
            "barrier_sign_agreement_fraction": sum(
                row[f"{name}_barrier_sign_agrees"] for row in rows
            ) / len(rows),
            "versus_production": _comparison(rows, name, "production"),
            "versus_v0": _comparison(rows, name, "v0"),
        }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--box-size", type=float, default=30.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    rows, summary = compare(args.input, args.limit, args.box_size, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
