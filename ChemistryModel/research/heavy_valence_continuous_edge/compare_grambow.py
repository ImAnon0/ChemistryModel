"""Add the continuous-edge result to the frozen four-formulation benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from research.benchmark.benchmark_reaction_barriers import group_by_reaction
from research.heavy_valence_continuous_edge import (
    ContinuousSharedEdgeHeavyValencePrototype,
)
from research.heavy_valence_state.compare_formulations import (
    _comparison,
    _distribution,
)
from research.heavy_valence_state.compare_grambow import evaluate


DEFAULT_ENDPOINTS = Path("research_data/benchmark/grambow_endpoints.json")
DEFAULT_BASELINE = Path(
    "research_data/benchmark/diagnostics/heavy_valence_formulations.csv"
)
DEFAULT_OUTPUT = Path(
    "research_data/benchmark/diagnostics/continuous_edge_comparison.csv"
)
DEFAULT_SUMMARY = Path(
    "research_data/benchmark/diagnostics/continuous_edge_comparison.json"
)


def compare(endpoints_path, baseline_path, box_size, device):
    payload = json.loads(endpoints_path.read_text(encoding="utf-8"))
    geometries = group_by_reaction(payload["geometries"])
    with baseline_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    failures = []
    output = []
    for index, raw in enumerate(rows, 1):
        row = {
            key: (value if key == "reaction_id" else float(value))
            for key, value in raw.items()
        }
        reaction_id = row["reaction_id"]
        try:
            endpoints = geometries[reaction_id]
            values = {
                region: evaluate(
                    ContinuousSharedEdgeHeavyValencePrototype,
                    endpoints[region], box_size, device,
                )
                for region in ("reactant", "transition_state", "product")
            }
            barrier = values["transition_state"] - values["reactant"]
            reaction = values["product"] - values["reactant"]
            row.update({
                "continuous_edge_barrier_eV": barrier,
                "continuous_edge_barrier_error_eV": (
                    barrier - row["reference_barrier_eV"]
                ),
                "continuous_edge_reaction_eV": reaction,
                "continuous_edge_reaction_error_eV": (
                    reaction - row["reference_reaction_eV"]
                ),
                "continuous_edge_barrier_sign_agrees": int(
                    (barrier > 0.0) == (row["reference_barrier_eV"] > 0.0)
                ),
            })
            output.append(row)
        except Exception as problem:
            failures.append(f"{reaction_id}: {type(problem).__name__}: {problem}")
        if index == 1 or index % 25 == 0:
            print(f"[{index:3d}] {reaction_id}", flush=True)

    barrier_errors = [row["continuous_edge_barrier_error_eV"] for row in output]
    reaction_errors = [row["continuous_edge_reaction_error_eV"] for row in output]
    summary = {
        "evaluated_reactions": len(output),
        "failures": failures,
        "continuous_edge": {
            "barrier": _distribution(barrier_errors),
            "reaction": _distribution(reaction_errors),
            "barrier_sign_agreement_fraction": sum(
                row["continuous_edge_barrier_sign_agrees"] for row in output
            ) / len(output),
            "versus_production": _comparison(
                output, "continuous_edge", "production"
            ),
            "versus_v0": _comparison(output, "continuous_edge", "v0"),
            "versus_joint_edge": _comparison(
                output, "continuous_edge", "joint_edge"
            ),
        },
    }
    return output, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoints", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--box-size", type=float, default=30.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    rows, summary = compare(
        args.endpoints, args.baseline, args.box_size, args.device
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
