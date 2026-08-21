"""Add shared bond-order channels to the frozen Grambow comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from research.benchmark.benchmark_reaction_barriers import group_by_reaction
from research.heavy_valence_bond_channels import SharedBondOrderChannelPrototype
from research.heavy_valence_state.compare_formulations import (
    _comparison,
    _distribution,
)
from research.heavy_valence_state.compare_grambow import evaluate


DEFAULT_ENDPOINTS = Path("research_data/benchmark/grambow_endpoints.json")
DEFAULT_BASELINE = Path(
    "research_data/benchmark/diagnostics/continuous_edge_comparison.csv"
)
DEFAULT_OUTPUT = Path(
    "research_data/benchmark/diagnostics/bond_channel_comparison.csv"
)
DEFAULT_SUMMARY = Path(
    "research_data/benchmark/diagnostics/bond_channel_comparison.json"
)


def compare(endpoints_path, baseline_path, box_size, device):
    payload = json.loads(endpoints_path.read_text(encoding="utf-8"))
    geometries = group_by_reaction(payload["geometries"])
    with baseline_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    failures = []
    output = []
    for index, raw in enumerate(source_rows, 1):
        row = {
            key: (value if key == "reaction_id" else float(value))
            for key, value in raw.items()
        }
        reaction_id = row["reaction_id"]
        try:
            endpoints = geometries[reaction_id]
            values = {
                region: evaluate(
                    SharedBondOrderChannelPrototype,
                    endpoints[region],
                    box_size,
                    device,
                )
                for region in ("reactant", "transition_state", "product")
            }
            barrier = values["transition_state"] - values["reactant"]
            reaction = values["product"] - values["reactant"]
            row.update({
                "bond_channels_barrier_eV": barrier,
                "bond_channels_barrier_error_eV": (
                    barrier - row["reference_barrier_eV"]
                ),
                "bond_channels_reaction_eV": reaction,
                "bond_channels_reaction_error_eV": (
                    reaction - row["reference_reaction_eV"]
                ),
                "bond_channels_barrier_sign_agrees": int(
                    (barrier > 0.0) == (row["reference_barrier_eV"] > 0.0)
                ),
            })
            output.append(row)
        except Exception as problem:
            failures.append(f"{reaction_id}: {type(problem).__name__}: {problem}")
        if index == 1 or index % 25 == 0:
            print(f"[{index:3d}] {reaction_id}", flush=True)

    barrier_errors = [row["bond_channels_barrier_error_eV"] for row in output]
    reaction_errors = [row["bond_channels_reaction_error_eV"] for row in output]
    summary = {
        "evaluated_reactions": len(output),
        "failures": failures,
        "bond_channels": {
            "barrier": _distribution(barrier_errors),
            "reaction": _distribution(reaction_errors),
            "barrier_sign_agreement_fraction": sum(
                row["bond_channels_barrier_sign_agrees"] for row in output
            ) / len(output),
            "versus_production": _comparison(output, "bond_channels", "production"),
            "versus_v0": _comparison(output, "bond_channels", "v0"),
            "versus_continuous_edge": _comparison(
                output, "bond_channels", "continuous_edge"
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
