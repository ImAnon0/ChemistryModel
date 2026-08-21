"""Summarise continuous-edge benchmark tails and required microscope cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_INPUT = Path(
    "research_data/benchmark/diagnostics/continuous_edge_comparison.csv"
)
DEFAULT_OUTPUT = Path(
    "research_data/benchmark/diagnostics/continuous_edge_analysis.json"
)
REQUIRED = (
    "rxn006559", "rxn011804", "rxn004353", "rxn000096", "rxn010742",
    "rxn000105",
)


def analyse(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {}
    for row in rows:
        numeric[row["reaction_id"]] = {
            name: float(value) if name != "reaction_id" else value
            for name, value in row.items()
            if name == "reaction_id" or value not in (None, "")
        }

    continuous = sorted(
        numeric.values(),
        key=lambda row: abs(row["continuous_edge_barrier_error_eV"]),
        reverse=True,
    )
    result = {
        "required_microscopes": {
            reaction: numeric[reaction] for reaction in REQUIRED
        },
        "continuous_edge_barrier_over_10_eV": [
            row["reaction_id"] for row in continuous
            if abs(row["continuous_edge_barrier_error_eV"]) > 10.0
        ],
        "continuous_edge_new_over_10_eV_vs_production": [
            row["reaction_id"] for row in continuous
            if abs(row["continuous_edge_barrier_error_eV"]) > 10.0
            and abs(row["production_barrier_error_eV"]) <= 10.0
        ],
        "continuous_edge_new_over_10_eV_vs_v0": [
            row["reaction_id"] for row in continuous
            if abs(row["continuous_edge_barrier_error_eV"]) > 10.0
            and abs(row["v0_barrier_error_eV"]) <= 10.0
        ],
        "worst_continuous_edge_barriers": continuous[:20],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyse(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
