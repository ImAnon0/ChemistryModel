"""Compare current Optimised-Valence and the research heavy-energy prototype."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from research.benchmark.benchmark_reaction_barriers import (
    DEFAULT_BOX_SIZE,
    group_by_reaction,
    statistics,
)
from research.heavy_valence_state import HeavyValenceStateEnergyPrototype
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


DEFAULT_INPUT = Path("research_data/benchmark/grambow_endpoints.json")
DEFAULT_OUTPUT = Path(
    "research_data/benchmark/diagnostics/heavy_valence_state_comparison.csv"
)
DEFAULT_SUMMARY = Path(
    "research_data/benchmark/diagnostics/heavy_valence_state_comparison.json"
)


def evaluate(simulation_class, geometry, box_size, device):
    simulation = simulation_class(
        boxes=[(
            list(geometry["symbols"]),
            geometry["coordinates_angstrom"],
        )],
        box_size=box_size,
        target_temperature=0.0,
        friction=0.0,
        device=device,
        dtype=torch.float64,
        random_seed=0,
        relax_on_start=False,
    )
    energy = float(simulation.potential_per_box[0])
    if not math.isfinite(energy):
        raise ValueError(f"non-finite potential energy: {energy}")
    return energy


def metric_summary(rows, prefix):
    barrier_errors = [row[f"{prefix}_barrier_error_eV"] for row in rows]
    reaction_errors = [row[f"{prefix}_reaction_error_eV"] for row in rows]
    return {
        "barrier": statistics(barrier_errors),
        "reaction": statistics(reaction_errors),
        "barrier_sign_agreement_fraction": sum(
            row[f"{prefix}_barrier_sign_agrees"] for row in rows
        ) / len(rows),
    }


def compare(input_path, limit, box_size, device):
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    reactions = group_by_reaction(payload["geometries"])
    rows = []
    failures = []

    for reaction_id, endpoints in sorted(reactions.items()):
        if limit is not None and len(rows) >= limit:
            break
        try:
            missing = {
                "reactant", "transition_state", "product"
            } - set(endpoints)
            if missing:
                raise ValueError(f"missing endpoint(s): {sorted(missing)}")
            current = {
                region: evaluate(
                    OptimisedValenceStateBatchedSimulation,
                    endpoints[region], box_size, device,
                )
                for region in ("reactant", "transition_state", "product")
            }
            prototype = {
                region: evaluate(
                    HeavyValenceStateEnergyPrototype,
                    endpoints[region], box_size, device,
                )
                for region in ("reactant", "transition_state", "product")
            }
        except Exception as problem:
            failures.append(
                f"{reaction_id}: {type(problem).__name__}: {problem}"
            )
            continue

        reference = {
            region: float(endpoints[region]["reference_energy_eV"])
            for region in current
        }
        reference_barrier = reference["transition_state"] - reference["reactant"]
        reference_reaction = reference["product"] - reference["reactant"]
        row = {
            "reaction_id": reaction_id,
            "atom_count": len(endpoints["reactant"]["symbols"]),
            "reference_barrier_eV": reference_barrier,
            "reference_reaction_eV": reference_reaction,
        }
        for name, values in (("current", current), ("prototype", prototype)):
            barrier = values["transition_state"] - values["reactant"]
            reaction = values["product"] - values["reactant"]
            row.update({
                f"{name}_barrier_eV": barrier,
                f"{name}_barrier_error_eV": barrier - reference_barrier,
                f"{name}_reaction_eV": reaction,
                f"{name}_reaction_error_eV": reaction - reference_reaction,
                f"{name}_barrier_sign_agrees": int(
                    (barrier > 0.0) == (reference_barrier > 0.0)
                ),
            })
        row.update({
            "barrier_change_eV": (
                row["prototype_barrier_eV"] - row["current_barrier_eV"]
            ),
            "reaction_change_eV": (
                row["prototype_reaction_eV"] - row["current_reaction_eV"]
            ),
            "barrier_absolute_error_improvement_eV": (
                abs(row["current_barrier_error_eV"])
                - abs(row["prototype_barrier_error_eV"])
            ),
            "reaction_absolute_error_improvement_eV": (
                abs(row["current_reaction_error_eV"])
                - abs(row["prototype_reaction_error_eV"])
            ),
        })
        rows.append(row)

    summary = {
        "prototype": "research_mean_state_heavy_valence_energy_v0",
        "evaluated_reactions": len(rows),
        "failures": failures,
        "current": metric_summary(rows, "current"),
        "prototype_metrics": metric_summary(rows, "prototype"),
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--box-size", type=float, default=DEFAULT_BOX_SIZE)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    rows, summary = compare(
        args.input, args.limit, args.box_size, args.device
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.summary.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
