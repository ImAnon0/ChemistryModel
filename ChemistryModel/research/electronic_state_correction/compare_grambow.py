"""Frozen, post-gate Grambow diagnostic for all electronic hypotheses.

The parameters are loaded from the pre-existing observable/non-water fit.  No
benchmark value is used to alter a coefficient.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.benchmark.benchmark_reaction_barriers import group_by_reaction
from research.electronic_state_correction import (
    CombinedElectronicStatePrototype,
    LocalElectronicDescriptorPrototype,
    MultipoleDensityPrototype,
    PolarisationResponsePrototype,
)
from research.heavy_valence_state.compare_grambow import evaluate


DEFAULT_INPUT = Path("research_data/benchmark/grambow_endpoints.json")
DEFAULT_BASELINE = Path(
    "research_data/benchmark/diagnostics/unified_bond_capacity_comparison.csv"
)
DEFAULT_OUTPUT = Path(
    "research_data/benchmark/diagnostics/electronic_state_grambow.csv"
)
DEFAULT_SUMMARY = Path(
    "research_data/benchmark/diagnostics/electronic_state_grambow.json"
)
MODELS = {
    "local_scalar": LocalElectronicDescriptorPrototype,
    "polarisation_vector": PolarisationResponsePrototype,
    "multipole_tensor": MultipoleDensityPrototype,
    "combined": CombinedElectronicStatePrototype,
}


def _metrics(errors, signs):
    values = np.asarray(errors, dtype=float)
    return {
        "count": int(len(values)),
        "mae_eV": float(np.abs(values).mean()),
        "rmse_eV": float(np.sqrt(np.mean(values * values))),
        "signed_mean_eV": float(values.mean()),
        "maximum_absolute_eV": float(np.abs(values).max()),
        "sign_agreement_fraction": float(np.mean(signs)),
    }


def compare(input_path, baseline_path, box_size, device):
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    reactions = group_by_reaction(payload["geometries"])
    with baseline_path.open("r", newline="", encoding="utf-8") as handle:
        baseline = {row["reaction_id"]: row for row in csv.DictReader(handle)}
    rows, failures = [], []
    for index, (reaction_id, endpoints) in enumerate(sorted(reactions.items())):
        old = baseline[reaction_id]
        row = {
            key: (value if key == "reaction_id" else float(value))
            for key, value in old.items()
        }
        reference_barrier = float(row["reference_barrier_eV"])
        reference_reaction = float(row["reference_reaction_eV"])
        for name, model in MODELS.items():
            try:
                energies = {
                    region: evaluate(model, endpoints[region], box_size, device)
                    for region in ("reactant", "transition_state", "product")
                }
                barrier = energies["transition_state"] - energies["reactant"]
                reaction = energies["product"] - energies["reactant"]
                row.update({
                    f"{name}_barrier_eV": barrier,
                    f"{name}_barrier_error_eV": barrier - reference_barrier,
                    f"{name}_reaction_eV": reaction,
                    f"{name}_reaction_error_eV": reaction - reference_reaction,
                    f"{name}_barrier_sign_agrees": int(
                        (barrier > 0) == (reference_barrier > 0)
                    ),
                })
            except Exception as exc:
                failures.append(
                    f"{reaction_id}/{name}: {type(exc).__name__}: {exc}"
                )
        rows.append(row)
        if (index + 1) % 25 == 0:
            print(f"Grambow: {index + 1}/200", flush=True)

    summary = {
        "scope": "frozen diagnostic only; all candidates already failed water promotion",
        "evaluated_reactions": len(rows),
        "failures": failures,
        "models": {},
    }
    for name in ("production", "unified_radial", *MODELS):
        summary["models"][name] = {
            "barrier": _metrics(
                [row[f"{name}_barrier_error_eV"] for row in rows],
                [row[f"{name}_barrier_sign_agrees"] for row in rows],
            ),
            "reaction": _metrics(
                [row[f"{name}_reaction_error_eV"] for row in rows],
                [1 for _ in rows],
            ),
        }
    for name in MODELS:
        summary["models"][name]["changed_from_unified"] = {
            "maximum_barrier_change_eV": max(abs(
                row[f"{name}_barrier_eV"] - row["unified_radial_barrier_eV"]
            ) for row in rows),
            "maximum_reaction_change_eV": max(abs(
                row[f"{name}_reaction_eV"] - row["unified_radial_reaction_eV"]
            ) for row in rows),
        }
        summary["models"][name]["worst_barriers"] = sorted(
            ({
                "reaction_id": row["reaction_id"],
                "error_eV": row[f"{name}_barrier_error_eV"],
                "unified_error_eV": row["unified_radial_barrier_error_eV"],
            } for row in rows),
            key=lambda item: abs(item["error_eV"]), reverse=True,
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
        raise SystemExit(f"{len(summary['failures'])} failures")


if __name__ == "__main__":
    main()
