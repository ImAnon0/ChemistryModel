"""Score ChemistryModel barrier heights and reaction energies against DFT.

This is the external yardstick the project currently lacks.  Every number in
``research_data/`` so far compares ChemistryModel against QM runs on geometries
ChemistryModel itself proposed.  This script instead evaluates a frozen public
benchmark that the model has never seen and that was produced independently.

Two quantities are scored per reaction:

    barrier_eV        = E(transition_state) - E(reactant)
    reaction_eV       = E(product)          - E(reactant)

Both are differences between endpoints with identical composition, so
ChemistryModel's absolute energy zero cancels exactly.  This is the same
alignment argument already documented in build_qm_residual_dataset.py, applied
to a dataset with published reference values.

No atoms are moved.  Each geometry is evaluated exactly as stored, matching
the convention in research/qm_residual/evaluate_qm_residual_base.py.

    python benchmark_reaction_barriers.py --limit 200
    python benchmark_reaction_barriers.py --physics optimised-valence

Coverage is a result, not an inconvenience.  A geometry that produces a
non-finite energy, or a reaction the model cannot evaluate at all, is recorded
and reported rather than dropped silently -- "evaluated 40% of reactions" is
itself a finding about where the model's domain ends.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

import reactive as R
from physics_provenance import physics_source_identity
from batched_torch import BatchedReactiveSimulation
from reactive_torch import ReactiveSimulation


DEFAULT_INPUT = Path("research_data/benchmark/transition1x_endpoints.json")
DEFAULT_OUTPUT = Path("research_data/benchmark/transition1x_scores.csv")
DEFAULT_SUMMARY = Path("research_data/benchmark/transition1x_summary.json")
DEFAULT_BOX_SIZE = 30.0


def _normalised_physics_name(physics):
    aliases = {
        None: "base",
        "": "base",
        "base": "base",
        "standard": "base",
        "reactive": "base",
        "h_state": "h-state",
        "h-state": "h-state",
        "high_fidelity": "high-fidelity",
        "high-fidelity": "high-fidelity",
        "optimised_valence": "optimised-valence",
        "optimised-valence": "optimised-valence",
        "unified_radial": "unified-radial",
        "unified-radial": "unified-radial",
        "unified_radial_v1": "unified-radial",
    }
    try:
        return aliases[physics]
    except KeyError as problem:
        raise ValueError(f"unknown benchmark physics: {physics!r}") from problem


def resolve_physics(physics=None, topology="current"):
    """Resolve an explicit benchmark backend without changing production."""
    name = _normalised_physics_name(physics)
    if name == "base":
        simulation_class = ReactiveSimulation
        h_state_active = False
        high_fidelity_active = False
    elif name == "h-state":
        from h_state_torch import HStateReferenceBatchedSimulation
        simulation_class = HStateReferenceBatchedSimulation
        h_state_active = True
        high_fidelity_active = False
    elif name == "high-fidelity":
        from high_fidelity_torch import HighFidelityBatchedReactiveSimulation
        simulation_class = HighFidelityBatchedReactiveSimulation
        h_state_active = False
        high_fidelity_active = True
    elif name == "optimised-valence":
        from valence_state_optimised_torch import (
            OptimisedValenceStateBatchedSimulation,
        )
        simulation_class = OptimisedValenceStateBatchedSimulation
        h_state_active = True
        high_fidelity_active = False
    else:
        from research.unified_bond_capacity import (
            UnifiedBondCapacityEnergyPrototype,
        )
        simulation_class = UnifiedBondCapacityEnergyPrototype
        h_state_active = True
        high_fidelity_active = False

    topology = str(topology)
    if topology not in ("current", "exclude-oo"):
        raise ValueError(f"unknown topology mode: {topology!r}")

    if topology == "exclude-oo":
        oxygen = int(R.ELEMENT_INDEX["O"])

        class OOTopologyExcludedSimulation(simulation_class):
            diagnostic_topology_exclusions = (("O", "O"),)

            def topology_taper(self, taper, centre_types, other_types, mask):
                current = super().topology_taper(
                    taper, centre_types, other_types, mask
                )
                oo_contact = (
                    (centre_types == oxygen)
                    & (other_types == oxygen)
                    & (mask > 0.0)
                )
                return torch.where(
                    oo_contact, torch.zeros_like(current), current
                )

        OOTopologyExcludedSimulation.__name__ = (
            f"OOExcluded{simulation_class.__name__}"
        )
        simulation_class = OOTopologyExcludedSimulation

    return {
        "name": name,
        "topology": topology,
        "class": simulation_class,
        "h_state_active": h_state_active,
        "high_fidelity_active": high_fidelity_active,
    }


def _git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def physics_fingerprint(physics=None, topology="current"):
    backend = resolve_physics(physics, topology)
    simulation_class = backend["class"]
    production_exclusions = [
        [str(first), str(second)]
        for first, second in getattr(R, "TOPOLOGY_ONLY_EXCLUDED_PAIRS", ())
    ]
    effective_exclusions = list(production_exclusions)
    if topology == "exclude-oo" and ["O", "O"] not in effective_exclusions:
        effective_exclusions.append(["O", "O"])

    constants = {}
    if backend["h_state_active"]:
        import h_state_torch as H
        constants.update({
            "h_state_model": H.H_STATE_MODEL_NAME,
            "h_state_revision": H.H_STATE_MODEL_REVISION,
            "h_state_mixing": H.H_STATE_MIXING,
        })
    if backend["high_fidelity_active"]:
        import high_fidelity_torch as HF
        constants.update({
            "high_fidelity_model": HF.HF_MODEL_NAME,
            "high_fidelity_revision": HF.HF_MODEL_REVISION,
            "h_transfer_state_mixing_fraction": (
                HF.H_TRANSFER_STATE_MIXING_FRACTION
            ),
            "h_transfer_gate_start": HF.H_TRANSFER_GATE_START,
            "h_transfer_gate_full": HF.H_TRANSFER_GATE_FULL,
        })

    source_identity = physics_source_identity(simulation_class)
    source_sha256 = source_identity["sha256"]
    source_files = source_identity["files"]
    physics_spec = (
        simulation_class.default_physics_spec()
        if hasattr(simulation_class, "default_physics_spec") else None
    )
    payload = {
        "requested_physics": backend["name"],
        "simulation_class": (
            f"{simulation_class.__module__}.{simulation_class.__name__}"
        ),
        "physics_model": getattr(
            simulation_class, "physics_model_name", "reactive_v1"
        ),
        "physics_model_revision": getattr(
            simulation_class, "physics_model_revision", None
        ),
        "physics_model_id": getattr(simulation_class, "model_id", None),
        "h_state_active": backend["h_state_active"],
        "high_fidelity_active": backend["high_fidelity_active"],
        "topology_mode": topology,
        "production_topology_exclusions": production_exclusions,
        "effective_topology_exclusions": effective_exclusions,
        "constants": constants,
        "git_revision": _git_revision(),
        "source_sha256": source_sha256,
        "source_files": source_files,
        "source_algorithm": source_identity["algorithm"],
    }
    if physics_spec is not None:
        payload.update({
            "parameter_sha256": physics_spec.parameter_sha256,
            "enabled_terms": list(physics_spec.enabled_terms),
            "capacity_solver": physics_spec.capacity.solver,
            "geometry_convention": physics_spec.geometry.convention,
        })
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint_sha256"] = hashlib.sha256(
        encoded.encode("utf-8")
    ).hexdigest()
    return payload


def evaluate_geometry(row, box_size, device, physics=None, topology="current"):
    """Return the ChemistryModel potential energy for one stored geometry."""
    simulation_class = resolve_physics(physics, topology)["class"]
    common = dict(
        box_size=box_size, target_temperature=0.0, friction=0.0,
        device=device, dtype=torch.float64, random_seed=0,
        relax_on_start=False,
    )
    if issubclass(simulation_class, BatchedReactiveSimulation):
        simulation = simulation_class(
            boxes=[(
                list(row["symbols"]), row["coordinates_angstrom"],
            )],
            **common,
        )
        energy = float(simulation.potential_per_box[0])
    else:
        simulation = simulation_class(
            symbols=list(row["symbols"]),
            positions=torch.tensor(
                row["coordinates_angstrom"], dtype=torch.float64
            ),
            **common,
        )
        energy = float(simulation.potential_energy)
    if not math.isfinite(energy):
        raise ValueError(f"non-finite potential energy: {energy}")
    return energy


def group_by_reaction(geometries):
    reactions = defaultdict(dict)
    for row in geometries:
        reactions[row["reaction_id"]][row["region"]] = row
    return reactions


def score(input_path, *, limit, box_size, device, physics=None,
          topology="current"):
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    reactions = group_by_reaction(payload["geometries"])

    rows = []
    failures = []
    attempted = 0

    for reaction_id, endpoints in sorted(reactions.items()):
        if limit is not None and attempted >= limit:
            break
        attempted += 1

        missing = [
            region for region in ("reactant", "transition_state", "product")
            if region not in endpoints
        ]
        if missing:
            failures.append(f"{reaction_id}: missing {', '.join(missing)}")
            continue

        try:
            model = {
                region: evaluate_geometry(
                    endpoints[region], box_size, device, physics, topology
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
            for region in model
        }

        model_barrier = model["transition_state"] - model["reactant"]
        model_reaction = model["product"] - model["reactant"]
        reference_barrier = (
            reference["transition_state"] - reference["reactant"]
        )
        reference_reaction = reference["product"] - reference["reactant"]

        rows.append({
            "reaction_id": reaction_id,
            "atom_count": len(endpoints["reactant"]["symbols"]),
            "model_barrier_eV": model_barrier,
            "reference_barrier_eV": reference_barrier,
            "barrier_error_eV": model_barrier - reference_barrier,
            "model_reaction_eV": model_reaction,
            "reference_reaction_eV": reference_reaction,
            "reaction_error_eV": model_reaction - reference_reaction,
            "barrier_sign_agrees": int(
                (model_barrier > 0) == (reference_barrier > 0)
            ),
        })

    return rows, failures, attempted


def compare_rows(rows_a, rows_b):
    by_id_a = {row["reaction_id"]: row for row in rows_a}
    by_id_b = {row["reaction_id"]: row for row in rows_b}
    compared = []
    for reaction_id in sorted(set(by_id_a) & set(by_id_b)):
        old = by_id_a[reaction_id]
        new = by_id_b[reaction_id]
        compared.append({
            "reaction_id": reaction_id,
            "old_barrier_eV": old["model_barrier_eV"],
            "new_barrier_eV": new["model_barrier_eV"],
            "barrier_delta_eV": (
                new["model_barrier_eV"] - old["model_barrier_eV"]
            ),
            "old_reaction_eV": old["model_reaction_eV"],
            "new_reaction_eV": new["model_reaction_eV"],
            "reaction_delta_eV": (
                new["model_reaction_eV"] - old["model_reaction_eV"]
            ),
        })
    return compared


def statistics(values):
    if not values:
        return {"count": 0}
    absolute = [abs(value) for value in values]
    ordered = sorted(absolute)
    middle = len(ordered) // 2
    return {
        "count": len(values),
        "mae_eV": sum(absolute) / len(absolute),
        "rmse_eV": math.sqrt(sum(value ** 2 for value in values) / len(values)),
        "median_absolute_eV": (
            ordered[middle]
            if len(ordered) % 2
            else 0.5 * (ordered[middle - 1] + ordered[middle])
        ),
        "max_absolute_eV": max(absolute),
        "signed_mean_eV": sum(values) / len(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--box-size", type=float, default=DEFAULT_BOX_SIZE)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--physics", default="base",
        choices=(
            "base", "h-state", "high-fidelity", "optimised-valence",
            "unified-radial",
        ),
        help="explicit ChemistryModel backend (default preserves the old base benchmark)",
    )
    parser.add_argument(
        "--topology", default="current", choices=("current", "exclude-oo"),
        help="topology policy; exclude-oo is a diagnostic only",
    )
    parser.add_argument(
        "--compare-physics", default=None,
        choices=(
            "base", "h-state", "high-fidelity", "optimised-valence",
            "unified-radial",
        ),
        help="run an A/B comparison against this second backend",
    )
    parser.add_argument(
        "--compare-topology", default=None,
        choices=("current", "exclude-oo"),
        help="run an A/B comparison against this second topology policy",
    )
    parser.add_argument(
        "--comparison-output", type=Path, default=None,
        help="per-reaction A/B CSV (default: beside --output)",
    )
    arguments = parser.parse_args()

    if not arguments.input.is_file():
        parser.error(
            f"{arguments.input} not found. Run extract_transition1x_endpoints.py first."
        )

    device = arguments.device or ("cuda" if torch.cuda.is_available() else "cpu")

    rows, failures, attempted = score(
        arguments.input,
        limit=arguments.limit,
        box_size=arguments.box_size,
        device=device,
        physics=arguments.physics,
        topology=arguments.topology,
    )

    fingerprint = physics_fingerprint(
        arguments.physics, arguments.topology
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with arguments.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    barrier = statistics([row["barrier_error_eV"] for row in rows])
    reaction = statistics([row["reaction_error_eV"] for row in rows])
    summary = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input": str(arguments.input),
        "reactions_attempted": attempted,
        "reactions_scored": len(rows),
        "reactions_failed": len(failures),
        "coverage_fraction": (len(rows) / attempted) if attempted else 0.0,
        "barrier_height": barrier,
        "reaction_energy": reaction,
        "barrier_sign_agreement": (
            sum(row["barrier_sign_agrees"] for row in rows) / len(rows)
            if rows else 0.0
        ),
        "box_size_angstrom": arguments.box_size,
        "device": device,
        "physics": arguments.physics,
        "topology": arguments.topology,
        "physics_fingerprint": fingerprint,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "failures": failures[:100],
    }

    comparison = None
    compare_requested = (
        arguments.compare_physics is not None
        or arguments.compare_topology is not None
    )
    if compare_requested:
        compare_physics = arguments.compare_physics or arguments.physics
        compare_topology = arguments.compare_topology or arguments.topology
        rows_b, failures_b, attempted_b = score(
            arguments.input,
            limit=arguments.limit,
            box_size=arguments.box_size,
            device=device,
            physics=compare_physics,
            topology=compare_topology,
        )
        compared = compare_rows(rows, rows_b)
        comparison_path = arguments.comparison_output or arguments.output.with_name(
            arguments.output.stem + "_comparison.csv"
        )
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        if compared:
            with comparison_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(compared[0]))
                writer.writeheader()
                writer.writerows(compared)
        barrier_deltas = [row["barrier_delta_eV"] for row in compared]
        reaction_deltas = [row["reaction_delta_eV"] for row in compared]
        largest = sorted(
            compared,
            key=lambda row: max(
                abs(row["barrier_delta_eV"]),
                abs(row["reaction_delta_eV"]),
            ),
            reverse=True,
        )[:20]
        comparison = {
            "physics_b": compare_physics,
            "topology_b": compare_topology,
            "physics_fingerprint_b": physics_fingerprint(
                compare_physics, compare_topology
            ),
            "reactions_attempted_b": attempted_b,
            "reactions_scored_b": len(rows_b),
            "failures_b": failures_b[:100],
            "reactions_compared": len(compared),
            "barrier_delta": statistics(barrier_deltas),
            "reaction_delta": statistics(reaction_deltas),
            "all_deltas_zero": all(
                row["barrier_delta_eV"] == 0.0
                and row["reaction_delta_eV"] == 0.0
                for row in compared
            ),
            "largest_changes": largest,
            "comparison_output": str(comparison_path),
        }
        summary["comparison"] = comparison
    arguments.summary.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"scored {len(rows)}/{attempted} reactions "
          f"({summary['coverage_fraction']:.1%} coverage)")
    print(
        "physics  "
        f"{fingerprint['simulation_class']}  "
        f"model={fingerprint['physics_model']}  "
        f"revision={fingerprint['physics_model_revision']}"
    )
    print(
        "topology "
        f"{fingerprint['topology_mode']}  "
        f"exclusions={fingerprint['effective_topology_exclusions']}"
    )
    print(f"fingerprint {fingerprint['fingerprint_sha256']}")
    if rows:
        print(f"barrier  MAE {barrier['mae_eV']:.3f} eV  "
              f"RMSE {barrier['rmse_eV']:.3f} eV  "
              f"signed mean {barrier['signed_mean_eV']:+.3f} eV")
        print(f"reaction MAE {reaction['mae_eV']:.3f} eV  "
              f"RMSE {reaction['rmse_eV']:.3f} eV  "
              f"signed mean {reaction['signed_mean_eV']:+.3f} eV")
        print(f"barrier sign agreement {summary['barrier_sign_agreement']:.1%}")
    if failures:
        print(f"\n{len(failures)} failures; first few:")
        for line in failures[:5]:
            print(f"  {line}")
    if comparison is not None:
        print()
        print(
            f"A/B compared {comparison['reactions_compared']} reactions; "
            f"all deltas zero: {comparison['all_deltas_zero']}"
        )
        if comparison["reactions_compared"]:
            print(
                "mean absolute barrier delta "
                f"{comparison['barrier_delta']['mae_eV']:.6f} eV"
            )
            print(
                "mean absolute reaction delta "
                f"{comparison['reaction_delta']['mae_eV']:.6f} eV"
            )
            for row in comparison["largest_changes"][:5]:
                print(
                    f"  {row['reaction_id']}  "
                    f"barrier {row['barrier_delta_eV']:+.6f} eV  "
                    f"reaction {row['reaction_delta_eV']:+.6f} eV"
                )


if __name__ == "__main__":
    main()
