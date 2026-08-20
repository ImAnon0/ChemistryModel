"""Inspect exact Optimised-Valence energy/topology state for Grambow failures.

This is a read-only diagnostic subclass.  It composes the same production
energy methods and only retains detached tensors that production normally
discards after a force evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

import reactive as R
from batched_torch import BatchedReactiveSimulation
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


DEFAULT_SCORES = Path("research_data/benchmark/grambow_optimised.json")
DEFAULT_ENDPOINTS = Path("research_data/benchmark/grambow_endpoints.json")
DEFAULT_OUTPUT = Path("research_data/benchmark/diagnostics")
DEFAULT_IDS = (
    "rxn006559", "rxn011804", "rxn004353", "rxn000096",
    "rxn010742", "rxn011394", "rxn011223",
)
BOX_SIZE = 30.0


def _load_scores(path):
    text = path.read_text(encoding="utf-8-sig")
    if text.lstrip().startswith(("{", "[")):
        payload = json.loads(text)
        rows = payload if isinstance(payload, list) else payload.get("rows", payload.get("scores", []))
    else:
        rows = list(csv.DictReader(text.splitlines()))
    return {row["reaction_id"]: row for row in rows}


def _json_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class MicroscopeOptimisedValenceSimulation(OptimisedValenceStateBatchedSimulation):
    """Production equations plus detached, read-only endpoint diagnostics."""

    def _local_valence_membership(self, values):
        membership = super()._local_valence_membership(values)
        self._microscope_membership = membership.detach().clone()
        return membership

    def energy_per_atom(self, positions):
        base = BatchedReactiveSimulation.energy_per_atom(self, positions)
        try:
            hydrogen = self._hydrogen_state_correction(positions, base)
            valence = self._valence_topology_correction(positions)
            values = self._reactive_intermediates[1]
            live_parts = self._profile_energy_parts
            membership = getattr(self, "_microscope_membership", values["mask"])
            keep = (
                "neighbours", "mask", "distances", "taper", "topology_taper",
                "coordination", "topology_coordination", "valence", "order",
                "repulsive", "pair_depth", "pair_width", "shift",
            )
            captured_values = {
                key: values[key].detach().clone()
                for key in keep
            }
            attractive = values.get("state_attractive")
            if attractive is None:
                attractive = 2.0 * values["pair_depth"] * torch.exp(
                    -values["pair_width"] * values["shift"]
                )
            captured_values["pair_energy"] = (
                values["taper"] * (values["repulsive"] - attractive)
            ).detach().clone()
            captured_values["valence_membership"] = membership.detach().clone()
            captured_values["valence_topology_taper"] = (
                values["taper"] * membership
            ).detach().clone()
            self._microscope_capture = {
                "values": captured_values,
                "parts": {
                    "base_bond": live_parts["bond"].detach().clone(),
                    "base_overcoordination": live_parts["over"].detach().clone(),
                    "base_angle": live_parts["angle"].detach().clone(),
                    "h_state_correction": hydrogen.detach().clone(),
                    "valence_topology_correction": valence.detach().clone(),
                },
                "h_state_diagnostics": _json_safe(
                    getattr(self, "_h_component_diagnostics", {})
                ),
                "heavy_valence_diagnostics": _json_safe(
                    getattr(self, "_heavy_valence_diagnostics", {})
                ),
            }
        finally:
            self._reactive_intermediates = None
        return base + hydrogen + valence


def _reverse_slots(neighbours, mask):
    lookup = {}
    for centre in range(neighbours.shape[0]):
        for slot in range(neighbours.shape[1]):
            if float(mask[centre, slot]) <= 0.0:
                continue
            lookup[(centre, int(neighbours[centre, slot]))] = slot
    return lookup


def _endpoint_diagnostics(geometry, device):
    simulation = MicroscopeOptimisedValenceSimulation(
        boxes=[(geometry["symbols"], geometry["coordinates_angstrom"])],
        box_size=BOX_SIZE,
        target_temperature=0.0,
        friction=0.0,
        device=device,
        dtype=torch.float64,
        random_seed=0,
        relax_on_start=False,
    )
    capture = simulation._microscope_capture
    values = {key: value.detach().cpu() for key, value in capture["values"].items()}
    parts = {key: value.detach().cpu() for key, value in capture["parts"].items()}
    symbols = list(geometry["symbols"])

    component_totals = {key: float(value.sum()) for key, value in parts.items()}
    component_totals["base_total"] = sum(
        component_totals[key]
        for key in ("base_bond", "base_overcoordination", "base_angle")
    )
    component_totals["effective_angle_total"] = (
        component_totals["base_angle"]
        + component_totals["valence_topology_correction"]
    )
    component_totals["total"] = (
        component_totals["base_total"]
        + component_totals["h_state_correction"]
        + component_totals["valence_topology_correction"]
    )
    component_totals["reported_potential"] = float(simulation.potential_per_box[0])
    component_totals["composition_residual"] = (
        component_totals["total"] - component_totals["reported_potential"]
    )

    atoms = []
    for index, symbol in enumerate(symbols):
        atoms.append({
            "atom": index,
            "element": symbol,
            "radial_coordination": float(values["coordination"][index]),
            "base_topology_coordination": float(values["topology_coordination"][index]),
            "valence_topology_coordination": float(values["valence_topology_taper"][index].sum()),
            "elemental_valence": float(values["valence"][index]),
            "base_bond_eV": float(parts["base_bond"][index]),
            "base_overcoordination_eV": float(parts["base_overcoordination"][index]),
            "base_angle_eV": float(parts["base_angle"][index]),
            "h_state_correction_eV": float(parts["h_state_correction"][index]),
            "valence_topology_correction_eV": float(parts["valence_topology_correction"][index]),
        })

    neighbours = values["neighbours"]
    mask = values["mask"]
    reverse = _reverse_slots(neighbours, mask)
    contacts = []
    for first in range(len(symbols)):
        for slot in range(neighbours.shape[1]):
            if float(mask[first, slot]) <= 0.0:
                continue
            second = int(neighbours[first, slot])
            if first >= second or second >= len(symbols):
                continue
            reverse_slot = reverse.get((second, first))
            contacts.append({
                "atoms": [first, second],
                "elements": [symbols[first], symbols[second]],
                "pair": "-".join(sorted((symbols[first], symbols[second]))),
                "distance_angstrom": float(values["distances"][first, slot]),
                "radial_taper": float(values["taper"][first, slot]),
                "base_topology_taper": float(values["topology_taper"][first, slot]),
                "bond_order": float(values["order"][first, slot]),
                "radial_pair_energy_eV": float(values["pair_energy"][first, slot]),
                "membership_first_to_second": float(values["valence_membership"][first, slot]),
                "membership_second_to_first": (
                    float(values["valence_membership"][second, reverse_slot])
                    if reverse_slot is not None else None
                ),
                "valence_topology_first_to_second": float(
                    values["valence_topology_taper"][first, slot]
                ),
                "valence_topology_second_to_first": (
                    float(values["valence_topology_taper"][second, reverse_slot])
                    if reverse_slot is not None else None
                ),
            })
    contacts.sort(key=lambda row: (row["distance_angstrom"], row["atoms"]))
    coordination_gaps = [
        atom["radial_coordination"] - atom["valence_topology_coordination"]
        for atom in atoms
    ]
    radial_excesses = [
        max(atom["radial_coordination"] - atom["elemental_valence"], 0.0)
        for atom in atoms
    ]
    state_summary = {
        "base_overcoordination_eV": component_totals["base_overcoordination"],
        "heavy_base_overcoordination_eV": sum(
            atom["base_overcoordination_eV"]
            for atom in atoms if atom["element"] != "H"
        ),
        "hydrogen_base_overcoordination_eV": sum(
            atom["base_overcoordination_eV"]
            for atom in atoms if atom["element"] == "H"
        ),
        "base_angle_eV": component_totals["base_angle"],
        "valence_topology_correction_eV": component_totals["valence_topology_correction"],
        "coordination_gap_sum": sum(coordination_gaps),
        "max_coordination_gap": max(coordination_gaps, default=0.0),
        "radially_overcoordinated_atom_count": sum(value > 1e-9 for value in radial_excesses),
        "max_radial_valence_excess": max(radial_excesses, default=0.0),
        "suppressed_contact_count": sum(
            contact["radial_taper"] > 0.05
            and min(
                contact["membership_first_to_second"],
                contact["membership_second_to_first"]
                if contact["membership_second_to_first"] is not None else 1.0,
            ) < 0.10
            for contact in contacts
        ),
    }
    return {
        "geometry_id": geometry["geometry_id"],
        "components_eV": component_totals,
        "state_summary": state_summary,
        "atoms": atoms,
        "contacts": contacts,
        "h_state_diagnostics": capture["h_state_diagnostics"],
        "heavy_valence_diagnostics": capture["heavy_valence_diagnostics"],
    }


def _component_difference(target, source):
    return {
        key: target["components_eV"][key] - source["components_eV"][key]
        for key in (
            "base_bond", "base_overcoordination", "base_angle",
            "h_state_correction", "valence_topology_correction",
            "effective_angle_total", "total",
        )
    }


def _selected_ids(scores, explicit_ids, top):
    chosen = list(dict.fromkeys(explicit_ids))
    if top:
        barrier = sorted(scores.values(), key=lambda row: abs(float(row["barrier_error_eV"])), reverse=True)
        reaction = sorted(scores.values(), key=lambda row: abs(float(row["reaction_error_eV"])), reverse=True)
        for row in barrier[:top] + reaction[:top]:
            if row["reaction_id"] not in chosen:
                chosen.append(row["reaction_id"])
    return chosen


def microscope(scores_path, endpoints_path, reaction_ids, top, device, compact=False):
    scores = _load_scores(scores_path)
    endpoint_payload = json.loads(endpoints_path.read_text(encoding="utf-8"))
    endpoint_map = defaultdict(dict)
    for geometry in endpoint_payload["geometries"]:
        endpoint_map[geometry["reaction_id"]][geometry["region"]] = geometry

    selected = _selected_ids(scores, reaction_ids, top)
    reactions = []
    for position, reaction_id in enumerate(selected, 1):
        print(f"[{position:02d}/{len(selected):02d}] {reaction_id}")
        score = scores[reaction_id]
        regions = {
            region: _endpoint_diagnostics(endpoint_map[reaction_id][region], device)
            for region in ("reactant", "transition_state", "product")
        }
        barrier_parts = _component_difference(regions["transition_state"], regions["reactant"])
        reaction_parts = _component_difference(regions["product"], regions["reactant"])
        scored_barrier = float(score["model_barrier_eV"])
        scored_reaction = float(score["model_reaction_eV"])
        largest_barrier = sorted(
            ((key, value) for key, value in barrier_parts.items() if key != "total"),
            key=lambda item: abs(item[1]), reverse=True,
        )
        largest_reaction = sorted(
            ((key, value) for key, value in reaction_parts.items() if key != "total"),
            key=lambda item: abs(item[1]), reverse=True,
        )
        pressure_fields = tuple(regions["reactant"]["state_summary"])
        barrier_pressure = {
            key: (
                regions["transition_state"]["state_summary"][key]
                - regions["reactant"]["state_summary"][key]
            )
            for key in pressure_fields
        }
        reaction_pressure = {
            key: (
                regions["product"]["state_summary"][key]
                - regions["reactant"]["state_summary"][key]
            )
            for key in pressure_fields
        }
        endpoint_pressure = {
            region: dict(details["state_summary"])
            for region, details in regions.items()
        }
        if compact:
            for details in regions.values():
                details.pop("atoms", None)
                details.pop("contacts", None)
        reactions.append({
            "reaction_id": reaction_id,
            "atom_count": int(score["atom_count"]),
            "reference_barrier_eV": float(score["reference_barrier_eV"]),
            "scored_model_barrier_eV": scored_barrier,
            "barrier_error_eV": float(score["barrier_error_eV"]),
            "reference_reaction_eV": float(score["reference_reaction_eV"]),
            "scored_model_reaction_eV": scored_reaction,
            "reaction_error_eV": float(score["reaction_error_eV"]),
            "barrier_sign_agrees": bool(int(score["barrier_sign_agrees"])),
            "recomputed_model_barrier_eV": barrier_parts["total"],
            "recomputed_model_reaction_eV": reaction_parts["total"],
            "barrier_score_residual_eV": barrier_parts["total"] - scored_barrier,
            "reaction_score_residual_eV": reaction_parts["total"] - scored_reaction,
            "barrier_components_eV": barrier_parts,
            "reaction_components_eV": reaction_parts,
            "largest_barrier_components": [
                {"component": key, "delta_eV": value} for key, value in largest_barrier
            ],
            "largest_reaction_components": [
                {"component": key, "delta_eV": value} for key, value in largest_reaction
            ],
            "endpoint_pressure": endpoint_pressure,
            "barrier_pressure_delta": barrier_pressure,
            "reaction_pressure_delta": reaction_pressure,
            "regions": regions,
        })
    return {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "physics": "optimised-valence",
        "simulation_class": (
            f"{OptimisedValenceStateBatchedSimulation.__module__}."
            f"{OptimisedValenceStateBatchedSimulation.__name__}"
        ),
        "device": device,
        "dtype": "float64",
        "compact": bool(compact),
        "scores": str(scores_path),
        "endpoints": str(endpoints_path),
        "reaction_count": len(reactions),
        "reactions": reactions,
    }


def _write_outputs(result, output_dir, stem=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    if stem:
        json_path = output_dir / f"{stem}.json"
        csv_path = output_dir / f"{stem}_components.csv"
        report_path = output_dir / f"{stem}.md"
    else:
        json_path = output_dir / "grambow_reaction_microscope.json"
        csv_path = output_dir / "grambow_reaction_components.csv"
        report_path = output_dir / "grambow_reaction_microscope.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    component_rows = []
    for reaction in result["reactions"]:
        row = {
            key: reaction[key]
            for key in (
                "reaction_id", "atom_count", "reference_barrier_eV",
                "scored_model_barrier_eV", "barrier_error_eV",
                "reference_reaction_eV", "scored_model_reaction_eV",
                "reaction_error_eV", "barrier_sign_agrees",
                "barrier_score_residual_eV", "reaction_score_residual_eV",
            )
        }
        for kind in ("barrier", "reaction"):
            for component, value in reaction[f"{kind}_components_eV"].items():
                row[f"{kind}_{component}_eV"] = value
            for metric, value in reaction[f"{kind}_pressure_delta"].items():
                row[f"{kind}_pressure_{metric}"] = value
        for region, metrics in reaction["endpoint_pressure"].items():
            for metric, value in metrics.items():
                row[f"{region}_{metric}"] = value
        component_rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(component_rows[0]))
        writer.writeheader()
        writer.writerows(component_rows)

    lines = [
        "# Grambow per-reaction Optimised-Valence microscope", "",
        "All energy terms are evaluated by the current production equations; the subclass only retains detached diagnostics.", "",
        "| Reaction | Barrier error | Reaction error | Largest barrier term | Largest reaction term |",
        "| --- | --- | --- | --- | --- |",
    ]
    for reaction in result["reactions"]:
        barrier = reaction["largest_barrier_components"][0]
        reaction_term = reaction["largest_reaction_components"][0]
        lines.append(
            f"| {reaction['reaction_id']} | {reaction['barrier_error_eV']:+.3f} | "
            f"{reaction['reaction_error_eV']:+.3f} | "
            f"{barrier['component']} {barrier['delta_eV']:+.3f} | "
            f"{reaction_term['component']} {reaction_term['delta_eV']:+.3f} |"
        )
    lines.extend([""])
    for reaction in result["reactions"]:
        lines.extend([
            f"## {reaction['reaction_id']}", "",
            f"Barrier: reference {reaction['reference_barrier_eV']:+.6f} eV; "
            f"model {reaction['scored_model_barrier_eV']:+.6f} eV; "
            f"error {reaction['barrier_error_eV']:+.6f} eV.", "",
            f"Reaction: reference {reaction['reference_reaction_eV']:+.6f} eV; "
            f"model {reaction['scored_model_reaction_eV']:+.6f} eV; "
            f"error {reaction['reaction_error_eV']:+.6f} eV.", "",
            "| Component | TS - reactant | Product - reactant |",
            "| --- | --- | --- |",
        ])
        for component in (
            "base_bond", "base_overcoordination", "base_angle",
            "h_state_correction", "valence_topology_correction", "total",
        ):
            lines.append(
                f"| {component} | {reaction['barrier_components_eV'][component]:+.6f} | "
                f"{reaction['reaction_components_eV'][component]:+.6f} |"
            )
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, csv_path, report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--endpoints", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reaction", action="append", dest="reactions")
    parser.add_argument("--top", type=int, default=0, help="also inspect top N barrier and reaction errors")
    parser.add_argument("--device", default=None)
    parser.add_argument("--compact", action="store_true", help="omit per-atom/contact detail")
    parser.add_argument("--output-stem", default=None)
    arguments = parser.parse_args()
    device = arguments.device or ("cuda" if torch.cuda.is_available() else "cpu")
    reaction_ids = tuple(arguments.reactions) if arguments.reactions else DEFAULT_IDS
    result = microscope(
        arguments.scores, arguments.endpoints, reaction_ids, arguments.top,
        device, compact=arguments.compact,
    )
    paths = _write_outputs(result, arguments.output_dir, arguments.output_stem)
    print(f"inspected {result['reaction_count']} reactions on {device} float64")
    for reaction in result["reactions"]:
        print(
            f"  {reaction['reaction_id']}  barrier error={reaction['barrier_error_eV']:+8.3f}  "
            f"reaction error={reaction['reaction_error_eV']:+8.3f} eV"
        )
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
