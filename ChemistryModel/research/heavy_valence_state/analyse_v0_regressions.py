"""Microscope the reactions whose absolute barrier error worsens under v0."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

import reactive as R
from batched_torch import BatchedReactiveSimulation
from research.benchmark.benchmark_reaction_barriers import group_by_reaction
from research.heavy_valence_state import HeavyValenceStateEnergyPrototype


DEFAULT_SCORES = Path(
    "research_data/benchmark/diagnostics/heavy_valence_formulations.csv"
)
DEFAULT_ENDPOINTS = Path("research_data/benchmark/grambow_endpoints.json")
DEFAULT_CSV = Path(
    "research_data/benchmark/diagnostics/heavy_valence_v0_regressions.csv"
)
DEFAULT_JSON = Path(
    "research_data/benchmark/diagnostics/heavy_valence_v0_regressions.json"
)


class _CaptureV0(HeavyValenceStateEnergyPrototype):
    def energy_per_atom(self, positions):
        base = BatchedReactiveSimulation.energy_per_atom(self, positions)
        try:
            hydrogen = self._hydrogen_state_correction(positions, base)
            topology = self._valence_topology_correction(positions)
            values = self._reactive_intermediates[1]
            attractive = values.get("state_attractive")
            if attractive is None:
                attractive = 2.0 * values["pair_depth"] * torch.exp(
                    -values["pair_width"] * values["shift"]
                )
            self._forensic_capture = {
                key: values[key].detach().cpu().clone()
                for key in ("neighbours", "mask", "distances", "taper")
            }
            self._forensic_capture["attractive"] = attractive.detach().cpu().clone()
            self._forensic_capture["membership"] = (
                self._heavy_valence_energy_diagnostics["membership"]
                .detach().cpu().clone()
            )
            self._forensic_capture["diagnostics"] = {
                key: value.detach().cpu().clone()
                for key, value in self._heavy_valence_energy_diagnostics.items()
                if isinstance(value, torch.Tensor)
            }
        finally:
            self._reactive_intermediates = None
        return base + hydrogen + topology


def _severity(delta):
    if delta <= 0.5:
        return "minor"
    if delta <= 2.0:
        return "moderate"
    if delta <= 5.0:
        return "major"
    return "new catastrophic/outlier"


def _endpoint(geometry, device):
    simulation = _CaptureV0(
        boxes=[(geometry["symbols"], geometry["coordinates_angstrom"])],
        box_size=30.0,
        target_temperature=0.0,
        friction=0.0,
        device=device,
        dtype=torch.float64,
        random_seed=0,
        relax_on_start=False,
    )
    captured = simulation._forensic_capture
    symbols = list(geometry["symbols"])
    hydrogen = "H"
    reverse = {}
    for atom in range(len(symbols)):
        for slot in range(captured["neighbours"].shape[1]):
            if float(captured["mask"][atom, slot]) <= 0.0:
                continue
            reverse[(atom, int(captured["neighbours"][atom, slot]))] = slot
    contacts = {}
    for (first, second), slot in reverse.items():
        if first >= second or symbols[first] == hydrogen or symbols[second] == hydrogen:
            continue
        reverse_slot = reverse.get((second, first))
        if reverse_slot is None:
            continue
        first_membership = float(captured["membership"][first, slot])
        second_membership = float(captured["membership"][second, reverse_slot])
        attraction = 0.5 * (
            float(captured["taper"][first, slot] * captured["attractive"][first, slot])
            + float(captured["taper"][second, reverse_slot] * captured["attractive"][second, reverse_slot])
        )
        contacts[f"{first}-{second}"] = {
            "atoms": [first, second],
            "pair": "-".join(sorted((symbols[first], symbols[second]))),
            "distance_angstrom": float(captured["distances"][first, slot]),
            "taper": float(captured["taper"][first, slot]),
            "membership_first": first_membership,
            "membership_second": second_membership,
            "membership_asymmetry": abs(first_membership - second_membership),
            "both_endpoints_competing": (
                first_membership < 0.999 and second_membership < 0.999
            ),
            "attraction_magnitude_eV": attraction,
            "v0_rejected_attraction_eV": attraction * (
                1.0 - 0.5 * (first_membership + second_membership)
            ),
            "bonded_for_family": (
                float(captured["taper"][first, slot])
                * 0.5 * (first_membership + second_membership) > 0.5
            ),
        }
    diagnostics = captured["diagnostics"]
    heavy_indices = [
        atom for atom, symbol in enumerate(symbols) if symbol != hydrogen
    ]
    heavy_competing = 0
    for atom, symbol in enumerate(symbols):
        if symbol == hydrogen:
            continue
        active = captured["mask"][atom] > 0.0
        if bool(torch.any(captured["membership"][atom][active] < 0.999)):
            heavy_competing += 1
    atoms = []
    for atom, symbol in enumerate(symbols):
        atoms.append({
            "atom": atom,
            "element": symbol,
            "raw_radial_coordination": float(captured["taper"][atom].sum()),
            "effective_coordination": float(diagnostics["effective_coordination"][atom]),
            "original_over_eV": float(diagnostics["original_over_per_atom"][atom]),
            "replacement_over_eV": float(diagnostics["prototype_over_per_atom"][atom]),
            "rejected_attraction_eV": float(diagnostics["rejected_attraction_per_atom"][atom]),
        })
    return {
        "energy_eV": float(simulation.potential_per_box[0]),
        "contacts": contacts,
        "atoms": atoms,
        "heavy_competing_centres": heavy_competing,
        "maximum_membership_asymmetry": max(
            (row["membership_asymmetry"] for row in contacts.values()), default=0.0
        ),
        "both_endpoints_competing_edges": sum(
            row["both_endpoints_competing"] for row in contacts.values()
        ),
        "original_heavy_over_eV": float(
            diagnostics["original_over_per_atom"][heavy_indices].sum()
        ),
        "replacement_heavy_over_eV": float(
            diagnostics["prototype_over_per_atom"][heavy_indices].sum()
        ),
        "rejected_attraction_eV": float(diagnostics["rejected_attraction_per_atom"].sum()),
    }


def _bond_changes(reactant, product):
    old = {
        key: row for key, row in reactant["contacts"].items()
        if row["bonded_for_family"]
    }
    new = {
        key: row for key, row in product["contacts"].items()
        if row["bonded_for_family"]
    }
    broken = [old[key]["pair"] for key in sorted(set(old) - set(new))]
    formed = [new[key]["pair"] for key in sorted(set(new) - set(old))]
    pieces = []
    if broken:
        pieces.append("break " + "+".join(sorted(broken)))
    if formed:
        pieces.append("form " + "+".join(sorted(formed)))
    return broken, formed, "; ".join(pieces) if pieces else "no >0.5 topology change"


def analyse(scores_path, endpoints_path, device):
    with scores_path.open("r", newline="", encoding="utf-8") as handle:
        scores = list(csv.DictReader(handle))
    regressions = [
        row for row in scores
        if abs(float(row["v0_barrier_error_eV"]))
        > abs(float(row["production_barrier_error_eV"])) + 1e-12
    ]
    geometries = group_by_reaction(json.loads(
        endpoints_path.read_text(encoding="utf-8")
    )["geometries"])
    details = []
    flat = []
    for index, score in enumerate(regressions, start=1):
        reaction_id = score["reaction_id"]
        regions = {
            region: _endpoint(geometries[reaction_id][region], device)
            for region in ("reactant", "transition_state", "product")
        }
        broken, formed, family = _bond_changes(regions["reactant"], regions["product"])
        delta = (
            abs(float(score["v0_barrier_error_eV"]))
            - abs(float(score["production_barrier_error_eV"]))
        )
        composition = Counter(geometries[reaction_id]["reactant"]["symbols"])
        transition = regions["transition_state"]
        reactant = regions["reactant"]
        contact_keys = set(transition["contacts"]) | set(reactant["contacts"])
        contact_changes = []
        for key in contact_keys:
            ts = transition["contacts"].get(key)
            re = reactant["contacts"].get(key)
            contact_changes.append({
                "edge": key,
                "pair": (ts or re)["pair"],
                "rejected_attraction_change_eV": (
                    (ts or {}).get("v0_rejected_attraction_eV", 0.0)
                    - (re or {}).get("v0_rejected_attraction_eV", 0.0)
                ),
                "membership_asymmetry_ts": (ts or {}).get("membership_asymmetry", 0.0),
                "both_endpoints_competing_ts": (ts or {}).get("both_endpoints_competing", False),
            })
        contact_changes.sort(
            key=lambda row: abs(row["rejected_attraction_change_eV"]), reverse=True
        )
        detail = {
            "reaction_id": reaction_id,
            "production_barrier_error_eV": float(score["production_barrier_error_eV"]),
            "v0_barrier_error_eV": float(score["v0_barrier_error_eV"]),
            "absolute_error_regression_eV": delta,
            "severity": _severity(delta),
            "composition": dict(sorted(composition.items())),
            "reaction_family": family,
            "broken_pair_types": broken,
            "formed_pair_types": formed,
            "barrier_attraction_correction_change_eV": (
                transition["rejected_attraction_eV"] - reactant["rejected_attraction_eV"]
            ),
            "barrier_overcoordination_change_current_eV": (
                transition["original_heavy_over_eV"] - reactant["original_heavy_over_eV"]
            ),
            "barrier_overcoordination_change_v0_eV": (
                transition["replacement_heavy_over_eV"] - reactant["replacement_heavy_over_eV"]
            ),
            "maximum_ts_membership_asymmetry": transition["maximum_membership_asymmetry"],
            "ts_both_endpoints_competing_edges": transition["both_endpoints_competing_edges"],
            "ts_competing_heavy_centres": transition["heavy_competing_centres"],
            "largest_contact_changes": contact_changes[:5],
            "regions": regions,
        }
        details.append(detail)
        flat.append({
            key: detail[key] for key in (
                "reaction_id", "production_barrier_error_eV", "v0_barrier_error_eV",
                "absolute_error_regression_eV", "severity", "reaction_family",
                "barrier_attraction_correction_change_eV",
                "barrier_overcoordination_change_current_eV",
                "barrier_overcoordination_change_v0_eV",
                "maximum_ts_membership_asymmetry",
                "ts_both_endpoints_competing_edges", "ts_competing_heavy_centres",
            )
        } | {
            "composition": json.dumps(detail["composition"], sort_keys=True),
            "relevant_heavy_heavy_contacts": "; ".join(
                f"{row['pair']}:{row['rejected_attraction_change_eV']:+.3f}eV"
                for row in contact_changes[:5]
            ),
        })
        if index == 1 or index % 10 == 0:
            print(f"[{index:2d}/{len(regressions)}] {reaction_id}", flush=True)
    summary = {
        "regression_count": len(details),
        "severity_counts": dict(Counter(row["severity"] for row in details)),
        "family_counts": dict(Counter(row["reaction_family"] for row in details)),
        "pair_change_counts": dict(Counter(
            pair for row in details for pair in row["broken_pair_types"] + row["formed_pair_types"]
        )),
        "both_endpoint_competition_count": sum(
            row["ts_both_endpoints_competing_edges"] > 0 for row in details
        ),
        "details": details,
    }
    return flat, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--endpoints", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    rows, summary = analyse(args.scores, args.endpoints, args.device)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
