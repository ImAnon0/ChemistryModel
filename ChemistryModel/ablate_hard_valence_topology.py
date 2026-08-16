"""
Hard valence-selection topology ablation.

Diagnostic question
-------------------
If radial contacts are treated as candidates, and each atom allows only its
strongest V contacts (V = elemental valence) to participate in *chemical
topology*, does the H-state model improve across the QM microscope without
breaking genuine O-O topology in peroxide?

This is deliberately NOT production physics:
  - selection is hard/discrete and therefore not force-continuous
  - radial Morse/H-state energies are untouched
  - only heavy-atom overcoordination and heavy-centred angles are replaced

If this works, the next task is to replace the hard top-V rule with a smooth
state/membership formulation.

Run:
    py ablate_hard_valence_topology.py

Output:
    research_data/qm_residual/hard_valence_topology_ablation.csv
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch

import reactive as R
from bond_calibration import hydrogen_peroxide_geometry
from h_state_torch import HStateReferenceBatchedSimulation


GEOMETRIES = Path("research_data/qm_residual/dense_scan_geometries.json")
QM_RESULTS = Path("research_data/qm_residual/dense_scan_qm.csv")
OUTPUT = Path(
    "research_data/qm_residual/hard_valence_topology_ablation.csv"
)

BOX_SIZE = 30.0
EPS = 1e-12


class CaptureHState(HStateReferenceBatchedSimulation):
    """Normal H-state model plus detached read-only intermediates."""

    def _hydrogen_state_correction(self, positions, base_per_atom):
        cached = getattr(self, "_reactive_intermediates", None)

        if cached is None or cached[0] is not positions:
            raise RuntimeError("Missing reactive intermediates")

        values = cached[1]

        self._hard_valence_values = {
            key: value.detach().clone()
            for key, value in values.items()
            if isinstance(value, torch.Tensor)
        }

        return super()._hydrogen_state_correction(
            positions,
            base_per_atom,
        )


def load_qm(path: Path):
    result = {}

    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                result[row["geometry_id"]] = float(
                    row["qm_energy_eV"]
                )

    return result


def rmse(values):
    return math.sqrt(
        sum(value * value for value in values)
        / len(values)
    )


def build_hard_membership(simulation, values):
    """
    For each centre atom, choose up to V strongest active contacts.

    Score:
        radial taper * exp[-a_single * (r - r0_single)]

    The score is only used to rank contacts. The topology strength of a
    selected contact remains the existing radial taper.
    """

    taper = values["taper"]
    mask = values["mask"]
    distances = values["distances"]
    centre_types = values["centre_types"]
    other_types = values["other_types"]

    single_r0 = simulation.bond_length[
        centre_types,
        other_types,
    ]

    single_width = simulation.bond_width[
        centre_types,
        other_types,
    ]

    overlap = torch.exp(
        -single_width * (distances - single_r0)
    )

    score = taper * overlap * mask

    membership = torch.zeros_like(taper)

    valence = simulation.valence[simulation.types]

    selections = []

    for atom in range(taper.shape[0]):
        active_slots = torch.nonzero(
            (mask[atom] > 0.0)
            & (taper[atom] > EPS),
            as_tuple=False,
        ).flatten()

        capacity = max(
            int(round(float(valence[atom].detach().cpu()))),
            0,
        )

        if capacity <= 0 or active_slots.numel() == 0:
            selections.append([])
            continue

        take = min(capacity, int(active_slots.numel()))

        local_scores = score[atom, active_slots]

        _, local_indices = torch.topk(
            local_scores,
            k=take,
            largest=True,
            sorted=True,
        )

        chosen_slots = active_slots[local_indices]

        membership[atom, chosen_slots] = 1.0

        atom_selection = []

        for slot in chosen_slots.detach().cpu().tolist():
            atom_selection.append({
                "slot": int(slot),
                "neighbour": int(
                    values["neighbours"][atom, slot]
                    .detach()
                    .cpu()
                ),
                "score": float(
                    score[atom, slot].detach().cpu()
                ),
                "taper": float(
                    taper[atom, slot].detach().cpu()
                ),
                "overlap": float(
                    overlap[atom, slot].detach().cpu()
                ),
            })

        selections.append(atom_selection)

    return membership, overlap, score, selections


def hard_topology_counterfactual(simulation):
    values = simulation._hard_valence_values

    taper = values["taper"]
    order = values["order"]
    mask = values["mask"]
    neighbours = values["neighbours"]
    distances = values["distances"]
    unsoftened_depth = values["unsoftened_depth"]

    membership, overlap, score, selections = (
        build_hard_membership(
            simulation,
            values,
        )
    )

    topology_taper = taper * membership

    hydrogen = int(R.ELEMENT_INDEX["H"])

    heavy_mask = (
        simulation.types != hydrogen
    )

    # ------------------------------------------------------------
    # Heavy-atom overcoordination only.
    # H-state already removes the ordinary hydrogen overcoordination term,
    # so replacing H over here would double-count that correction.
    # ------------------------------------------------------------

    topology_coordination = torch.sum(
        topology_taper,
        dim=1,
    )

    valence = simulation.valence[simulation.types]

    topology_excess = torch.clamp(
        topology_coordination - valence,
        min=0.0,
    )

    topology_over_scale = simulation.over_coordination_scale(
        topology_taper,
        unsoftened_depth,
        mask,
        cache_key=None,
    )

    topology_over_per_atom = (
        simulation.over_penalty
        * topology_over_scale
        * topology_excess ** 2
    )

    original_over_per_atom = (
        simulation._energy_parts["over"]
        .detach()
        .to(
            device=topology_over_per_atom.device,
            dtype=topology_over_per_atom.dtype,
        )
    )

    original_heavy_over = float(
        original_over_per_atom[heavy_mask]
        .sum()
        .detach()
        .cpu()
    )

    topology_heavy_over = float(
        topology_over_per_atom[heavy_mask]
        .sum()
        .detach()
        .cpu()
    )

    # ------------------------------------------------------------
    # Heavy-centred angles only.
    # Mirrors reactive_torch.py with topology_taper replacing taper.
    # ------------------------------------------------------------

    topology_bonded_order = torch.sum(
        topology_taper * order,
        dim=1,
    )

    outer = simulation.outer_electrons[
        simulation.types
    ]

    topology_lone_pairs = torch.clamp(
        (outer - topology_bonded_order) / 2.0,
        min=0.0,
    )

    topology_steric = torch.clamp(
        topology_coordination + topology_lone_pairs,
        2.0,
        4.0,
    )

    low_angle = torch.where(
        topology_steric < 3.0,
        180.0
        + (120.0 - 180.0)
        * (topology_steric - 2.0),
        120.0
        + (109.47 - 120.0)
        * (topology_steric - 3.0),
    )

    topology_rest = torch.deg2rad(
        low_angle
        - simulation.lone_pair_squeeze
        * topology_lone_pairs
    )

    stiffness = simulation.angle_stiffness[
        simulation.types
    ]

    positions = simulation.positions.detach()

    gathered = simulation._gather_neighbours(
        positions,
        neighbours,
        "positions",
    )

    offsets = gathered - positions[:, None, :]

    offsets = (
        offsets
        - simulation.box_size
        * torch.round(
            offsets / simulation.box_size
        )
    )

    left = offsets[:, :, None, :]
    right = offsets[:, None, :, :]

    dot = torch.sum(
        left * right,
        dim=3,
    )

    cosine = torch.clamp(
        dot
        / torch.clamp(
            distances[:, :, None]
            * distances[:, None, :],
            min=1e-9,
        ),
        -1.0 + 1e-7,
        1.0 - 1e-7,
    )

    angle = torch.arccos(cosine)

    angle_pair_taper = (
        topology_taper[:, :, None]
        * topology_taper[:, None, :]
    )

    first_taper = topology_taper[:, :, None]
    second_taper = topology_taper[:, None, :]

    taper_difference = (
        first_taper - second_taper
    )

    weaker_taper = 0.5 * (
        first_taper
        + second_taper
        - torch.sqrt(
            taper_difference ** 2
            + 1e-8
        )
        + 1e-4
    )

    lone_pair_directionality = torch.clamp(
        0.5 * topology_lone_pairs,
        0.0,
        1.0,
    )[:, None, None]

    angle_engagement = (
        weaker_taper
        + (1.0 - weaker_taper)
        * lone_pair_directionality
    )

    weight = (
        angle_pair_taper
        * angle_engagement
    )

    upper_triangle = torch.triu(
        torch.ones(
            weight.shape[1],
            weight.shape[2],
            device=simulation.device,
            dtype=simulation.dtype,
        ),
        diagonal=1,
    )

    topology_angle_energy = (
        0.5
        * stiffness[:, None, None]
        * weight
        * upper_triangle
        * (
            angle
            - topology_rest[:, None, None]
        ) ** 2
    )

    topology_angle_per_atom = torch.sum(
        topology_angle_energy,
        dim=(1, 2),
    )

    original_angle_per_atom = (
        simulation._energy_parts["angle"]
        .detach()
        .to(
            device=topology_angle_per_atom.device,
            dtype=topology_angle_per_atom.dtype,
        )
    )

    original_heavy_angle = float(
        original_angle_per_atom[heavy_mask]
        .sum()
        .detach()
        .cpu()
    )

    topology_heavy_angle = float(
        topology_angle_per_atom[heavy_mask]
        .sum()
        .detach()
        .cpu()
    )

    return {
        "over_delta_eV": (
            topology_heavy_over
            - original_heavy_over
        ),
        "angle_delta_eV": (
            topology_heavy_angle
            - original_heavy_angle
        ),
        "original_heavy_over_eV": original_heavy_over,
        "selected_heavy_over_eV": topology_heavy_over,
        "original_heavy_angle_eV": original_heavy_angle,
        "selected_heavy_angle_eV": topology_heavy_angle,
        "membership": membership.detach().cpu(),
        "overlap": overlap.detach().cpu(),
        "score": score.detach().cpu(),
        "selections": selections,
        "topology_taper": topology_taper.detach().cpu(),
    }


def evaluate_geometry(geometry):
    simulation = CaptureHState(
        boxes=[(
            geometry["symbols"],
            geometry["coordinates_angstrom"],
        )],
        box_size=BOX_SIZE,
        target_temperature=0.0,
        friction=0.0,
        device="cpu",
        dtype=torch.float64,
        random_seed=0,
        relax_on_start=False,
    )

    normal = float(
        simulation.potential_per_box[0]
    )

    counterfactual = (
        hard_topology_counterfactual(
            simulation
        )
    )

    selected = (
        normal
        + counterfactual["over_delta_eV"]
        + counterfactual["angle_delta_eV"]
    )

    return normal, selected, counterfactual


def pair_selection_summary(
    symbols,
    simulation_values,
    counterfactual,
):
    neighbours = simulation_values["neighbours"].detach().cpu()
    taper = simulation_values["taper"].detach().cpu()

    items = []

    for atom, chosen in enumerate(
        counterfactual["selections"]
    ):
        for entry in chosen:
            neighbour = entry["neighbour"]

            if neighbour < atom:
                continue

            items.append(
                (
                    f"{symbols[atom]}{atom}-"
                    f"{symbols[neighbour]}{neighbour}",
                    entry["score"],
                    entry["taper"],
                    entry["overlap"],
                )
            )

    return items


def peroxide_audit(distance):
    symbols, coordinates = hydrogen_peroxide_geometry(
        oo_distance=distance
    )

    geometry = {
        "symbols": symbols,
        "coordinates_angstrom": coordinates,
    }

    simulation = CaptureHState(
        boxes=[(
            symbols,
            coordinates,
        )],
        box_size=BOX_SIZE,
        target_temperature=0.0,
        friction=0.0,
        device="cpu",
        dtype=torch.float64,
        random_seed=0,
        relax_on_start=False,
    )

    counterfactual = hard_topology_counterfactual(
        simulation
    )

    values = simulation._hard_valence_values

    oxygen_atoms = [
        index
        for index, symbol in enumerate(symbols)
        if symbol == "O"
    ]

    oo_details = []

    for oxygen in oxygen_atoms:
        chosen = {
            entry["neighbour"]: entry
            for entry in counterfactual["selections"][oxygen]
        }

        other_oxygen = (
            oxygen_atoms[1]
            if oxygen == oxygen_atoms[0]
            else oxygen_atoms[0]
        )

        oo_details.append({
            "oxygen": oxygen,
            "other_oxygen": other_oxygen,
            "selected": other_oxygen in chosen,
            "entry": chosen.get(other_oxygen),
        })

    return oo_details


def main():
    payload = json.loads(
        GEOMETRIES.read_text(
            encoding="utf-8"
        )
    )

    qm = load_qm(QM_RESULTS)

    geometries = payload["geometries"]

    missing = [
        geometry["geometry_id"]
        for geometry in geometries
        if geometry["geometry_id"] not in qm
    ]

    if missing:
        raise RuntimeError(
            f"QM missing {len(missing)} rows; "
            f"first={missing[0]}"
        )

    print("HARD VALENCE-SELECTION TOPOLOGY ABLATION")
    print("radial/H-state physics : unchanged")
    print("heavy overcoordination : hard top-V topology")
    print("heavy-centred angles   : hard top-V topology")
    print("production safe        : NO (diagnostic only)")
    print()

    raw = []

    total = len(geometries)

    for index, geometry in enumerate(
        geometries,
        start=1,
    ):
        gid = geometry["geometry_id"]

        normal, selected, parts = (
            evaluate_geometry(geometry)
        )

        rc = geometry.get(
            "reaction_coordinate",
            {},
        )

        raw.append({
            "geometry_id": gid,
            "system": geometry["system"],
            "sample_kind": geometry["sample_kind"],
            "region": geometry["region"],
            "transfer_distance_angstrom": rc.get(
                "transfer_distance_angstrom",
                "",
            ),
            "donor_distance_angstrom": rc.get(
                "donor_distance_angstrom",
                "",
            ),
            "qm_energy_eV": qm[gid],
            "hstate_energy_eV": normal,
            "hard_valence_energy_eV": selected,
            "over_delta_eV": parts["over_delta_eV"],
            "angle_delta_eV": parts["angle_delta_eV"],
            "original_heavy_over_eV": (
                parts["original_heavy_over_eV"]
            ),
            "selected_heavy_over_eV": (
                parts["selected_heavy_over_eV"]
            ),
            "original_heavy_angle_eV": (
                parts["original_heavy_angle_eV"]
            ),
            "selected_heavy_angle_eV": (
                parts["selected_heavy_angle_eV"]
            ),
        })

        if (
            index == 1
            or index % 20 == 0
            or index == total
        ):
            print(
                f"[{index:3d}/{total:3d}] "
                f"{gid:<38s} "
                f"H={normal:+.6f}  "
                f"topV={selected:+.6f}"
            )

    refs = {}

    for row in raw:
        if row["sample_kind"] == "reactant_reference":
            refs[row["system"]] = {
                "qm": row["qm_energy_eV"],
                "h": row["hstate_energy_eV"],
                "topv": row["hard_valence_energy_eV"],
            }

    final = []

    for row in raw:
        ref = refs[row["system"]]

        qm_rel = row["qm_energy_eV"] - ref["qm"]
        h_rel = row["hstate_energy_eV"] - ref["h"]
        topv_rel = (
            row["hard_valence_energy_eV"]
            - ref["topv"]
        )

        merged = dict(row)

        merged.update({
            "qm_relative_eV": qm_rel,
            "hstate_relative_eV": h_rel,
            "hard_valence_relative_eV": topv_rel,
            "hstate_residual_eV": qm_rel - h_rel,
            "hard_valence_residual_eV": qm_rel - topv_rel,
        })

        final.append(merged)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                final[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(final)

    dense = defaultdict(list)

    for row in final:
        if row["sample_kind"] == "dense_transfer_scan":
            dense[row["system"]].append(row)

    print()
    print(f"wrote : {OUTPUT}")
    print()

    print("DENSE TRANSFER RESIDUALS VS QM")
    print(
        f"{'system':14s} "
        f"{'H MAE':>9s} {'H RMSE':>9s} {'H max':>9s} "
        f"{'topV MAE':>10s} {'topV RMSE':>10s} {'topV max':>10s}"
    )

    for system in sorted(dense):
        rows = dense[system]

        h_values = [
            row["hstate_residual_eV"]
            for row in rows
        ]

        topv_values = [
            row["hard_valence_residual_eV"]
            for row in rows
        ]

        print(
            f"{system:14s} "
            f"{statistics.fmean(abs(v) for v in h_values):9.4f} "
            f"{rmse(h_values):9.4f} "
            f"{max(abs(v) for v in h_values):9.4f} "
            f"{statistics.fmean(abs(v) for v in topv_values):10.4f} "
            f"{rmse(topv_values):10.4f} "
            f"{max(abs(v) for v in topv_values):10.4f}"
        )

    print()
    print("WORST ADJACENT RESIDUAL STEP")

    for system in sorted(dense):
        rows = sorted(
            dense[system],
            key=lambda row: float(
                row[
                    "transfer_distance_angstrom"
                ]
            ),
        )

        for column, label in (
            (
                "hstate_residual_eV",
                "H-state",
            ),
            (
                "hard_valence_residual_eV",
                "top-V",
            ),
        ):
            pairs = []

            for left, right in zip(
                rows,
                rows[1:],
            ):
                jump = (
                    right[column]
                    - left[column]
                )

                pairs.append(
                    (
                        abs(jump),
                        jump,
                        left,
                        right,
                    )
                )

            _, jump, left, right = max(
                pairs,
                key=lambda item: item[0],
            )

            print(
                f"{system:14s} "
                f"{label:8s} "
                f"{float(left['transfer_distance_angstrom']):.3f}->"
                f"{float(right['transfer_distance_angstrom']):.3f} A  "
                f"dResidual={jump:+.6f} eV"
            )

    print()
    print("PRODUCT REFERENCE ENERGIES")
    print(
        f"{'system':14s} "
        f"{'QM':>9s} "
        f"{'H-state':>9s} "
        f"{'top-V':>9s}"
    )

    for system in sorted(refs):
        products = [
            row
            for row in final
            if (
                row["system"] == system
                and row["sample_kind"]
                == "product_reference"
            )
        ]

        if len(products) == 1:
            row = products[0]

            print(
                f"{system:14s} "
                f"{row['qm_relative_eV']:+9.4f} "
                f"{row['hstate_relative_eV']:+9.4f} "
                f"{row['hard_valence_relative_eV']:+9.4f}"
            )

    print()
    print("PEROXIDE O-O SELECTION AUDIT")

    for distance in (
        1.4750,
        1.8100,
        2.0635,
        2.1086,
        2.2125,
    ):
        details = peroxide_audit(
            distance
        )

        status = ", ".join(
            (
                f"O{detail['oxygen']}->O{detail['other_oxygen']}:"
                f"{'selected' if detail['selected'] else 'NOT selected'}"
                + (
                    (
                        f"(score={detail['entry']['score']:.5f},"
                        f" taper={detail['entry']['taper']:.4f},"
                        f" overlap={detail['entry']['overlap']:.5f})"
                    )
                    if detail["entry"] is not None
                    else ""
                )
            )
            for detail in details
        )

        print(
            f"  O-O={distance:.4f} A  {status}"
        )


if __name__ == "__main__":
    main()
