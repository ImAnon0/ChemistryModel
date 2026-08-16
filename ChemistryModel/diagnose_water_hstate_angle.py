from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch

from reactive import ELEMENT_INDEX
from h_state_torch import HStateReferenceBatchedSimulation


GEOMETRIES = Path("research_data/qm_residual/dense_scan_geometries.json")
QM_RESULTS = Path("research_data/qm_residual/dense_scan_qm.csv")

BOX_SIZE = 30.0
LOW_X = 1.04
HIGH_X = 1.14


class DiagnosticHState(HStateReferenceBatchedSimulation):
    """
    H-state model with read-only diagnostic capture.

    This does NOT modify the energy equation.
    """

    def _hydrogen_state_correction(self, positions, base_per_atom):
        cached = getattr(self, "_reactive_intermediates", None)

        if cached is None or cached[0] is not positions:
            raise RuntimeError("Missing reactive intermediates")

        values = cached[1]

        taper = values["taper"]
        pair_depth = values["pair_depth"]
        pair_width = values["pair_width"]
        shift = values["shift"]
        repulsive = values["repulsive"]

        attractive = (
            2.0
            * pair_depth
            * torch.exp(-pair_width * shift)
        )

        pair_morse = taper * (repulsive - attractive)

        over_scale = self.over_coordination_scale(
            taper,
            values["unsoftened_depth"],
            values["mask"],
            cache_key=positions,
        )

        excess = torch.clamp(
            values["coordination"] - values["valence"],
            min=0.0,
        )

        base_over = (
            self.over_penalty
            * over_scale
            * excess ** 2
        )

        neighbours_numpy = (
            values["neighbours"]
            .detach()
            .cpu()
            .numpy()
        )

        active_numpy = (
            (
                taper.detach().cpu().numpy()
                > 1e-12
            )
            & self.neighbour_mask.detach().cpu().numpy()
        )

        hydrogen = int(ELEMENT_INDEX["H"])

        state_total = torch.zeros(
            (),
            dtype=base_per_atom.dtype,
            device=base_per_atom.device,
        )

        h_pair_total = torch.zeros_like(state_total)
        h_over_total = torch.zeros_like(state_total)

        for box in range(self.box_count):
            start = box * self.per_box
            stop = start + self.per_box

            edge_atoms, edge_rows, edge_slots = (
                self._active_edges_for_box(
                    box,
                    values,
                    neighbours_numpy,
                    active_numpy,
                )
            )

            if not edge_atoms:
                continue

            state_energy = self._box_state_energy(
                edge_atoms,
                edge_rows,
                edge_slots,
                values,
            )

            state_total = state_total + state_energy

            pair_terms = [
                pair_morse[row, slot]
                for row, slot in zip(edge_rows, edge_slots)
            ]

            if pair_terms:
                h_pair_total = (
                    h_pair_total
                    + torch.stack(pair_terms).sum()
                )

            hydrogen_atoms = [
                atom
                for atom in range(start, stop)
                if int(self.types_numpy[atom]) == hydrogen
            ]

            if hydrogen_atoms:
                h_over_total = (
                    h_over_total
                    + torch.stack([
                        base_over[atom]
                        for atom in hydrogen_atoms
                    ]).sum()
                )

        # Capture before the parent method clears the intermediate cache.
        self._water_diag = {
            "state_energy": float(state_total.detach().cpu()),
            "base_h_pair": float(h_pair_total.detach().cpu()),
            "base_h_over": float(h_over_total.detach().cpu()),
            "values": {
                key: value.detach().cpu().clone()
                for key, value in values.items()
                if isinstance(value, torch.Tensor)
            },
        }

        return super()._hydrogen_state_correction(
            positions,
            base_per_atom,
        )


def load_qm():
    result = {}

    with QM_RESULTS.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue

            result[row["geometry_id"]] = float(
                row["qm_energy_eV"]
            )

    return result


def angle_decomposition(simulation, symbols):
    diag = simulation._water_diag
    values = diag["values"]

    oxygen_indices = [
        index
        for index, symbol in enumerate(symbols)
        if symbol == "O"
    ]

    if not oxygen_indices:
        raise RuntimeError("No oxygen atoms found")

    # The water transfer benchmark can contain more than one oxygen
    # (donor oxygen + oxygen-containing collision partner).  We want the
    # donor oxygen whose H-centred angular environment is being dismantled.
    # Select the oxygen with the strongest total active H contact strength.
    #
    # This is diagnostic-only bookkeeping; it does not affect the potential.
    oxygen_candidates = []

    for candidate in oxygen_indices:
        candidate_taper = values["taper"][candidate]
        candidate_order = values["order"][candidate]
        candidate_neighbours = values["neighbours"][candidate]

        h_taper_sum = 0.0
        h_ordered_taper_sum = 0.0
        h_contact_count = 0

        for slot in range(len(candidate_taper)):
            t = float(candidate_taper[slot])

            if t <= 1e-12:
                continue

            neighbour = int(candidate_neighbours[slot])

            if neighbour < 0 or neighbour >= len(symbols):
                continue

            if symbols[neighbour] != "H":
                continue

            h_contact_count += 1
            h_taper_sum += t
            h_ordered_taper_sum += t * float(candidate_order[slot])

        oxygen_candidates.append({
            "index": candidate,
            "h_contact_count": h_contact_count,
            "h_taper_sum": h_taper_sum,
            "h_ordered_taper_sum": h_ordered_taper_sum,
        })

    # Strongest total H contact wins.  Ordered taper and contact count break
    # near-ties deterministically.
    selected = max(
        oxygen_candidates,
        key=lambda item: (
            item["h_taper_sum"],
            item["h_ordered_taper_sum"],
            item["h_contact_count"],
            -item["index"],
        ),
    )

    oxygen = selected["index"]

    taper = values["taper"][oxygen]
    order = values["order"][oxygen]
    neighbours = values["neighbours"][oxygen]
    distances = values["distances"][oxygen]

    coordination = float(
        values["coordination"][oxygen]
    )

    bonded_order = float(
        torch.sum(taper * order)
    )

    outer = float(
        simulation.outer_electrons[
            simulation.types[oxygen]
        ]
        .detach()
        .cpu()
    )

    lone_pairs = max(
        (outer - bonded_order) / 2.0,
        0.0,
    )

    steric = min(
        max(coordination + lone_pairs, 2.0),
        4.0,
    )

    if steric < 3.0:
        low_angle_deg = (
            180.0
            + (120.0 - 180.0)
            * (steric - 2.0)
        )
    else:
        low_angle_deg = (
            120.0
            + (109.47 - 120.0)
            * (steric - 3.0)
        )

    rest_deg = (
        low_angle_deg
        - simulation.lone_pair_squeeze
        * lone_pairs
    )

    lone_pair_directionality = min(
        max(0.5 * lone_pairs, 0.0),
        1.0,
    )

    # Total angle energy centred on oxygen is already calculated
    # by reactive_torch and stored in _energy_parts.
    angle_total = float(
        simulation._energy_parts["angle"][oxygen]
        .detach()
        .cpu()
    )

    active = []

    for slot in range(len(taper)):
        t = float(taper[slot])

        if t <= 1e-12:
            continue

        neighbour = int(neighbours[slot])

        if neighbour < 0 or neighbour >= len(symbols):
            continue

        active.append({
            "slot": slot,
            "atom": neighbour,
            "symbol": symbols[neighbour],
            "taper": t,
            "order": float(order[slot]),
            "distance": float(distances[slot]),
        })

    # Reconstruct each O-centred angular pair from the actual
    # simulation coordinates.
    positions = (
        simulation.positions
        .detach()
        .cpu()
        .to(torch.float64)
    )

    pair_details = []

    stiffness = float(
        simulation.angle_stiffness[
            simulation.types[oxygen]
        ]
        .detach()
        .cpu()
    )

    rest_rad = math.radians(rest_deg)

    for left_index in range(len(active)):
        for right_index in range(
            left_index + 1,
            len(active),
        ):
            left = active[left_index]
            right = active[right_index]

            vector_left = (
                positions[left["atom"]]
                - positions[oxygen]
            )

            vector_right = (
                positions[right["atom"]]
                - positions[oxygen]
            )

            cosine = float(
                torch.dot(
                    vector_left,
                    vector_right,
                )
                / (
                    torch.linalg.norm(vector_left)
                    * torch.linalg.norm(vector_right)
                )
            )

            cosine = min(
                max(cosine, -1.0 + 1e-7),
                1.0 - 1e-7,
            )

            angle_rad = math.acos(cosine)
            angle_deg = math.degrees(angle_rad)

            first_taper = left["taper"]
            second_taper = right["taper"]

            difference = (
                first_taper - second_taper
            )

            weaker_taper = 0.5 * (
                first_taper
                + second_taper
                - math.sqrt(
                    difference * difference
                    + 1e-8
                )
                + 1e-4
            )

            angle_engagement = (
                weaker_taper
                + (1.0 - weaker_taper)
                * lone_pair_directionality
            )

            pair_taper = (
                first_taper * second_taper
            )

            weight = (
                pair_taper * angle_engagement
            )

            energy = (
                0.5
                * stiffness
                * weight
                * (angle_rad - rest_rad) ** 2
            )

            pair_details.append({
                "left": left["atom"],
                "right": right["atom"],
                "left_symbol": left["symbol"],
                "right_symbol": right["symbol"],
                "left_taper": first_taper,
                "right_taper": second_taper,
                "weaker_taper": weaker_taper,
                "engagement": angle_engagement,
                "weight": weight,
                "angle_deg": angle_deg,
                "energy": energy,
            })

    return {
        "oxygen_index": oxygen,
        "oxygen_candidates": oxygen_candidates,
        "coordination": coordination,
        "bonded_order": bonded_order,
        "lone_pairs": lone_pairs,
        "steric": steric,
        "rest_angle": rest_deg,
        "lone_direction": lone_pair_directionality,
        "angle_total": angle_total,
        "active": active,
        "pairs": pair_details,
    }


def main():
    payload = json.loads(
        GEOMETRIES.read_text(encoding="utf-8")
    )

    qm = load_qm()

    geometries = []

    for geometry in payload["geometries"]:
        if geometry["system"] != "water":
            continue

        if geometry["sample_kind"] != "dense_transfer_scan":
            continue

        rc = geometry.get("reaction_coordinate", {})
        x = rc.get("transfer_distance_angstrom")

        if x is None:
            continue

        x = float(x)

        if LOW_X <= x <= HIGH_X:
            geometries.append(geometry)

    geometries.sort(
        key=lambda g: float(
            g["reaction_coordinate"][
                "transfer_distance_angstrom"
            ]
        )
    )

    if not geometries:
        raise RuntimeError(
            "No water dense-scan geometries found"
        )

    rows = []

    qm_reference = None
    h_reference = None

    # Get reactant reference first.
    references = [
        g
        for g in payload["geometries"]
        if (
            g["system"] == "water"
            and g["sample_kind"]
            == "reactant_reference"
        )
    ]

    if len(references) != 1:
        raise RuntimeError(
            f"Expected one water reactant reference, got {len(references)}"
        )

    reference = references[0]

    reference_sim = DiagnosticHState(
        boxes=[(
            reference["symbols"],
            reference["coordinates_angstrom"],
        )],
        box_size=BOX_SIZE,
        target_temperature=0.0,
        friction=0.0,
        device="cpu",
        dtype=torch.float64,
        random_seed=0,
        relax_on_start=False,
    )

    qm_reference = qm[reference["geometry_id"]]
    h_reference = float(
        reference_sim.potential_per_box[0]
    )

    print()
    print("WATER H-STATE ANGLE MICROSCOPE")
    print()

    for geometry in geometries:
        gid = geometry["geometry_id"]
        symbols = geometry["symbols"]

        simulation = DiagnosticHState(
            boxes=[(
                symbols,
                geometry[
                    "coordinates_angstrom"
                ],
            )],
            box_size=BOX_SIZE,
            target_temperature=0.0,
            friction=0.0,
            device="cpu",
            dtype=torch.float64,
            random_seed=0,
            relax_on_start=False,
        )

        h_total = float(
            simulation.potential_per_box[0]
        )

        base_parts = simulation._energy_parts

        bond = float(
            base_parts["bond"].sum()
        )
        over = float(
            base_parts["over"].sum()
        )
        angle = float(
            base_parts["angle"].sum()
        )

        base_total = bond + over + angle

        diag = simulation._water_diag
        oxygen = angle_decomposition(
            simulation,
            symbols,
        )

        qm_rel = qm[gid] - qm_reference
        h_rel = h_total - h_reference

        x = float(
            geometry["reaction_coordinate"][
                "transfer_distance_angstrom"
            ]
        )

        row = {
            "x": x,
            "gid": gid,
            "qm_rel": qm_rel,
            "h_rel": h_rel,
            "residual": qm_rel - h_rel,
            "base_total": base_total,
            "bond": bond,
            "over": over,
            "angle": angle,
            "state": diag["state_energy"],
            "h_pair_removed": diag["base_h_pair"],
            "h_over_removed": diag["base_h_over"],
            "oxygen_index": oxygen["oxygen_index"],
            "oxygen_candidates": oxygen["oxygen_candidates"],
            "coordination": oxygen["coordination"],
            "bonded_order": oxygen["bonded_order"],
            "lone_pairs": oxygen["lone_pairs"],
            "steric": oxygen["steric"],
            "rest_angle": oxygen["rest_angle"],
            "lone_direction": oxygen["lone_direction"],
            "oxygen_angle": oxygen["angle_total"],
            "pairs": oxygen["pairs"],
            "active": oxygen["active"],
        }

        rows.append(row)

        print(
            f"x={x:.3f} A  "
            f"QM={qm_rel:+.6f}  "
            f"H={h_rel:+.6f}  "
            f"res={row['residual']:+.6f}"
        )

        print(
            f"    base={base_total:+.6f}  "
            f"bond={bond:+.6f}  "
            f"over={over:+.6f}  "
            f"angle={angle:+.6f}"
        )

        print(
            f"    state={row['state']:+.6f}  "
            f"Hpair_removed={row['h_pair_removed']:+.6f}  "
            f"Hover_removed={row['h_over_removed']:+.6f}"
        )

        candidate_text = ", ".join(
            f"O{item['index']}:Hcontacts={item['h_contact_count']},"
            f"Htaper={item['h_taper_sum']:.4f}"
            for item in row["oxygen_candidates"]
        )

        print(
            f"    selected donor O={row['oxygen_index']}  "
            f"candidates=[{candidate_text}]"
        )

        print(
            f"    O coordination={row['coordination']:.6f}  "
            f"bonded_order={row['bonded_order']:.6f}"
        )

        print(
            f"    lone_pairs={row['lone_pairs']:.6f}  "
            f"steric={row['steric']:.6f}"
        )

        print(
            f"    rest_angle={row['rest_angle']:.3f} deg  "
            f"lone_direction={row['lone_direction']:.6f}"
        )

        for pair in row["pairs"]:
            print(
                "      angle "
                f"{pair['left_symbol']}{pair['left']}"
                f"-O-"
                f"{pair['right_symbol']}{pair['right']}  "
                f"actual={pair['angle_deg']:.3f} deg  "
                f"tapers="
                f"{pair['left_taper']:.4f}/"
                f"{pair['right_taper']:.4f}  "
                f"weak={pair['weaker_taper']:.4f}  "
                f"engage={pair['engagement']:.4f}  "
                f"weight={pair['weight']:.4f}  "
                f"E={pair['energy']:+.6f}"
            )

        print()

    print()
    print("ADJACENT 0.02 A CHANGES")
    print()

    columns = [
        ("qm_rel", "QM"),
        ("h_rel", "H-state"),
        ("residual", "residual"),
        ("bond", "base bond"),
        ("over", "base over"),
        ("angle", "base angle"),
        ("state", "state"),
        ("h_pair_removed", "H pair removed"),
        ("h_over_removed", "H over removed"),
        ("coordination", "O coordination"),
        ("bonded_order", "O bonded order"),
        ("lone_pairs", "O lone pairs"),
        ("steric", "O steric"),
        ("rest_angle", "rest angle deg"),
    ]

    for left, right in zip(rows, rows[1:]):
        print(
            f"{left['x']:.3f} -> "
            f"{right['x']:.3f} A"
        )

        for key, label in columns:
            change = right[key] - left[key]

            print(
                f"    {label:<18s} "
                f"{change:+.6f}"
            )

        print()


if __name__ == "__main__":
    main()