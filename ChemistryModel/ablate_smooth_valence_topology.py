"""
Smooth valence-state topology ablation.

This is the differentiable/state-mixing counterpart of
ablate_hard_valence_topology.py.

It deliberately reuses the already-validated hard-ablation plumbing for
recomputing heavy-atom overcoordination and heavy-centred angles. The ONLY
thing replaced is the per-atom topology membership builder:

    hard top-V selection
        ->
    local valence-state Hamiltonian + ground-state edge membership

No production physics files are modified.

State construction
------------------
For one centre atom with N active candidate contacts and elemental valence V:

    if N <= V:
        all contacts have membership 1

    if N > V:
        enumerate all size-V contact sets
        diagonal energy = -sum(selected attractive magnitudes)
        connect states differing by one exchanged contact
        coupling uses the same H-state:
            contact-overlap gate
            depth scale
            crowding normalisation
            H_STATE_MIXING

For ground-state eigenvector c_s:

    membership(edge e) = sum_s |c_s|^2 * I(e in state s)

Then:

    topology_taper = radial_taper * membership

Run:
    py ablate_smooth_valence_topology.py

Output:
    research_data/qm_residual/smooth_valence_topology_ablation.csv
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch

import reactive as R
import ablate_hard_valence_topology as HARD
from bond_calibration import hydrogen_peroxide_geometry
from h_state_torch import (
    _contact_overlap,
    _crowding_normalisation,
)


GEOMETRIES = Path("research_data/qm_residual/dense_scan_geometries.json")
QM_RESULTS = Path("research_data/qm_residual/dense_scan_qm.csv")
OUTPUT = Path(
    "research_data/qm_residual/smooth_valence_topology_ablation.csv"
)

MAX_LOCAL_CANDIDATES = 12
EPS = 1e-12


def load_qm(path: Path):
    result = {}

    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                result[row["geometry_id"]] = float(row["qm_energy_eV"])

    return result


def rmse(values):
    return math.sqrt(
        sum(value * value for value in values) / len(values)
    )


def states_differ_by_one_exchange(first, second):
    first_set = set(first)
    second_set = set(second)

    removed = list(first_set - second_set)
    added = list(second_set - first_set)

    if len(removed) != 1 or len(added) != 1:
        return None

    return removed[0], added[0]


def build_smooth_membership(simulation, values):
    """
    Match HARD.build_hard_membership's return signature so the already
    validated topology counterfactual can be reused unchanged.
    """

    taper = values["taper"]
    mask = values["mask"]
    distances = values["distances"]
    centre_types = values["centre_types"]
    other_types = values["other_types"]

    pair_depth = values["pair_depth"]
    pair_width = values["pair_width"]
    shift = values["shift"]

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

    # Keep the same simple diagnostic score returned by the hard builder.
    # It is useful for printing/ranking audits, but the Hamiltonian diagonal
    # below uses the same attractive energy form as H-state itself.
    score = taper * overlap * mask

    attractive = (
        taper
        * 2.0
        * pair_depth
        * torch.exp(
            -pair_width * shift
        )
        * mask
    )

    membership = torch.zeros_like(taper)

    valence = simulation.valence[simulation.types]

    selections = []

    for atom in range(taper.shape[0]):
        active_slots = torch.nonzero(
            (mask[atom] > 0.0)
            & (taper[atom] > EPS),
            as_tuple=False,
        ).flatten()

        active_list = [
            int(slot)
            for slot in active_slots.detach().cpu().tolist()
        ]

        capacity = max(
            int(round(float(valence[atom].detach().cpu()))),
            0,
        )

        if capacity <= 0 or not active_list:
            selections.append([])
            continue

        # No competition: every radial contact fits into the atom's valence.
        if len(active_list) <= capacity:
            membership[atom, active_slots] = 1.0

            selections.append([
                {
                    "slot": slot,
                    "neighbour": int(
                        values["neighbours"][atom, slot]
                        .detach()
                        .cpu()
                    ),
                    "score": float(score[atom, slot].detach().cpu()),
                    "taper": float(taper[atom, slot].detach().cpu()),
                    "overlap": float(overlap[atom, slot].detach().cpu()),
                    "membership": 1.0,
                }
                for slot in active_list
            ])
            continue

        if len(active_list) > MAX_LOCAL_CANDIDATES:
            raise RuntimeError(
                f"Atom {atom} has {len(active_list)} active contacts; "
                f"smooth local-state diagnostic limit is "
                f"{MAX_LOCAL_CANDIDATES}"
            )

        states = tuple(
            itertools.combinations(
                range(len(active_list)),
                capacity,
            )
        )

        zero = taper.sum() * 0.0

        diagonals = []

        for state in states:
            selected_attractions = [
                attractive[
                    atom,
                    active_list[local_index],
                ]
                for local_index in state
            ]

            if selected_attractions:
                diagonal = -torch.stack(
                    selected_attractions
                ).sum()
            else:
                diagonal = zero

            diagonals.append(diagonal)

        diagonal = torch.stack(diagonals)

        if len(states) == 1:
            probabilities = torch.ones_like(diagonal)
        else:
            transitions = {}
            weighted_degree = [
                zero for _ in states
            ]

            for first in range(len(states)):
                for second in range(first + 1, len(states)):
                    exchange = states_differ_by_one_exchange(
                        states[first],
                        states[second],
                    )

                    if exchange is None:
                        continue

                    old_local, new_local = exchange

                    old_slot = active_list[old_local]
                    new_slot = active_list[new_local]

                    contact_gate = _contact_overlap(
                        taper[atom, old_slot],
                        taper[atom, new_slot],
                    )

                    transitions[(first, second)] = (
                        old_slot,
                        new_slot,
                        contact_gate,
                    )

                    weighted_degree[first] = (
                        weighted_degree[first]
                        + contact_gate * contact_gate
                    )

                    weighted_degree[second] = (
                        weighted_degree[second]
                        + contact_gate * contact_gate
                    )

            normalisation = torch.stack([
                _crowding_normalisation(value)
                for value in weighted_degree
            ])

            couplings = {}

            for (first, second), (
                old_slot,
                new_slot,
                contact_gate,
            ) in transitions.items():
                depth_scale = torch.sqrt(
                    torch.clamp(
                        pair_depth[atom, old_slot]
                        * pair_depth[atom, new_slot],
                        min=1e-12,
                    )
                )

                denominator = torch.sqrt(
                    torch.clamp(
                        normalisation[first]
                        * normalisation[second],
                        min=1e-12,
                    )
                )

                coupling = (
                    simulation.h_state_mixing
                    * depth_scale
                    * contact_gate
                    / denominator
                )

                couplings[(first, second)] = coupling

            rows = []

            for first in range(len(states)):
                row = []

                for second in range(len(states)):
                    if first == second:
                        value = diagonal[first]
                    else:
                        key = (
                            min(first, second),
                            max(first, second),
                        )

                        value = (
                            -couplings[key]
                            if key in couplings
                            else zero
                        )

                    row.append(value)

                rows.append(torch.stack(row))

            hamiltonian = torch.stack(rows)

            eigenvalues, eigenvectors = torch.linalg.eigh(
                hamiltonian
            )

            ground = eigenvectors[:, 0]

            probabilities = ground * ground

            # Numerical audit: symmetric real eigenvectors should give
            # probabilities summing to 1.
            probabilities = (
                probabilities
                / torch.clamp(
                    probabilities.sum(),
                    min=1e-12,
                )
            )

        atom_selection = []

        for local_index, slot in enumerate(active_list):
            present = [
                probabilities[state_index]
                for state_index, state in enumerate(states)
                if local_index in state
            ]

            if present:
                edge_membership = torch.stack(present).sum()
            else:
                edge_membership = zero

            membership[atom, slot] = edge_membership

            atom_selection.append({
                "slot": slot,
                "neighbour": int(
                    values["neighbours"][atom, slot]
                    .detach()
                    .cpu()
                ),
                "score": float(score[atom, slot].detach().cpu()),
                "taper": float(taper[atom, slot].detach().cpu()),
                "overlap": float(overlap[atom, slot].detach().cpu()),
                "membership": float(
                    edge_membership.detach().cpu()
                ),
            })

        selections.append(atom_selection)

    return membership, overlap, score, selections


def evaluate_geometry(geometry):
    simulation = HARD.CaptureHState(
        boxes=[(
            geometry["symbols"],
            geometry["coordinates_angstrom"],
        )],
        box_size=HARD.BOX_SIZE,
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

    original_builder = HARD.build_hard_membership

    try:
        HARD.build_hard_membership = build_smooth_membership

        counterfactual = HARD.hard_topology_counterfactual(
            simulation
        )
    finally:
        HARD.build_hard_membership = original_builder

    smooth = (
        normal
        + counterfactual["over_delta_eV"]
        + counterfactual["angle_delta_eV"]
    )

    return simulation, normal, smooth, counterfactual


def print_atom_memberships(
    symbols,
    simulation,
    counterfactual,
    atom,
):
    entries = counterfactual["selections"][atom]

    formatted = []

    for entry in sorted(
        entries,
        key=lambda item: item["membership"],
        reverse=True,
    ):
        neighbour = entry["neighbour"]

        formatted.append(
            f"{symbols[neighbour]}{neighbour}:"
            f"m={entry['membership']:.4f},"
            f"t={entry['taper']:.4f},"
            f"s={entry['score']:.4f}"
        )

    print(
        f"    {symbols[atom]}{atom} -> "
        + " | ".join(formatted)
    )


def peroxide_audit(distance):
    symbols, coordinates = hydrogen_peroxide_geometry(
        oo_distance=distance
    )

    geometry = {
        "symbols": symbols,
        "coordinates_angstrom": coordinates,
    }

    simulation, _, _, counterfactual = evaluate_geometry(
        geometry
    )

    oxygen_atoms = [
        index
        for index, symbol in enumerate(symbols)
        if symbol == "O"
    ]

    print(f"  O-O={distance:.4f} A")

    for atom in oxygen_atoms:
        print_atom_memberships(
            symbols,
            simulation,
            counterfactual,
            atom,
        )


def water_audit(geometry):
    simulation, normal, smooth, counterfactual = (
        evaluate_geometry(geometry)
    )

    x = float(
        geometry["reaction_coordinate"][
            "transfer_distance_angstrom"
        ]
    )

    print(
        f"  water x={x:.3f} A  "
        f"H={normal:+.6f}  "
        f"smooth={smooth:+.6f}"
    )

    for atom, symbol in enumerate(geometry["symbols"]):
        if symbol == "O":
            print_atom_memberships(
                geometry["symbols"],
                simulation,
                counterfactual,
                atom,
            )


def main():
    payload = json.loads(
        GEOMETRIES.read_text(encoding="utf-8")
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
            f"QM missing {len(missing)} rows; first={missing[0]}"
        )

    print("SMOOTH VALENCE-STATE TOPOLOGY ABLATION")
    print("radial/H-state physics : unchanged")
    print("heavy overcoordination : smooth valence-state topology")
    print("heavy-centred angles   : smooth valence-state topology")
    print("state mixer            : existing H-state Hamiltonian recipe")
    print("new fitted parameters  : none")
    print("production safe        : NOT YET — diagnostic")
    print()

    raw = []

    total = len(geometries)

    for index, geometry in enumerate(
        geometries,
        start=1,
    ):
        gid = geometry["geometry_id"]

        _, normal, smooth, parts = evaluate_geometry(
            geometry
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
            "smooth_valence_energy_eV": smooth,
            "over_delta_eV": parts["over_delta_eV"],
            "angle_delta_eV": parts["angle_delta_eV"],
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
                f"smooth={smooth:+.6f}"
            )

    refs = {}

    for row in raw:
        if row["sample_kind"] == "reactant_reference":
            refs[row["system"]] = {
                "qm": row["qm_energy_eV"],
                "h": row["hstate_energy_eV"],
                "smooth": row["smooth_valence_energy_eV"],
            }

    final = []

    for row in raw:
        ref = refs[row["system"]]

        qm_rel = row["qm_energy_eV"] - ref["qm"]
        h_rel = row["hstate_energy_eV"] - ref["h"]
        smooth_rel = (
            row["smooth_valence_energy_eV"]
            - ref["smooth"]
        )

        merged = dict(row)

        merged.update({
            "qm_relative_eV": qm_rel,
            "hstate_relative_eV": h_rel,
            "smooth_valence_relative_eV": smooth_rel,
            "hstate_residual_eV": qm_rel - h_rel,
            "smooth_valence_residual_eV": qm_rel - smooth_rel,
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
            fieldnames=list(final[0].keys()),
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
        f"{'smooth MAE':>11s} {'smooth RMSE':>11s} {'smooth max':>11s}"
    )

    for system in sorted(dense):
        rows = dense[system]

        h_values = [
            row["hstate_residual_eV"]
            for row in rows
        ]

        smooth_values = [
            row["smooth_valence_residual_eV"]
            for row in rows
        ]

        print(
            f"{system:14s} "
            f"{statistics.fmean(abs(v) for v in h_values):9.4f} "
            f"{rmse(h_values):9.4f} "
            f"{max(abs(v) for v in h_values):9.4f} "
            f"{statistics.fmean(abs(v) for v in smooth_values):11.4f} "
            f"{rmse(smooth_values):11.4f} "
            f"{max(abs(v) for v in smooth_values):11.4f}"
        )

    print()
    print("WORST ADJACENT RESIDUAL STEP")

    for system in sorted(dense):
        rows = sorted(
            dense[system],
            key=lambda row: float(
                row["transfer_distance_angstrom"]
            ),
        )

        for column, label in (
            ("hstate_residual_eV", "H-state"),
            ("smooth_valence_residual_eV", "smooth"),
        ):
            pairs = []

            for left, right in zip(rows, rows[1:]):
                jump = right[column] - left[column]

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
        f"{'smooth':>9s}"
    )

    for system in sorted(refs):
        products = [
            row
            for row in final
            if (
                row["system"] == system
                and row["sample_kind"] == "product_reference"
            )
        ]

        if len(products) == 1:
            row = products[0]

            print(
                f"{system:14s} "
                f"{row['qm_relative_eV']:+9.4f} "
                f"{row['hstate_relative_eV']:+9.4f} "
                f"{row['smooth_valence_relative_eV']:+9.4f}"
            )

    print()
    print("WATER MEMBERSHIP AUDIT")

    audit_water = [
        geometry
        for geometry in geometries
        if (
            geometry["system"] == "water"
            and geometry["sample_kind"] == "dense_transfer_scan"
            and geometry.get("reaction_coordinate", {}).get(
                "transfer_distance_angstrom"
            ) is not None
            and abs(
                float(
                    geometry["reaction_coordinate"][
                        "transfer_distance_angstrom"
                    ]
                )
                - 1.08
            ) < 1e-9
        )
    ]

    if audit_water:
        water_audit(audit_water[0])

    print()
    print("PEROXIDE MEMBERSHIP AUDIT")

    for distance in (
        1.4750,
        1.8100,
        2.0635,
        2.1086,
        2.2125,
    ):
        peroxide_audit(distance)


if __name__ == "__main__":
    main()
