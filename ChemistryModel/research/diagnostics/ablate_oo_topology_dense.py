"""
Temporary O-O topology ablation for the water dense QM microscope.

Purpose
-------
Test one hypothesis only:

    A nearby O...O radial contact should keep its ordinary radial/H-state
    interaction, but should not automatically count as chemical topology
    for heavy-atom overcoordination or angle construction.

This script DOES NOT modify ChemistryModel physics or parameter files.

It evaluates the normal H-state model, then post-hoc replaces only:
    - overcoordination computed from O-O-excluded topology taper
    - angles computed from O-O-excluded topology taper

Everything else remains exactly the H-state result:
    - radial Morse terms
    - bond-order interpolation used by the radial potential
    - environment softening
    - H-state energies/mixing
    - H-pair subtraction
    - all non-O-O topology

Run on sapt-h-state:

    py ablate_oo_topology_dense.py

Output:
    research_data/qm_residual/dense_scan_oo_topology_ablation.csv
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch

import reactive as R
from h_state_torch import HStateReferenceBatchedSimulation


GEOMETRIES = Path("research_data/qm_residual/dense_scan_geometries.json")
QM_RESULTS = Path("research_data/qm_residual/dense_scan_qm.csv")
OUTPUT = Path("research_data/qm_residual/dense_scan_oo_topology_ablation.csv")

BOX_SIZE = 30.0


class CaptureHState(HStateReferenceBatchedSimulation):
    """Normal H-state model with detached read-only reactive intermediates."""

    def _hydrogen_state_correction(self, positions, base_per_atom):
        cached = getattr(self, "_reactive_intermediates", None)

        if cached is None or cached[0] is not positions:
            raise RuntimeError("Missing reactive intermediates")

        values = cached[1]

        self._ablation_values = {
            key: value.detach().clone()
            for key, value in values.items()
            if isinstance(value, torch.Tensor)
        }

        return super()._hydrogen_state_correction(
            positions,
            base_per_atom,
        )


def load_qm(path: Path) -> dict[str, float]:
    rows = {}

    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            rows[row["geometry_id"]] = float(row["qm_energy_eV"])

    return rows


def rmse(values):
    return math.sqrt(sum(value * value for value in values) / len(values))


def oo_topology_counterfactual(simulation: CaptureHState):
    """
    Recompute only topology-derived overcoordination + angle terms after
    removing O-O contacts from topology.

    The original radial taper/order/Morse/H-state terms are not changed.
    """

    values = simulation._ablation_values

    taper = values["taper"]
    order = values["order"]
    mask = values["mask"]
    neighbours = values["neighbours"]
    centre_types = values["centre_types"]
    other_types = values["other_types"]
    distances = values["distances"]
    unsoftened_depth = values["unsoftened_depth"]

    oxygen = int(R.ELEMENT_INDEX["O"])

    oo_contact = (
        (centre_types == oxygen)
        & (other_types == oxygen)
        & (mask > 0.0)
    )

    topology_taper = torch.where(
        oo_contact,
        torch.zeros_like(taper),
        taper,
    )

    # ------------------------------------------------------------
    # Heavy/topological overcoordination counterfactual
    # ------------------------------------------------------------

    topology_coordination = torch.sum(topology_taper, dim=1)
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

    # ------------------------------------------------------------
    # Angle counterfactual
    # Mirrors reactive_torch.py, changing only the topology taper.
    # ------------------------------------------------------------

    topology_bonded_order = torch.sum(
        topology_taper * order,
        dim=1,
    )

    outer = simulation.outer_electrons[simulation.types]

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

    stiffness = simulation.angle_stiffness[simulation.types]

    positions = simulation.positions.detach()

    gathered = simulation._gather_neighbours(
        positions,
        neighbours,
        "positions",
    )

    offsets = gathered - positions[:, None, :]

    offsets = offsets - simulation.box_size * torch.round(
        offsets / simulation.box_size
    )

    # Use the exact distances already evaluated by the base model for the
    # denominator, matching the production angle equation.
    left = offsets[:, :, None, :]
    right = offsets[:, None, :, :]

    dot = torch.sum(left * right, dim=3)

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

    taper_difference = first_taper - second_taper

    weaker_taper = 0.5 * (
        first_taper
        + second_taper
        - torch.sqrt(taper_difference ** 2 + 1e-8)
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

    weight = angle_pair_taper * angle_engagement

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
        * (angle - topology_rest[:, None, None]) ** 2
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

    original_over = float(
        original_over_per_atom.sum().detach().cpu()
    )
    topology_over = float(
        topology_over_per_atom.sum().detach().cpu()
    )

    original_angle = float(
        original_angle_per_atom.sum().detach().cpu()
    )
    topology_angle = float(
        topology_angle_per_atom.sum().detach().cpu()
    )

    over_delta = topology_over - original_over
    angle_delta = topology_angle - original_angle

    # Auditing information: O-O taper in each directed O row.
    oo_tapers = taper[oo_contact].detach().cpu().tolist()

    return {
        "original_over_eV": original_over,
        "ablated_over_eV": topology_over,
        "over_delta_eV": over_delta,
        "original_angle_eV": original_angle,
        "ablated_angle_eV": topology_angle,
        "angle_delta_eV": angle_delta,
        "oo_taper_sum": float(sum(oo_tapers)),
        "oo_directed_contacts": len(oo_tapers),
    }


def evaluate(geometry):
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

    normal_hstate = float(simulation.potential_per_box[0])

    counterfactual = oo_topology_counterfactual(simulation)

    ablated_hstate = (
        normal_hstate
        + counterfactual["over_delta_eV"]
        + counterfactual["angle_delta_eV"]
    )

    return normal_hstate, ablated_hstate, counterfactual


def main():
    payload = json.loads(
        GEOMETRIES.read_text(encoding="utf-8")
    )

    qm = load_qm(QM_RESULTS)

    water = [
        geometry
        for geometry in payload["geometries"]
        if geometry["system"] == "water"
    ]

    missing = [
        geometry["geometry_id"]
        for geometry in water
        if geometry["geometry_id"] not in qm
    ]

    if missing:
        raise RuntimeError(
            f"QM missing {len(missing)} water rows; first={missing[0]}"
        )

    print("O-O TOPOLOGY ABLATION — WATER")
    print("radial/H-state physics : unchanged")
    print("O-O overcoord topology : removed")
    print("O-O angle topology     : removed")
    print()

    raw = []

    for index, geometry in enumerate(water, start=1):
        gid = geometry["geometry_id"]

        normal, ablated, parts = evaluate(geometry)

        rc = geometry.get("reaction_coordinate", {})

        row = {
            "geometry_id": gid,
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
            "ablated_hstate_energy_eV": ablated,
            **parts,
        }

        raw.append(row)

        if (
            geometry["sample_kind"] != "dense_transfer_scan"
            or (
                row["transfer_distance_angstrom"] != ""
                and 1.06 <= float(row["transfer_distance_angstrom"]) <= 1.16
            )
        ):
            x_text = (
                f"{float(row['transfer_distance_angstrom']):.3f}"
                if row["transfer_distance_angstrom"] != ""
                else "-"
            )

            print(
                f"{gid:<34s} x={x_text:>5s}  "
                f"H={normal:+.6f}  "
                f"abl={ablated:+.6f}  "
                f"dOver={parts['over_delta_eV']:+.6f}  "
                f"dAngle={parts['angle_delta_eV']:+.6f}"
            )

    references = [
        row
        for row in raw
        if row["sample_kind"] == "reactant_reference"
    ]

    if len(references) != 1:
        raise RuntimeError(
            f"Expected one water reactant reference, got {len(references)}"
        )

    reference = references[0]

    final = []

    for row in raw:
        merged = dict(row)

        qm_rel = (
            row["qm_energy_eV"]
            - reference["qm_energy_eV"]
        )

        h_rel = (
            row["hstate_energy_eV"]
            - reference["hstate_energy_eV"]
        )

        ablated_rel = (
            row["ablated_hstate_energy_eV"]
            - reference["ablated_hstate_energy_eV"]
        )

        merged.update({
            "qm_relative_eV": qm_rel,
            "hstate_relative_eV": h_rel,
            "ablated_hstate_relative_eV": ablated_rel,
            "hstate_residual_eV": qm_rel - h_rel,
            "ablated_residual_eV": qm_rel - ablated_rel,
        })

        final.append(merged)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

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

    dense = [
        row
        for row in final
        if row["sample_kind"] == "dense_transfer_scan"
    ]

    dense.sort(
        key=lambda row: float(
            row["transfer_distance_angstrom"]
        )
    )

    normal_residuals = [
        row["hstate_residual_eV"]
        for row in dense
    ]

    ablated_residuals = [
        row["ablated_residual_eV"]
        for row in dense
    ]

    print()
    print(f"wrote : {OUTPUT}")
    print()

    print("DENSE TRANSFER RESIDUALS VS QM")
    print(
        f"{'model':18s} {'MAE':>10s} {'RMSE':>10s} {'max':>10s}"
    )

    for label, values in (
        ("H-state", normal_residuals),
        ("O-O topology off", ablated_residuals),
    ):
        print(
            f"{label:18s} "
            f"{statistics.fmean(abs(v) for v in values):10.4f} "
            f"{rmse(values):10.4f} "
            f"{max(abs(v) for v in values):10.4f}"
        )

    print()
    print("WORST ADJACENT RESIDUAL STEP")

    for key, label in (
        ("hstate_residual_eV", "H-state"),
        ("ablated_residual_eV", "O-O topology off"),
    ):
        candidates = []

        for left_row, right_row in zip(dense, dense[1:]):
            jump = right_row[key] - left_row[key]

            candidates.append((
                abs(jump),
                jump,
                left_row,
                right_row,
            ))

        _, jump, left_row, right_row = max(
            candidates,
            key=lambda item: item[0],
        )

        print(
            f"{label:18s} "
            f"{float(left_row['transfer_distance_angstrom']):.3f}->"
            f"{float(right_row['transfer_distance_angstrom']):.3f} A  "
            f"dResidual={jump:+.6f} eV"
        )

    print()
    print("MICROSCOPE — ADJACENT STEPS 1.06 TO 1.16 A")

    local_rows = [
        row
        for row in dense
        if 1.06 <= float(row["transfer_distance_angstrom"]) <= 1.16
    ]

    for left_row, right_row in zip(local_rows, local_rows[1:]):
        left_x = float(
            left_row["transfer_distance_angstrom"]
        )
        right_x = float(
            right_row["transfer_distance_angstrom"]
        )

        d_qm = (
            right_row["qm_relative_eV"]
            - left_row["qm_relative_eV"]
        )

        d_h = (
            right_row["hstate_relative_eV"]
            - left_row["hstate_relative_eV"]
        )

        d_ablated = (
            right_row["ablated_hstate_relative_eV"]
            - left_row["ablated_hstate_relative_eV"]
        )

        d_over_delta = (
            right_row["over_delta_eV"]
            - left_row["over_delta_eV"]
        )

        d_angle_delta = (
            right_row["angle_delta_eV"]
            - left_row["angle_delta_eV"]
        )

        normal_residual_step = d_qm - d_h
        ablated_residual_step = d_qm - d_ablated

        print(
            f"{left_x:.3f}->{right_x:.3f} A  "
            f"dQM={d_qm:+.6f}  "
            f"dH={d_h:+.6f}  "
            f"dAbl={d_ablated:+.6f}  "
            f"dOverFix={d_over_delta:+.6f}  "
            f"dAngleFix={d_angle_delta:+.6f}  "
            f"resStep {normal_residual_step:+.6f}"
            f" -> {ablated_residual_step:+.6f}"
        )

    products = [
        row
        for row in final
        if row["sample_kind"] == "product_reference"
    ]

    if len(products) == 1:
        product = products[0]

        print()
        print("PRODUCT REFERENCE ENERGY")
        print(
            f"QM               {product['qm_relative_eV']:+.6f} eV"
        )
        print(
            f"H-state          {product['hstate_relative_eV']:+.6f} eV"
        )
        print(
            f"O-O topology off {product['ablated_hstate_relative_eV']:+.6f} eV"
        )


if __name__ == "__main__":
    main()
