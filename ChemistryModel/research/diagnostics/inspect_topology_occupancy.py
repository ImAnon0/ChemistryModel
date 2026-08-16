"""
Inspect whether existing Morse geometry can distinguish real covalent bonds
from close-but-nonbonded contacts.

Diagnostic only. It does not modify ChemistryModel physics.

Run:
    py inspect_topology_occupancy.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

import reactive as R
from bond_calibration import hydrogen_peroxide_geometry
from h_state_torch import HStateReferenceBatchedSimulation


GEOMETRIES = Path("research_data/qm_residual/dense_scan_geometries.json")
BOX_SIZE = 30.0


class CaptureHState(HStateReferenceBatchedSimulation):
    def _hydrogen_state_correction(self, positions, base_per_atom):
        cached = getattr(self, "_reactive_intermediates", None)
        if cached is None or cached[0] is not positions:
            raise RuntimeError("Missing reactive intermediates")

        values = cached[1]
        self._occupancy_values = {
            key: value.detach().cpu().clone()
            for key, value in values.items()
            if isinstance(value, torch.Tensor)
        }

        return super()._hydrogen_state_correction(positions, base_per_atom)


def evaluate(symbols, coordinates):
    sim = CaptureHState(
        boxes=[(symbols, coordinates)],
        box_size=BOX_SIZE,
        target_temperature=0.0,
        friction=0.0,
        device="cpu",
        dtype=torch.float64,
        random_seed=0,
        relax_on_start=False,
    )
    return sim, sim._occupancy_values


def pair_record(sim, values, symbols, atom_i, atom_j, label, category):
    neighbours = values["neighbours"][atom_i]
    mask = values["mask"][atom_i]

    slots = [
        slot
        for slot in range(len(neighbours))
        if bool(mask[slot]) and int(neighbours[slot]) == atom_j
    ]
    if not slots:
        raise RuntimeError(f"No active neighbour slot for {label}: {atom_i}-{atom_j}")

    slot = slots[0]

    type_i = int(sim.types_numpy[atom_i])
    type_j = int(sim.types_numpy[atom_j])

    distance = float(values["distances"][atom_i, slot])
    taper = float(values["taper"][atom_i, slot])
    order = float(values["order"][atom_i, slot])

    single_r0 = float(R.BOND_LENGTH[type_i, type_j])
    single_width = float(R.BOND_WIDTH[type_i, type_j])

    current_r0 = float(values["pair_length"][atom_i, slot])
    current_width = float(values["pair_width"][atom_i, slot])

    dr = distance - single_r0
    overlap = math.exp(-single_width * dr)
    current_overlap = math.exp(-current_width * (distance - current_r0))

    return {
        "category": category,
        "label": label,
        "pair": f"{symbols[atom_i]}-{symbols[atom_j]}",
        "atoms": f"{atom_i}-{atom_j}",
        "r": distance,
        "r0": single_r0,
        "a": single_width,
        "dr": dr,
        "taper": taper,
        "overlap": overlap,
        "order": order,
        "current_r0": current_r0,
        "current_a": current_width,
        "current_overlap": current_overlap,
    }


def unique_active_pairs(values, symbols, allowed_pair=None):
    neighbours = values["neighbours"]
    mask = values["mask"]
    result = []

    for atom_i in range(len(symbols)):
        for slot in range(neighbours.shape[1]):
            if not bool(mask[atom_i, slot]):
                continue

            atom_j = int(neighbours[atom_i, slot])
            if atom_j <= atom_i or atom_j >= len(symbols):
                continue

            pair = tuple(sorted((symbols[atom_i], symbols[atom_j])))

            if allowed_pair is not None:
                if pair != tuple(sorted(allowed_pair)):
                    continue

            result.append((atom_i, atom_j))

    return result


def strongest_pair(sim, values, symbols, allowed_pair, label):
    candidates = unique_active_pairs(values, symbols, allowed_pair=allowed_pair)
    if not candidates:
        raise RuntimeError(f"No {allowed_pair} pair found for {label}")

    scored = []
    for atom_i, atom_j in candidates:
        record = pair_record(
            sim, values, symbols, atom_i, atom_j, label, "real bond"
        )
        scored.append((record["overlap"], record["taper"], -record["r"], record))

    return max(scored, key=lambda item: item[:3])[-1]


def find_reference(payload, system):
    matches = [
        g for g in payload["geometries"]
        if g["system"] == system and g["sample_kind"] == "reactant_reference"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {system} reactant reference, got {len(matches)}"
        )
    return matches[0]


def print_record(record):
    print(
        f"{record['category']:<18s} "
        f"{record['label']:<26s} "
        f"{record['pair']:<5s} "
        f"r={record['r']:7.4f}  "
        f"r0={record['r0']:7.4f}  "
        f"a={record['a']:7.4f}  "
        f"dr={record['dr']:+7.4f}  "
        f"taper={record['taper']:7.4f}  "
        f"overlap={record['overlap']:9.5f}  "
        f"order={record['order']:7.4f}"
    )


def main():
    payload = json.loads(GEOMETRIES.read_text(encoding="utf-8"))
    records = []

    reference_specs = [
        ("h3", ("H", "H"), "H3 reference H-H"),
        ("methane", ("C", "H"), "methane reference C-H"),
        ("formaldehyde", ("C", "H"), "formaldehyde reference C-H"),
        ("formaldehyde", ("C", "O"), "formaldehyde reference C-O"),
        ("water", ("O", "H"), "water reference O-H"),
    ]

    for system, pair_type, label in reference_specs:
        geometry = find_reference(payload, system)
        sim, values = evaluate(
            geometry["symbols"], geometry["coordinates_angstrom"]
        )
        records.append(
            strongest_pair(sim, values, geometry["symbols"], pair_type, label)
        )

    peroxide_symbols, peroxide_positions = hydrogen_peroxide_geometry(
        oo_distance=1.475
    )
    sim, values = evaluate(peroxide_symbols, peroxide_positions)

    peroxide = pair_record(
        sim, values, peroxide_symbols, 0, 1,
        "H2O2 real O-O", "real bond"
    )
    records.append(peroxide)

    dense_water = [
        g for g in payload["geometries"]
        if (
            g["system"] == "water"
            and g["sample_kind"] == "dense_transfer_scan"
            and g.get("reaction_coordinate", {}).get(
                "transfer_distance_angstrom"
            ) is not None
        )
    ]
    dense_water.sort(
        key=lambda g: float(
            g["reaction_coordinate"]["transfer_distance_angstrom"]
        )
    )

    for geometry in dense_water:
        x = float(
            geometry["reaction_coordinate"]["transfer_distance_angstrom"]
        )
        if not (1.08 <= x <= 1.16 + 1e-9):
            continue

        sim, values = evaluate(
            geometry["symbols"], geometry["coordinates_angstrom"]
        )

        oo_pairs = unique_active_pairs(
            values, geometry["symbols"], allowed_pair=("O", "O")
        )
        if not oo_pairs:
            raise RuntimeError(f"No O-O contact found at water x={x:.3f}")

        candidates = []
        for atom_i, atom_j in oo_pairs:
            record = pair_record(
                sim, values, geometry["symbols"], atom_i, atom_j,
                f"water encounter x={x:.3f}", "nonbonded contact"
            )
            candidates.append(
                (record["overlap"], record["taper"], -record["r"], record)
            )

        records.append(max(candidates, key=lambda item: item[:3])[-1])

    print()
    print("TOPOLOGY OCCUPANCY MICROSCOPE")
    print()

    for record in records:
        print_record(record)

    real = [r for r in records if r["category"] == "real bond"]
    nonbonded = [r for r in records if r["category"] == "nonbonded contact"]

    print()
    print("SEPARATION SUMMARY")

    if real:
        print(
            f"real bonds overlap range       : "
            f"{min(r['overlap'] for r in real):.6f} .. "
            f"{max(r['overlap'] for r in real):.6f}"
        )

    if nonbonded:
        print(
            f"nonbonded overlap range        : "
            f"{min(r['overlap'] for r in nonbonded):.6f} .. "
            f"{max(r['overlap'] for r in nonbonded):.6f}"
        )

    if real and nonbonded:
        real_floor = min(r["overlap"] for r in real)
        nonbonded_ceiling = max(r["overlap"] for r in nonbonded)
        gap = real_floor - nonbonded_ceiling

        print(f"overlap separation gap         : {gap:+.6f}")

        if gap > 0.0:
            midpoint = 0.5 * (real_floor + nonbonded_ceiling)
            print(
                f"clean gap exists; midpoint only "
                f"as diagnostic = {midpoint:.6f}"
            )
        else:
            print(
                "no clean overlap-only gap: occupancy will need "
                "additional state/environment information."
            )

    print()
    print("O-O SAME-PAIR CHECK")
    print(
        f"H2O2 O-O:        r={peroxide['r']:.4f}  "
        f"taper={peroxide['taper']:.4f}  "
        f"overlap={peroxide['overlap']:.6f}  "
        f"order={peroxide['order']:.4f}"
    )

    for record in nonbonded:
        if record["pair"] == "O-O":
            print(
                f"{record['label']:<23s} "
                f"r={record['r']:.4f}  "
                f"taper={record['taper']:.4f}  "
                f"overlap={record['overlap']:.6f}  "
                f"order={record['order']:.4f}"
            )


if __name__ == "__main__":
    main()
