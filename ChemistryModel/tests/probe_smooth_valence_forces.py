"""
Differentiable smooth-valence topology force microscope.

This is the first NON-DETACHED prototype of the smooth valence-state topology.

It does not modify production physics files. Instead it subclasses the current
H-state simulation and, during the live autograd energy evaluation:

    1. evaluates the ordinary reactive base,
    2. evaluates the existing H-state correction,
    3. builds smooth heavy-centred valence-state memberships from the SAME
       live reactive intermediates,
    4. replaces only heavy-atom overcoordination + heavy-centred angles,
    5. returns one fully differentiable energy.

Then it checks:
    - autograd force vs central finite difference,
    - force continuity through water state competition,
    - force behaviour around the O-H outer cutoff.

Run:
    py probe_smooth_valence_forces.py

No production files are changed and no output data file is required.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

import reactive as R
from batched_torch import BatchedReactiveSimulation
from h_state_torch import (
    HStateReferenceBatchedSimulation,
    _contact_overlap,
    _crowding_normalisation,
)


GEOMETRIES = Path("research_data/qm_residual/dense_scan_geometries.json")

BOX_SIZE = 30.0
MAX_LOCAL_CANDIDATES = 12

FD_STEP_A = 1.0e-5
PATH_STEP_A = 5.0e-4

# Representative locations:
#   1.08  : strongest diagnosed water topology error
#   1.16  : active state competition
#   1.32  : later residual region
#   1.535 : immediately before the known O-H outer cutoff ~1.536 A
FD_POINTS = (1.080, 1.160, 1.320, 1.535)


def states_differ_by_one_exchange(first, second):
    first_set = set(first)
    second_set = set(second)

    removed = list(first_set - second_set)
    added = list(second_set - first_set)

    if len(removed) != 1 or len(added) != 1:
        return None

    return removed[0], added[0]


class SmoothValenceForceSimulation(HStateReferenceBatchedSimulation):
    """
    H-state plus live, differentiable heavy-centred smooth valence topology.
    """

    def __init__(self, *args, **kwargs):
        # Ask reactive_torch to retain LIVE energy parts rather than only its
        # normal detached diagnostics.
        self._profile_energy_part_gradients = True

        super().__init__(*args, **kwargs)

    def energy_per_atom(self, positions):
        # Call the parent below HStateReferenceBatchedSimulation directly.
        # Calling super().energy_per_atom() here would execute H-state's
        # wrapper, which clears _reactive_intermediates before we can consume
        # them for heavy topology.
        base = BatchedReactiveSimulation.energy_per_atom(
            self,
            positions,
        )

        try:
            hydrogen_correction = self._hydrogen_state_correction(
                positions,
                base,
            )

            topology_correction = self._smooth_topology_correction(
                positions,
            )
        finally:
            self._reactive_intermediates = None

        return (
            base
            + hydrogen_correction
            + topology_correction
        )

    def _smooth_membership(self, values):
        taper = values["taper"]
        mask = values["mask"]
        pair_depth = values["pair_depth"]
        pair_width = values["pair_width"]
        shift = values["shift"]

        membership_rows = []

        hydrogen = int(R.ELEMENT_INDEX["H"])

        for atom in range(taper.shape[0]):
            row_zero = taper[atom] * 0.0

            # This correction replaces only HEAVY-centred topology.
            # Hydrogen topology remains handled by H-state.
            if int(self.types_numpy[atom]) == hydrogen:
                membership_rows.append(
                    row_zero + mask[atom]
                )
                continue

            active_slots_tensor = torch.nonzero(
                (mask[atom] > 0.0)
                & (taper[atom] > 0.0),
                as_tuple=False,
            ).flatten()

            active_slots = [
                int(slot)
                for slot in active_slots_tensor.detach().cpu().tolist()
            ]

            capacity = max(
                int(round(float(
                    self.valence[self.types[atom]]
                    .detach()
                    .cpu()
                ))),
                0,
            )

            if capacity <= 0 or not active_slots:
                membership_rows.append(row_zero)
                continue

            if len(active_slots) <= capacity:
                membership_rows.append(
                    row_zero + mask[atom]
                )
                continue

            if len(active_slots) > MAX_LOCAL_CANDIDATES:
                raise RuntimeError(
                    f"Atom {atom} has {len(active_slots)} active contacts; "
                    f"local smooth-state limit is {MAX_LOCAL_CANDIDATES}"
                )

            states = tuple(
                itertools.combinations(
                    range(len(active_slots)),
                    capacity,
                )
            )

            zero = taper.sum() * 0.0

            attractive = (
                taper[atom]
                * 2.0
                * pair_depth[atom]
                * torch.exp(
                    -pair_width[atom] * shift[atom]
                )
                * mask[atom]
            )

            diagonals = []

            for state in states:
                chosen = [
                    attractive[active_slots[local_index]]
                    for local_index in state
                ]

                diagonal = (
                    -torch.stack(chosen).sum()
                    if chosen
                    else zero
                )

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

                        old_slot = active_slots[old_local]
                        new_slot = active_slots[new_local]

                        overlap = _contact_overlap(
                            taper[atom, old_slot],
                            taper[atom, new_slot],
                        )

                        transitions[(first, second)] = (
                            old_slot,
                            new_slot,
                            overlap,
                        )

                        weighted_degree[first] = (
                            weighted_degree[first]
                            + overlap * overlap
                        )

                        weighted_degree[second] = (
                            weighted_degree[second]
                            + overlap * overlap
                        )

                normalisation = torch.stack([
                    _crowding_normalisation(value)
                    for value in weighted_degree
                ])

                couplings = {}

                for (first, second), (
                    old_slot,
                    new_slot,
                    overlap,
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
                        self.h_state_mixing
                        * depth_scale
                        * overlap
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

                _, eigenvectors = torch.linalg.eigh(
                    hamiltonian
                )

                ground = eigenvectors[:, 0]

                probabilities = ground * ground

                probabilities = (
                    probabilities
                    / torch.clamp(
                        probabilities.sum(),
                        min=1e-12,
                    )
                )

            slot_values = [zero for _ in range(taper.shape[1])]

            for local_index, slot in enumerate(active_slots):
                present = [
                    probabilities[state_index]
                    for state_index, state in enumerate(states)
                    if local_index in state
                ]

                slot_values[slot] = (
                    torch.stack(present).sum()
                    if present
                    else zero
                )

            membership_rows.append(
                torch.stack(slot_values)
            )

        return torch.stack(membership_rows)

    def _smooth_topology_correction(self, positions):
        cached = getattr(
            self,
            "_reactive_intermediates",
            None,
        )

        if cached is None or cached[0] is not positions:
            raise RuntimeError(
                "Smooth topology correction missing live "
                "reactive intermediates"
            )

        values = cached[1]

        taper = values["taper"]
        order = values["order"]
        mask = values["mask"]
        neighbours = values["neighbours"]
        distances = values["distances"]
        unsoftened_depth = values["unsoftened_depth"]

        membership = self._smooth_membership(values)

        topology_taper = taper * membership

        hydrogen = int(R.ELEMENT_INDEX["H"])

        heavy = (
            self.types != hydrogen
        ).to(self.dtype)

        # ------------------------------------------------------------
        # New heavy-atom overcoordination
        # ------------------------------------------------------------

        topology_coordination = torch.sum(
            topology_taper,
            dim=1,
        )

        valence = self.valence[self.types]

        topology_excess = torch.clamp(
            topology_coordination - valence,
            min=0.0,
        )

        topology_over_scale = self.over_coordination_scale(
            topology_taper,
            unsoftened_depth,
            mask,
            cache_key=None,
        )

        topology_over = (
            self.over_penalty
            * topology_over_scale
            * topology_excess ** 2
        )

        # ------------------------------------------------------------
        # New heavy-centred angles
        # ------------------------------------------------------------

        topology_bonded_order = torch.sum(
            topology_taper * order,
            dim=1,
        )

        outer = self.outer_electrons[self.types]

        topology_lone_pairs = torch.clamp(
            (outer - topology_bonded_order) / 2.0,
            min=0.0,
        )

        topology_steric = torch.clamp(
            topology_coordination
            + topology_lone_pairs,
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
            - self.lone_pair_squeeze
            * topology_lone_pairs
        )

        stiffness = self.angle_stiffness[self.types]

        gathered = self._gather_neighbours(
            positions,
            neighbours,
            "positions",
        )

        offsets = gathered - positions[:, None, :]

        offsets = (
            offsets
            - self.box_size
            * torch.round(
                offsets / self.box_size
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

        upper_triangle = getattr(
            self,
            "_angle_upper_triangle",
            None,
        )

        if upper_triangle is None:
            upper_triangle = torch.triu(
                torch.ones(
                    weight.shape[1],
                    weight.shape[2],
                    device=self.device,
                    dtype=self.dtype,
                ),
                diagonal=1,
            )

            self._angle_upper_triangle = upper_triangle

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

        topology_angle = torch.sum(
            topology_angle_energy,
            dim=(1, 2),
        )

        # reactive_torch exposed the original LIVE tensors for exactly this
        # diagnostic/profiling purpose.
        original_parts = getattr(
            self,
            "_profile_energy_parts",
            None,
        )

        if original_parts is None:
            raise RuntimeError(
                "reactive_torch did not expose live profile energy parts"
            )

        original_over = original_parts["over"]
        original_angle = original_parts["angle"]

        # Replace only heavy-centred old topology with heavy-centred new
        # topology. H-state already handles ordinary H overcoordination.
        return heavy * (
            topology_over
            - original_over
            + topology_angle
            - original_angle
        )


def load_dense_water():
    payload = json.loads(
        GEOMETRIES.read_text(encoding="utf-8")
    )

    rows = [
        geometry
        for geometry in payload["geometries"]
        if (
            geometry["system"] == "water"
            and geometry["sample_kind"] == "dense_transfer_scan"
            and geometry.get("reaction_coordinate", {}).get(
                "transfer_distance_angstrom"
            ) is not None
        )
    ]

    rows.sort(
        key=lambda geometry: float(
            geometry["reaction_coordinate"][
                "transfer_distance_angstrom"
            ]
        )
    )

    return rows


def interpolate_geometry(dense, x):
    for left, right in zip(dense, dense[1:]):
        x0 = float(
            left["reaction_coordinate"][
                "transfer_distance_angstrom"
            ]
        )

        x1 = float(
            right["reaction_coordinate"][
                "transfer_distance_angstrom"
            ]
        )

        if x0 - 1e-12 <= x <= x1 + 1e-12:
            fraction = (
                0.0
                if x1 == x0
                else (x - x0) / (x1 - x0)
            )

            p0 = np.asarray(
                left["coordinates_angstrom"],
                dtype=float,
            )

            p1 = np.asarray(
                right["coordinates_angstrom"],
                dtype=float,
            )

            positions = (
                p0 + fraction * (p1 - p0)
            )

            return {
                "symbols": list(left["symbols"]),
                "coordinates_angstrom": positions,
                "x": float(x),
            }

    raise ValueError(
        f"x={x:.6f} outside dense water path"
    )


def build_simulation(geometry):
    return SmoothValenceForceSimulation(
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


def live_energy_and_force(simulation, coordinates):
    positions = torch.tensor(
        np.asarray(coordinates, dtype=float),
        dtype=torch.float64,
        device="cpu",
        requires_grad=True,
    )

    energy = simulation.energy_per_atom(
        positions
    ).sum()

    gradient = torch.autograd.grad(
        energy,
        positions,
        create_graph=False,
        retain_graph=False,
    )[0]

    force = -gradient

    return (
        float(energy.detach().cpu()),
        force.detach().cpu().numpy(),
    )


def finite_difference_force(
    simulation,
    coordinates,
    step=FD_STEP_A,
):
    coordinates = np.asarray(
        coordinates,
        dtype=float,
    )

    result = np.zeros_like(coordinates)

    for atom in range(coordinates.shape[0]):
        for axis in range(3):
            plus = coordinates.copy()
            minus = coordinates.copy()

            plus[atom, axis] += step
            minus[atom, axis] -= step

            plus_energy, _ = live_energy_and_force(
                simulation,
                plus,
            )

            minus_energy, _ = live_energy_and_force(
                simulation,
                minus,
            )

            result[atom, axis] = -(
                plus_energy - minus_energy
            ) / (2.0 * step)

    return result


def force_error_metrics(analytic, numeric):
    difference = analytic - numeric

    max_abs = float(
        np.max(np.abs(difference))
    )

    rms = float(
        np.sqrt(np.mean(difference ** 2))
    )

    reference = max(
        float(np.max(np.abs(numeric))),
        1e-12,
    )

    return {
        "max_abs": max_abs,
        "rms": rms,
        "max_reference": reference,
        "max_relative": max_abs / reference,
    }


def force_continuity_window(dense, centre, half_width, step):
    xs = np.arange(
        centre - half_width,
        centre + half_width + 0.5 * step,
        step,
    )

    rows = []

    # One simulation is enough because this tiny microscope keeps the same
    # atom list and only applies very small geometric changes.
    initial = interpolate_geometry(
        dense,
        float(xs[0]),
    )

    simulation = build_simulation(initial)

    for x in xs:
        geometry = interpolate_geometry(
            dense,
            float(x),
        )

        energy, force = live_energy_and_force(
            simulation,
            geometry["coordinates_angstrom"],
        )

        rows.append({
            "x": float(x),
            "energy": energy,
            "force": force,
        })

    worst_force_jump = None
    worst_energy_step = None

    for left, right in zip(rows, rows[1:]):
        force_jump = float(
            np.max(
                np.abs(
                    right["force"]
                    - left["force"]
                )
            )
        )

        energy_step = (
            right["energy"]
            - left["energy"]
        )

        candidate_force = (
            force_jump,
            left["x"],
            right["x"],
        )

        candidate_energy = (
            abs(energy_step),
            energy_step,
            left["x"],
            right["x"],
        )

        if (
            worst_force_jump is None
            or candidate_force[0]
            > worst_force_jump[0]
        ):
            worst_force_jump = candidate_force

        if (
            worst_energy_step is None
            or candidate_energy[0]
            > worst_energy_step[0]
        ):
            worst_energy_step = candidate_energy

    return rows, worst_force_jump, worst_energy_step


def main():
    dense = load_dense_water()

    print("DIFFERENTIABLE SMOOTH-VALENCE FORCE MICROSCOPE")
    print()
    print("engine modification : none")
    print("device              : CPU / float64")
    print(f"finite diff step    : {FD_STEP_A:.1e} A")
    print()

    print("AUTOGRAD VS CENTRAL FINITE DIFFERENCE")
    print(
        f"{'x / A':>8s} "
        f"{'E / eV':>13s} "
        f"{'max |F|':>12s} "
        f"{'max abs err':>13s} "
        f"{'RMS err':>12s} "
        f"{'max rel':>11s}"
    )

    for x in FD_POINTS:
        geometry = interpolate_geometry(
            dense,
            x,
        )

        simulation = build_simulation(
            geometry
        )

        energy, analytic = live_energy_and_force(
            simulation,
            geometry["coordinates_angstrom"],
        )

        numeric = finite_difference_force(
            simulation,
            geometry["coordinates_angstrom"],
        )

        metrics = force_error_metrics(
            analytic,
            numeric,
        )

        print(
            f"{x:8.3f} "
            f"{energy:13.6f} "
            f"{np.max(np.abs(analytic)):12.6f} "
            f"{metrics['max_abs']:13.6e} "
            f"{metrics['rms']:12.6e} "
            f"{metrics['max_relative']:11.3e}"
        )

    print()
    print("FORCE CONTINUITY — STATE COMPETITION REGION")

    _, worst_force, worst_energy = force_continuity_window(
        dense,
        centre=1.160,
        half_width=0.010,
        step=PATH_STEP_A,
    )

    print(
        f"window              : "
        f"1.150 -> 1.170 A in {PATH_STEP_A:.4f} A steps"
    )

    print(
        f"worst max-component force jump : "
        f"{worst_force[1]:.4f}->{worst_force[2]:.4f} A  "
        f"{worst_force[0]:.6e} eV/A"
    )

    print(
        f"worst energy step             : "
        f"{worst_energy[2]:.4f}->{worst_energy[3]:.4f} A  "
        f"{worst_energy[1]:+.6e} eV"
    )

    print()
    print("FORCE CONTINUITY — O-H CUTOFF REGION")

    _, worst_force, worst_energy = force_continuity_window(
        dense,
        centre=1.536,
        half_width=0.004,
        step=PATH_STEP_A,
    )

    print(
        f"window              : "
        f"1.532 -> 1.540 A in {PATH_STEP_A:.4f} A steps"
    )

    print(
        f"worst max-component force jump : "
        f"{worst_force[1]:.4f}->{worst_force[2]:.4f} A  "
        f"{worst_force[0]:.6e} eV/A"
    )

    print(
        f"worst energy step             : "
        f"{worst_energy[2]:.4f}->{worst_energy[3]:.4f} A  "
        f"{worst_energy[1]:+.6e} eV"
    )


if __name__ == "__main__":
    main()
