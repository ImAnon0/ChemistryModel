"""
Experimental general valence-state topology engine.

This module extends the existing H-state engine with a smooth, local
valence-state model for HEAVY-ATOM chemical topology.

Why this exists
---------------
The base reactive model uses radial contact taper both for physical pair
interaction and for chemical topology. That means a close nonbonded contact
can incorrectly consume valence, create overcoordination, and generate angles.

The valence-state layer separates those ideas:

    radial taper
        -> physical contact / radial interaction

    radial candidates
        -> allowed local valence states
        -> smooth Hamiltonian mixing
        -> edge membership
        -> topology taper = radial taper * membership
        -> coordination / heavy overcoordination / heavy-centred angles

Hydrogen remains handled by h_state_torch.py. This module generalises the same
state-mixing idea to heavy-centred topology.

Validated before promotion into this module
-------------------------------------------
The scratch implementation was checked against:
    - dense QM residual scans for H3, water, formaldehyde, methane
    - genuine H2O2 O-O topology
    - hard top-V topology ablation
    - dense 0.001 A continuity scans
    - autograd vs central finite-difference forces (~1e-8 eV/A agreement)
    - short thermostat-free NVE trajectories

This is still EXPERIMENTAL. In particular, local state enumeration is
combinatorial and is not yet suitable for unrestricted dense production runs.
"""

from __future__ import annotations

import itertools
import math

import torch

import reactive as R
from batched_torch import BatchedReactiveSimulation
from h_state_torch import (
    HStateReferenceBatchedSimulation,
    _contact_overlap,
    _crowding_normalisation,
)


VALENCE_STATE_MODEL_NAME = "reactive_v3_valence_state_experimental"
VALENCE_STATE_MODEL_REVISION = 0

# Safety limits for the research implementation. These are not chemistry
# parameters; they prevent accidental combinatorial explosions before a more
# efficient state solver is implemented.
MAX_LOCAL_CANDIDATES = 12
MAX_LOCAL_STATES = 128


def _states_differ_by_one_exchange(first, second):
    """Return exchanged local contact indices for two valence states."""

    first_set = set(first)
    second_set = set(second)

    removed = list(first_set - second_set)
    added = list(second_set - first_set)

    if len(removed) != 1 or len(added) != 1:
        return None

    return removed[0], added[0]


class ValenceStateBatchedSimulation(HStateReferenceBatchedSimulation):
    """
    H-state reactive model plus smooth heavy-atom valence-state topology.

    The radial Morse/H-state energy is unchanged. Only the heavy-centred
    topology-dependent pieces are replaced:
        - heavy-atom coordination used by overcoordination
        - heavy-centred electron-domain counting
        - heavy-centred angle engagement / angle energy
    """

    physics_model_name = VALENCE_STATE_MODEL_NAME
    physics_model_revision = VALENCE_STATE_MODEL_REVISION

    def __init__(self, *args, **kwargs):
        # reactive_torch normally keeps detached diagnostic energy parts.
        # We also need the live tensors so the old heavy topology terms can
        # be replaced without breaking autograd.
        self._profile_energy_part_gradients = True

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Energy composition
    # ------------------------------------------------------------------

    def energy_per_atom(self, positions):
        """
        Evaluate base + H-state + differentiable heavy valence topology.

        We deliberately call BatchedReactiveSimulation.energy_per_atom
        directly rather than HStateReferenceBatchedSimulation.energy_per_atom,
        because the H-state wrapper clears _reactive_intermediates after its
        correction. The heavy topology layer needs those same live tensors.
        """

        base = BatchedReactiveSimulation.energy_per_atom(
            self,
            positions,
        )

        try:
            hydrogen_correction = self._hydrogen_state_correction(
                positions,
                base,
            )

            topology_correction = self._valence_topology_correction(
                positions,
            )
        finally:
            self._reactive_intermediates = None

        return (
            base
            + hydrogen_correction
            + topology_correction
        )

    # ------------------------------------------------------------------
    # Local valence-state mixing
    # ------------------------------------------------------------------

    def _local_valence_membership(self, values):
        """
        Return directed per-centre contact membership in [0, 1].

        For a heavy atom with N active candidate contacts and elemental
        valence V:

            N <= V:
                every active contact has membership 1

            N > V:
                enumerate all size-V contact sets
                build an H-state-style Hamiltonian
                diagonalise it
                convert ground-state probabilities into expected contact
                membership

        The returned tensor has the same shape as values["taper"].
        """

        taper = values["taper"]
        mask = values["mask"]

        pair_depth = values["pair_depth"]
        pair_width = values["pair_width"]
        shift = values["shift"]

        hydrogen = int(R.ELEMENT_INDEX["H"])

        membership_rows = []

        for atom in range(taper.shape[0]):
            row_zero = taper[atom] * 0.0

            # This layer replaces only HEAVY-centred topology.
            # H-state remains responsible for hydrogen valence physics.
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
                int(
                    round(
                        float(
                            self.valence[self.types[atom]]
                            .detach()
                            .cpu()
                        )
                    )
                ),
                0,
            )

            if capacity <= 0 or not active_slots:
                membership_rows.append(row_zero)
                continue

            # No competition for valence: all contacts can participate in
            # topology exactly as before.
            if len(active_slots) <= capacity:
                membership_rows.append(
                    row_zero + mask[atom]
                )
                continue

            if len(active_slots) > MAX_LOCAL_CANDIDATES:
                raise RuntimeError(
                    "valence-state topology has "
                    f"{len(active_slots)} active contacts around atom {atom}; "
                    f"research limit is {MAX_LOCAL_CANDIDATES}"
                )

            state_count = math.comb(
                len(active_slots),
                capacity,
            )

            if state_count > MAX_LOCAL_STATES:
                raise RuntimeError(
                    "valence-state topology would require "
                    f"{state_count} local states around atom {atom} "
                    f"({len(active_slots)} candidates, valence {capacity}); "
                    f"research limit is {MAX_LOCAL_STATES}"
                )

            states = tuple(
                itertools.combinations(
                    range(len(active_slots)),
                    capacity,
                )
            )

            zero = taper.sum() * 0.0

            # Same attractive magnitude used by the H-state state diagonals.
            # The common repulsive core is state-independent and therefore
            # does not affect the eigenvectors / memberships.
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
                selected = [
                    attractive[
                        active_slots[local_index]
                    ]
                    for local_index in state
                ]

                diagonals.append(
                    -torch.stack(selected).sum()
                    if selected
                    else zero
                )

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
                        exchange = _states_differ_by_one_exchange(
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

                    couplings[(first, second)] = (
                        self.h_state_mixing
                        * depth_scale
                        * overlap
                        / denominator
                    )

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

                # Eigenvector sign is arbitrary, but |c_s|^2 is invariant.
                probabilities = ground * ground

                probabilities = (
                    probabilities
                    / torch.clamp(
                        probabilities.sum(),
                        min=1e-12,
                    )
                )

            slot_values = [
                zero for _ in range(taper.shape[1])
            ]

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

    # ------------------------------------------------------------------
    # Heavy topology replacement
    # ------------------------------------------------------------------

    def _valence_topology_correction(self, positions):
        cached = getattr(
            self,
            "_reactive_intermediates",
            None,
        )

        if cached is None or cached[0] is not positions:
            raise RuntimeError(
                "valence-state topology missing live reactive intermediates"
            )

        values = cached[1]

        taper = values["taper"]
        order = values["order"]
        mask = values["mask"]
        neighbours = values["neighbours"]
        distances = values["distances"]
        unsoftened_depth = values["unsoftened_depth"]

        membership = self._local_valence_membership(
            values
        )

        topology_taper = taper * membership

        hydrogen = int(R.ELEMENT_INDEX["H"])

        heavy = (
            self.types != hydrogen
        ).to(self.dtype)

        # --------------------------------------------------------------
        # Heavy-atom overcoordination
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Heavy-centred electron domains and angles
        # --------------------------------------------------------------

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

        original_parts = getattr(
            self,
            "_profile_energy_parts",
            None,
        )

        if original_parts is None:
            raise RuntimeError(
                "reactive_torch did not expose live energy parts"
            )

        original_over = original_parts["over"]
        original_angle = original_parts["angle"]

        # H-state already replaces the hydrogen share of ordinary
        # overcoordination. This layer changes only heavy-centred topology.
        return heavy * (
            topology_over
            - original_over
            + topology_angle
            - original_angle
        )


# Short alias matching the naming convention used by other engine modules.
ValenceStateSimulation = ValenceStateBatchedSimulation
