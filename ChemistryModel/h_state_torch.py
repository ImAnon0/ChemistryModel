"""
Differentiable Torch adapter for the experimental v2 hydrogen-state model.

REFERENCE ONLY.

This does not modify reactive_torch.py or production ChemistryModel physics.

It keeps the existing base potential for:
    - heavy-heavy bonding
    - bond-order interpolation
    - environment softening
    - heavy-atom over-coordination
    - angles
    - all existing parameter tables

For hydrogen-containing pairs it replaces:

    every contact gets a full Morse interaction
    + hydrogen over-coordination punishment

with:

    allowed one-valence H bonding states
    + smooth mixing between states differing by transfer of one H bond

The current taper/Morse decomposition is deliberately retained for this
experiment. Nonbonded repulsion/cutoff redesign is a later, separate step.
"""

import itertools

import numpy as np
import torch

import reactive as R

from batched_torch import BatchedReactiveSimulation

from h_state_reference import (
    H_STATE_MIXING,
    CROWDING_TRANSITION_WIDTH,
)


H_STATE_MODEL_NAME = "reactive_v2_h_state_reference"
H_STATE_MODEL_REVISION = 0

MAX_REFERENCE_EDGES = 18


def _state_is_valid(edge_indices, edge_atoms, types):
    """Hydrogen may occur in at most one occupied covalent edge."""

    hydrogen = int(R.ELEMENT_INDEX["H"])
    used = set()

    for edge_index in edge_indices:
        first, second = edge_atoms[edge_index]

        for atom in (first, second):
            if int(types[atom]) != hydrogen:
                continue

            if atom in used:
                return False

            used.add(atom)

    return True


def _maximal_states(edge_atoms, types):
    """Enumerate maximal H matchings for one small reference box."""

    edge_count = len(edge_atoms)

    if edge_count > MAX_REFERENCE_EDGES:
        raise RuntimeError(
            "h-state reference has "
            f"{edge_count} active H-containing edges; "
            f"limit is {MAX_REFERENCE_EDGES}"
        )

    valid = []

    for count in range(edge_count + 1):
        for chosen in itertools.combinations(range(edge_count), count):
            if _state_is_valid(chosen, edge_atoms, types):
                valid.append(tuple(chosen))

    maximal = []

    for state in valid:
        can_extend = False

        for edge_index in range(edge_count):
            if edge_index in state:
                continue

            candidate = state + (edge_index,)

            if _state_is_valid(candidate, edge_atoms, types):
                can_extend = True
                break

        if not can_extend:
            maximal.append(state)

    if not maximal:
        return (tuple(),)

    return tuple(maximal)


def _single_h_transfer(first_state, second_state, edge_atoms, types):
    """Return the two exchanged edges if states differ by one H transfer."""

    first = set(first_state)
    second = set(second_state)

    removed = list(first - second)
    added = list(second - first)

    if len(removed) != 1 or len(added) != 1:
        return None

    old_index = removed[0]
    new_index = added[0]

    old_atoms = set(edge_atoms[old_index])
    new_atoms = set(edge_atoms[new_index])

    shared = old_atoms.intersection(new_atoms)

    hydrogen = int(R.ELEMENT_INDEX["H"])

    shared_hydrogens = [
        atom
        for atom in shared
        if int(types[atom]) == hydrogen
    ]

    if len(shared_hydrogens) != 1:
        return None

    return old_index, new_index, shared_hydrogens[0]


def _contact_overlap(first_taper, second_taper):
    """Balanced simultaneous-contact gate used for state mixing."""

    total = first_taper + second_taper

    balance = (
        4.0
        * first_taper
        * second_taper
        / torch.clamp(total * total, min=1e-12)
    )

    return (
        torch.sqrt(
            torch.clamp(
                first_taper * second_taper,
                min=0.0,
            )
        )
        * balance
    )


def _crowding_normalisation(degree):
    """C1 normalisation preventing free N-state resonance stabilisation."""

    width = float(CROWDING_TRANSITION_WIDTH)

    fraction = torch.clamp(
        (degree - 1.0) / width,
        0.0,
        1.0,
    )

    smooth = (
        fraction * fraction
        * (3.0 - 2.0 * fraction)
    )

    transition = 1.0 + smooth * (degree - 1.0)

    return torch.where(
        degree <= 1.0,
        torch.ones_like(degree),
        torch.where(
            degree >= 1.0 + width,
            degree,
            transition,
        ),
    )


class HStateReferenceBatchedSimulation(BatchedReactiveSimulation):
    """Batched base potential with experimental one-valence H states."""

    physics_model_name = H_STATE_MODEL_NAME
    physics_model_revision = H_STATE_MODEL_REVISION

    def __init__(
        self,
        *args,
        h_state_mixing=None,
        **kwargs,
    ):
        self.h_state_mixing = float(
            H_STATE_MIXING
            if h_state_mixing is None
            else h_state_mixing
        )

        # Ask reactive_torch to expose exactly the pair quantities it used
        # when evaluating the base energy.
        self._share_reactive_intermediates = True

        super().__init__(*args, **kwargs)

    def energy_per_atom(self, positions):
        base = super().energy_per_atom(positions)

        try:
            correction = self._hydrogen_state_correction(
                positions,
                base,
            )
        finally:
            self._reactive_intermediates = None

        return base + correction

    def _active_edges_for_box(
        self,
        box,
        values,
        neighbours_numpy,
        active_numpy,
    ):
        """H-containing directed neighbour entries, deduplicated by pair."""

        start = box * self.per_box
        stop = start + self.per_box

        hydrogen = int(R.ELEMENT_INDEX["H"])

        found = {}

        for centre in range(start, stop):
            for slot in range(neighbours_numpy.shape[1]):
                if not active_numpy[centre, slot]:
                    continue

                other = int(neighbours_numpy[centre, slot])

                if not start <= other < stop:
                    continue

                first = min(centre, other)
                second = max(centre, other)

                if first == second:
                    continue

                if (
                    int(self.types_numpy[first]) != hydrogen
                    and int(self.types_numpy[second]) != hydrogen
                ):
                    continue

                pair = (first, second)

                # Keep whichever directed copy was encountered first.
                # Pair quantities are symmetric in the reference model.
                if pair not in found:
                    found[pair] = (centre, slot)

        ordered = sorted(found)

        edge_atoms = []
        edge_rows = []
        edge_slots = []

        for pair in ordered:
            row, slot = found[pair]

            edge_atoms.append(pair)
            edge_rows.append(row)
            edge_slots.append(slot)

        return (
            tuple(edge_atoms),
            tuple(edge_rows),
            tuple(edge_slots),
        )

    def _box_state_energy(
        self,
        edge_atoms,
        edge_rows,
        edge_slots,
        values,
    ):
        """Lowest mixed H-valence state for one box."""

        taper = values["taper"]
        pair_depth = values["pair_depth"]
        pair_width = values["pair_width"]
        shift = values["shift"]
        repulsive = values["repulsive"]

        edge_tapers = []
        edge_depths = []
        edge_repulsive = []
        edge_attractive = []

        for row, slot in zip(edge_rows, edge_slots):
            contact = taper[row, slot]
            depth = pair_depth[row, slot]

            attractive = (
                2.0
                * depth
                * torch.exp(
                    -pair_width[row, slot]
                    * shift[row, slot]
                )
            )

            edge_tapers.append(contact)
            edge_depths.append(depth)
            edge_repulsive.append(
                contact * repulsive[row, slot]
            )
            edge_attractive.append(
                contact * attractive
            )

        zero = values["taper"].sum() * 0.0

        if not edge_atoms:
            return zero

        states = _maximal_states(
            edge_atoms,
            self.types_numpy,
        )

        common_core = torch.stack(
            edge_repulsive
        ).sum()

        diagonals = []

        for state in states:
            if state:
                attraction = torch.stack([
                    edge_attractive[index]
                    for index in state
                ]).sum()
            else:
                attraction = zero

            diagonals.append(
                common_core - attraction
            )

        diagonal = torch.stack(diagonals)

        if len(states) == 1:
            return diagonal[0]

        transitions = {}
        weighted_degree = [
            zero for _ in states
        ]

        for first in range(len(states)):
            for second in range(first + 1, len(states)):
                transfer = _single_h_transfer(
                    states[first],
                    states[second],
                    edge_atoms,
                    self.types_numpy,
                )

                if transfer is None:
                    continue

                old_index, new_index, _ = transfer

                overlap = _contact_overlap(
                    edge_tapers[old_index],
                    edge_tapers[new_index],
                )

                transitions[(first, second)] = (
                    old_index,
                    new_index,
                    overlap,
                )

                # Weak alternatives should barely change the ordinary
                # two-state result.
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
            old_index,
            new_index,
            overlap,
        ) in transitions.items():

            depth_scale = torch.sqrt(
                torch.clamp(
                    edge_depths[old_index]
                    * edge_depths[new_index],
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

        # Build without mutating a tensor in-place, so autograd has a clean
        # graph through every matrix entry.
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

        # Only the lowest adiabatic state contributes to the reference
        # potential.
        eigenvalues = torch.linalg.eigvalsh(
            hamiltonian
        )

        return eigenvalues[0]

    def _hydrogen_state_correction(
        self,
        positions,
        base_per_atom,
    ):
        cached = getattr(
            self,
            "_reactive_intermediates",
            None,
        )

        if cached is None or cached[0] is not positions:
            raise RuntimeError(
                "h-state reference requires current reactive intermediates"
            )

        values = cached[1]

        neighbours = values["neighbours"]

        # State topology is a discrete question, just like neighbour-list
        # membership. Detach it. Energies and couplings remain Torch tensors,
        # so gradients still flow through the actual potential.
        neighbours_numpy = (
            neighbours.detach().cpu().numpy()
        )

        active_numpy = (
            (
                values["taper"].detach().cpu().numpy()
                > 1e-12
            )
            & self.neighbour_mask.detach().cpu().numpy()
        )

        taper = values["taper"]
        pair_depth = values["pair_depth"]
        pair_width = values["pair_width"]
        shift = values["shift"]
        repulsive = values["repulsive"]

        attractive = (
            2.0
            * pair_depth
            * torch.exp(
                -pair_width * shift
            )
        )

        pair_morse = (
            taper
            * (repulsive - attractive)
        )

        # The base potential currently creates H-transfer barriers by
        # penalising radial coordination above valence one. The reference
        # state model enforces valence one directly, so remove the H share of
        # that penalty while leaving heavy-atom over-coordination untouched.
        over_scale = self.over_coordination_scale(
            taper,
            values["unsoftened_depth"],
            values["mask"],
            cache_key=positions,
        )

        excess = torch.clamp(
            values["coordination"]
            - values["valence"],
            min=0.0,
        )

        base_over = (
            self.over_penalty
            * over_scale
            * excess * excess
        )

        hydrogen = int(R.ELEMENT_INDEX["H"])

        correction = torch.zeros_like(
            base_per_atom
        )

        all_indices = torch.arange(
            len(base_per_atom),
            device=base_per_atom.device,
        )

        for box in range(self.box_count):
            start = box * self.per_box
            stop = start + self.per_box

            (
                edge_atoms,
                edge_rows,
                edge_slots,
            ) = self._active_edges_for_box(
                box,
                values,
                neighbours_numpy,
                active_numpy,
            )

            if not edge_atoms:
                continue

            state_energy = self._box_state_energy(
                edge_atoms,
                edge_rows,
                edge_slots,
                values,
            )

            base_pair_terms = [
                pair_morse[row, slot]
                for row, slot
                in zip(edge_rows, edge_slots)
            ]

            base_h_pair = torch.stack(
                base_pair_terms
            ).sum()

            hydrogen_atoms = [
                atom
                for atom in range(start, stop)
                if int(self.types_numpy[atom]) == hydrogen
            ]

            if hydrogen_atoms:
                base_h_over = torch.stack([
                    base_over[atom]
                    for atom in hydrogen_atoms
                ]).sum()
            else:
                base_h_over = state_energy * 0.0

            delta = (
                state_energy
                - base_h_pair
                - base_h_over
            )

            # Per-atom placement is only bookkeeping. Put the box correction
            # on its first hydrogen so summing by box gives the correct total.
            anchor = (
                hydrogen_atoms[0]
                if hydrogen_atoms
                else start
            )

            weight = (
                all_indices == anchor
            ).to(base_per_atom.dtype)

            correction = (
                correction
                + weight * delta
            )

        return correction