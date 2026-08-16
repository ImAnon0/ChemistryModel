"""
Factorisable local H-state experiment.

This is an experimental successor to h_state_component_torch.py.

It addresses the component merge/split discontinuity found when two
independent H-transfer networks are joined by an infinitesimally weak
candidate H-containing edge.

Two architecture changes are tested together:

1. ALL valence-valid H matchings are retained as basis states.
   The old "maximal states only" basis changes discontinuously when a weak
   edge makes an otherwise valid state extendable.

2. Crowding normalisation is LOCAL TO THE TRANSFERRED HYDROGEN.
   The old weighted state degree sums unrelated transfers anywhere in the
   state graph.  For a transition involving hydrogen h, only alternative
   transitions involving the same h contribute to its crowding denominator.

The diagonal energy, contact overlap gate, H_STATE_MIXING, Morse quantities,
and one-H transfer definition are unchanged.

The component splitter from h_state_component_torch.py is retained, so
disconnected H-valence networks are solved independently.

This is a research model.  Do not promote it until the accompanying
validation passes.
"""

from __future__ import annotations

import itertools

import torch

from h_state_component_torch import (
    HStateComponentBatchedSimulation,
)
from h_state_torch import (
    _contact_overlap,
    _crowding_normalisation,
    _single_h_transfer,
    _state_is_valid,
)


H_STATE_FACTORISED_MODEL_NAME = (
    "reactive_v2_h_state_factorisable_components"
)
H_STATE_FACTORISED_MODEL_REVISION = 0

# Computational safety limit only, not a chemistry parameter.
MAX_FACTORISED_STATES = 512


def _all_valid_states(edge_atoms, types):
    """Enumerate all H-valence-valid edge subsets."""

    edge_count = len(edge_atoms)

    states = []

    for count in range(edge_count + 1):
        for chosen in itertools.combinations(
            range(edge_count),
            count,
        ):
            if _state_is_valid(
                chosen,
                edge_atoms,
                types,
            ):
                states.append(
                    tuple(chosen)
                )

                if len(states) > MAX_FACTORISED_STATES:
                    raise RuntimeError(
                        "factorisable H-state component exceeds "
                        f"{MAX_FACTORISED_STATES} valid states; "
                        "this is a research safety limit"
                    )

    if not states:
        return (tuple(),)

    return tuple(states)


class FactorisedHStateBatchedSimulation(
    HStateComponentBatchedSimulation
):
    """
    Local-component H-state with a factorisable zero-bridge limit.
    """

    physics_model_name = H_STATE_FACTORISED_MODEL_NAME
    physics_model_revision = H_STATE_FACTORISED_MODEL_REVISION

    def _box_state_energy(
        self,
        edge_atoms,
        edge_rows,
        edge_slots,
        values,
    ):
        taper = values["taper"]
        pair_depth = values["pair_depth"]
        pair_width = values["pair_width"]
        shift = values["shift"]
        repulsive = values["repulsive"]

        edge_tapers = []
        edge_depths = []
        edge_repulsive = []
        edge_attractive = []

        for row, slot in zip(
            edge_rows,
            edge_slots,
        ):
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
                contact
                * repulsive[row, slot]
            )

            edge_attractive.append(
                contact
                * attractive
            )

        zero = (
            values["taper"].sum()
            * 0.0
        )

        if not edge_atoms:
            return zero

        states = _all_valid_states(
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

        diagonal = torch.stack(
            diagonals
        )

        if len(states) == 1:
            return diagonal[0]

        # Build transfer list first.  The shared H returned by
        # _single_h_transfer is the locality key for crowding.
        transitions = {}

        weighted_degree_by_h = {}

        for first in range(len(states)):
            for second in range(
                first + 1,
                len(states),
            ):
                transfer = _single_h_transfer(
                    states[first],
                    states[second],
                    edge_atoms,
                    self.types_numpy,
                )

                if transfer is None:
                    continue

                (
                    old_index,
                    new_index,
                    hydrogen,
                ) = transfer

                overlap = _contact_overlap(
                    edge_tapers[old_index],
                    edge_tapers[new_index],
                )

                transitions[
                    (first, second)
                ] = (
                    old_index,
                    new_index,
                    hydrogen,
                    overlap,
                )

                weight = (
                    overlap * overlap
                )

                first_key = (
                    first,
                    hydrogen,
                )

                second_key = (
                    second,
                    hydrogen,
                )

                weighted_degree_by_h[
                    first_key
                ] = (
                    weighted_degree_by_h.get(
                        first_key,
                        zero,
                    )
                    + weight
                )

                weighted_degree_by_h[
                    second_key
                ] = (
                    weighted_degree_by_h.get(
                        second_key,
                        zero,
                    )
                    + weight
                )

        couplings = {}

        for (
            first,
            second,
        ), (
            old_index,
            new_index,
            hydrogen,
            overlap,
        ) in transitions.items():

            first_degree = (
                weighted_degree_by_h[
                    (first, hydrogen)
                ]
            )

            second_degree = (
                weighted_degree_by_h[
                    (second, hydrogen)
                ]
            )

            first_normalisation = (
                _crowding_normalisation(
                    first_degree
                )
            )

            second_normalisation = (
                _crowding_normalisation(
                    second_degree
                )
            )

            depth_scale = torch.sqrt(
                torch.clamp(
                    edge_depths[old_index]
                    * edge_depths[new_index],
                    min=1e-12,
                )
            )

            denominator = torch.sqrt(
                torch.clamp(
                    first_normalisation
                    * second_normalisation,
                    min=1e-12,
                )
            )

            coupling = (
                self.h_state_mixing
                * depth_scale
                * overlap
                / denominator
            )

            couplings[
                (first, second)
            ] = coupling

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

            rows.append(
                torch.stack(row)
            )

        hamiltonian = torch.stack(
            rows
        )

        eigenvalues = torch.linalg.eigvalsh(
            hamiltonian
        )

        return eigenvalues[0]


FactorisableHStateBatchedSimulation = (
    FactorisedHStateBatchedSimulation
)
