"""
Experimental SAPT H-state adapter with state-aware base valence.

This tests a plumbing inconsistency in the current reference architecture:

    H-state diagonal:
        says an H-containing edge is occupied or unoccupied

    ordinary ChemistryModel base:
        still lets every geometrically close H contact contribute to
        coordination, bond order, environment softening, heavy-atom
        over-coordination and angles.

For each diabatic state this adapter instead rebuilds the EXISTING
ChemistryModel valence machinery using a state-specific reactive taper:

    heavy-heavy contacts:
        unchanged

    occupied H-state edges:
        keep their ordinary smooth reactive taper

    unoccupied H-state edges:
        covalent taper is zero for base valence/bond/angle bookkeeping;
        the independently calibrated SAPT wall still acts on them

No new fitted energy term is introduced.

The entire ordinary base energy of a small reference box is replaced by the
lowest eigenvalue of state-specific full-box diagonals. This is cleaner than
trying to patch only selected heavy-atom terms and avoids double counting.

The off-diagonal coupling law is deliberately unchanged.

REFERENCE / EXPERIMENTAL ONLY.
"""

from __future__ import annotations

import torch

import reactive as R
import h_state_torch as hs
import nonbonded_continuous_torch as nb

from sapt_h_state_torch import (
    SAPT_H_STATE_MIXING,
    SaptHStateBatchedSimulation,
    _descriptor_weights_for_state,
    _sapt_pair_energy,
    _symbol_lookup,
)


STATE_AWARE_MODEL_REVISION = (
    "sapt-wall-v3-state-aware-base-valence-experimental"
)


def _state_covalent_taper(
    *,
    taper,
    neighbours,
    edge_atoms,
    state,
):
    """
    Return the directed reactive taper used by the ordinary base terms in one
    H-valence state.

    The neighbour topology itself is unchanged. Only unoccupied H-state
    candidate edges are removed from *covalent* bookkeeping.
    """

    occupied = set(
        state
    )

    row_atoms = torch.arange(
        taper.shape[0],
        dtype=neighbours.dtype,
        device=neighbours.device,
    )[:, None].expand_as(
        neighbours
    )

    keep = torch.ones_like(
        taper
    )

    for edge_index, (
        first,
        second,
    ) in enumerate(
        edge_atoms
    ):
        if edge_index in occupied:
            continue

        pair_selector = (
            (
                (row_atoms == first)
                & (neighbours == second)
            )
            | (
                (row_atoms == second)
                & (neighbours == first)
            )
        ).to(
            taper.dtype
        )

        keep = (
            keep
            * (
                1.0
                - pair_selector
            )
        )

    return (
        taper
        * keep
    )


def _bond_order_from_taper(
    simulation,
    state_taper,
    neighbours,
):
    """Mirror ReactiveSimulation.energy_per_atom bond-order algebra."""

    coordination = torch.sum(
        state_taper,
        dim=1,
    )

    valence = simulation.valence[
        simulation.types
    ]

    spare = torch.clamp(
        valence
        - coordination,
        min=0.0,
    )

    spare_other = simulation._gather_neighbours(
        spare,
        neighbours,
        "state_spare",
    )

    weighted = (
        state_taper
        * spare_other
    )

    totals = torch.sum(
        weighted,
        dim=1,
    )

    totals_other = simulation._gather_neighbours(
        totals,
        neighbours,
        "state_totals",
    )

    onset = 1.0e-4

    share_fraction = torch.clamp(
        totals / onset,
        0.0,
        1.0,
    )

    share_gate = (
        share_fraction
        * share_fraction
        * (
            3.0
            - 2.0
            * share_fraction
        )
    )

    share_out = (
        spare[:, None]
        * weighted
        / torch.clamp(
            totals[:, None],
            min=1.0e-12,
        )
        * share_gate[:, None]
    )

    share_fraction_other = torch.clamp(
        totals_other / onset,
        0.0,
        1.0,
    )

    share_gate_other = (
        share_fraction_other
        * share_fraction_other
        * (
            3.0
            - 2.0
            * share_fraction_other
        )
    )

    share_back = (
        spare_other
        * (
            state_taper
            * spare[:, None]
        )
        / torch.clamp(
            totals_other,
            min=1.0e-12,
        )
        * share_gate_other
    )

    # This matches the current live Torch engine, including the already
    # accepted final taper weighting on extra bond order.
    extra = (
        torch.minimum(
            share_out,
            share_back,
        )
        * state_taper
    )

    order = torch.clamp(
        1.0 + extra,
        1.0,
        3.0,
    )

    return (
        order,
        coordination,
        valence,
    )


def _blend_pair_tables(
    simulation,
    *,
    order,
    centre_types,
    other_types,
):
    lower = torch.clamp(
        order - 1.0,
        0.0,
        1.0,
    )

    upper = torch.clamp(
        order - 2.0,
        0.0,
        1.0,
    )

    def blend(
        single_table,
        double_table,
        triple_table,
    ):
        single = single_table[
            centre_types,
            other_types,
        ]

        double = double_table[
            centre_types,
            other_types,
        ]

        triple = triple_table[
            centre_types,
            other_types,
        ]

        first = (
            single
            + (
                double
                - single
            )
            * lower
        )

        return (
            first
            + (
                triple
                - first
            )
            * upper
        )

    return (
        lower,
        upper,
        blend(
            simulation.bond_length,
            simulation.double_length,
            simulation.triple_length,
        ),
        blend(
            simulation.bond_depth,
            simulation.double_depth,
            simulation.triple_depth,
        ),
        blend(
            simulation.bond_width,
            simulation.double_width,
            simulation.triple_width,
        ),
    )


def _state_base_energy_per_atom(
    simulation,
    *,
    positions,
    values,
    state_taper,
):
    """
    Re-evaluate the existing ChemistryModel bond, overcoordination and angle
    terms with one state's covalent H occupancy.

    No new potential terms are introduced here.
    """

    neighbours = values[
        "neighbours"
    ]

    mask = values[
        "mask"
    ]

    distances = values[
        "distances"
    ]

    centre_types = values[
        "centre_types"
    ]

    other_types = values[
        "other_types"
    ]

    (
        order,
        coordination,
        valence,
    ) = _bond_order_from_taper(
        simulation,
        state_taper,
        neighbours,
    )

    (
        lower,
        upper,
        pair_length,
        unsoftened_depth,
        pair_width,
    ) = _blend_pair_tables(
        simulation,
        order=order,
        centre_types=centre_types,
        other_types=other_types,
    )

    pair_depth = (
        unsoftened_depth
        * simulation.environment_softening_factor(
            state_taper,
            order,
            lower,
            mask,
            neighbours,
            # State-specific taper/order must never hit the base evaluation's
            # cache just because the positions tensor is identical.
            cache_key=None,
        )
    )

    shift = (
        distances
        - pair_length
    )

    repulsive = (
        pair_depth
        * torch.exp(
            -2.0
            * pair_width
            * shift
        )
    )

    attractive = (
        2.0
        * pair_depth
        * torch.exp(
            -pair_width
            * shift
        )
    )

    pair_energy = (
        state_taper
        * (
            repulsive
            - attractive
        )
    )

    bond_per_atom = (
        0.5
        * torch.sum(
            pair_energy,
            dim=1,
        )
    )

    excess = torch.clamp(
        coordination
        - valence,
        min=0.0,
    )

    over_per_atom = (
        simulation.over_penalty
        * simulation.over_coordination_scale(
            state_taper,
            unsoftened_depth,
            mask,
            cache_key=None,
        )
        * excess
        * excess
    )

    bonded_order = torch.sum(
        state_taper
        * order,
        dim=1,
    )

    outer = simulation.outer_electrons[
        simulation.types
    ]

    lone_pairs = torch.clamp(
        (
            outer
            - bonded_order
        )
        / 2.0,
        min=0.0,
    )

    steric = torch.clamp(
        coordination
        + lone_pairs,
        2.0,
        4.0,
    )

    low_angle = torch.where(
        steric < 3.0,
        (
            180.0
            + (
                120.0
                - 180.0
            )
            * (
                steric
                - 2.0
            )
        ),
        (
            120.0
            + (
                109.47
                - 120.0
            )
            * (
                steric
                - 3.0
            )
        ),
    )

    rest = torch.deg2rad(
        low_angle
        - simulation.lone_pair_squeeze
        * lone_pairs
    )

    stiffness = simulation.angle_stiffness[
        simulation.types
    ]

    offsets = (
        simulation._gather_neighbours(
            positions,
            neighbours,
            "state_positions",
        )
        - positions[:, None, :]
    )

    offsets = (
        offsets
        - simulation.box_size
        * torch.round(
            offsets
            / simulation.box_size
        )
    )

    left = offsets[
        :,
        :,
        None,
        :,
    ]

    right = offsets[
        :,
        None,
        :,
        :,
    ]

    dot = torch.sum(
        left * right,
        dim=3,
    )

    cosine = torch.clamp(
        dot
        / torch.clamp(
            distances[
                :,
                :,
                None,
            ]
            * distances[
                :,
                None,
                :,
            ],
            min=1.0e-9,
        ),
        -1.0 + 1.0e-7,
        1.0 - 1.0e-7,
    )

    angle = torch.arccos(
        cosine
    )

    first_taper = state_taper[
        :,
        :,
        None,
    ]

    second_taper = state_taper[
        :,
        None,
        :,
    ]

    angle_pair_taper = (
        first_taper
        * second_taper
    )

    taper_difference = (
        first_taper
        - second_taper
    )

    weaker_taper = (
        0.5
        * (
            first_taper
            + second_taper
            - torch.sqrt(
                taper_difference
                * taper_difference
                + 1.0e-8
            )
            + 1.0e-4
        )
    )

    lone_pair_directionality = torch.clamp(
        0.5
        * lone_pairs,
        0.0,
        1.0,
    )[
        :,
        None,
        None,
    ]

    angle_engagement = (
        weaker_taper
        + (
            1.0
            - weaker_taper
        )
        * lone_pair_directionality
    )

    weight = (
        angle_pair_taper
        * angle_engagement
    )

    upper_triangle = getattr(
        simulation,
        "_angle_upper_triangle",
        None,
    )

    if upper_triangle is None:
        upper_triangle = torch.triu(
            torch.ones(
                weight.shape[
                    1
                ],
                weight.shape[
                    2
                ],
                dtype=simulation.dtype,
                device=simulation.device,
            ),
            diagonal=1,
        )

        simulation._angle_upper_triangle = (
            upper_triangle
        )

    angle_energy = (
        0.5
        * stiffness[
            :,
            None,
            None,
        ]
        * weight
        * upper_triangle
        * (
            angle
            - rest[
                :,
                None,
                None,
            ]
        )
        ** 2
    )

    angle_per_atom = torch.sum(
        angle_energy,
        dim=(
            1,
            2,
        ),
    )

    return {
        "total": (
            bond_per_atom
            + over_per_atom
            + angle_per_atom
        ),
        "bond": bond_per_atom,
        "over": over_per_atom,
        "angle": angle_per_atom,
        "coordination": coordination,
        "order": order,
        "lone_pairs": lone_pairs,
        "steric": steric,
        "pair_depth": pair_depth,
        "unsoftened_depth": unsoftened_depth,
    }


class StateAwareValenceSaptHStateBatchedSimulation(
    SaptHStateBatchedSimulation
):
    """
    SAPT H-state reference with state occupancy propagated back into the
    existing ChemistryModel base valence machinery.
    """

    physics_model_name = (
        "SAPT-wall H-state + state-aware ChemistryModel valence"
    )

    physics_model_revision = (
        STATE_AWARE_MODEL_REVISION
    )

    def _full_box_state_energy(
        self,
        *,
        box,
        edge_atoms,
        edge_rows,
        edge_slots,
        values,
    ):
        taper = values[
            "taper"
        ]

        neighbours = values[
            "neighbours"
        ]

        zero = (
            taper.sum()
            * 0.0
        )

        states = hs._maximal_states(
            edge_atoms,
            self.types_numpy,
        )

        cached = getattr(
            self,
            "_reactive_intermediates",
            None,
        )

        if cached is None:
            raise RuntimeError(
                "state-aware SAPT H-state requires reactive intermediates"
            )

        positions = cached[
            0
        ]

        start = (
            box
            * self.per_box
        )

        stop = (
            start
            + self.per_box
        )

        edge_tapers = [
            taper[
                row,
                slot,
            ]
            for row, slot
            in zip(
                edge_rows,
                edge_slots,
            )
        ]

        # Keep the existing coupling depth scale unchanged in this experiment.
        edge_depths = [
            values[
                "pair_depth"
            ][
                row,
                slot,
            ]
            for row, slot
            in zip(
                edge_rows,
                edge_slots,
            )
        ]

        symbol_for = _symbol_lookup()

        local_symbols = [
            symbol_for[
                int(
                    self.types_numpy[
                        atom
                    ]
                )
            ]
            for atom in range(
                start,
                stop,
            )
        ]

        local_positions = positions[
            start:stop
        ]

        diagonals = []

        for state in states:
            state_taper = _state_covalent_taper(
                taper=taper,
                neighbours=neighbours,
                edge_atoms=edge_atoms,
                state=state,
            )

            base_parts = _state_base_energy_per_atom(
                self,
                positions=positions,
                values=values,
                state_taper=state_taper,
            )

            state_base_total = torch.sum(
                base_parts[
                    "total"
                ][
                    start:stop
                ]
            )

            descriptor_weights = (
                _descriptor_weights_for_state(
                    box=box,
                    per_box=self.per_box,
                    types_numpy=self.types_numpy,
                    neighbours=neighbours,
                    neighbour_mask=self.neighbour_mask,
                    taper=taper,
                    edge_atoms=edge_atoms,
                    edge_tapers=edge_tapers,
                    state=state,
                )
            )

            fragment = nb.ContinuousTorchFragment(
                symbols=local_symbols,
                positions=local_positions,
                bond_weights=descriptor_weights,
            )

            occupied = set(
                state
            )

            wall = zero

            for edge_index, (
                global_a,
                global_b,
            ) in enumerate(
                edge_atoms
            ):
                if edge_index in occupied:
                    continue

                wall = (
                    wall
                    + edge_tapers[
                        edge_index
                    ]
                    * _sapt_pair_energy(
                        fragment,
                        global_a
                        - start,
                        global_b
                        - start,
                    )
                )

            diagonals.append(
                state_base_total
                + wall
            )

        diagonal = torch.stack(
            diagonals
        )

        if len(
            states
        ) == 1:
            return diagonal[
                0
            ]

        weighted_degree = [
            zero
            for _ in states
        ]

        transitions = {}

        for first in range(
            len(
                states
            )
        ):
            for second in range(
                first + 1,
                len(
                    states
                ),
            ):
                transition = hs._single_h_transfer(
                    states[
                        first
                    ],
                    states[
                        second
                    ],
                    edge_atoms,
                    self.types_numpy,
                )

                if transition is None:
                    continue

                (
                    old_index,
                    new_index,
                    _,
                ) = transition

                overlap = hs._contact_overlap(
                    edge_tapers[
                        old_index
                    ],
                    edge_tapers[
                        new_index
                    ],
                )

                transitions[
                    (
                        first,
                        second,
                    )
                ] = (
                    old_index,
                    new_index,
                    overlap,
                )

                weighted_degree[
                    first
                ] = (
                    weighted_degree[
                        first
                    ]
                    + overlap
                    * overlap
                )

                weighted_degree[
                    second
                ] = (
                    weighted_degree[
                        second
                    ]
                    + overlap
                    * overlap
                )

        normalisation = torch.stack(
            [
                hs._crowding_normalisation(
                    value
                )
                for value
                in weighted_degree
            ]
        )

        couplings = {}

        for (
            first,
            second,
        ), (
            old_index,
            new_index,
            overlap,
        ) in transitions.items():
            depth_scale = torch.sqrt(
                torch.clamp(
                    edge_depths[
                        old_index
                    ]
                    * edge_depths[
                        new_index
                    ],
                    min=1.0e-12,
                )
            )

            denominator = torch.sqrt(
                torch.clamp(
                    normalisation[
                        first
                    ]
                    * normalisation[
                        second
                    ],
                    min=1.0e-12,
                )
            )

            coupling = (
                self.h_state_mixing
                * depth_scale
                * overlap
                / denominator
            )

            couplings[
                (
                    first,
                    second,
                )
            ] = coupling

        rows = []

        for first in range(
            len(
                states
            )
        ):
            row = []

            for second in range(
                len(
                    states
                )
            ):
                if first == second:
                    value = diagonal[
                        first
                    ]
                else:
                    key = (
                        min(
                            first,
                            second,
                        ),
                        max(
                            first,
                            second,
                        ),
                    )

                    value = (
                        -couplings[
                            key
                        ]
                        if key
                        in couplings
                        else zero
                    )

                row.append(
                    value
                )

            rows.append(
                torch.stack(
                    row
                )
            )

        hamiltonian = torch.stack(
            rows
        )

        eigenvalues = torch.linalg.eigvalsh(
            hamiltonian
        )

        return eigenvalues[
            0
        ]

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

        if (
            cached is None
            or cached[
                0
            ]
            is not positions
        ):
            raise RuntimeError(
                "state-aware H-state requires current reactive intermediates"
            )

        values = cached[
            1
        ]

        neighbours_numpy = (
            values[
                "neighbours"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        active_numpy = (
            (
                values[
                    "taper"
                ]
                .detach()
                .cpu()
                .numpy()
                > 1.0e-12
            )
            & self.neighbour_mask
            .detach()
            .cpu()
            .numpy()
        )

        correction = torch.zeros_like(
            base_per_atom
        )

        all_indices = torch.arange(
            len(
                base_per_atom
            ),
            dtype=torch.long,
            device=base_per_atom.device,
        )

        for box in range(
            self.box_count
        ):
            start = (
                box
                * self.per_box
            )

            stop = (
                start
                + self.per_box
            )

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

            mixed_full_box = (
                self._full_box_state_energy(
                    box=box,
                    edge_atoms=edge_atoms,
                    edge_rows=edge_rows,
                    edge_slots=edge_slots,
                    values=values,
                )
            )

            ordinary_base_box = torch.sum(
                base_per_atom[
                    start:stop
                ]
            )

            delta = (
                mixed_full_box
                - ordinary_base_box
            )

            # Pure bookkeeping: place the box-level correction on the first H
            # if possible so summing per box recovers exactly the replacement.
            hydrogen = int(
                R.ELEMENT_INDEX[
                    "H"
                ]
            )

            hydrogen_atoms = [
                atom
                for atom in range(
                    start,
                    stop,
                )
                if int(
                    self.types_numpy[
                        atom
                    ]
                )
                == hydrogen
            ]

            anchor = (
                hydrogen_atoms[
                    0
                ]
                if hydrogen_atoms
                else start
            )

            correction = (
                correction
                + (
                    all_indices
                    == anchor
                ).to(
                    base_per_atom.dtype
                )
                * delta
            )

        return correction
