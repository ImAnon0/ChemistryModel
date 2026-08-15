"""
Component ablations for the state-aware SAPT H-state experiment.

Purpose
-------
The all-at-once state-aware experiment raised the frozen barriers by roughly:

    formaldehyde  +0.078 eV
    water         +0.024 eV
    methane       +0.104 eV

while leaving settled reaction energies unchanged.

Before changing architecture again, this module isolates WHICH part of the
original ChemistryModel valence machinery causes that transition-state shift.

Every mode starts from the CURRENT SAPT H-state diagonal and changes only one
downstream base mechanism:

    bond_order
        State occupancy changes coordination/spare valence -> bond order ->
        interpolated Morse length/depth/width. Environment-softening factor is
        held at its ordinary current value so this does not include the
        softening response.

    softening
        Bond order and raw Morse tables stay ordinary, but the existing
        environment-softening factor is recomputed from the state-specific
        covalent taper.

    overcoord
        Only HEAVY-ATOM over-coordination is made state-aware. Hydrogen
        over-coordination is already removed by the H-state architecture and
        is deliberately not reintroduced here.

    angles
        Only the existing angle/lone-pair/steric-domain energy is replaced by
        its state-aware value.

The independently calibrated SAPT wall and the existing off-diagonal coupling
law are unchanged in every mode.

This is an experimental ablation tool, not production physics.
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

from sapt_state_aware_valence_torch import (
    _state_covalent_taper,
    _bond_order_from_taper,
    _blend_pair_tables,
    _state_base_energy_per_atom,
)


ABLATION_MODES = (
    "bond_order",
    "softening",
    "overcoord",
    "angles",
)


def _pair_energy_total(
    *,
    covalent_taper,
    distances,
    pair_length,
    pair_depth,
    pair_width,
):
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

    # Directed neighbour representation: halve because every physical pair is
    # present from both ends.
    return (
        0.5
        * torch.sum(
            covalent_taper
            * (
                repulsive
                - attractive
            )
        )
    )


def _ordinary_softening_factor(
    values,
):
    return (
        values[
            "pair_depth"
        ]
        / torch.clamp(
            values[
                "unsoftened_depth"
            ],
            min=1.0e-12,
        )
    )


def _bond_order_correction(
    simulation,
    *,
    values,
    state_taper,
):
    """
    Change only bond-order interpolation / raw Morse tables.

    The ordinary environment-softening factor is frozen so the state-specific
    softening response is not counted here.
    """

    neighbours = values[
        "neighbours"
    ]

    (
        state_order,
        _,
        _,
    ) = _bond_order_from_taper(
        simulation,
        state_taper,
        neighbours,
    )

    (
        _,
        _,
        state_length,
        state_unsoftened_depth,
        state_width,
    ) = _blend_pair_tables(
        simulation,
        order=state_order,
        centre_types=values[
            "centre_types"
        ],
        other_types=values[
            "other_types"
        ],
    )

    frozen_factor = (
        _ordinary_softening_factor(
            values
        )
    )

    state_depth = (
        state_unsoftened_depth
        * frozen_factor
    )

    baseline = _pair_energy_total(
        covalent_taper=state_taper,
        distances=values[
            "distances"
        ],
        pair_length=values[
            "pair_length"
        ],
        pair_depth=values[
            "pair_depth"
        ],
        pair_width=values[
            "pair_width"
        ],
    )

    changed = _pair_energy_total(
        covalent_taper=state_taper,
        distances=values[
            "distances"
        ],
        pair_length=state_length,
        pair_depth=state_depth,
        pair_width=state_width,
    )

    return (
        changed
        - baseline
    )


def _softening_correction(
    simulation,
    *,
    values,
    state_taper,
):
    """
    Change only the environment-softening response.

    Order, Morse length, raw depth and width remain exactly the current
    ordinary values.
    """

    state_factor = (
        simulation.environment_softening_factor(
            state_taper,
            values[
                "order"
            ],
            values[
                "lower"
            ],
            values[
                "mask"
            ],
            values[
                "neighbours"
            ],
            cache_key=None,
        )
    )

    state_depth = (
        values[
            "unsoftened_depth"
        ]
        * state_factor
    )

    baseline = _pair_energy_total(
        covalent_taper=state_taper,
        distances=values[
            "distances"
        ],
        pair_length=values[
            "pair_length"
        ],
        pair_depth=values[
            "pair_depth"
        ],
        pair_width=values[
            "pair_width"
        ],
    )

    changed = _pair_energy_total(
        covalent_taper=state_taper,
        distances=values[
            "distances"
        ],
        pair_length=values[
            "pair_length"
        ],
        pair_depth=state_depth,
        pair_width=values[
            "pair_width"
        ],
    )

    return (
        changed
        - baseline
    )


def _heavy_overcoord_correction(
    simulation,
    *,
    values,
    state_taper,
    start,
    stop,
):
    """
    Change only heavy-atom over-coordination.

    H overcoordination is intentionally excluded because the parent H-state
    correction already removes the hydrogen share of this original barrier
    mechanism.
    """

    valence = values[
        "valence"
    ]

    ordinary_excess = torch.clamp(
        values[
            "coordination"
        ]
        - valence,
        min=0.0,
    )

    ordinary_scale = (
        simulation.over_coordination_scale(
            values[
                "taper"
            ],
            values[
                "unsoftened_depth"
            ],
            values[
                "mask"
            ],
            cache_key=None,
        )
    )

    ordinary = (
        simulation.over_penalty
        * ordinary_scale
        * ordinary_excess
        * ordinary_excess
    )

    state_coordination = torch.sum(
        state_taper,
        dim=1,
    )

    state_excess = torch.clamp(
        state_coordination
        - valence,
        min=0.0,
    )

    state_scale = (
        simulation.over_coordination_scale(
            state_taper,
            values[
                "unsoftened_depth"
            ],
            values[
                "mask"
            ],
            cache_key=None,
        )
    )

    changed = (
        simulation.over_penalty
        * state_scale
        * state_excess
        * state_excess
    )

    hydrogen = int(
        R.ELEMENT_INDEX[
            "H"
        ]
    )

    heavy_mask = torch.tensor(
        [
            0.0
            if int(
                simulation.types_numpy[
                    atom
                ]
            ) == hydrogen
            else 1.0
            for atom in range(
                start,
                stop,
            )
        ],
        dtype=values[
            "taper"
        ].dtype,
        device=values[
            "taper"
        ].device,
    )

    return torch.sum(
        (
            changed[
                start:stop
            ]
            - ordinary[
                start:stop
            ]
        )
        * heavy_mask
    )


def _angle_correction(
    simulation,
    *,
    positions,
    values,
    state_taper,
    start,
    stop,
):
    """
    Replace only the existing angle/lone-pair/steric-domain term with its
    state-aware value.
    """

    ordinary_parts = (
        _state_base_energy_per_atom(
            simulation,
            positions=positions,
            values=values,
            state_taper=values[
                "taper"
            ],
        )
    )

    state_parts = (
        _state_base_energy_per_atom(
            simulation,
            positions=positions,
            values=values,
            state_taper=state_taper,
        )
    )

    return torch.sum(
        state_parts[
            "angle"
        ][
            start:stop
        ]
        - ordinary_parts[
            "angle"
        ][
            start:stop
        ]
    )


class StateAwareAblationSaptHStateBatchedSimulation(
    SaptHStateBatchedSimulation
):
    """Current SAPT H-state model plus exactly one state-aware base ablation."""

    physics_model_name = (
        "SAPT H-state state-aware component ablation"
    )

    physics_model_revision = (
        "sapt-state-aware-ablation-v1"
    )

    def __init__(
        self,
        *args,
        ablation_mode,
        h_state_mixing=SAPT_H_STATE_MIXING,
        **kwargs,
    ):
        mode = str(
            ablation_mode
        )

        if mode not in ABLATION_MODES:
            raise ValueError(
                "ablation_mode must be one of: "
                + ", ".join(
                    ABLATION_MODES
                )
            )

        self.ablation_mode = mode

        super().__init__(
            *args,
            h_state_mixing=h_state_mixing,
            **kwargs,
        )

    def _component_correction(
        self,
        *,
        state_taper,
        positions,
        values,
        start,
        stop,
    ):
        if self.ablation_mode == "bond_order":
            return _bond_order_correction(
                self,
                values=values,
                state_taper=state_taper,
            )

        if self.ablation_mode == "softening":
            return _softening_correction(
                self,
                values=values,
                state_taper=state_taper,
            )

        if self.ablation_mode == "overcoord":
            return _heavy_overcoord_correction(
                self,
                values=values,
                state_taper=state_taper,
                start=start,
                stop=stop,
            )

        if self.ablation_mode == "angles":
            return _angle_correction(
                self,
                positions=positions,
                values=values,
                state_taper=state_taper,
                start=start,
                stop=stop,
            )

        raise RuntimeError(
            f"unhandled ablation mode {self.ablation_mode}"
        )

    def _box_state_energy(
        self,
        edge_atoms,
        edge_rows,
        edge_slots,
        values,
    ):
        taper = values[
            "taper"
        ]

        pair_depth = values[
            "pair_depth"
        ]

        pair_width = values[
            "pair_width"
        ]

        shift = values[
            "shift"
        ]

        repulsive = values[
            "repulsive"
        ]

        zero = (
            taper.sum()
            * 0.0
        )

        if not edge_atoms:
            return zero

        edge_tapers = []
        edge_depths = []
        edge_repulsive = []
        edge_attractive = []

        for row, slot in zip(
            edge_rows,
            edge_slots,
        ):
            contact = taper[
                row,
                slot,
            ]

            depth = pair_depth[
                row,
                slot,
            ]

            attractive = (
                2.0
                * depth
                * torch.exp(
                    -pair_width[
                        row,
                        slot,
                    ]
                    * shift[
                        row,
                        slot,
                    ]
                )
            )

            edge_tapers.append(
                contact
            )

            edge_depths.append(
                depth
            )

            edge_repulsive.append(
                contact
                * repulsive[
                    row,
                    slot,
                ]
            )

            edge_attractive.append(
                contact
                * attractive
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
                "state-aware ablation requires reactive intermediates"
            )

        positions = cached[
            0
        ]

        first_atom = min(
            atom
            for pair in edge_atoms
            for atom in pair
        )

        box = (
            first_atom
            // self.per_box
        )

        start = (
            box
            * self.per_box
        )

        stop = (
            start
            + self.per_box
        )

        local_positions = positions[
            start:stop
        ]

        symbol_for = (
            _symbol_lookup()
        )

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

        neighbours = values[
            "neighbours"
        ]

        neighbour_mask = (
            self.neighbour_mask
        )

        diagonals = []

        for state in states:
            weights = (
                _descriptor_weights_for_state(
                    box=box,
                    per_box=self.per_box,
                    types_numpy=self.types_numpy,
                    neighbours=neighbours,
                    neighbour_mask=neighbour_mask,
                    taper=taper,
                    edge_atoms=edge_atoms,
                    edge_tapers=edge_tapers,
                    state=state,
                )
            )

            fragment = (
                nb.ContinuousTorchFragment(
                    symbols=local_symbols,
                    positions=local_positions,
                    bond_weights=weights,
                )
            )

            if state:
                covalent = torch.stack(
                    [
                        edge_repulsive[
                            index
                        ]
                        - edge_attractive[
                            index
                        ]
                        for index in state
                    ]
                ).sum()
            else:
                covalent = zero

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

            state_taper = (
                _state_covalent_taper(
                    taper=taper,
                    neighbours=neighbours,
                    edge_atoms=edge_atoms,
                    state=state,
                )
            )

            correction = (
                self._component_correction(
                    state_taper=state_taper,
                    positions=positions,
                    values=values,
                    start=start,
                    stop=stop,
                )
            )

            diagonals.append(
                covalent
                + wall
                + correction
            )

        diagonal = torch.stack(
            diagonals
        )

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
                transition = (
                    hs._single_h_transfer(
                        states[
                            first
                        ],
                        states[
                            second
                        ],
                        edge_atoms,
                        self.types_numpy,
                    )
                )

                if transition is None:
                    continue

                (
                    old_index,
                    new_index,
                    _,
                ) = transition

                overlap = (
                    hs._contact_overlap(
                        edge_tapers[
                            old_index
                        ],
                        edge_tapers[
                            new_index
                        ],
                    )
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

        return torch.linalg.eigvalsh(
            hamiltonian
        )[
            0
        ]
