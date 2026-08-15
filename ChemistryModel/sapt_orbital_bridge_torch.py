"""
Experimental SAPT/H-state short-range orbital bridge.

This module deliberately leaves the current SAPT wall, production reactive
potential, and frozen SAPT parameters untouched.

Motivation
----------
In a diabatic H-transfer state, the alternative edge is currently treated as

    occupied edge      -> full tapered Morse covalent interaction
    unoccupied edge    -> pure tapered SAPT exchange wall

The formaldehyde path diagnostics show that this makes the product-like state
several eV too high while the breaking C-H bond is still at covalent distance.
"Unoccupied in this diabatic state" is therefore being asked to mean "pure
closed-shell nonbonded contact" too early.

This experimental adapter adds one state-diagonal term only while an H is
actually exchanging partners:

    E_bridge = -lambda * G * A_Morse

where A_Morse is the existing tapered Morse attractive component of the
currently unoccupied alternative edge and G is built from the SAME balanced
contact-overlap gate already used for state coupling.

Properties:
- no pair-specific parameters
- no formaldehyde-specific branch
- no bridge for an isolated bond with no competing transfer state
- vanishes smoothly when either competing contact vanishes
- preserves the frozen SAPT wall itself
- lambda=0 delegates exactly to the current SAPT adapter

This is a research probe, not production physics.
"""

from __future__ import annotations

import torch

import h_state_torch as hs
import nonbonded_continuous_torch as nb

from sapt_h_state_torch import (
    SAPT_H_STATE_MIXING,
    SaptHStateBatchedSimulation,
    _descriptor_weights_for_state,
    _sapt_pair_energy,
    _symbol_lookup,
)


ORBITAL_BRIDGE_MODEL_REVISION = (
    "sapt-wall-v3-orbital-bridge-experimental"
)


def _bridge_gate_maps(
    states,
    edge_atoms,
    edge_tapers,
    types_numpy,
    *,
    like,
):
    """
    Return per-state gates for transitionable *unoccupied* alternative edges.

    For a direct state transition

        state A: old edge occupied, new edge unoccupied
        state B: old edge unoccupied, new edge occupied

    the balanced overlap of old/new contacts is assigned to:
        state A -> new edge
        state B -> old edge

    If the same unoccupied edge is reachable through more than one directly
    coupled state, squared overlaps are accumulated. The same smooth crowding
    normalisation used by the H-state coupling bounds the resulting gate
    without introducing a hard max or an N-state free stabilisation.

    With one ordinary two-state alternative this reduces exactly to the
    existing contact-overlap gate.
    """

    zero = like.sum() * 0.0

    edge_degree = [
        {}
        for _ in states
    ]

    transitions = {}

    for first in range(
        len(states)
    ):
        for second in range(
            first + 1,
            len(states),
        ):
            transition = (
                hs._single_h_transfer(
                    states[first],
                    states[second],
                    edge_atoms,
                    types_numpy,
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
                (first, second)
            ] = (
                old_index,
                new_index,
                overlap,
            )

            # In the first state the "new" edge is the unoccupied alternative.
            current = edge_degree[
                first
            ].get(
                new_index,
                zero,
            )

            edge_degree[
                first
            ][
                new_index
            ] = (
                current
                + overlap * overlap
            )

            # In the second state the "old" edge is the unoccupied alternative.
            current = edge_degree[
                second
            ].get(
                old_index,
                zero,
            )

            edge_degree[
                second
            ][
                old_index
            ] = (
                current
                + overlap * overlap
            )

    gate_maps = []

    for per_state in edge_degree:
        gates = {}

        for edge_index, degree in per_state.items():
            normalisation = (
                hs._crowding_normalisation(
                    degree
                )
            )

            gates[
                edge_index
            ] = torch.sqrt(
                torch.clamp(
                    degree
                    / torch.clamp(
                        normalisation,
                        min=1.0e-12,
                    ),
                    min=0.0,
                )
            )

        gate_maps.append(
            gates
        )

    return (
        gate_maps,
        transitions,
    )


class OrbitalBridgeSaptHStateBatchedSimulation(
    SaptHStateBatchedSimulation
):
    """
    Current SAPT-wall H-state adapter plus an experimental orbital bridge.

    ``orbital_bridge_strength`` is dimensionless. A value of zero is exactly
    the current SAPT adapter. Positive values recover a fraction of the
    existing Morse attraction for an unoccupied edge *only* when that edge is
    the competing partner in a recognised one-H state transfer.
    """

    physics_model_name = (
        "SAPT-wall H-state + experimental orbital bridge"
    )

    physics_model_revision = (
        ORBITAL_BRIDGE_MODEL_REVISION
    )

    def __init__(
        self,
        *args,
        orbital_bridge_strength=0.0,
        h_state_mixing=SAPT_H_STATE_MIXING,
        **kwargs,
    ):
        strength = float(
            orbital_bridge_strength
        )

        if strength < 0.0:
            raise ValueError(
                "orbital_bridge_strength must be non-negative"
            )

        self.orbital_bridge_strength = strength

        super().__init__(
            *args,
            h_state_mixing=h_state_mixing,
            **kwargs,
        )

    def _box_state_energy(
        self,
        edge_atoms,
        edge_rows,
        edge_slots,
        values,
    ):
        # This guarantees lambda=0 is not merely algebraically intended to
        # match the current adapter; it executes the exact current code path.
        if self.orbital_bridge_strength == 0.0:
            return super()._box_state_energy(
                edge_atoms,
                edge_rows,
                edge_slots,
                values,
            )

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

            # Already tapered. Multiplying this by the transfer-overlap gate
            # makes the bridge vanish if either the alternative edge itself or
            # the competing occupied contact leaves the active region.
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
                "orbital-bridge SAPT H-state requires reactive intermediates"
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

        (
            bridge_gates,
            transitions,
        ) = _bridge_gate_maps(
            states,
            edge_atoms,
            edge_tapers,
            self.types_numpy,
            like=taper,
        )

        diagonals = []

        for state_index, state in enumerate(
            states
        ):
            weights = (
                _descriptor_weights_for_state(
                    box=box,
                    per_box=self.per_box,
                    types_numpy=(
                        self.types_numpy
                    ),
                    neighbours=neighbours,
                    neighbour_mask=(
                        neighbour_mask
                    ),
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
                        global_a - start,
                        global_b - start,
                    )
                )

            bridge_terms = []

            for edge_index, gate in (
                bridge_gates[
                    state_index
                ].items()
            ):
                # This should be guaranteed by _bridge_gate_maps. Keep the
                # condition explicit so an enumeration change cannot silently
                # add extra attraction to an already occupied bond.
                if edge_index in occupied:
                    continue

                bridge_terms.append(
                    gate
                    * edge_attractive[
                        edge_index
                    ]
                )

            if bridge_terms:
                bridge = (
                    -self.orbital_bridge_strength
                    * torch.stack(
                        bridge_terms
                    ).sum()
                )
            else:
                bridge = zero

            diagonals.append(
                covalent
                + wall
                + bridge
            )

        diagonal = torch.stack(
            diagonals
        )

        # Coupling is deliberately unchanged. The experiment asks whether the
        # missing *diagonal* short-range electronic interaction is the issue.
        weighted_degree = [
            zero
            for _ in states
        ]

        for (
            first,
            second,
        ), (
            _,
            _,
            overlap,
        ) in transitions.items():
            weighted_degree[
                first
            ] = (
                weighted_degree[
                    first
                ]
                + overlap * overlap
            )

            weighted_degree[
                second
            ] = (
                weighted_degree[
                    second
                ]
                + overlap * overlap
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
                (first, second)
            ] = coupling

        rows = []

        for first in range(
            len(states)
        ):
            row = []

            for second in range(
                len(states)
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
                        if key in couplings
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
