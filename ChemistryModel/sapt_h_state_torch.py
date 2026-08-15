import math

import torch

import reactive as R
import h_state_torch as hs
import nonbonded_continuous_torch as nb


SAPT_H_STATE_MIXING = 0.534590721


def _symbol_lookup():
    return {
        int(index): symbol
        for symbol, index
        in R.ELEMENT_INDEX.items()
    }


def _sapt_pair_energy(
    fragment,
    atom_a,
    atom_b,
):
    delta = (
        fragment.positions[atom_b]
        - fragment.positions[atom_a]
    )

    distance = torch.linalg.vector_norm(
        delta
    )

    rhat = (
        delta
        / torch.clamp(
            distance,
            min=nb.EPS,
        )
    )

    symbol_a = fragment.symbols[
        atom_a
    ]

    symbol_b = fragment.symbols[
        atom_b
    ]

    A_a = nb.effective_A(
        fragment,
        atom_a,
    )

    A_b = nb.effective_A(
        fragment,
        atom_b,
    )

    B_a = nb.effective_B(
        fragment,
        atom_a,
        rhat,
    )

    B_b = nb.effective_B(
        fragment,
        atom_b,
        -rhat,
    )

    beta = torch.sqrt(
        B_a * B_b
    )

    x = beta * distance

    radial = (
        A_a
        * A_b
        * (
            1.0
            + x
            + x*x/3.0
        )
        * torch.exp(-x)
    )

    q_a = nb.amplitude_q2(
        fragment,
        atom_a,
        rhat,
    )

    q_b = nb.amplitude_q2(
        fragment,
        atom_b,
        -rhat,
    )

    angular = torch.exp(
        nb.ELEMENT_PARAMETERS[
            symbol_a
        ].k * q_a
        + nb.ELEMENT_PARAMETERS[
            symbol_b
        ].k * q_b
    )

    return radial * angular


class SaptHStateBatchedSimulation(
    hs.HStateReferenceBatchedSimulation
):
    """
    Full ChemistryModel base potential with the H-state radial
    decomposition replaced by the independently calibrated SAPT wall.

    Everything outside the H state correction remains the ordinary
    ChemistryModel potential.
    """

    physics_model_name = (
        "SAPT-wall hydrogen-state reference"
    )

    physics_model_revision = (
        "sapt-wall-v1"
    )

    def __init__(
        self,
        *args,
        h_state_mixing=SAPT_H_STATE_MIXING,
        **kwargs,
    ):
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
        taper = values["taper"]
        pair_depth = values["pair_depth"]
        pair_width = values["pair_width"]
        shift = values["shift"]
        repulsive = values["repulsive"]

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
                "SAPT H-state requires reactive intermediates"
            )

        positions = cached[0]

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

        diagonals = []

        for state in states:

            weights = torch.zeros(
                (
                    self.per_box,
                    self.per_box,
                ),
                dtype=positions.dtype,
                device=positions.device,
            )

            for edge_index in state:

                global_a, global_b = (
                    edge_atoms[
                        edge_index
                    ]
                )

                local_a = (
                    global_a
                    - start
                )

                local_b = (
                    global_b
                    - start
                )

                # Continuous occupancy follows the same contact taper
                # already used by the reactive state machinery.
                occupancy = (
                    edge_tapers[
                        edge_index
                    ]
                )

                weights[
                    local_a,
                    local_b,
                ] = occupancy

                weights[
                    local_b,
                    local_a,
                ] = occupancy

            fragment = (
                nb.ContinuousTorchFragment(
                    symbols=local_symbols,
                    positions=local_positions,
                    bond_weights=weights,
                )
            )

            # Occupied H edges retain the complete tapered Morse
            # covalent interaction.
            if state:
                covalent = torch.stack([
                    edge_repulsive[index]
                    - edge_attractive[index]
                    for index in state
                ]).sum()
            else:
                covalent = zero

            # Unoccupied H candidate edges receive the SAPT exchange wall.
            wall = zero

            occupied = set(
                state
            )

            for edge_index, (
                global_a,
                global_b,
            ) in enumerate(
                edge_atoms
            ):
                if edge_index in occupied:
                    continue

                local_a = (
                    global_a
                    - start
                )

                local_b = (
                    global_b
                    - start
                )

                wall = (
                    wall
                    + edge_tapers[
                        edge_index
                    ]
                    * _sapt_pair_energy(
                        fragment,
                        local_a,
                        local_b,
                    )
                )

            diagonals.append(
                covalent
                + wall
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
                    (first, second)
                ] = (
                    old_index,
                    new_index,
                    overlap,
                )

                weighted_degree[first] = (
                    weighted_degree[first]
                    + overlap*overlap
                )

                weighted_degree[second] = (
                    weighted_degree[second]
                    + overlap*overlap
                )

        normalisation = torch.stack([
            hs._crowding_normalisation(
                value
            )
            for value
            in weighted_degree
        ])

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
                        -couplings[key]
                        if key in couplings
                        else zero
                    )

                row.append(
                    value
                )

            rows.append(
                torch.stack(row)
            )

        hamiltonian = (
            torch.stack(rows)
        )

        eigenvalues = (
            torch.linalg.eigvalsh(
                hamiltonian
            )
        )

        return eigenvalues[0]