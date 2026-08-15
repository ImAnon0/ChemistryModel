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


def _symmetric_pair_basis(
    atom_count,
    first,
    second,
    *,
    like,
):
    """Constant symmetric matrix selecting one undirected atom pair."""

    basis = torch.zeros(
        (
            atom_count,
            atom_count,
        ),
        dtype=like.dtype,
        device=like.device,
    )

    basis[
        first,
        second,
    ] = 1.0

    basis[
        second,
        first,
    ] = 1.0

    return basis


def _descriptor_weights_for_state(
    *,
    box,
    per_box,
    types_numpy,
    neighbours,
    neighbour_mask,
    taper,
    edge_atoms,
    edge_tapers,
    state,
):
    """
    Build the continuous covalent-environment matrix seen by SAPT.

    Two kinds of covalent information enter:

      heavy-heavy:
          state-independent reactive contacts, weighted continuously by
          the same smooth taper already used by ChemistryModel.

      H-containing:
          only the H-state edges occupied in this diabatic state, again
          weighted by their current smooth contact taper.

    This is the bridge required for carbonyl chemistry. Without the
    heavy-heavy part a C=O bond disappears from the SAPT descriptor, so
    the independently calibrated polar-pi / zeta response cannot operate.

    Important limitation:
    ChemistryModel does not yet have an explicit heavy-atom covalent-state
    graph. The reactive taper is therefore the only smooth bond-activation
    variable available for heavy-heavy contacts. This helper does NOT use
    a hard bond threshold, so forces can still flow continuously through
    the descriptor weights.
    """

    zero = (
        taper.sum()
        * 0.0
    )

    weights = torch.zeros(
        (
            per_box,
            per_box,
        ),
        dtype=taper.dtype,
        device=taper.device,
    ) + zero

    start = (
        box
        * per_box
    )

    stop = (
        start
        + per_box
    )

    hydrogen = int(
        R.ELEMENT_INDEX[
            "H"
        ]
    )

    neighbours_numpy = (
        neighbours
        .detach()
        .cpu()
        .numpy()
    )

    mask_numpy = (
        neighbour_mask
        .detach()
        .cpu()
        .numpy()
        .astype(bool)
    )

    seen_heavy_pairs = set()

    for centre in range(
        start,
        stop,
    ):
        for slot in range(
            neighbours_numpy.shape[1]
        ):
            if not mask_numpy[
                centre,
                slot,
            ]:
                continue

            other = int(
                neighbours_numpy[
                    centre,
                    slot,
                ]
            )

            if not (
                start
                <= other
                < stop
            ):
                continue

            first = min(
                centre,
                other,
            )

            second = max(
                centre,
                other,
            )

            if first == second:
                continue

            pair = (
                first,
                second,
            )

            if pair in seen_heavy_pairs:
                continue

            if (
                int(
                    types_numpy[
                        first
                    ]
                )
                == hydrogen
                or int(
                    types_numpy[
                        second
                    ]
                )
                == hydrogen
            ):
                continue

            seen_heavy_pairs.add(
                pair
            )

            local_first = (
                first
                - start
            )

            local_second = (
                second
                - start
            )

            # Use this directed copy of the symmetric reactive taper.
            # No threshold: outside the covalent cutoff it is already
            # exactly zero, while inside the fade region it remains smooth.
            contact = taper[
                centre,
                slot,
            ]

            weights = (
                weights
                + _symmetric_pair_basis(
                    per_box,
                    local_first,
                    local_second,
                    like=taper,
                )
                * contact
            )

    # H edges are state-specific. Unoccupied alternatives must NOT appear
    # as covalent neighbours in the SAPT environment descriptor.
    for edge_index in state:

        global_first, global_second = (
            edge_atoms[
                edge_index
            ]
        )

        local_first = (
            global_first
            - start
        )

        local_second = (
            global_second
            - start
        )

        weights = (
            weights
            + _symmetric_pair_basis(
                per_box,
                local_first,
                local_second,
                like=taper,
            )
            * edge_tapers[
                edge_index
            ]
        )

    # Every pair belongs to exactly one category above, so values should
    # already lie in [0, 1]. Clamp only for sub-ulp numerical excursions.
    return torch.clamp(
        weights,
        min=0.0,
        max=1.0,
    )


class SaptHStateBatchedSimulation(
    hs.HStateReferenceBatchedSimulation
):
    """
    Full ChemistryModel base potential with the H-state radial
    decomposition replaced by the independently calibrated SAPT wall.

    Everything outside the H state correction remains the ordinary
    ChemistryModel potential.

    SAPT environment descriptors receive:
      - state-independent heavy-heavy reactive covalent contacts
      - state-dependent occupied H contacts

    That keeps C=O/C=C/etc. chemistry visible to the frozen SAPT descriptor
    while unoccupied H-transfer alternatives remain noncovalent.
    """

    physics_model_name = (
        "SAPT-wall hydrogen-state reference"
    )

    physics_model_revision = (
        "sapt-wall-v2-heavy-env"
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
