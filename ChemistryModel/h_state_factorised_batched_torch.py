"""
Grouped / batched execution for the validated factorisable local H-state.

Physics is unchanged from h_state_factorised_torch.py.

The reference factorisable implementation:
    box -> H component -> Python Hamiltonian -> eigvalsh
repeats many tiny eigensolves.

This execution layer:
    1. discovers the exact same active H edges
    2. builds the exact same local H-conflict components
    3. converts each component to an H-incidence topology signature
    4. groups topology-identical components across ALL simulation boxes
    5. builds their Hamiltonians as a batch
    6. calls torch.linalg.eigvalsh once per topology group

Only discrete topology-derived tensors are cached.  Tapers, depths, Morse
terms, overlaps, normalisations and couplings remain live tensors on every
energy evaluation, so autograd still differentiates the same potential.

No chemistry parameter is added or changed.
"""

from __future__ import annotations

from collections import defaultdict
import itertools

import torch

import reactive as R

from h_state_factorised_torch import (
    FactorisedHStateBatchedSimulation,
    MAX_FACTORISED_STATES,
)
from h_state_torch import (
    _contact_overlap,
    _crowding_normalisation,
)


BATCHED_FACTORISABLE_H_STATE_MODEL_NAME = (
    "reactive_v2_h_state_factorisable_components_grouped_execution"
)

BATCHED_FACTORISABLE_H_STATE_MODEL_REVISION = 0


def _states_for_incidence(edge_hydrogens):
    """
    All H-valence-valid edge subsets using only H incidence.

    Heavy-atom identity is irrelevant to the H valence rule.  A state is
    valid iff no hydrogen occurs in more than one selected edge.
    """

    edge_count = len(edge_hydrogens)
    states = []

    for count in range(edge_count + 1):
        for chosen in itertools.combinations(
            range(edge_count),
            count,
        ):
            used_hydrogens = set()
            valid = True

            for edge_index in chosen:
                for hydrogen in edge_hydrogens[
                    edge_index
                ]:
                    if hydrogen in used_hydrogens:
                        valid = False
                        break

                    used_hydrogens.add(
                        hydrogen
                    )

                if not valid:
                    break

            if valid:
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


class GroupedFactorisedHStateBatchedSimulation(
    FactorisedHStateBatchedSimulation
):
    """
    Factorisable H-state with topology-grouped batched Hamiltonian execution.
    """

    physics_model_name = (
        BATCHED_FACTORISABLE_H_STATE_MODEL_NAME
    )

    physics_model_revision = (
        BATCHED_FACTORISABLE_H_STATE_MODEL_REVISION
    )

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # Purely discrete cache:
        # signature -> state masks / transition tensors.
        self._factorised_h_structure_cache = {}

        super().__init__(
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Canonical local topology
    # ------------------------------------------------------------------

    def _factorised_component_signature(
        self,
        edge_atoms,
    ):
        """
        Canonicalise a local H component by H incidence.

        Global atom numbers and heavy-atom identities do not affect which
        H-valence states are allowed.  Scan the component's already-stable
        edge order and renumber H atoms by first occurrence.

        Examples:
            heavy-H / heavy-H competition:
                ((0,), (0,))

            H-H bridge followed by H-heavy:
                ((0, 1), (1,))
        """

        hydrogen_type = int(
            R.ELEMENT_INDEX["H"]
        )

        hydrogen_labels = {}
        next_label = 0

        incidence = []

        for first, second in edge_atoms:
            edge_hydrogens = []

            for atom in (
                first,
                second,
            ):
                if (
                    int(
                        self.types_numpy[
                            atom
                        ]
                    )
                    != hydrogen_type
                ):
                    continue

                label = (
                    hydrogen_labels.get(
                        atom
                    )
                )

                if label is None:
                    label = next_label
                    hydrogen_labels[
                        atom
                    ] = label
                    next_label += 1

                edge_hydrogens.append(
                    label
                )

            if not edge_hydrogens:
                raise RuntimeError(
                    "factorised H component contained a non-H edge"
                )

            incidence.append(
                tuple(
                    sorted(
                        edge_hydrogens
                    )
                )
            )

        return tuple(
            incidence
        )

    def _factorised_structure_for_signature(
        self,
        signature,
    ):
        cached = (
            self._factorised_h_structure_cache.get(
                signature
            )
        )

        if cached is not None:
            return cached

        edge_hydrogens = tuple(
            tuple(
                int(value)
                for value in edge
            )
            for edge in signature
        )

        states = _states_for_incidence(
            edge_hydrogens
        )

        edge_count = len(
            edge_hydrogens
        )

        state_count = len(
            states
        )

        state_mask = torch.zeros(
            (
                state_count,
                edge_count,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        for state_index, state in enumerate(
            states
        ):
            if state:
                indices = torch.tensor(
                    state,
                    device=self.device,
                    dtype=torch.long,
                )

                state_mask[
                    state_index,
                    indices,
                ] = 1.0

        transition_state_first = []
        transition_state_second = []
        transition_old_edge = []
        transition_new_edge = []
        transition_hydrogen = []

        for first_state in range(
            state_count
        ):
            first_set = set(
                states[
                    first_state
                ]
            )

            for second_state in range(
                first_state + 1,
                state_count,
            ):
                second_set = set(
                    states[
                        second_state
                    ]
                )

                removed = list(
                    first_set
                    - second_set
                )

                added = list(
                    second_set
                    - first_set
                )

                if (
                    len(removed) != 1
                    or len(added) != 1
                ):
                    continue

                old_edge = removed[0]
                new_edge = added[0]

                shared_hydrogens = set(
                    edge_hydrogens[
                        old_edge
                    ]
                ).intersection(
                    edge_hydrogens[
                        new_edge
                    ]
                )

                if len(
                    shared_hydrogens
                ) != 1:
                    continue

                shared_hydrogen = int(
                    next(
                        iter(
                            shared_hydrogens
                        )
                    )
                )

                transition_state_first.append(
                    first_state
                )

                transition_state_second.append(
                    second_state
                )

                transition_old_edge.append(
                    old_edge
                )

                transition_new_edge.append(
                    new_edge
                )

                transition_hydrogen.append(
                    shared_hydrogen
                )

        state_first = torch.tensor(
            transition_state_first,
            device=self.device,
            dtype=torch.long,
        )

        state_second = torch.tensor(
            transition_state_second,
            device=self.device,
            dtype=torch.long,
        )

        old_edge = torch.tensor(
            transition_old_edge,
            device=self.device,
            dtype=torch.long,
        )

        new_edge = torch.tensor(
            transition_new_edge,
            device=self.device,
            dtype=torch.long,
        )

        # Local crowding is keyed by (state, transferred H).
        degree_keys = {}
        first_degree_key = []
        second_degree_key = []

        for (
            first_state,
            second_state,
            hydrogen,
        ) in zip(
            transition_state_first,
            transition_state_second,
            transition_hydrogen,
        ):
            first_key = (
                first_state,
                hydrogen,
            )

            second_key = (
                second_state,
                hydrogen,
            )

            if first_key not in degree_keys:
                degree_keys[
                    first_key
                ] = len(
                    degree_keys
                )

            if second_key not in degree_keys:
                degree_keys[
                    second_key
                ] = len(
                    degree_keys
                )

            first_degree_key.append(
                degree_keys[
                    first_key
                ]
            )

            second_degree_key.append(
                degree_keys[
                    second_key
                ]
            )

        first_degree_key = torch.tensor(
            first_degree_key,
            device=self.device,
            dtype=torch.long,
        )

        second_degree_key = torch.tensor(
            second_degree_key,
            device=self.device,
            dtype=torch.long,
        )

        transition_count = len(
            transition_state_first
        )

        if transition_count:
            transition_basis = torch.zeros(
                (
                    transition_count,
                    state_count,
                    state_count,
                ),
                device=self.device,
                dtype=self.dtype,
            )

            transition_index = torch.arange(
                transition_count,
                device=self.device,
                dtype=torch.long,
            )

            transition_basis[
                transition_index,
                state_first,
                state_second,
            ] = 1.0

            transition_basis[
                transition_index,
                state_second,
                state_first,
            ] = 1.0
        else:
            transition_basis = torch.zeros(
                (
                    0,
                    state_count,
                    state_count,
                ),
                device=self.device,
                dtype=self.dtype,
            )

        cached = {
            "states": states,
            "edge_count": edge_count,
            "state_count": state_count,
            "state_mask": state_mask,
            "state_first": state_first,
            "state_second": state_second,
            "old_edge": old_edge,
            "new_edge": new_edge,
            "first_degree_key": (
                first_degree_key
            ),
            "second_degree_key": (
                second_degree_key
            ),
            "degree_key_count": len(
                degree_keys
            ),
            "transition_basis": (
                transition_basis
            ),
        }

        self._factorised_h_structure_cache[
            signature
        ] = cached

        return cached

    # ------------------------------------------------------------------
    # Batched local Hamiltonian
    # ------------------------------------------------------------------

    def _factorised_group_state_energies(
        self,
        group,
        structure,
        values,
    ):
        """
        Lowest state energy for topology-identical local H components.

        group entries contain equal-length edge_rows / edge_slots tuples.
        """

        component_count = len(
            group
        )

        edge_count = structure[
            "edge_count"
        ]

        if edge_count == 0:
            return (
                values["taper"].sum()
                * 0.0
                * torch.ones(
                    component_count,
                    device=self.device,
                    dtype=self.dtype,
                )
            )

        row_index = torch.tensor(
            [
                entry[
                    "edge_rows"
                ]
                for entry in group
            ],
            device=self.device,
            dtype=torch.long,
        )

        slot_index = torch.tensor(
            [
                entry[
                    "edge_slots"
                ]
                for entry in group
            ],
            device=self.device,
            dtype=torch.long,
        )

        taper = values[
            "taper"
        ][
            row_index,
            slot_index,
        ]

        pair_depth = values[
            "pair_depth"
        ][
            row_index,
            slot_index,
        ]

        pair_width = values[
            "pair_width"
        ][
            row_index,
            slot_index,
        ]

        shift = values[
            "shift"
        ][
            row_index,
            slot_index,
        ]

        repulsive = values[
            "repulsive"
        ][
            row_index,
            slot_index,
        ]

        attractive = (
            2.0
            * pair_depth
            * torch.exp(
                -pair_width
                * shift
            )
        )

        edge_repulsive = (
            taper
            * repulsive
        )

        edge_attractive = (
            taper
            * attractive
        )

        common_core = torch.sum(
            edge_repulsive,
            dim=1,
        )

        state_attraction = (
            edge_attractive
            @ structure[
                "state_mask"
            ].T
        )

        diagonal = (
            common_core[
                :,
                None,
            ]
            - state_attraction
        )

        if (
            structure[
                "state_count"
            ]
            == 1
        ):
            return diagonal[
                :,
                0,
            ]

        old_edge = structure[
            "old_edge"
        ]

        new_edge = structure[
            "new_edge"
        ]

        if len(
            old_edge
        ):
            overlap = _contact_overlap(
                taper[
                    :,
                    old_edge,
                ],
                taper[
                    :,
                    new_edge,
                ],
            )

            weighted = (
                overlap
                * overlap
            )

            degree = torch.zeros(
                (
                    component_count,
                    structure[
                        "degree_key_count"
                    ],
                ),
                device=self.device,
                dtype=self.dtype,
            )

            first_degree_key = structure[
                "first_degree_key"
            ]

            second_degree_key = structure[
                "second_degree_key"
            ]

            degree = (
                degree.scatter_add(
                    1,
                    first_degree_key[
                        None,
                        :,
                    ].expand(
                        component_count,
                        -1,
                    ),
                    weighted,
                )
            )

            degree = (
                degree.scatter_add(
                    1,
                    second_degree_key[
                        None,
                        :,
                    ].expand(
                        component_count,
                        -1,
                    ),
                    weighted,
                )
            )

            normalisation = (
                _crowding_normalisation(
                    degree
                )
            )

            depth_scale = torch.sqrt(
                torch.clamp(
                    pair_depth[
                        :,
                        old_edge,
                    ]
                    * pair_depth[
                        :,
                        new_edge,
                    ],
                    min=1e-12,
                )
            )

            denominator = torch.sqrt(
                torch.clamp(
                    normalisation[
                        :,
                        first_degree_key,
                    ]
                    * normalisation[
                        :,
                        second_degree_key,
                    ],
                    min=1e-12,
                )
            )

            coupling = (
                self.h_state_mixing
                * depth_scale
                * overlap
                / denominator
            )

            off_diagonal = torch.einsum(
                "bt,tij->bij",
                -coupling,
                structure[
                    "transition_basis"
                ],
            )

            hamiltonian = (
                torch.diag_embed(
                    diagonal
                )
                + off_diagonal
            )
        else:
            hamiltonian = (
                torch.diag_embed(
                    diagonal
                )
            )

        eigenvalues = torch.linalg.eigvalsh(
            hamiltonian
        )

        return eigenvalues[
            :,
            0,
        ]

    # ------------------------------------------------------------------
    # Exact component correction, grouped by topology
    # ------------------------------------------------------------------

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
            or cached[0]
            is not positions
        ):
            raise RuntimeError(
                "grouped factorisable H-state requires current "
                "reactive intermediates"
            )

        values = cached[1]

        # Keep the reference discrete discovery unchanged for this first
        # performance layer.  A later optimisation can compact the GPU->CPU
        # topology transfer once grouped execution is independently validated.
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
                > 1e-12
            )
            & self.neighbour_mask
            .detach()
            .cpu()
            .numpy()
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

        attractive = (
            2.0
            * pair_depth
            * torch.exp(
                -pair_width
                * shift
            )
        )

        pair_morse = (
            taper
            * (
                repulsive
                - attractive
            )
        )

        over_scale = (
            self.over_coordination_scale(
                taper,
                values[
                    "unsoftened_depth"
                ],
                values[
                    "mask"
                ],
                cache_key=positions,
            )
        )

        excess = torch.clamp(
            values[
                "coordination"
            ]
            - values[
                "valence"
            ],
            min=0.0,
        )

        base_over = (
            self.over_penalty
            * over_scale
            * excess
            * excess
        )

        hydrogen_type = int(
            R.ELEMENT_INDEX["H"]
        )

        grouped = defaultdict(
            list
        )

        component_counts = []
        largest_component_edges = 0
        component_total = 0

        for box in range(
            self.box_count
        ):
            start = (
                box
                * self.per_box
            )

            (
                edge_atoms,
                edge_rows,
                edge_slots,
            ) = (
                self._active_edges_for_box(
                    box,
                    values,
                    neighbours_numpy,
                    active_numpy,
                )
            )

            if not edge_atoms:
                component_counts.append(
                    0
                )
                continue

            components = (
                self._hydrogen_edge_components(
                    edge_atoms
                )
            )

            component_counts.append(
                len(
                    components
                )
            )

            for component in components:
                component_total += 1

                largest_component_edges = max(
                    largest_component_edges,
                    len(
                        component
                    ),
                )

                component_edge_atoms = tuple(
                    edge_atoms[
                        index
                    ]
                    for index in component
                )

                component_edge_rows = tuple(
                    int(
                        edge_rows[
                            index
                        ]
                    )
                    for index in component
                )

                component_edge_slots = tuple(
                    int(
                        edge_slots[
                            index
                        ]
                    )
                    for index in component
                )

                component_hydrogens = tuple(
                    sorted({
                        atom
                        for first, second
                        in component_edge_atoms
                        for atom
                        in (
                            first,
                            second,
                        )
                        if int(
                            self.types_numpy[
                                atom
                            ]
                        )
                        == hydrogen_type
                    })
                )

                if not component_hydrogens:
                    raise RuntimeError(
                        "H component unexpectedly contains no H atoms"
                    )

                signature = (
                    self._factorised_component_signature(
                        component_edge_atoms
                    )
                )

                grouped[
                    signature
                ].append({
                    "box": int(
                        box
                    ),
                    "edge_rows": (
                        component_edge_rows
                    ),
                    "edge_slots": (
                        component_edge_slots
                    ),
                    "hydrogens": (
                        component_hydrogens
                    ),
                    "anchor": int(
                        component_hydrogens[
                            0
                        ]
                    ),
                })

        correction = (
            torch.zeros_like(
                base_per_atom
            )
            + base_per_atom.sum()
            * 0.0
        )

        largest_group = 0

        for signature, group in (
            grouped.items()
        ):
            largest_group = max(
                largest_group,
                len(
                    group
                ),
            )

            structure = (
                self._factorised_structure_for_signature(
                    signature
                )
            )

            state_energies = (
                self._factorised_group_state_energies(
                    group,
                    structure,
                    values,
                )
            )

            row_index = torch.tensor(
                [
                    entry[
                        "edge_rows"
                    ]
                    for entry in group
                ],
                device=self.device,
                dtype=torch.long,
            )

            slot_index = torch.tensor(
                [
                    entry[
                        "edge_slots"
                    ]
                    for entry in group
                ],
                device=self.device,
                dtype=torch.long,
            )

            base_h_pair = torch.sum(
                pair_morse[
                    row_index,
                    slot_index,
                ],
                dim=1,
            )

            base_h_over_values = []

            for entry_index, entry in enumerate(
                group
            ):
                atom_index = torch.tensor(
                    entry[
                        "hydrogens"
                    ],
                    device=self.device,
                    dtype=torch.long,
                )

                if len(
                    entry[
                        "hydrogens"
                    ]
                ):
                    base_h_over_values.append(
                        torch.sum(
                            base_over[
                                atom_index
                            ]
                        )
                    )
                else:
                    base_h_over_values.append(
                        state_energies[
                            entry_index
                        ]
                        * 0.0
                    )

            base_h_over = torch.stack(
                base_h_over_values
            )

            delta = (
                state_energies
                - base_h_pair
                - base_h_over
            )

            anchor_index = torch.tensor(
                [
                    entry[
                        "anchor"
                    ]
                    for entry in group
                ],
                device=self.device,
                dtype=torch.long,
            )

            correction = (
                correction.scatter_add(
                    0,
                    anchor_index,
                    delta,
                )
            )

        self._h_component_diagnostics = {
            "component_counts_per_box": tuple(
                component_counts
            ),
            "largest_component_edges": int(
                largest_component_edges
            ),
            "component_total": int(
                component_total
            ),
            "topology_group_count": int(
                len(
                    grouped
                )
            ),
            "largest_topology_group": int(
                largest_group
            ),
        }

        return correction


# Aliases.
BatchedFactorisedHStateBatchedSimulation = (
    GroupedFactorisedHStateBatchedSimulation
)

GroupedFactorisableHStateBatchedSimulation = (
    GroupedFactorisedHStateBatchedSimulation
)
