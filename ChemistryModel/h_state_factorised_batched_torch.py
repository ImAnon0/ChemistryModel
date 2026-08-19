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
    6. calls torch.linalg.eigvalsh in bounded S=2 chunks per topology group

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

DEFAULT_H_S2_EIGVALSH_CHUNK_SIZE = 512
DEFAULT_H_TRANSITION_ASSEMBLY = "dense"


def _resolved_h_s2_eigvalsh_chunk_size(simulation):
    """Resolve the shared execution bound for every legal H-state caller."""

    chunk_size = int(
        getattr(
            simulation,
            "h_s2_eigvalsh_chunk_size",
            DEFAULT_H_S2_EIGVALSH_CHUNK_SIZE,
        )
    )
    if chunk_size < 0:
        raise ValueError(
            "h_s2_eigvalsh_chunk_size must be non-negative"
        )
    return chunk_size


def _bounded_factorised_eigvalsh(
    hamiltonian,
    s2_chunk_size,
):
    """Run S=2 eigvalsh in fixed full chunks, preserving batch order."""

    batch_size = int(hamiltonian.shape[0])
    state_count = int(hamiltonian.shape[-1])
    chunk_size = int(s2_chunk_size)

    if (
        state_count != 2
        or chunk_size <= 0
        or batch_size <= chunk_size
    ):
        return torch.linalg.eigvalsh(
            hamiltonian
        )

    chunks = []
    full_stop = (
        batch_size
        // chunk_size
        * chunk_size
    )

    for start in range(
        0,
        full_stop,
        chunk_size,
    ):
        chunks.append(
            torch.linalg.eigvalsh(
                hamiltonian[
                    start:
                    start + chunk_size
                ]
            )
        )

    if full_stop < batch_size:
        chunks.append(
            torch.linalg.eigvalsh(
                hamiltonian[
                    full_stop:
                ]
            )
        )

    return torch.cat(
        chunks,
        dim=0,
    )


def _record_h_eigvalsh_execution(
    simulation,
    hamiltonian,
):
    run = _h_state_run_diagnostics(
        simulation
    )
    batch_size = int(hamiltonian.shape[0])
    state_count = int(hamiltonian.shape[-1])
    chunk_size = _resolved_h_s2_eigvalsh_chunk_size(
        simulation
    )

    run["s2_eigvalsh_chunk_size"] = chunk_size

    if state_count != 2:
        run["largest_actual_eigvalsh_batch"] = max(
            int(run["largest_actual_eigvalsh_batch"]),
            batch_size,
        )
        return

    run["max_original_s2_batch"] = max(
        int(run["max_original_s2_batch"]),
        batch_size,
    )

    if chunk_size > 0 and batch_size > chunk_size:
        call_count = (
            batch_size
            + chunk_size
            - 1
        ) // chunk_size
        largest_submitted = chunk_size
        run["chunked_s2_group_count"] += 1
        run["chunked_s2_eigvalsh_calls"] += call_count
    else:
        call_count = 1
        largest_submitted = batch_size

    run["total_s2_eigvalsh_calls"] += call_count
    run["largest_actual_eigvalsh_batch"] = max(
        int(run["largest_actual_eigvalsh_batch"]),
        largest_submitted,
    )


def _fresh_h_state_run_diagnostics():
    """Whole-run, observational H-state pressure counters."""

    return {
        "evaluation_count": 0,
        "max_component_edge_count": 0,
        "max_state_count": 0,
        "max_transition_count": 0,
        "max_topology_group": 0,
        "max_hamiltonian_shape": (),
        "max_hamiltonian_elements": 0,
        "max_hamiltonian_bytes": 0,
        "max_total_h_states_solved": 0,
        "max_topology_groups_per_evaluation": 0,
        "structure_cache_entries": 0,
        "structure_cache_cuda_bytes": 0,
        "cuda_memory_allocated": 0,
        "cuda_memory_reserved": 0,
        "cuda_max_memory_allocated": 0,
        "cuda_max_memory_reserved": 0,
        "memory_pressure_hamiltonian_shape": (),
        "memory_pressure_component_edges": 0,
        "memory_pressure_state_count": 0,
        "memory_pressure_transition_count": 0,
        "memory_pressure_topology_group": 0,
        "s2_eigvalsh_chunk_size": 0,
        "max_original_s2_batch": 0,
        "chunked_s2_group_count": 0,
        "chunked_s2_eigvalsh_calls": 0,
        "total_s2_eigvalsh_calls": 0,
        "largest_actual_eigvalsh_batch": 0,
    }


def _h_state_run_diagnostics(simulation):
    run = getattr(simulation, "_h_state_run_diagnostics", None)
    if run is None:
        run = _fresh_h_state_run_diagnostics()
        simulation._h_state_run_diagnostics = run
    return run


def _profile_stage_begin(simulation, name):
    sink = getattr(simulation, "_h_state_profile_sink", None)
    if sink is None:
        return None
    return sink.stage_begin(
        name,
        context=getattr(simulation, "_h_state_profile_context", None),
    )


def _profile_stage_end(simulation, name, token):
    if token is not None:
        simulation._h_state_profile_sink.stage_end(
            name,
            token,
            context=getattr(simulation, "_h_state_profile_context", None),
        )


def _begin_h_state_evaluation(simulation, topology_group_count):
    run = _h_state_run_diagnostics(simulation)
    run["evaluation_count"] += 1
    run["max_topology_groups_per_evaluation"] = max(
        int(run["max_topology_groups_per_evaluation"]),
        int(topology_group_count),
    )
    simulation._h_state_current_states_solved = 0


def _record_h_state_group_pressure(simulation, structure, component_count):
    run = _h_state_run_diagnostics(simulation)
    edge_count = int(structure["edge_count"])
    state_count = int(structure["state_count"])
    transition_count = int(structure["state_first"].numel())
    component_count = int(component_count)
    shape = (component_count, state_count, state_count)
    element_count = component_count * state_count * state_count
    raw_bytes = element_count * int(structure["state_mask"].element_size())

    run["max_component_edge_count"] = max(
        int(run["max_component_edge_count"]), edge_count
    )
    run["max_state_count"] = max(
        int(run["max_state_count"]), state_count
    )
    run["max_transition_count"] = max(
        int(run["max_transition_count"]), transition_count
    )
    run["max_topology_group"] = max(
        int(run["max_topology_group"]), component_count
    )
    if element_count > int(run["max_hamiltonian_elements"]):
        run["max_hamiltonian_elements"] = element_count
        run["max_hamiltonian_shape"] = shape
        run["max_hamiltonian_bytes"] = raw_bytes

    simulation._h_state_current_states_solved += component_count * state_count
    pressure = {
        "shape": shape,
        "edge_count": edge_count,
        "state_count": state_count,
        "transition_count": transition_count,
        "component_count": component_count,
    }

    if getattr(simulation, "_h_state_profile_sink", None) is not None:
        simulation._h_state_profile_context = dict(pressure)

    return pressure


def _record_h_state_cuda_memory(simulation, group_pressure):
    device = torch.device(simulation.device)
    if device.type != "cuda":
        return

    run = _h_state_run_diagnostics(simulation)
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    previous_allocated = int(run["cuda_memory_allocated"])
    previous_pressure = int(run["cuda_memory_reserved"])

    run["cuda_memory_allocated"] = max(
        int(run["cuda_memory_allocated"]), allocated
    )
    run["cuda_memory_reserved"] = max(previous_pressure, reserved)
    run["cuda_max_memory_allocated"] = max(
        int(run["cuda_max_memory_allocated"]),
        int(torch.cuda.max_memory_allocated(device)),
    )
    run["cuda_max_memory_reserved"] = max(
        int(run["cuda_max_memory_reserved"]),
        int(torch.cuda.max_memory_reserved(device)),
    )

    if (
        reserved > previous_pressure
        or allocated > previous_allocated
    ):
        run["memory_pressure_hamiltonian_shape"] = tuple(
            group_pressure["shape"]
        )
        run["memory_pressure_component_edges"] = int(
            group_pressure["edge_count"]
        )
        run["memory_pressure_state_count"] = int(
            group_pressure["state_count"]
        )
        run["memory_pressure_transition_count"] = int(
            group_pressure["transition_count"]
        )
        run["memory_pressure_topology_group"] = int(
            group_pressure["component_count"]
        )


def _finish_h_state_evaluation(simulation):
    run = _h_state_run_diagnostics(simulation)
    run["max_total_h_states_solved"] = max(
        int(run["max_total_h_states_solved"]),
        int(getattr(simulation, "_h_state_current_states_solved", 0)),
    )

    structure_cache = getattr(simulation, "_factorised_h_structure_cache", {})
    run["structure_cache_entries"] = max(
        int(run["structure_cache_entries"]), len(structure_cache)
    )

    cuda_bytes = 0
    for structure in structure_cache.values():
        for value in structure.values():
            if torch.is_tensor(value) and value.device.type == "cuda":
                cuda_bytes += int(value.numel()) * int(value.element_size())
    run["structure_cache_cuda_bytes"] = max(
        int(run["structure_cache_cuda_bytes"]), cuda_bytes
    )


def _assemble_factorised_hamiltonian(
    diagonal,
    coupling,
    transition_flat_index,
):
    """Assemble the exact symmetric H-state Hamiltonian compactly."""

    hamiltonian = torch.diag_embed(
        diagonal
    )

    if transition_flat_index.numel() == 0:
        return hamiltonian

    component_count = int(
        diagonal.shape[0]
    )

    state_count = int(
        diagonal.shape[1]
    )

    flat_values = torch.cat(
        (
            -coupling,
            -coupling,
        ),
        dim=1,
    )

    off_diagonal = torch.zeros(
        (
            component_count,
            state_count * state_count,
        ),
        device=diagonal.device,
        dtype=diagonal.dtype,
    )

    off_diagonal = off_diagonal.scatter_add(
        1,
        transition_flat_index[
            None,
            :,
        ].expand(
            component_count,
            -1,
        ),
        flat_values,
    )

    return (
        hamiltonian
        + off_diagonal.reshape(
            component_count,
            state_count,
            state_count,
        )
    )


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
        h_s2_eigvalsh_chunk_size=(
            DEFAULT_H_S2_EIGVALSH_CHUNK_SIZE
        ),
        h_transition_assembly=(
            DEFAULT_H_TRANSITION_ASSEMBLY
        ),
        **kwargs,
    ):
        # Purely discrete cache:
        # signature -> state masks / transition tensors.
        self._factorised_h_structure_cache = {}
        self.h_s2_eigvalsh_chunk_size = int(
            h_s2_eigvalsh_chunk_size
        )
        if self.h_s2_eigvalsh_chunk_size < 0:
            raise ValueError(
                "h_s2_eigvalsh_chunk_size must be non-negative"
            )
        self.h_transition_assembly = str(
            h_transition_assembly
        )
        if self.h_transition_assembly not in {"compact", "dense"}:
            raise ValueError(
                "h_transition_assembly must be 'compact' or 'dense'"
            )

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
            transition_flat_index = torch.cat(
                (
                    state_first * state_count
                    + state_second,
                    state_second * state_count
                    + state_first,
                )
            )
        else:
            transition_flat_index = torch.zeros(
                (0,),
                device=self.device,
                dtype=torch.long,
            )

        if getattr(
            self,
            "h_transition_assembly",
            DEFAULT_H_TRANSITION_ASSEMBLY,
        ) == "dense":
            transition_basis = torch.zeros(
                (transition_count, state_count, state_count),
                device=self.device,
                dtype=self.dtype,
            )
            if transition_count:
                transition_index = torch.arange(
                    transition_count,
                    device=self.device,
                    dtype=torch.long,
                )
                transition_basis[
                    transition_index, state_first, state_second
                ] = 1.0
                transition_basis[
                    transition_index, state_second, state_first
                ] = 1.0
        else:
            transition_basis = None

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
            "transition_flat_index": (
                transition_flat_index
            ),
        }
        if transition_basis is not None:
            cached["transition_basis"] = transition_basis

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

        group_pressure = _record_h_state_group_pressure(
            self,
            structure,
            component_count,
        )

        edge_count = structure[
            "edge_count"
        ]

        if edge_count == 0:
            _record_h_state_cuda_memory(
                self,
                group_pressure,
            )
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

        physics_token = _profile_stage_begin(
            self,
            "H coupling/physics construction",
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
            _profile_stage_end(
                self,
                "H coupling/physics construction",
                physics_token,
            )
            _record_h_state_cuda_memory(
                self,
                group_pressure,
            )
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

            _profile_stage_end(
                self,
                "H coupling/physics construction",
                physics_token,
            )
            assembly_token = _profile_stage_begin(
                self,
                "Hamiltonian assembly",
            )

            if getattr(
                self,
                "h_transition_assembly",
                DEFAULT_H_TRANSITION_ASSEMBLY,
            ) == "dense":
                hamiltonian = torch.diag_embed(diagonal) + torch.einsum(
                    "bt,tij->bij",
                    -coupling,
                    structure["transition_basis"],
                )
            else:
                hamiltonian = _assemble_factorised_hamiltonian(
                    diagonal,
                    coupling,
                    structure["transition_flat_index"],
                )
            _profile_stage_end(
                self,
                "Hamiltonian assembly",
                assembly_token,
            )
        else:
            _profile_stage_end(
                self,
                "H coupling/physics construction",
                physics_token,
            )
            assembly_token = _profile_stage_begin(
                self,
                "Hamiltonian assembly",
            )
            hamiltonian = (
                torch.diag_embed(
                    diagonal
                )
            )
            _profile_stage_end(
                self,
                "Hamiltonian assembly",
                assembly_token,
            )

        eig_token = _profile_stage_begin(
            self,
            "torch.linalg.eigvalsh",
        )
        _record_h_eigvalsh_execution(
            self,
            hamiltonian,
        )
        eigenvalues = _bounded_factorised_eigvalsh(
            hamiltonian,
            _resolved_h_s2_eigvalsh_chunk_size(self),
        )
        _profile_stage_end(
            self,
            "torch.linalg.eigvalsh",
            eig_token,
        )
        _record_h_state_cuda_memory(
            self,
            group_pressure,
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

        _begin_h_state_evaluation(
            self,
            len(grouped),
        )

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

        _finish_h_state_evaluation(
            self
        )

        return correction


# Aliases.
BatchedFactorisedHStateBatchedSimulation = (
    GroupedFactorisedHStateBatchedSimulation
)

GroupedFactorisableHStateBatchedSimulation = (
    GroupedFactorisedHStateBatchedSimulation
)
