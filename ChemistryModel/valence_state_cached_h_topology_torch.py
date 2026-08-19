"""
Cached-topology execution for the validated grouped factorisable H-state.

Physics is unchanged.

Why this layer exists
---------------------
The grouped H-state already batches topology-identical local Hamiltonians, but
it still repeats the DISCRETE work on every force evaluation:

    copy full neighbour/taper tables to CPU
    scan every padded neighbour slot
    deduplicate H-containing pairs
    rebuild H-conflict connected components
    rebuild Python group dictionaries
    recreate row/slot/anchor tensors

In MD, that graph is usually unchanged for many force steps.  It changes only
when:
    1. the neighbour table is rebuilt, or
    2. an H-containing candidate crosses the active taper threshold.

This implementation therefore:
    - caches all possible unique H-containing candidate pairs for the CURRENT
      neighbour table
    - transfers only their active/inactive boolean vector each evaluation
    - reuses fully tensorised component/group metadata while that vector is
      unchanged
    - rebuilds metadata immediately when the active vector changes
    - invalidates everything when rebuild_count changes

Live taper, depth, width, shift, repulsive, overlap, coupling and eigensolve
quantities are still evaluated from the current autograd tensors every call.
Only discrete topology/index bookkeeping is cached.

No chemistry parameter or equation is changed.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

import reactive as R

from h_state_torch import (
    _contact_overlap,
    _crowding_normalisation,
)
from h_state_factorised_batched_torch import (
    DEFAULT_H_S2_EIGVALSH_CHUNK_SIZE,
    DEFAULT_H_TRANSITION_ASSEMBLY,
    _begin_h_state_evaluation,
    _assemble_factorised_hamiltonian,
    _bounded_factorised_eigvalsh,
    _finish_h_state_evaluation,
    _profile_stage_begin,
    _profile_stage_end,
    _record_h_state_cuda_memory,
    _record_h_state_group_pressure,
    _record_h_eigvalsh_execution,
    _resolved_h_s2_eigvalsh_chunk_size,
)

from valence_state_batched_membership_torch import (
    BatchedHeavyValenceStateBatchedSimulation,
)


CACHED_H_TOPOLOGY_MODEL_NAME = (
    "reactive_v6_factorisable_h_cached_topology_batched_heavy_experimental"
)

CACHED_H_TOPOLOGY_MODEL_REVISION = 1


class CachedHFastValenceStateBatchedSimulation(
    BatchedHeavyValenceStateBatchedSimulation
):
    """
    Fully batched heavy membership plus cached grouped H topology.
    """

    physics_model_name = CACHED_H_TOPOLOGY_MODEL_NAME
    physics_model_revision = CACHED_H_TOPOLOGY_MODEL_REVISION

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
        # These must exist before parent construction: ReactiveSimulation's
        # __init__ evaluates initial forces and can dispatch here.
        self._h_candidate_rebuild_count = None
        self._h_candidate_cache = None
        self._h_last_active_signature = None
        self._h_last_topology_metadata = None
        self._h_topology_cache_hits = 0
        self._h_topology_cache_misses = 0
        self.h_s2_eigvalsh_chunk_size = int(
            h_s2_eigvalsh_chunk_size
        )
        if self.h_s2_eigvalsh_chunk_size < 0:
            raise ValueError(
                "h_s2_eigvalsh_chunk_size must be non-negative"
            )
        self.h_transition_assembly = str(h_transition_assembly)
        if self.h_transition_assembly not in {"compact", "dense"}:
            raise ValueError(
                "h_transition_assembly must be 'compact' or 'dense'"
            )

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Candidate pairs fixed by the current neighbour table
    # ------------------------------------------------------------------

    def _prepare_h_candidate_cache(self):
        if (
            self._h_candidate_cache is not None
            and self._h_candidate_rebuild_count
            == int(self.rebuild_count)
        ):
            return self._h_candidate_cache

        neighbours_numpy = (
            self.neighbours
            .detach()
            .cpu()
            .numpy()
        )

        mask_numpy = (
            self.neighbour_mask
            .detach()
            .cpu()
            .numpy()
        )

        hydrogen = int(
            R.ELEMENT_INDEX["H"]
        )

        pairs = []
        rows = []
        slots = []
        boxes = []
        box_ranges = []

        for box in range(self.box_count):
            start = box * self.per_box
            stop = start + self.per_box

            found = {}

            # Same row-major representative rule as
            # HStateReferenceBatchedSimulation._active_edges_for_box(),
            # except candidate discovery ignores taper activity.
            for centre in range(start, stop):
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

                    if (
                        int(
                            self.types_numpy[
                                first
                            ]
                        )
                        != hydrogen
                        and int(
                            self.types_numpy[
                                second
                            ]
                        )
                        != hydrogen
                    ):
                        continue

                    pair = (
                        first,
                        second,
                    )

                    if pair not in found:
                        found[pair] = (
                            centre,
                            slot,
                        )

            begin = len(
                pairs
            )

            for pair in sorted(
                found
            ):
                row, slot = (
                    found[pair]
                )

                pairs.append(
                    pair
                )
                rows.append(
                    int(row)
                )
                slots.append(
                    int(slot)
                )
                boxes.append(
                    int(box)
                )

            box_ranges.append(
                (
                    begin,
                    len(pairs),
                )
            )

        if rows:
            row_tensor = torch.tensor(
                rows,
                device=self.device,
                dtype=torch.long,
            )

            slot_tensor = torch.tensor(
                slots,
                device=self.device,
                dtype=torch.long,
            )
        else:
            row_tensor = torch.empty(
                (0,),
                device=self.device,
                dtype=torch.long,
            )

            slot_tensor = torch.empty(
                (0,),
                device=self.device,
                dtype=torch.long,
            )

        cache = {
            "pairs": tuple(
                pairs
            ),
            "rows": tuple(
                rows
            ),
            "slots": tuple(
                slots
            ),
            "boxes": tuple(
                boxes
            ),
            "box_ranges": tuple(
                box_ranges
            ),
            "row_tensor": (
                row_tensor
            ),
            "slot_tensor": (
                slot_tensor
            ),
            "candidate_count": int(
                len(
                    pairs
                )
            ),
        }

        self._h_candidate_rebuild_count = int(
            self.rebuild_count
        )

        self._h_candidate_cache = (
            cache
        )

        # Directed row/slot representatives can change on neighbour rebuild,
        # even if the chemical pair graph happens to look identical.
        self._h_last_active_signature = None
        self._h_last_topology_metadata = None

        return cache

    # ------------------------------------------------------------------
    # Build component metadata only on a topology change
    # ------------------------------------------------------------------

    def _build_h_topology_metadata(
        self,
        candidate_cache,
        active_numpy,
    ):
        grouped_python = defaultdict(
            list
        )

        component_counts = []
        largest_component_edges = 0
        component_total = 0

        pairs = candidate_cache[
            "pairs"
        ]

        rows = candidate_cache[
            "rows"
        ]

        slots = candidate_cache[
            "slots"
        ]

        hydrogen = int(
            R.ELEMENT_INDEX["H"]
        )

        for box, (
            begin,
            end,
        ) in enumerate(
            candidate_cache[
                "box_ranges"
            ]
        ):
            active_indices = [
                index
                for index
                in range(
                    begin,
                    end,
                )
                if bool(
                    active_numpy[
                        index
                    ]
                )
            ]

            if not active_indices:
                component_counts.append(
                    0
                )
                continue

            edge_atoms = tuple(
                pairs[
                    index
                ]
                for index in active_indices
            )

            edge_rows = tuple(
                rows[
                    index
                ]
                for index in active_indices
            )

            edge_slots = tuple(
                slots[
                    index
                ]
                for index in active_indices
            )

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
                        local_index
                    ]
                    for local_index
                    in component
                )

                component_edge_rows = tuple(
                    int(
                        edge_rows[
                            local_index
                        ]
                    )
                    for local_index
                    in component
                )

                component_edge_slots = tuple(
                    int(
                        edge_slots[
                            local_index
                        ]
                    )
                    for local_index
                    in component
                )

                component_hydrogens = tuple(
                    sorted({
                        atom
                        for first, second
                        in component_edge_atoms
                        for atom in (
                            first,
                            second,
                        )
                        if int(
                            self.types_numpy[
                                atom
                            ]
                        )
                        == hydrogen
                    })
                )

                if not component_hydrogens:
                    raise RuntimeError(
                        "cached H component contains no hydrogen"
                    )

                signature = (
                    self._factorised_component_signature(
                        component_edge_atoms
                    )
                )

                grouped_python[
                    signature
                ].append({
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

        grouped = []

        largest_group = 0

        for signature, entries in (
            grouped_python.items()
        ):
            structure = (
                self._factorised_structure_for_signature(
                    signature
                )
            )

            largest_group = max(
                largest_group,
                len(
                    entries
                ),
            )

            row_index = torch.tensor(
                [
                    entry[
                        "edge_rows"
                    ]
                    for entry in entries
                ],
                device=self.device,
                dtype=torch.long,
            )

            slot_index = torch.tensor(
                [
                    entry[
                        "edge_slots"
                    ]
                    for entry in entries
                ],
                device=self.device,
                dtype=torch.long,
            )

            hydrogen_counts = {
                len(
                    entry[
                        "hydrogens"
                    ]
                )
                for entry in entries
            }

            if len(
                hydrogen_counts
            ) != 1:
                raise RuntimeError(
                    "same H topology signature produced unequal "
                    "hydrogen counts"
                )

            hydrogen_index = torch.tensor(
                [
                    entry[
                        "hydrogens"
                    ]
                    for entry in entries
                ],
                device=self.device,
                dtype=torch.long,
            )

            anchor_index = torch.tensor(
                [
                    entry[
                        "anchor"
                    ]
                    for entry in entries
                ],
                device=self.device,
                dtype=torch.long,
            )

            grouped.append({
                "signature": signature,
                "structure": (
                    structure
                ),
                "row_index": (
                    row_index
                ),
                "slot_index": (
                    slot_index
                ),
                "hydrogen_index": (
                    hydrogen_index
                ),
                "anchor_index": (
                    anchor_index
                ),
                "component_count": int(
                    len(
                        entries
                    )
                ),
            })

        return {
            "groups": tuple(
                grouped
            ),
            "diagnostics": {
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
                "candidate_h_pairs": int(
                    candidate_cache[
                        "candidate_count"
                    ]
                ),
            },
        }

    def _cached_h_topology(
        self,
        values,
    ):
        candidate_cache = (
            self._prepare_h_candidate_cache()
        )

        candidate_count = (
            candidate_cache[
                "candidate_count"
            ]
        )

        if candidate_count == 0:
            active_numpy = np.zeros(
                (0,),
                dtype=bool,
            )
        else:
            active = (
                values[
                    "taper"
                ][
                    candidate_cache[
                        "row_tensor"
                    ],
                    candidate_cache[
                        "slot_tensor"
                    ],
                ]
                > 1e-12
            )

            # Only this compact boolean vector crosses the device boundary.
            # No full neighbour or taper table transfer occurs here.
            active_numpy = (
                active
                .detach()
                .cpu()
                .numpy()
            )

        # Include rebuild_count because representative directed entries can
        # change after a neighbour rebuild.
        signature = (
            int(
                self.rebuild_count
            ),
            active_numpy.tobytes(),
        )

        if (
            self._h_last_active_signature
            == signature
            and self._h_last_topology_metadata
            is not None
        ):
            self._h_topology_cache_hits += 1

            return (
                self._h_last_topology_metadata
            )

        self._h_topology_cache_misses += 1

        metadata = (
            self._build_h_topology_metadata(
                candidate_cache,
                active_numpy,
            )
        )

        self._h_last_active_signature = (
            signature
        )

        self._h_last_topology_metadata = (
            metadata
        )

        return metadata

    # ------------------------------------------------------------------
    # Same factorisable Hamiltonian, using prebuilt index tensors
    # ------------------------------------------------------------------

    def _cached_factorised_group_state_energies(
        self,
        group,
        values,
    ):
        structure = group[
            "structure"
        ]

        row_index = group[
            "row_index"
        ]

        slot_index = group[
            "slot_index"
        ]

        component_count = int(
            group[
                "component_count"
            ]
        )

        group_pressure = _record_h_state_group_pressure(
            self,
            structure,
            component_count,
        )

        edge_count = int(
            structure[
                "edge_count"
            ]
        )

        if edge_count == 0:
            _record_h_state_cuda_memory(
                self,
                group_pressure,
            )
            return (
                values[
                    "taper"
                ].sum()
                * 0.0
                * torch.ones(
                    component_count,
                    device=self.device,
                    dtype=self.dtype,
                )
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

        repulsive = values[
            "repulsive"
        ][
            row_index,
            slot_index,
        ]

        shared_attractive = values.get(
            "state_attractive"
        )

        if shared_attractive is None:
            # Defensive fallback for direct research/test calls that bypass
            # _hydrogen_state_correction(). Normal optimised-valence execution
            # populates this once before any group solve.
            shared_attractive = (
                2.0
                * values[
                    "pair_depth"
                ]
                * torch.exp(
                    -values[
                        "pair_width"
                    ]
                    * values[
                        "shift"
                    ]
                )
            )

            values[
                "state_attractive"
            ] = shared_attractive

        attractive = shared_attractive[
            row_index,
            slot_index,
        ]

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
            int(
                structure[
                    "state_count"
                ]
            )
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
                    int(
                        structure[
                            "degree_key_count"
                        ]
                    ),
                ),
                device=self.device,
                dtype=self.dtype,
            )

            first_degree_key = (
                structure[
                    "first_degree_key"
                ]
            )

            second_degree_key = (
                structure[
                    "second_degree_key"
                ]
            )

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

            if self.h_transition_assembly == "dense":
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
        eigenvalues = (
            _bounded_factorised_eigvalsh(
                hamiltonian,
                _resolved_h_s2_eigvalsh_chunk_size(self),
            )
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
    # H correction with cached discrete topology
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
                "cached H topology requires current reactive intermediates"
            )

        values = cached[1]

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

        # This exact Morse attractive magnitude is consumed by the H-state
        # correction, every grouped H Hamiltonian, and then the heavy-valence
        # membership layer later in the same energy evaluation.  Compute it
        # once and keep the live autograd tensor in the per-evaluation
        # intermediates instead of rebuilding identical exp() branches.
        attractive = values.get(
            "state_attractive"
        )

        if attractive is None:
            attractive = (
                2.0
                * pair_depth
                * torch.exp(
                    -pair_width
                    * shift
                )
            )

            values[
                "state_attractive"
            ] = attractive

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

        if getattr(self, "_h_state_profile_sink", None) is not None:
            self._h_state_profile_context = {
                "candidate_h_pairs": int(
                    self._prepare_h_candidate_cache()["candidate_count"]
                )
            }

        preparation_token = _profile_stage_begin(
            self,
            "H group preparation",
        )
        topology = (
            self._cached_h_topology(
                values
            )
        )
        if getattr(self, "_h_state_profile_sink", None) is not None:
            self._h_state_profile_context = dict(
                topology["diagnostics"]
            )
        _profile_stage_end(
            self,
            "H group preparation",
            preparation_token,
        )

        _begin_h_state_evaluation(
            self,
            len(topology["groups"]),
        )

        correction = (
            torch.zeros_like(
                base_per_atom
            )
            + base_per_atom.sum()
            * 0.0
        )

        for group in topology[
            "groups"
        ]:
            state_energies = (
                self._cached_factorised_group_state_energies(
                    group,
                    values,
                )
            )

            row_index = group[
                "row_index"
            ]

            slot_index = group[
                "slot_index"
            ]

            base_h_pair = torch.sum(
                pair_morse[
                    row_index,
                    slot_index,
                ],
                dim=1,
            )

            base_h_over = torch.sum(
                base_over[
                    group[
                        "hydrogen_index"
                    ]
                ],
                dim=1,
            )

            delta = (
                state_energies
                - base_h_pair
                - base_h_over
            )

            correction = (
                correction.scatter_add(
                    0,
                    group[
                        "anchor_index"
                    ],
                    delta,
                )
            )

        diagnostics = dict(
            topology[
                "diagnostics"
            ]
        )

        diagnostics.update({
            "topology_cache_hits": int(
                self._h_topology_cache_hits
            ),
            "topology_cache_misses": int(
                self._h_topology_cache_misses
            ),
            "candidate_rebuild_count": int(
                self._h_candidate_rebuild_count
                if self._h_candidate_rebuild_count
                is not None
                else -1
            ),
        })

        self._h_component_diagnostics = (
            diagnostics
        )

        _finish_h_state_evaluation(
            self
        )

        return correction


CachedTopologyValenceStateBatchedSimulation = (
    CachedHFastValenceStateBatchedSimulation
)
