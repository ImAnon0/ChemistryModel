"""
Batched heavy-valence membership execution.

Physics is unchanged from valence_state_torch.py.

The validated reference implementation evaluates competing heavy centres one
atom at a time:
    active slots -> enumerate C(N,V) states -> build Hamiltonian -> eigh
and repeats the same discrete state graph for every equivalent centre.

This execution layer groups competing heavy centres by:
    (N active candidate contacts, V elemental valence capacity)

For a fixed (N,V), the state basis, state/contact membership matrix, and
one-exchange transition graph are identical.  Only the live taper/depth/shift
values differ.  Those live tensors remain in Torch/autograd.

One batched matrix exponential is therefore used per (N,V) group.  Heavy
state membership is the diagonal of a finite-temperature density matrix,
which is basis independent and remains differentiable at degeneracy.

The 10 meV density-matrix temperature is the only intentional physics
parameter introduced by this revision; base and hydrogen physics are unchanged.
"""

from __future__ import annotations

from collections import defaultdict
import itertools
import math

import numpy as np
import torch

import reactive as R

from h_state_torch import (
    _contact_overlap,
    _crowding_normalisation,
)
from heavy_valence_density import (
    DEFAULT_HEAVY_VALENCE_TEMPERATURE,
    thermal_state_probabilities,
)
from valence_state_torch import (
    MAX_LOCAL_CANDIDATES,
    MAX_LOCAL_STATES,
)
from valence_state_factorised_batched_torch import (
    GroupedFactorisableValenceStateBatchedSimulation,
)


BATCHED_HEAVY_VALENCE_MODEL_NAME = (
    "reactive_v5_factorisable_h_grouped_heavy_valence_experimental"
)

BATCHED_HEAVY_VALENCE_MODEL_REVISION = 1

class BatchedHeavyValenceStateBatchedSimulation(
    GroupedFactorisableValenceStateBatchedSimulation
):
    """
    Grouped factorisable H-state plus grouped heavy-valence membership.
    """

    physics_model_name = (
        BATCHED_HEAVY_VALENCE_MODEL_NAME
    )

    physics_model_revision = (
        BATCHED_HEAVY_VALENCE_MODEL_REVISION
    )

    def __init__(
        self,
        *args,
        heavy_valence_temperature=DEFAULT_HEAVY_VALENCE_TEMPERATURE,
        **kwargs,
    ):
        # Must exist before parent construction because the parent can evaluate
        # initial forces and dispatch to this override during __init__.
        self._heavy_valence_structure_cache = {}
        self._heavy_valence_capacity_numpy = None
        self._heavy_valence_diagnostics = {}
        self.heavy_valence_temperature = float(
            heavy_valence_temperature
        )
        if self.heavy_valence_temperature <= 0.0:
            raise ValueError(
                "heavy_valence_temperature must be positive"
            )

        # Cumulative state-pressure diagnostics for a whole simulation/group.
        # Observational only: solver decisions and forces do not depend on it.
        self._heavy_valence_run_diagnostics = {
            "evaluation_count": 0,
            "max_candidate_count": 0,
            "max_state_count": 0,
            "centre_evaluations_over_128": 0,
            "centre_evaluations_over_200": 0,
            "evaluations_with_over_128": 0,
            "evaluations_with_over_200": 0,
            "max_topology_group": 0,
            "max_total_states_solved": 0,
            "max_competitive_atom_count": 0,
            "max_state_shape": (),
        }

        super().__init__(
            *args,
            **kwargs,
        )

    def _record_heavy_valence_run_diagnostics(
        self,
        current,
    ):
        # Accumulate cheap Python-side state-space pressure counters.

        run = self._heavy_valence_run_diagnostics

        run["evaluation_count"] += 1

        run["max_candidate_count"] = max(
            int(run["max_candidate_count"]),
            int(current.get("largest_candidate_count", 0)),
        )

        current_state_count = int(
            current.get("largest_state_count", 0)
        )

        if current_state_count > int(run["max_state_count"]):
            run["max_state_count"] = current_state_count
            run["max_state_shape"] = tuple(
                int(value)
                for value in current.get(
                    "largest_state_shape",
                    (),
                )
            )

        over_128 = int(
            current.get("centres_over_128", 0)
        )
        over_200 = int(
            current.get("centres_over_200", 0)
        )

        run["centre_evaluations_over_128"] += over_128
        run["centre_evaluations_over_200"] += over_200

        if over_128:
            run["evaluations_with_over_128"] += 1

        if over_200:
            run["evaluations_with_over_200"] += 1

        run["max_topology_group"] = max(
            int(run["max_topology_group"]),
            int(current.get("largest_topology_group", 0)),
        )

        run["max_total_states_solved"] = max(
            int(run["max_total_states_solved"]),
            int(current.get("total_states_solved", 0)),
        )

        run["max_competitive_atom_count"] = max(
            int(run["max_competitive_atom_count"]),
            int(current.get("competitive_atom_count", 0)),
        )

    # ------------------------------------------------------------------
    # Discrete (N,V) state graph cache
    # ------------------------------------------------------------------

    def _heavy_capacity_numpy(self):
        cached = (
            self._heavy_valence_capacity_numpy
        )

        if cached is not None:
            return cached

        capacities = (
            self.valence[
                self.types
            ]
            .detach()
            .cpu()
            .numpy()
        )

        cached = np.maximum(
            np.rint(
                capacities
            ).astype(
                np.int64
            ),
            0,
        )

        self._heavy_valence_capacity_numpy = (
            cached
        )

        return cached

    def _heavy_structure(
        self,
        candidate_count,
        capacity,
    ):
        key = (
            int(candidate_count),
            int(capacity),
        )

        cached = (
            self._heavy_valence_structure_cache.get(
                key
            )
        )

        if cached is not None:
            return cached

        candidate_count = int(
            candidate_count
        )

        capacity = int(
            capacity
        )

        if candidate_count > MAX_LOCAL_CANDIDATES:
            raise RuntimeError(
                "valence-state topology has "
                f"{candidate_count} active contacts; "
                f"research limit is {MAX_LOCAL_CANDIDATES}"
            )

        state_count = math.comb(
            candidate_count,
            capacity,
        )

        if state_count > MAX_LOCAL_STATES:
            raise RuntimeError(
                "valence-state topology would require "
                f"{state_count} local states "
                f"({candidate_count} candidates, valence {capacity}); "
                f"research limit is {MAX_LOCAL_STATES}"
            )

        states = tuple(
            itertools.combinations(
                range(
                    candidate_count
                ),
                capacity,
            )
        )

        state_count = len(
            states
        )

        state_mask = torch.zeros(
            (
                state_count,
                candidate_count,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        for state_index, state in enumerate(
            states
        ):
            if state:
                state_mask[
                    state_index,
                    torch.tensor(
                        state,
                        device=self.device,
                        dtype=torch.long,
                    ),
                ] = 1.0

        transition_first = []
        transition_second = []
        transition_old = []
        transition_new = []

        for first in range(
            state_count
        ):
            first_set = set(
                states[first]
            )

            for second in range(
                first + 1,
                state_count,
            ):
                second_set = set(
                    states[second]
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

                transition_first.append(
                    first
                )
                transition_second.append(
                    second
                )
                transition_old.append(
                    removed[0]
                )
                transition_new.append(
                    added[0]
                )

        transition_first_tensor = torch.tensor(
            transition_first,
            device=self.device,
            dtype=torch.long,
        )

        transition_second_tensor = torch.tensor(
            transition_second,
            device=self.device,
            dtype=torch.long,
        )

        transition_old_tensor = torch.tensor(
            transition_old,
            device=self.device,
            dtype=torch.long,
        )

        transition_new_tensor = torch.tensor(
            transition_new,
            device=self.device,
            dtype=torch.long,
        )

        transition_count = len(
            transition_first
        )

        # Compact representation of symmetric off-diagonal placements.
        # The old T x S x S transition basis exploded in memory for larger
        # exact local state spaces.
        if transition_count:
            transition_flat_index = torch.cat(
                (
                    transition_first_tensor * state_count
                    + transition_second_tensor,
                    transition_second_tensor * state_count
                    + transition_first_tensor,
                )
            )
        else:
            transition_flat_index = torch.zeros(
                (0,),
                device=self.device,
                dtype=torch.long,
            )

        cached = {
            "candidate_count": (
                candidate_count
            ),
            "capacity": capacity,
            "states": states,
            "state_count": (
                state_count
            ),
            "state_mask": (
                state_mask
            ),
            "transition_first": (
                transition_first_tensor
            ),
            "transition_second": (
                transition_second_tensor
            ),
            "transition_old": (
                transition_old_tensor
            ),
            "transition_new": (
                transition_new_tensor
            ),
            "transition_flat_index": (
                transition_flat_index
            ),
        }

        self._heavy_valence_structure_cache[
            key
        ] = cached

        return cached

    # ------------------------------------------------------------------
    # Compact differentiable Hamiltonian assembly
    # ------------------------------------------------------------------

    def _assemble_heavy_hamiltonian(
        self,
        diagonal,
        coupling,
        structure,
    ):
        # Same symmetric Hamiltonian as the former transition-basis einsum,
        # assembled from compact flat transition indices.

        hamiltonian = torch.diag_embed(
            diagonal
        )

        transition_count = int(
            structure[
                "transition_first"
            ].numel()
        )

        if transition_count == 0:
            return hamiltonian

        state_count = int(
            structure[
                "state_count"
            ]
        )

        group_size = int(
            diagonal.shape[0]
        )

        flat_index = structure[
            "transition_flat_index"
        ]

        flat_values = torch.cat(
            (
                -coupling,
                -coupling,
            ),
            dim=1,
        )

        off_diagonal = torch.zeros(
            (
                group_size,
                state_count * state_count,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        off_diagonal = off_diagonal.scatter_add(
            1,
            flat_index[
                None,
                :,
            ].expand(
                group_size,
                -1,
            ),
            flat_values,
        )

        return (
            hamiltonian
            + off_diagonal.reshape(
                group_size,
                state_count,
                state_count,
            )
        )

    # ------------------------------------------------------------------
    # Batched heavy membership
    # ------------------------------------------------------------------

    def _local_valence_membership(
        self,
        values,
    ):
        # Optional profiling sink installed only by the dedicated profiler.
        # Solver decisions and forces never depend on these timings.
        profile_sink = getattr(
            self,
            "_heavy_profile_sink",
            None,
        )

        taper = values[
            "taper"
        ]

        mask = values[
            "mask"
        ]

        pair_depth = values[
            "pair_depth"
        ]

        shared_attractive = values.get(
            "state_attractive"
        )

        if shared_attractive is None:
            # Defensive fallback for direct membership tests. In the real
            # optimised-valence energy path the H-state runs first and
            # populates this live tensor once per evaluation.
            shared_attractive = (
                2.0
                * pair_depth
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

        hydrogen = int(
            R.ELEMENT_INDEX["H"]
        )

        # Preserve the reference active-contact semantics exactly.  The wider
        # neighbour list includes skin entries whose taper is zero, so mask
        # alone must not define the heavy-valence candidate topology.
        # One bulk snapshot avoids one GPU->CPU synchronisation per atom.
        if profile_sink is not None:
            profile_token = profile_sink.stage_begin(
                "heavy.membership.snapshot"
            )

        active_numpy = (
            ((mask > 0.0) & (taper > 0.0))
            .detach()
            .cpu()
            .numpy()
        )

        if profile_sink is not None:
            profile_sink.stage_end(
                "heavy.membership.snapshot",
                profile_token,
            )
            profile_token = profile_sink.stage_begin(
                "heavy.membership.cpu_grouping"
            )

        capacities = (
            self._heavy_capacity_numpy()
        )

        groups = defaultdict(
            list
        )

        zero_rows = []

        competitive_atom_count = 0

        largest_candidate_count = 0
        largest_state_count = 0
        largest_state_shape = ()
        centres_over_128 = 0
        centres_over_200 = 0

        for atom in range(
            taper.shape[0]
        ):
            if (
                int(
                    self.types_numpy[
                        atom
                    ]
                )
                == hydrogen
            ):
                continue

            active_slots = np.flatnonzero(
                active_numpy[
                    atom
                ]
            )

            capacity = int(
                capacities[
                    atom
                ]
            )

            if (
                capacity <= 0
                or active_slots.size == 0
            ):
                # Reference implementation returns row_zero, not mask.
                zero_rows.append(
                    atom
                )
                continue

            candidate_count = int(
                active_slots.size
            )

            largest_candidate_count = max(
                largest_candidate_count,
                candidate_count,
            )

            if (
                candidate_count
                <= capacity
            ):
                # Reference implementation returns mask exactly.
                continue

            if (
                candidate_count
                > MAX_LOCAL_CANDIDATES
            ):
                raise RuntimeError(
                    "valence-state topology has "
                    f"{candidate_count} active contacts around atom {atom}; "
                    f"research limit is {MAX_LOCAL_CANDIDATES}"
                )

            state_count = math.comb(
                candidate_count,
                capacity,
            )

            if state_count > largest_state_count:
                largest_state_count = state_count
                largest_state_shape = (
                    candidate_count,
                    capacity,
                    state_count,
                )

            if state_count > 128:
                centres_over_128 += 1

            if state_count > 200:
                centres_over_200 += 1

            if (
                state_count
                > MAX_LOCAL_STATES
            ):
                raise RuntimeError(
                    "valence-state topology would require "
                    f"{state_count} local states around atom {atom} "
                    f"({candidate_count} candidates, valence {capacity}); "
                    f"research limit is {MAX_LOCAL_STATES}"
                )

            groups[
                (
                    candidate_count,
                    capacity,
                )
            ].append(
                (
                    atom,
                    tuple(
                        int(slot)
                        for slot in active_slots
                    ),
                )
            )

            competitive_atom_count += 1

        if profile_sink is not None:
            profile_sink.stage_end(
                "heavy.membership.cpu_grouping",
                profile_token,
            )

        # Start from the exact reference result for H atoms and heavy centres
        # without competition.
        membership = (
            mask
            + taper.sum()
            * 0.0
        )

        correction = torch.zeros_like(
            membership
        )

        if zero_rows:
            zero_index = torch.tensor(
                zero_rows,
                device=self.device,
                dtype=torch.long,
            )

            correction = (
                correction.index_add(
                    0,
                    zero_index,
                    -mask[
                        zero_index
                    ],
                )
            )

        largest_group = 0
        total_states_solved = 0

        for (
            candidate_count,
            capacity,
        ), entries in groups.items():
            group_size = len(
                entries
            )

            largest_group = max(
                largest_group,
                group_size,
            )

            if profile_sink is not None:
                profile_token = profile_sink.stage_begin(
                    "heavy.membership.group_prepare",
                    centres=group_size,
                )

            structure = (
                self._heavy_structure(
                    candidate_count,
                    capacity,
                )
            )

            total_states_solved += (
                group_size
                * structure[
                    "state_count"
                ]
            )

            atom_index = torch.tensor(
                [
                    atom
                    for atom, _
                    in entries
                ],
                device=self.device,
                dtype=torch.long,
            )

            slot_index = torch.tensor(
                [
                    slots
                    for _, slots
                    in entries
                ],
                device=self.device,
                dtype=torch.long,
            )

            group_taper = taper[
                atom_index[
                    :,
                    None,
                ],
                slot_index,
            ]

            group_depth = pair_depth[
                atom_index[
                    :,
                    None,
                ],
                slot_index,
            ]

            group_attractive_magnitude = shared_attractive[
                atom_index[
                    :,
                    None,
                ],
                slot_index,
            ]

            if profile_sink is not None:
                profile_sink.stage_end(
                    "heavy.membership.group_prepare",
                    profile_token,
                )
                profile_token = profile_sink.stage_begin(
                    "heavy.membership.group_physics",
                    centres=group_size,
                )

            group_attractive = (
                group_taper
                * group_attractive_magnitude
            )

            diagonal = -(
                group_attractive
                @ structure[
                    "state_mask"
                ].T
            )

            if (
                structure[
                    "state_count"
                ]
                == 1
            ):
                probabilities = (
                    torch.ones_like(
                        diagonal
                    )
                )

                if profile_sink is not None:
                    profile_sink.stage_end(
                        "heavy.membership.group_physics",
                        profile_token,
                    )
                    profile_token = profile_sink.stage_begin(
                        "heavy.membership.reconstruct",
                        centres=group_size,
                    )
            else:
                old_index = structure[
                    "transition_old"
                ]

                new_index = structure[
                    "transition_new"
                ]

                overlap = _contact_overlap(
                    group_taper[
                        :,
                        old_index,
                    ],
                    group_taper[
                        :,
                        new_index,
                    ],
                )

                weighted = (
                    overlap
                    * overlap
                )

                degree = torch.zeros(
                    (
                        group_size,
                        structure[
                            "state_count"
                        ],
                    ),
                    device=self.device,
                    dtype=self.dtype,
                )

                transition_first = structure[
                    "transition_first"
                ]

                transition_second = structure[
                    "transition_second"
                ]

                degree = degree.scatter_add(
                    1,
                    transition_first[
                        None,
                        :,
                    ].expand(
                        group_size,
                        -1,
                    ),
                    weighted,
                )

                degree = degree.scatter_add(
                    1,
                    transition_second[
                        None,
                        :,
                    ].expand(
                        group_size,
                        -1,
                    ),
                    weighted,
                )

                normalisation = (
                    _crowding_normalisation(
                        degree
                    )
                )

                depth_scale = torch.sqrt(
                    torch.clamp(
                        group_depth[
                            :,
                            old_index,
                        ]
                        * group_depth[
                            :,
                            new_index,
                        ],
                        min=1e-12,
                    )
                )

                denominator = torch.sqrt(
                    torch.clamp(
                        normalisation[
                            :,
                            transition_first,
                        ]
                        * normalisation[
                            :,
                            transition_second,
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

                hamiltonian = (
                    self._assemble_heavy_hamiltonian(
                        diagonal,
                        coupling,
                        structure,
                    )
                )

                if profile_sink is not None:
                    profile_sink.stage_end(
                        "heavy.membership.group_physics",
                        profile_token,
                    )

                probabilities = (
                    thermal_state_probabilities(
                        hamiltonian,
                        self.heavy_valence_temperature,
                    )
                )

                if profile_sink is not None:
                    profile_token = profile_sink.stage_begin(
                        "heavy.membership.reconstruct",
                        centres=group_size,
                    )

            candidate_membership = (
                probabilities
                @ structure[
                    "state_mask"
                ]
            )

            desired_rows = torch.zeros(
                (
                    group_size,
                    taper.shape[1],
                ),
                device=self.device,
                dtype=self.dtype,
            )

            desired_rows = (
                desired_rows.scatter(
                    1,
                    slot_index,
                    candidate_membership,
                )
            )

            correction = (
                correction.index_add(
                    0,
                    atom_index,
                    desired_rows
                    - mask[
                        atom_index
                    ],
                )
            )

            if profile_sink is not None:
                profile_sink.stage_end(
                    "heavy.membership.reconstruct",
                    profile_token,
                )

        membership = (
            membership
            + correction
        )

        self._heavy_valence_diagnostics = {
            "competitive_atom_count": int(
                competitive_atom_count
            ),
            "topology_group_count": int(
                len(
                    groups
                )
            ),
            "largest_topology_group": int(
                largest_group
            ),
            "zero_row_count": int(
                len(
                    zero_rows
                )
            ),
            "total_states_solved": int(
                total_states_solved
            ),
            "group_shapes": tuple(
                sorted(
                    (
                        int(key[0]),
                        int(key[1]),
                        int(len(entries)),
                    )
                    for key, entries
                    in groups.items()
                )
            ),
            "largest_candidate_count": int(
                largest_candidate_count
            ),
            "largest_state_count": int(
                largest_state_count
            ),
            "largest_state_shape": tuple(
                int(value)
                for value in largest_state_shape
            ),
            "centres_over_128": int(
                centres_over_128
            ),
            "centres_over_200": int(
                centres_over_200
            ),
        }

        self._record_heavy_valence_run_diagnostics(
            self._heavy_valence_diagnostics
        )

        return membership


GroupedHeavyValenceStateBatchedSimulation = (
    BatchedHeavyValenceStateBatchedSimulation
)
