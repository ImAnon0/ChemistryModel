"""
Final execution-optimised valence-state candidate.

Physics stack:
    reactive base
      -> factorisable/local H-state
      -> grouped H Hamiltonians
      -> batched heavy-valence membership
      -> cached discrete H topology
      -> device-aware neighbour gather backend

The only change introduced in THIS module is execution selection for the
existing reactive_torch._gather_neighbours() backend:

    CPU  -> advanced indexing (values[neighbours])
    CUDA -> torch.index_select(...)

Benchmarks on the validated 8 x 330 workload showed:
    - CPU: advanced indexing is faster
    - CUDA: index_select for all gather roles is faster

The gather backend is selected before parent construction because initial
forces are evaluated during ReactiveSimulation.__init__.

No chemistry equation, parameter, topology definition, or force derivative is
changed.
"""

from __future__ import annotations

import torch

from valence_state_cached_h_topology_torch import (
    CachedHFastValenceStateBatchedSimulation,
)


OPTIMISED_VALENCE_MODEL_NAME = (
    "reactive_v7_factorisable_valence_optimised_experimental"
)

OPTIMISED_VALENCE_MODEL_REVISION = 1


def _resolved_device_type(requested_device):
    if requested_device is None:
        return (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(
        requested_device
    ).type


class OptimisedValenceStateBatchedSimulation(
    CachedHFastValenceStateBatchedSimulation
):
    """
    Current validated valence-state engine with device-aware gather execution.
    """

    physics_model_name = (
        OPTIMISED_VALENCE_MODEL_NAME
    )

    physics_model_revision = (
        OPTIMISED_VALENCE_MODEL_REVISION
    )

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        requested_device = kwargs.get(
            "device",
            None,
        )

        resolved_type = (
            _resolved_device_type(
                requested_device
            )
        )

        # reactive_torch._gather_neighbours() reads this attribute.
        #
        # True means use index_select for every gather role.
        # False means preserve the existing advanced-indexing backend.
        #
        # This must be assigned before super().__init__ because the base
        # constructor computes initial forces.
        self.experimental_index_select_gather = (
            resolved_type
            == "cuda"
        )

        super().__init__(
            *args,
            **kwargs,
        )

        # Diagnostic only; not used by the equations.
        self.selected_neighbour_gather_backend = (
            "index_select"
            if self.experimental_index_select_gather
            else "advanced_indexing"
        )


# Concise aliases for production wiring experiments.
FinalValenceStateBatchedSimulation = (
    OptimisedValenceStateBatchedSimulation
)

OptimizedValenceStateBatchedSimulation = (
    OptimisedValenceStateBatchedSimulation
)
