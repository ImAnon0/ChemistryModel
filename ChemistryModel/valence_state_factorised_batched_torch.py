"""
Heavy-valence candidate using grouped execution of the validated
factorisable local H-state.

Heavy-valence equations are inherited unchanged from
valence_state_factorised_torch.py.  Only execution of the H-state correction
is replaced by the topology-grouped implementation.
"""

from __future__ import annotations

from valence_state_factorised_torch import (
    FactorisableValenceStateBatchedSimulation,
    FACTORISABLE_VALENCE_STATE_MODEL_REVISION,
)
from h_state_factorised_batched_torch import (
    GroupedFactorisedHStateBatchedSimulation,
)


GROUPED_FACTORISABLE_VALENCE_MODEL_NAME = (
    "reactive_v4_factorisable_h_valence_state_grouped_execution_experimental"
)


class GroupedFactorisableValenceStateBatchedSimulation(
    FactorisableValenceStateBatchedSimulation
):
    physics_model_name = (
        GROUPED_FACTORISABLE_VALENCE_MODEL_NAME
    )

    physics_model_revision = (
        FACTORISABLE_VALENCE_STATE_MODEL_REVISION
    )

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self._factorised_h_structure_cache = {}

        super().__init__(
            *args,
            **kwargs,
        )

    _factorised_component_signature = (
        GroupedFactorisedHStateBatchedSimulation
        ._factorised_component_signature
    )

    _factorised_structure_for_signature = (
        GroupedFactorisedHStateBatchedSimulation
        ._factorised_structure_for_signature
    )

    _factorised_group_state_energies = (
        GroupedFactorisedHStateBatchedSimulation
        ._factorised_group_state_energies
    )

    _hydrogen_state_correction = (
        GroupedFactorisedHStateBatchedSimulation
        ._hydrogen_state_correction
    )


BatchedFactorisableValenceStateBatchedSimulation = (
    GroupedFactorisableValenceStateBatchedSimulation
)
