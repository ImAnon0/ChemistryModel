"""
Experimental heavy-valence engine using the factorisable local H-state.

Architecture
------------
    reactive_torch.py
        ↓
    factorisable local H-state
        ↓
    existing heavy valence-state topology

This module intentionally reuses the validated heavy-topology implementation
from valence_state_torch.py without changing its equations or parameters.

Only the hydrogen-state machinery is replaced:

    historical whole-box H-state
        -> local H-competition components
        -> all H-valence-valid matchings
        -> crowding normalisation local to the transferred H

The factorisable H-state has already been checked for:
    - exact preservation of the existing 106 single-reaction microscope
    - exact size consistency for disconnected H-transfer networks
    - smooth static component merge/split
    - live 2->1 component merge under NVE
    - exact historical-vs-factorisable single-component trajectories

This file is still EXPERIMENTAL.  It exists so the corrected H-state can be
validated underneath the already-validated heavy-valence layer before any
performance optimisation or promotion.

No heavy-valence chemistry parameter is added or changed.
"""

from __future__ import annotations

from valence_state_torch import (
    ValenceStateBatchedSimulation,
    VALENCE_STATE_MODEL_REVISION,
)

from h_state_component_torch import (
    HStateComponentBatchedSimulation,
)

from h_state_factorised_torch import (
    FactorisedHStateBatchedSimulation,
)


FACTORISABLE_VALENCE_STATE_MODEL_NAME = (
    "reactive_v4_factorisable_h_valence_state_experimental"
)

FACTORISABLE_VALENCE_STATE_MODEL_REVISION = (
    VALENCE_STATE_MODEL_REVISION
)


class FactorisableValenceStateBatchedSimulation(
    ValenceStateBatchedSimulation
):
    """
    Existing heavy-valence topology with the corrected factorisable H-state.

    valence_state_torch.ValenceStateBatchedSimulation already composes energy
    as:

        base
        + self._hydrogen_state_correction(...)
        + self._valence_topology_correction(...)

    so the safest integration is to inherit its heavy-topology implementation
    untouched and replace only the H-state methods.

    The component correction calls self._box_state_energy(), so aliasing the
    factorisable version here automatically uses the new all-valid-state /
    per-H-crowding Hamiltonian inside every local H component.
    """

    physics_model_name = FACTORISABLE_VALENCE_STATE_MODEL_NAME
    physics_model_revision = FACTORISABLE_VALENCE_STATE_MODEL_REVISION

    # Local H-valence component construction and correction bookkeeping.
    _hydrogen_edge_components = (
        HStateComponentBatchedSimulation._hydrogen_edge_components
    )

    _hydrogen_state_correction = (
        HStateComponentBatchedSimulation._hydrogen_state_correction
    )

    # Factorisable local H-state Hamiltonian.
    _box_state_energy = (
        FactorisedHStateBatchedSimulation._box_state_energy
    )


# Naming aliases used by other engine modules / experiments.
FactorisedValenceStateBatchedSimulation = (
    FactorisableValenceStateBatchedSimulation
)

FactorisableValenceStateSimulation = (
    FactorisableValenceStateBatchedSimulation
)
