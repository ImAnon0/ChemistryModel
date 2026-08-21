"""Interface for the established geometry/topology correction."""

from __future__ import annotations

from typing import Protocol

from ..context import InteractionContext


class GeometryEnergy(Protocol):
    def energy(self, context: InteractionContext):
        ...


class ExistingGeometryCorrection:
    """Adapter to the established topology/geometry correction unchanged."""

    def __init__(self, simulation):
        self.simulation = simulation

    def energy(self, context: InteractionContext):
        return self.simulation._valence_topology_correction(context.positions)
