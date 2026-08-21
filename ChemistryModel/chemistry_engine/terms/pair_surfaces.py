"""Interface for the complete established reactive base surface."""

from __future__ import annotations

from typing import Protocol

from ..context import InteractionContext


class PairSurfaceEnergy(Protocol):
    def energy(self, context: InteractionContext):
        ...

    def release_intermediates(self):
        ...


class ExistingReactiveBaseEnergy:
    """Adapter to the exact base call used by unified radial v1."""

    def __init__(self, simulation):
        self.simulation = simulation

    def energy(self, context: InteractionContext):
        from batched_torch import BatchedReactiveSimulation

        return BatchedReactiveSimulation.energy_per_atom(
            self.simulation, context.positions
        )

    def components(self):
        parts = self.simulation._profile_energy_parts
        return {
            "base_bond": parts["bond"],
            "base_overcoordination": parts["over"],
            "base_angle": parts["angle"],
        }

    def release_intermediates(self):
        self.simulation._reactive_intermediates = None
