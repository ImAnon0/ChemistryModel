"""Protocol for the unchanged reference state solver."""

from __future__ import annotations

from typing import Protocol

from ..context import InteractionContext


class ReferenceStateSolver(Protocol):
    def solve_energy(self, context: InteractionContext, base_energy):
        ...


class ExistingUnifiedCapacityReferenceSolver:
    """Delegates to the frozen SciPy dual solve and state construction."""

    def __init__(self, simulation):
        self.simulation = simulation

    def solve_energy(self, context: InteractionContext, base_energy):
        return self.simulation._unified_capacity_correction(
            context.positions, base_energy
        )

    def state(self):
        return {
            "membership": self.simulation._unified_membership,
            "diagnostics": self.simulation._unified_diagnostics,
        }
