"""Boundary for the existing unified-capacity correction."""

from __future__ import annotations

from dataclasses import dataclass

from ..context import InteractionContext


@dataclass(frozen=True)
class UnifiedCapacityEnergy:
    state_solver: object

    def energy(self, context: InteractionContext, base_energy):
        return self.state_solver.solve_energy(context, base_energy)

    def state(self):
        return self.state_solver.state()
