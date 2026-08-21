from __future__ import annotations

from typing import Protocol

from ..context import InteractionContext


class EnergyTerm(Protocol):
    """Optional Hamiltonian contribution boundary."""

    name: str

    def energy(self, context: InteractionContext, current_energy):
        ...
