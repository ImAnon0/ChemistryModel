"""Canonical owner of a selected ChemistryModel Hamiltonian."""

from __future__ import annotations

from dataclasses import dataclass

from .config import PhysicsSpec
from .context import InteractionContext
from .hamiltonian import Hamiltonian
from .results import EnergyResult


@dataclass(frozen=True)
class ChemistryEngine:
    physics: PhysicsSpec
    hamiltonian: Hamiltonian

    def energy(self, context: InteractionContext) -> EnergyResult:
        return self.hamiltonian.energy(context)
