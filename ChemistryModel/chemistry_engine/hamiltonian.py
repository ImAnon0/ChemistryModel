"""Hamiltonian interfaces; no production physics is defined here yet."""

from __future__ import annotations

from typing import Protocol

from .context import InteractionContext
from .results import EnergyResult


class Hamiltonian(Protocol):
    def energy(self, context: InteractionContext) -> EnergyResult:
        ...


class UnifiedRadialHamiltonian:
    """Exact-order composition of the frozen unified radial formulation."""

    def __init__(self, base_energy, capacity_energy, geometry_energy):
        self.base_energy = base_energy
        self.capacity_energy = capacity_energy
        self.geometry_energy = geometry_energy

    def energy(self, context: InteractionContext) -> EnergyResult:
        # Do not rearrange this sequence. It is the Stage 2A reference order.
        base = self.base_energy.energy(context)
        try:
            capacity_correction = self.capacity_energy.energy(context, base)
            topology_correction = self.geometry_energy.energy(context)
        finally:
            self.base_energy.release_intermediates()
        total = base + capacity_correction + topology_correction
        components = {
            "base": base,
            **self.base_energy.components(),
            "capacity_correction": capacity_correction,
            "topology_correction": topology_correction,
            "total": total,
        }
        return EnergyResult(
            per_atom=total,
            components=components,
            state=self.capacity_energy.state(),
        )
