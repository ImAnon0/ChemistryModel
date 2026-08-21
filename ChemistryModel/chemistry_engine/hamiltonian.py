"""Hamiltonian interfaces; no production physics is defined here yet."""

from __future__ import annotations

from typing import Protocol

from .context import InteractionContext
from .results import EnergyResult


class Hamiltonian(Protocol):
    def energy(self, context: InteractionContext) -> EnergyResult:
        ...


class UnifiedRadialHamiltonian:
    """Exact-order composition of the frozen unified radial formulation.

    Optional extension terms are evaluated only after the frozen reference
    energy composition. With no extensions selected this path is unchanged.
    """

    def __init__(
        self,
        base_energy,
        capacity_energy,
        geometry_energy,
        extensions=(),
    ):
        self.base_energy = base_energy
        self.capacity_energy = capacity_energy
        self.geometry_energy = geometry_energy
        self.extensions = tuple(extensions)

    def energy(self, context: InteractionContext) -> EnergyResult:
        # Do not rearrange this sequence. It is the Stage 2A reference order.
        base = self.base_energy.energy(context)
        try:
            capacity_correction = self.capacity_energy.energy(context, base)
            topology_correction = self.geometry_energy.energy(context)

            total = base + capacity_correction + topology_correction

            extension_components = {}
            for term in self.extensions:
                contribution = term.energy(context, total)
                extension_components[term.name] = contribution
                total = total + contribution

        finally:
            self.base_energy.release_intermediates()

        components = {
            "base": base,
            **self.base_energy.components(),
            "capacity_correction": capacity_correction,
            "topology_correction": topology_correction,
            **extension_components,
            "total": total,
        }

        return EnergyResult(
            per_atom=total,
            components=components,
            state=self.capacity_energy.state(),
        )
