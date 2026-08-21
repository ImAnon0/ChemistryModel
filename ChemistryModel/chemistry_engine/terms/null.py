from __future__ import annotations


class NullEnergyTerm:
    """Explicit zero contribution extension."""

    name = "null"

    def energy(self, context, current_energy):
        return current_energy * 0.0
