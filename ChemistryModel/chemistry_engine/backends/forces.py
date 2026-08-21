"""Force differentiation is an execution backend responsibility."""

from __future__ import annotations

from typing import Protocol


class ForceBackend(Protocol):
    def forces(self, energy_per_atom, positions):
        ...
