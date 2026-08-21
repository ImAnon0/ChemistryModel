"""Neighbour construction is an execution concern, not a Hamiltonian term."""

from __future__ import annotations

from typing import Protocol


class NeighbourBackend(Protocol):
    def build(self, positions, box_information):
        ...
