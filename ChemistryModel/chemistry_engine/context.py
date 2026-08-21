"""Data-only input passed from a simulation runtime to a Hamiltonian."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class InteractionContext:
    positions: Any
    element_types: Any
    atomic_numbers: tuple[int, ...]
    neighbours: Any
    neighbour_mask: Any
    box_size: float
    box_count: int
    atoms_per_box: int
    batch_assignment: tuple[int, ...]
    tensors: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self):
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))
