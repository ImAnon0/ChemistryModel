"""Canonical energy/state result objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class EnergyResult:
    per_atom: Any
    components: Mapping[str, Any]
    state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(
            self, "components", MappingProxyType(dict(self.components))
        )
        object.__setattr__(self, "state", MappingProxyType(dict(self.state)))
