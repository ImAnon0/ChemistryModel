"""Canonical ChemistryModel engine interfaces.

Stage 2B deliberately keeps the validated equations in their existing source
locations.  This package establishes the boundary through which those exact
operations are composed.
"""

from .config import CapacitySpec, ExecutionConfig, GeometrySpec, PhysicsSpec
from .context import InteractionContext
from .engine import ChemistryEngine
from .results import EnergyResult

__all__ = [
    "CapacitySpec",
    "ChemistryEngine",
    "EnergyResult",
    "ExecutionConfig",
    "GeometrySpec",
    "InteractionContext",
    "PhysicsSpec",
]
