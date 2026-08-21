"""Construction of unified_radial_v1 from unchanged legacy operations."""

from __future__ import annotations

from .engine import ChemistryEngine
from .hamiltonian import UnifiedRadialHamiltonian
from .registry import register
from .state.reference_solver import ExistingUnifiedCapacityReferenceSolver
from .terms.geometry import ExistingGeometryCorrection
from .terms.pair_surfaces import ExistingReactiveBaseEnergy
from .terms.unified_capacity import UnifiedCapacityEnergy
from .terms.registry import build_extensions


def build_unified_radial_engine(simulation, physics_spec):
    state_solver = ExistingUnifiedCapacityReferenceSolver(simulation)
    hamiltonian = UnifiedRadialHamiltonian(
        base_energy=ExistingReactiveBaseEnergy(simulation),
        capacity_energy=UnifiedCapacityEnergy(state_solver),
        geometry_energy=ExistingGeometryCorrection(simulation),
        extensions=build_extensions(
            physics_spec.enabled_extensions
        ),
    )
    return ChemistryEngine(physics=physics_spec, hamiltonian=hamiltonian)


register("unified_radial_v1", build_unified_radial_engine)
