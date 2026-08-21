from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from chemistry_engine.backends.torch_backend import interaction_context
from chemistry_engine.hamiltonian import UnifiedRadialHamiltonian
from chemistry_engine.registry import registered_models
from research.unified_bond_capacity import UnifiedBondCapacityEnergyPrototype
from unified_radial_equivalence import build_simulation, load_fixture


def _case(name="h2_equilibrium"):
    return next(case for case in load_fixture()["cases"] if case["name"] == name)


def _simulation(dtype=torch.float64):
    return build_simulation(
        UnifiedBondCapacityEnergyPrototype,
        [_case()],
        device="cpu",
        dtype=dtype,
        box_size=40.0,
        capture_equivalence_state=True,
    )


def test_unified_radial_is_registered_with_canonical_hamiltonian():
    simulation = _simulation()
    assert "unified_radial_v1" in registered_models()
    assert simulation.chemistry_engine.physics.model_id == "unified_radial_v1"
    assert isinstance(
        simulation.chemistry_engine.hamiltonian, UnifiedRadialHamiltonian
    )
    assert simulation.use_canonical_engine is True


def test_physics_and_execution_configuration_are_separate_and_immutable():
    float64 = _simulation(torch.float64)
    float32 = _simulation(torch.float32)
    physics = float64.chemistry_physics_spec
    assert physics.parameter_sha256 == float32.chemistry_physics_spec.parameter_sha256
    assert physics.capacity.solver == "existing_scipy_l_bfgs_b_dual"
    assert physics.geometry.convention == "established_heavy_angle_topology_v1"
    assert float64.chemistry_execution_config.dtype == "torch.float64"
    assert float32.chemistry_execution_config.dtype == "torch.float32"
    with pytest.raises(FrozenInstanceError):
        physics.model_id = "changed"


def test_interaction_context_contains_runtime_data_without_owning_runtime_steps():
    simulation = _simulation()
    context = interaction_context(simulation, simulation.positions)
    assert context.positions is simulation.positions
    assert context.element_types is simulation.types
    assert context.atomic_numbers == (1, 1)
    assert context.box_count == 1
    assert context.atoms_per_box == 2
    assert context.batch_assignment == (0, 0)
    assert context.neighbours is simulation.neighbours
    assert context.neighbour_mask is simulation.neighbour_mask
    assert not hasattr(context, "step")


def test_canonical_result_exposes_exact_components_and_state():
    simulation = _simulation()
    result = simulation._last_chemistry_result
    assert tuple(result.components) == (
        "base",
        "base_bond",
        "base_overcoordination",
        "base_angle",
        "capacity_correction",
        "topology_correction",
        "total",
    )
    assert torch.equal(result.per_atom, result.components["total"])
    assert result.state["membership"] is simulation._unified_membership
    assert result.state["diagnostics"] is simulation._unified_diagnostics
