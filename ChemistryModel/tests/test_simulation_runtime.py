from __future__ import annotations

import numpy as np
import torch

from chemistry_engine.backends.runtime_adapters import (
    ExistingAutogradForceBackend,
    ExistingLangevinThermostat,
    ExistingNeighbourBackend,
    ExistingVelocityVerletIntegrator,
)
from chemistry_engine.runtime import SimulationRuntime
from research.unified_bond_capacity import UnifiedBondCapacityEnergyPrototype


def _simulation(seed=37):
    symbols = ["O", "H", "H"]
    positions = np.asarray([
        [20.0, 20.0, 20.0],
        [20.9572, 20.0, 20.0],
        [19.7600128, 20.9266272, 20.0],
    ])
    return UnifiedBondCapacityEnergyPrototype(
        boxes=[(symbols, positions)],
        box_size=40.0,
        time_step=0.1,
        target_temperature=300.0,
        friction=0.01,
        device="cpu",
        dtype=torch.float64,
        random_seed=seed,
        relax_on_start=False,
    )


def test_runtime_owns_live_state_backends_and_execution_caches():
    simulation = _simulation()
    runtime = simulation.runtime
    assert isinstance(runtime, SimulationRuntime)
    assert isinstance(runtime.integrator, ExistingVelocityVerletIntegrator)
    assert isinstance(runtime.thermostat, ExistingLangevinThermostat)
    assert isinstance(runtime.force_backend, ExistingAutogradForceBackend)
    assert isinstance(runtime.neighbour_backend, ExistingNeighbourBackend)
    assert simulation.positions is runtime.state.positions
    assert simulation.neighbours is runtime.state.neighbours
    assert simulation.torch_generator is runtime.state.torch_generator
    assert "_potential_per_atom" in runtime.execution_caches
    assert "_unified_lambda_cache" in runtime.execution_caches
    assert not hasattr(runtime, "chemistry_engine")


def test_legacy_attributes_remain_bidirectional_runtime_adapters():
    simulation = _simulation()
    replacement = simulation.positions.clone()
    simulation.positions = replacement
    assert simulation.runtime.positions is replacement
    second = replacement.clone()
    simulation.runtime.positions = second
    assert simulation.positions is second


def test_runtime_step_is_bit_identical_to_retained_pre_split_step():
    runtime_route = _simulation(seed=91)
    legacy_route = _simulation(seed=91)

    runtime_route.step(8)
    legacy_route._legacy_step(8)

    assert torch.equal(runtime_route.positions, legacy_route.positions)
    assert torch.equal(runtime_route.velocities, legacy_route.velocities)
    assert torch.equal(runtime_route.forces, legacy_route.forces)
    assert torch.equal(
        runtime_route._potential_energy, legacy_route._potential_energy
    )
    assert torch.equal(
        runtime_route.torch_generator.get_state(),
        legacy_route.torch_generator.get_state(),
    )
    assert runtime_route.steps_taken == legacy_route.steps_taken == 8
    assert runtime_route.elapsed_femtoseconds == (
        legacy_route.elapsed_femtoseconds
    )
    assert runtime_route.rebuild_count == legacy_route.rebuild_count
    assert runtime_route.capped_steps == legacy_route.capped_steps
