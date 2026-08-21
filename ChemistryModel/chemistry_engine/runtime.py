"""Concrete simulation runtime, separate from chemistry definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


RUNTIME_STATE_FIELDS = (
    "time_step", "target_temperature", "friction", "thermostat_is_on",
    "positions", "velocities", "forces", "steps_taken",
    "elapsed_femtoseconds", "neighbours", "neighbour_mask",
    "reference_positions", "rebuild_count", "_potential_energy",
    "maximum_step", "capped_steps", "capped_atom_counts",
    "last_capped_atoms", "_last_move_capped_mask", "random_generator",
    "torch_generator",
)

RUNTIME_CACHE_FIELDS = (
    "_potential_per_atom", "_potential_cache_source",
    "_potential_cache_version", "_chemical_pair_cache",
    "_chemical_pair_cache_source", "_chemical_pair_cache_version",
    "_softening_cache", "_over_scale_cache",
    "_heavy_membership_topology_cache",
    "_heavy_membership_cache_signature", "_factorised_h_structure_cache",
    "_heavy_valence_structure_cache",
    "_cached_factorised_group_state_energies", "_cached_h_topology",
    "_h_candidate_cache", "_h_topology_cache_hits",
    "_h_topology_cache_misses", "_unified_lambda_cache",
)


def _pending_key(name):
    return f"_runtime_pending:{name}"


def _pending_cache_key(name):
    return f"_runtime_pending_cache:{name}"


class RuntimeOwnedField:
    """Compatibility descriptor whose storage moves into RuntimeState."""

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        runtime = instance.__dict__.get("_simulation_runtime")
        if runtime is not None:
            return getattr(runtime.state, self.name)
        try:
            return instance.__dict__[_pending_key(self.name)]
        except KeyError as error:
            raise AttributeError(self.name) from error

    def __set__(self, instance, value):
        runtime = instance.__dict__.get("_simulation_runtime")
        if runtime is not None:
            setattr(runtime.state, self.name, value)
        else:
            instance.__dict__[_pending_key(self.name)] = value


class RuntimeCacheField:
    """Compatibility descriptor backed by the runtime cache store."""

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        runtime = instance.__dict__.get("_simulation_runtime")
        if runtime is not None:
            try:
                return runtime.execution_caches[self.name]
            except KeyError as error:
                raise AttributeError(self.name) from error
        try:
            return instance.__dict__[_pending_cache_key(self.name)]
        except KeyError as error:
            raise AttributeError(self.name) from error

    def __set__(self, instance, value):
        runtime = instance.__dict__.get("_simulation_runtime")
        if runtime is not None:
            runtime.execution_caches[self.name] = value
        else:
            instance.__dict__[_pending_cache_key(self.name)] = value


@dataclass
class RuntimeState:
    time_step: float
    target_temperature: float
    friction: float
    thermostat_is_on: bool
    positions: Any
    velocities: Any
    forces: Any
    steps_taken: int
    elapsed_femtoseconds: float
    neighbours: Any
    neighbour_mask: Any
    reference_positions: Any
    rebuild_count: int
    _potential_energy: Any
    maximum_step: float
    capped_steps: int
    capped_atom_counts: Any
    last_capped_atoms: tuple
    _last_move_capped_mask: Any
    random_generator: Any
    torch_generator: Any


class Integrator(Protocol):
    def step(self, runtime, number_of_steps=1):
        ...


class Thermostat(Protocol):
    def apply(self, runtime):
        ...


class SimulationRuntime:
    """Own dynamics state and coordinate execution backends."""

    def __init__(self, simulation_adapter, state, *, integrator, thermostat,
                 force_backend, neighbour_backend, execution_caches=None):
        object.__setattr__(self, "simulation_adapter", simulation_adapter)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "masses", simulation_adapter.masses)
        object.__setattr__(self, "box_size", float(simulation_adapter.box_size))
        object.__setattr__(self, "device", simulation_adapter.device)
        object.__setattr__(self, "dtype", simulation_adapter.dtype)
        object.__setattr__(self, "integrator", integrator)
        object.__setattr__(self, "thermostat", thermostat)
        object.__setattr__(self, "force_backend", force_backend)
        object.__setattr__(self, "neighbour_backend", neighbour_backend)
        object.__setattr__(self, "execution_caches", dict(execution_caches or {}))

    def __getattr__(self, name):
        if name in RUNTIME_STATE_FIELDS:
            return getattr(self.state, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in RUNTIME_STATE_FIELDS and "state" in self.__dict__:
            setattr(self.state, name, value)
            return
        object.__setattr__(self, name, value)

    def step(self, number_of_steps=1):
        return self.integrator.step(self, number_of_steps)

    def compute_forces(self):
        return self.force_backend.compute(self)

    def needs_rebuild(self):
        return self.neighbour_backend.needs_rebuild(self)

    def build_neighbours(self):
        return self.neighbour_backend.build(self)

    def limit_move(self, movement):
        return self.simulation_adapter.limit_move(movement)


def attach_runtime(simulation, *, integrator, thermostat, force_backend,
                   neighbour_backend):
    """Move existing state references into a runtime without copying tensors."""

    values = {name: getattr(simulation, name) for name in RUNTIME_STATE_FIELDS}
    caches = {}
    for name in RUNTIME_CACHE_FIELDS:
        try:
            caches[name] = getattr(simulation, name)
        except AttributeError:
            pass
    runtime = SimulationRuntime(
        simulation, RuntimeState(**values), integrator=integrator,
        thermostat=thermostat, force_backend=force_backend,
        neighbour_backend=neighbour_backend, execution_caches=caches,
    )
    simulation.__dict__["_simulation_runtime"] = runtime
    for name in RUNTIME_STATE_FIELDS:
        simulation.__dict__.pop(_pending_key(name), None)
    for name in RUNTIME_CACHE_FIELDS:
        simulation.__dict__.pop(_pending_cache_key(name), None)
    return runtime
