"""Build an InteractionContext from the current Torch runtime."""

from __future__ import annotations

from ..config import ExecutionConfig
from ..context import InteractionContext


_ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8}


def interaction_context(simulation, positions):
    atomic_numbers = getattr(simulation, "_chemistry_atomic_numbers", None)
    if atomic_numbers is None:
        atomic_numbers = tuple(_ATOMIC_NUMBERS[symbol] for symbol in simulation.symbols)
        simulation._chemistry_atomic_numbers = atomic_numbers
    assignments = getattr(simulation, "_chemistry_batch_assignment", None)
    if assignments is None:
        assignments = tuple(
            index // int(simulation.per_box)
            for index in range(len(simulation.symbols))
        )
        simulation._chemistry_batch_assignment = assignments
    return InteractionContext(
        positions=positions,
        element_types=simulation.types,
        atomic_numbers=atomic_numbers,
        neighbours=simulation.neighbours,
        neighbour_mask=simulation.neighbour_mask,
        box_size=float(simulation.box_size),
        box_count=int(simulation.box_count),
        atoms_per_box=int(simulation.per_box),
        batch_assignment=assignments,
        tensors={
            "bond_length": simulation.bond_length,
            "bond_depth": simulation.bond_depth,
            "bond_width": simulation.bond_width,
            "valence": simulation.valence,
        },
    )


def execution_config(simulation):
    return ExecutionConfig(
        device=str(simulation.device),
        dtype=str(simulation.dtype),
        box_count=int(simulation.box_count),
        atoms_per_box=int(simulation.per_box),
        neighbour_strategy=type(simulation).build_neighbours.__qualname__,
        caching="existing_runtime_caches",
        solver_execution_mode="scipy_cpu_reference",
    )
