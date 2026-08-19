"""Observational H-state pressure diagnostic regression checks."""

import torch
from pathlib import Path

from h_state_factorised_batched_torch import (
    GroupedFactorisedHStateBatchedSimulation,
    _begin_h_state_evaluation,
    _finish_h_state_evaluation,
    _record_h_state_group_pressure,
)


def _solver():
    solver = object.__new__(GroupedFactorisedHStateBatchedSimulation)
    solver.device = torch.device("cpu")
    solver.dtype = torch.float64
    solver._factorised_h_structure_cache = {}
    return solver


def test_h_state_pressure_accumulates_shapes_bytes_and_evaluation_totals():
    solver = _solver()
    first = solver._factorised_structure_for_signature(
        ((0,), (0,), (1,), (1,), (0, 1))
    )
    second = solver._factorised_structure_for_signature(
        ((0,), (0,))
    )

    _begin_h_state_evaluation(solver, 2)
    first_pressure = _record_h_state_group_pressure(solver, first, 4)
    _record_h_state_group_pressure(solver, second, 7)
    _finish_h_state_evaluation(solver)

    run = solver._h_state_run_diagnostics
    first_states = int(first["state_count"])
    second_states = int(second["state_count"])
    expected_elements = 4 * first_states * first_states

    assert run["evaluation_count"] == 1
    assert run["max_component_edge_count"] == 5
    assert run["max_state_count"] == first_states
    assert run["max_transition_count"] == int(first["state_first"].numel())
    assert run["max_topology_group"] == 7
    assert run["max_hamiltonian_shape"] == first_pressure["shape"]
    assert run["max_hamiltonian_elements"] == expected_elements
    assert run["max_hamiltonian_bytes"] == expected_elements * 8
    assert run["max_total_h_states_solved"] == (
        4 * first_states + 7 * second_states
    )
    assert run["max_topology_groups_per_evaluation"] == 2
    assert run["structure_cache_entries"] == 2
    assert run["structure_cache_cuda_bytes"] == 0
    assert run["cuda_max_memory_allocated"] == 0


def test_h_state_diagnostics_do_not_feed_back_into_structure_or_hamiltonian():
    solver = _solver()
    signature = ((0,), (0,), (1,), (1,), (0, 1))
    before = solver._factorised_structure_for_signature(signature)

    _begin_h_state_evaluation(solver, 1)
    _record_h_state_group_pressure(solver, before, 3)
    _finish_h_state_evaluation(solver)

    solver._h_state_run_diagnostics["max_state_count"] = 999999
    after = solver._factorised_structure_for_signature(signature)

    assert after is before
    assert after["state_count"] != 999999
    torch.testing.assert_close(after["state_mask"], before["state_mask"])
    torch.testing.assert_close(
        after["transition_flat_index"],
        before["transition_flat_index"],
    )


def test_batch_runner_persists_and_prints_h_state_pressure():
    source = Path("batch_runner.py").read_text(encoding="utf-8")
    for token in (
        '"h_state_group_diagnostics"',
        "H state pressure (group)",
        "largest Hamiltonian MB",
        "peak allocated",
        "pressure H solve",
    ):
        assert token in source
