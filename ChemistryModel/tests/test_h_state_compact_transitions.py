"""Exactness and storage checks for compact grouped H-state transitions."""

import torch

from h_state_factorised_batched_torch import (
    DEFAULT_H_S2_EIGVALSH_CHUNK_SIZE,
    GroupedFactorisedHStateBatchedSimulation,
    _assemble_factorised_hamiltonian,
    _bounded_factorised_eigvalsh,
    _record_h_eigvalsh_execution,
    _resolved_h_s2_eigvalsh_chunk_size,
)


def _solver():
    solver = object.__new__(GroupedFactorisedHStateBatchedSimulation)
    solver.device = torch.device("cpu")
    solver.dtype = torch.float64
    solver.h_transition_assembly = "compact"
    solver._factorised_h_structure_cache = {}
    return solver


def _dense_reference(diagonal, coupling, structure):
    batch_count, state_count = diagonal.shape
    transition_count = int(structure["state_first"].numel())
    basis = torch.zeros(
        transition_count, state_count, state_count,
        dtype=diagonal.dtype, device=diagonal.device,
    )
    if transition_count:
        transition = torch.arange(transition_count, device=diagonal.device)
        first = structure["state_first"]
        second = structure["state_second"]
        basis[transition, first, second] = 1.0
        basis[transition, second, first] = 1.0
    return torch.diag_embed(diagonal) + torch.einsum(
        "bt,tij->bij", -coupling, basis
    )


def test_shared_chunk_resolution_covers_aliased_lower_layers():
    class AliasedExecutionLayer:
        pass

    solver = AliasedExecutionLayer()
    assert DEFAULT_H_S2_EIGVALSH_CHUNK_SIZE == 512
    assert _resolved_h_s2_eigvalsh_chunk_size(solver) == 512

    hamiltonian = torch.zeros(1056, 2, 2, dtype=torch.float64)
    _record_h_eigvalsh_execution(solver, hamiltonian)
    diagnostics = solver._h_state_run_diagnostics
    assert diagnostics["s2_eigvalsh_chunk_size"] == 512
    assert diagnostics["max_original_s2_batch"] == 1056
    assert diagnostics["largest_actual_eigvalsh_batch"] == 512

    solver.h_s2_eigvalsh_chunk_size = 128
    assert _resolved_h_s2_eigvalsh_chunk_size(solver) == 128


def test_compact_structure_has_no_dense_transition_basis():
    solver = _solver()
    # Multiple shared-H competitions plus an H-H bridge produce a nontrivial
    # state graph with several transitions.
    signature = ((0,), (0,), (1,), (1,), (0, 1))
    structure = solver._factorised_structure_for_signature(signature)
    transitions = int(structure["state_first"].numel())
    states = int(structure["state_count"])
    assert states > 2
    assert transitions > 2
    assert "transition_basis" not in structure
    assert structure["transition_flat_index"].dtype == torch.long
    assert structure["transition_flat_index"].numel() == 2 * transitions

    old_dense_bytes = transitions * states * states * torch.tensor(
        [], dtype=torch.float64
    ).element_size()
    compact_bytes = structure["transition_flat_index"].numel() * torch.tensor(
        [], dtype=torch.long
    ).element_size()
    assert compact_bytes < old_dense_bytes


def test_compact_hamiltonian_energy_and_gradients_match_dense_reference():
    torch.manual_seed(1729)
    structure = _solver()._factorised_structure_for_signature(
        ((0,), (0,), (1,), (1,), (0, 1))
    )
    states = int(structure["state_count"])
    transitions = int(structure["state_first"].numel())
    diagonal = torch.randn(4, states, dtype=torch.float64, requires_grad=True)
    coupling = torch.rand(4, transitions, dtype=torch.float64, requires_grad=True)
    compact = _assemble_factorised_hamiltonian(
        diagonal, coupling, structure["transition_flat_index"]
    )
    dense = _dense_reference(diagonal, coupling, structure)
    torch.testing.assert_close(compact, dense, rtol=0.0, atol=0.0)

    compact_energy = torch.linalg.eigvalsh(compact)[:, 0].sum()
    compact_gradient = torch.autograd.grad(
        compact_energy, (diagonal, coupling), retain_graph=True
    )
    dense_energy = torch.linalg.eigvalsh(dense)[:, 0].sum()
    dense_gradient = torch.autograd.grad(dense_energy, (diagonal, coupling))
    torch.testing.assert_close(compact_energy, dense_energy, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        compact_gradient[0], dense_gradient[0], rtol=1e-13, atol=1e-13
    )
    torch.testing.assert_close(
        compact_gradient[1], dense_gradient[1], rtol=1e-13, atol=1e-13
    )


def test_compact_assembly_handles_topology_without_transitions():
    diagonal = torch.tensor([[1.0]], dtype=torch.float64, requires_grad=True)
    coupling = torch.empty((1, 0), dtype=torch.float64, requires_grad=True)
    result = _assemble_factorised_hamiltonian(
        diagonal, coupling, torch.empty(0, dtype=torch.long)
    )
    torch.testing.assert_close(result, torch.tensor([[[1.0]]], dtype=torch.float64))


def _devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def test_bounded_s2_eigvalsh_matches_full_values_and_gradients():
    chunk_size = 8
    for device in _devices():
        for batch_size in (3, 8, 9, 27):
            torch.manual_seed(1000 + batch_size)
            raw = torch.randn(
                batch_size,
                2,
                2,
                dtype=torch.float64,
                device=device,
            )
            symmetric = 0.5 * (
                raw + raw.transpose(-1, -2)
            )
            full_input = symmetric.detach().clone().requires_grad_(True)
            chunked_input = symmetric.detach().clone().requires_grad_(True)

            full = torch.linalg.eigvalsh(full_input)
            chunked = _bounded_factorised_eigvalsh(
                chunked_input,
                chunk_size,
            )
            torch.testing.assert_close(
                chunked,
                full,
                rtol=0.0,
                atol=0.0,
            )

            weights = torch.linspace(
                0.5,
                1.5,
                full.numel(),
                dtype=torch.float64,
                device=device,
            ).reshape_as(full)
            full_gradient, = torch.autograd.grad(
                torch.sum(full * weights),
                full_input,
            )
            chunked_gradient, = torch.autograd.grad(
                torch.sum(chunked * weights),
                chunked_input,
            )
            torch.testing.assert_close(
                chunked_gradient,
                full_gradient,
                rtol=0.0,
                atol=0.0,
            )


def test_bounded_eigvalsh_submits_fixed_full_chunks_then_remainder(monkeypatch):
    original = torch.linalg.eigvalsh
    submitted = []

    def recording_eigvalsh(values):
        submitted.append(tuple(values.shape))
        return original(values)

    monkeypatch.setattr(
        torch.linalg,
        "eigvalsh",
        recording_eigvalsh,
    )
    values = torch.randn(27, 2, 2, dtype=torch.float64)
    values = 0.5 * (values + values.transpose(-1, -2))
    _bounded_factorised_eigvalsh(values, 8)
    assert submitted == [
        (8, 2, 2),
        (8, 2, 2),
        (8, 2, 2),
        (3, 2, 2),
    ]


def test_bounded_eigvalsh_preserves_s_greater_than_two_fallback(monkeypatch):
    original = torch.linalg.eigvalsh
    submitted = []

    def recording_eigvalsh(values):
        submitted.append(tuple(values.shape))
        return original(values)

    monkeypatch.setattr(
        torch.linalg,
        "eigvalsh",
        recording_eigvalsh,
    )
    values = torch.randn(27, 3, 3, dtype=torch.float64)
    values = 0.5 * (values + values.transpose(-1, -2))
    expected = original(values)
    actual = _bounded_factorised_eigvalsh(values, 8)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert submitted == [(27, 3, 3)]
