
from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

import math
import torch

from valence_state_torch import (
    MAX_LOCAL_CANDIDATES,
    MAX_LOCAL_STATES,
)
from valence_state_batched_membership_torch import (
    BatchedHeavyValenceStateBatchedSimulation,
)

DEVICE = torch.device("cpu")
DTYPE = torch.float64


def dummy_solver():
    solver = object.__new__(
        BatchedHeavyValenceStateBatchedSimulation
    )
    solver.device = DEVICE
    solver.dtype = DTYPE
    solver._heavy_valence_structure_cache = {}
    return solver


def dense_reference_hamiltonian(diagonal, coupling, structure):
    state_count = int(structure["state_count"])
    first = structure["transition_first"]
    second = structure["transition_second"]
    transition_count = int(first.numel())

    basis = torch.zeros(
        (transition_count, state_count, state_count),
        dtype=DTYPE,
        device=DEVICE,
    )

    if transition_count:
        t = torch.arange(
            transition_count,
            dtype=torch.long,
            device=DEVICE,
        )
        basis[t, first, second] = 1.0
        basis[t, second, first] = 1.0

    return (
        torch.diag_embed(diagonal)
        + torch.einsum(
            "bt,tij->bij",
            -coupling,
            basis,
        )
    )


def small_equivalence(solver):
    structure = solver._heavy_structure(5, 2)

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(87123)

    group_size = 3
    states = int(structure["state_count"])
    transitions = int(
        structure["transition_first"].numel()
    )

    diagonal_old = torch.randn(
        (group_size, states),
        generator=generator,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    coupling_old = (
        0.05
        * torch.randn(
            (group_size, transitions),
            generator=generator,
            dtype=DTYPE,
            device=DEVICE,
        )
    ).requires_grad_(True)

    diagonal_new = (
        diagonal_old.detach().clone().requires_grad_(True)
    )
    coupling_new = (
        coupling_old.detach().clone().requires_grad_(True)
    )

    old = dense_reference_hamiltonian(
        diagonal_old,
        coupling_old,
        structure,
    )
    new = solver._assemble_heavy_hamiltonian(
        diagonal_new,
        coupling_new,
        structure,
    )

    matrix_error = float(
        torch.max(torch.abs(old - new))
    )

    old_energy = torch.linalg.eigvalsh(old)[:, 0].sum()
    new_energy = torch.linalg.eigvalsh(new)[:, 0].sum()

    old_grad = torch.autograd.grad(
        old_energy,
        (diagonal_old, coupling_old),
    )
    new_grad = torch.autograd.grad(
        new_energy,
        (diagonal_new, coupling_new),
    )

    gradient_error = max(
        float(torch.max(torch.abs(a - b)))
        for a, b in zip(old_grad, new_grad)
    )

    return (
        matrix_error,
        float(torch.abs(old_energy - new_energy)),
        gradient_error,
        states,
        transitions,
    )


def structure_check(solver, candidates, capacity):
    structure = solver._heavy_structure(
        candidates,
        capacity,
    )

    expected_states = math.comb(
        candidates,
        capacity,
    )
    expected_transitions = (
        expected_states
        * capacity
        * (candidates - capacity)
        // 2
    )

    return {
        "structure": structure,
        "states": int(structure["state_count"]),
        "expected_states": expected_states,
        "transitions": int(
            structure["transition_first"].numel()
        ),
        "expected_transitions": expected_transitions,
        "flat_count": int(
            structure["transition_flat_index"].numel()
        ),
        "no_dense_basis": (
            "transition_basis" not in structure
        ),
    }


def large_assembly_check(solver):
    info = structure_check(solver, 10, 4)
    structure = info["structure"]

    state_count = info["states"]
    transition_count = info["transitions"]

    diagonal = torch.linspace(
        -2.0,
        1.0,
        state_count,
        dtype=DTYPE,
        device=DEVICE,
    )[None, :].requires_grad_(True)

    coupling = torch.full(
        (1, transition_count),
        0.003,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )

    hamiltonian = solver._assemble_heavy_hamiltonian(
        diagonal,
        coupling,
        structure,
    )

    symmetry_error = float(
        torch.max(
            torch.abs(
                hamiltonian
                - hamiltonian.transpose(1, 2)
            )
        )
    )

    lowest = torch.linalg.eigvalsh(
        hamiltonian
    )[:, 0].sum()

    gradients = torch.autograd.grad(
        lowest,
        (diagonal, coupling),
    )

    finite = (
        bool(torch.isfinite(hamiltonian).all())
        and bool(torch.isfinite(lowest))
        and all(
            bool(torch.isfinite(value).all())
            for value in gradients
        )
    )

    return info, tuple(hamiltonian.shape), symmetry_error, finite


def main():
    print("LARGE HEAVY-VALENCE STATE EXECUTION VALIDATION")
    print()

    solver = dummy_solver()

    limit_pass = (
        MAX_LOCAL_CANDIDATES == 12
        and MAX_LOCAL_STATES >= math.comb(12, 4)
    )

    print(f"candidate limit : {MAX_LOCAL_CANDIDATES}")
    print(f"state limit     : {MAX_LOCAL_STATES}")
    print(
        "covers C(12,4) : "
        + ("PASS" if limit_pass else "FAIL")
    )

    (
        matrix_error,
        energy_error,
        gradient_error,
        small_states,
        small_transitions,
    ) = small_equivalence(solver)

    small_pass = (
        matrix_error <= 1e-15
        and energy_error <= 1e-12
        and gradient_error <= 1e-11
    )

    print()
    print(
        f"small control   : states={small_states}, "
        f"transitions={small_transitions}"
    )
    print(f"matrix error    : {matrix_error:.3e}")
    print(f"energy error    : {energy_error:.3e}")
    print(f"gradient error  : {gradient_error:.3e}")
    print(
        "old/new exact  : "
        + ("PASS" if small_pass else "FAIL")
    )

    crash, shape, symmetry_error, finite = (
        large_assembly_check(solver)
    )

    crash_pass = (
        crash["states"] == 210
        and crash["transitions"] == 2520
        and crash["flat_count"] == 5040
        and crash["no_dense_basis"]
        and shape == (1, 210, 210)
        and symmetry_error <= 1e-15
        and finite
    )

    print()
    print("former crash case, N=10 V=4")
    print(f"states          : {crash['states']}")
    print(f"transitions     : {crash['transitions']}")
    print(f"flat indices    : {crash['flat_count']}")
    print(f"assembled shape : {shape}")
    print(f"symmetry error  : {symmetry_error:.3e}")
    print(
        "eigh/autograd   : "
        + ("PASS" if finite else "FAIL")
    )

    maximum = structure_check(
        solver, 12, 4
    )

    max_pass = (
        maximum["states"] == 495
        and maximum["transitions"] == 7920
        and maximum["flat_count"] == 15840
        and maximum["no_dense_basis"]
    )

    print()
    print("maximum carbon case, N=12 V=4")
    print(f"states          : {maximum['states']}")
    print(f"transitions     : {maximum['transitions']}")
    print(f"flat indices    : {maximum['flat_count']}")

    old_bytes = (
        maximum["transitions"]
        * maximum["states"]
        * maximum["states"]
        * 8
    )
    compact_bytes = (
        maximum["flat_count"]
        * 8
    )

    print(
        "old basis size  : "
        f"{old_bytes / (1024 ** 3):.2f} GiB"
    )
    print(
        "compact indices : "
        f"{compact_bytes / 1024:.1f} KiB"
    )

    passed = all((
        limit_pass,
        small_pass,
        crash_pass,
        max_pass,
    ))

    print()
    if passed:
        print(
            "FINAL PASS - compact Hamiltonian assembly is equivalent on "
            "the control and supports the 210-state production case "
            "without the transition-basis memory explosion."
        )
        return

    print(
        "FINAL FAIL - do not rerun the production smoke test."
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
