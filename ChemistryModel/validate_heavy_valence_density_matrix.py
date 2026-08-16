"""Targeted validation for finite-temperature heavy-valence membership."""

from __future__ import annotations

import torch

from heavy_valence_density import (
    DEFAULT_HEAVY_VALENCE_TEMPERATURE,
    thermal_state_probabilities,
)


def old_ground_probabilities(hamiltonian):
    """Historical eigenvector formulation, retained here as a reference."""
    _, vectors = torch.linalg.eigh(hamiltonian)
    ground = vectors[..., 0]
    return ground.square() / ground.square().sum(dim=-1, keepdim=True)


def density_force(x_value, temperature):
    x = torch.tensor(x_value, dtype=torch.float64, requires_grad=True)
    zero = x * 0.0
    hamiltonian = torch.stack(
        (torch.stack((x, zero)), torch.stack((zero, -x)))
    ).unsqueeze(0)
    probability = thermal_state_probabilities(
        hamiltonian, temperature
    )[0]
    observable = probability[0]
    force = -torch.autograd.grad(observable, x)[0]
    return float(observable.detach()), float(force.detach())


def main():
    temperature = DEFAULT_HEAVY_VALENCE_TEMPERATURE

    # Exact degeneracy: the density matrix must be the basis-independent
    # equal mixture and its derivative must remain finite.
    exact_probability, exact_force = density_force(0.0, temperature)
    assert abs(exact_probability - 0.5) < 1e-12
    assert torch.isfinite(torch.tensor(exact_force))

    # The two one-sided evaluations must converge to the same finite force.
    left_probability, left_force = density_force(-1e-8, temperature)
    right_probability, right_force = density_force(1e-8, temperature)
    assert abs(left_probability - right_probability) < 2e-6
    assert abs(left_force - right_force) < 1e-7
    assert abs(left_force - exact_force) < 1e-7

    # Far from a crossing, thermal membership should reproduce the historical
    # ground eigenvector to exponentially small error.
    far = torch.tensor(
        [[[-0.20, -0.03], [-0.03, 0.20]]], dtype=torch.float64
    )
    old = old_ground_probabilities(far)
    new = thermal_state_probabilities(far, temperature)
    max_difference = float((old - new).abs().max())
    assert max_difference < 1e-12

    print("HEAVY-VALENCE DENSITY-MATRIX VALIDATION")
    print(f"  temperature       : {temperature:.6f} eV")
    print(f"  exact probability : {exact_probability:.12f}")
    print(f"  exact force       : {exact_force:.12f}")
    print(f"  near forces       : {left_force:.12f}, {right_force:.12f}")
    print(f"  far max |new-old| : {max_difference:.3e}")
    print("FINAL PASS")


if __name__ == "__main__":
    main()
