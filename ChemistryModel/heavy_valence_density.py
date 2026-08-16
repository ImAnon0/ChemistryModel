"""Basis-independent finite-temperature heavy-valence state density."""

from __future__ import annotations

import torch


# Energy scale (eV) over which competing heavy-valence states are mixed.
# 10 meV is deliberately small relative to the calibrated ~0.47 eV state
# coupling, while making an exact state crossing a well-defined ensemble.
DEFAULT_HEAVY_VALENCE_TEMPERATURE = 0.01


class _ThermalStateDiagonal(torch.autograd.Function):
    """Exact diagonal of a normalized matrix exponential with stable forces.

    The forward diagonalizes the real-symmetric Hamiltonian under no-grad.
    The backward is the spectral Fréchet derivative of the matrix exponential,
    evaluated explicitly with the divided-difference limit at degeneracy.  No
    derivative of an eigenvector is requested, so repeated eigenvalues remain
    finite and basis independent.
    """

    @staticmethod
    def forward(ctx, hamiltonian, temperature):
        temperature = float(temperature)
        with torch.no_grad():
            eigenvalues, eigenvectors = torch.linalg.eigh(hamiltonian)
            ground = eigenvalues[..., :1]
            scaled = -(eigenvalues - ground) / temperature
            weights = torch.exp(scaled)
            partition = weights.sum(dim=-1, keepdim=True)
            probabilities = (
                eigenvectors.square() * weights.unsqueeze(-2)
            ).sum(dim=-1) / partition

        ctx.temperature = temperature
        ctx.save_for_backward(
            eigenvectors, scaled, weights, partition, probabilities
        )
        return probabilities

    @staticmethod
    def backward(ctx, grad_output):
        eigenvectors, scaled, weights, partition, probabilities = (
            ctx.saved_tensors
        )

        # For L = Tr(G exp(A)) / Tr(exp(A)), the matrix multiplying d exp(A)
        # is (G - L I) / Z. G is diagonal because only diag(rho) is returned.
        objective = (grad_output * probabilities).sum(
            dim=-1, keepdim=True
        )
        centred = grad_output - objective
        transformed = torch.matmul(
            eigenvectors.transpose(-2, -1),
            centred.unsqueeze(-1) * eigenvectors,
        ) / partition.unsqueeze(-1)

        difference = scaled.unsqueeze(-1) - scaled.unsqueeze(-2)
        weight_difference = weights.unsqueeze(-1) - weights.unsqueeze(-2)
        close = torch.abs(difference) < 1e-7
        safe_difference = torch.where(
            close, torch.ones_like(difference), difference
        )
        quotient = weight_difference / safe_difference

        # exp((x+y)/2) sinh((x-y)/2)/((x-y)/2), expanded at zero.
        half_difference = 0.5 * difference
        half_square = half_difference.square()
        limit = torch.exp(
            0.5 * (scaled.unsqueeze(-1) + scaled.unsqueeze(-2))
        ) * (1.0 + half_square / 6.0 + half_square.square() / 120.0)
        divided_difference = torch.where(close, limit, quotient)

        gradient_scaled = torch.matmul(
            eigenvectors,
            torch.matmul(
                divided_difference * transformed,
                eigenvectors.transpose(-2, -1),
            ),
        )
        gradient_hamiltonian = -gradient_scaled / ctx.temperature
        gradient_hamiltonian = 0.5 * (
            gradient_hamiltonian
            + gradient_hamiltonian.transpose(-2, -1)
        )
        return gradient_hamiltonian, None


def thermal_state_probabilities(hamiltonian, temperature):
    """Return diag(exp(-H/T)) / trace(exp(-H/T)) exactly.

    The detached lowest eigenvalue is subtracted solely to put the largest
    Boltzmann eigenvalue at one.  No eigensolver derivative enters autograd.
    Scalar shifts cancel exactly in the normalized density matrix, so this
    changes neither its value nor its derivative and does not open a gap.
    """
    return _ThermalStateDiagonal.apply(hamiltonian, float(temperature))
