"""Basis-independent finite-temperature heavy-valence state density."""

from __future__ import annotations

import torch


# Energy scale (eV) over which competing heavy-valence states are mixed.
# 10 meV is deliberately small relative to the calibrated ~0.47 eV state
# coupling, while making an exact state crossing a well-defined ensemble.
DEFAULT_HEAVY_VALENCE_TEMPERATURE = 0.01


def thermal_state_probabilities(hamiltonian, temperature):
    """Return diag(exp(-H/T)) / trace(exp(-H/T)), without eigensolvers.

    The detached lowest eigenvalue is subtracted solely to put the largest
    Boltzmann eigenvalue at one.  No eigensolver derivative enters autograd.
    Scalar shifts cancel exactly in the normalized density matrix, so this
    changes neither its value nor its derivative and does not open a gap.
    """
    with torch.no_grad():
        ground_energy = torch.linalg.eigvalsh(hamiltonian)[..., :1]
    identity = torch.eye(
        hamiltonian.shape[-1],
        device=hamiltonian.device,
        dtype=hamiltonian.dtype,
    )
    shifted = hamiltonian - ground_energy.unsqueeze(-1) * identity
    boltzmann = torch.matrix_exp(-shifted / temperature)
    diagonal_density = torch.diagonal(
        boltzmann, dim1=-2, dim2=-1
    )
    return diagonal_density / diagonal_density.sum(
        dim=-1, keepdim=True
    )
