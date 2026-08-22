"""Smooth electronic-redistribution hypotheses over unified radial bonding.

The candidates deliberately vanish for a contact taper of exactly zero or
one.  They therefore describe only the extra electronic response while a
contact is being formed/broken, rather than adding permanent electrostatics or
re-counting settled bond energies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import reactive as R
from research.unified_bond_capacity import UnifiedBondCapacityEnergyPrototype


DEFAULT_PARAMETERS = Path(
    "research_data/benchmark/diagnostics/electronic_state_parameters.json"
)


def smooth_contact_numpy(distance, first, second):
    first_index = R.ELEMENT_INDEX[first]
    second_index = R.ELEMENT_INDEX[second]
    inner = float(R.CUTOFF_INNER[first_index, second_index])
    outer = float(R.CUTOFF_OUTER[first_index, second_index])
    fraction = np.clip((float(distance) - inner) / (outer - inner), 0.0, 1.0)
    return float(0.5 * (1.0 + np.cos(np.pi * fraction)))


def numpy_electronic_features(symbols, coordinates, source_values):
    """Return scalar, vector and traceless-tensor ambiguity features."""

    positions = np.asarray(coordinates, dtype=float)
    count = len(symbols)
    scalar_source = np.zeros(count)
    vector_source = np.zeros((count, 3))
    tensor_source = np.zeros((count, 3, 3))
    identity = np.eye(3) / 3.0
    for first in range(count):
        for second in range(first + 1, count):
            offset = positions[second] - positions[first]
            distance = float(np.linalg.norm(offset))
            if distance <= 1e-12:
                continue
            taper = smooth_contact_numpy(
                distance, symbols[first], symbols[second]
            )
            ambiguity = taper * (1.0 - taper)
            if ambiguity == 0.0:
                continue
            direction = offset / distance
            difference = float(
                source_values[symbols[second]]
                - source_values[symbols[first]]
            )
            magnitude = abs(difference)
            quadrupole_axis = np.outer(direction, direction) - identity
            # Reversing the directed edge reverses both delta-source and its
            # unit vector, so the vector product has the same global direction
            # at both endpoints.
            for atom, sign in ((first, 1.0), (second, 1.0)):
                scalar_source[atom] += ambiguity * magnitude
                vector_source[atom] += sign * ambiguity * difference * direction
                tensor_source[atom] += ambiguity * magnitude * quadrupole_axis
    return {
        "local_scalar": float(np.square(scalar_source).sum()),
        "polarisation_vector": float(np.square(vector_source).sum()),
        "multipole_tensor": float(np.square(tensor_source).sum()),
    }


def _load_parameters(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"electronic-state parameter audit not found: {path}; "
            "run fit_observable_parameters.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


class ElectronicStateCorrectionPrototype(UnifiedBondCapacityEnergyPrototype):
    """Base class for independently frozen electronic feature hypotheses."""

    research_only = True
    electronic_hypothesis = "combined"
    physics_model_name = "research_electronic_state_correction"
    physics_model_revision = 0

    def __init__(
        self,
        *args,
        electronic_parameter_path=DEFAULT_PARAMETERS,
        electronic_source_values=None,
        electronic_coefficients=None,
        **kwargs,
    ):
        if electronic_source_values is None or electronic_coefficients is None:
            payload = _load_parameters(electronic_parameter_path)
            electronic_source_values = payload["source_values_e"]
            electronic_coefficients = payload["hypotheses"][
                self.electronic_hypothesis
            ]["coefficients_eV"]
        self.electronic_source_values = {
            symbol: float(electronic_source_values.get(symbol, 0.0))
            for symbol in R.ELEMENTS
        }
        self.electronic_coefficients = {
            key: float(value) for key, value in electronic_coefficients.items()
        }
        self._electronic_state_diagnostics = None
        super().__init__(*args, **kwargs)

    def _electronic_features(self, positions):
        neighbours = self.neighbours
        mask = self._neighbour_weight
        offsets = (
            self._gather_neighbours(positions, neighbours, "positions")
            - positions[:, None, :]
        )
        offsets = offsets - self.box_size * torch.round(offsets / self.box_size)
        distances = torch.sqrt(
            torch.clamp(torch.sum(offsets * offsets, dim=2), min=1e-12)
        )
        centre_types = self.types[:, None].expand_as(neighbours)
        other_types = self.types[neighbours]
        inner = self.cutoff_inner[centre_types, other_types]
        outer = self.cutoff_outer[centre_types, other_types]
        fraction = torch.clamp(
            (distances - inner) / torch.clamp(outer - inner, min=1e-12),
            0.0,
            1.0,
        )
        taper = 0.5 * (1.0 + torch.cos(torch.pi * fraction)) * mask
        ambiguity = taper * (1.0 - taper)

        source_table = torch.as_tensor(
            [self.electronic_source_values[symbol] for symbol in R.ELEMENTS],
            device=positions.device,
            dtype=positions.dtype,
        )
        centre_source = source_table[self.types][:, None]
        other_source = source_table[other_types]
        difference = other_source - centre_source
        direction = offsets / distances[:, :, None]

        scalar_source = torch.sum(ambiguity * torch.abs(difference), dim=1)
        vector_source = torch.sum(
            ambiguity[:, :, None] * difference[:, :, None] * direction,
            dim=1,
        )
        outer_product = direction[:, :, :, None] * direction[:, :, None, :]
        identity = torch.eye(3, device=positions.device, dtype=positions.dtype) / 3.0
        tensor_source = torch.sum(
            ambiguity[:, :, None, None]
            * torch.abs(difference)[:, :, None, None]
            * (outer_product - identity),
            dim=1,
        )
        return {
            "local_scalar": scalar_source.square(),
            "polarisation_vector": vector_source.square().sum(dim=1),
            "multipole_tensor": tensor_source.square().sum(dim=(1, 2)),
        }

    def energy_per_atom(self, positions):
        base = super().energy_per_atom(positions)
        features = self._electronic_features(positions)
        correction = torch.zeros_like(base)
        for key, coefficient in self.electronic_coefficients.items():
            correction = correction + coefficient * features[key]
        self._electronic_state_diagnostics = {
            "hypothesis": self.electronic_hypothesis,
            "source_values_e": dict(self.electronic_source_values),
            "coefficients_eV": dict(self.electronic_coefficients),
            "feature_totals": {
                key: float(value.sum().detach().cpu())
                for key, value in features.items()
            },
            "correction_eV": float(correction.sum().detach().cpu()),
        }
        return base + correction


class LocalElectronicDescriptorPrototype(ElectronicStateCorrectionPrototype):
    electronic_hypothesis = "local_scalar"
    physics_model_name = "research_local_electronic_descriptor_v0"


class PolarisationResponsePrototype(ElectronicStateCorrectionPrototype):
    electronic_hypothesis = "polarisation_vector"
    physics_model_name = "research_polarisation_response_v0"


class MultipoleDensityPrototype(ElectronicStateCorrectionPrototype):
    electronic_hypothesis = "multipole_tensor"
    physics_model_name = "research_multipole_density_v0"


class CombinedElectronicStatePrototype(ElectronicStateCorrectionPrototype):
    electronic_hypothesis = "combined"
    physics_model_name = "research_combined_electronic_state_v0"
