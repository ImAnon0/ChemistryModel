"""Immutable physics and execution configuration records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


def _normalise_parameter(value):
    if hasattr(value, "tolist"):
        return _normalise_parameter(value.tolist())
    if isinstance(value, dict):
        return {
            str(key): _normalise_parameter(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise_parameter(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def public_parameter_payload(module) -> dict:
    """Capture public constant data without maintaining another file list."""

    payload = {}
    for name, value in vars(module).items():
        if not name.isupper() or callable(value):
            continue
        if getattr(value, "__spec__", None) is not None:
            continue
        payload[name] = _normalise_parameter(value)
    return payload


def parameter_identity(parameters) -> tuple[str, str]:
    encoded = json.dumps(
        _normalise_parameter(parameters), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CapacitySpec:
    solver: str
    temperature_eV: float
    h_regularisation_temperature_eV: float


@dataclass(frozen=True)
class GeometrySpec:
    convention: str


@dataclass(frozen=True)
class PhysicsSpec:
    model_id: str
    parameter_sha256: str
    parameter_payload_json: str
    capacity: CapacitySpec
    geometry: GeometrySpec
    enabled_terms: tuple[str, ...]

    @classmethod
    def unified_radial_v1(
        cls,
        parameters,
        *,
        capacity_temperature,
        h_regularisation_temperature,
    ):
        payload, digest = parameter_identity(parameters)
        return cls(
            model_id="unified_radial_v1",
            parameter_sha256=digest,
            parameter_payload_json=payload,
            capacity=CapacitySpec(
                solver="existing_scipy_l_bfgs_b_dual",
                temperature_eV=float(capacity_temperature),
                h_regularisation_temperature_eV=float(
                    h_regularisation_temperature
                ),
            ),
            geometry=GeometrySpec(
                convention="established_heavy_angle_topology_v1"
            ),
            enabled_terms=(
                "base_reactive",
                "unified_capacity_correction",
                "established_geometry_correction",
            ),
        )


@dataclass(frozen=True)
class ExecutionConfig:
    device: str
    dtype: str
    box_count: int
    atoms_per_box: int
    neighbour_strategy: str
    caching: str
    solver_execution_mode: str
