"""Runtime boundaries intentionally kept outside the Hamiltonian."""

from __future__ import annotations

from typing import Protocol


class Integrator(Protocol):
    def step(self, runtime):
        ...


class Thermostat(Protocol):
    def apply(self, runtime):
        ...


class SimulationRuntime(Protocol):
    chemistry_engine: object
    integrator: object
    thermostat: object

    def step(self):
        ...
