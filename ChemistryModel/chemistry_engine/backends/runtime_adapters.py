"""Exact adapters around the established runtime algorithms."""

from __future__ import annotations

import numpy as np
import torch

from ..runtime import attach_runtime


class ExistingVelocityVerletIntegrator:
    """The existing velocity-Verlet sequence, moved without rearrangement."""

    def step(self, runtime, number_of_steps=1):
        dt = runtime.time_step
        conversion = 1.0 / 103.642
        masses = runtime.masses[:, None]

        for _ in range(int(number_of_steps)):
            acceleration = runtime.forces * conversion / masses
            movement = runtime.limit_move(
                runtime.velocities * dt
                + 0.5 * acceleration * dt * dt
            )
            runtime.positions = (runtime.positions + movement) % runtime.box_size

            if runtime.needs_rebuild():
                runtime.build_neighbours()

            new_forces, potential = runtime.compute_forces()
            new_acceleration = new_forces * conversion / masses
            ordinary_velocity = runtime.velocities + 0.5 * (
                acceleration + new_acceleration
            ) * dt
            capped_velocity = runtime.velocities
            runtime.velocities = torch.where(
                runtime._last_move_capped_mask[:, None],
                capped_velocity,
                ordinary_velocity,
            )

            runtime.forces = new_forces
            runtime._potential_energy = potential

            if runtime.thermostat_is_on:
                runtime.thermostat.apply(runtime)

            runtime.elapsed_femtoseconds += dt
            runtime.steps_taken += 1


class ExistingLangevinThermostat:
    """The established Langevin draw and update, unchanged."""

    def apply(self, runtime):
        decay = float(np.exp(-runtime.friction * runtime.time_step))
        masses = runtime.masses[:, None]
        scale = torch.sqrt(
            torch.tensor(
                8.617333e-5 * runtime.target_temperature / 103.642,
                device=runtime.device,
                dtype=runtime.dtype,
            ) / masses
        ) * float(np.sqrt(1.0 - decay ** 2))
        runtime.velocities = (
            runtime.velocities * decay
            + scale * torch.randn(
                runtime.velocities.shape,
                generator=runtime.torch_generator,
                device=runtime.device,
                dtype=runtime.dtype,
            )
        )


class ExistingAutogradForceBackend:
    def compute(self, runtime):
        return runtime.simulation_adapter.compute_forces()


class ExistingNeighbourBackend:
    def needs_rebuild(self, runtime):
        return runtime.simulation_adapter.needs_rebuild()

    def build(self, runtime):
        return runtime.simulation_adapter.build_neighbours()


def build_existing_runtime(simulation):
    return attach_runtime(
        simulation,
        integrator=ExistingVelocityVerletIntegrator(),
        thermostat=ExistingLangevinThermostat(),
        force_backend=ExistingAutogradForceBackend(),
        neighbour_backend=ExistingNeighbourBackend(),
    )
