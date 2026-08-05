import numpy as np

from interactions import calculate_all_interactions


def run_simulation(
    starting_positions,
    starting_velocities,
    particle_mass,
    box_size,
    time_step,
    total_simulation_time,
    record_every,
    degrees_of_freedom
):
    # Copy the arrays so this function does not modify
    # the original arrays supplied by ChemistryModel.py.

    particle_positions = starting_positions.copy()
    particle_velocities = starting_velocities.copy()

    number_of_steps = int(
        total_simulation_time / time_step
    )

    forces, potential_energy = (
        calculate_all_interactions(
            particle_positions,
            box_size
        )
    )

    initial_kinetic_energy = (
        0.5
        * particle_mass
        * np.sum(particle_velocities ** 2)
    )

    initial_total_energy = (
        initial_kinetic_energy
        + potential_energy
    )


    position_history = []
    time_history = []

    kinetic_energy_history = []
    potential_energy_history = []
    total_energy_history = []
    energy_drift_history = []
    temperature_history = []


    for step_number in range(number_of_steps):
        # Velocity-Verlet position update.

        particle_positions = (
            particle_positions
            + particle_velocities * time_step
            + 0.5
            * (forces / particle_mass)
            * time_step ** 2
        )

        # Periodic boundaries.

        particle_positions %= box_size

        new_forces, potential_energy = (
            calculate_all_interactions(
                particle_positions,
                box_size
            )
        )

        # Velocity-Verlet velocity update.

        particle_velocities = (
            particle_velocities
            + 0.5
            * (forces + new_forces)
            / particle_mass
            * time_step
        )

        forces = new_forces

        if step_number % record_every == 0:
            kinetic_energy = (
                0.5
                * particle_mass
                * np.sum(particle_velocities ** 2)
            )

            temperature = (
                2.0
                * kinetic_energy
                / degrees_of_freedom
            )

            total_energy = (
                kinetic_energy
                + potential_energy
            )

            energy_drift = (
                total_energy
                - initial_total_energy
            )

            current_time = (
                (step_number + 1)
                * time_step
            )

            position_history.append(
                particle_positions.copy()
            )

            time_history.append(
                current_time
            )

            kinetic_energy_history.append(
                kinetic_energy
            )

            potential_energy_history.append(
                potential_energy
            )

            total_energy_history.append(
                total_energy
            )

            energy_drift_history.append(
                energy_drift
            )

            temperature_history.append(
                temperature
            )


    return (
        position_history,
        time_history,
        kinetic_energy_history,
        potential_energy_history,
        total_energy_history,
        energy_drift_history,
        temperature_history
    )