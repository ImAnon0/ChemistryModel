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
    degrees_of_freedom,
    cutoff_distance=None,
    progress_every=None
):
    # Dimension agnostic: works for 2D and 3D without changes.

    particle_positions = starting_positions.copy()
    particle_velocities = starting_velocities.copy()

    number_of_steps = int(
        total_simulation_time / time_step
    )

    forces, potential_energy = calculate_all_interactions(
        particle_positions,
        box_size,
        cutoff_distance=cutoff_distance
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

        particle_positions %= box_size

        new_forces, potential_energy = calculate_all_interactions(
            particle_positions,
            box_size,
            cutoff_distance=cutoff_distance
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

        if (
            progress_every is not None
            and step_number % progress_every == 0
        ):
            print(
                f"  step {step_number} / {number_of_steps}",
                end="\r"
            )

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

            position_history.append(
                particle_positions.copy()
            )

            time_history.append(
                (step_number + 1) * time_step
            )

            kinetic_energy_history.append(kinetic_energy)
            potential_energy_history.append(potential_energy)
            total_energy_history.append(total_energy)

            energy_drift_history.append(
                total_energy - initial_total_energy
            )

            temperature_history.append(temperature)

    if progress_every is not None:
        print(" " * 40, end="\r")

    return (
        position_history,
        time_history,
        kinetic_energy_history,
        potential_energy_history,
        total_energy_history,
        energy_drift_history,
        temperature_history
    )
