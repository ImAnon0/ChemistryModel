import numpy as np

from interactions import calculate_all_interactions

from thermostat import calculate_temperature


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
    thermostat=None,
    equilibration_time=0.0,
    progress_every=None
):
    # Dimension agnostic: works for 2D and 3D without changes.
    #
    # If a thermostat is supplied it acts only for the first
    # `equilibration_time` of the run. After that the system is
    # left alone, so total energy should be conserved again and
    # the drift figure means something.

    particle_positions = starting_positions.copy()
    particle_velocities = starting_velocities.copy()

    number_of_steps = int(
        total_simulation_time / time_step
    )

    equilibration_steps = int(
        equilibration_time / time_step
    )

    forces, potential_energy = calculate_all_interactions(
        particle_positions,
        box_size,
        cutoff_distance=cutoff_distance
    )

    reference_total_energy = None

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

        thermostat_is_active = (
            thermostat is not None
            and step_number < equilibration_steps
        )

        if thermostat_is_active:
            particle_velocities = thermostat.apply(
                particle_velocities,
                time_step,
                particle_mass,
                degrees_of_freedom
            )

        kinetic_energy = (
            0.5
            * particle_mass
            * np.sum(particle_velocities ** 2)
        )

        total_energy = kinetic_energy + potential_energy

        # Energy is deliberately not conserved while the
        # thermostat is running, so the drift baseline is only
        # fixed once the thermostat lets go.

        if not thermostat_is_active and reference_total_energy is None:
            reference_total_energy = total_energy

        if (
            progress_every is not None
            and step_number % progress_every == 0
        ):
            phase_label = (
                "equilibrating"
                if thermostat_is_active
                else "production   "
            )

            current_temperature = (
                2.0 * kinetic_energy / degrees_of_freedom
            )

            print(
                f"  {phase_label} "
                f"step {step_number} / {number_of_steps}  "
                f"T = {current_temperature:.4f}",
                end="\r"
            )

        if step_number % record_every == 0:
            temperature = (
                2.0
                * kinetic_energy
                / degrees_of_freedom
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
                total_energy - reference_total_energy
                if reference_total_energy is not None
                else 0.0
            )

            temperature_history.append(temperature)

    if progress_every is not None:
        print(" " * 70, end="\r")

    return (
        position_history,
        time_history,
        kinetic_energy_history,
        potential_energy_history,
        total_energy_history,
        energy_drift_history,
        temperature_history
    )
