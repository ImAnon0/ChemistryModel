import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from argon import (
    ARGON_SIGMA_METERS,
    ARGON_EPSILON_OVER_KELVIN,
    ARGON_TIME_UNIT_SECONDS
)

from interactions import calculate_all_interactions
from simulation import run_simulation

# ============================================================
# Simulation settings
# ============================================================

box_size = 15.0
particle_mass = 1.0

time_step = 0.001
total_simulation_time = 8.0

number_of_steps = int(
    total_simulation_time / time_step
)

record_every = 8

# Change this to compare colder and hotter argon systems.
target_temperature_kelvin = 90

target_starting_temperature = (
    target_temperature_kelvin
    / ARGON_EPSILON_OVER_KELVIN
)


# ============================================================
# Initial particle state
# ============================================================

particle_positions = np.array(
    [
        [1.0, 1.0],
        [2.5, 1.0],
        [4.0, 1.0],

        [1.0, 2.5],
        [2.5, 2.5],
        [4.0, 2.5],

        [1.0, 4.0],
        [2.5, 4.0],
        [4.0, 4.0],

        [5.2, 5.2]
    ],
    dtype=float
)

particle_count = len(particle_positions)
number_of_dimensions = particle_positions.shape[1]

# Two centre-of-mass directions were removed below.
degrees_of_freedom = (
    particle_count * number_of_dimensions
    - number_of_dimensions
)


# Create repeatable random starting velocities.

random_generator = np.random.default_rng(seed=12)

particle_velocities = random_generator.normal(
    loc=0.0,
    scale=0.18,
    size=particle_positions.shape
)


# Remove overall centre-of-mass movement.

average_velocity = np.mean(
    particle_velocities,
    axis=0
)

particle_velocities -= average_velocity


# Scale the random velocities to the requested starting temperature.

current_kinetic_energy = (
    0.5
    * particle_mass
    * np.sum(particle_velocities ** 2)
)

current_temperature = (
    2.0
    * current_kinetic_energy
    / degrees_of_freedom
)

velocity_scale = np.sqrt(
    target_starting_temperature
    / current_temperature
)

particle_velocities *= velocity_scale


# ============================================================
# Run the simulation
# ============================================================

forces, potential_energy = calculate_all_interactions(
    particle_positions,
    box_size
)

initial_kinetic_energy = (
    0.5
    * particle_mass
    * np.sum(particle_velocities ** 2)
)

initial_total_energy = (
    initial_kinetic_energy + potential_energy
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

    new_forces, potential_energy = calculate_all_interactions(
        particle_positions,
        box_size
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
            kinetic_energy + potential_energy
        )

        energy_drift = (
            total_energy - initial_total_energy
        )

        current_time = (
            (step_number + 1) * time_step
        )

        position_history.append(
            particle_positions.copy()
        )

        time_history.append(current_time)
        kinetic_energy_history.append(kinetic_energy)
        potential_energy_history.append(potential_energy)
        total_energy_history.append(total_energy)
        energy_drift_history.append(energy_drift)
        temperature_history.append(temperature)


# ============================================================
# Animate the recorded simulation
# ============================================================

box_size_nanometers = (
    box_size
    * ARGON_SIGMA_METERS
    * 1e9
)

figure, axes = plt.subplots()

axes.set_xlim(0.0, box_size_nanometers)
axes.set_ylim(0.0, box_size_nanometers)
axes.set_aspect("equal")

axes.set_xlabel("X position (nm)")
axes.set_ylabel("Y position (nm)")
axes.set_title("Two-Dimensional Lennard-Jones Argon Model")
axes.grid()

particle_markers = axes.scatter(
    [],
    [],
    s=120
)

information_text = axes.text(
    0.02,
    0.98,
    "",
    transform=axes.transAxes,
    verticalalignment="top"
)


def update_animation(frame_number):
    current_positions_nanometers = (
        position_history[frame_number]
        * ARGON_SIGMA_METERS
        * 1e9
    )

    time_picoseconds = (
        time_history[frame_number]
        * ARGON_TIME_UNIT_SECONDS
        * 1e12
    )

    temperature_kelvin = (
        temperature_history[frame_number]
        * ARGON_EPSILON_OVER_KELVIN
    )

    particle_markers.set_offsets(
        current_positions_nanometers
    )

    information_text.set_text(
        f"Time: {time_picoseconds:.3f} ps\n"
        f"Temperature: {temperature_kelvin:.2f} K\n"
        f"Kinetic: "
        f"{kinetic_energy_history[frame_number]:.4f}\n"
        f"Potential: "
        f"{potential_energy_history[frame_number]:.4f}\n"
        f"Total: "
        f"{total_energy_history[frame_number]:.6f}\n"
        f"Energy drift: "
        f"{energy_drift_history[frame_number]:+.8f}"
    )

    return (
        particle_markers,
        information_text
    )


animation = FuncAnimation(
    figure,
    update_animation,
    frames=len(position_history),
    interval=10,
    blit=True,
    repeat=True
)

plt.show()
