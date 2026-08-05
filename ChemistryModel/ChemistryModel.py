import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from interactions import calculate_all_interactions

from argon import (
    ARGON_SIGMA_METERS,
    ARGON_EPSILON_OVER_KELVIN,
    ARGON_TIME_UNIT_SECONDS
)

# ============================================================
# Simulation settings
# ============================================================

box_size = 6.0
particle_mass = 1.0

time_step = 0.001
total_simulation_time = 8.0

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

(
    position_history,
    time_history,
    kinetic_energy_history,
    potential_energy_history,
    total_energy_history,
    energy_drift_history,
    temperature_history
) = run_simulation(
    starting_positions=particle_positions,
    starting_velocities=particle_velocities,
    particle_mass=particle_mass,
    box_size=box_size,
    time_step=time_step,
    total_simulation_time=total_simulation_time,
    record_every=record_every,
    degrees_of_freedom=degrees_of_freedom
)

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
