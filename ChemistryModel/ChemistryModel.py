import numpy as np

from simulation import run_simulation

from visualisation import show_animation

from argon import ARGON_EPSILON_OVER_KELVIN

from particle_setup import create_square_grid

# ============================================================
# Simulation settings
# ============================================================

box_size = 6.0
particle_mass = 1.0

time_step = 0.001
total_simulation_time = 8.0

record_every = 8

# Change this to compare colder and hotter argon systems.
target_temperature_kelvin = 0

target_starting_temperature = (
    target_temperature_kelvin
    / ARGON_EPSILON_OVER_KELVIN
)


# ============================================================
# Initial particle state
# ============================================================

particle_positions = create_square_grid(
    particles_per_side=5,
    box_size=box_size
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

show_animation(
    position_history=position_history,
    time_history=time_history,
    kinetic_energy_history=kinetic_energy_history,
    potential_energy_history=potential_energy_history,
    total_energy_history=total_energy_history,
    energy_drift_history=energy_drift_history,
    temperature_history=temperature_history,
    box_size=box_size
)