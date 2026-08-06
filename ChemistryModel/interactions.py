import numpy as np


def calculate_lennard_jones_pair(
    position_a,
    position_b,
    box_size,
    epsilon=1.0,
    sigma=1.0
):
    displacement_from_a_to_b = position_b - position_a

    displacement_from_a_to_b -= (
        box_size
        * np.round(displacement_from_a_to_b / box_size)
    )

    distance = np.linalg.norm(displacement_from_a_to_b)

    if distance == 0:
        raise ValueError(
            "Two particles cannot occupy exactly the same position."
        )

    direction_from_a_to_b = (
        displacement_from_a_to_b / distance
    )

    sigma_over_distance = sigma / distance

    distance_power_6 = sigma_over_distance ** 6
    distance_power_12 = sigma_over_distance ** 12

    force_strength_on_a = (
        24.0
        * epsilon
        / distance
        * (
            distance_power_6
            - 2.0 * distance_power_12
        )
    )

    force_on_a = (
        direction_from_a_to_b
        * force_strength_on_a
    )

    potential_energy = (
        4.0
        * epsilon
        * (
            distance_power_12
            - distance_power_6
        )
    )

    return force_on_a, potential_energy


def calculate_all_interactions_slow(
    particle_positions,
    box_size,
    epsilon=1.0,
    sigma=1.0
):
    particle_count = len(particle_positions)

    total_forces = np.zeros_like(
        particle_positions
    )

    total_potential_energy = 0.0

    for particle_a_index in range(particle_count - 1):
        for particle_b_index in range(
            particle_a_index + 1,
            particle_count
        ):
            force_on_a, pair_potential_energy = (
                calculate_lennard_jones_pair(
                    particle_positions[particle_a_index],
                    particle_positions[particle_b_index],
                    box_size,
                    epsilon,
                    sigma
                )
            )

            total_forces[particle_a_index] += force_on_a
            total_forces[particle_b_index] -= force_on_a

            total_potential_energy += (
                pair_potential_energy
            )

    return total_forces, total_potential_energy

def calculate_all_interactions(
    particle_positions,
    box_size,
    epsilon=1.0,
    sigma=1.0
):
    particle_displacements = (
        particle_positions[np.newaxis, :, :]
        - particle_positions[:, np.newaxis, :]
    )

    particle_displacements -= (
        box_size
        * np.round(particle_displacements / box_size)
    )

    distance_squared = np.sum(
        particle_displacements ** 2,
        axis=2
    )

    particle_count = len(particle_positions)

    self_interaction_mask = np.eye(
        particle_count,
        dtype=bool
    )

    overlapping_particles = (
        (distance_squared == 0.0)
        & ~self_interaction_mask
    )

    if np.any(overlapping_particles):
        raise ValueError(
            "Two particles cannot occupy exactly the same position."
        )

    distance_squared[self_interaction_mask] = np.inf

    return calculate_all_interactions_slow(
        particle_positions,
        box_size,
        epsilon,
        sigma
    )