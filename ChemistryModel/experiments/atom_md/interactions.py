import numpy as np


def lennard_jones_potential_at(
    distance,
    epsilon=1.0,
    sigma=1.0
):
    sigma_over_distance_6 = (sigma / distance) ** 6

    return (
        4.0
        * epsilon
        * (
            sigma_over_distance_6 ** 2
            - sigma_over_distance_6
        )
    )


def calculate_all_interactions_slow(
    particle_positions,
    box_size,
    epsilon=1.0,
    sigma=1.0,
    cutoff_distance=None
):
    # Kept only as a readable reference to test the fast
    # version against. Not used by the simulation.

    particle_count = len(particle_positions)

    total_forces = np.zeros_like(particle_positions)
    total_potential_energy = 0.0

    if cutoff_distance is None:
        potential_at_cutoff = 0.0
    else:
        potential_at_cutoff = lennard_jones_potential_at(
            cutoff_distance,
            epsilon,
            sigma
        )

    for particle_a_index in range(particle_count - 1):
        for particle_b_index in range(
            particle_a_index + 1,
            particle_count
        ):
            displacement = (
                particle_positions[particle_b_index]
                - particle_positions[particle_a_index]
            )

            displacement -= (
                box_size
                * np.round(displacement / box_size)
            )

            distance = np.linalg.norm(displacement)

            if distance == 0.0:
                raise ValueError(
                    "Two particles cannot occupy exactly "
                    "the same position."
                )

            if (
                cutoff_distance is not None
                and distance >= cutoff_distance
            ):
                continue

            sigma_over_distance_6 = (sigma / distance) ** 6
            sigma_over_distance_12 = sigma_over_distance_6 ** 2

            force_strength = (
                24.0
                * epsilon
                / distance
                * (
                    sigma_over_distance_6
                    - 2.0 * sigma_over_distance_12
                )
            )

            force_on_a = (
                displacement
                / distance
                * force_strength
            )

            total_forces[particle_a_index] += force_on_a
            total_forces[particle_b_index] -= force_on_a

            total_potential_energy += (
                4.0
                * epsilon
                * (
                    sigma_over_distance_12
                    - sigma_over_distance_6
                )
                - potential_at_cutoff
            )

    return total_forces, total_potential_energy


def calculate_all_interactions(
    particle_positions,
    box_size,
    epsilon=1.0,
    sigma=1.0,
    cutoff_distance=None
):
    # Works unchanged in any number of dimensions.
    # particle_positions has shape (particle_count, dimensions).

    particle_count = len(particle_positions)

    # displacements[i, j] is the vector FROM particle i TO particle j.

    particle_displacements = (
        particle_positions[np.newaxis, :, :]
        - particle_positions[:, np.newaxis, :]
    )

    # Minimum image convention.

    particle_displacements -= (
        box_size
        * np.round(particle_displacements / box_size)
    )

    distance_squared = np.sum(
        particle_displacements ** 2,
        axis=2
    )

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

    # Infinity on the diagonal removes self-interaction for free:
    # sigma^2 / inf is 0, so every term below vanishes there.

    distance_squared[self_interaction_mask] = np.inf

    if cutoff_distance is None:
        pairs_in_range = ~self_interaction_mask
        potential_at_cutoff = 0.0
    else:
        if cutoff_distance > box_size / 2.0:
            raise ValueError(
                f"Cutoff distance {cutoff_distance:.3f} exceeds half the "
                f"box size {box_size / 2.0:.3f}. The minimum image "
                f"convention would count the same neighbour twice."
            )

        pairs_in_range = distance_squared < cutoff_distance ** 2

        potential_at_cutoff = lennard_jones_potential_at(
            cutoff_distance,
            epsilon,
            sigma
        )

    sigma_over_distance_squared = (
        sigma ** 2 / distance_squared
    )

    distance_power_6 = sigma_over_distance_squared ** 3
    distance_power_12 = distance_power_6 ** 2

    # Force on i from j, expressed per unit displacement vector.

    force_coefficients = (
        24.0
        * epsilon
        * (
            distance_power_6
            - 2.0 * distance_power_12
        )
        / distance_squared
    )

    force_coefficients = np.where(
        pairs_in_range,
        force_coefficients,
        0.0
    )

    # Sum along axis 1: row i holds every displacement from i,
    # so the force on i is the sum across that row.

    total_forces = np.sum(
        force_coefficients[:, :, np.newaxis]
        * particle_displacements,
        axis=1
    )

    pair_potential_energies = (
        4.0
        * epsilon
        * (
            distance_power_12
            - distance_power_6
        )
        - potential_at_cutoff
    )

    pair_potential_energies = np.where(
        pairs_in_range,
        pair_potential_energies,
        0.0
    )

    # The full matrix counts every pair twice.

    total_potential_energy = 0.5 * np.sum(
        pair_potential_energies
    )

    return total_forces, total_potential_energy
