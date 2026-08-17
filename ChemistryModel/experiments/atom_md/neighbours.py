import numpy as np

from scipy.spatial import cKDTree


class NeighbourList:
    # Finds every pair of particles closer than the cutoff, without
    # ever building the full N x N matrix.
    #
    # A "skin" is added to the cutoff so the list stays valid for
    # several steps. It only needs rebuilding once any particle has
    # moved more than half the skin since the last build.

    def __init__(
        self,
        cutoff_distance,
        skin_distance=0.4
    ):
        self.cutoff_distance = cutoff_distance
        self.skin_distance = skin_distance

        self.reference_positions = None

        self.first_indices = None
        self.second_indices = None

        self.build_count = 0

    def needs_rebuild(self, particle_positions, box_size):
        if self.reference_positions is None:
            return True

        displacement = (
            particle_positions
            - self.reference_positions
        )

        displacement -= (
            box_size
            * np.round(displacement / box_size)
        )

        largest_movement = np.sqrt(
            np.max(np.sum(displacement ** 2, axis=1))
        )

        return largest_movement > 0.5 * self.skin_distance

    def build(self, particle_positions, box_size):
        wrapped_positions = particle_positions % box_size

        # boxsize makes the tree periodic, so it handles the
        # minimum image convention for us.

        tree = cKDTree(
            wrapped_positions,
            boxsize=box_size
        )

        pairs = tree.query_pairs(
            r=self.cutoff_distance + self.skin_distance,
            output_type="ndarray"
        )

        self.first_indices = pairs[:, 0]
        self.second_indices = pairs[:, 1]

        self.reference_positions = particle_positions.copy()

        self.build_count += 1

    def update(self, particle_positions, box_size):
        if self.needs_rebuild(particle_positions, box_size):
            self.build(particle_positions, box_size)

        return self.first_indices, self.second_indices


def calculate_interactions_with_neighbour_list(
    particle_positions,
    box_size,
    neighbour_list,
    epsilon=1.0,
    sigma=1.0
):
    from interactions import lennard_jones_potential_at

    first_indices, second_indices = neighbour_list.update(
        particle_positions,
        box_size
    )

    particle_count, number_of_dimensions = (
        particle_positions.shape
    )

    total_forces = np.zeros_like(particle_positions)

    if len(first_indices) == 0:
        return total_forces, 0.0

    displacements = (
        particle_positions[second_indices]
        - particle_positions[first_indices]
    )

    displacements -= (
        box_size
        * np.round(displacements / box_size)
    )

    distance_squared = np.sum(
        displacements ** 2,
        axis=1
    )

    cutoff_distance = neighbour_list.cutoff_distance

    within_cutoff = (
        distance_squared < cutoff_distance ** 2
    )

    displacements = displacements[within_cutoff]
    distance_squared = distance_squared[within_cutoff]

    active_first = first_indices[within_cutoff]
    active_second = second_indices[within_cutoff]

    sigma_over_distance_squared = (
        sigma ** 2 / distance_squared
    )

    distance_power_6 = sigma_over_distance_squared ** 3
    distance_power_12 = distance_power_6 ** 2

    force_coefficients = (
        24.0
        * epsilon
        * (
            distance_power_6
            - 2.0 * distance_power_12
        )
        / distance_squared
    )

    pair_forces = (
        force_coefficients[:, np.newaxis]
        * displacements
    )

    # Scatter each pair force onto both particles. Newton's third
    # law means the second particle gets the negative.

    for dimension in range(number_of_dimensions):
        total_forces[:, dimension] = (
            np.bincount(
                active_first,
                weights=pair_forces[:, dimension],
                minlength=particle_count
            )
            - np.bincount(
                active_second,
                weights=pair_forces[:, dimension],
                minlength=particle_count
            )
        )

    potential_at_cutoff = lennard_jones_potential_at(
        cutoff_distance,
        epsilon,
        sigma
    )

    total_potential_energy = np.sum(
        4.0
        * epsilon
        * (
            distance_power_12
            - distance_power_6
        )
        - potential_at_cutoff
    )

    return total_forces, total_potential_energy
