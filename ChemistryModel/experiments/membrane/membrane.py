import numpy as np

from scipy.spatial import cKDTree


# ============================================================
# Coarse-grained amphiphile model
# ============================================================
#
# Each lipid is three beads in a line:
#
#     HEAD - TAIL - TAIL
#
# joined by FENE springs, with a stiffening spring between the
# two ends to stop it folding back on itself.
#
# There is no water in this model. Instead of simulating solvent
# explicitly, the hydrophobic effect is folded into a direct
# attraction between tail beads. This is the Cooke-Deserno
# solvent-free model, and it is what makes the whole thing fast
# enough to watch: roughly 95 percent of a normal membrane
# simulation is spent pushing water around.
#
# Nothing below knows what a bilayer is. Heads repel, tails
# stick together, and the geometry does the rest.

def scatter_add(target, indices, values):
    # np.add.at is correct but famously slow. bincount does the
    # same accumulation an order of magnitude faster.

    count = len(target)

    for dimension in range(target.shape[1]):
        target[:, dimension] += np.bincount(
            indices,
            weights=values[:, dimension],
            minlength=count
        )


HEAD = 0
TAIL = 1

BEADS_PER_LIPID = 3


class AmphiphileModel:

    def __init__(
        self,
        number_of_lipids,
        box_size,
        epsilon=1.0,
        sigma=1.0,
        head_sigma_factor=0.95,
        attraction_range=1.6,
        fene_stiffness=30.0,
        fene_maximum_extension=1.5,
        bending_stiffness=10.0,
        bending_rest_length=4.0
    ):
        self.number_of_lipids = number_of_lipids
        self.box_size = float(box_size)

        self.epsilon = epsilon
        self.sigma = sigma

        # Heads are given a slightly smaller effective diameter so
        # that head-head and head-tail contacts are purely
        # repulsive while tail-tail contacts can attract.

        head_sigma = head_sigma_factor * sigma

        self.sigma_matrix = np.array([
            [head_sigma, head_sigma],
            [head_sigma, sigma]
        ])

        self.attraction_range = attraction_range

        self.fene_stiffness = fene_stiffness
        self.fene_maximum_extension = fene_maximum_extension

        self.bending_stiffness = bending_stiffness
        self.bending_rest_length = bending_rest_length

        self.bead_count = number_of_lipids * BEADS_PER_LIPID

        self.bead_types = np.tile(
            np.array([HEAD, TAIL, TAIL]),
            number_of_lipids
        )

        self.molecule_index = np.repeat(
            np.arange(number_of_lipids),
            BEADS_PER_LIPID
        )

        # Bonded pairs, as index arrays.

        lipid_starts = np.arange(number_of_lipids) * BEADS_PER_LIPID

        self.bond_first = np.concatenate([
            lipid_starts,
            lipid_starts + 1
        ])

        self.bond_second = np.concatenate([
            lipid_starts + 1,
            lipid_starts + 2
        ])

        self.bend_first = lipid_starts
        self.bend_second = lipid_starts + 2

        # The longest range any non-bonded force reaches.

        self.interaction_cutoff = (
            2.0 ** (1.0 / 6.0) * sigma
            + attraction_range
        )

        self.masses = np.ones(self.bead_count)

    # --------------------------------------------------------

    def minimum_image(self, displacements):
        return displacements - self.box_size * np.round(
            displacements / self.box_size
        )

    def random_configuration(self, random_generator):
        # Lipids dropped in at random positions and random
        # orientations, well separated, with no structure at all.

        positions = np.zeros((self.bead_count, 3))

        bond_length = 0.95 * self.sigma

        # Lipids go on a loose lattice rather than at uniformly
        # random points. Two lipids dropped at random will
        # sometimes overlap, and an overlap inside the repulsive
        # core produces a force large enough to blow the
        # integrator apart on the very first step.

        sites_per_side = int(
            np.ceil(self.number_of_lipids ** (1.0 / 3.0))
        )

        spacing = self.box_size / sites_per_side

        grid = np.stack(
            np.meshgrid(
                *(np.arange(sites_per_side),) * 3,
                indexing="ij"
            ),
            axis=-1
        ).reshape(-1, 3)

        chosen = random_generator.choice(
            len(grid),
            size=self.number_of_lipids,
            replace=False
        )

        centres = (grid[chosen] + 0.5) * spacing

        centres += random_generator.uniform(
            -0.15 * spacing,
            0.15 * spacing,
            size=centres.shape
        )

        directions = random_generator.normal(
            size=(self.number_of_lipids, 3)
        )

        directions /= np.linalg.norm(
            directions,
            axis=1,
            keepdims=True
        )

        for bead in range(BEADS_PER_LIPID):
            positions[bead::BEADS_PER_LIPID] = (
                centres
                + directions * bond_length * bead
            )

        return positions % self.box_size

    def vesicle_configuration(self, random_generator, radius=None):
        # A pre-formed sphere, for when you want to start from a
        # membrane and break it rather than wait for one to build.

        if radius is None:
            area_per_lipid = 1.25
            radius = np.sqrt(
                self.number_of_lipids
                * area_per_lipid
                / (8.0 * np.pi)
            )

        positions = np.zeros((self.bead_count, 3))

        centre = np.full(3, self.box_size / 2.0)

        inner_count = self.number_of_lipids // 2
        outer_count = self.number_of_lipids - inner_count

        bond_length = 0.95 * self.sigma
        half_thickness = bond_length * 2.5

        def fibonacci_sphere(count):
            index = np.arange(count) + 0.5
            phi = np.arccos(1.0 - 2.0 * index / count)
            theta = np.pi * (1.0 + 5.0 ** 0.5) * index

            return np.stack([
                np.cos(theta) * np.sin(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(phi)
            ], axis=1)

        outer_directions = fibonacci_sphere(outer_count)
        inner_directions = fibonacci_sphere(inner_count)

        lipid = 0

        for direction in outer_directions:
            head = centre + direction * (radius + half_thickness)
            for bead in range(BEADS_PER_LIPID):
                positions[lipid * BEADS_PER_LIPID + bead] = (
                    head - direction * bond_length * bead
                )
            lipid += 1

        for direction in inner_directions:
            head = centre + direction * max(
                radius - half_thickness,
                1.0
            )
            for bead in range(BEADS_PER_LIPID):
                positions[lipid * BEADS_PER_LIPID + bead] = (
                    head + direction * bond_length * bead
                )
            lipid += 1

        jitter = random_generator.normal(
            scale=0.05,
            size=positions.shape
        )

        return (positions + jitter) % self.box_size

    # --------------------------------------------------------

    def bonded_forces(self, positions):
        forces = np.zeros_like(positions)
        energy = 0.0

        # FENE springs along the lipid.

        displacements = self.minimum_image(
            positions[self.bond_second]
            - positions[self.bond_first]
        )

        distances = np.linalg.norm(displacements, axis=1)

        extension_ratio = (
            distances / self.fene_maximum_extension
        )

        # Guard against a bond being stretched past the FENE
        # divergence, which would produce infinities.

        extension_ratio = np.clip(extension_ratio, 0.0, 0.98)

        denominator = 1.0 - extension_ratio ** 2

        force_magnitude = (
            self.fene_stiffness
            * distances
            / denominator
        )

        unit_vectors = displacements / distances[:, None]
        pair_forces = unit_vectors * force_magnitude[:, None]

        scatter_add(forces, self.bond_first, pair_forces)
        scatter_add(forces, self.bond_second, -pair_forces)

        energy += np.sum(
            -0.5
            * self.fene_stiffness
            * self.fene_maximum_extension ** 2
            * np.log(denominator)
        )

        # Stiffening spring between the two ends of each lipid.

        bend_displacements = self.minimum_image(
            positions[self.bend_second]
            - positions[self.bend_first]
        )

        bend_distances = np.linalg.norm(
            bend_displacements,
            axis=1
        )

        stretch = bend_distances - self.bending_rest_length

        bend_magnitude = self.bending_stiffness * stretch

        bend_units = (
            bend_displacements / bend_distances[:, None]
        )

        bend_forces = bend_units * bend_magnitude[:, None]

        scatter_add(forces, self.bend_first, bend_forces)
        scatter_add(forces, self.bend_second, -bend_forces)

        energy += np.sum(
            0.5 * self.bending_stiffness * stretch ** 2
        )

        return forces, energy

    def nonbonded_forces(self, positions, pair_first, pair_second):
        forces = np.zeros_like(positions)

        if len(pair_first) == 0:
            return forces, 0.0

        displacements = self.minimum_image(
            positions[pair_second]
            - positions[pair_first]
        )

        distance_squared = np.sum(displacements ** 2, axis=1)

        # A hard floor stops a bad starting configuration from
        # producing an infinite force on the first step.

        distance_squared = np.maximum(distance_squared, 1e-4)
        distances = np.sqrt(distance_squared)

        type_first = self.bead_types[pair_first]
        type_second = self.bead_types[pair_second]

        pair_sigma = self.sigma_matrix[type_first, type_second]

        repulsive_cutoff = 2.0 ** (1.0 / 6.0) * pair_sigma

        force_magnitude = np.zeros_like(distances)
        energy = 0.0

        # Weeks-Chandler-Andersen: Lennard-Jones truncated at its
        # minimum and shifted up, so it is purely repulsive.

        in_core = distances < repulsive_cutoff

        if np.any(in_core):
            sigma_over_r = (
                pair_sigma[in_core] / distances[in_core]
            )

            power_6 = sigma_over_r ** 6
            power_12 = power_6 ** 2

            force_magnitude[in_core] += (
                24.0
                * self.epsilon
                * (2.0 * power_12 - power_6)
                / distances[in_core]
            )

            energy += np.sum(
                4.0
                * self.epsilon
                * (power_12 - power_6)
                + self.epsilon
            )

        # Tail-tail attraction. This single term is standing in
        # for every water molecule that is not in the simulation.

        both_tails = (
            (type_first == TAIL)
            & (type_second == TAIL)
        )

        flat_region = both_tails & in_core

        energy += -self.epsilon * np.count_nonzero(flat_region)

        tail_cutoff = repulsive_cutoff + self.attraction_range

        in_tail_well = (
            both_tails
            & (distances >= repulsive_cutoff)
            & (distances < tail_cutoff)
        )

        if np.any(in_tail_well):
            phase = (
                np.pi
                * (
                    distances[in_tail_well]
                    - repulsive_cutoff[in_tail_well]
                )
                / (2.0 * self.attraction_range)
            )

            energy += np.sum(
                -self.epsilon * np.cos(phase) ** 2
            )

            force_magnitude[in_tail_well] += (
                -self.epsilon
                * np.pi
                * np.sin(2.0 * phase)
                / (2.0 * self.attraction_range)
            )

        unit_vectors = displacements / distances[:, None]
        pair_forces = unit_vectors * force_magnitude[:, None]

        # Force on the first particle points away from the second
        # when the term is repulsive, so the sign convention here
        # matches the displacement direction.

        scatter_add(forces, pair_first, -pair_forces)
        scatter_add(forces, pair_second, pair_forces)

        return forces, energy

    # --------------------------------------------------------

    def build_pairs(self, positions):
        # Bonded neighbours inside a lipid are excluded, since
        # their interaction is already handled by the springs.

        tree = cKDTree(
            positions % self.box_size,
            boxsize=self.box_size
        )

        pairs = tree.query_pairs(
            r=self.interaction_cutoff,
            output_type="ndarray"
        )

        if len(pairs) == 0:
            return pairs[:, 0] if pairs.size else np.array([], dtype=int), \
                   pairs[:, 1] if pairs.size else np.array([], dtype=int)

        first = pairs[:, 0]
        second = pairs[:, 1]

        same_molecule = (
            self.molecule_index[first]
            == self.molecule_index[second]
        )

        keep = ~same_molecule

        return first[keep], second[keep]

    def forces_and_energy(self, positions, pair_first, pair_second):
        bonded, bonded_energy = self.bonded_forces(positions)

        nonbonded, nonbonded_energy = self.nonbonded_forces(
            positions,
            pair_first,
            pair_second
        )

        return (
            bonded + nonbonded,
            bonded_energy + nonbonded_energy
        )
