import numpy as np

from scipy.spatial import cKDTree

from particle_setup import (
    create_face_centred_cubic_lattice,
    create_thermal_velocities,
    count_degrees_of_freedom,
)
from argon import ARGON_EPSILON_OVER_KELVIN, ARGON_TIME_UNIT_SECONDS


class AtomSimulation:
    # CPU reference for TorchAtomSimulation. Same equations, same
    # conventions, so validate_atoms.py can compare the two.

    def __init__(self, unit_cells_per_side=8, number_density=0.85,
                 particle_mass=1.0, time_step=0.002,
                 cutoff_distance=2.5, skin_distance=0.6,
                 target_temperature=1.0, friction=1.0,
                 random_seed=12):
        self.unit_cells_per_side = int(unit_cells_per_side)
        self.number_density = float(number_density)
        self.particle_mass = float(particle_mass)
        self.time_step = float(time_step)
        self.cutoff_distance = float(cutoff_distance)
        self.skin_distance = float(skin_distance)
        self.target_temperature = float(target_temperature)
        self.friction = float(friction)
        self.random_seed = int(random_seed)
        self.thermostat_is_on = True
        self.coordination_cutoff = 1.4
        self.epsilon_over_kelvin = float(ARGON_EPSILON_OVER_KELVIN)
        self.time_unit_seconds = float(ARGON_TIME_UNIT_SECONDS)
        self.reset()

    @property
    def particle_count(self):
        return int(self.positions.shape[0])

    @property
    def volume(self):
        return self.box_size ** 3

    @property
    def density(self):
        return self.particle_count / self.volume

    @property
    def kinetic_energy(self):
        return 0.5 * self.particle_mass * np.sum(self.velocities ** 2)

    @property
    def potential_energy(self):
        return self._potential_energy

    @property
    def total_energy(self):
        return self.kinetic_energy + self.potential_energy

    @property
    def temperature(self):
        return 2.0 * self.kinetic_energy / max(self.degrees_of_freedom, 1)

    @property
    def temperature_kelvin(self):
        return self.temperature * self.epsilon_over_kelvin

    @property
    def pressure(self):
        return (
            self.density * self.temperature
            + self._virial / (3.0 * self.volume)
        )

    def reset(self):
        positions, self.box_size = create_face_centred_cubic_lattice(
            unit_cells_per_side=self.unit_cells_per_side,
            number_density=self.number_density,
        )

        self.degrees_of_freedom = count_degrees_of_freedom(positions)

        self.positions = positions
        self.velocities = create_thermal_velocities(
            particle_positions=positions,
            particle_mass=self.particle_mass,
            target_temperature=self.target_temperature,
            random_seed=self.random_seed,
        )

        self.random_generator = np.random.default_rng(self.random_seed)

        self.unwrapped = positions.copy()
        self.msd_reference = positions.copy()
        self.msd_reference_time = 0.0

        self.reference_positions = None
        self.pair_first = np.array([], dtype=int)
        self.pair_second = np.array([], dtype=int)
        self.rebuild_count = 0
        self.elapsed_time = 0.0

        self.build_pairs(force=True)
        self.forces, self._potential_energy = self.compute_forces()

    def minimum_image(self, displacement):
        return displacement - self.box_size * np.round(
            displacement / self.box_size
        )

    def build_pairs(self, force=False):
        if force or self.reference_positions is None:
            rebuild = True
        else:
            displacement = self.minimum_image(
                self.positions - self.reference_positions
            )
            max_move = np.sqrt(
                np.max(np.sum(displacement ** 2, axis=1))
            )
            rebuild = max_move > 0.5 * self.skin_distance

        if not rebuild:
            return False

        tree = cKDTree(
            self.positions % self.box_size,
            boxsize=self.box_size
        )

        pairs = tree.query_pairs(
            self.cutoff_distance + self.skin_distance,
            output_type="ndarray"
        )

        if len(pairs) == 0:
            self.pair_first = np.array([], dtype=int)
            self.pair_second = np.array([], dtype=int)
        else:
            self.pair_first = pairs[:, 0]
            self.pair_second = pairs[:, 1]

        self.reference_positions = self.positions.copy()
        self.rebuild_count += 1

        return True

    def compute_forces(self):
        forces = np.zeros_like(self.positions)

        count = len(self.positions)

        self.per_atom_energy = np.zeros(count)
        self.coordination = np.zeros(count)
        self._virial = 0.0

        if len(self.pair_first) == 0:
            return forces, 0.0

        displacement = self.minimum_image(
            self.positions[self.pair_second]
            - self.positions[self.pair_first]
        )

        r2 = np.maximum(np.sum(displacement ** 2, axis=1), 1e-12)
        active = r2 < self.cutoff_distance ** 2

        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2

        coefficient = 24.0 * (inv_r6 - 2.0 * inv_r12) * inv_r2
        coefficient = np.where(active, coefficient, 0.0)

        pair_force = coefficient[:, None] * displacement

        for dimension in range(3):
            forces[:, dimension] = (
                np.bincount(self.pair_first,
                            weights=pair_force[:, dimension],
                            minlength=count)
                - np.bincount(self.pair_second,
                              weights=pair_force[:, dimension],
                              minlength=count)
            )

        cutoff = self.cutoff_distance
        cutoff_energy = 4.0 * ((1.0 / cutoff) ** 12 - (1.0 / cutoff) ** 6)

        pair_energy = 4.0 * (inv_r12 - inv_r6) - cutoff_energy
        pair_energy = np.where(active, pair_energy, 0.0)

        self._virial = -np.sum(displacement * pair_force)

        half_energy = 0.5 * pair_energy

        self.per_atom_energy = (
            np.bincount(self.pair_first, weights=half_energy,
                        minlength=count)
            + np.bincount(self.pair_second, weights=half_energy,
                          minlength=count)
        )

        contacts = (
            r2 < self.coordination_cutoff ** 2
        ).astype(float)

        self.coordination = (
            np.bincount(self.pair_first, weights=contacts,
                        minlength=count)
            + np.bincount(self.pair_second, weights=contacts,
                          minlength=count)
        )

        return forces, np.sum(pair_energy)

    def step(self, number_of_steps=1):
        dt = self.time_step
        mass = self.particle_mass

        for _ in range(int(number_of_steps)):
            acceleration = self.forces / mass

            movement = (
                self.velocities * dt + 0.5 * acceleration * dt * dt
            )

            self.unwrapped = self.unwrapped + movement
            self.positions = (self.positions + movement) % self.box_size

            self.build_pairs()

            new_forces, potential = self.compute_forces()

            self.velocities = (
                self.velocities
                + 0.5 * (self.forces + new_forces) / mass * dt
            )

            self.forces = new_forces
            self._potential_energy = potential

            if self.thermostat_is_on:
                self._apply_langevin()

            self.elapsed_time += dt

    def _apply_langevin(self):
        decay = np.exp(-self.friction * self.time_step)

        noise_scale = np.sqrt(
            self.target_temperature * (1.0 - decay ** 2)
            / self.particle_mass
        )

        self.velocities = (
            self.velocities * decay
            + noise_scale
            * self.random_generator.normal(size=self.velocities.shape)
        )

        self.velocities -= np.mean(self.velocities, axis=0)

    def reset_msd(self):
        self.msd_reference = self.unwrapped.copy()
        self.msd_reference_time = self.elapsed_time

    @property
    def mean_squared_displacement(self):
        offset = self.unwrapped - self.msd_reference
        return float(np.mean(np.sum(offset ** 2, axis=1)))

    @property
    def diffusion_coefficient(self):
        elapsed = self.elapsed_time - self.msd_reference_time

        if elapsed < 1e-9:
            return 0.0

        return self.mean_squared_displacement / (6.0 * elapsed)

    def radial_distribution(self, bin_count=120, maximum=None):
        if maximum is None:
            maximum = self.cutoff_distance + self.skin_distance

        if len(self.pair_first) == 0:
            return np.linspace(0.0, maximum, bin_count), np.zeros(bin_count)

        displacement = self.minimum_image(
            self.positions[self.pair_second]
            - self.positions[self.pair_first]
        )

        distances = np.sqrt(np.sum(displacement ** 2, axis=1))

        counts, edges = np.histogram(
            distances, bins=bin_count, range=(0.0, maximum)
        )

        centres = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]

        shell_volume = 4.0 * np.pi * centres ** 2 * width

        expected = (
            0.5 * self.particle_count * self.density * shell_volume
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(expected > 0, counts / expected, 0.0)

        return centres, g

    def set_phase(self, number_density, target_temperature):
        self.number_density = float(number_density)
        self.target_temperature = float(target_temperature)
        self.reset()

    def scale_velocities_to(self, target_temperature):
        current = self.temperature

        if current <= 0.0:
            return

        self.velocities = self.velocities * float(
            np.sqrt(target_temperature / current)
        )

    @property
    def target_temperature_kelvin(self):
        return self.target_temperature * self.epsilon_over_kelvin

    @target_temperature_kelvin.setter
    def target_temperature_kelvin(self, value):
        self.target_temperature = float(value) / self.epsilon_over_kelvin

    @property
    def elapsed_picoseconds(self):
        return self.elapsed_time * self.time_unit_seconds * 1e12

    @property
    def neighbour_pair_count(self):
        return len(self.pair_first)
