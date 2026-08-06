import numpy as np

from ase import units


class AseLiveSimulation:
    # Same interface as LiveSimulation, but the forces come from
    # an ASE calculator instead of the Lennard-Jones code, and
    # everything is in real units (Angstroms, eV, amu, Kelvin)
    # rather than argon-reduced units.
    #
    # Because the property names match, this object can be handed
    # straight to live.run_live_window with no changes to it.

    def __init__(
        self,
        atoms,
        target_temperature_kelvin=300.0,
        time_step_femtoseconds=0.5,
        friction_per_femtosecond=0.01,
        random_seed=12
    ):
        self.initial_atoms = atoms.copy()
        self.initial_calculator = atoms.calc

        self.time_step = time_step_femtoseconds * units.fs
        self.friction = friction_per_femtosecond / units.fs

        self._target_temperature_kelvin = target_temperature_kelvin

        self.random_seed = random_seed
        self.thermostat_is_on = True

        self.reset()

    # --------------------------------------------------------

    @property
    def target_temperature_kelvin(self):
        return self._target_temperature_kelvin

    @target_temperature_kelvin.setter
    def target_temperature_kelvin(self, value):
        self._target_temperature_kelvin = value

    @property
    def particle_positions(self):
        return self.atoms.get_positions()

    @property
    def degrees_of_freedom(self):
        return 3 * len(self.atoms) - 3

    @property
    def kinetic_energy(self):
        return 0.5 * np.sum(
            self.masses[:, np.newaxis]
            * self.velocities ** 2
        )

    @property
    def potential_energy(self):
        return self._potential_energy

    @property
    def total_energy(self):
        return self.kinetic_energy + self.potential_energy

    @property
    def temperature_kelvin(self):
        return (
            2.0
            * self.kinetic_energy
            / (self.degrees_of_freedom * units.kB)
        )

    @property
    def elapsed_picoseconds(self):
        return self.elapsed_time / (1000.0 * units.fs)

    @property
    def positions_in_nanometers(self):
        return self.atoms.get_positions() * 0.1

    @property
    def box_size_nanometers(self):
        return float(self.atoms.cell.lengths().max()) * 0.1

    # --------------------------------------------------------

    def reset(self):
        self.atoms = self.initial_atoms.copy()
        self.atoms.calc = self.initial_calculator

        self.masses = self.atoms.get_masses()

        self.random_generator = np.random.default_rng(
            seed=self.random_seed
        )

        # Maxwell-Boltzmann velocities at the target temperature.

        target_energy = (
            self._target_temperature_kelvin * units.kB
        )

        self.velocities = (
            self.random_generator.normal(
                size=(len(self.atoms), 3)
            )
            * np.sqrt(
                target_energy
                / self.masses
            )[:, np.newaxis]
        )

        self.velocities -= np.average(
            self.velocities,
            axis=0,
            weights=self.masses
        )

        self.forces = self.atoms.get_forces()
        self._potential_energy = self.atoms.get_potential_energy()

        self.elapsed_time = 0.0

    def step(self, number_of_steps=1):
        for _ in range(number_of_steps):
            accelerations = (
                self.forces
                / self.masses[:, np.newaxis]
            )

            new_positions = (
                self.atoms.get_positions()
                + self.velocities * self.time_step
                + 0.5 * accelerations * self.time_step ** 2
            )

            self.atoms.set_positions(new_positions)
            self.atoms.wrap()

            new_forces = self.atoms.get_forces()
            self._potential_energy = (
                self.atoms.get_potential_energy()
            )

            self.velocities += (
                0.5
                * (self.forces + new_forces)
                / self.masses[:, np.newaxis]
                * self.time_step
            )

            self.forces = new_forces

            if self.thermostat_is_on:
                self._apply_langevin()

            self.elapsed_time += self.time_step

    def _apply_langevin(self):
        decay = np.exp(-self.friction * self.time_step)

        target_energy = (
            self._target_temperature_kelvin * units.kB
        )

        noise_scale = np.sqrt(
            target_energy
            / self.masses
            * (1.0 - decay ** 2)
        )[:, np.newaxis]

        self.velocities = (
            self.velocities * decay
            + noise_scale
            * self.random_generator.normal(
                size=self.velocities.shape
            )
        )
