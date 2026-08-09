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
        # The usual -3 assumes centre-of-mass momentum is
        # conserved. Langevin kicks each atom independently, so it
        # puts COM drift straight back every step, and that motion
        # is thermal like everything else. Subtracting it while
        # the thermostat runs biases the reported temperature.

        if self.thermostat_is_on:
            return 3 * len(self.atoms)

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

        self.diverged = False
        self.divergence_reason = ""

    def step(self, number_of_steps=1):
        for _ in range(number_of_steps):
            self._check_stability()

            if self.diverged:
                return

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

    # Any run this far above target has diverged, not heated up.
    # Once velocities run away the readout shows numbers like
    # 1e33 K, which is a broken integrator rather than a result.

    RUNAWAY_TEMPERATURE_FACTOR = 20.0

    def _check_stability(self):
        # Sets a flag rather than raising, so the live window can
        # freeze the run and show the reason instead of the whole
        # program dying with a traceback. Anything driving this
        # class headlessly should check `diverged` after stepping.

        if self.diverged:
            return

        if not np.all(np.isfinite(self.forces)):
            self.diverged = True

            self.divergence_reason = (
                "Forces came back NaN or infinite. The calculator "
                "has been pushed outside anything it can evaluate."
                + self._divergence_advice()
            )

            return

        runaway_limit = (
            self.RUNAWAY_TEMPERATURE_FACTOR
            * max(self._target_temperature_kelvin, 1.0)
        )

        if self.temperature_kelvin > runaway_limit:
            self.diverged = True

            self.divergence_reason = (
                f"Temperature reached "
                f"{self.temperature_kelvin:.3e} K against a "
                f"target of {self._target_temperature_kelvin:.0f} "
                f"K. This is numerical divergence, not heating."
                + self._divergence_advice()
            )

    def _divergence_advice(self):
        # A neural network potential has no analytic repulsive
        # core, so outside its training data it returns arbitrary
        # forces rather than large ones. A single bad force gives
        # a huge velocity, which lands the next step further out
        # of domain still. That feedback loop is what this catches.

        return (
            "\n"
            "\n  Likely causes, in order:"
            "\n    1. Free atoms or radicals in the box. MACE-OFF"
            "\n       is trained on neutral closed-shell molecules,"
            "\n       so a lone C, N or O atom is outside anything"
            "\n       it has seen, at any temperature. Start from"
            "\n       whole molecules, not scattered atoms."
            "\n    2. Temperature too high. Above roughly 600 K"
            "\n       bonds dissociate into radicals, which is the"
            "\n       same problem arriving by another route."
            "\n       Try 500 K."
            "\n    3. Molecules placed overlapping. Check the"
            "\n       closest contact printed at startup is above"
            "\n       about 2 A."
            "\n    4. Timestep too large. Try 1.0 fs with"
            "\n       hydrogen_mass = 3.0, or 0.5 fs with None."
            "\n"
            "\n  A neural network potential has no analytic"
            "\n  repulsive core, so outside its training data it"
            "\n  returns arbitrary forces rather than large ones."
            "\n  One bad force gives a huge velocity, which lands"
            "\n  the next step further out of domain still."
        )

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