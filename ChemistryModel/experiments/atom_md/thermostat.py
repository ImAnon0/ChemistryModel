import numpy as np


def calculate_temperature(
    particle_velocities,
    particle_mass,
    degrees_of_freedom
):
    kinetic_energy = (
        0.5
        * particle_mass
        * np.sum(particle_velocities ** 2)
    )

    return (
        2.0
        * kinetic_energy
        / degrees_of_freedom
    )


class BerendsenThermostat:
    # Rescales every velocity by the same factor, nudging the
    # temperature towards the target over a chosen coupling time.
    #
    # Simple, stable, and good for getting a system to the
    # temperature you asked for. It does NOT produce correct
    # canonical fluctuations, so use it to equilibrate and then
    # switch it off before measuring anything.

    def __init__(
        self,
        target_temperature,
        coupling_time=0.1,
        maximum_scaling=1.1
    ):
        self.target_temperature = target_temperature
        self.coupling_time = coupling_time
        self.maximum_scaling = maximum_scaling

    def apply(
        self,
        particle_velocities,
        time_step,
        particle_mass,
        degrees_of_freedom
    ):
        current_temperature = calculate_temperature(
            particle_velocities,
            particle_mass,
            degrees_of_freedom
        )

        if current_temperature <= 0.0:
            return particle_velocities

        temperature_ratio = (
            self.target_temperature
            / current_temperature
        )

        scaling_factor_squared = 1.0 + (
            time_step
            / self.coupling_time
            * (temperature_ratio - 1.0)
        )

        # A cold start or a sudden burst of bonding energy can
        # make this negative. Clamping keeps the square root real
        # and stops a single step from rescaling everything wildly.

        scaling_factor_squared = max(
            scaling_factor_squared,
            0.0
        )

        scaling_factor = np.sqrt(scaling_factor_squared)

        scaling_factor = min(
            scaling_factor,
            self.maximum_scaling
        )

        scaling_factor = max(
            scaling_factor,
            1.0 / self.maximum_scaling
        )

        return particle_velocities * scaling_factor


class LangevinThermostat:
    # Adds friction plus matching random kicks, so each particle
    # is coupled to the heat bath individually rather than the
    # whole system being scaled at once.
    #
    # This one DOES sample the correct canonical distribution, and
    # it copes far better with heat released locally by bond
    # formation. Slower to equilibrate than Berendsen.

    def __init__(
        self,
        target_temperature,
        friction=1.0,
        random_seed=None
    ):
        self.target_temperature = target_temperature
        self.friction = friction

        self.random_generator = np.random.default_rng(
            seed=random_seed
        )

    def apply(
        self,
        particle_velocities,
        time_step,
        particle_mass,
        degrees_of_freedom
    ):
        decay = np.exp(
            -self.friction * time_step
        )

        noise_scale = np.sqrt(
            self.target_temperature
            / particle_mass
            * (1.0 - decay ** 2)
        )

        random_kicks = self.random_generator.normal(
            loc=0.0,
            scale=1.0,
            size=particle_velocities.shape
        )

        return (
            particle_velocities * decay
            + noise_scale * random_kicks
        )
