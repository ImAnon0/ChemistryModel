import numpy as np
import torch
from scipy.spatial import cKDTree

from particle_setup import (
    create_face_centred_cubic_lattice,
    create_thermal_velocities,
    count_degrees_of_freedom,
)
from argon import ARGON_EPSILON_OVER_KELVIN, ARGON_TIME_UNIT_SECONDS


class TorchAtomSimulation:
    """Torch/vectorised Lennard-Jones atom simulation with a Verlet neighbour list."""

    def __init__(self, unit_cells_per_side=8, number_density=0.85,
                 particle_mass=1.0, time_step=0.002,
                 cutoff_distance=2.5, skin_distance=0.6,
                 target_temperature=1.0, friction=1.0, device=None,
                 dtype=torch.float32, random_seed=12):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
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
        self.reference_positions = None
        self.pair_first = None
        self.pair_second = None
        self.rebuild_count = 0
        self.steps_taken = 0
        self.elapsed_time = 0.0
        self.reset()

    @property
    def particle_count(self):
        return int(self.positions.shape[0])

    @property
    def positions_numpy(self):
        return self.positions.detach().cpu().numpy()

    @property
    def particle_positions(self):
        return self.positions_numpy

    @property
    def temperature(self):
        return 2.0 * self.kinetic_energy / max(self.degrees_of_freedom, 1)

    @property
    def temperature_kelvin(self):
        return self.temperature * self.epsilon_over_kelvin

    @property
    def target_temperature_kelvin(self):
        return self.target_temperature * self.epsilon_over_kelvin

    @target_temperature_kelvin.setter
    def target_temperature_kelvin(self, value):
        self.target_temperature = float(value) / self.epsilon_over_kelvin

    @property
    def kinetic_energy(self):
        return float(0.5 * self.particle_mass * torch.sum(self.velocities ** 2))

    @property
    def potential_energy(self):
        return float(self._potential_energy)

    @property
    def total_energy(self):
        return self.kinetic_energy + self.potential_energy

    @property
    def elapsed_picoseconds(self):
        return self.elapsed_time * self.time_unit_seconds * 1e12

    @property
    def neighbour_pair_count(self):
        return 0 if self.pair_first is None else int(self.pair_first.numel())

    @property
    def neighbour_rebuild_count(self):
        return int(self.rebuild_count)

    def reset(self):
        positions, self.box_size = create_face_centred_cubic_lattice(
            unit_cells_per_side=self.unit_cells_per_side,
            number_density=self.number_density,
        )
        self.degrees_of_freedom = count_degrees_of_freedom(positions)
        velocities = create_thermal_velocities(
            particle_positions=positions,
            particle_mass=self.particle_mass,
            target_temperature=self.target_temperature,
            random_seed=self.random_seed,
        )
        self.positions = torch.as_tensor(positions, dtype=self.dtype, device=self.device)
        self.velocities = torch.as_tensor(velocities, dtype=self.dtype, device=self.device)
        self.reference_positions = None
        self.pair_first = torch.empty(0, dtype=torch.long, device=self.device)
        self.pair_second = torch.empty(0, dtype=torch.long, device=self.device)
        self.rebuild_count = 0
        self.steps_taken = 0
        self.elapsed_time = 0.0
        self.unwrapped = self.positions.clone()
        self.msd_reference = self.positions.clone()
        self.msd_reference_time = 0.0

        self.build_pairs(force=True)
        self.forces, self._potential_energy = self.compute_forces()

    def minimum_image(self, displacement):
        return displacement - self.box_size * torch.round(displacement / self.box_size)

    def build_pairs(self, force=False):
        if force or self.reference_positions is None:
            rebuild = True
        else:
            displacement = self.minimum_image(self.positions - self.reference_positions)
            max_move = float(torch.sqrt(torch.max(torch.sum(displacement ** 2, dim=1))))
            rebuild = max_move > 0.5 * self.skin_distance
        if not rebuild:
            return False

        positions_cpu = self.positions.detach().cpu().numpy() % self.box_size
        tree = cKDTree(positions_cpu, boxsize=self.box_size)
        pairs = tree.query_pairs(self.cutoff_distance + self.skin_distance,
                                 output_type="ndarray")
        if len(pairs) == 0:
            self.pair_first = torch.empty(0, dtype=torch.long, device=self.device)
            self.pair_second = torch.empty(0, dtype=torch.long, device=self.device)
        else:
            self.pair_first = torch.as_tensor(pairs[:, 0], dtype=torch.long, device=self.device)
            self.pair_second = torch.as_tensor(pairs[:, 1], dtype=torch.long, device=self.device)
        self.reference_positions = self.positions.clone()
        self.rebuild_count += 1
        return True

    def compute_forces(self):
        forces = torch.zeros_like(self.positions)

        count = self.positions.shape[0]

        self.per_atom_energy = torch.zeros(
            count, dtype=self.dtype, device=self.device)
        self.coordination = torch.zeros(
            count, dtype=self.dtype, device=self.device)
        self._virial = 0.0

        if self.pair_first.numel() == 0:
            return forces, torch.zeros((), dtype=self.dtype, device=self.device)

        displacement = self.minimum_image(
            self.positions[self.pair_second] - self.positions[self.pair_first]
        )
        r2 = torch.clamp(torch.sum(displacement * displacement, dim=1), min=1e-12)
        active = r2 < self.cutoff_distance ** 2

        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2

        # Same reduced-unit Lennard-Jones force as interactions.py.
        coefficient = 24.0 * (inv_r6 - 2.0 * inv_r12) * inv_r2
        coefficient = torch.where(active, coefficient, torch.zeros_like(coefficient))
        pair_force = coefficient[:, None] * displacement

        forces.index_add_(0, self.pair_first, pair_force)
        forces.index_add_(0, self.pair_second, -pair_force)

        cutoff = self.cutoff_distance
        cutoff_energy = 4.0 * ((1.0 / cutoff) ** 12 - (1.0 / cutoff) ** 6)
        pair_energy = 4.0 * (inv_r12 - inv_r6) - cutoff_energy
        pair_energy = torch.where(active, pair_energy, torch.zeros_like(pair_energy))

        # Virial, for the pressure. W is the sum over pairs of
        # r_ij . f_ij, with r_ij from i to j and f_ij the force
        # on i. Here displacement runs first -> second while
        # pair_force is the force on first, so the dot product
        # picks up a minus sign.

        self._virial = float(-torch.sum(displacement * pair_force))

        # Split each pair's energy between its two atoms and
        # count neighbours inside the cutoff. Both ride along on
        # the loop that is already running.

        half_energy = 0.5 * pair_energy

        self.per_atom_energy.index_add_(0, self.pair_first, half_energy)
        self.per_atom_energy.index_add_(0, self.pair_second, half_energy)

        # Only the first shell counts as a neighbour. Inside
        # the full 2.5 sigma cutoff a face centred cubic solid
        # has about 55 atoms, which says nothing useful; inside
        # 1.4 sigma it has its 12 nearest neighbours, so this
        # number tells you whether an atom is in the bulk, at a
        # surface, or next to a defect.

        contacts = (
            r2 < self.coordination_cutoff ** 2
        ).to(self.dtype)

        self.coordination.index_add_(0, self.pair_first, contacts)
        self.coordination.index_add_(0, self.pair_second, contacts)

        return forces, torch.sum(pair_energy)

    def step(self, number_of_steps=1):
        dt = self.time_step
        mass = self.particle_mass
        for _ in range(int(number_of_steps)):
            acceleration = self.forces / mass

            movement = (self.velocities * dt
                        + 0.5 * acceleration * dt * dt)

            # Kept unwrapped alongside the wrapped copy, so mean
            # squared displacement is not destroyed every time an
            # atom crosses a periodic boundary.

            self.unwrapped = self.unwrapped + movement

            self.positions = (self.positions + movement) % self.box_size
            self.build_pairs()
            new_forces, potential = self.compute_forces()
            self.velocities = (self.velocities
                               + 0.5 * (self.forces + new_forces) / mass * dt)
            self.forces = new_forces
            self._potential_energy = potential
            if self.thermostat_is_on:
                self._apply_langevin()
            self.elapsed_time += dt
            self.steps_taken += 1

    def _apply_langevin(self):
        decay = np.exp(-self.friction * self.time_step)
        noise_scale = np.sqrt(self.target_temperature * (1.0 - decay ** 2)
                               / self.particle_mass)
        self.velocities = (self.velocities * decay
                           + noise_scale * torch.randn_like(self.velocities))
        self.velocities -= torch.mean(self.velocities, dim=0, keepdim=True)

    def set_target_temperature_kelvin(self, kelvin):
        self.target_temperature_kelvin = kelvin

    def set_thermostat(self, enabled):
        self.thermostat_is_on = bool(enabled)

    def force_rebuild_neighbours(self):
        self.build_pairs(force=True)

    # --------------------------------------------------------
    # Measurements

    @property
    def volume(self):
        return self.box_size ** 3

    @property
    def density(self):
        return self.particle_count / self.volume

    @property
    def pressure(self):
        # Ideal gas term plus the virial correction. In reduced
        # units P = rho * T + W / (3V).

        return (
            self.density * self.temperature
            + self._virial / (3.0 * self.volume)
        )

    def reset_msd(self):
        self.msd_reference = self.unwrapped.clone()
        self.msd_reference_time = self.elapsed_time

    @property
    def mean_squared_displacement(self):
        offset = self.unwrapped - self.msd_reference

        return float(torch.mean(torch.sum(offset ** 2, dim=1)))

    @property
    def diffusion_coefficient(self):
        # Einstein relation in three dimensions: MSD = 6 D t.
        # A solid gives a flat MSD and therefore D near zero; a
        # liquid gives a straight line and a real D.

        elapsed = self.elapsed_time - self.msd_reference_time

        if elapsed < 1e-9:
            return 0.0

        return self.mean_squared_displacement / (6.0 * elapsed)

    def radial_distribution(self, bin_count=120, maximum=None):
        # g(r): the probability of finding another atom a given
        # distance away, relative to a uniform gas. A crystal
        # gives sharp peaks, a liquid gives a couple of broad
        # ones, a gas gives almost nothing.
        #
        # Only valid out to the neighbour list range, since no
        # pairs beyond that are known.

        if maximum is None:
            maximum = self.cutoff_distance + self.skin_distance

        if self.pair_first.numel() == 0:
            centres = np.linspace(0.0, maximum, bin_count)
            return centres, np.zeros(bin_count)

        displacement = self.minimum_image(
            self.positions[self.pair_second]
            - self.positions[self.pair_first]
        )

        distances = torch.sqrt(
            torch.sum(displacement ** 2, dim=1)
        ).detach().cpu().numpy()

        counts, edges = np.histogram(
            distances,
            bins=bin_count,
            range=(0.0, maximum)
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

    # --------------------------------------------------------
    # Phases

    def set_phase(self, number_density, target_temperature):
        # Density fixes the box size, so changing it means
        # rebuilding the lattice from scratch.

        self.number_density = float(number_density)
        self.target_temperature = float(target_temperature)

        self.reset()

    def scale_velocities_to(self, target_temperature):
        # An instant temperature jump, for melting or quenching
        # without waiting for the thermostat.

        current = self.temperature

        if current <= 0.0:
            return

        self.velocities = self.velocities * float(
            np.sqrt(target_temperature / current)
        )