import numpy as np

import torch

from scipy.spatial import cKDTree

from membrane import AmphiphileModel, HEAD, TAIL, BEADS_PER_LIPID


# ============================================================
# GPU port of the amphiphile force loop
# ============================================================
#
# Deliberately mirrors membrane.py term for term, so that
# validate_torch.py can check the two against each other. If you
# change the physics in one, change it in the other and re-run
# the validation.
#
# The neighbour search stays on the CPU. cKDTree is very good and
# rebuilding is rare; everything else lives on the GPU and never
# comes back.


class TorchMembraneSimulation:

    def __init__(
        self,
        number_of_lipids=2000,
        box_size=45.0,
        target_temperature=1.1,
        time_step=0.01,
        friction=1.0,
        skin_distance=0.6,
        start="random",
        device=None,
        dtype=torch.float32,
        random_seed=3
    ):
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)
        self.dtype = dtype

        # The CPU model is reused for geometry, bond topology and
        # the starting configuration. Only the hot loop moves.

        self.model = AmphiphileModel(
            number_of_lipids=number_of_lipids,
            box_size=box_size
        )

        self.box_size = float(box_size)

        self.time_step = time_step
        self.friction = friction

        self._target_temperature = target_temperature

        self.skin_distance = skin_distance
        self.start = start
        self.random_seed = random_seed

        self.thermostat_is_on = True
        self.last_event = "none"

        # Growth is off until growth_interval is a step count.

        self.growth_interval = 0
        self.growth_batch = 6
        self.growth_outer_fraction = 0.85
        self.steps_taken = 0

        # Constants pulled onto the device once.

        self.epsilon = float(self.model.epsilon)
        self.sigma = float(self.model.sigma)
        self.attraction_range = float(self.model.attraction_range)

        self.fene_stiffness = float(self.model.fene_stiffness)
        self.fene_maximum_extension = float(
            self.model.fene_maximum_extension
        )

        self.bending_stiffness = float(self.model.bending_stiffness)
        self.bending_rest_length = float(
            self.model.bending_rest_length
        )

        self.sigma_matrix = torch.tensor(
            self.model.sigma_matrix,
            device=self.device,
            dtype=self.dtype
        )

        self.bead_types = torch.tensor(
            self.model.bead_types,
            device=self.device,
            dtype=torch.long
        )

        self.molecule_index_cpu = self.model.molecule_index

        self.bond_first = torch.tensor(
            self.model.bond_first,
            device=self.device,
            dtype=torch.long
        )

        self.bond_second = torch.tensor(
            self.model.bond_second,
            device=self.device,
            dtype=torch.long
        )

        self.bend_first = torch.tensor(
            self.model.bend_first,
            device=self.device,
            dtype=torch.long
        )

        self.bend_second = torch.tensor(
            self.model.bend_second,
            device=self.device,
            dtype=torch.long
        )

        self.masses = torch.ones(
            self.model.bead_count,
            device=self.device,
            dtype=self.dtype
        )

        self.reset()

    # --------------------------------------------------------

    @property
    def number_of_lipids(self):
        return self.model.number_of_lipids

    @property
    def positions_numpy(self):
        return self.positions.detach().cpu().numpy()

    @property
    def particle_positions(self):
        return self.positions_numpy

    @property
    def box_size_nanometers(self):
        return self.box_size

    @property
    def degrees_of_freedom(self):
        return 3 * self.model.bead_count - 3

    @property
    def kinetic_energy(self):
        return float(
            0.5
            * torch.sum(
                self.masses[:, None] * self.velocities ** 2
            )
        )

    @property
    def potential_energy(self):
        return float(self._potential_energy)

    @property
    def total_energy(self):
        return self.kinetic_energy + self.potential_energy

    @property
    def temperature_kelvin(self):
        return (
            2.0
            * self.kinetic_energy
            / self.degrees_of_freedom
        )

    @property
    def target_temperature_kelvin(self):
        return self._target_temperature

    @target_temperature_kelvin.setter
    def target_temperature_kelvin(self, value):
        self._target_temperature = float(value)

    @property
    def elapsed_picoseconds(self):
        return self.elapsed_time

    # --------------------------------------------------------

    def reset(self):
        generator = np.random.default_rng(self.random_seed)
        self.random_generator = generator

        legacy = {
            "random": "random scatter",
            "vesicle": "single vesicle"
        }

        name = legacy.get(self.start, self.start)

        import structures

        starting_positions = structures.build(
            name,
            self.model,
            generator
        )

        self.positions = torch.tensor(
            starting_positions,
            device=self.device,
            dtype=self.dtype
        )

        velocities = generator.normal(
            size=starting_positions.shape
        ) * np.sqrt(self._target_temperature)

        velocities -= velocities.mean(axis=0)

        self.velocities = torch.tensor(
            velocities,
            device=self.device,
            dtype=self.dtype
        )

        self.reference_positions = None
        self.pair_first = None
        self.pair_second = None
        self.rebuild_count = 0

        self.build_pairs(force=True)

        self.forces, self._potential_energy = self.compute_forces()

        self.elapsed_time = 0.0

        self.relax()

    # --------------------------------------------------------

    def minimum_image(self, displacements):
        return displacements - self.box_size * torch.round(
            displacements / self.box_size
        )

    def build_pairs(self, force=False):
        if force or self.reference_positions is None:
            needs_rebuild = True
        else:
            displacement = self.minimum_image(
                self.positions - self.reference_positions
            )

            largest_movement = float(
                torch.sqrt(
                    torch.max(
                        torch.sum(displacement ** 2, dim=1)
                    )
                )
            )

            needs_rebuild = (
                largest_movement > 0.5 * self.skin_distance
            )

        if not needs_rebuild:
            return

        # The one place per rebuild where data crosses back.

        positions_cpu = (
            self.positions.detach().cpu().numpy() % self.box_size
        )

        tree = cKDTree(positions_cpu, boxsize=self.box_size)

        pairs = tree.query_pairs(
            r=self.model.interaction_cutoff + self.skin_distance,
            output_type="ndarray"
        )

        if len(pairs) == 0:
            empty = torch.zeros(
                0,
                device=self.device,
                dtype=torch.long
            )

            self.pair_first = empty
            self.pair_second = empty
        else:
            first = pairs[:, 0]
            second = pairs[:, 1]

            same_molecule = (
                self.molecule_index_cpu[first]
                == self.molecule_index_cpu[second]
            )

            keep = ~same_molecule

            self.pair_first = torch.tensor(
                first[keep],
                device=self.device,
                dtype=torch.long
            )

            self.pair_second = torch.tensor(
                second[keep],
                device=self.device,
                dtype=torch.long
            )

        self.reference_positions = self.positions.clone()
        self.rebuild_count += 1

    # --------------------------------------------------------

    def compute_forces(self):
        positions = self.positions

        forces = torch.zeros_like(positions)
        energy = torch.zeros((), device=self.device, dtype=self.dtype)

        # ---- FENE bonds ----

        displacements = self.minimum_image(
            positions[self.bond_second]
            - positions[self.bond_first]
        )

        distances = torch.linalg.norm(displacements, dim=1)

        extension_ratio = torch.clamp(
            distances / self.fene_maximum_extension,
            max=0.98
        )

        denominator = 1.0 - extension_ratio ** 2

        magnitude = (
            self.fene_stiffness * distances / denominator
        )

        pair_forces = (
            displacements / distances[:, None]
            * magnitude[:, None]
        )

        forces.index_add_(0, self.bond_first, pair_forces)
        forces.index_add_(0, self.bond_second, -pair_forces)

        energy = energy + torch.sum(
            -0.5
            * self.fene_stiffness
            * self.fene_maximum_extension ** 2
            * torch.log(denominator)
        )

        # ---- stiffening spring ----

        bend_displacements = self.minimum_image(
            positions[self.bend_second]
            - positions[self.bend_first]
        )

        bend_distances = torch.linalg.norm(
            bend_displacements,
            dim=1
        )

        stretch = bend_distances - self.bending_rest_length

        bend_forces = (
            bend_displacements / bend_distances[:, None]
            * (self.bending_stiffness * stretch)[:, None]
        )

        forces.index_add_(0, self.bend_first, bend_forces)
        forces.index_add_(0, self.bend_second, -bend_forces)

        energy = energy + torch.sum(
            0.5 * self.bending_stiffness * stretch ** 2
        )

        # ---- non-bonded ----

        if self.pair_first is None or len(self.pair_first) == 0:
            self._potential_energy = energy
            return forces, energy

        pair_displacements = self.minimum_image(
            positions[self.pair_second]
            - positions[self.pair_first]
        )

        distance_squared = torch.clamp(
            torch.sum(pair_displacements ** 2, dim=1),
            min=1e-4
        )

        pair_distances = torch.sqrt(distance_squared)

        type_first = self.bead_types[self.pair_first]
        type_second = self.bead_types[self.pair_second]

        pair_sigma = self.sigma_matrix[type_first, type_second]

        repulsive_cutoff = 2.0 ** (1.0 / 6.0) * pair_sigma

        in_core = pair_distances < repulsive_cutoff

        sigma_over_r = pair_sigma / pair_distances

        power_6 = sigma_over_r ** 6
        power_12 = power_6 ** 2

        core_magnitude = (
            24.0
            * self.epsilon
            * (2.0 * power_12 - power_6)
            / pair_distances
        )

        force_magnitude = torch.where(
            in_core,
            core_magnitude,
            torch.zeros_like(core_magnitude)
        )

        energy = energy + torch.sum(
            torch.where(
                in_core,
                4.0 * self.epsilon * (power_12 - power_6)
                + self.epsilon,
                torch.zeros_like(power_6)
            )
        )

        # ---- tail-tail attraction ----

        both_tails = (
            (type_first == TAIL) & (type_second == TAIL)
        )

        flat_region = both_tails & in_core

        energy = energy - self.epsilon * torch.count_nonzero(
            flat_region
        ).to(self.dtype)

        tail_cutoff = repulsive_cutoff + self.attraction_range

        in_well = (
            both_tails
            & (pair_distances >= repulsive_cutoff)
            & (pair_distances < tail_cutoff)
        )

        phase = (
            np.pi
            * (pair_distances - repulsive_cutoff)
            / (2.0 * self.attraction_range)
        )

        well_energy = -self.epsilon * torch.cos(phase) ** 2

        energy = energy + torch.sum(
            torch.where(
                in_well,
                well_energy,
                torch.zeros_like(well_energy)
            )
        )

        well_magnitude = (
            -self.epsilon
            * np.pi
            * torch.sin(2.0 * phase)
            / (2.0 * self.attraction_range)
        )

        force_magnitude = force_magnitude + torch.where(
            in_well,
            well_magnitude,
            torch.zeros_like(well_magnitude)
        )

        pair_forces = (
            pair_displacements / pair_distances[:, None]
            * force_magnitude[:, None]
        )

        forces.index_add_(0, self.pair_first, -pair_forces)
        forces.index_add_(0, self.pair_second, pair_forces)

        self._potential_energy = energy

        return forces, energy

    # --------------------------------------------------------

    def relax(self, steps=400, maximum_force=20.0, step_size=0.005):
        for _ in range(steps):
            self.build_pairs()

            forces, _ = self.compute_forces()

            magnitudes = torch.linalg.norm(forces, dim=1)

            # Per bead, not global: one badly overlapped pair
            # would otherwise scale every other force to nothing
            # and the overlap could never clear.

            scale = torch.clamp(
                maximum_force / torch.clamp(magnitudes, min=1e-12),
                max=1.0
            )

            forces = forces * scale[:, None]

            self.positions = (
                self.positions + forces * step_size
            ) % self.box_size

        self.build_pairs(force=True)

        self.forces, self._potential_energy = self.compute_forces()

        velocities = self.random_generator.normal(
            size=(self.model.bead_count, 3)
        ) * np.sqrt(self._target_temperature)

        velocities -= velocities.mean(axis=0)

        self.velocities = torch.tensor(
            velocities,
            device=self.device,
            dtype=self.dtype
        )

    def step(self, number_of_steps=1):
        dt = self.time_step
        masses = self.masses[:, None]

        for _ in range(number_of_steps):
            accelerations = self.forces / masses

            self.positions = (
                self.positions
                + self.velocities * dt
                + 0.5 * accelerations * dt ** 2
            ) % self.box_size

            self.build_pairs()

            new_forces, _ = self.compute_forces()

            self.velocities = self.velocities + (
                0.5 * (self.forces + new_forces) / masses * dt
            )

            self.forces = new_forces

            if self.thermostat_is_on:
                self._apply_langevin()

            self.elapsed_time += dt
            self.steps_taken += 1

            if (
                self.growth_interval
                and self.steps_taken % self.growth_interval == 0
            ):
                self.add_lipids(
                    self.growth_batch,
                    outer_fraction=self.growth_outer_fraction
                )

                masses = self.masses[:, None]

    def _apply_langevin(self):
        decay = float(np.exp(-self.friction * self.time_step))

        noise_scale = float(
            np.sqrt(
                self._target_temperature * (1.0 - decay ** 2)
            )
        )

        self.velocities = (
            self.velocities * decay
            + noise_scale
            * torch.randn(
                self.velocities.shape,
                device=self.device,
                dtype=self.dtype
            )
        )

    # --------------------------------------------------------

    def cluster_report(self, contact_distance=1.6):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        positions = (
            self.positions.detach().cpu().numpy() % self.box_size
        )

        tails = positions[1::BEADS_PER_LIPID]

        tree = cKDTree(tails, boxsize=self.box_size)

        pairs = tree.query_pairs(
            r=contact_distance,
            output_type="ndarray"
        )

        count = len(tails)

        if len(pairs) == 0:
            return count, 1

        graph = coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
            shape=(count, count)
        )

        number_of_clusters, labels = connected_components(
            graph,
            directed=False
        )

        return int(np.max(np.bincount(labels))), int(
            number_of_clusters
        )

    # --------------------------------------------------------
    # Changing the number or type of lipids
    #
    # These are rare events, so the arrays come back to the CPU,
    # get rebuilt, and go straight back to the device. One
    # transfer per growth batch is nothing next to the force
    # evaluations in between.

    def _rebuild(self, positions, velocities, bead_types,
                 lipid_count):
        new_model = AmphiphileModel(
            number_of_lipids=lipid_count,
            box_size=self.box_size
        )

        new_model.bead_types = bead_types

        self.model = new_model
        self.molecule_index_cpu = new_model.molecule_index

        self.bead_types = torch.tensor(
            bead_types,
            device=self.device,
            dtype=torch.long
        )

        for name in (
            "bond_first", "bond_second", "bend_first", "bend_second"
        ):
            setattr(
                self,
                name,
                torch.tensor(
                    getattr(new_model, name),
                    device=self.device,
                    dtype=torch.long
                )
            )

        self.masses = torch.ones(
            new_model.bead_count,
            device=self.device,
            dtype=self.dtype
        )

        self.positions = torch.tensor(
            positions,
            device=self.device,
            dtype=self.dtype
        )

        self.velocities = torch.tensor(
            velocities,
            device=self.device,
            dtype=self.dtype
        )

        self.reference_positions = None
        self.build_pairs(force=True)

        self.forces, self._potential_energy = self.compute_forces()

    def settle(self, steps=120, maximum_force=25.0,
               step_size=0.004):
        for _ in range(steps):
            self.build_pairs()

            forces, _ = self.compute_forces()

            magnitudes = torch.linalg.norm(forces, dim=1)

            scale = torch.clamp(
                maximum_force / torch.clamp(magnitudes, min=1e-12),
                max=1.0
            )

            self.positions = (
                self.positions + forces * scale[:, None] * step_size
            ) % self.box_size

        self.build_pairs(force=True)

        self.forces, self._potential_energy = self.compute_forces()

    def minimum_image_numpy(self, displacements):
        return displacements - self.box_size * np.round(
            displacements / self.box_size
        )

    def leaflet_sign(self, positions):
        heads = positions[0::BEADS_PER_LIPID]
        ends = positions[2::BEADS_PER_LIPID]

        anchor = heads[0]

        offsets = self.minimum_image_numpy(heads - anchor)

        centre = anchor + offsets.mean(axis=0)

        outward = self.minimum_image_numpy(heads - centre)
        along = self.minimum_image_numpy(heads - ends)

        return np.sign(np.sum(outward * along, axis=1))

    def add_lipids(self, count, outer_fraction=1.0):
        if count <= 0:
            return 0

        positions = self.positions.detach().cpu().numpy()
        velocities = self.velocities.detach().cpu().numpy()
        bead_types = self.bead_types.detach().cpu().numpy()

        signs = self.leaflet_sign(positions)

        outer = np.where(signs > 0)[0]
        inner = np.where(signs <= 0)[0]

        if len(outer) == 0:
            outer = np.arange(self.model.number_of_lipids)

        if len(inner) == 0:
            inner = outer

        outer_count = int(round(count * outer_fraction))
        inner_count = count - outer_count

        templates = np.concatenate([
            self.random_generator.choice(outer, size=outer_count),
            self.random_generator.choice(inner, size=inner_count)
        ]).astype(int)

        new_positions = np.zeros((count * BEADS_PER_LIPID, 3))

        for slot, template in enumerate(templates):
            base = template * BEADS_PER_LIPID

            head = positions[base]

            axis = self.minimum_image_numpy(
                positions[base + BEADS_PER_LIPID - 1] - head
            )

            length = np.linalg.norm(axis)

            if length < 1e-8:
                axis = np.array([0.0, 0.0, 1.0])
                length = 1.0

            axis = axis / length

            best_head = None
            best_clearance = -1.0

            for _ in range(8):
                sideways = self.random_generator.normal(size=3)
                sideways -= np.dot(sideways, axis) * axis

                norm = np.linalg.norm(sideways)

                if norm < 1e-8:
                    continue

                candidate = head + (sideways / norm) * 1.15

                offsets = self.minimum_image_numpy(
                    positions - candidate
                )

                clearance = float(
                    np.min(np.sum(offsets ** 2, axis=1))
                )

                if clearance > best_clearance:
                    best_clearance = clearance
                    best_head = candidate

            if best_head is None:
                best_head = head + np.array([1.15, 0.0, 0.0])

            for bead in range(BEADS_PER_LIPID):
                new_positions[slot * BEADS_PER_LIPID + bead] = (
                    best_head + axis * 0.95 * bead
                )

        positions = np.vstack([
            positions,
            new_positions % self.box_size
        ])

        velocities = np.vstack([
            velocities,
            self.random_generator.normal(
                size=new_positions.shape
            ) * np.sqrt(self._target_temperature)
        ])

        new_types = np.tile(
            np.array([HEAD, TAIL, TAIL]),
            count
        )

        bead_types = np.concatenate([bead_types, new_types])

        self._rebuild(
            positions,
            velocities,
            bead_types,
            self.model.number_of_lipids + count
        )

        self.settle(steps=120)

        self.last_event = f"grew by {count} lipids"

        return count

    def puncture(self, radius=4.0):
        positions = self.positions.detach().cpu().numpy()
        velocities = self.velocities.detach().cpu().numpy()
        bead_types = self.bead_types.detach().cpu().numpy()

        heads = positions[0::BEADS_PER_LIPID]

        target = heads[
            self.random_generator.integers(len(heads))
        ]

        offsets = self.minimum_image_numpy(heads - target)

        doomed = np.linalg.norm(offsets, axis=1) < radius

        removed = int(np.count_nonzero(doomed))

        if removed == 0 or removed == len(heads):
            return 0

        keep = np.where(~doomed)[0]

        bead_indices = (
            keep[:, None] * BEADS_PER_LIPID
            + np.arange(BEADS_PER_LIPID)[None, :]
        ).ravel()

        self._rebuild(
            positions[bead_indices],
            velocities[bead_indices],
            bead_types[bead_indices],
            len(keep)
        )

        self.last_event = f"punctured, {removed} lipids removed"

        return removed

    def add_detergent(self, fraction=0.25):
        bead_types = self.bead_types.detach().cpu().numpy()

        lipid_count = self.model.number_of_lipids

        chosen = self.random_generator.choice(
            lipid_count,
            size=max(1, int(fraction * lipid_count)),
            replace=False
        )

        bead_types[chosen * BEADS_PER_LIPID + 2] = HEAD

        self.bead_types = torch.tensor(
            bead_types,
            device=self.device,
            dtype=torch.long
        )

        self.model.bead_types = bead_types

        self.forces, self._potential_energy = self.compute_forces()

        self.last_event = (
            f"detergent added to {len(chosen)} lipids"
        )