import numpy as np

from scipy.spatial import cKDTree

from membrane import AmphiphileModel, HEAD, TAIL, BEADS_PER_LIPID


class MembraneSimulation:
    # Deliberately exposes the same property names as
    # AseLiveSimulation, so the existing live window can drive it
    # without changes.

    def __init__(
        self,
        number_of_lipids=500,
        box_size=24.0,
        target_temperature=1.1,
        time_step=0.01,
        friction=1.0,
        skin_distance=0.5,
        start="random",
        random_seed=3
    ):
        self.model = AmphiphileModel(
            number_of_lipids=number_of_lipids,
            box_size=box_size
        )

        self.time_step = time_step
        self.friction = friction

        self._target_temperature = target_temperature

        self.skin_distance = skin_distance
        self.start = start
        self.random_seed = random_seed

        self.thermostat_is_on = True

        # Growth is off until growth_interval is set to a number
        # of steps between lipid additions.

        self.growth_interval = 0
        self.growth_batch = 6
        self.growth_outer_fraction = 0.85
        self.steps_taken = 0

        # Set by the disruption methods, purely for display.

        self.last_event = "none"

        self.reset()

    # --------------------------------------------------------
    # Properties mirroring AseLiveSimulation

    @property
    def target_temperature_kelvin(self):
        return self._target_temperature

    @target_temperature_kelvin.setter
    def target_temperature_kelvin(self, value):
        self._target_temperature = value

    @property
    def particle_positions(self):
        return self.positions

    @property
    def box_size_nanometers(self):
        return self.model.box_size

    @property
    def positions_in_nanometers(self):
        return self.positions

    @property
    def degrees_of_freedom(self):
        return 3 * self.model.bead_count - 3

    @property
    def kinetic_energy(self):
        return 0.5 * np.sum(
            self.model.masses[:, None]
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
            / self.degrees_of_freedom
        )

    @property
    def elapsed_picoseconds(self):
        return self.elapsed_time

    # --------------------------------------------------------

    def reset(self):
        generator = np.random.default_rng(self.random_seed)
        self.random_generator = generator

        if self.start in ("random", "vesicle"):
            legacy = {
                "random": "random scatter",
                "vesicle": "single vesicle"
            }
            name = legacy[self.start]
        else:
            name = self.start

        import structures

        self.positions = structures.build(name, self.model, generator)

        self.velocities = generator.normal(
            size=self.positions.shape
        ) * np.sqrt(
            self._target_temperature
            / self.model.masses
        )[:, None]

        self.velocities -= np.average(
            self.velocities,
            axis=0,
            weights=self.model.masses
        )

        self.reference_positions = None
        self.pair_first = np.array([], dtype=int)
        self.pair_second = np.array([], dtype=int)
        self.rebuild_count = 0

        self._update_pairs(force=True)

        self.forces, self._potential_energy = (
            self.model.forces_and_energy(
                self.positions,
                self.pair_first,
                self.pair_second
            )
        )

        self.elapsed_time = 0.0
        self.last_event = "none"

        self.relax(steps=400)

    # --------------------------------------------------------

    def relax(self, steps=400, maximum_force=20.0, step_size=0.005):
        # Steepest descent with a hard cap on how far any bead can
        # move in one step. This clears leftover overlaps without
        # letting a large force turn into a large velocity, which
        # is what destroys the integrator on step one.

        for _ in range(steps):
            self._update_pairs()

            forces, self._potential_energy = (
                self.model.forces_and_energy(
                    self.positions,
                    self.pair_first,
                    self.pair_second
                )
            )

            magnitudes = np.linalg.norm(forces, axis=1)

            # Capping by the single largest force would scale
            # every other force to nothing whenever one pair is
            # badly overlapped, so the overlap would never clear.
            # Each bead is limited on its own instead.

            scale = np.ones_like(magnitudes)

            too_large = magnitudes > maximum_force

            scale[too_large] = (
                maximum_force / magnitudes[too_large]
            )

            forces = forces * scale[:, None]

            self.positions = (
                self.positions + forces * step_size
            ) % self.model.box_size

        self._update_pairs(force=True)

        self.forces, self._potential_energy = (
            self.model.forces_and_energy(
                self.positions,
                self.pair_first,
                self.pair_second
            )
        )

        self.velocities = self.random_generator.normal(
            size=self.positions.shape
        ) * np.sqrt(
            self._target_temperature
            / self.model.masses
        )[:, None]

        self.velocities -= np.average(
            self.velocities,
            axis=0,
            weights=self.model.masses
        )

    def _update_pairs(self, force=False):
        if force or self.reference_positions is None:
            needs_rebuild = True
        else:
            displacement = self.model.minimum_image(
                self.positions - self.reference_positions
            )

            largest_movement = np.sqrt(
                np.max(np.sum(displacement ** 2, axis=1))
            )

            needs_rebuild = (
                largest_movement > 0.5 * self.skin_distance
            )

        if not needs_rebuild:
            return

        tree = cKDTree(
            self.positions % self.model.box_size,
            boxsize=self.model.box_size
        )

        pairs = tree.query_pairs(
            r=self.model.interaction_cutoff + self.skin_distance,
            output_type="ndarray"
        )

        if len(pairs) == 0:
            self.pair_first = np.array([], dtype=int)
            self.pair_second = np.array([], dtype=int)
        else:
            first = pairs[:, 0]
            second = pairs[:, 1]

            same_molecule = (
                self.model.molecule_index[first]
                == self.model.molecule_index[second]
            )

            self.pair_first = first[~same_molecule]
            self.pair_second = second[~same_molecule]

        self.reference_positions = self.positions.copy()
        self.rebuild_count += 1

    def step(self, number_of_steps=1):
        dt = self.time_step
        masses = self.model.masses[:, None]

        for _ in range(number_of_steps):
            accelerations = self.forces / masses

            self.positions = (
                self.positions
                + self.velocities * dt
                + 0.5 * accelerations * dt ** 2
            )

            self.positions %= self.model.box_size

            self._update_pairs()

            new_forces, self._potential_energy = (
                self.model.forces_and_energy(
                    self.positions,
                    self.pair_first,
                    self.pair_second
                )
            )

            self.velocities += (
                0.5
                * (self.forces + new_forces)
                / masses
                * dt
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

                dt = self.time_step
                masses = self.model.masses[:, None]

    def _apply_langevin(self):
        decay = np.exp(-self.friction * self.time_step)

        noise_scale = np.sqrt(
            self._target_temperature
            / self.model.masses
            * (1.0 - decay ** 2)
        )[:, None]

        self.velocities = (
            self.velocities * decay
            + noise_scale
            * self.random_generator.normal(
                size=self.velocities.shape
            )
        )

    # --------------------------------------------------------
    # Ways to break it

    def puncture(self, radius=3.0):
        # Delete every lipid whose head falls inside a sphere at a
        # random point on the structure, leaving a hole.

        head_positions = self.positions[HEAD::BEADS_PER_LIPID]

        target = head_positions[
            self.random_generator.integers(len(head_positions))
        ]

        offsets = self.model.minimum_image(
            head_positions - target
        )

        distances = np.linalg.norm(offsets, axis=1)

        doomed = distances < radius

        if np.count_nonzero(doomed) == 0:
            return 0

        keep_lipids = np.where(~doomed)[0]

        self._keep_only(keep_lipids)

        self.last_event = (
            f"punctured, {np.count_nonzero(doomed)} lipids removed"
        )

        return int(np.count_nonzero(doomed))

    def add_detergent(self, fraction=0.25):
        # Turn a fraction of lipids into single-tail molecules by
        # converting their end bead to a head. Cone-shaped
        # molecules cannot pack into a flat sheet, so the membrane
        # falls apart into micelles.

        lipid_count = self.model.number_of_lipids

        chosen = self.random_generator.choice(
            lipid_count,
            size=max(1, int(fraction * lipid_count)),
            replace=False
        )

        end_beads = chosen * BEADS_PER_LIPID + 2

        self.model.bead_types[end_beads] = HEAD

        self.last_event = (
            f"detergent added to {len(chosen)} lipids"
        )

    def heat_shock(self, temperature):
        self._target_temperature = temperature
        self.last_event = f"heated to T = {temperature:.2f}"

    def _keep_only(self, keep_lipids):
        bead_indices = (
            keep_lipids[:, None] * BEADS_PER_LIPID
            + np.arange(BEADS_PER_LIPID)[None, :]
        ).ravel()

        self.positions = self.positions[bead_indices]
        self.velocities = self.velocities[bead_indices]

        new_model = AmphiphileModel(
            number_of_lipids=len(keep_lipids),
            box_size=self.model.box_size
        )

        new_model.bead_types = self.model.bead_types[bead_indices]

        self.model = new_model

        self.reference_positions = None
        self._update_pairs(force=True)

        self.forces, self._potential_energy = (
            self.model.forces_and_energy(
                self.positions,
                self.pair_first,
                self.pair_second
            )
        )

    # --------------------------------------------------------
    # Measurement


    # --------------------------------------------------------
    # Growth and division

    def leaflet_sign(self):
        # +1 for lipids whose head points away from the local
        # centre of mass (outer leaflet), -1 for the inner one.

        positions = self.positions
        box = self.model.box_size

        heads = positions[0::BEADS_PER_LIPID]
        ends = positions[2::BEADS_PER_LIPID]

        anchor = heads[0]

        offsets = heads - anchor
        offsets -= box * np.round(offsets / box)

        centre = anchor + offsets.mean(axis=0)

        outward = self.model.minimum_image(heads - centre)

        along = self.model.minimum_image(heads - ends)

        return np.sign(np.sum(outward * along, axis=1)), centre

    def add_lipids(self, count, outer_fraction=1.0):
        # New lipids are inserted beside an existing one, copying
        # its orientation and shifted sideways by about one lipid
        # width. Sending most of them to the outer leaflet is what
        # drives division: the outer surface gains area faster
        # than the inner one, the bilayer is forced to curve, and
        # eventually a sphere is no longer the cheapest shape.

        if count <= 0:
            return 0

        signs, _ = self.leaflet_sign()

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

        new_positions = np.zeros(
            (count * BEADS_PER_LIPID, 3)
        )

        for slot, template in enumerate(templates):
            base = template * BEADS_PER_LIPID

            head = self.positions[base]
            axis = self.model.minimum_image(
                self.positions[base + BEADS_PER_LIPID - 1] - head
            )

            length = np.linalg.norm(axis)

            if length < 1e-8:
                axis = np.array([0.0, 0.0, 1.0])
                length = 1.0

            axis = axis / length

            # A random direction perpendicular to the lipid, so
            # the new one sits alongside it in the same leaflet.

            # Try several sideways directions and keep whichever
            # lands furthest from existing beads. Dropping a lipid
            # straight on top of a neighbour creates an overlap
            # no relaxation can recover from.

            best_head = None
            best_clearance = -1.0

            for _ in range(8):
                sideways = self.random_generator.normal(size=3)
                sideways -= np.dot(sideways, axis) * axis

                norm = np.linalg.norm(sideways)

                if norm < 1e-8:
                    continue

                sideways = sideways / norm

                candidate = head + sideways * 1.15

                offsets = self.model.minimum_image(
                    self.positions - candidate
                )

                clearance = np.min(
                    np.sum(offsets ** 2, axis=1)
                )

                if clearance > best_clearance:
                    best_clearance = clearance
                    best_head = candidate

            if best_head is None:
                best_head = head + np.array([1.15, 0.0, 0.0])

            new_head = best_head

            for bead in range(BEADS_PER_LIPID):
                new_positions[slot * BEADS_PER_LIPID + bead] = (
                    new_head + axis * 0.95 * bead
                )

        self.positions = np.vstack([
            self.positions,
            new_positions % self.model.box_size
        ])

        thermal = self.random_generator.normal(
            size=new_positions.shape
        ) * np.sqrt(self._target_temperature)

        self.velocities = np.vstack([
            self.velocities,
            thermal
        ])

        old_types = self.model.bead_types

        new_model = AmphiphileModel(
            number_of_lipids=self.model.number_of_lipids + count,
            box_size=self.model.box_size
        )

        new_model.bead_types[:len(old_types)] = old_types

        self.model = new_model

        self.reference_positions = None
        self._update_pairs(force=True)

        # New lipids land close to their neighbours, so a few
        # capped steps clear any overlap before real dynamics.

        self.settle(steps=120)

        self.last_event = f"grew by {count} lipids"

        return count

    def settle(self, steps=30, maximum_force=25.0, step_size=0.004):
        for _ in range(steps):
            self._update_pairs()

            forces, self._potential_energy = (
                self.model.forces_and_energy(
                    self.positions,
                    self.pair_first,
                    self.pair_second
                )
            )

            magnitudes = np.linalg.norm(forces, axis=1)

            # Capping by the single largest force would scale
            # every other force to nothing whenever one pair is
            # badly overlapped, so the overlap would never clear.
            # Each bead is limited on its own instead.

            scale = np.ones_like(magnitudes)

            too_large = magnitudes > maximum_force

            scale[too_large] = (
                maximum_force / magnitudes[too_large]
            )

            forces = forces * scale[:, None]

            self.positions = (
                self.positions + forces * step_size
            ) % self.model.box_size

        self._update_pairs(force=True)

        self.forces, self._potential_energy = (
            self.model.forces_and_energy(
                self.positions,
                self.pair_first,
                self.pair_second
            )
        )

    def cluster_report(self, contact_distance=1.6):
        # Groups lipids into connected aggregates by tail contact.
        # Nothing here knows about membranes; it just counts what
        # is touching what.

        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        tails = self.positions[1::BEADS_PER_LIPID]

        tree = cKDTree(
            tails % self.model.box_size,
            boxsize=self.model.box_size
        )

        pairs = tree.query_pairs(
            r=contact_distance,
            output_type="ndarray"
        )

        count = len(tails)

        if len(pairs) == 0:
            return count, 1

        graph = coo_matrix(
            (
                np.ones(len(pairs)),
                (pairs[:, 0], pairs[:, 1])
            ),
            shape=(count, count)
        )

        number_of_clusters, labels = connected_components(
            graph,
            directed=False
        )

        largest = np.max(np.bincount(labels))

        return int(largest), int(number_of_clusters)