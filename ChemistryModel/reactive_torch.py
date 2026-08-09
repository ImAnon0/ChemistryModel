import numpy as np

import torch

from scipy.spatial import cKDTree

import reactive as R


# ============================================================
# Reactive molecular dynamics on the GPU
# ============================================================
#
# Forces come from automatic differentiation of the energy rather
# than a hand-derived gradient. The bond order for one pair
# depends on every neighbour of both atoms, so the analytic
# derivative is long and easy to get subtly wrong. Autograd
# cannot disagree with the energy it differentiates, which
# removes a whole category of bug at a modest cost in speed.

MAXIMUM_NEIGHBOURS = 12


class ReactiveSimulation:

    def __init__(self, symbols, positions, box_size,
                 time_step=0.25, target_temperature=300.0,
                 friction=0.01, device=None, dtype=torch.float32,
                 random_seed=7, rebuild_every=20):
        # time_step is in femtoseconds, temperature in kelvin.

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.dtype = dtype

        self.symbols = list(symbols)
        self.box_size = float(box_size)

        self.time_step = float(time_step)
        self.target_temperature = float(target_temperature)
        self.friction = float(friction)
        self.rebuild_every = int(rebuild_every)

        self.thermostat_is_on = True

        self.random_generator = np.random.default_rng(random_seed)

        types = R.types_from_symbols(symbols)

        self.types_numpy = types

        self.types = torch.tensor(
            types, device=self.device, dtype=torch.long
        )

        masses = R.masses_from_symbols(symbols)

        self.masses = torch.tensor(
            masses, device=self.device, dtype=self.dtype
        )

        self.positions = torch.tensor(
            np.asarray(positions, dtype=float),
            device=self.device,
            dtype=self.dtype
        )

        self._load_tables()

        self.steps_taken = 0
        self.elapsed_femtoseconds = 0.0

        self.set_temperature(self.target_temperature)

        self.neighbours = None
        self.neighbour_mask = None
        self.build_neighbours()

        self.forces, self._potential_energy = self.compute_forces()

    # --------------------------------------------------------

    def _load_tables(self):
        def to_tensor(array):
            return torch.tensor(
                array, device=self.device, dtype=self.dtype
            )

        self.bond_length = to_tensor(R.BOND_LENGTH)
        self.bond_depth = to_tensor(R.BOND_DEPTH)
        self.bond_width = to_tensor(R.BOND_WIDTH)

        self.double_length = to_tensor(R.DOUBLE_LENGTH)
        self.double_depth = to_tensor(R.DOUBLE_DEPTH)
        self.double_width = to_tensor(R.DOUBLE_WIDTH)

        self.triple_length = to_tensor(R.TRIPLE_LENGTH)
        self.triple_depth = to_tensor(R.TRIPLE_DEPTH)
        self.triple_width = to_tensor(R.TRIPLE_WIDTH)

        self.cutoff_inner = to_tensor(R.CUTOFF_INNER)
        self.cutoff_outer = to_tensor(R.CUTOFF_OUTER)

        self.valence = to_tensor(R.VALENCE_ARRAY)
        self.outer_electrons = to_tensor(R.OUTER_ELECTRON_ARRAY)
        self.angle_stiffness = to_tensor(R.ANGLE_STIFFNESS_ARRAY)

        self.over_penalty = float(R.OVER_COORDINATION_PENALTY)
        self.lone_pair_squeeze = float(R.LONE_PAIR_SQUEEZE)

        self.maximum_cutoff = float(R.MAXIMUM_CUTOFF)

    @property
    def atom_count(self):
        return int(self.positions.shape[0])

    @property
    def particle_count(self):
        return self.atom_count

    @property
    def positions_numpy(self):
        return self.positions.detach().cpu().numpy()

    @property
    def degrees_of_freedom(self):
        return 3 * self.atom_count - 3

    @property
    def kinetic_energy(self):
        # In eV, with masses in amu and velocities in A/fs.
        # 1 amu (A/fs)^2 = 103.642 eV.

        return float(
            0.5
            * torch.sum(self.masses[:, None] * self.velocities ** 2)
        ) * 103.642

    @property
    def potential_energy(self):
        return float(self._potential_energy)

    @property
    def total_energy(self):
        return self.kinetic_energy + self.potential_energy

    @property
    def temperature(self):
        # Kelvin, from the equipartition theorem. Boltzmann's
        # constant is 8.617333e-5 eV per kelvin.

        return (
            2.0 * self.kinetic_energy
            / (self.degrees_of_freedom * 8.617333e-5)
        )

    @property
    def temperature_kelvin(self):
        return self.temperature

    @property
    def target_temperature_kelvin(self):
        return self.target_temperature

    @target_temperature_kelvin.setter
    def target_temperature_kelvin(self, value):
        self.target_temperature = float(value)

    @property
    def elapsed_picoseconds(self):
        return self.elapsed_femtoseconds / 1000.0

    # --------------------------------------------------------

    def set_temperature(self, kelvin):
        # Maxwell-Boltzmann velocities in angstroms per
        # femtosecond.

        masses = self.masses.detach().cpu().numpy()

        scale = np.sqrt(
            8.617333e-5 * kelvin / (masses * 103.642)
        )

        velocities = (
            self.random_generator.normal(
                size=(len(masses), 3)
            ) * scale[:, None]
        )

        velocities -= np.average(
            velocities, axis=0, weights=masses
        )

        self.velocities = torch.tensor(
            velocities, device=self.device, dtype=self.dtype
        )

    def build_neighbours(self):
        # A padded neighbour table of fixed width. Fixed width
        # means the angular term can be evaluated as one batched
        # tensor operation instead of a Python loop over atoms.

        positions = self.positions_numpy % self.box_size

        tree = cKDTree(positions, boxsize=self.box_size)

        lists = tree.query_ball_point(
            positions, r=self.maximum_cutoff * 1.25
        )

        count = len(positions)

        table = np.zeros(
            (count, MAXIMUM_NEIGHBOURS), dtype=np.int64
        )
        mask = np.zeros(
            (count, MAXIMUM_NEIGHBOURS), dtype=bool
        )

        for index, entries in enumerate(lists):
            others = [item for item in entries if item != index]

            if len(others) > MAXIMUM_NEIGHBOURS:
                offsets = positions[others] - positions[index]
                offsets -= self.box_size * np.round(
                    offsets / self.box_size
                )

                order = np.argsort(np.sum(offsets ** 2, axis=1))

                others = [
                    others[position]
                    for position in order[:MAXIMUM_NEIGHBOURS]
                ]

            table[index, :len(others)] = others
            mask[index, :len(others)] = True

        self.neighbours = torch.tensor(
            table, device=self.device, dtype=torch.long
        )

        self.neighbour_mask = torch.tensor(
            mask, device=self.device, dtype=torch.bool
        )

    # --------------------------------------------------------

    def energy(self, positions):
        # Mirrors reactive.potential_energy term for term, in the
        # padded neighbour-table form.

        neighbours = self.neighbours
        mask = self.neighbour_mask.to(self.dtype)

        offsets = positions[neighbours] - positions[:, None, :]

        offsets = offsets - self.box_size * torch.round(
            offsets / self.box_size
        )

        distances = torch.sqrt(
            torch.clamp(torch.sum(offsets ** 2, dim=2), min=1e-12)
        )

        centre_types = self.types[:, None].expand_as(neighbours)
        other_types = self.types[neighbours]

        pair_inner = self.cutoff_inner[centre_types, other_types]
        pair_outer = self.cutoff_outer[centre_types, other_types]

        span = torch.clamp(pair_outer - pair_inner, min=1e-9)

        fraction = torch.clamp(
            (distances - pair_inner) / span, 0.0, 1.0
        )

        taper = 0.5 * (1.0 + torch.cos(np.pi * fraction)) * mask

        coordination = torch.sum(taper, dim=1)

        valence = self.valence[self.types]

        # ---- bond order ----
        #
        # Spare valence is handed out to whichever partners can
        # take it, and a bond ends up as strong as the poorer
        # partner allows. Everything here is built from per-atom
        # quantities gathered at both ends, so no transpose of
        # the neighbour table is needed.

        spare = torch.clamp(valence - coordination, min=0.0)

        spare_other = spare[neighbours]

        weighted = taper * spare_other

        totals = torch.sum(weighted, dim=1)

        totals_other = totals[neighbours]

        share_out = torch.where(
            totals[:, None] > 1e-9,
            spare[:, None] * weighted
            / torch.clamp(totals[:, None], min=1e-9),
            torch.zeros_like(weighted)
        )

        share_back = torch.where(
            totals_other > 1e-9,
            spare_other * (taper * spare[:, None])
            / torch.clamp(totals_other, min=1e-9),
            torch.zeros_like(weighted)
        )

        extra = torch.minimum(share_out, share_back)

        order = torch.clamp(1.0 + extra, 0.0, 3.0) * mask

        # ---- Morse parameters, blended by bond order ----

        lower = torch.clamp(order - 1.0, 0.0, 1.0)
        upper = torch.clamp(order - 2.0, 0.0, 1.0)

        def blend(single_table, double_table, triple_table):
            single = single_table[centre_types, other_types]
            double = double_table[centre_types, other_types]
            triple = triple_table[centre_types, other_types]

            first = single + (double - single) * lower

            return first + (triple - first) * upper

        pair_length = blend(
            self.bond_length, self.double_length, self.triple_length
        )
        pair_depth = blend(
            self.bond_depth, self.double_depth, self.triple_depth
        )
        pair_width = blend(
            self.bond_width, self.double_width, self.triple_width
        )

        shift = distances - pair_length

        repulsive = pair_depth * torch.exp(-2.0 * pair_width * shift)
        attractive = 2.0 * pair_depth * torch.exp(
            -pair_width * shift
        )

        pair_energy = taper * (repulsive - attractive)

        bond_total = 0.5 * torch.sum(pair_energy)

        # ---- over-coordination penalty ----
        #
        # The whole of the model's activation barriers. Halfway
        # through a transfer the moving atom touches two partners
        # at once; without this the halfway point is the most
        # stable place to be and nothing has a barrier at all.

        excess = torch.clamp(coordination - valence, min=0.0)

        over_total = self.over_penalty * torch.sum(excess ** 2)

        # ---- angles, from electron domain counting ----

        bonded_order = torch.sum(taper * order, dim=1)

        outer = self.outer_electrons[self.types]

        lone_pairs = torch.clamp(
            (outer - bonded_order) / 2.0, min=0.0
        )

        steric = torch.clamp(coordination + lone_pairs, 2.0, 4.0)

        # Linear at two domains, trigonal at three, tetrahedral
        # at four, interpolated in between.

        low_angle = torch.where(
            steric < 3.0,
            180.0 + (120.0 - 180.0) * (steric - 2.0),
            120.0 + (109.47 - 120.0) * (steric - 3.0)
        )

        rest = torch.deg2rad(
            low_angle - self.lone_pair_squeeze * lone_pairs
        )

        stiffness = self.angle_stiffness[self.types]

        left = offsets[:, :, None, :]
        right = offsets[:, None, :, :]

        dot = torch.sum(left * right, dim=3)

        cosine = torch.clamp(
            dot / torch.clamp(
                distances[:, :, None] * distances[:, None, :],
                min=1e-9
            ),
            -1.0 + 1e-7,
            1.0 - 1e-7
        )

        angle = torch.arccos(cosine)

        weight = taper[:, :, None] * taper[:, None, :]

        upper_triangle = torch.triu(
            torch.ones(
                weight.shape[1],
                weight.shape[2],
                device=self.device,
                dtype=self.dtype
            ),
            diagonal=1
        )

        angle_energy = (
            0.5
            * stiffness[:, None, None]
            * weight
            * upper_triangle
            * (angle - rest[:, None, None]) ** 2
        )

        return bond_total + over_total + torch.sum(angle_energy)

    def compute_forces(self):
        positions = self.positions.detach().requires_grad_(True)

        total = self.energy(positions)

        gradient, = torch.autograd.grad(total, positions)

        return -gradient.detach(), total.detach()

    # --------------------------------------------------------

    def step(self, number_of_steps=1):
        dt = self.time_step

        # Convert eV/A into amu * A / fs^2.

        conversion = 1.0 / 103.642

        masses = self.masses[:, None]

        for _ in range(int(number_of_steps)):
            acceleration = self.forces * conversion / masses

            self.positions = (
                self.positions
                + self.velocities * dt
                + 0.5 * acceleration * dt * dt
            ) % self.box_size

            if self.steps_taken % self.rebuild_every == 0:
                self.build_neighbours()

            new_forces, potential = self.compute_forces()

            new_acceleration = new_forces * conversion / masses

            self.velocities = self.velocities + 0.5 * (
                acceleration + new_acceleration
            ) * dt

            self.forces = new_forces
            self._potential_energy = potential

            if self.thermostat_is_on:
                self._apply_langevin()

            self.elapsed_femtoseconds += dt
            self.steps_taken += 1

    def _apply_langevin(self):
        decay = float(np.exp(-self.friction * self.time_step))

        masses = self.masses[:, None]

        scale = torch.sqrt(
            torch.tensor(
                8.617333e-5 * self.target_temperature / 103.642,
                device=self.device,
                dtype=self.dtype
            ) / masses
        ) * float(np.sqrt(1.0 - decay ** 2))

        self.velocities = (
            self.velocities * decay
            + scale * torch.randn_like(self.velocities)
        )

    # --------------------------------------------------------

    def bond_list(self, threshold=0.35):
        # Which atoms are currently bonded, for drawing sticks
        # and for counting molecules. A bond counts when its
        # taper is above the threshold.

        positions = self.positions_numpy

        neighbours = self.neighbours.detach().cpu().numpy()
        mask = self.neighbour_mask.detach().cpu().numpy()

        first = []
        second = []

        for index in range(len(positions)):
            for slot in range(neighbours.shape[1]):
                if not mask[index, slot]:
                    continue

                other = int(neighbours[index, slot])

                if other <= index:
                    continue

                offset = positions[other] - positions[index]
                offset -= self.box_size * np.round(
                    offset / self.box_size
                )

                distance = float(np.linalg.norm(offset))

                inner = R.CUTOFF_INNER[
                    self.types_numpy[index], self.types_numpy[other]
                ]
                outer = R.CUTOFF_OUTER[
                    self.types_numpy[index], self.types_numpy[other]
                ]

                taper = R.smooth_cutoff(
                    np.array([distance]),
                    np.array([inner]),
                    np.array([outer])
                )[0]

                if taper > threshold:
                    first.append(index)
                    second.append(other)

        return np.array(first, dtype=int), np.array(second, dtype=int)

    def molecule_formulas(self):
        # Group bonded atoms and report what has formed.

        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        first, second = self.bond_list()

        count = self.atom_count

        if len(first) == 0:
            labels = np.arange(count)
        else:
            graph = coo_matrix(
                (np.ones(len(first)), (first, second)),
                shape=(count, count)
            )

            _, labels = connected_components(graph, directed=False)

        formulas = {}

        for label in np.unique(labels):
            members = np.where(labels == label)[0]

            counts = {}

            for member in members:
                symbol = R.ELEMENTS[self.types_numpy[member]]
                counts[symbol] = counts.get(symbol, 0) + 1

            formula = "".join(
                symbol + (str(counts[symbol]) if counts[symbol] > 1 else "")
                for symbol in ["C", "N", "O", "H"]
                if symbol in counts
            )

            formulas[formula] = formulas.get(formula, 0) + 1

        return formulas