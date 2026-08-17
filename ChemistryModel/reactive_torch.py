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
                 random_seed=7, rebuild_every=20,
                 relax_on_start=True):
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

        # The largest distance any atom may travel in one step,
        # in angstroms.
        #
        # Capping the force was the wrong handle: a hydrogen given
        # a large force for one step gains velocity in proportion
        # to that force over its mass, so a limit loose enough to
        # leave ordinary forces alone would still let a thousand
        # electronvolts into the lightest atom in the box.
        #
        # Limiting the move itself is direct and mass aware. At
        # 250 K a hydrogen covers about 0.006 A per step and a
        # carbon 0.002. The largest legitimate case is a hydrogen
        # inside a 30,000 K discharge channel, which manages
        # 0.068, so the limit must sit above that or it would
        # quietly clip every lightning strike. At 0.15 there is
        # room for the hottest real motion and none at all for the
        # jumps of a third of an angstrom and more that wreck a
        # run.

        self.maximum_step = 0.15
        self.capped_steps = 0
        self.capped_atom_counts = np.zeros(len(self.symbols), dtype=np.uint64)
        self.last_capped_atoms = ()
        self._last_move_capped_mask = None

        self.random_generator = np.random.default_rng(random_seed)

        # The thermostat draws from torch, and torch's global
        # generator is shared and unseeded. Without a generator of
        # its own, two runs started from identical positions
        # diverge immediately, which makes matched seeds across
        # conditions meaningless.

        self.torch_generator = torch.Generator(device=self.device)
        self.torch_generator.manual_seed(int(random_seed))

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
        self.reference_positions = None
        self.rebuild_count = 0
        self.build_neighbours()

        self.forces, self._potential_energy = self.compute_forces()

        # Atoms are placed with a minimum separation that is
        # comfortable in a roomy box and much too tight in a
        # crowded one. Packing 330 atoms into 12 angstroms leaves
        # pairs deep inside each other's repulsive wall, and the
        # first few steps then fling the box apart: measured on
        # one such run, the opening kinetic energy was 2075 eV
        # against 691 eV released by every bond formed in the
        # following twenty picoseconds. The chemistry that
        # followed was a product of that explosion rather than of
        # the density it was supposed to be testing.
        #
        # Steepest descent with a per-atom cap on how far anything
        # can move clears the overlaps without letting a large
        # force become a large velocity.

        # Skipped when picking up an existing run: the atoms
        # are already settled, and relaxing them would discard
        # the state being resumed.

        if relax_on_start:
            self.relax()

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
        self.over_depth_weight = float(R.OVER_COORDINATION_DEPTH_WEIGHT)
        self.over_reference_depth = float(R.OVER_COORDINATION_REFERENCE_DEPTH)
        self.lone_pair_squeeze = float(R.LONE_PAIR_SQUEEZE)

        # Zero reproduces the previous behaviour exactly, so this whole
        # addition is inert until it is turned on deliberately. It affects
        # every molecule with a multiply bonded atom, not just H transfer,
        # so measure the effect before switching it on globally.
        self.environment_softening = float(R.ENVIRONMENT_SOFTENING)
        self.environment_softening_epsilon = float(
            R.ENVIRONMENT_SOFTENING_SMOOTH_EPSILON_SQUARED
        )

        self.maximum_cutoff = float(R.MAXIMUM_CUTOFF)

        # How much room the neighbour table has before it goes
        # stale. The search radius below is the cutoff plus this.

        self.neighbour_skin = 0.25 * self.maximum_cutoff

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

    def needs_rebuild(self):
        # Rebuilding on a fixed step count is not safe. The
        # neighbour search reaches 1.25 times the bond cutoff, so
        # there is only about 0.6 A of slack, and a hydrogen that
        # has just taken a few eV from a lightning strike covers
        # nearly 2 A in twenty steps. It would sail straight past
        # atoms it should have bonded to, because the table still
        # described where everything was five femtoseconds ago.
        #
        # Rebuilding when something has actually moved half the
        # slack costs nothing when the box is cold and keeps the
        # table honest when it is not.

        if self.reference_positions is None:
            return True

        displacement = self.positions - self.reference_positions

        displacement = displacement - self.box_size * torch.round(
            displacement / self.box_size
        )

        largest = float(
            torch.sqrt(
                torch.max(torch.sum(displacement ** 2, dim=1))
            )
        )

        return largest > 0.5 * self.neighbour_skin

    def build_neighbours(self):
        # A padded neighbour table of fixed width. Fixed width
        # means the angular term can be evaluated as one batched
        # tensor operation instead of a Python loop over atoms.

        positions = self.positions_numpy % self.box_size

        tree = cKDTree(positions, boxsize=self.box_size)

        lists = tree.query_ball_point(
            positions,
            r=self.maximum_cutoff + self.neighbour_skin
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
        self._neighbour_weight = self.neighbour_mask.to(self.dtype)

        # Neighbour topology changed. Any derived heavy-valence topology
        # state based on the old contact graph is now invalid.
        if hasattr(self, "_heavy_membership_topology_cache"):
            self._heavy_membership_topology_cache = None
            self._heavy_membership_cache_signature = None

        self.reference_positions = self.positions.clone()
        self.rebuild_count += 1

    # --------------------------------------------------------

    def energy(self, positions):
        return torch.sum(self.energy_per_atom(positions))

    def _gather_neighbours(self, values, neighbours, role):
        """Gather padded neighbour rows; experimental index-select backend."""
        selected = getattr(self, "experimental_index_select_gather", False)
        if selected is True or (selected and role in selected):
            return torch.index_select(
                values, 0, neighbours.reshape(-1)
            ).reshape(*neighbours.shape, *values.shape[1:])
        return values[neighbours]

    def environment_softening_factor(self, taper, order, lower, mask,
                                     neighbours, cache_key=None):
        """How much each single bond is weakened by its partner's commitment.

        The tables carry one depth per element pair, so every C-H is the
        methane C-H at 4.291 eV. The aldehydic C-H is nearer 3.760, because
        that carbon is already committed to a double bond and its remaining
        single bonds are correspondingly weaker. A single generic entry
        cannot express that, and the error is not small: the formaldehyde
        abstraction came out at -0.228 eV against a thermochemical -0.718.

        So a single bond's depth is reduced in proportion to the multiple
        bond character its partner already carries elsewhere. The idea is the
        bond-order trade-off that Tersoff and REBO potentials use, though the
        form here is much simpler than either. It stays element agnostic: no
        molecule is named, and an atom holding only single bonds is untouched.

        Returns a multiplier of the same shape as the pair tables, and
        exactly one when the feature is switched off.
        """
        if self.environment_softening <= 0.0:
            return torch.ones_like(taper)

        # Cached within a single energy evaluation. The high fidelity
        # correction runs immediately after the base energy on the same
        # geometry and needs the identical factor, so computing it twice
        # duplicates a gather and a square root over every pair, which is
        # enough work to matter on a large box.
        #
        # Keyed on the positions tensor itself. Identity is exact and free,
        # where comparing values would cost more than the calculation being
        # saved. The tensor is held rather than its id, so a later object
        # cannot land on a recycled address and collide. Base and correction
        # each build their own taper, so keying on that would never hit.
        if cache_key is not None:
            cache = getattr(self, "_softening_cache", None)
            if cache is not None and cache[0] is cache_key:
                return cache[1]

        # Excess bond order beyond one per neighbour: zero for methane's
        # carbon, one for the carbon in a carbonyl.
        commitment = torch.clamp(
            torch.sum(taper * (order - 1.0) * mask, dim=1), min=0.0
        )

        # Both ends have to agree, because the bond energy is assembled from
        # a half contribution in each atom's row. Keying on the partner alone
        # softens only the half seen from the hydrogen and delivers exactly
        # half the intended effect.
        #
        # Smoothed rather than a bare maximum: max() is continuous in value
        # but not in slope where the two commitments cross, which would put a
        # force flip on that surface. Same treatment, and same reasoning, as
        # the two-contact minimum in the transfer gate.
        own = commitment[:, None]
        partner = self._gather_neighbours(
            commitment, neighbours, "commitment"
        )
        gap = own - partner
        pair_commitment = 0.5 * (
            own + partner
            + torch.sqrt(gap * gap + self.environment_softening_epsilon)
        )

        # Only bonds that are themselves single are discounted: a double
        # bond's depth already comes from the double table and must not be
        # reduced twice.
        single_character = torch.clamp(1.0 - lower, 0.0, 1.0)

        factor = 1.0 - (
            self.environment_softening
            * single_character
            * torch.clamp(pair_commitment, 0.0, 1.0)
        )

        if cache_key is not None:
            self._softening_cache = (cache_key, factor)
        return factor

    def over_coordination_scale(self, taper, pair_depth, mask,
                                cache_key=None):
        """Per-atom multiplier on the over-coordination penalty.

        That penalty is most of every activation barrier the model has, and
        it is a single constant for every element. So nothing in the barrier
        distinguishes one bond from another, and reactions that should differ
        by hundreds of meV come out within tens.

        The cost of crowding an extra partner onto an atom is not a constant
        in reality: it is larger when the bonds already present are strong.
        So the penalty is scaled by the mean depth of the atom's own
        contacts, weighted by how much each contact is actually engaged, and
        measured against a reference depth at which the scale is one.

        Element agnostic, and no new table: the depths come from the same
        entries the bond energies already use. Returns exactly one when the
        weight is zero.
        """
        if self.over_depth_weight <= 0.0:
            return torch.ones(
                taper.shape[0], device=taper.device, dtype=taper.dtype
            )

        if cache_key is not None:
            cache = getattr(self, "_over_scale_cache", None)
            if cache is not None and cache[0] is cache_key:
                return cache[1]

        # Engagement-weighted mean depth of this atom's contacts. An atom
        # with no contacts at all falls back to the reference, so its scale
        # is one and the penalty is unchanged -- it has no over-coordination
        # to be penalised for in any case.
        weight = taper * mask
        total = torch.sum(weight, dim=1)
        mean_depth = torch.where(
            total > 1e-9,
            torch.sum(weight * pair_depth, dim=1)
            / torch.clamp(total, min=1e-9),
            torch.full_like(total, self.over_reference_depth),
        )

        ratio = torch.sqrt(mean_depth / self.over_reference_depth)
        scale = 1.0 + self.over_depth_weight * (ratio - 1.0)

        # Kept positive: a deep enough well should make crowding harder, not
        # turn the penalty into a reward.
        scale = torch.clamp(scale, min=0.05)

        if cache_key is not None:
            self._over_scale_cache = (cache_key, scale)
        return scale

    def energy_per_atom(self, positions):
        # Mirrors reactive.potential_energy term for term, in the
        # padded neighbour-table form.
        #
        # Kept per atom rather than summed, so that several
        # independent boxes held in one tensor can each be given
        # their own total. Summing it reproduces the old result
        # exactly: every term here was already a sum over atoms.

        neighbours = self.neighbours
        mask = self._neighbour_weight

        offsets = (
            self._gather_neighbours(positions, neighbours, "positions")
            - positions[:, None, :]
        )

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

        # Read-only diagnostics reuse the exact pair state already evaluated
        # for forces. Detached views do not retain the autograd graph and are
        # replaced by the next force calculation.
        if not getattr(self, "_suppress_force_diagnostic_caches", False):
            self._chemical_pair_cache = {
                "neighbours": neighbours,
                "mask": self.neighbour_mask,
                "distances": distances.detach(),
                "inner": pair_inner.detach(),
                "taper": taper.detach(),
            }

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

        spare_other = self._gather_neighbours(spare, neighbours, "spare")

        weighted = taper * spare_other

        totals = torch.sum(weighted, dim=1)

        totals_other = self._gather_neighbours(totals, neighbours, "totals")

        onset = 1e-4
        share_fraction = torch.clamp(totals / onset, 0.0, 1.0)
        share_gate = share_fraction ** 2 * (3.0 - 2.0 * share_fraction)
        share_out = (
            spare[:, None] * weighted
            / torch.clamp(totals[:, None], min=1e-12)
            * share_gate[:, None]
        )

        share_fraction_other = torch.clamp(
            totals_other / onset, 0.0, 1.0
        )
        share_gate_other = (
            share_fraction_other ** 2
            * (3.0 - 2.0 * share_fraction_other)
        )
        share_back = (
            spare_other * (taper * spare[:, None])
            / torch.clamp(totals_other, min=1e-12)
            * share_gate_other
        )

        extra = torch.minimum(share_out, share_back) * taper

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

        # ---- environment softening of single bonds ----
        #
        # Factored into a method because the high fidelity correction has to
        # apply exactly the same factor. That correction rebuilds the Morse
        # terms in order to subtract the local picture the base already
        # counted, so if the two computed depths differently the subtraction
        # would remove an energy that was never added, and the transfer
        # surface would be a hybrid of two potentials.
        # Kept before softening, for the over-coordination scale below. That
        # scale is meant to read how strong an atom's bonds are as an element
        # property; reading the softened value would apply the environment
        # discount twice, once in the well depth and again in the barrier.
        unsoftened_depth = pair_depth
        pair_depth = pair_depth * self.environment_softening_factor(
            taper, order, lower, mask, neighbours, cache_key=positions
        )

        shift = distances - pair_length

        repulsive = pair_depth * torch.exp(-2.0 * pair_width * shift)
        attractive = 2.0 * pair_depth * torch.exp(
            -pair_width * shift
        )

        if getattr(self, "_share_reactive_intermediates", False):
            self._reactive_intermediates = (positions, {
                "neighbours": neighbours,
                "mask": mask,
                "distances": distances,
                "centre_types": centre_types,
                "other_types": other_types,
                "taper": taper,
                "coordination": coordination,
                "valence": valence,
                "order": order,
                "lower": lower,
                "upper": upper,
                "pair_length": pair_length,
                "pair_depth": pair_depth,
                "unsoftened_depth": unsoftened_depth,
                "pair_width": pair_width,
                "shift": shift,
                "repulsive": repulsive,
            })

        pair_energy = taper * (repulsive - attractive)

        # Halved because each bond is counted from both ends.

        bond_per_atom = 0.5 * torch.sum(pair_energy, dim=1)

        # ---- over-coordination penalty ----
        #
        # The whole of the model's activation barriers. Halfway
        # through a transfer the moving atom touches two partners
        # at once; without this the halfway point is the most
        # stable place to be and nothing has a barrier at all.

        excess = torch.clamp(coordination - valence, min=0.0)

        over_per_atom = (
            self.over_penalty
            * self.over_coordination_scale(
                taper, unsoftened_depth, mask, cache_key=positions
            )
            * excess ** 2
        )

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

        angle_pair_taper = taper[:, :, None] * taper[:, None, :]

        # A new angle needs both of its bonds to be mature. The old product
        # made an emerging contact impose geometry too early: in CH3 + CH3,
        # six H-C-C angles overwhelmed the attractive C-C force at the edge
        # of the cutoff. Use a smoothly rounded weaker-contact taper so the
        # angle grows with the less established bond without a force cusp
        # when the two contacts exchange roles.
        first_taper = taper[:, :, None]
        second_taper = taper[:, None, :]
        taper_difference = first_taper - second_taper
        weaker_taper = 0.5 * (
            first_taper
            + second_taper
            - torch.sqrt(taper_difference ** 2 + 1e-8)
            + 1e-4
        )

        # Existing lone-pair directionality already resists extra contacts
        # around oxygen and, more weakly, nitrogen. Preserve that restraint
        # while letting centres without lone pairs acquire new geometry only
        # as the new bond matures. Complete bonds and settled molecules are
        # unchanged because weaker_taper is one there.
        lone_pair_directionality = torch.clamp(
            0.5 * lone_pairs, 0.0, 1.0
        )[:, None, None]
        angle_engagement = (
            weaker_taper
            + (1.0 - weaker_taper) * lone_pair_directionality
        )
        weight = angle_pair_taper * angle_engagement

        upper_triangle = getattr(self, "_angle_upper_triangle", None)
        if upper_triangle is None:
            upper_triangle = torch.triu(
                torch.ones(
                    weight.shape[1],
                    weight.shape[2],
                    device=self.device,
                    dtype=self.dtype
                ),
                diagonal=1
            )
            self._angle_upper_triangle = upper_triangle

        angle_energy = (
            0.5
            * stiffness[:, None, None]
            * weight
            * upper_triangle
            * (angle - rest[:, None, None]) ** 2
        )

        angle_per_atom = torch.sum(angle_energy, dim=(1, 2))

        # Temporary: stash the pieces so a scratch script can see which term
        # a barrier is made of. Delete once measured.
        if not getattr(self, "_suppress_force_diagnostic_caches", False):
            self._energy_parts = {
                "bond": bond_per_atom.detach(),
                "over": over_per_atom.detach(),
                "angle": angle_per_atom.detach(),
            }
        if getattr(self, "_profile_energy_part_gradients", False):
            self._profile_energy_parts = {
                "bond": bond_per_atom,
                "over": over_per_atom,
                "angle": angle_per_atom,
            }

        return bond_per_atom + over_per_atom + angle_per_atom

    def replace_atoms(self, slots, symbols, positions):
        # Swap the atoms in these slots for different ones.
        #
        # Replacing rather than adding and removing keeps the box
        # the same size, which matters because every recording is
        # a fixed-shape array and every tool downstream assumes
        # the atom list does not change length. It is also the
        # more honest picture: the box stands for a region of a
        # much larger system at steady density, with material
        # passing through rather than piling up.

        import numpy as np

        if not len(slots):
            return

        index = torch.tensor(
            np.asarray(slots, dtype=np.int64),
            device=self.device,
            dtype=torch.long,
        )

        new_types = R.types_from_symbols(symbols)
        new_masses = R.masses_from_symbols(symbols)

        self.positions[index] = torch.tensor(
            np.asarray(positions, dtype=float) % self.box_size,
            device=self.device,
            dtype=self.dtype,
        )

        # Arriving already at the temperature the box is held at,
        # rather than as a cold lump that would have to be warmed
        # by everything around it.

        scale = np.sqrt(
            8.617333e-5 * self.target_temperature
            / (new_masses * 103.642)
        )

        drawn = (
            self.random_generator.normal(size=(len(symbols), 3))
            * scale[:, None]
        )

        self.velocities[index] = torch.tensor(
            drawn, device=self.device, dtype=self.dtype
        )

        self.masses[index] = torch.tensor(
            new_masses, device=self.device, dtype=self.dtype
        )

        self.types[index] = torch.tensor(
            new_types, device=self.device, dtype=torch.long
        )

        self.types_numpy = self.types_numpy.copy()
        self.types_numpy[np.asarray(slots)] = new_types

        self.symbols = list(self.symbols)

        for slot, symbol in zip(slots, symbols):
            self.symbols[slot] = symbol

        # Everything about where these atoms were is now wrong.

        self.reference_positions = None

        self.build_neighbours()

        self.forces, self._potential_energy = self.compute_forces()

    def limit_move(self, movement):
        # Trims any atom that would travel further than the limit
        # in one step, leaving its direction alone.
        #
        # Velocity Verlet conserves energy well, but only while
        # the potential is smooth across a timestep. Morse
        # repulsion is not: two atoms that end a step inside each
        # other feel thousands of electronvolts per angstrom, and
        # the next step throws them across the box. Measured over
        # forty runs that happened in about one in seven, each
        # time adding hundreds of eV that no chemistry supplied.
        #
        # Trimming the step turns that throw into a shove. Nothing
        # in ordinary running approaches the limit, so no normal
        # trajectory changes; it acts only in the moments the
        # integrator could not have handled anyway, and a shove is
        # a far better approximation than an explosion.

        if self.maximum_step <= 0:
            self.last_capped_atoms = ()
            self._last_move_capped_mask = torch.zeros(
                len(movement), device=movement.device, dtype=torch.bool
            )
            return movement

        distances = torch.linalg.norm(movement, dim=1)

        scale = torch.clamp(
            self.maximum_step
            / torch.clamp(distances, min=1e-12),
            max=1.0,
        )

        caught_mask = scale < 1.0
        self._last_move_capped_mask = caught_mask
        caught = int(torch.count_nonzero(caught_mask))

        self.last_capped_atoms = tuple(
            torch.nonzero(caught_mask, as_tuple=False)
            .flatten().detach().cpu().tolist()
        )

        if caught:
            self.capped_steps += caught
            self.capped_atom_counts[np.asarray(self.last_capped_atoms)] += 1

        return movement * scale[:, None]

    def compute_forces(self):
        positions = self.positions.detach().requires_grad_(True)

        compiled = getattr(self, "_compiled_energy_per_atom", None)
        per_atom = (
            compiled(positions) if compiled is not None
            else self.energy_per_atom(positions)
        )
        total = torch.sum(per_atom)

        gradient, = torch.autograd.grad(total, positions)

        # Keep the already-evaluated per-atom energies for batched reporting.
        # A force calculation necessarily constructs these values, so running
        # energy_per_atom again just to split the total by box duplicates the
        # full potential. Track the exact source tensor and its in-place
        # version so callers that replace or edit positions still get a fresh
        # evaluation rather than a stale result.
        self._potential_per_atom = per_atom.detach()
        self._potential_cache_source = self.positions
        self._potential_cache_version = self.positions._version
        if not getattr(self, "_suppress_force_diagnostic_caches", False):
            self._chemical_pair_cache_source = self.positions
            self._chemical_pair_cache_version = self.positions._version
        else:
            self._chemical_pair_cache_source = None
            self._chemical_pair_cache_version = None

        return -gradient.detach(), total.detach()

    def enable_compiled_forces(self, mode="reduce-overhead", max_fusion_size=8):
        """Enable the experimental Triton/Inductor force path.

        This is deliberately opt-in until ensemble equivalence is established.
        It changes execution/floating-point reduction order, not the energy
        equation or parameters.
        """
        if self.device.type != "cuda":
            raise ValueError("compiled forces require a CUDA device")
        if getattr(self, "_share_reactive_intermediates", False):
            raise ValueError(
                "compiled forces are not yet validated for high-fidelity "
                "reactive corrections"
            )
        try:
            import triton  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "compiled forces require a working Triton installation"
            ) from error
        torch._inductor.config.max_fusion_size = int(max_fusion_size)
        self._suppress_force_diagnostic_caches = True
        self._compiled_energy_per_atom = torch.compile(
            self.energy_per_atom, backend="inductor", fullgraph=True,
            mode=mode,
        )
        self.forces, self._potential_energy = self.compute_forces()
        return self

    def _refresh_chemical_pair_cache(self):
        with torch.no_grad():
            neighbours = self.neighbours
            mask = self._neighbour_weight
            offsets = (
                self._gather_neighbours(
                    self.positions, neighbours, "positions"
                ) - self.positions[:, None, :]
            )
            offsets -= self.box_size * torch.round(offsets / self.box_size)
            distances = torch.sqrt(torch.clamp(
                torch.sum(offsets ** 2, dim=2), min=1e-12
            ))
            centre_types = self.types[:, None].expand_as(neighbours)
            other_types = self.types[neighbours]
            inner = self.cutoff_inner[centre_types, other_types]
            outer = self.cutoff_outer[centre_types, other_types]
            fraction = torch.clamp(
                (distances - inner) / torch.clamp(outer - inner, min=1e-9),
                0.0, 1.0,
            )
            taper = 0.5 * (1.0 + torch.cos(np.pi * fraction)) * mask
            self._chemical_pair_cache = {
                "neighbours": neighbours, "mask": self.neighbour_mask,
                "distances": distances, "inner": inner, "taper": taper,
            }
            self._chemical_pair_cache_source = self.positions
            self._chemical_pair_cache_version = self.positions._version

    def chemical_observation(self):
        """Return compact current pair diagnostics without recomputation."""
        cache = getattr(self, "_chemical_pair_cache", None)
        if (
            cache is None
            or getattr(self, "_chemical_pair_cache_source", None)
            is not self.positions
            or getattr(self, "_chemical_pair_cache_version", None)
            != self.positions._version
        ):
            if getattr(self, "_compiled_energy_per_atom", None) is None:
                raise RuntimeError("chemical observation requested before forces")
            self._refresh_chemical_pair_cache()
            cache = self._chemical_pair_cache
        neighbours = cache["neighbours"]
        centres = torch.arange(
            len(self.positions), device=self.device, dtype=torch.long
        )[:, None].expand_as(neighbours)
        keep = cache["mask"] & (centres < neighbours) & (cache["taper"] > 0)
        values = torch.stack((
            centres[keep].to(self.dtype),
            neighbours[keep].to(self.dtype),
            cache["distances"][keep],
            cache["inner"][keep],
            cache["taper"][keep],
        ), dim=1).detach().cpu().numpy()
        return {
            "first": values[:, 0].astype(np.int32, copy=False),
            "second": values[:, 1].astype(np.int32, copy=False),
            "distance": values[:, 2],
            "inner": values[:, 3],
            "taper": values[:, 4],
        }

    def relax(self, steps=300, maximum_force=25.0,
              step_size=0.002):
        # Nudges every atom downhill, with each one limited to a
        # small move per step. Capping per atom rather than
        # globally matters: scaling by the single largest force
        # would reduce every other force to nothing whenever one
        # pair is badly overlapped, and that pair would never
        # separate.

        for _ in range(steps):
            if self.needs_rebuild():
                self.build_neighbours()

            forces, _ = self.compute_forces()

            magnitudes = torch.linalg.norm(forces, dim=1)

            scale = torch.clamp(
                maximum_force
                / torch.clamp(magnitudes, min=1e-12),
                max=1.0,
            )

            self.positions = (
                self.positions + forces * scale[:, None] * step_size
            ) % self.box_size

        self.reference_positions = None
        self.build_neighbours()

        self.forces, self._potential_energy = self.compute_forces()

        # Velocities are redrawn afterwards, since the relaxation
        # is not dynamics and whatever they were before means
        # nothing now.

        self.set_temperature(self.target_temperature)

        self.unwrapped = self.positions.clone()
        self.msd_reference = self.positions.clone()

    # --------------------------------------------------------

    def step(self, number_of_steps=1):
        dt = self.time_step

        # Convert eV/A into amu * A / fs^2.

        conversion = 1.0 / 103.642

        masses = self.masses[:, None]

        for _ in range(int(number_of_steps)):
            acceleration = self.forces * conversion / masses

            movement = self.limit_move(
                self.velocities * dt
                + 0.5 * acceleration * dt * dt
            )

            self.positions = (
                self.positions + movement
            ) % self.box_size

            # Checked every step but only acted on when
            # something has moved far enough to matter.

            if self.needs_rebuild():
                self.build_neighbours()

            new_forces, potential = self.compute_forces()

            new_acceleration = new_forces * conversion / masses

            ordinary_velocity = self.velocities + 0.5 * (
                acceleration + new_acceleration
            ) * dt
            # A capped move means this timestep did not resolve the force
            # impulse. Do not turn either the rejected old force or the still
            # potentially steep new force into kinetic energy. Keep the
            # pre-step velocity and let the freshly evaluated force act from
            # the separated position on the next ordinary step.
            capped_velocity = self.velocities
            self.velocities = torch.where(
                self._last_move_capped_mask[:, None],
                capped_velocity,
                ordinary_velocity,
            )

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
            + scale * torch.randn(
                self.velocities.shape,
                generator=self.torch_generator,
                device=self.device,
                dtype=self.dtype
            )
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
