import numpy as np

import build_box
import reactive as R


# ============================================================
# Making the box open
# ============================================================
#
# A sealed box reaches equilibrium and stops. Measured on a
# thirty picosecond run: seventy-five bonds form, sixty-eight of
# them in the first frame, and across the remaining twenty-nine
# picoseconds not one heavy bond breaks. Thirty-seven hydrogen
# molecules sit there having taken most of the hydrogen out of
# play. Heating that box, cooling it or striking it changes
# nothing, because there is no reaction left to influence.
#
# That is thermodynamics rather than a fault. Closed systems end
# at equilibrium; life does not happen in closed systems.
#
# Miller's apparatus was not a sealed flask either. It was a
# loop: water boiled, the vapour passed the spark, products
# condensed in a cold trap and drained back below. Material moved
# through the reaction zone continuously, so the reaction zone
# never settled.
#
# Three processes here, each one something the early Earth
# actually did.
#
#   Hydrogen escapes. H2 is light enough to leave the planet
#   entirely, which is why the atmosphere grew less reducing over
#   time. Taking it out of the box is modelling a real loss.
#
#   Volcanoes vent. Fresh methane, ammonia, water and carbon
#   dioxide arrived continuously rather than being present once
#   at the start.
#
#   Products leave the reaction zone. Anything that condensed in
#   Miller's trap was beyond the spark's reach, so it accumulated
#   instead of being broken apart again. That is how yield
#   builds.


def molecule_labels(positions, types, box_size, threshold=0.35):
    # Which connected group each atom belongs to, by the same
    # bonding rule the analysis uses.

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    count = len(positions)

    first, second = np.triu_indices(count, k=1)

    offsets = positions[second] - positions[first]
    offsets -= box_size * np.round(offsets / box_size)

    distances = np.linalg.norm(offsets, axis=1)

    inner = R.CUTOFF_INNER[types[first], types[second]]
    outer = R.CUTOFF_OUTER[types[first], types[second]]

    taper = R.smooth_cutoff(distances, inner, outer)

    keep = taper > threshold

    if not np.any(keep):
        return np.arange(count)

    graph = coo_matrix(
        (
            np.ones(int(keep.sum())),
            (first[keep], second[keep]),
        ),
        shape=(count, count),
    )

    _, labels = connected_components(graph, directed=False)

    return labels


def group_members(labels):
    groups = {}

    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)

    return groups


def formula_of(symbols, members):
    counts = {}

    for index in members:
        symbol = symbols[index]
        counts[symbol] = counts.get(symbol, 0) + 1

    return "".join(
        symbol + (str(counts[symbol]) if counts[symbol] > 1 else "")
        for symbol in ["C", "N", "O", "H"]
        if symbol in counts
    )


def choose_escaping(symbols, labels, wanted, generator,
                    escapes=("H2",)):
    # Which whole molecules leave. Only complete molecules go: a
    # fragment of one would leave a dangling bond behind, which
    # is not what escape means.

    groups = group_members(labels)

    candidates = []

    for label, members in groups.items():
        if formula_of(symbols, members) in escapes:
            candidates.append(members)

    if not candidates:
        return []

    generator.shuffle(candidates)

    return candidates[:wanted]


class OpenBox:
    # Material flows through rather than accumulating.
    #
    # Atoms are replaced rather than added and removed, so the box
    # holds the same number throughout. That keeps every recording
    # a fixed size and every downstream tool working unchanged,
    # and it is the more honest picture anyway: the box stands for
    # a region of a much larger system at steady density, with
    # material passing through it. What leaves is balanced by what
    # arrives, which is what a planet with volcanoes venting and
    # hydrogen escaping actually looks like.
    #
    # Applied between chunks of stepping rather than inside the
    # integrator, so nothing changes underneath a half-finished
    # step.

    def __init__(self, escape_per_ps=0.0, escapes=("H2",),
                 feed=None, trap_per_ps=0.0,
                 trap_minimum_heavy=3, seed=0):
        self.escape_per_ps = escape_per_ps
        self.escapes = tuple(escapes)

        # What arrives to take the place of whatever left. A
        # composition by element, in the proportions a volcano
        # vents.

        self.feed = feed or {"C": 2, "H": 5, "N": 1, "O": 1}

        self.trap_per_ps = trap_per_ps
        self.trap_minimum_heavy = trap_minimum_heavy

        self.generator = np.random.default_rng(seed)

        self.escaped = {}
        self.trapped = []
        self.replaced = 0

        # Slots replaced by the most recent choose/apply call. The
        # batch runner uses this to give each incoming atom a fresh
        # identity, including H -> H replacements.
        self.last_replaced_slots = []

        self.last_time = 0.0

    @property
    def active(self):
        return bool(self.escape_per_ps or self.trap_per_ps)

    def due(self, rate, elapsed_ps):
        # Rates are per picosecond and a chunk is a fraction of
        # one, so the whole number of events is drawn rather than
        # rounded down. Otherwise anything below one per
        # picosecond would round to nothing every time and never
        # happen at all.

        expected = rate * elapsed_ps

        whole = int(expected)

        if self.generator.random() < expected - whole:
            whole += 1

        return whole

    def fresh_symbols(self, count):
        # Elements drawn in the proportions being vented.

        names = list(self.feed)

        weights = np.array(
            [float(self.feed[name]) for name in names]
        )

        weights = weights / weights.sum()

        return list(
            self.generator.choice(names, size=count, p=weights)
        )

    def free_positions(self, positions, box_size, count,
                       minimum=3.0, attempts=120):
        # Somewhere for the new atoms to arrive that is clear of
        # everything already there.
        #
        # Clear means further than bonding range, not merely not
        # overlapping. Dropped at 1.6 angstroms an arriving carbon
        # lands already touching two or three neighbours and is
        # counted as over-coordinated the moment it appears, which
        # showed up in the first open run as atoms exceeding their
        # valence in molecules that had done nothing wrong. Beyond
        # about 2.5 the taper is zero, so at 3.0 an atom arrives
        # genuinely alone and has to move to find a partner.

        chosen = []

        existing = positions % box_size

        for _ in range(count):
            best = None
            best_gap = -1.0

            for _ in range(attempts):
                candidate = self.generator.uniform(0, box_size, 3)

                offsets = existing - candidate
                offsets -= box_size * np.round(offsets / box_size)

                gap = float(
                    np.sqrt(np.min(np.sum(offsets ** 2, axis=1)))
                )

                if gap > best_gap:
                    best_gap = gap
                    best = candidate

                if gap >= minimum:
                    break

            chosen.append(best)

            existing = np.vstack([existing, best[None, :]])

        return np.array(chosen)

    def apply(self, simulation, symbols):
        # Decide and act in one go, for a single box.

        leaving, arriving, places = self.choose(
            simulation.positions_numpy,
            symbols,
            simulation.box_size,
            simulation.elapsed_femtoseconds,
        )

        if not leaving:
            return symbols

        simulation.replace_atoms(leaving, arriving, places)

        symbols = list(symbols)

        for slot, symbol in zip(leaving, arriving):
            symbols[slot] = symbol

        return symbols

    def choose(self, positions, symbols, box_size, now):
        # Work out what leaves and what arrives, without touching
        # the simulation. Kept separate so a group of boxes can
        # each decide and then be updated together, which rebuilds
        # the neighbour table once instead of once per box.

        self.last_replaced_slots = []

        elapsed_ps = (now - self.last_time) / 1000.0

        if elapsed_ps <= 0:
            return [], [], []

        self.last_time = now

        types = R.types_from_symbols(symbols)

        labels = None

        leaving = []

        # ---- hydrogen escaping ----

        if self.escape_per_ps:
            wanted = self.due(self.escape_per_ps, elapsed_ps)

            if wanted:
                labels = molecule_labels(
                    positions, types, box_size
                )

                for members in choose_escaping(
                    symbols, labels, wanted, self.generator,
                    self.escapes,
                ):
                    formula = formula_of(symbols, members)

                    self.escaped[formula] = (
                        self.escaped.get(formula, 0) + 1
                    )

                    leaving.extend(members)

        # ---- products condensing out of reach ----

        if self.trap_per_ps:
            wanted = self.due(self.trap_per_ps, elapsed_ps)

            if wanted:
                if labels is None:
                    labels = molecule_labels(
                        positions, types, box_size
                    )

                groups = group_members(labels)

                already = set(leaving)

                candidates = []

                for label, members in groups.items():
                    if already & set(members):
                        continue

                    heavy = sum(
                        1 for index in members
                        if symbols[index] != "H"
                    )

                    if heavy >= self.trap_minimum_heavy:
                        candidates.append((heavy, members))

                # Heaviest first: a trap catches what condenses,
                # and the larger a molecule is the more readily it
                # does.

                candidates.sort(key=lambda item: -item[0])

                for heavy, members in candidates[:wanted]:
                    self.trapped.append(
                        formula_of(symbols, members)
                    )

                    leaving.extend(members)

        if not leaving:
            return [], [], []

        leaving = sorted(set(leaving))

        # ---- fresh material takes their place ----

        arriving = self.fresh_symbols(len(leaving))

        keep = np.ones(len(symbols), dtype=bool)
        keep[np.array(leaving)] = False

        places = self.free_positions(
            positions[keep], box_size, len(leaving)
        )

        self.replaced += len(leaving)
        self.last_replaced_slots = list(leaving)

        return leaving, arriving, places

    def report(self):
        lines = []

        if self.escaped:
            lines.append(
                "escaped: "
                + ", ".join(
                    f"{name} x{number}"
                    for name, number in sorted(self.escaped.items())
                )
            )

        if self.trapped:
            counts = {}

            for name in self.trapped:
                counts[name] = counts.get(name, 0) + 1

            ordered = sorted(
                counts.items(), key=lambda item: -item[1]
            )

            lines.append(
                f"trapped {len(self.trapped)}: "
                + ", ".join(
                    f"{name} x{number}"
                    for name, number in ordered[:10]
                )
            )

        if self.replaced:
            lines.append(
                f"{self.replaced} atoms replaced by fresh material"
            )

        return lines