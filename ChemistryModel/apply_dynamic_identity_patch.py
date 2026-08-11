from pathlib import Path
import shutil
import datetime

ROOT = Path(__file__).resolve().parent

FILES = [
    "recorder.py",
    "bonding.py",
    "open_box.py",
    "batch_runner.py",
    "analysis.py",
    "analysis_cache.py",
]


def replace_once(text, old, new, label):
    if new in text:
        print(f"  already applied: {label}")
        return text

    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly 1 matching block, found {count}. "
            "Your file differs from the master version this patch was built for."
        )

    print(f"  patching: {label}")
    return text.replace(old, new, 1)


def patch_recorder(text):
    text = replace_once(
        text,
        '''        self.types = R.types_from_symbols(self.symbols)

        # How many captures are skipped between stored frames.
''',
        '''        self.types = np.asarray(
            R.types_from_symbols(self.symbols), dtype=np.uint8
        )

        # A slot is only a place in the fixed-size arrays. In an
        # open box, the atom occupying that slot can leave and be
        # replaced by a completely new atom, even by another atom
        # of the same element. IDs preserve that distinction.
        self.atom_ids = np.arange(len(self.symbols), dtype=np.uint32)

        # How many captures are skipped between stored frames.
''',
        "recorder: initial atom IDs",
    )

    text = replace_once(
        text,
        '''        self.positions = []
        self.velocities = []

        # The cell can change during a run, since the viewer can
''',
        '''        self.positions = []
        self.velocities = []

        # Element type and atom identity at every stored frame.
        # These are essential for open boxes: coordinates alone
        # cannot tell whether slot 143 still contains the original
        # hydrogen or a newly arrived carbon (or a new hydrogen).
        self.frame_types = []
        self.frame_atom_ids = []

        # The cell can change during a run, since the viewer can
''',
        "recorder: per-frame identity buffers",
    )

    text = replace_once(
        text,
        '''    def capture(self, positions, time, potential, kinetic,
                temperature, velocities=None, box_size=None):
''',
        '''    def capture(self, positions, time, potential, kinetic,
                temperature, velocities=None, box_size=None,
                symbols=None, types=None, atom_ids=None):
''',
        "recorder: capture signature",
    )

    text = replace_once(
        text,
        '''        self.positions.append(
            np.asarray(positions, dtype=np.float32).copy()
        )

        # Velocities are kept so a run can be picked up again
''',
        '''        frame_positions = np.asarray(
            positions, dtype=np.float32
        ).copy()

        self.positions.append(frame_positions)

        if types is None:
            if symbols is None:
                frame_types = self.types
            else:
                frame_types = R.types_from_symbols(symbols)
        else:
            frame_types = types

        frame_types = np.asarray(frame_types, dtype=np.uint8).copy()

        if len(frame_types) != len(frame_positions):
            raise ValueError(
                "Recorder.capture received a different number of "
                "atom types and positions."
            )

        if atom_ids is None:
            frame_atom_ids = self.atom_ids
        else:
            frame_atom_ids = atom_ids

        frame_atom_ids = np.asarray(
            frame_atom_ids, dtype=np.uint32
        ).copy()

        if len(frame_atom_ids) != len(frame_positions):
            raise ValueError(
                "Recorder.capture received a different number of "
                "atom IDs and positions."
            )

        self.frame_types.append(frame_types)
        self.frame_atom_ids.append(frame_atom_ids)

        # Velocities are kept so a run can be picked up again
''',
        "recorder: capture frame types and IDs",
    )

    text = replace_once(
        text,
        '''        self.positions = self.positions[::2]
        self.velocities = self.velocities[::2]
        self.box_sizes = self.box_sizes[::2]
''',
        '''        self.positions = self.positions[::2]
        self.velocities = self.velocities[::2]
        self.frame_types = self.frame_types[::2]
        self.frame_atom_ids = self.frame_atom_ids[::2]
        self.box_sizes = self.box_sizes[::2]
''',
        "recorder: thin identity history",
    )

    text = replace_once(
        text,
        '''        self.positions.clear()
        self.velocities.clear()
        self.box_sizes.clear()
''',
        '''        self.positions.clear()
        self.velocities.clear()
        self.frame_types.clear()
        self.frame_atom_ids.clear()
        self.box_sizes.clear()
''',
        "recorder: clear identity history",
    )

    text = replace_once(
        text,
        '''    # --------------------------------------------------------
    def bonds_at(self, index, threshold=0.35):
''',
        '''    # --------------------------------------------------------
    def types_at(self, index):
        # Old recordings predate changing atom identities. For
        # those, the initial types are the only information that
        # exists, and are therefore the best possible fallback.
        if index < len(self.frame_types):
            return np.asarray(self.frame_types[index], dtype=np.uint8)

        return self.types

    def atom_ids_at(self, index):
        # Same compatibility rule as types_at(). A legacy sealed
        # recording has one atom per slot for its whole life, so
        # slot number is a perfectly good identity.
        if index < len(self.frame_atom_ids):
            return np.asarray(
                self.frame_atom_ids[index], dtype=np.uint32
            )

        return self.atom_ids

    def symbols_at(self, index):
        return [
            str(R.ELEMENTS[int(atom_type)])
            for atom_type in self.types_at(index)
        ]

    @property
    def has_atom_history(self):
        return (
            len(self.frame_types) == len(self.positions) > 0
            and len(self.frame_atom_ids) == len(self.positions)
        )

    # --------------------------------------------------------
    def bonds_at(self, index, threshold=0.35):
''',
        "recorder: frame lookup helpers",
    )

    text = replace_once(
        text,
        '''        positions = self.positions[index]

        box = self.box_at(index)

        count = len(positions)
''',
        '''        positions = self.positions[index]
        types = self.types_at(index)

        box = self.box_at(index)

        count = len(positions)
''',
        "recorder: bonds use frame types",
    )

    text = replace_once(
        text,
        '''            partners = self.types[i + 1:]

            inner = R.CUTOFF_INNER[self.types[i], partners]
            outer = R.CUTOFF_OUTER[self.types[i], partners]
''',
        '''            partners = types[i + 1:]

            inner = R.CUTOFF_INNER[types[i], partners]
            outer = R.CUTOFF_OUTER[types[i], partners]
''',
        "recorder: dynamic bond cutoffs",
    )

    text = replace_once(
        text,
        '''        first, second = self.bonds_at(index)

        count = len(self.positions[index])
''',
        '''        first, second = self.bonds_at(index)
        types = self.types_at(index)

        count = len(self.positions[index])
''',
        "recorder: formulas frame types",
    )

    text = replace_once(
        text,
        '''                symbol = R.ELEMENTS[self.types[member]]
''',
        '''                symbol = R.ELEMENTS[types[member]]
''',
        "recorder: formulas dynamic symbols",
    )

    text = replace_once(
        text,
        '''        if len(self.box_sizes) == len(self.positions):
            extra["box_sizes"] = np.array(
                self.box_sizes, dtype=np.float32
            )
        np.savez_compressed(
''',
        '''        if len(self.box_sizes) == len(self.positions):
            extra["box_sizes"] = np.array(
                self.box_sizes, dtype=np.float32
            )

        if len(self.frame_types) == len(self.positions):
            extra["frame_types"] = np.array(
                self.frame_types, dtype=np.uint8
            )

        if len(self.frame_atom_ids) == len(self.positions):
            extra["frame_atom_ids"] = np.array(
                self.frame_atom_ids, dtype=np.uint32
            )

        np.savez_compressed(
''',
        "recorder: save identity history",
    )

    text = replace_once(
        text,
        '''        recorder.positions = [
            frame for frame in data["positions"]
        ]
        if "box_sizes" in data.files:
''',
        '''        recorder.positions = [
            frame for frame in data["positions"]
        ]

        if "frame_types" in data.files:
            recorder.frame_types = [
                np.asarray(frame, dtype=np.uint8)
                for frame in data["frame_types"]
            ]

        if "frame_atom_ids" in data.files:
            recorder.frame_atom_ids = [
                np.asarray(frame, dtype=np.uint32)
                for frame in data["frame_atom_ids"]
            ]

        if "box_sizes" in data.files:
''',
        "recorder: load identity history",
    )

    text = replace_once(
        text,
        '''                for symbol, point in zip(self.symbols, frame):
''',
        '''                for symbol, point in zip(
                    self.symbols_at(index), frame
                ):
''',
        "recorder: dynamic XYZ export",
    )

    return text


def patch_bonding(text):
    text = replace_once(
        text,
        '''    def __init__(self, types, formation_time=FORMATION_TIME,
                 breaking_time=BREAKING_TIME, threshold=0.35):
        self.types = types
        self.formation_time = formation_time
''',
        '''    def __init__(self, types, formation_time=FORMATION_TIME,
                 breaking_time=BREAKING_TIME, threshold=0.35,
                 atom_ids=None):
        self.types = np.asarray(types, dtype=int).copy()
        self.formation_time = formation_time
''',
        "bonding: constructor accepts atom IDs",
    )

    text = replace_once(
        text,
        '''        count = len(types)

        self.first, self.second = np.triu_indices(count, k=1)
        self.inner = R.CUTOFF_INNER[
            types[self.first], types[self.second]
        ]
        self.outer = R.CUTOFF_OUTER[
            types[self.first], types[self.second]
        ]
''',
        '''        count = len(self.types)

        if atom_ids is None:
            self.atom_ids = np.arange(count, dtype=np.uint32)
        else:
            self.atom_ids = np.asarray(
                atom_ids, dtype=np.uint32
            ).copy()

        if len(self.atom_ids) != count:
            raise ValueError("BondTracker types and atom IDs differ in length.")

        self.first, self.second = np.triu_indices(count, k=1)
        self.inner = R.CUTOFF_INNER[
            self.types[self.first], self.types[self.second]
        ]
        self.outer = R.CUTOFF_OUTER[
            self.types[self.first], self.types[self.second]
        ]
''',
        "bonding: initialise identity state",
    )

    text = replace_once(
        text,
        '''        self.formed_at = np.zeros(len(self.first))

        self.events = []

    def in_range(self, positions, box_size):
''',
        '''        self.formed_at = np.zeros(len(self.first))
        self.arrived_at = np.zeros(len(self.first))

        self.events = []

    def set_atoms(self, types=None, atom_ids=None):
        # A fixed array slot can receive a new atom in an open box.
        # Any persistence history involving that slot belongs to the
        # atom that left and must not be inherited by the newcomer.
        new_types = (
            self.types if types is None
            else np.asarray(types, dtype=int)
        )
        new_ids = (
            self.atom_ids if atom_ids is None
            else np.asarray(atom_ids, dtype=np.uint32)
        )

        if len(new_types) != len(self.types):
            raise ValueError("BondTracker atom count changed.")
        if len(new_ids) != len(self.atom_ids):
            raise ValueError("BondTracker atom ID count changed.")

        changed_atoms = (
            (new_types != self.types)
            | (new_ids != self.atom_ids)
        )

        affected = (
            changed_atoms[self.first]
            | changed_atoms[self.second]
        )

        if np.any(affected):
            self.bonded[affected] = False
            self.in_range_for[affected] = 0.0
            self.out_of_range_for[affected] = np.inf
            self.formed_at[affected] = 0.0
            self.arrived_at[affected] = 0.0

        self.types = new_types.copy()
        self.atom_ids = new_ids.copy()

        self.inner = R.CUTOFF_INNER[
            self.types[self.first], self.types[self.second]
        ]
        self.outer = R.CUTOFF_OUTER[
            self.types[self.first], self.types[self.second]
        ]

        return affected

    def in_range(self, positions, box_size):
''',
        "bonding: reset only replaced atom histories",
    )

    text = replace_once(
        text,
        '''    def update(self, positions, box_size, time):
        # Advance the tracker by one frame and return the pairs
        # currently counted as bonded.

        if self.last_time is None:
''',
        '''    def update(self, positions, box_size, time,
               types=None, atom_ids=None):
        # Advance the tracker by one frame and return the pairs
        # currently counted as bonded.

        affected = self.set_atoms(types=types, atom_ids=atom_ids)

        if self.last_time is None:
''',
        "bonding: update accepts current identity",
    )

    text = replace_once(
        text,
        '''        self.arrived_at = getattr(
            self, "arrived_at", np.zeros(len(self.first))
        )

        self.arrived_at = np.where(
            arriving, float(time), self.arrived_at
        )

        forming = (
''',
        '''        self.arrived_at = np.where(
            arriving, float(time), self.arrived_at
        )

        # Time before a replacement happened cannot count toward a
        # bond made by the new atom. Even if the newcomer is already
        # close in its first saved frame, its clock starts now.
        if np.any(affected):
            self.in_range_for[affected] = 0.0
            self.out_of_range_for[affected] = np.where(
                close[affected], 0.0, np.inf
            )
            self.arrived_at[affected & close] = float(time)

        forming = (
''',
        "bonding: replacement clock starts at zero",
    )

    return text


def patch_open_box(text):
    text = replace_once(
        text,
        '''        self.replaced = 0

        self.last_time = 0.0
''',
        '''        self.replaced = 0

        # Which fixed array slots received brand-new atoms during
        # the most recent choose/apply call.
        self.last_replaced_slots = []

        self.last_time = 0.0
''',
        "open_box: expose replaced slots",
    )

    text = replace_once(
        text,
        '''        elapsed_ps = (now - self.last_time) / 1000.0

        if elapsed_ps <= 0:
''',
        '''        self.last_replaced_slots = []

        elapsed_ps = (now - self.last_time) / 1000.0

        if elapsed_ps <= 0:
''',
        "open_box: reset last replacement list",
    )

    text = replace_once(
        text,
        '''        self.replaced += len(leaving)

        return leaving, arriving, places
''',
        '''        self.replaced += len(leaving)
        self.last_replaced_slots = list(leaving)

        return leaving, arriving, places
''',
        "open_box: remember replacement slots",
    )

    return text


def patch_batch_runner(text):
    text = replace_once(
        text,
        '''    symbols = list(simulation.symbols)

    next_save = (
''',
        '''    symbols = list(simulation.symbols)

    # Slots stay fixed for GPU efficiency, but their occupants do
    # not in an open box.
    atom_ids = np.arange(len(symbols), dtype=np.uint32)
    next_atom_id = len(symbols)

    next_save = (
''',
        "batch_runner: single-run atom IDs",
    )

    text = replace_once(
        text,
        '''            ),
            box_size=simulation.box_size,
        )

        if opening.active:
            symbols = opening.apply(simulation, symbols)
''',
        '''            ),
            box_size=simulation.box_size,
            symbols=symbols,
            atom_ids=atom_ids,
        )

        if opening.active:
            symbols = opening.apply(simulation, symbols)

            if opening.last_replaced_slots:
                slots = np.asarray(
                    opening.last_replaced_slots, dtype=int
                )
                count = len(slots)

                atom_ids[slots] = np.arange(
                    next_atom_id,
                    next_atom_id + count,
                    dtype=np.uint32,
                )
                next_atom_id += count
''',
        "batch_runner: single-run capture identity",
    )

    text = replace_once(
        text,
        '''    symbol_lists = [
        list(simulation.symbols_for(box))
        for box in range(len(seeds))
    ]
    while steps_done < total_steps:
''',
        '''    symbol_lists = [
        list(simulation.symbols_for(box))
        for box in range(len(seeds))
    ]

    atom_id_lists = [
        np.arange(len(symbols), dtype=np.uint32)
        for symbols in symbol_lists
    ]
    next_atom_ids = [
        len(symbols) for symbols in symbol_lists
    ]

    while steps_done < total_steps:
''',
        "batch_runner: grouped atom IDs",
    )

    text = replace_once(
        text,
        '''                velocities=simulation.velocities_for(box),
                box_size=simulation.box_size,
            )
        if openings and openings[0].active:
            symbol_lists = apply_openings_to_group(
                simulation, openings, symbol_lists
            )
''',
        '''                velocities=simulation.velocities_for(box),
                box_size=simulation.box_size,
                symbols=symbol_lists[box],
                atom_ids=atom_id_lists[box],
            )

        if openings and openings[0].active:
            symbol_lists = apply_openings_to_group(
                simulation, openings, symbol_lists
            )

            for box, opening in enumerate(openings):
                if not opening.last_replaced_slots:
                    continue

                slots = np.asarray(
                    opening.last_replaced_slots, dtype=int
                )
                count = len(slots)
                start_id = next_atom_ids[box]

                atom_id_lists[box][slots] = np.arange(
                    start_id,
                    start_id + count,
                    dtype=np.uint32,
                )
                next_atom_ids[box] += count
''',
        "batch_runner: grouped capture identity",
    )

    text = replace_once(
        text,
        '''    last = len(recorder) - 1

    if not recorder.has_velocities:
''',
        '''    last = len(recorder) - 1

    current_symbols = recorder.symbols_at(last)
    current_atom_ids = recorder.atom_ids_at(last).copy()

    if not recorder.has_velocities:
''',
        "batch_runner: continuation final identity",
    )

    text = replace_once(
        text,
        '''    simulation = ReactiveSimulation(
        symbols=recorder.symbols,
        positions=recorder.positions[last].astype(float),
''',
        '''    simulation = ReactiveSimulation(
        symbols=current_symbols,
        positions=recorder.positions[last].astype(float),
''',
        "batch_runner: continue with true final elements",
    )

    text = replace_once(
        text,
        '''            ),
            box_size=simulation.box_size,
        )
        if (
            next_strike is not None
''',
        '''            ),
            box_size=simulation.box_size,
            symbols=current_symbols,
            atom_ids=current_atom_ids,
        )
        if (
            next_strike is not None
''',
        "batch_runner: continuation records identity",
    )

    return text


def patch_analysis(text):
    text = replace_once(
        text,
        '''def analyse(recorder, stride=1, threshold=0.35, late_fs=2500.0,
            structures=True, persistent=True):
    symbols = recorder.symbols
    types = recorder.types

    # The last frame's cell, used for the final-state analysis.
    # Individual frames use their own, since the box can change
    # during a run.

    box = recorder.box_at(len(recorder) - 1)

    count = len(symbols)

    first, second = pair_indices(count)
    inner, outer = cutoffs_for(types, first, second)
    times = np.array(recorder.times)
''',
        '''def analyse(recorder, stride=1, threshold=0.35, late_fs=2500.0,
            structures=True, persistent=True):
    if len(recorder) == 0:
        raise ValueError("cannot analyse an empty recording")

    count = len(recorder.positions[0])

    first, second = pair_indices(count)
    times = np.array(recorder.times)

    initial_types = recorder.types_at(0)
    initial_ids = recorder.atom_ids_at(0)
''',
        "analysis: dynamic setup",
    )

    text = replace_once(
        text,
        '''    previous_heavy = set()
    examined = 0
''',
        '''    previous_heavy = set()
    previous_symbols_by_id = {}
    examined = 0
''',
        "analysis: identity-aware heavy history",
    )

    text = replace_once(
        text,
        '''    tracker = (
        bonding.BondTracker(types, threshold=threshold)
        if persistent else None
    )
''',
        '''    tracker = (
        bonding.BondTracker(
            initial_types,
            threshold=threshold,
            atom_ids=initial_ids,
        )
        if persistent else None
    )
''',
        "analysis: tracker receives identities",
    )

    text = replace_once(
        text,
        '''    for index in range(0, len(recorder), stride):
        positions = recorder.positions[index]

        frame_box = recorder.box_at(index)
        if tracker is not None:
            bond_first, bond_second = tracker.update(
                positions, frame_box, float(times[index])
            )
        else:
            bond_first, bond_second = bonds_in_frame(
                positions, frame_box, first, second,
                inner, outer, threshold
            )

        if float(times[index]) < census_from:
            continue
''',
        '''    indices = list(range(0, len(recorder), stride))

    # Final-state measurements must use the actual final frame even
    # when the requested stride does not land on it.
    final_index = len(recorder) - 1
    if indices[-1] != final_index:
        indices.append(final_index)

    for index in indices:
        positions = recorder.positions[index]
        frame_types = recorder.types_at(index)
        frame_symbols = recorder.symbols_at(index)
        frame_ids = recorder.atom_ids_at(index)

        frame_box = recorder.box_at(index)

        if tracker is not None:
            bond_first, bond_second = tracker.update(
                positions,
                frame_box,
                float(times[index]),
                types=frame_types,
                atom_ids=frame_ids,
            )
        else:
            frame_inner, frame_outer = cutoffs_for(
                frame_types, first, second
            )

            bond_first, bond_second = bonds_in_frame(
                positions, frame_box, first, second,
                frame_inner, frame_outer, threshold
            )

        if float(times[index]) < census_from:
            continue
''',
        "analysis: per-frame types, IDs and cutoffs",
    )

    text = replace_once(
        text,
        '''        if structures:
            table, _, _ = F.bond_table(
                positions, types, frame_box, threshold
            )
''',
        '''        if structures:
            table, _, _ = F.bond_table(
                positions, frame_types, frame_box, threshold
            )
''',
        "analysis: fingerprint uses frame types",
    )

    text = replace_once(
        text,
        '''            if structures:
                name = registry.register(symbols, members, table)
            else:
                name = formula_for(symbols, members)
''',
        '''            if structures:
                name = registry.register(
                    frame_symbols, members, table
                )
            else:
                name = formula_for(frame_symbols, members)
''',
        "analysis: species census uses frame symbols",
    )

    text = replace_once(
        text,
        '''        # Bonds between heavy atoms build structure. Hydrogen
        # comes and goes constantly and would drown the log.

        heavy_now = set()

        for a, b in zip(bond_first, bond_second):
            if symbols[a] in HEAVY and symbols[b] in HEAVY:
                heavy_now.add((int(a), int(b)))
        for pair in sorted(heavy_now - previous_heavy):
            heavy_events.append((
                float(times[index]), "formed",
                symbols[pair[0]], pair[0],
                symbols[pair[1]], pair[1]
            ))

        for pair in sorted(previous_heavy - heavy_now):
            heavy_events.append((
                float(times[index]), "broke",
                symbols[pair[0]], pair[0],
                symbols[pair[1]], pair[1]
            ))

        previous_heavy = heavy_now
''',
        '''        # Bonds between heavy atoms build structure. Hydrogen
        # comes and goes constantly and would drown the log.
        #
        # Events are keyed by atom identity, not array slot. If a
        # product leaves the open box, its internal bonds disappear
        # because the atoms left the system; that is transport, not
        # chemical bond breaking.

        symbols_by_id = {
            int(frame_ids[slot]): frame_symbols[slot]
            for slot in range(count)
        }

        heavy_now = set()

        for a, b in zip(bond_first, bond_second):
            if (
                frame_symbols[a] in HEAVY
                and frame_symbols[b] in HEAVY
            ):
                ida = int(frame_ids[a])
                idb = int(frame_ids[b])

                heavy_now.add(
                    (min(ida, idb), max(ida, idb))
                )

        for pair in sorted(heavy_now - previous_heavy):
            heavy_events.append((
                float(times[index]), "formed",
                symbols_by_id[pair[0]], pair[0],
                symbols_by_id[pair[1]], pair[1]
            ))

        for pair in sorted(previous_heavy - heavy_now):
            if (
                pair[0] not in symbols_by_id
                or pair[1] not in symbols_by_id
            ):
                continue

            heavy_events.append((
                float(times[index]), "broke",
                previous_symbols_by_id.get(
                    pair[0], symbols_by_id[pair[0]]
                ),
                pair[0],
                previous_symbols_by_id.get(
                    pair[1], symbols_by_id[pair[1]]
                ),
                pair[1],
            ))

        previous_heavy = heavy_now
        previous_symbols_by_id = symbols_by_id
''',
        "analysis: heavy events follow real atoms",
    )

    text = replace_once(
        text,
        '''    # ---- the final frame, in detail ----

    positions = recorder.positions[len(recorder) - 1]

    if tracker is not None:
        bond_first, bond_second = tracker.confirmed_now()
    else:
        bond_first, bond_second = bonds_in_frame(
            positions, box, first, second, inner, outer, threshold
        )

    labels = label_frame(count, bond_first, bond_second)

    degree = bond_counts(positions, types, box, threshold)
    if structures:
        final_table, _, _ = F.bond_table(
            positions, types, box, threshold
        )
''',
        '''    # ---- the final frame, in detail ----

    positions = recorder.positions[final_index]
    symbols = recorder.symbols_at(final_index)
    types = recorder.types_at(final_index)
    box = recorder.box_at(final_index)

    if tracker is not None:
        # final_index was deliberately included in `indices`, so
        # the tracker is aligned with these exact positions/types.
        bond_first, bond_second = tracker.confirmed_now()
    else:
        inner, outer = cutoffs_for(types, first, second)

        bond_first, bond_second = bonds_in_frame(
            positions, box, first, second, inner, outer, threshold
        )

    labels = label_frame(count, bond_first, bond_second)

    degree = bond_counts(positions, types, box, threshold)
    if structures:
        final_table, _, _ = F.bond_table(
            positions, types, box, threshold
        )
''',
        "analysis: true final-frame chemistry",
    )

    return text


def patch_analysis_cache(text):
    return replace_once(
        text,
        "CACHE_VERSION = 3",
        "CACHE_VERSION = 4",
        "analysis_cache: invalidate old cached chemistry",
    )


PATCHERS = {
    "recorder.py": patch_recorder,
    "bonding.py": patch_bonding,
    "open_box.py": patch_open_box,
    "batch_runner.py": patch_batch_runner,
    "analysis.py": patch_analysis,
    "analysis_cache.py": patch_analysis_cache,
}


def main():
    missing = [name for name in FILES if not (ROOT / name).exists()]
    if missing:
        raise SystemExit(
            "Run this from the ChemistryModel folder. Missing: "
            + ", ".join(missing)
        )

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ROOT / f"dynamic_identity_backup_{stamp}"
    backup.mkdir()

    originals = {}

    try:
        for name in FILES:
            path = ROOT / name
            originals[name] = path.read_text(encoding="utf-8")
            shutil.copy2(path, backup / name)

        changed = {}

        print("Applying dynamic atom identity patch...\n")

        for name in FILES:
            changed[name] = PATCHERS[name](originals[name])

        for name, changed_text in changed.items():
            compile(changed_text, name, "exec")

        for name, changed_text in changed.items():
            (ROOT / name).write_text(changed_text, encoding="utf-8")

    except Exception:
        for name, original_text in originals.items():
            (ROOT / name).write_text(original_text, encoding="utf-8")

        print("\nPATCH ABORTED. Original files restored.")
        print(f"Backups are also in: {backup.name}")
        raise

    print("\nDone.")
    print(f"Original files backed up to: {backup.name}")
    print()
    print("New recordings now store:")
    print("  - element type at every frame")
    print("  - atom identity at every frame")
    print()
    print("Old open-box recordings cannot be repaired exactly because they")
    print("never recorded replacement identity. Closed-box recordings remain")
    print("fully re-analysable.")


if __name__ == "__main__":
    main()
