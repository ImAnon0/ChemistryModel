import numpy as np
from collections import deque
import json

import reactive as R


# ============================================================
# Recording a run
# ============================================================
#
# Stores coordinates rather than pixels, which is both far
# smaller and far more useful: you can scrub back to any moment,
# recompute the bonds, count what had formed, and export the
# whole thing for a proper molecular viewer.
#
# Eighty atoms costs about a kilobyte a frame, so ten thousand
# frames is around ten megabytes. There is a frame limit anyway,
# and once it is reached the oldest frames drop off the front.


class Recorder:

    FORMAT_VERSION = 2

    def __init__(self, symbols, box_size, maximum_frames=40000):
        self.symbols = list(symbols)
        self.box_size = float(box_size)
        self.maximum_frames = int(maximum_frames)
        # Recorder remains the explicit legacy writer. Production batch/Lab
        # runs select AdaptiveRecorder by default, while old callers and the
        # standalone live runner retain this compatibility path.
        self.format_version = 1

        self.types = np.asarray(
            R.types_from_symbols(self.symbols), dtype=np.uint8
        )

        # Array slots stay fixed, but an open box can replace the
        # atom occupying a slot. Atom IDs distinguish a replacement
        # even when the incoming element happens to be the same.
        self.atom_ids = np.arange(
            len(self.symbols), dtype=np.uint32
        )

        # How many captures are skipped between stored frames.
        # Starts at one and doubles whenever the buffer fills.

        self.stride = 1
        self.since_last = 0
        self.thinned_count = 0

        self.positions = []
        self.velocities = []

        # Open-box chemistry can change which element occupies a
        # slot. Keep both the element type and atom identity for
        # every stored frame so later analysis can reconstruct the
        # actual chemistry rather than assuming the starting atoms
        # remained forever.
        self.frame_types = []
        self.frame_atom_ids = []

        # The cell can change during a run, since the viewer can
        # squeeze or stretch it. Storing only the size it started
        # at means anything re-analysed later wraps coordinates
        # against the wrong cell: a box saved as 12 A but actually
        # grown to 20 A folds distant atoms onto each other and
        # invents bonds between them.

        self.box_sizes = []
        self.times = []
        self.potential = []
        self.kinetic = []
        self.temperature = []
        # 0 = ordinary frame. Future adaptive captures use additional values;
        # keeping this parallel array makes the new format explicit while old
        # readers can continue ignoring it.
        self.frame_kinds = []
        self.event_reasons = []
        self.events = []
        self.adaptive_dropped_frames = 0

    def __len__(self):
        return len(self.positions)

    @property
    def is_empty(self):
        return len(self.positions) == 0

    def capture(self, positions, time, potential, kinetic,
                temperature, velocities=None, box_size=None,
                symbols=None, types=None, atom_ids=None):
        # Frames arrive at a fixed rate but are only kept every
        # `stride` of them.

        self.since_last += 1

        if self.since_last < self.stride:
            return False

        self.since_last = 0

        frame_positions = np.asarray(
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

        frame_types = np.asarray(
            frame_types, dtype=np.uint8
        ).copy()

        if len(frame_types) != len(frame_positions):
            raise ValueError(
                "Recorder.capture got a different number of "
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
                "Recorder.capture got a different number of "
                "atom IDs and positions."
            )

        self.frame_types.append(frame_types)
        self.frame_atom_ids.append(frame_atom_ids)

        # Velocities are kept so a run can be picked up again
        # exactly where it stopped. Without them a resumed run has
        # to draw fresh thermal velocities, which throws away all
        # the momentum and gives a small artificial kick.

        if velocities is not None:
            self.velocities.append(
                np.asarray(velocities, dtype=np.float32).copy()
            )

        self.box_sizes.append(
            float(box_size) if box_size is not None
            else float(self.box_size)
        )

        self.times.append(float(time))
        self.potential.append(float(potential))
        self.kinetic.append(float(kinetic))
        self.temperature.append(float(temperature))
        self.frame_kinds.append(0)
        self.event_reasons.append("")

        if len(self.positions) > self.maximum_frames:
            self.thin()

        return True

    def thin(self):
        # When the buffer fills, throw away every other frame and
        # halve the capture rate from here on.
        #
        # Dropping the oldest frames instead would lose the start
        # of the run, which is usually the part worth keeping. This
        # way the whole run survives, at steadily coarser time
        # resolution, and the buffer never grows.

        self.positions = self.positions[::2]
        self.velocities = self.velocities[::2]
        self.frame_types = self.frame_types[::2]
        self.frame_atom_ids = self.frame_atom_ids[::2]
        self.box_sizes = self.box_sizes[::2]
        self.times = self.times[::2]
        self.potential = self.potential[::2]
        self.kinetic = self.kinetic[::2]
        self.temperature = self.temperature[::2]
        self.frame_kinds = self.frame_kinds[::2]
        self.event_reasons = self.event_reasons[::2]

        self.stride *= 2
        self.thinned_count += 1

    @property
    def span_femtoseconds(self):
        if len(self.times) < 2:
            return 0.0

        return self.times[-1] - self.times[0]

    @property
    def memory_megabytes(self):
        if not self.positions:
            return 0.0

        return (
            len(self.positions)
            * self.positions[0].nbytes
            / (1024.0 * 1024.0)
        )

    def clear(self):
        self.stride = 1
        self.since_last = 0
        self.thinned_count = 0

        self.positions.clear()
        self.velocities.clear()
        self.frame_types.clear()
        self.frame_atom_ids.clear()
        self.box_sizes.clear()
        self.times.clear()
        self.potential.clear()
        self.kinetic.clear()
        self.temperature.clear()
        self.frame_kinds.clear()
        self.event_reasons.clear()
        self.events.clear()

    # --------------------------------------------------------

    def types_at(self, index):
        # Legacy recordings have no per-frame type history. That is
        # exact for sealed boxes, where atom identities never
        # change, and the best possible fallback for old files.
        if index < len(self.frame_types):
            return np.asarray(
                self.frame_types[index], dtype=np.uint8
            )

        return self.types

    def atom_ids_at(self, index):
        # For a legacy sealed run, array slot is a valid permanent
        # identity. New open-box runs store the real identity
        # history explicitly.
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

    def bonds_at(self, index, threshold=0.35):
        # Bonds are not stored. They are recomputed from the
        # coordinates, which keeps the file small and means a
        # change to the bonding rules applies to old recordings
        # too.

        positions = self.positions[index]
        types = self.types_at(index)

        box = self.box_at(index)

        count = len(positions)

        first = []
        second = []

        for i in range(count):
            offsets = positions[i + 1:] - positions[i]

            offsets -= box * np.round(offsets / box)

            distances = np.linalg.norm(offsets, axis=1)

            partners = types[i + 1:]

            inner = R.CUTOFF_INNER[types[i], partners]
            outer = R.CUTOFF_OUTER[types[i], partners]

            taper = R.smooth_cutoff(distances, inner, outer)

            found = np.where(taper > threshold)[0]

            for offset in found:
                first.append(i)
                second.append(i + 1 + int(offset))

        return np.array(first, dtype=int), np.array(second, dtype=int)

    def formulas_at(self, index):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        first, second = self.bonds_at(index)
        types = self.types_at(index)

        count = len(self.positions[index])

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
                symbol = R.ELEMENTS[types[member]]
                counts[symbol] = counts.get(symbol, 0) + 1

            name = "".join(
                symbol
                + (str(counts[symbol]) if counts[symbol] > 1 else "")
                for symbol in ["C", "N", "O", "H"]
                if symbol in counts
            )

            formulas[name] = formulas.get(name, 0) + 1

        return formulas

    # --------------------------------------------------------

    def box_at(self, index):
        # Falls back to the single stored size for recordings made
        # before the cell could change.

        if index < len(self.box_sizes):
            return self.box_sizes[index]

        return self.box_size

    @property
    def box_changed(self):
        if len(self.box_sizes) < 2:
            return False

        return (
            max(self.box_sizes) - min(self.box_sizes) > 0.01
        )

    @property
    def has_velocities(self):
        return len(self.velocities) == len(self.positions) > 0

    def save(self, path):
        extra = {}

        if self.has_velocities:
            extra["velocities"] = np.array(
                self.velocities, dtype=np.float32
            )

        if len(self.box_sizes) == len(self.positions):
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

        versioned = {}
        if self.format_version >= 2:
            versioned = {
                "format_version": np.array(
                    self.format_version, dtype=np.uint16
                ),
                "frame_kinds": np.asarray(
                    self.frame_kinds, dtype=np.uint8
                ),
            }
            versioned["event_reasons"] = np.asarray(
                self.event_reasons, dtype="U96"
            )
            versioned["adaptive_dropped_frames"] = np.array(
                self.adaptive_dropped_frames, dtype=np.uint64
            )
            versioned["events_json"] = np.asarray(
                [json.dumps(event, separators=(",", ":")) for event in self.events],
                dtype=np.str_,
            )

        np.savez_compressed(
            path,
            symbols=np.array(self.symbols),
            box_size=self.box_size,
            positions=np.array(self.positions, dtype=np.float32),
            **extra,
            **versioned,
            times=np.array(self.times),
            potential=np.array(self.potential),
            kinetic=np.array(self.kinetic),
            temperature=np.array(self.temperature),
        )

        return path


class AdaptiveRecorder(Recorder):
    """Opt-in event-aware recorder fed by observational candidate frames.

    Callers may observe more frequently than the ordinary saved cadence. Quiet
    candidates stay only in a bounded rolling buffer. An explicit event or a
    large total-energy jump flushes the pre-event buffer and retains candidates
    through the post-event window. No simulation state is modified.
    """

    KIND_ORDINARY = 0
    KIND_PRE_EVENT = 1
    KIND_EVENT = 2
    KIND_POST_EVENT = 3

    def __init__(self, symbols, box_size, maximum_frames=40000,
                 ordinary_interval_fs=10.0, pre_event_fs=100.0,
                 post_event_fs=100.0, energy_jump_ev=20.0,
                 detect_chemical_events=True, close_contact_scale=0.35,
                 reaction_window_fs=20.0, chemical_context_fs=10.0):
        super().__init__(symbols, box_size, maximum_frames)
        self.format_version = self.FORMAT_VERSION
        self.ordinary_interval_fs = float(ordinary_interval_fs)
        self.pre_event_fs = float(pre_event_fs)
        self.post_event_fs = float(post_event_fs)
        self.energy_jump_ev = float(energy_jump_ev)
        self.detect_chemical_events = bool(detect_chemical_events)
        self.close_contact_scale = float(close_contact_scale)
        self.reaction_window_fs = float(reaction_window_fs)
        self.chemical_context_fs = float(chemical_context_fs)
        self._rolling = deque()
        self._next_ordinary_time = float("-inf")
        self._dense_until = float("-inf")
        self._last_total_energy = None
        self._last_bonds = None
        self._last_close_pairs = set()
        self._reaction_window_until = float("-inf")
        self._reaction_window_id = -1

    def _chemical_state(self, candidate, observation=None):
        positions = candidate["positions"]
        types = candidate["types"]
        ids = candidate["atom_ids"]
        if observation is None:
            first, second = np.triu_indices(len(positions), k=1)
            offsets = positions[second] - positions[first]
            box = candidate["box_size"]
            offsets -= box * np.round(offsets / box)
            distances = np.linalg.norm(offsets, axis=1)
            inner = R.CUTOFF_INNER[types[first], types[second]]
            outer = R.CUTOFF_OUTER[types[first], types[second]]
            taper = R.smooth_cutoff(distances, inner, outer)
        else:
            first = np.asarray(observation["first"], dtype=int)
            second = np.asarray(observation["second"], dtype=int)
            distances = np.asarray(observation["distance"], dtype=float)
            inner = np.asarray(observation["inner"], dtype=float)
            taper = np.asarray(observation["taper"], dtype=float)
        pairs = [
            tuple(sorted((int(ids[left]), int(ids[right]))))
            for left, right in zip(first, second)
        ]
        # Hysteresis prevents a vibrating pair near one threshold from being
        # reported as formed/broken every candidate frame.
        if self._last_bonds is None:
            bonded = taper > 0.35
        else:
            bonded = np.array([
                value > 0.65 or (pair in self._last_bonds and value >= 0.15)
                for pair, value in zip(pairs, taper)
            ], dtype=bool)
        bonds = {
            pair for pair, keep in zip(pairs, bonded) if keep
        }
        close = distances < inner * self.close_contact_scale
        close_pairs = [
            [int(ids[left]), int(ids[right]), float(distance)]
            for left, right, distance in zip(
                first[close], second[close], distances[close]
            )
        ]
        return bonds, close_pairs

    def _candidate(self, positions, time, potential, kinetic, temperature,
                   velocities=None, box_size=None, symbols=None, types=None,
                   atom_ids=None):
        positions = np.asarray(positions, dtype=np.float32).copy()
        if types is None:
            types = self.types if symbols is None else R.types_from_symbols(symbols)
        if atom_ids is None:
            atom_ids = self.atom_ids
        return {
            "positions": positions,
            "time": float(time),
            "potential": float(potential),
            "kinetic": float(kinetic),
            "temperature": float(temperature),
            "velocities": (
                None if velocities is None else
                np.asarray(velocities, dtype=np.float32).copy()
            ),
            "box_size": float(self.box_size if box_size is None else box_size),
            "types": np.asarray(types, dtype=np.uint8).copy(),
            "atom_ids": np.asarray(atom_ids, dtype=np.uint32).copy(),
        }

    def _append_candidate(self, candidate, kind, reason=""):
        # Duplicate timestamps can arise when an ordinary capture is also the
        # event frame. Upgrade its metadata instead of storing it twice.
        matches = np.where(
            np.isclose(np.asarray(self.times), candidate["time"], atol=1e-9)
        )[0]
        if len(matches):
            index = int(matches[0])
            if kind > self.frame_kinds[index]:
                self.frame_kinds[index] = int(kind)
                self.event_reasons[index] = str(reason)
            return False
        saved_stride = self.stride
        saved_since = self.since_last
        self.stride = 1
        self.since_last = 0
        frame_limit = self.maximum_frames
        # Metadata is assigned immediately after capture. Prevent the base
        # class from thinning this new frame while it still looks ordinary.
        self.maximum_frames = max(frame_limit, len(self.positions) + 2)
        stored = super().capture(
            candidate["positions"], candidate["time"],
            candidate["potential"], candidate["kinetic"],
            candidate["temperature"], velocities=candidate["velocities"],
            box_size=candidate["box_size"], types=candidate["types"],
            atom_ids=candidate["atom_ids"],
        )
        self.maximum_frames = frame_limit
        self.stride = saved_stride
        self.since_last = saved_since
        if stored:
            self.frame_kinds[-1] = int(kind)
            self.event_reasons[-1] = str(reason)
            if len(self.positions) > self.maximum_frames:
                self.thin()
        return stored

    def _sort_frames(self):
        order = np.argsort(np.asarray(self.times), kind="stable")
        names = (
            "positions", "velocities", "frame_types", "frame_atom_ids",
            "box_sizes", "times", "potential", "kinetic", "temperature",
            "frame_kinds", "event_reasons",
        )
        for name in names:
            values = getattr(self, name)
            if len(values) == len(order):
                setattr(self, name, [values[int(index)] for index in order])

    def observe(self, positions, time, potential, kinetic, temperature,
                velocities=None, box_size=None, symbols=None, types=None,
                atom_ids=None, event_reason=None, chemical_observation=None):
        candidate = self._candidate(
            positions, time, potential, kinetic, temperature, velocities,
            box_size, symbols, types, atom_ids,
        )
        total = candidate["potential"] + candidate["kinetic"]
        energy_delta = None
        if event_reason is None and self._last_total_energy is not None:
            energy_delta = total - self._last_total_energy
            if abs(energy_delta) >= self.energy_jump_ev:
                event_reason = f"total_energy_jump:{energy_delta:+.6g}eV"
        self._last_total_energy = total

        formed = set()
        broken = set()
        close_pairs = []
        if self.detect_chemical_events:
            bonds, close_pairs = self._chemical_state(
                candidate, chemical_observation
            )
            close_ids = {
                tuple(sorted((int(item[0]), int(item[1]))))
                for item in close_pairs
            }
            newly_close = close_ids - self._last_close_pairs
            if self._last_bonds is not None:
                formed = bonds - self._last_bonds
                broken = self._last_bonds - bonds
                if (formed or broken) and event_reason is None:
                    event_reason = "bond_change"
            self._last_bonds = bonds
            if newly_close and event_reason is None:
                event_reason = "close_contact"
            if newly_close:
                close_pairs = [
                    item for item in close_pairs
                    if tuple(sorted((int(item[0]), int(item[1])))) in newly_close
                ]
            else:
                close_pairs = []
            self._last_close_pairs = close_ids

        cutoff = candidate["time"] - self.pre_event_fs
        while self._rolling and self._rolling[0]["time"] < cutoff:
            self._rolling.popleft()
        self._rolling.append(candidate)

        if event_reason:
            event_type = str(event_reason).split(":", 1)[0]
            chemical = event_type in {"bond_change", "close_contact"}
            same_window = (
                chemical and candidate["time"] <= self._reaction_window_until
            )
            if chemical and not same_window:
                self._reaction_window_id += 1
            if chemical:
                self._reaction_window_until = (
                    candidate["time"] + self.reaction_window_fs
                )
            protect_frame = not same_window
            event = {
                "time_fs": candidate["time"],
                "type": event_type,
                "reason": str(event_reason),
                "formed": [list(pair) for pair in sorted(formed)],
                "broken": [list(pair) for pair in sorted(broken)],
                "close_contacts": close_pairs,
                "window_id": self._reaction_window_id if chemical else None,
                "window_start": bool(chemical and not same_window),
            }
            if energy_delta is not None:
                event["energy_delta_eV"] = float(energy_delta)
            self.events.append(event)
            if protect_frame:
                pre_context = (
                    self.chemical_context_fs if chemical
                    else self.pre_event_fs
                )
                buffered_frames = [
                    buffered for buffered in self._rolling
                    if buffered["time"] >= candidate["time"] - pre_context
                ]
                for buffered in buffered_frames:
                    kind = (
                        self.KIND_EVENT if buffered is candidate
                        else self.KIND_PRE_EVENT
                    )
                    self._append_candidate(buffered, kind, event_reason)
                self._sort_frames()
            else:
                self._append_candidate(
                    candidate, self.KIND_POST_EVENT, event_reason
                )
            self._rolling.clear()
            # Only a new window opens/extends its surrounding dense envelope.
            # Exact changes inside a sustained burst are still stored above,
            # but must not keep a nominal 50 fs post-window alive forever.
            if protect_frame:
                post_context = (
                    self.chemical_context_fs if chemical
                    else self.post_event_fs
                )
                self._dense_until = max(
                    self._dense_until, candidate["time"] + post_context
                )
            self._next_ordinary_time = max(
                self._next_ordinary_time,
                candidate["time"] + self.ordinary_interval_fs,
            )
            return True

        if candidate["time"] <= self._dense_until:
            self._append_candidate(candidate, self.KIND_POST_EVENT)
            self._rolling.clear()
            return True

        if candidate["time"] + 1e-9 >= self._next_ordinary_time:
            self._append_candidate(candidate, self.KIND_ORDINARY)
            self._next_ordinary_time = (
                candidate["time"] + self.ordinary_interval_fs
            )
            return True
        return False

    def thin(self):
        excess = len(self.positions) - self.maximum_frames
        if excess <= 0:
            return
        kinds = np.asarray(self.frame_kinds, dtype=np.uint8)
        protected = set(np.where(kinds == self.KIND_EVENT)[0].tolist())
        if self.positions:
            protected.update((0, len(self.positions) - 1))
        removable = []
        # Quiet evidence is least valuable; post-event and then pre-event
        # context are reduced only if quiet frames cannot satisfy the limit.
        for kind in (self.KIND_ORDINARY, self.KIND_POST_EVENT,
                     self.KIND_PRE_EVENT):
            removable.extend(
                index for index in np.where(kinds == kind)[0]
                if int(index) not in protected
            )
        if len(removable) < excess:
            raise RuntimeError(
                "AdaptiveRecorder contains more protected event frames than "
                "maximum_frames; increase --max-frames."
            )
        # Spread removals through each priority stream instead of deleting one
        # contiguous time range. This preserves whole-run temporal coverage.
        chosen = set()
        remaining = excess
        for kind in (self.KIND_ORDINARY, self.KIND_POST_EVENT,
                     self.KIND_PRE_EVENT):
            candidates = [
                int(index) for index in np.where(kinds == kind)[0]
                if int(index) not in protected
            ]
            take = min(remaining, len(candidates))
            if take:
                positions = np.linspace(0, len(candidates) - 1, take, dtype=int)
                chosen.update(candidates[int(position)] for position in positions)
                remaining -= take
            if not remaining:
                break
        keep = [index for index in range(len(self.positions)) if index not in chosen]
        for name in (
            "positions", "velocities", "frame_types", "frame_atom_ids",
            "box_sizes", "times", "potential", "kinetic", "temperature",
            "frame_kinds", "event_reasons",
        ):
            values = getattr(self, name)
            if len(values) == len(kinds):
                setattr(self, name, [values[index] for index in keep])
        self.adaptive_dropped_frames += len(chosen)
        self.thinned_count += 1

    def clear(self):
        super().clear()
        self.event_reasons.clear()
        self._rolling.clear()
        self._next_ordinary_time = float("-inf")
        self._dense_until = float("-inf")
        self._last_total_energy = None
        self._last_bonds = None
        self._last_close_pairs = set()
        self._reaction_window_until = float("-inf")
        self._reaction_window_id = -1
        self.adaptive_dropped_frames = 0

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)

        recorder = cls(
            symbols=[str(item) for item in data["symbols"]],
            box_size=float(data["box_size"]),
            maximum_frames=len(data["positions"]) + 1
        )
        recorder.format_version = (
            int(data["format_version"]) if "format_version" in data.files else 1
        )

        recorder.positions = [
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
            recorder.box_sizes = [
                float(value) for value in data["box_sizes"]
            ]

            # The nominal size becomes the one the run ended at,
            # so anything that only looks at box_size still gets
            # something sensible.

            if recorder.box_sizes:
                recorder.box_size = recorder.box_sizes[-1]

        if "velocities" in data.files:
            recorder.velocities = [
                frame for frame in data["velocities"]
            ]

        recorder.times = list(data["times"])
        recorder.potential = list(data["potential"])
        recorder.kinetic = list(data["kinetic"])
        recorder.temperature = list(data["temperature"])
        recorder.frame_kinds = (
            list(np.asarray(data["frame_kinds"], dtype=np.uint8))
            if "frame_kinds" in data.files else [0] * len(recorder.positions)
        )
        recorder.event_reasons = (
            [str(value) for value in data["event_reasons"]]
            if "event_reasons" in data.files else [""] * len(recorder)
        )
        recorder.adaptive_dropped_frames = (
            int(data["adaptive_dropped_frames"])
            if "adaptive_dropped_frames" in data.files else 0
        )
        recorder.events = (
            [json.loads(str(value)) for value in data["events_json"]]
            if "events_json" in data.files else []
        )

        return recorder

    def export_xyz(self, path, every=1):
        # Multi-frame XYZ. Any molecular viewer will open this,
        # so a recording can be replayed properly in VMD or
        # Ovito rather than only here.

        with open(path, "w") as handle:
            for index in range(0, len(self.positions), every):
                frame = self.positions[index]

                handle.write(f"{len(frame)}\n")
                handle.write(
                    f"time {self.times[index]:.3f} fs  "
                    f"T {self.temperature[index]:.1f} K\n"
                )

                for symbol, point in zip(
                    self.symbols_at(index), frame
                ):
                    handle.write(
                        f"{symbol} {point[0]:.5f} "
                        f"{point[1]:.5f} {point[2]:.5f}\n"
                    )

        return path


# AdaptiveRecorder is intentionally declared beside the serialization methods
# above. Restore the legacy public class methods explicitly so existing imports
# keep their historical Recorder.load()/export_xyz() API during the opt-in
# development period.
Recorder.load = classmethod(AdaptiveRecorder.load.__func__)
Recorder.export_xyz = AdaptiveRecorder.export_xyz
