import numpy as np

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

    def __init__(self, symbols, box_size, maximum_frames=40000):
        self.symbols = list(symbols)
        self.box_size = float(box_size)
        self.maximum_frames = int(maximum_frames)

        self.types = R.types_from_symbols(self.symbols)

        # How many captures are skipped between stored frames.
        # Starts at one and doubles whenever the buffer fills.

        self.stride = 1
        self.since_last = 0
        self.thinned_count = 0

        self.positions = []
        self.velocities = []

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

    def __len__(self):
        return len(self.positions)

    @property
    def is_empty(self):
        return len(self.positions) == 0

    def capture(self, positions, time, potential, kinetic,
                temperature, velocities=None, box_size=None):
        # Frames arrive at a fixed rate but are only kept every
        # `stride` of them.

        self.since_last += 1

        if self.since_last < self.stride:
            return False

        self.since_last = 0

        self.positions.append(
            np.asarray(positions, dtype=np.float32).copy()
        )

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
        self.box_sizes = self.box_sizes[::2]
        self.times = self.times[::2]
        self.potential = self.potential[::2]
        self.kinetic = self.kinetic[::2]
        self.temperature = self.temperature[::2]

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
        self.box_sizes.clear()
        self.times.clear()
        self.potential.clear()
        self.kinetic.clear()
        self.temperature.clear()

    # --------------------------------------------------------

    def bonds_at(self, index, threshold=0.35):
        # Bonds are not stored. They are recomputed from the
        # coordinates, which keeps the file small and means a
        # change to the bonding rules applies to old recordings
        # too.

        positions = self.positions[index]

        box = self.box_at(index)

        count = len(positions)

        first = []
        second = []

        for i in range(count):
            offsets = positions[i + 1:] - positions[i]

            offsets -= box * np.round(offsets / box)

            distances = np.linalg.norm(offsets, axis=1)

            partners = self.types[i + 1:]

            inner = R.CUTOFF_INNER[self.types[i], partners]
            outer = R.CUTOFF_OUTER[self.types[i], partners]

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
                symbol = R.ELEMENTS[self.types[member]]
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

        np.savez_compressed(
            path,
            symbols=np.array(self.symbols),
            box_size=self.box_size,
            positions=np.array(self.positions, dtype=np.float32),
            **extra,
            times=np.array(self.times),
            potential=np.array(self.potential),
            kinetic=np.array(self.kinetic),
            temperature=np.array(self.temperature),
        )

        return path

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)

        recorder = cls(
            symbols=[str(item) for item in data["symbols"]],
            box_size=float(data["box_size"]),
            maximum_frames=len(data["positions"]) + 1
        )

        recorder.positions = [
            frame for frame in data["positions"]
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

                for symbol, point in zip(self.symbols, frame):
                    handle.write(
                        f"{symbol} {point[0]:.5f} "
                        f"{point[1]:.5f} {point[2]:.5f}\n"
                    )

        return path