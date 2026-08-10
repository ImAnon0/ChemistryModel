import numpy as np

import reactive as R


# ============================================================
# Bonds by persistence, not by distance
# ============================================================
#
# A distance test asks whether two atoms are close. That works
# while close and bonded mean the same thing, and stops working
# the moment they do not.
#
# Squeeze a box to twice the density of liquid water and every
# atom sits inside its neighbours' bond range whether or not
# anything chemical has happened. Connected components then run
# away into structures of forty carbons that dissolve as soon as
# the box is opened again. Patching that with a density cutoff
# only hides it: frames get thrown away, and every bond in the
# run appears to form at the first frame that survives the cut.
#
# What actually separates a bond from a collision is time. A
# carbon-carbon bond vibrates about once every twenty
# femtoseconds and survives many thousands of those cycles. Two
# atoms merely passing each other are within range for a few
# femtoseconds and then gone.
#
# So a pair counts as bonded once it has stayed within range for
# longer than a handful of vibrations, and stays bonded until it
# has been out of range for the same span. No density threshold,
# no discarded frames, and a formation time that means something.
#
# It also makes a real question answerable. Compressing a system
# until atoms are forced to bonding distance is how pressure
# drives polymerisation, and the products are genuinely bonded
# while the pressure holds. A persistence test can tell that
# apart from crowding: a pressure-bonded network survives the
# test and then breaks on expansion, which is a result rather
# than an artefact.


# How long a contact must hold before it counts, and how long it
# must be gone before it stops counting. Around five vibrational
# periods, which is short enough to catch genuine chemistry and
# long enough to reject a collision.

FORMATION_TIME = 100.0
BREAKING_TIME = 100.0


class BondTracker:

    def __init__(self, types, formation_time=FORMATION_TIME,
                 breaking_time=BREAKING_TIME, threshold=0.35):
        self.types = types
        self.formation_time = formation_time
        self.breaking_time = breaking_time
        self.threshold = threshold

        count = len(types)

        self.first, self.second = np.triu_indices(count, k=1)

        self.inner = R.CUTOFF_INNER[
            types[self.first], types[self.second]
        ]
        self.outer = R.CUTOFF_OUTER[
            types[self.first], types[self.second]
        ]

        # How long each pair has been continuously in range, and
        # continuously out of it. Both are in femtoseconds.

        self.in_range_for = np.zeros(len(self.first))
        self.out_of_range_for = np.full(len(self.first), np.inf)

        self.bonded = np.zeros(len(self.first), dtype=bool)

        self.last_time = None
        self.warned = False

        # When each currently bonded pair first came into range,
        # which is the honest formation time rather than the
        # moment it crossed the persistence threshold.

        self.formed_at = np.zeros(len(self.first))

        self.events = []

    def in_range(self, positions, box_size):
        offsets = positions[self.second] - positions[self.first]
        offsets -= box_size * np.round(offsets / box_size)

        distances = np.linalg.norm(offsets, axis=1)

        taper = R.smooth_cutoff(distances, self.inner, self.outer)

        return taper > self.threshold

    def update(self, positions, box_size, time):
        # Advance the tracker by one frame and return the pairs
        # currently counted as bonded.

        if self.last_time is None:
            elapsed = 0.0
        else:
            elapsed = float(time) - self.last_time

            # Frames further apart than the formation time make
            # persistence meaningless: every contact clears the
            # threshold the moment it is first seen, and the
            # tracker quietly degenerates into the distance test
            # it was written to replace. Measured on one run,
            # sampling every 160 fs invented over-coordinated
            # atoms that vanish entirely at 40 fs.

            if elapsed >= self.formation_time and not self.warned:
                self.warned = True

                print(
                    f"  warning: frames are {elapsed:.0f} fs apart "
                    f"but a bond is confirmed after "
                    f"{self.formation_time:.0f} fs, so persistence "
                    f"cannot do its job. Use a smaller stride."
                )

        self.last_time = float(time)

        close = self.in_range(positions, box_size)

        # A pair that has just come into range starts its clock
        # now; one that has just left starts the other clock.

        arriving = close & (self.in_range_for == 0.0)

        self.in_range_for = np.where(
            close, self.in_range_for + elapsed, 0.0
        )

        self.out_of_range_for = np.where(
            close, 0.0, self.out_of_range_for + elapsed
        )

        # The moment a pair came into range, remembered so a bond
        # can be dated from when it started rather than from when
        # it was confirmed.

        self.arrived_at = getattr(
            self, "arrived_at", np.zeros(len(self.first))
        )

        self.arrived_at = np.where(
            arriving, float(time), self.arrived_at
        )

        forming = (
            (~self.bonded)
            & close
            & (self.in_range_for >= self.formation_time)
        )

        breaking = (
            self.bonded
            & (~close)
            & (self.out_of_range_for >= self.breaking_time)
        )

        for index in np.where(forming)[0]:
            self.events.append((
                float(self.arrived_at[index]),
                "formed",
                int(self.first[index]),
                int(self.second[index]),
            ))

            self.formed_at[index] = self.arrived_at[index]

        for index in np.where(breaking)[0]:
            self.events.append((
                float(time) - self.breaking_time,
                "broke",
                int(self.first[index]),
                int(self.second[index]),
            ))

        self.bonded = (self.bonded | forming) & (~breaking)

        return self.first[self.bonded], self.second[self.bonded]

    def confirmed_now(self):
        return self.first[self.bonded], self.second[self.bonded]

    @property
    def pending(self):
        # Pairs in range but not yet held long enough to count.
        # A large number here at high density is the tracker
        # doing its job: contacts that never mature into bonds.

        close = self.in_range_for > 0.0

        return int(np.count_nonzero(close & (~self.bonded)))