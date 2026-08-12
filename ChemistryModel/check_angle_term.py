"""Does the angle term actually act at water's relaxed geometry?

Water relaxes its donor angle to 75.53 degrees, nearly 30 away from its
104.5 rest value, and does so identically at two different oxygen angle
stiffnesses. If the angle term were doing what it looks like it does, a
stiffer oxygen would make that distortion cost more and the energy there
would rise. It did not move at all.

Either the term is switched off at that geometry, or the angle being
computed is not the one the spectator coordinate names. Both are worth
knowing and neither is visible from a barrier.

No relaxation anywhere here: the same fixed geometries are evaluated under
each stiffness, so any difference is the term itself rather than the
optimiser finding somewhere else to sit.
"""
import numpy as np
import torch

import reactive as R
from hf_surface_scan import (
    CENTRE, active_geometry, apply_system, build, energy_at,
)

apply_system("water")

# Geometries to probe: the frozen start, the relaxed donor angle, and two
# extremes. Held fixed, only the stiffness changes between runs.
ANGLES = (104.47, 95.0, 85.0, 75.53, 65.0)

# Reactant and saddle, from the scan.
POINTS = (("reactant", 0.980, 1.700), ("saddle", 1.100, 1.300))

# ANGLE_STIFFNESS_ARRAY is built once at import from the ANGLE_STIFFNESS
# dictionary, and the simulation reads the array. Setting the dictionary
# afterwards therefore does nothing at all -- which is what an earlier
# version of this script did, giving zero difference everywhere and looking
# exactly like an angle term that had been switched off.
oxygen = int(R.ELEMENT_INDEX["O"])
original = float(R.ANGLE_STIFFNESS_ARRAY[oxygen])

# REST_ANGLE is not used by the torch path at all. The rest angle is derived
# instead: steric = coordination + lone pairs, interpolated 180 to 120 to
# 109.47, then reduced by LONE_PAIR_SQUEEZE per lone pair. For an oxygen
# with two bonds that gives 109.47 - 2.5 * 2 = 104.47, which happens to land
# within a twentieth of a degree of the 104.5 in the dictionary. Worth
# knowing they agree by construction rather than because one is read.
lone_pairs = (R.OUTER_ELECTRONS["O"] - 2.0) / 2.0
expected_rest = 109.47 - R.LONE_PAIR_SQUEEZE * lone_pairs

print(f"oxygen stiffness in the array {original}")
print(f"REST_ANGLE['O'] is {R.REST_ANGLE['O']}, but the torch path does not")
print(f"use it; with two bonds and {lone_pairs:.0f} lone pairs it wants "
      f"{expected_rest:.2f} degrees\n")

for label, donor, transfer in POINTS:
    print(f"--- {label}: donor {donor:.3f}, transfer {transfer:.3f} ---")
    print(f"{'donor angle':>12}{'stiff 2.6':>12}{'stiff 3.5':>12}"
          f"{'difference':>12}")

    rows = []
    for stiffness in (2.6, 3.5):
        R.ANGLE_STIFFNESS_ARRAY[oxygen] = stiffness
        sim = build("high_fidelity", mixing=0.52)

        column = []
        for angle in ANGLES:
            spectators = np.array([0.96, 0.96, angle, 104.5], float)
            column.append(energy_at(sim, donor, transfer, spectators))
        rows.append(column)

    for index, angle in enumerate(ANGLES):
        low, high = rows[0][index], rows[1][index]
        print(f"{angle:12.2f}{low:12.4f}{high:12.4f}{high - low:+12.4f}")
    print()

R.ANGLE_STIFFNESS_ARRAY[oxygen] = original

print("a zero difference means the angle term contributes nothing at that")
print("geometry, whatever the spectator coordinate is called")


# How many neighbours does each oxygen actually have? An angle needs two.
print("\n\nneighbour counts, since an angle needs two of them\n")
print(f"{'point':>10}{'r(H...O acceptor)':>20}{'donor O':>10}"
      f"{'acceptor O':>12}")

outer = R.CUTOFF_OUTER[R.ELEMENT_INDEX["O"], R.ELEMENT_INDEX["H"]]
for label, donor, transfer in POINTS:
    _, positions = active_geometry(donor, transfer)
    donor_o, moving, donor_h, acceptor_o, acceptor_h = positions

    def within(centre, others):
        return sum(
            1 for other in others
            if np.linalg.norm(centre - other) < outer
        )

    print(f"{label:>10}{transfer:20.3f}"
          f"{within(donor_o, [moving, donor_h]):10d}"
          f"{within(acceptor_o, [moving, acceptor_h]):12d}")

print(f"\nO-H outer cutoff is {outer:.3f} A")