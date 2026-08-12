"""Where does relaxation change the barrier -- at the reactant or the saddle?

A relaxed barrier being larger than a frozen one does not mean anything went
uphill. Each relaxed point is by construction no higher than the same frozen
point, so a barrier can only grow if relaxation stabilises the reactant more
than it stabilises the saddle. Which of the two moved, and by how much, is
the thing worth knowing: it says whether the transfer surface is wrong or
whether the rest of the molecule is being allowed to do something it
shouldn't.

Water is the case in point. Its frozen barrier is 0.089 eV above the
reference and its relaxed one 0.305 above, while formaldehyde and methane
barely move between the two.
"""
import numpy as np

from hf_surface_scan import (
    active_frozen, apply_system, build, energy_at, relaxed_energy,
)

# system, reactant (donor, transfer), saddle (donor, transfer)
CASES = [
    ("water", (0.980, 1.700), (1.100, 1.300)),
    ("formaldehyde", (1.080, 1.310), (1.160, 1.050)),
    ("methane", (1.100, 1.300), (1.120, 1.055)),
    ("ammonia", (1.020, 1.300), (1.120, 1.010)),
]

print(f"{'system':>14}{'point':>10}{'frozen':>10}{'relaxed':>10}"
      f"{'gain':>9}   relaxed spectators")

for name, reactant_point, saddle_point in CASES:
    apply_system(name)
    sim = build("high_fidelity", mixing=0.52)

    gains = {}
    for label, (donor, transfer) in [("reactant", reactant_point),
                                     ("saddle", saddle_point)]:
        frozen = energy_at(sim, donor, transfer)
        relaxed, spectators = relaxed_energy(sim, donor, transfer)
        gains[label] = frozen - relaxed

        shown = " ".join(f"{value:.2f}" for value in spectators)
        print(f"{name if label == 'reactant' else '':>14}{label:>10}"
              f"{frozen:10.3f}{relaxed:10.3f}{frozen - relaxed:+9.3f}"
              f"   {shown}")

    difference = gains["reactant"] - gains["saddle"]
    if difference > 0:
        note = "reactant gained more, so the barrier grew by this"
    elif difference < 0:
        note = "saddle gained more, so the barrier shrank by this"
    else:
        note = "both gained equally, so the barrier is unchanged"

    print(f"{'':>14}{'':>10}{'':>10}{'':>10}{difference:+9.3f}   <- {note}")
    print()

print("rest angles for reference: C 109.47, N 107.0, O 104.5")


# ----------------------------------------------------------------------
# Is the relaxed geometry a minimum, or just where the optimiser stopped?
# ----------------------------------------------------------------------
#
# Water's spectators come back at 75.53 degrees, identically at two
# different oxygen angle stiffnesses and at both the reactant and the
# saddle. An optimiser landing on the same value to two decimals under two
# different potentials is worth questioning: it could be a genuine minimum
# that the stiffness happens not to move, or it could be somewhere Powell
# stops for a reason that has nothing to do with the surface.
#
# The test is to start it somewhere else. A real minimum is found from any
# direction; a stopping point is not.

print("\n\nwater relaxation from three different starting angles")
print("(the default start is 104.5, water's rest angle)\n")

apply_system("water")
sim = build("high_fidelity", mixing=0.52)

print(f"{'start angle':>12}{'energy':>12}   relaxed spectators")

for angle in (104.5, 80.0, 130.0, 65.0):
    start = np.array([0.96, 0.96, angle, angle], float)
    value, spectators = relaxed_energy(sim, 0.980, 1.700, start=start)
    shown = " ".join(f"{item:.2f}" for item in spectators)
    print(f"{angle:12.1f}{value:12.4f}   {shown}")

print("\nif these disagree, the relaxed surface depends on where the")
print("optimiser began, and every relaxed number measured so far is soft")