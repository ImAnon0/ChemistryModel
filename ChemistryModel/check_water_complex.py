import numpy as np
import torch

from hf_surface_scan import (CENTRE, active_geometry, apply_system, build,
                             relaxed_energy)

apply_system("water")
sim = build("high_fidelity", mixing=0.52)


def energy(donor, transfer):
    value, _ = relaxed_energy(sim, donor, transfer)
    return value


# The scan's reactant cell, from the last run.
scan_reactant = energy(0.980, 1.700)

# Genuinely separated: push the transferring hydrogen far from the acceptor
# oxygen, well past the 1.536 A O-H cutoff, so the two molecules do not
# interact at all.
print(f"{'r(H...O acceptor)':>20}{'energy':>12}{'vs scan reactant':>20}")
for far in (1.70, 2.20, 2.80, 3.50, 4.50):
    value = energy(0.980, far)
    print(f"{far:20.2f}{value:12.4f}{value - scan_reactant:+20.4f}")

print("\nif the energy keeps falling past the cutoff, the scan's reactant is")
print("a hydrogen-bonded complex rather than separated molecules, and the")
print("barrier measured from it is larger than one measured from separation")