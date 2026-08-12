"""Why do two methyl radicals stop at 2.4 A instead of bonding?

Eight seeds all approached to between 2.34 and 2.44 A and bounced. Ethane's
C-C is 1.54, and 2.464 is the outer cutoff for C-C -- so they stop exactly
where the bond term switches on, which is the wrong place for a wall. Two
radicals with a spare valence each should be pulled in from there.

This walks the two carbons together along their axis and prints each energy
term, so whichever is doing the pushing is visible rather than guessed at.
"""
import numpy as np
import torch

import reactive as R
from batched_torch import BatchedReactiveSimulation

BOX = 20.0
CENTRE = np.array([BOX / 2, BOX / 2, BOX / 2])


def methyl_pair(separation, angle=109.47):
    """Two planar-ish CH3 groups facing each other along z."""
    radians = np.radians(180.0 - angle)

    def group(origin, direction):
        atoms = [origin]
        for index in range(3):
            turn = 2.0 * np.pi * index / 3.0
            atoms.append(origin + 1.09 * np.array([
                np.sin(radians) * np.cos(turn),
                np.sin(radians) * np.sin(turn),
                direction * np.cos(radians),
            ]))
        return atoms

    first = group(np.array([0.0, 0.0, 0.0]), -1.0)
    second = group(np.array([0.0, 0.0, separation]), 1.0)

    symbols = ["C", "H", "H", "H", "C", "H", "H", "H"]
    return symbols, np.array(first + second)


symbols, start = methyl_pair(3.0)
sim = BatchedReactiveSimulation(
    boxes=[(symbols, start + CENTRE)], box_size=BOX,
    random_seed=0, relax_on_start=False, device="cpu", dtype=torch.float64,
)

print(f"C-C: equilibrium {R.BOND_LENGTH[1, 1]:.3f}, "
      f"depth {R.BOND_DEPTH[1, 1]:.3f} eV, "
      f"cutoff {R.CUTOFF_INNER[1, 1]:.3f} to {R.CUTOFF_OUTER[1, 1]:.3f}\n")

print(f"{'r(C-C)':>8}{'bond':>10}{'over':>10}{'angle':>10}{'TOTAL':>10}"
      f"{'vs 3.0 A':>10}")

reference = None
for separation in (3.00, 2.60, 2.46, 2.40, 2.20, 2.00, 1.80, 1.60, 1.54,
                   1.40):
    _, positions = methyl_pair(separation)

    sim.positions = torch.tensor(
        positions + CENTRE, device=sim.device, dtype=sim.dtype
    )
    sim.build_neighbours()

    with torch.no_grad():
        total = float(torch.sum(sim.energy_per_atom(sim.positions)))

    parts = {
        name: float(torch.sum(value))
        for name, value in sim._energy_parts.items()
    }

    if reference is None:
        reference = total

    print(f"{separation:8.2f}{parts['bond']:10.3f}{parts['over']:10.3f}"
          f"{parts['angle']:10.3f}{total:10.3f}{total - reference:+10.3f}")

print("\nif the total rises as they close, whichever column rises with it is")
print("the term doing the pushing")
