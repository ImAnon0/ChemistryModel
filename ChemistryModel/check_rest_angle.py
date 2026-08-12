"""What rest angle does the model actually want for water's donor oxygen?

The angle energy there is exactly zero at 75.53 degrees, so that is where
the term's minimum sits. But the rest angle is supposed to come from domain
counting -- steric number, interpolated 180 to 120 to 109.47, less a squeeze
per lone pair -- which for a two-bonded oxygen gives 104.47.

Those disagree by 29 degrees, so one of the quantities feeding that
calculation is not what it looks like. This prints every one of them at the
geometry in question rather than inferring from the energy.
"""
import numpy as np
import torch

import reactive as R
from hf_surface_scan import CENTRE, active_geometry, apply_system, build

apply_system("water")
sim = build("high_fidelity", mixing=0.52)

OXYGEN = int(R.ELEMENT_INDEX["O"])

for label, donor, transfer, angle in (
    ("reactant", 0.980, 1.700, 75.53),
    ("reactant", 0.980, 1.700, 104.47),
    ("saddle", 1.100, 1.300, 75.53),
):
    spectators = np.array([0.96, 0.96, angle, 104.5], float)
    _, positions = active_geometry(donor, transfer, spectators)

    sim.positions = torch.tensor(
        positions + CENTRE, device=sim.device, dtype=sim.dtype
    )
    sim.build_neighbours()

    with torch.no_grad():
        pos = sim.positions
        offsets = pos[sim.neighbours] - pos[:, None, :]
        distances = torch.sqrt(
            torch.clamp((offsets ** 2).sum(2), min=1e-12)
        )
        mask = sim.neighbour_mask.to(sim.dtype)

        kinds = sim.types
        inner = sim.cutoff_inner[kinds[:, None], kinds[sim.neighbours]]
        outer = sim.cutoff_outer[kinds[:, None], kinds[sim.neighbours]]
        fraction = torch.clamp(
            (distances - inner) / torch.clamp(outer - inner, min=1e-9), 0, 1
        )
        taper = 0.5 * (1 + torch.cos(np.pi * fraction)) * mask

        coordination = taper.sum(1)

        # order is not recomputed here; at these geometries every bond is
        # single, so taper * order reduces to taper.
        bonded_order = coordination
        outer_electrons = sim.outer_electrons[kinds]
        lone = torch.clamp((outer_electrons - bonded_order) / 2.0, min=0.0)
        steric = torch.clamp(coordination + lone, 2.0, 4.0)

        low = torch.where(
            steric < 3.0,
            180.0 + (120.0 - 180.0) * (steric - 2.0),
            120.0 + (109.47 - 120.0) * (steric - 3.0),
        )
        rest = low - sim.lone_pair_squeeze * lone

    # Index 0 is the donor oxygen, index 3 the acceptor.
    print(f"\n--- {label}, donor angle set to {angle} ---")
    print(f"{'atom':>10}{'coord':>9}{'lone':>7}{'steric':>8}"
          f"{'low angle':>11}{'rest':>9}")
    for index, name in ((0, "donor O"), (3, "acceptor O")):
        print(f"{name:>10}{coordination[index]:9.3f}{lone[index]:7.3f}"
              f"{steric[index]:8.3f}{low[index]:11.2f}{rest[index]:9.2f}")

    # And the angle actually present between the donor's two neighbours.
    donor_o, moving, donor_h = positions[0], positions[1], positions[2]
    first, second = moving - donor_o, donor_h - donor_o
    measured = np.degrees(np.arccos(
        first @ second
        / (np.linalg.norm(first) * np.linalg.norm(second))
    ))
    print(f"{'':>10}measured H-O-H at the donor: {measured:.2f} degrees")
