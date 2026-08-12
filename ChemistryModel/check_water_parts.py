"""What is water's barrier actually made of?

Its 0.669 eV barrier barely moved when the O-H depth was cut by 0.36 eV, so
it is not the bond term. This prints each term separately at the reactant
and at the saddle, relaxed, so the one carrying it can be identified rather
than guessed at.

Formaldehyde is included as a control, since its barrier is essentially
exact and its decomposition is what a working case looks like.
"""
import numpy as np
import torch

from hf_surface_scan import (
    CENTRE, active_geometry, apply_system, build, relaxed_energy,
)


def parts_at(sim, donor, transfer, spectators):
    _, positions = active_geometry(donor, transfer, spectators)
    sim.positions = torch.tensor(
        positions + CENTRE, device=sim.device, dtype=sim.dtype
    )
    sim.build_neighbours()
    with torch.no_grad():
        total = float(torch.sum(sim.energy_per_atom(sim.positions)))
    pieces = {
        name: float(torch.sum(value))
        for name, value in sim._energy_parts.items()
    }
    pieces["TOTAL"] = total
    return pieces


# system, reactant (donor, transfer), saddle (donor, transfer)
CASES = [
    ("water", (0.980, 1.620), (1.020, 1.380)),
    ("formaldehyde", (1.080, 1.310), (1.160, 1.050)),
]

for name, reactant_point, saddle_point in CASES:
    apply_system(name)
    sim = build("high_fidelity", mixing=0.52)

    print(f"\n=== {name} ===")
    both = []
    for label, (donor, transfer) in [("reactant", reactant_point),
                                     ("saddle", saddle_point)]:
        # Relax the spectators at that point, as the scan does, then read the
        # terms there. Reading them at the frozen geometry would describe a
        # different surface from the one the barrier came from.
        _, spectators = relaxed_energy(sim, donor, transfer)
        both.append(parts_at(sim, donor, transfer, spectators))
        print(f"  {label:>8} spectators: "
              + " ".join(f"{v:.3f}" for v in spectators))

    reactant, saddle = both
    print(f"\n  {'term':>8}{'reactant':>12}{'saddle':>12}{'barrier':>12}")
    for term in ("bond", "over", "angle", "TOTAL"):
        print(f"  {term:>8}{reactant[term]:12.3f}{saddle[term]:12.3f}"
              f"{saddle[term] - reactant[term]:+12.3f}")
