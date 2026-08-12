"""Which energy term accounts for the missing reaction energy?

The bond depths say formaldehyde's abstraction should release 0.750 eV:
the aldehydic C-H is softened to 3.758 and H-H is 4.508. The model reports
0.432. Something on the product side is 0.318 eV higher than the depths
alone predict, and this prints each term separately at both ends so the
culprit is measured rather than guessed.
"""
import torch
from hf_surface_scan import build, apply_system, active_geometry, CENTRE

apply_system("formaldehyde")
sim = build("high_fidelity")


def parts_at(donor, transfer):
    _, positions = active_geometry(donor, transfer)
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


reactant = parts_at(1.09, 1.60)
product = parts_at(1.80, 0.74)

print(f"{'term':>8}{'reactant':>12}{'product':>12}{'change':>12}")
for name in ("bond", "over", "angle", "TOTAL"):
    change = product[name] - reactant[name]
    print(f"{name:>8}{reactant[name]:12.3f}{product[name]:12.3f}"
          f"{change:+12.3f}")

print("\nbond depths alone predict -0.750 eV")
print("whichever term contributes a large positive change is the one")
print("cancelling it")
