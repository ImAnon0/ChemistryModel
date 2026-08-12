"""Does the carbon's bond order change when it loses a hydrogen?

If the C=O order rises in the product, the carbon's commitment rises with it,
the environment softening deepens, and the remaining C-H is weakened further.
That would stabilise the product and cancel part of the reaction energy --
which is the 0.32 eV the softening is currently failing to deliver.
"""
import torch
from hf_surface_scan import build, apply_system, active_geometry, CENTRE

apply_system("formaldehyde")
sim = build("high_fidelity")

for label, donor, transfer in [
    ("reactant  H2CO + H", 1.09, 1.60),
    ("saddle",             1.16, 1.065),
    ("product   HCO + H2", 1.80, 0.74),
]:
    print(f"\n--- {label} ---")
    _, positions = active_geometry(donor, transfer)
    sim.positions = torch.tensor(
        positions + CENTRE, device=sim.device, dtype=sim.dtype
    )
    sim.build_neighbours()
    with torch.no_grad():
        sim.energy_per_atom(sim.positions)