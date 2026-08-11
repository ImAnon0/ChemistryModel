import numpy as np, torch
from hf_surface_scan import build, formaldehyde_geometry, CENTRE

sim = build("high_fidelity")
symbols, positions = formaldehyde_geometry(1.09, 1.60)
sim.positions = torch.tensor(positions + CENTRE, device=sim.device, dtype=sim.dtype)
sim.build_neighbours()

with torch.no_grad():
    sim.energy_per_atom(sim.positions)

print("atoms:", symbols)
print("C is index 0, O is index 1")