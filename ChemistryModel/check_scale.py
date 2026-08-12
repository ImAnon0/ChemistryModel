import numpy as np, torch
from hf_surface_scan import build, apply_system, active_geometry, CENTRE
import reactive as R

R.OVER_COORDINATION_DEPTH_WEIGHT = 1.5

for name, donor, transfer in [
    ("water", 1.38, 1.00),
    ("formaldehyde", 1.14, 1.06),
    ("methane", 1.12, 1.06),
]:
    apply_system(name)
    sim = build("high_fidelity")
    symbols, positions = active_geometry(donor, transfer)

    sim.positions = torch.tensor(positions + CENTRE,
                                 device=sim.device, dtype=sim.dtype)
    sim.build_neighbours()

    with torch.no_grad():
        energies = sim.energy_per_atom(sim.positions)

    print(f"\n{name}  (saddle geometry)")
    for symbol, energy in zip(symbols, energies.tolist()):
        print(f"   {symbol}  per-atom energy {energy:+8.3f} eV")