import numpy as np, torch
from hf_surface_scan import build, apply_system, active_geometry, CENTRE
import reactive as R

for name, donor, transfer in [
    ("formaldehyde", 1.14, 1.06),
    ("water", 1.38, 1.00),
    ("methane", 1.12, 1.06),
]:
    apply_system(name)
    sim = build("high_fidelity")
    symbols, positions = active_geometry(donor, transfer)

    pos = torch.tensor(positions + CENTRE, device=sim.device, dtype=sim.dtype)
    sim.positions = pos
    sim.build_neighbours()

    off = pos[sim.neighbours] - pos[:, None, :]
    dist = torch.sqrt(torch.clamp((off ** 2).sum(2), min=1e-12))
    mask = sim.neighbour_mask.to(sim.dtype)
    kinds = sim.types
    inner = sim.cutoff_inner[kinds[:, None], kinds[sim.neighbours]]
    outer = sim.cutoff_outer[kinds[:, None], kinds[sim.neighbours]]
    frac = torch.clamp((dist - inner) / torch.clamp(outer - inner, min=1e-9), 0, 1)
    taper = 0.5 * (1 + torch.cos(np.pi * frac)) * mask

    coord = taper.sum(1)
    valence = sim.valence[kinds]
    excess = torch.clamp(coord - valence, min=0.0)

    print(f"\n{name} at its saddle")
    for symbol, c, v, e in zip(symbols, coord.tolist(),
                               valence.tolist(), excess.tolist()):
        print(f"   {symbol}  coordination {c:5.3f}  valence {v:3.1f}  "
              f"excess {e:5.3f}")