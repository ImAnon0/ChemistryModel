import numpy as np
from hf_surface_scan import build, surface, basin_seeds, flood_saddle
import high_fidelity_torch as h

print("cap:", h.H_TRANSFER_LOWERING_CAP, " mixing:", h.H_TRANSFER_STATE_MIXING_FRACTION)

d = np.arange(1.00, 1.90, 0.02)
t = np.arange(0.65, 1.60, 0.005)

sim = build("high_fidelity")
g = surface(sim, d, t)
r, p = basin_seeds(g, d, t)
cell, e = flood_saddle(g, r, p)

print("reactant r(C-H) %.3f  r(H-H) %.3f" % (d[r[0]], t[r[1]]))
print("saddle   r(C-H) %.3f  r(H-H) %.3f" % (d[cell[0]], t[cell[1]]))
print("height   %+.3f eV" % (e - g[r]))
print("stretch  %.3f A" % (d[cell[0]] - d[r[0]]))