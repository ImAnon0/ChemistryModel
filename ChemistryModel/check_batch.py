import numpy as np
from hf_surface_scan import (apply_system, build, active_geometry,
                             batched_surface, energy_at, CENTRE)

apply_system("formaldehyde")
donor = np.arange(1.00, 1.20, 0.04)
transfer = np.arange(0.90, 1.10, 0.02)

batched = batched_surface("high_fidelity", donor, transfer,
                          progress=False, mixing=0.52)

sim = build("high_fidelity", mixing=0.52)
worst = 0.0
for i, d in enumerate(donor):
    for j, t in enumerate(transfer):
        one = energy_at(sim, d, t)
        worst = max(worst, abs(one - batched[i, j]))

print("cells compared:", donor.size * transfer.size)
print("largest difference: %.3e eV" % worst)