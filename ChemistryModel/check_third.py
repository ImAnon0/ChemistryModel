import numpy as np
from test_high_fidelity import make, energy_at

def geometry(third):
    return np.array([[0,0,0],[1.05,0,0],[2.30,0,0],[1.05,third,0]], float)

sim = make(["O","H","C","O"], geometry(2.20))
far = energy_at(sim, geometry(2.20))

for y in np.arange(0.90, 2.21, 0.05):
    print("y=%.2f  E-far = %+.4f eV" % (y, energy_at(sim, geometry(y)) - far))