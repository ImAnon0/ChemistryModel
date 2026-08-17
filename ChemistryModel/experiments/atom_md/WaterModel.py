import numpy as np

from ase import Atoms

from ase_live import AseLiveSimulation

from live import run_live_window


# ============================================================
# Choose where the forces come from
# ============================================================
#
# "mace"  = neural network trained on quantum (DFT) calculations.
#           Real chemistry: bonds can form and break. Needs a GPU
#           to be watchable, and `pip install mace-torch`.
#
# "lj"    = ASE's own Lennard-Jones. Fast, but argon-like: no
#           bonding. Useful for checking the setup works.

force_field = "mace"

molecules_per_side = 2
box_size_angstroms = 9.0

target_temperature_kelvin = 300.0
time_step_femtoseconds = 0.5


# ============================================================
# Build a small box of water
# ============================================================

spacing = box_size_angstroms / molecules_per_side

positions = []
symbols = []

for x_index in range(molecules_per_side):
    for y_index in range(molecules_per_side):
        for z_index in range(molecules_per_side):
            oxygen_position = (
                np.array([x_index, y_index, z_index]) * spacing
                + spacing / 2.0
            )

            positions.append(oxygen_position)
            positions.append(oxygen_position + [0.76, 0.59, 0.0])
            positions.append(oxygen_position + [-0.76, 0.59, 0.0])

            symbols += ["O", "H", "H"]

atoms = Atoms(
    symbols=symbols,
    positions=positions,
    cell=[box_size_angstroms] * 3,
    pbc=True
)


# ============================================================
# Attach a calculator
# ============================================================

if force_field == "mace":
    from mace.calculators import mace_mp

    atoms.calc = mace_mp(
        model="small",
        default_dtype="float32",
        device="cpu"      # change to "cuda" if you have an NVIDIA GPU
    )
else:
    from ase.calculators.lj import LennardJones

    atoms.calc = LennardJones(
        sigma=3.166,
        epsilon=0.0067,
        rc=8.0
    )


simulation = AseLiveSimulation(
    atoms=atoms,
    target_temperature_kelvin=target_temperature_kelvin,
    time_step_femtoseconds=time_step_femtoseconds,
    friction_per_femtosecond=0.01
)

print(
    f"{len(atoms)} atoms, {force_field} forces, "
    f"box {box_size_angstroms} A"
)

# The same window as before. It does not know or care that the
# forces now come from a neural network.

run_live_window(
    simulation=simulation,
    steps_per_frame=1
)
