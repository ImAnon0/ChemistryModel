# Thread count must be set before torch does anything else, so
# this import block stays at the very top of the file.

try:
    import torch

    torch.set_num_threads(cpu_thread_count := 8)
except ImportError:
    pass

import numpy as np

from ase import Atoms

from ase_live import AseLiveSimulation

from live import run_live_window


# ============================================================
# Settings
# ============================================================
#
# "mace" = neural network trained on quantum calculations.
#          Bonds can form and break.
# "lj"   = ASE's Lennard-Jones. No bonding. Fast sanity check.

force_field = "mace"

# "cpu" always works. Only use "cuda" if this prints True:
#   python -c "import torch; print(torch.cuda.is_available())"
compute_device = "cuda"

box_size_angstroms = 9.0

# Put whatever you like in here. MACE covers most of the
# periodic table, so adding an element costs nothing but a line.
#
#   {"O": 6, "H": 12}            water-ish
#   {"C": 3, "H": 8, "O": 4}     organic chemistry
#   {"C": 2, "H": 6, "O": 3, "N": 2}   Miller-Urey ingredients

composition = {
    "C": 7,
    "H": 4,
    "O": 14,
    "N": 8
}

# Start cold. Bond formation releases heat, so the system warms
# itself. Starting hot means nothing gets a chance to bond.
target_temperature_kelvin = 150.0

# Hydrogen is light, so its bonds vibrate fastest and normally
# force a tiny timestep. Weighting hydrogen to 3 amu slows that
# vibration and lets the timestep go four times larger, with no
# change to equilibrium structure or thermodynamics. Only
# hydrogen's own vibrational frequencies are affected.
#
# Set to None to use real hydrogen mass, and drop the timestep
# back to 0.5 fs if you do.

hydrogen_mass = 3.0

time_step_femtoseconds = 2.0

# How many simulation steps happen between screen redraws.
# Higher means less smooth but far more simulated time.

steps_per_frame = 4

minimum_starting_separation = 1.4

random_seed = 1


# ============================================================
# Scatter loose atoms, with nothing bonded
# ============================================================

random_generator = np.random.default_rng(seed=random_seed)

symbols = []

for element, count in composition.items():
    symbols += [element] * count

positions = []

while len(positions) < len(symbols):
    candidate = random_generator.uniform(
        0.0,
        box_size_angstroms,
        size=3
    )

    too_close = False

    for existing in positions:
        separation = (
            candidate - existing + box_size_angstroms / 2.0
        ) % box_size_angstroms - box_size_angstroms / 2.0

        if np.linalg.norm(separation) < minimum_starting_separation:
            too_close = True
            break

    if not too_close:
        positions.append(candidate)

atoms = Atoms(
    symbols=symbols,
    positions=positions,
    cell=[box_size_angstroms] * 3,
    pbc=True
)


if hydrogen_mass is not None:
    masses = atoms.get_masses()

    for index, symbol in enumerate(
        atoms.get_chemical_symbols()
    ):
        if symbol == "H":
            masses[index] = hydrogen_mass

    atoms.set_masses(masses)


# ============================================================
# Attach a calculator
# ============================================================

if force_field == "mace":
    from mace.calculators import mace_mp

    atoms.calc = mace_mp(
        model="small",
        default_dtype="float32",
        device=compute_device
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
    friction_per_femtosecond=0.05
)

composition_text = "  ".join(
    f"{count} {element}"
    for element, count in composition.items()
)

print(
    f"{len(atoms)} loose atoms ({composition_text}), "
    f"{force_field} forces on {compute_device}, "
    f"box {box_size_angstroms} A"
)

print(
    f"Starting potential energy: "
    f"{simulation.potential_energy:.3f} eV"
)

simulated_femtoseconds_per_frame = (
    time_step_femtoseconds * steps_per_frame
)

print(
    f"Timestep {time_step_femtoseconds} fs x "
    f"{steps_per_frame} steps per frame = "
    f"{simulated_femtoseconds_per_frame} fs per frame"
)

print("Watch the potential energy drop as bonds form.")

run_live_window(
    simulation=simulation,
    steps_per_frame=steps_per_frame
)