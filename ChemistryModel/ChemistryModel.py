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

force_field = "mace"

compute_device = "cuda"

# FIX 1: box size.
#
# MACE-MP uses a 6 A radial cutoff, and message passing over
# several layers makes its effective receptive field larger
# still. The old 9 A box gives a half-box of 4.5 A, so every
# atom was interacting with its own periodic images. This is
# exactly the error the guard in interactions.py raises for the
# Lennard-Jones path, and it applies just as much here.

box_size_angstroms = 16.0


# FIX 2: composition.
#
# The old mixture was 14 oxygens to 4 hydrogens: strongly
# oxidising, so the thermodynamic sink is CO2, N2 and NO. No
# organic chemistry can survive there.
#
# Prebiotic synthesis needs reducing conditions. Miller-Urey ran
# methane, ammonia, water and hydrogen, where hydrogen is by far
# the most abundant atom. Molecules are listed here rather than
# bare element counts, for FIX 3.

starting_molecules = {
    "CH4": 4,
    "NH3": 3,
    "H2O": 4,
    "H2": 6
}


# FIX 3 is in build_molecule below: the simulation now starts
# from real molecules rather than loose atoms.


# FIX 4: temperature.
#
# 150 K is too cold for anything with an activation barrier to
# happen. Miller-Urey style chemistry is driven by energy input,
# not by ambient heat, so the bath is set near room temperature
# and the energy arrives as sparks instead.

target_temperature_kelvin = 300.0


hydrogen_mass = 3.0

time_step_femtoseconds = 2.0

steps_per_frame = 4


# FIX 5: thermostat friction.
#
# 0.05 per fs is a damping time of 20 fs, which is faster than a
# bond vibrates. That turns Newtonian dynamics into Brownian
# motion and drains the heat of every bond-forming event before
# it can drive anything. Standard Langevin friction for this kind
# of run is 100 to 500 times weaker.

friction_per_femtosecond = 0.003


# Spark settings. Every so often one atom is given a large
# velocity kick, standing in for a UV photon or a lightning
# discharge. Without something like this a warm equilibrium
# mixture simply sits there.

spark_interval_steps = 2000
spark_energy_electronvolts = 8.0

random_seed = 1


# ============================================================
# Build real molecules, not loose atoms
# ============================================================
#
# Starting from free atoms means every radical recombines
# barrierlessly in the first picosecond, freezing the system into
# whatever it happened to touch first. Real chemistry starts from
# stable molecules and needs energy to break them open.

BOND_LENGTHS = {
    "CH4": 1.09,
    "NH3": 1.02,
    "H2O": 0.96,
    "H2": 0.74
}


def build_molecule(name, generator):
    length = BOND_LENGTHS[name]

    if name == "H2":
        return ["H", "H"], np.array([
            [0.0, 0.0, 0.0],
            [length, 0.0, 0.0]
        ])

    if name == "H2O":
        angle = np.deg2rad(104.5)
        return ["O", "H", "H"], np.array([
            [0.0, 0.0, 0.0],
            [length, 0.0, 0.0],
            [
                length * np.cos(angle),
                length * np.sin(angle),
                0.0
            ]
        ])

    if name == "NH3":
        angle = np.deg2rad(107.0)
        height = length * np.cos(angle / 2.0)
        radius = length * np.sin(angle / 2.0)

        coordinates = [[0.0, 0.0, 0.0]]

        for index in range(3):
            theta = 2.0 * np.pi * index / 3.0
            coordinates.append([
                radius * np.cos(theta),
                radius * np.sin(theta),
                -height
            ])

        return ["N", "H", "H", "H"], np.array(coordinates)

    if name == "CH4":
        directions = np.array([
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0]
        ]) / np.sqrt(3.0)

        coordinates = np.vstack([
            np.zeros(3),
            directions * length
        ])

        return ["C", "H", "H", "H", "H"], coordinates

    raise ValueError(f"Unknown molecule {name}")


def random_rotation(generator):
    # Uniform random rotation via QR decomposition of a Gaussian
    # matrix, with the determinant fixed to +1.

    matrix, _ = np.linalg.qr(generator.normal(size=(3, 3)))

    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1.0

    return matrix


def build_starting_atoms(molecules, box_size, generator):
    total = sum(molecules.values())

    sites_per_side = int(np.ceil(total ** (1.0 / 3.0)))
    spacing = box_size / sites_per_side

    grid = np.stack(
        np.meshgrid(
            *(np.arange(sites_per_side),) * 3,
            indexing="ij"
        ),
        axis=-1
    ).reshape(-1, 3)

    chosen = generator.choice(len(grid), size=total, replace=False)

    centres = (grid[chosen] + 0.5) * spacing

    symbols = []
    positions = []

    index = 0

    for name, count in molecules.items():
        for _ in range(count):
            molecule_symbols, coordinates = build_molecule(
                name,
                generator
            )

            rotated = coordinates @ random_rotation(generator).T

            symbols += molecule_symbols
            positions.append(rotated + centres[index])

            index += 1

    return symbols, np.vstack(positions)


random_generator = np.random.default_rng(seed=random_seed)

symbols, positions = build_starting_atoms(
    starting_molecules,
    box_size_angstroms,
    random_generator
)

atoms = Atoms(
    symbols=symbols,
    positions=positions,
    cell=[box_size_angstroms] * 3,
    pbc=True
)

if hydrogen_mass is not None:
    masses = atoms.get_masses()

    for index, symbol in enumerate(atoms.get_chemical_symbols()):
        if symbol == "H":
            masses[index] = hydrogen_mass

    atoms.set_masses(masses)


# ============================================================
# Calculator
# ============================================================

if force_field == "mace":
    from mace.calculators import mace_mp

    if compute_device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                compute_device = "cpu"
        except ImportError:
            compute_device = "cpu"

    atoms.calc = mace_mp(
        model="small",
        default_dtype="float32",
        device=compute_device
    )
else:
    from ase.calculators.lj import LennardJones

    atoms.calc = LennardJones(sigma=3.166, epsilon=0.0067, rc=8.0)


# ============================================================
# Sparks
# ============================================================

class SparkingSimulation(AseLiveSimulation):
    # Identical to AseLiveSimulation except that every so often a
    # single atom is kicked hard. Bond breaking has an activation
    # barrier that a 300 K bath will essentially never supply, so
    # without an energy source the mixture just sits at
    # equilibrium doing nothing.

    def __init__(
        self,
        *args,
        spark_interval_steps=2000,
        spark_energy_electronvolts=8.0,
        **keyword_arguments
    ):
        self.spark_interval_steps = spark_interval_steps
        self.spark_energy_electronvolts = spark_energy_electronvolts
        self.steps_taken = 0
        self.spark_count = 0

        super().__init__(*args, **keyword_arguments)

    def step(self, number_of_steps=1):
        for _ in range(number_of_steps):
            super().step(1)

            self.steps_taken += 1

            if (
                self.spark_interval_steps
                and self.steps_taken % self.spark_interval_steps == 0
            ):
                self._spark()

    def _spark(self):
        index = self.random_generator.integers(len(self.atoms))

        direction = self.random_generator.normal(size=3)
        direction /= np.linalg.norm(direction)

        mass = self.masses[index]

        speed = np.sqrt(
            2.0 * self.spark_energy_electronvolts / mass
        )

        self.velocities[index] += direction * speed

        self.spark_count += 1


simulation = SparkingSimulation(
    atoms=atoms,
    target_temperature_kelvin=target_temperature_kelvin,
    time_step_femtoseconds=time_step_femtoseconds,
    friction_per_femtosecond=friction_per_femtosecond,
    spark_interval_steps=spark_interval_steps,
    spark_energy_electronvolts=spark_energy_electronvolts
)

composition_text = "  ".join(
    f"{count} {name}"
    for name, count in starting_molecules.items()
)

print(
    f"{len(atoms)} atoms as molecules ({composition_text}), "
    f"{force_field} on {compute_device}, "
    f"box {box_size_angstroms} A"
)

print(
    f"Starting potential energy: "
    f"{simulation.potential_energy:.3f} eV"
)

print(
    f"Timestep {time_step_femtoseconds} fs x {steps_per_frame} "
    f"steps per frame, spark every {spark_interval_steps} steps"
)

run_live_window(
    simulation=simulation,
    steps_per_frame=steps_per_frame
)