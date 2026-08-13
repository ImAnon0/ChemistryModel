import numpy as np

import reactive as R


MOLECULE_GEOMETRY = {
    "H2":  (["H", "H"], np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])),
    "H2O": (["O", "H", "H"], None),
    "CH4": (["C", "H", "H", "H", "H"], None),
    "NH3": (["N", "H", "H", "H"], None),
}


def water():
    angle = np.deg2rad(104.5)
    r = 0.96

    return np.array([
        [0.0, 0.0, 0.0],
        [r, 0.0, 0.0],
        [r * np.cos(angle), r * np.sin(angle), 0.0]
    ])


def methane():
    r = 1.09

    directions = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float) / np.sqrt(3.0)

    return np.vstack([np.zeros(3), directions * r])


def ammonia():
    r = 1.01
    angle = np.deg2rad(107.0)

    height = r * np.cos(angle / 2.0)
    radius = r * np.sin(angle / 2.0)

    coordinates = [[0.0, 0.0, 0.0]]

    for index in range(3):
        theta = 2.0 * np.pi * index / 3.0
        coordinates.append([
            radius * np.cos(theta),
            radius * np.sin(theta),
            -height
        ])

    return np.array(coordinates)


def amidogen():
    """Experimental-like bent NH2 radical used by N-N calibration runs."""
    r = 1.0109
    half_angle = 0.5 * np.deg2rad(106.75)
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.cos(half_angle), r * np.sin(half_angle), 0.0],
        [r * np.cos(half_angle), -r * np.sin(half_angle), 0.0],
    ])


def hydroxyl():
    """OH radical geometry used by focused O-O calibration runs."""
    return np.array([
        [0.0, 0.0, 0.0],
        [0.96, 0.0, 0.0],
    ])


BUILDERS = {
    "H2": lambda: (["H", "H"],
                   np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])),
    "H2O": lambda: (["O", "H", "H"], water()),
    "CH4": lambda: (["C", "H", "H", "H", "H"], methane()),
    "NH3": lambda: (["N", "H", "H", "H"], ammonia()),
    "NH2": lambda: (["N", "H", "H"], amidogen()),
    "OH": lambda: (["O", "H"], hydroxyl()),
}


def random_rotation(generator):
    matrix, _ = np.linalg.qr(generator.normal(size=(3, 3)))

    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1.0

    return matrix


def build(composition, box_size, random_seed=0):
    # composition: {"H2O": 20, "CH4": 8, ...}

    generator = np.random.default_rng(random_seed)

    total = sum(composition.values())

    sites_per_side = int(np.ceil(total ** (1.0 / 3.0)))
    spacing = box_size / sites_per_side

    grid = np.stack(
        np.meshgrid(*(np.arange(sites_per_side),) * 3, indexing="ij"),
        axis=-1
    ).reshape(-1, 3)

    chosen = generator.choice(len(grid), size=total, replace=False)
    centres = (grid[chosen] + 0.5) * spacing

    symbols = []
    positions = []

    slot = 0

    for name, number in composition.items():
        for _ in range(number):
            molecule_symbols, coordinates = BUILDERS[name]()

            rotated = coordinates @ random_rotation(generator).T

            symbols += molecule_symbols
            positions.append(rotated + centres[slot])

            slot += 1

    return symbols, np.vstack(positions) % box_size


def loose_atoms(counts, box_size, minimum_separation=1.6,
                random_seed=0):
    # Free atoms, for watching molecules assemble themselves.

    generator = np.random.default_rng(random_seed)

    symbols = []

    for element, number in counts.items():
        symbols += [element] * number

    positions = []

    attempts = 0

    while len(positions) < len(symbols) and attempts < 200000:
        attempts += 1

        candidate = generator.uniform(0.0, box_size, size=3)

        if positions:
            offsets = np.array(positions) - candidate
            offsets -= box_size * np.round(offsets / box_size)

            if np.min(np.linalg.norm(offsets, axis=1)) < minimum_separation:
                continue

        positions.append(candidate)

    generator.shuffle(symbols)

    return symbols, np.array(positions)
