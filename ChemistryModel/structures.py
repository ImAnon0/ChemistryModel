import numpy as np

from membrane import BEADS_PER_LIPID


# ============================================================
# Pre-built starting structures
# ============================================================
#
# Every function takes the model and a random generator, and
# returns a positions array of shape (bead_count, 3).
#
# Lipids are laid out head first: bead 0 is the head, beads 1 and
# 2 are tails running away from it along a direction vector.

BOND_LENGTH = 0.95

LIPID_LENGTH = BOND_LENGTH * (BEADS_PER_LIPID - 1)


def place_lipids(positions, start_index, heads, directions):
    # Writes one lipid per (head, direction) pair. Tails run from
    # the head along the direction given.

    count = len(heads)

    for bead in range(BEADS_PER_LIPID):
        slice_start = (start_index + bead)
        indices = (
            np.arange(count) * BEADS_PER_LIPID
            + slice_start
        )

        positions[indices] = (
            heads + directions * BOND_LENGTH * bead
        )

    return start_index + count * BEADS_PER_LIPID


def fibonacci_sphere(count):
    index = np.arange(count) + 0.5

    phi = np.arccos(1.0 - 2.0 * index / count)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * index

    return np.stack([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi)
    ], axis=1)


def build_vesicle_at(positions, offset, centre, count, radius=None):
    # Two concentric shells of lipids, tails facing each other.

    if radius is None:
        radius = np.sqrt(count * 1.25 / (8.0 * np.pi))

    radius = max(radius, LIPID_LENGTH * 1.6)

    half_thickness = LIPID_LENGTH + 0.5

    # The two leaflets sit at different radii, so an even split
    # crams the inner shell far too tightly. Divide by surface
    # area instead: the outer shell holds more lipids in
    # proportion to the square of its radius.

    outer_radius = radius + half_thickness
    inner_radius = max(radius - half_thickness, LIPID_LENGTH * 0.6)

    outer_share = outer_radius ** 2
    inner_share = inner_radius ** 2

    outer_count = int(
        round(count * outer_share / (outer_share + inner_share))
    )

    outer_count = max(1, min(count - 1, outer_count))
    inner_count = count - outer_count

    outer_directions = fibonacci_sphere(outer_count)
    inner_directions = fibonacci_sphere(inner_count)

    # Outer leaflet: heads point away from centre, tails inward.

    outer_heads = centre + outer_directions * outer_radius

    offset = place_lipids(
        positions,
        offset,
        outer_heads,
        -outer_directions
    )

    # Inner leaflet: heads face the cavity, tails outward.

    inner_heads = centre + inner_directions * inner_radius

    offset = place_lipids(
        positions,
        offset,
        inner_heads,
        inner_directions
    )

    return offset


# ============================================================


def random_scatter(model, generator):
    return model.random_configuration(generator)


def single_vesicle(model, generator):
    positions = np.zeros((model.bead_count, 3))

    centre = np.full(3, model.box_size / 2.0)

    build_vesicle_at(
        positions,
        0,
        centre,
        model.number_of_lipids
    )

    jitter = generator.normal(scale=0.04, size=positions.shape)

    return (positions + jitter) % model.box_size


def two_vesicles(model, generator):
    # Two separate vesicles placed close together. Left running,
    # they will drift, touch and sometimes fuse.

    positions = np.zeros((model.bead_count, 3))

    half = model.number_of_lipids // 2

    radius = np.sqrt(half * 1.25 / (8.0 * np.pi))
    radius = max(radius, LIPID_LENGTH * 1.6)

    centre = model.box_size / 2.0

    separation = radius * 2.4 + LIPID_LENGTH * 2.0

    first_centre = np.array([
        centre - separation / 2.0,
        centre,
        centre
    ])

    second_centre = np.array([
        centre + separation / 2.0,
        centre,
        centre
    ])

    offset = build_vesicle_at(
        positions,
        0,
        first_centre,
        half,
        radius
    )

    build_vesicle_at(
        positions,
        offset,
        second_centre,
        model.number_of_lipids - half,
        radius
    )

    jitter = generator.normal(scale=0.04, size=positions.shape)

    return (positions + jitter) % model.box_size


def flat_bilayer(model, generator):
    # A sheet spanning the whole box, so the periodic boundary
    # makes it effectively infinite with no edges.

    positions = np.zeros((model.bead_count, 3))

    per_leaflet = model.number_of_lipids // 2

    columns = int(np.round(np.sqrt(per_leaflet)))
    rows = int(np.ceil(per_leaflet / columns))

    spacing_x = model.box_size / columns
    spacing_y = model.box_size / rows

    grid_x, grid_y = np.meshgrid(
        (np.arange(columns) + 0.5) * spacing_x,
        (np.arange(rows) + 0.5) * spacing_y,
        indexing="ij"
    )

    sites = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    middle = model.box_size / 2.0

    gap = LIPID_LENGTH + 0.55

    up = np.array([0.0, 0.0, 1.0])

    offset = 0
    used = 0

    for sign in (-1.0, 1.0):
        remaining = model.number_of_lipids - used

        take = min(per_leaflet, remaining)

        if sign > 0:
            take = remaining

        chosen = sites[:take]

        heads = np.column_stack([
            chosen[:, 0],
            chosen[:, 1],
            np.full(take, middle + sign * gap)
        ])

        directions = np.tile(-sign * up, (take, 1))

        offset = place_lipids(positions, offset, heads, directions)

        used += take

    jitter = generator.normal(scale=0.04, size=positions.shape)

    return (positions + jitter) % model.box_size


def micelle_field(model, generator, lipids_per_micelle=45):
    # Many small single-layer balls: tails in, heads out, no
    # cavity. Useful next to a vesicle for comparison.

    positions = np.zeros((model.bead_count, 3))

    count = model.number_of_lipids

    number_of_micelles = max(1, count // lipids_per_micelle)

    sites_per_side = int(np.ceil(number_of_micelles ** (1.0 / 3.0)))
    spacing = model.box_size / sites_per_side

    grid = np.stack(
        np.meshgrid(
            *(np.arange(sites_per_side),) * 3,
            indexing="ij"
        ),
        axis=-1
    ).reshape(-1, 3)

    chosen = generator.choice(
        len(grid),
        size=number_of_micelles,
        replace=False
    )

    centres = (grid[chosen] + 0.5) * spacing

    offset = 0
    placed = 0

    for index in range(number_of_micelles):
        if index == number_of_micelles - 1:
            take = count - placed
        else:
            take = lipids_per_micelle

        if take <= 0:
            break

        directions = fibonacci_sphere(take)

        radius = max(
            np.sqrt(take * 1.25 / (4.0 * np.pi)),
            LIPID_LENGTH * 1.6
        )

        heads = centres[index] + directions * radius

        # Every tail pointing exactly at the centre would put all
        # of them on the same tiny sphere and overlap horribly.
        # Real micelle cores are disordered, so each lipid is
        # given a small random tilt that grows along its length.

        tilt = generator.normal(scale=0.45, size=(take, 3))

        tilt -= (
            np.sum(tilt * directions, axis=1)[:, None] * directions
        )

        for bead in range(BEADS_PER_LIPID):
            indices = (
                np.arange(take) * BEADS_PER_LIPID
                + offset
                + bead
            )

            step = -directions + tilt * bead

            step /= np.linalg.norm(step, axis=1, keepdims=True)

            positions[indices] = (
                heads + step * BOND_LENGTH * bead
            )

        offset += take * BEADS_PER_LIPID

        placed += take

    jitter = generator.normal(scale=0.04, size=positions.shape)

    return (positions + jitter) % model.box_size


STRUCTURES = {
    "random scatter": random_scatter,
    "single vesicle": single_vesicle,
    "two vesicles": two_vesicles,
    "flat bilayer": flat_bilayer,
    "micelle field": micelle_field
}


def build(name, model, generator):
    if name not in STRUCTURES:
        raise ValueError(
            f"Unknown structure {name}. "
            f"Options: {sorted(STRUCTURES)}"
        )

    return STRUCTURES[name](model, generator)
