import numpy as np


def create_square_grid(
    particles_per_side,
    box_size,
    minimum_spacing=0.9
):
    # Two-dimensional starting grid. Kept from the original
    # version so the older 2D setup still runs.

    spacing = box_size / particles_per_side

    if spacing < minimum_spacing:
        raise ValueError(
            f"Particles are packed too closely. "
            f"Calculated spacing: {spacing:.3f}. "
            f"Minimum allowed spacing: {minimum_spacing:.3f}."
        )

    particle_positions = []

    for row_index in range(particles_per_side):
        for column_index in range(particles_per_side):
            particle_positions.append([
                (column_index + 0.5) * spacing,
                (row_index + 0.5) * spacing
            ])

    return np.array(
        particle_positions,
        dtype=float
    )


def create_cubic_grid(
    particles_per_side,
    box_size,
    minimum_spacing=0.9
):
    # Simple cubic lattice. Easy to reason about, but it is not
    # how argon actually packs, so it will collapse into a
    # different structure almost immediately.

    spacing = box_size / particles_per_side

    if spacing < minimum_spacing:
        raise ValueError(
            f"Particles are packed too closely. "
            f"Calculated spacing: {spacing:.3f}. "
            f"Minimum allowed spacing: {minimum_spacing:.3f}."
        )

    particle_positions = []

    for layer_index in range(particles_per_side):
        for row_index in range(particles_per_side):
            for column_index in range(particles_per_side):
                particle_positions.append([
                    (column_index + 0.5) * spacing,
                    (row_index + 0.5) * spacing,
                    (layer_index + 0.5) * spacing
                ])

    return np.array(
        particle_positions,
        dtype=float
    )


FACE_CENTRED_CUBIC_BASIS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0]
])


def create_face_centred_cubic_lattice(
    unit_cells_per_side,
    number_density
):
    # The face centred cubic lattice is the arrangement a
    # Lennard-Jones solid actually settles into, so starting
    # here means the crystal is stable rather than immediately
    # rearranging itself.
    #
    # Returns the positions AND the box size, because the box
    # size is fixed by the density once the cell count is chosen.
    #
    # There are 4 particles per unit cell, so:
    #     number_density = 4 / cell_length ** 3

    cell_length = (4.0 / number_density) ** (1.0 / 3.0)

    box_size = unit_cells_per_side * cell_length

    particle_positions = []

    for x_cell in range(unit_cells_per_side):
        for y_cell in range(unit_cells_per_side):
            for z_cell in range(unit_cells_per_side):
                cell_origin = np.array([
                    x_cell,
                    y_cell,
                    z_cell
                ], dtype=float)

                for basis_offset in FACE_CENTRED_CUBIC_BASIS:
                    particle_positions.append(
                        (cell_origin + basis_offset)
                        * cell_length
                    )

    return (
        np.array(particle_positions, dtype=float),
        box_size
    )


def create_thermal_velocities(
    particle_positions,
    particle_mass,
    target_temperature,
    random_seed=12
):
    # Random velocities, centre-of-mass drift removed, then
    # rescaled so the kinetic energy matches the requested
    # temperature exactly.

    random_generator = np.random.default_rng(seed=random_seed)

    particle_velocities = random_generator.normal(
        loc=0.0,
        scale=1.0,
        size=particle_positions.shape
    )

    particle_velocities -= np.mean(
        particle_velocities,
        axis=0
    )

    degrees_of_freedom = count_degrees_of_freedom(
        particle_positions
    )

    current_kinetic_energy = (
        0.5
        * particle_mass
        * np.sum(particle_velocities ** 2)
    )

    current_temperature = (
        2.0
        * current_kinetic_energy
        / degrees_of_freedom
    )

    particle_velocities *= np.sqrt(
        target_temperature
        / current_temperature
    )

    return particle_velocities


def count_degrees_of_freedom(particle_positions):
    # One whole dimension is removed because centre-of-mass
    # motion was subtracted and does not count as heat.

    particle_count, number_of_dimensions = (
        particle_positions.shape
    )

    return (
        particle_count * number_of_dimensions
        - number_of_dimensions
    )
