import numpy as np


def create_square_grid(
    particles_per_side,
    box_size,
    minimum_spacing=0.9
):
    spacing = box_size / (particles_per_side + 1)

    if spacing < minimum_spacing:
        raise ValueError(
            f"Particles are packed too closely. "
            f"Calculated spacing: {spacing:.3f}. "
            f"Minimum allowed spacing: {minimum_spacing:.3f}."
        )

    particle_positions = []

    for row_index in range(particles_per_side):
        for column_index in range(particles_per_side):
            x_position = (
                column_index + 1
            ) * spacing

            y_position = (
                row_index + 1
            ) * spacing

            particle_positions.append(
                [x_position, y_position]
            )

    return np.array(
        particle_positions,
        dtype=float
    )