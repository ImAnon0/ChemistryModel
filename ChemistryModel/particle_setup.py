import numpy as np


def create_square_grid(
    particles_per_side,
    box_size
):
    spacing = box_size / (particles_per_side + 1)

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