import numpy as np

from argon import ARGON_SIGMA_METERS


def write_xyz_trajectory(
    output_path,
    position_history,
    element_symbol="Ar",
    length_unit_meters=ARGON_SIGMA_METERS
):
    # Writes a standard multi-frame XYZ file.
    #
    # Open the result in OVITO (free) to get proper 3D rendering,
    # automatic bond detection by distance, and colouring by
    # coordination number. That last one is the diagnostic you
    # will want once bond-order potentials go in.

    metres_to_angstroms = 1e10

    with open(output_path, "w") as output_file:
        for frame_index, frame_positions in enumerate(
            position_history
        ):
            positions_in_angstroms = (
                np.asarray(frame_positions)
                * length_unit_meters
                * metres_to_angstroms
            )

            particle_count = len(positions_in_angstroms)

            output_file.write(f"{particle_count}\n")
            output_file.write(f"frame {frame_index}\n")

            for position in positions_in_angstroms:
                x_position, y_position, z_position = position

                output_file.write(
                    f"{element_symbol} "
                    f"{x_position:.5f} "
                    f"{y_position:.5f} "
                    f"{z_position:.5f}\n"
                )

    return output_path
