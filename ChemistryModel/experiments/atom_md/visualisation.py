import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from argon import (
    ARGON_SIGMA_METERS,
    ARGON_EPSILON_OVER_KELVIN,
    ARGON_TIME_UNIT_SECONDS
)


def show_animation(
    position_history,
    time_history,
    kinetic_energy_history,
    potential_energy_history,
    total_energy_history,
    energy_drift_history,
    temperature_history,
    box_size
):
    # Matplotlib has no real depth sorting, so this is a quick
    # look rather than a proper renderer. Past a few hundred
    # particles, use trajectory.write_xyz_trajectory and OVITO.

    box_size_nanometers = (
        box_size
        * ARGON_SIGMA_METERS
        * 1e9
    )

    figure = plt.figure(figsize=(8, 7))

    axes = figure.add_subplot(111, projection="3d")

    axes.set_xlim(0.0, box_size_nanometers)
    axes.set_ylim(0.0, box_size_nanometers)
    axes.set_zlim(0.0, box_size_nanometers)

    axes.set_box_aspect((1.0, 1.0, 1.0))

    axes.set_xlabel("X position (nm)")
    axes.set_ylabel("Y position (nm)")
    axes.set_zlabel("Z position (nm)")

    axes.set_title(
        "Three-Dimensional Lennard-Jones Argon Model"
    )

    starting_positions_nanometers = (
        position_history[0]
        * ARGON_SIGMA_METERS
        * 1e9
    )

    particle_markers = axes.scatter(
        starting_positions_nanometers[:, 0],
        starting_positions_nanometers[:, 1],
        starting_positions_nanometers[:, 2],
        s=40,
        depthshade=True
    )

    information_text = figure.text(
        0.02,
        0.97,
        "",
        verticalalignment="top",
        family="monospace",
        fontsize=9
    )

    def update_animation(frame_number):
        current_positions_nanometers = (
            position_history[frame_number]
            * ARGON_SIGMA_METERS
            * 1e9
        )

        time_picoseconds = (
            time_history[frame_number]
            * ARGON_TIME_UNIT_SECONDS
            * 1e12
        )

        temperature_kelvin = (
            temperature_history[frame_number]
            * ARGON_EPSILON_OVER_KELVIN
        )

        # 3D scatter needs both calls; set_offsets alone is 2D only.

        particle_markers._offsets3d = (
            current_positions_nanometers[:, 0],
            current_positions_nanometers[:, 1],
            current_positions_nanometers[:, 2]
        )

        information_text.set_text(
            f"Time:         {time_picoseconds:8.3f} ps\n"
            f"Temperature:  {temperature_kelvin:8.2f} K\n"
            f"Kinetic:      {kinetic_energy_history[frame_number]:8.3f}\n"
            f"Potential:    {potential_energy_history[frame_number]:8.3f}\n"
            f"Total:        {total_energy_history[frame_number]:8.4f}\n"
            f"Energy drift: {energy_drift_history[frame_number]:+8.5f}"
        )

        return particle_markers, information_text

    animation = FuncAnimation(
        figure,
        update_animation,
        frames=len(position_history),
        interval=30,
        blit=False,
        repeat=True
    )

    # Keeping a reference stops the animation being garbage
    # collected before the window opens.

    figure._chemistry_model_animation = animation

    plt.show()
