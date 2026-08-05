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
    box_size_nanometers = (
        box_size
        * ARGON_SIGMA_METERS
        * 1e9
    )

    figure, axes = plt.subplots()

    axes.set_xlim(0.0, box_size_nanometers)
    axes.set_ylim(0.0, box_size_nanometers)
    axes.set_aspect("equal")

    axes.set_xlabel("X position (nm)")
    axes.set_ylabel("Y position (nm)")
    axes.set_title(
        "Two-Dimensional Lennard-Jones Argon Model"
    )
    axes.grid()

    particle_markers = axes.scatter(
        [],
        [],
        s=120
    )

    information_text = axes.text(
        0.02,
        0.98,
        "",
        transform=axes.transAxes,
        verticalalignment="top"
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

        particle_markers.set_offsets(
            current_positions_nanometers
        )

        information_text.set_text(
            f"Time: {time_picoseconds:.3f} ps\n"
            f"Temperature: {temperature_kelvin:.2f} K\n"
            f"Kinetic: "
            f"{kinetic_energy_history[frame_number]:.4f}\n"
            f"Potential: "
            f"{potential_energy_history[frame_number]:.4f}\n"
            f"Total: "
            f"{total_energy_history[frame_number]:.6f}\n"
            f"Energy drift: "
            f"{energy_drift_history[frame_number]:+.8f}"
        )

        return (
            particle_markers,
            information_text
        )

    animation = FuncAnimation(
        figure,
        update_animation,
        frames=len(position_history),
        interval=10,
        blit=True,
        repeat=True
    )

    plt.show()