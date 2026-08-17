from collections import deque

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

from argon import (
    ARGON_SIGMA_METERS,
    ARGON_EPSILON_OVER_KELVIN,
    ARGON_TIME_UNIT_SECONDS
)

from interactions import calculate_all_interactions

from particle_setup import (
    create_face_centred_cubic_lattice,
    create_thermal_velocities,
    count_degrees_of_freedom
)

from thermostat import LangevinThermostat, calculate_temperature


class LiveSimulation:
    # Holds the simulation state and advances it on demand,
    # instead of running to completion and recording history.
    # Anything on this object can be changed while it runs.

    def __init__(
        self,
        unit_cells_per_side=3,
        number_density=0.85,
        particle_mass=1.0,
        time_step=0.002,
        cutoff_distance=2.5,
        target_temperature_kelvin=120.0,
        thermostat_friction=2.0,
        random_seed=12
    ):
        self.unit_cells_per_side = unit_cells_per_side
        self.number_density = number_density
        self.particle_mass = particle_mass
        self.time_step = time_step
        self.cutoff_distance = cutoff_distance
        self.random_seed = random_seed

        self.thermostat_is_on = True

        self.thermostat = LangevinThermostat(
            target_temperature=(
                target_temperature_kelvin
                / ARGON_EPSILON_OVER_KELVIN
            ),
            friction=thermostat_friction,
            random_seed=random_seed
        )

        self.reset()

    # --------------------------------------------------------

    @property
    def target_temperature_kelvin(self):
        return (
            self.thermostat.target_temperature
            * ARGON_EPSILON_OVER_KELVIN
        )

    @target_temperature_kelvin.setter
    def target_temperature_kelvin(self, value):
        self.thermostat.target_temperature = (
            value / ARGON_EPSILON_OVER_KELVIN
        )

    @property
    def temperature(self):
        return calculate_temperature(
            self.particle_velocities,
            self.particle_mass,
            self.degrees_of_freedom
        )

    @property
    def temperature_kelvin(self):
        return self.temperature * ARGON_EPSILON_OVER_KELVIN

    @property
    def elapsed_picoseconds(self):
        return (
            self.elapsed_time
            * ARGON_TIME_UNIT_SECONDS
            * 1e12
        )

    @property
    def positions_in_nanometers(self):
        return (
            self.particle_positions
            * ARGON_SIGMA_METERS
            * 1e9
        )

    @property
    def box_size_nanometers(self):
        return self.box_size * ARGON_SIGMA_METERS * 1e9

    # --------------------------------------------------------

    def reset(self):
        (
            self.particle_positions,
            self.box_size
        ) = create_face_centred_cubic_lattice(
            unit_cells_per_side=self.unit_cells_per_side,
            number_density=self.number_density
        )

        self.degrees_of_freedom = count_degrees_of_freedom(
            self.particle_positions
        )

        self.particle_velocities = create_thermal_velocities(
            particle_positions=self.particle_positions,
            particle_mass=self.particle_mass,
            target_temperature=self.thermostat.target_temperature,
            random_seed=self.random_seed
        )

        self.forces, self.potential_energy = (
            calculate_all_interactions(
                self.particle_positions,
                self.box_size,
                cutoff_distance=self.cutoff_distance
            )
        )

        self.elapsed_time = 0.0

    def step(self, number_of_steps=1):
        for _ in range(number_of_steps):
            self.particle_positions = (
                self.particle_positions
                + self.particle_velocities * self.time_step
                + 0.5
                * (self.forces / self.particle_mass)
                * self.time_step ** 2
            )

            self.particle_positions %= self.box_size

            new_forces, self.potential_energy = (
                calculate_all_interactions(
                    self.particle_positions,
                    self.box_size,
                    cutoff_distance=self.cutoff_distance
                )
            )

            self.particle_velocities = (
                self.particle_velocities
                + 0.5
                * (self.forces + new_forces)
                / self.particle_mass
                * self.time_step
            )

            self.forces = new_forces

            if self.thermostat_is_on:
                self.particle_velocities = self.thermostat.apply(
                    self.particle_velocities,
                    self.time_step,
                    self.particle_mass,
                    self.degrees_of_freedom
                )

            self.elapsed_time += self.time_step

    @property
    def kinetic_energy(self):
        return (
            0.5
            * self.particle_mass
            * np.sum(self.particle_velocities ** 2)
        )

    @property
    def total_energy(self):
        return self.kinetic_energy + self.potential_energy


def run_live_window(
    simulation=None,
    steps_per_frame=8,
    frame_interval_milliseconds=40,
    trace_length=400,
    species_every_frames=5,
    minimum_temperature_kelvin=1.0,
    maximum_temperature_kelvin=1500.0
):
    # Needs an interactive matplotlib backend (TkAgg, QtAgg).
    # If the window opens but nothing moves, that is the cause.

    if simulation is None:
        simulation = LiveSimulation()

    figure = plt.figure(figsize=(13.0, 7.0))

    figure.subplots_adjust(
        left=0.02,
        right=0.97,
        bottom=0.20,
        top=0.93,
        wspace=0.15
    )

    particle_axes = figure.add_subplot(1, 2, 1, projection="3d")
    trace_axes = figure.add_subplot(1, 2, 2)

    box_size_nanometers = simulation.box_size_nanometers

    particle_axes.set_xlim(0.0, box_size_nanometers)
    particle_axes.set_ylim(0.0, box_size_nanometers)
    particle_axes.set_zlim(0.0, box_size_nanometers)
    particle_axes.set_box_aspect((1.0, 1.0, 1.0))

    particle_axes.set_xlabel("X (nm)")
    particle_axes.set_ylabel("Y (nm)")
    particle_axes.set_zlabel("Z (nm)")

    starting_positions = simulation.positions_in_nanometers

    # Colour and size by element when the simulation is backed by
    # an ASE Atoms object. Falls back to plain dots otherwise.

    marker_colours = None
    marker_sizes = 45

    molecule_tracker = None

    chemistry_atoms = getattr(simulation, "atoms", None)

    if chemistry_atoms is not None:
        try:
            from species import (
                colours_for,
                sizes_for,
                MoleculeTracker
            )

            marker_colours = colours_for(chemistry_atoms)
            marker_sizes = sizes_for(chemistry_atoms)

            # Built once, here, because it carries bond history
            # between frames. Building it inside the draw callback
            # would reset that history every frame and defeat the
            # whole point of it.

            molecule_tracker = MoleculeTracker()
        except ImportError:
            pass

    particle_markers = particle_axes.scatter(
        starting_positions[:, 0],
        starting_positions[:, 1],
        starting_positions[:, 2],
        s=marker_sizes,
        c=marker_colours,
        edgecolors="black",
        linewidths=0.4,
        depthshade=True
    )

    if marker_colours is not None:
        from matplotlib.lines import Line2D
        from species import ELEMENT_COLOURS, DEFAULT_COLOUR

        present = []

        for symbol in chemistry_atoms.get_chemical_symbols():
            if symbol not in present:
                present.append(symbol)

        particle_axes.legend(
            handles=[
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="none",
                    markersize=8,
                    markerfacecolor=ELEMENT_COLOURS.get(
                        symbol, DEFAULT_COLOUR
                    ),
                    markeredgecolor="black",
                    label=symbol
                )
                for symbol in present
            ],
            loc="upper right",
            fontsize=9
        )

    time_trace = deque(maxlen=trace_length)
    temperature_trace = deque(maxlen=trace_length)

    # Single-element lists because the draw callback is a closure
    # and needs to mutate these without a nonlocal declaration.

    frames_drawn = [0]
    last_species_text = [""]

    measured_line, = trace_axes.plot(
        [],
        [],
        linewidth=1.4,
        label="Measured"
    )

    target_line, = trace_axes.plot(
        [],
        [],
        linewidth=1.2,
        linestyle="--",
        label="Target"
    )

    trace_axes.set_xlabel("Time (ps)")
    trace_axes.set_ylabel("Temperature (K)")
    trace_axes.set_ylim(
        0.0,
        maximum_temperature_kelvin * 1.1
    )
    trace_axes.grid(alpha=0.3)
    trace_axes.legend(loc="upper right", fontsize=9)

    readout_text = figure.text(
        0.02,
        0.965,
        "",
        family="monospace",
        fontsize=9,
        verticalalignment="top"
    )

    # ---- controls -------------------------------------------

    temperature_slider = Slider(
        ax=figure.add_axes([0.12, 0.09, 0.55, 0.03]),
        label="Target T (K)",
        valmin=minimum_temperature_kelvin,
        valmax=maximum_temperature_kelvin,
        valinit=simulation.target_temperature_kelvin,
        valstep=1.0
    )

    def on_temperature_changed(new_value):
        simulation.target_temperature_kelvin = new_value

    temperature_slider.on_changed(on_temperature_changed)

    control_state = {"paused": False}

    pause_button = Button(
        figure.add_axes([0.75, 0.085, 0.06, 0.04]),
        "Pause"
    )

    thermostat_button = Button(
        figure.add_axes([0.82, 0.085, 0.09, 0.04]),
        "Thermostat"
    )

    reset_button = Button(
        figure.add_axes([0.92, 0.085, 0.06, 0.04]),
        "Reset"
    )

    def on_pause_clicked(event):
        control_state["paused"] = not control_state["paused"]

        pause_button.label.set_text(
            "Resume" if control_state["paused"] else "Pause"
        )

    def on_thermostat_clicked(event):
        simulation.thermostat_is_on = (
            not simulation.thermostat_is_on
        )

    def on_reset_clicked(event):
        simulation.reset()

        time_trace.clear()
        temperature_trace.clear()

    pause_button.on_clicked(on_pause_clicked)
    thermostat_button.on_clicked(on_thermostat_clicked)
    reset_button.on_clicked(on_reset_clicked)

    # ---- animation ------------------------------------------

    def update_frame(frame_number):
        if not control_state["paused"]:
            simulation.step(steps_per_frame)

        if getattr(simulation, "diverged", False):
            # Freeze rather than carry on displaying nonsense.
            # Reset clears the flag and restarts cleanly.

            control_state["paused"] = True

            readout_text.set_text(
                "DIVERGED, run stopped\n\n"
                + simulation.divergence_reason
            )

            return particle_markers, measured_line, target_line

        positions = simulation.positions_in_nanometers

        particle_markers._offsets3d = (
            positions[:, 0],
            positions[:, 1],
            positions[:, 2]
        )

        time_trace.append(simulation.elapsed_picoseconds)
        temperature_trace.append(simulation.temperature_kelvin)

        measured_line.set_data(
            list(time_trace),
            list(temperature_trace)
        )

        target_line.set_data(
            [time_trace[0], time_trace[-1]],
            [
                simulation.target_temperature_kelvin,
                simulation.target_temperature_kelvin
            ]
        )

        if time_trace[-1] > time_trace[0]:
            trace_axes.set_xlim(
                time_trace[0],
                time_trace[-1]
            )

        thermostat_label = (
            "on " if simulation.thermostat_is_on else "off"
        )

        species_line = ""

        if molecule_tracker is not None:
            # The neighbour search and connected components pass
            # both run on the CPU while the GPU waits, so doing
            # this every frame throttles the whole animation.
            # Between updates the last result is reused.

            frames_drawn[0] += 1

            if frames_drawn[0] % species_every_frames == 0:
                last_species_text[0] = (
                    molecule_tracker.summarise(simulation.atoms)
                )

            species_line = (
                "\nMolecules    " + last_species_text[0]
            )

        readout_text.set_text(
            f"Time         {simulation.elapsed_picoseconds:8.2f} ps\n"
            f"Temperature  {simulation.temperature_kelvin:8.1f} K\n"
            f"Target       {simulation.target_temperature_kelvin:8.1f} K\n"
            f"Thermostat   {thermostat_label:>8}\n"
            f"Kinetic      {simulation.kinetic_energy:8.2f}\n"
            f"Potential    {simulation.potential_energy:8.2f}\n"
            f"Total        {simulation.total_energy:8.2f}"
            + species_line
        )

        return particle_markers, measured_line, target_line

    animation = FuncAnimation(
        figure,
        update_frame,
        interval=frame_interval_milliseconds,
        blit=False,
        cache_frame_data=False
    )

    figure._chemistry_model_animation = animation

    plt.show()

    return simulation