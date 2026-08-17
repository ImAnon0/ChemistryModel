import sys
import time

import numpy as np

import pyqtgraph as pg

from pyqtgraph.Qt import QtCore, QtWidgets

import pyqtgraph.opengl as gl


# ============================================================
# Settings
# ============================================================

UNIT_CELLS_PER_SIDE = 8
NUMBER_DENSITY = 0.85
TEMPERATURE = 1.0

# Reduced Lennard-Jones units. At 0.002 the energy drift with the
# thermostat off is about 4e-6 over 500 steps, which is plenty.
# Smaller values buy no accuracy worth having and cost speed
# proportionally.

TIME_STEP = 0.002

FRICTION = 1.0
CUTOFF_DISTANCE = 2.5
SKIN_DISTANCE = 0.6
STEPS_PER_FRAME = 20

# "torch" for the GPU force loop, "numpy" for the CPU reference.

BACKEND = "torch"

RDF_EVERY = 20


# Reduced density and temperature for each phase.

PHASES = {
    "solid": (1.05, 0.40),
    "dense liquid": (0.95, 0.80),
    "near triple point": (0.85, 0.70),
    "liquid": (0.85, 1.10),
    "supercritical": (0.30, 1.60),
    "gas": (0.05, 1.50)
}

COLOUR_MODES = ["uniform", "speed", "energy", "coordination"]


def make_simulation(density=NUMBER_DENSITY, temperature=TEMPERATURE):
    if BACKEND == "torch":
        from atom_torch import TorchAtomSimulation

        return TorchAtomSimulation(
            unit_cells_per_side=UNIT_CELLS_PER_SIDE,
            number_density=density,
            target_temperature=temperature,
            time_step=TIME_STEP,
            friction=FRICTION,
            cutoff_distance=CUTOFF_DISTANCE,
            skin_distance=SKIN_DISTANCE,
        )

    from atom_numpy import AtomSimulation

    return AtomSimulation(
        unit_cells_per_side=UNIT_CELLS_PER_SIDE,
        number_density=density,
        target_temperature=temperature,
        time_step=TIME_STEP,
        friction=FRICTION,
        cutoff_distance=CUTOFF_DISTANCE,
        skin_distance=SKIN_DISTANCE,
    )


def to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def colour_ramp(values, low, high):
    span = max(high - low, 1e-9)

    fraction = np.clip((values - low) / span, 0.0, 1.0)

    cold = np.array([0.25, 0.45, 0.90])
    middle = np.array([0.92, 0.92, 0.92])
    hot = np.array([0.90, 0.25, 0.15])

    lower = fraction < 0.5

    weight = np.where(lower, fraction * 2.0, (fraction - 0.5) * 2.0)

    start = np.where(lower[:, None], cold, middle)
    end = np.where(lower[:, None], middle, hot)

    colours = np.empty((len(values), 4), dtype=np.float32)

    colours[:, :3] = start + (end - start) * weight[:, None]
    colours[:, 3] = 0.92

    return colours


class Viewer(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.simulation = make_simulation()
        self.box_size = float(self.simulation.box_size)

        self.steps_per_frame = STEPS_PER_FRAME
        self.paused = False
        self.auto_orbit = False
        self.frame_index = 0
        self.colour_mode = 0
        self.slice_mode = 0
        self.slice_axis = 0
        self.slice_position = 50

        self.frame_times = []
        self.time_history = []
        self.energy_history = []
        self.last_time = time.perf_counter()

        self.setWindowTitle("Lennard-Jones atoms")
        self.resize(1400, 880)

        layout = QtWidgets.QHBoxLayout(self)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=self.box_size * 2.0)

        layout.addWidget(self.view, stretch=4)

        self.box_lines = []
        self.add_box_outline()

        self.scatter = gl.GLScatterPlotItem(
            pos=self.positions(),
            color=self.atom_colours(),
            size=0.34,
            pxMode=False,
        )

        self.scatter.setGLOptions("translucent")
        self.view.addItem(self.scatter)

        self.recentre()

        layout.addLayout(self.build_panel(), stretch=1)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(0)

    # --------------------------------------------------------

    def button(self, label, callback):
        widget = QtWidgets.QPushButton(label)
        widget.clicked.connect(callback)
        return widget

    def divider(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("color: #555;")
        return line

    def build_panel(self):
        panel = QtWidgets.QVBoxLayout()

        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet(
            "font-family: monospace; font-size: 12px;"
        )
        self.status.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        panel.addWidget(self.status)

        self.rdf_plot = pg.PlotWidget()
        self.rdf_plot.setLabel("left", "g(r)")
        self.rdf_plot.setLabel("bottom", "r")
        self.rdf_plot.setMaximumHeight(165)
        self.rdf_curve = self.rdf_plot.plot(pen="#e04a2a")
        self.rdf_plot.addLine(
            y=1.0,
            pen=pg.mkPen("#777", style=QtCore.Qt.PenStyle.DashLine)
        )

        panel.addWidget(self.rdf_plot)

        self.energy_plot = pg.PlotWidget()
        self.energy_plot.setLabel("left", "energy / atom")
        self.energy_plot.setLabel("bottom", "time")
        self.energy_plot.setMaximumHeight(140)
        self.energy_curve = self.energy_plot.plot(pen="#4a90c4")

        panel.addWidget(self.energy_plot)

        panel.addWidget(QtWidgets.QLabel("phase"))

        self.phase_box = QtWidgets.QComboBox()
        self.phase_box.addItems(list(PHASES))
        self.phase_box.setCurrentText("liquid")

        panel.addWidget(self.phase_box)
        panel.addWidget(
            self.button("Load phase", self.on_load_phase)
        )

        panel.addWidget(self.divider())

        self.pause_button = self.button("Pause", self.toggle_pause)
        panel.addWidget(self.pause_button)

        panel.addWidget(self.button("Melt now", self.on_melt))
        panel.addWidget(self.button("Quench now", self.on_quench))
        panel.addWidget(
            self.button("Thermostat on/off", self.toggle_thermostat)
        )
        panel.addWidget(
            self.button("Reset diffusion clock", self.on_reset_msd)
        )

        panel.addWidget(self.divider())

        self.colour_button = self.button(
            "Colour: uniform",
            self.cycle_colour
        )
        panel.addWidget(self.colour_button)

        self.slice_button = self.button("Cut: off", self.cycle_slice)
        panel.addWidget(self.slice_button)

        self.axis_button = self.button(
            "Cut axis: X",
            self.cycle_axis
        )
        panel.addWidget(self.axis_button)

        self.slice_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.slice_slider.setRange(0, 100)
        self.slice_slider.setValue(50)
        self.slice_slider.valueChanged.connect(self.on_slice)

        panel.addWidget(self.slice_slider)

        panel.addWidget(self.button("Auto-orbit", self.toggle_orbit))
        panel.addWidget(self.button("Recentre", self.recentre))

        panel.addWidget(self.divider())

        panel.addWidget(QtWidgets.QLabel("temperature (K)"))

        self.temperature_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.temperature_slider.setRange(20, 300)
        self.temperature_slider.setValue(
            int(self.simulation.target_temperature_kelvin)
        )
        self.temperature_slider.valueChanged.connect(
            self.on_temperature
        )

        panel.addWidget(self.temperature_slider)

        panel.addWidget(QtWidgets.QLabel("steps per frame"))

        self.steps_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.steps_slider.setRange(1, 400)
        self.steps_slider.setValue(STEPS_PER_FRAME)
        self.steps_slider.valueChanged.connect(self.on_steps)

        panel.addWidget(self.steps_slider)

        panel.addStretch(1)

        return panel

    # --------------------------------------------------------

    def add_box_outline(self):
        for item in self.box_lines:
            self.view.removeItem(item)

        self.box_lines = []

        size = self.box_size

        corners = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=np.float32) * size

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        for first, second in edges:
            line = gl.GLLinePlotItem(
                pos=np.array([corners[first], corners[second]]),
                color=(0.45, 0.45, 0.45, 0.5),
                width=1.0,
                antialias=True
            )

            self.view.addItem(line)
            self.box_lines.append(line)

    def recentre(self):
        half = self.box_size / 2.0

        self.view.opts["center"] = pg.Vector(half, half, half)
        self.view.setCameraPosition(distance=self.box_size * 2.0)
        self.view.update()

    # --------------------------------------------------------

    def positions(self):
        if hasattr(self.simulation, "positions_numpy"):
            return self.simulation.positions_numpy

        return self.simulation.positions

    def atom_colours(self):
        count = self.simulation.particle_count

        if self.colour_mode == 0:
            return np.tile(
                np.array(
                    [[0.30, 0.65, 0.95, 0.90]],
                    dtype=np.float32
                ),
                (count, 1)
            )

        if self.colour_mode == 1:
            velocities = to_numpy(self.simulation.velocities)

            values = np.linalg.norm(velocities, axis=1)

            typical = np.sqrt(
                max(3.0 * self.simulation.temperature, 1e-9)
            )

            return colour_ramp(values, 0.0, 2.0 * typical)

        if self.colour_mode == 2:
            values = to_numpy(self.simulation.per_atom_energy)

            return colour_ramp(
                values,
                float(np.percentile(values, 2)),
                float(np.percentile(values, 98))
            )

        values = to_numpy(self.simulation.coordination)

        return colour_ramp(values, 0.0, 14.0)

    def visible_mask(self, positions):
        if self.slice_mode == 0:
            return None

        coordinate = positions[:, self.slice_axis]

        cut = self.slice_position / 100.0 * self.box_size

        if self.slice_mode == 1:
            return coordinate <= cut

        return np.abs(coordinate - cut) <= self.box_size * 0.08

    def refresh_appearance(self):
        positions = self.positions() % self.box_size

        colours = self.atom_colours()

        mask = self.visible_mask(positions)

        if mask is not None:
            positions = positions[mask]
            colours = colours[mask]

        self.scatter.setData(
            pos=positions,
            color=colours,
            size=0.34,
            pxMode=False
        )

    # --------------------------------------------------------

    def cycle_colour(self):
        self.colour_mode = (
            self.colour_mode + 1
        ) % len(COLOUR_MODES)

        self.colour_button.setText(
            "Colour: " + COLOUR_MODES[self.colour_mode]
        )

    def cycle_slice(self):
        self.slice_mode = (self.slice_mode + 1) % 3

        self.slice_button.setText(
            ["Cut: off", "Cut: half", "Cut: slab"][self.slice_mode]
        )

    def cycle_axis(self):
        self.slice_axis = (self.slice_axis + 1) % 3

        self.axis_button.setText(
            "Cut axis: " + "XYZ"[self.slice_axis]
        )

    def on_slice(self, value):
        self.slice_position = int(value)

    def toggle_pause(self):
        self.paused = not self.paused

        self.pause_button.setText(
            "Resume" if self.paused else "Pause"
        )

    def toggle_thermostat(self):
        self.simulation.thermostat_is_on = (
            not self.simulation.thermostat_is_on
        )

    def toggle_orbit(self):
        self.auto_orbit = not self.auto_orbit

    def on_temperature(self, value):
        self.simulation.target_temperature_kelvin = value

    def on_steps(self, value):
        self.steps_per_frame = int(value)

    def on_reset_msd(self):
        self.simulation.reset_msd()

    def sync_temperature_slider(self):
        kelvin = self.simulation.target_temperature_kelvin

        self.temperature_slider.blockSignals(True)
        self.temperature_slider.setValue(
            int(min(max(kelvin, 20), 300))
        )
        self.temperature_slider.blockSignals(False)

    def on_melt(self):
        target = max(self.simulation.temperature * 2.5, 0.1)

        self.simulation.scale_velocities_to(target)
        self.simulation.target_temperature = target

        self.simulation.reset_msd()
        self.sync_temperature_slider()

    def on_quench(self):
        target = max(self.simulation.temperature * 0.35, 0.05)

        self.simulation.scale_velocities_to(target)
        self.simulation.target_temperature = target

        self.simulation.reset_msd()
        self.sync_temperature_slider()

    def on_load_phase(self):
        name = self.phase_box.currentText()

        density, temperature = PHASES[name]

        self.simulation = make_simulation(density, temperature)

        self.box_size = float(self.simulation.box_size)

        self.add_box_outline()
        self.recentre()

        self.time_history.clear()
        self.energy_history.clear()
        self.frame_times.clear()

        self.frame_index = 0

        self.sync_temperature_slider()

        print(
            f"loaded {name}: density {density}, T* {temperature}, "
            f"{self.simulation.particle_count} atoms"
        )

    # --------------------------------------------------------

    def update_frame(self):
        if not self.paused:
            self.simulation.step(self.steps_per_frame)

        self.frame_index += 1

        if self.auto_orbit:
            self.view.opts["azimuth"] = (
                self.view.opts["azimuth"] + 0.35
            ) % 360.0

        self.refresh_appearance()

        now = time.perf_counter()
        elapsed = max(now - self.last_time, 1e-9)
        self.last_time = now

        self.frame_times.append(elapsed)

        if len(self.frame_times) > 30:
            self.frame_times.pop(0)

        frames_per_second = 1.0 / float(np.mean(self.frame_times))

        count = max(self.simulation.particle_count, 1)

        self.time_history.append(
            self.simulation.elapsed_picoseconds
        )
        self.energy_history.append(
            self.simulation.total_energy / count
        )

        if len(self.time_history) > 600:
            self.time_history.pop(0)
            self.energy_history.pop(0)

        if self.frame_index % 5 == 0:
            self.energy_curve.setData(
                self.time_history,
                self.energy_history
            )

        if self.frame_index % RDF_EVERY == 0:
            radius, g = self.simulation.radial_distribution()
            self.rdf_curve.setData(radius, g)

        diffusion = self.simulation.diffusion_coefficient

        if diffusion < 0.005:
            state = "solid"
        elif diffusion < 0.5:
            state = "liquid"
        else:
            state = "gas"

        device = str(getattr(self.simulation, "device", "numpy"))

        self.status.setText(
            f"backend      {device:>10}\n"
            f"atoms        {self.simulation.particle_count:10d}\n"
            f"density      {self.simulation.density:10.3f}\n"
            f"time         "
            f"{self.simulation.elapsed_picoseconds:10.3f} ps\n"
            f"temperature  "
            f"{self.simulation.temperature_kelvin:10.2f} K\n"
            f"  reduced    {self.simulation.temperature:10.3f}\n"
            f"pressure     {self.simulation.pressure:10.3f}\n"
            f"potential    "
            f"{self.simulation.potential_energy / count:10.3f}\n"
            f"total        "
            f"{self.simulation.total_energy / count:10.3f}\n"
            f"MSD          "
            f"{self.simulation.mean_squared_displacement:10.4f}\n"
            f"diffusion    {diffusion:10.5f}\n"
            f"looks like   {state:>10}\n"
            f"pairs        "
            f"{self.simulation.neighbour_pair_count:10d}\n"
            f"steps/frame  {self.steps_per_frame:10d}\n"
            f"fps          {frames_per_second:10.1f}"
        )


def main():
    pg.setConfigOptions(antialias=True)

    application = QtWidgets.QApplication(sys.argv)

    viewer = Viewer()
    viewer.show()

    sys.exit(application.exec())


if __name__ == "__main__":
    main()