import sys
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph.opengl as gl

from atom_torch import TorchAtomSimulation

UNIT_CELLS_PER_SIDE = 8
NUMBER_DENSITY = 0.85
TEMPERATURE = 1.0
TIME_STEP = 0.0002
FRICTION = 1.0
CUTOFF_DISTANCE = 2.5
SKIN_DISTANCE = 0.6
STEPS_PER_FRAME = 1


class Viewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.simulation = TorchAtomSimulation(
            unit_cells_per_side=UNIT_CELLS_PER_SIDE,
            number_density=NUMBER_DENSITY,
            target_temperature=TEMPERATURE,
            time_step=TIME_STEP,
            friction=FRICTION,
            cutoff_distance=CUTOFF_DISTANCE,
            skin_distance=SKIN_DISTANCE,
        )
        self.box_size = float(self.simulation.box_size)
        self.steps_per_frame = STEPS_PER_FRAME
        self.paused = False
        self.auto_orbit = False
        self.frame_index = 0
        self.frame_times = []
        self.time_history = []
        self.energy_history = []
        self.last_time = time.perf_counter()

        self.setWindowTitle("Lennard-Jones atoms")
        self.resize(1340, 860)
        layout = QtWidgets.QHBoxLayout(self)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=self.box_size * 2.0)
        layout.addWidget(self.view, stretch=4)

        self.add_box_outline()
        self.scatter = gl.GLScatterPlotItem(
            pos=self.simulation.positions_numpy,
            color=self.atom_colours(),
            size=0.34,
            pxMode=False,
        )
        self.scatter.setGLOptions("translucent")
        self.view.addItem(self.scatter)
        self.centre_camera(np.full(3, self.box_size / 2.0))
        layout.addLayout(self.build_panel(), stretch=1)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(0)

    def button(self, label, callback):
        button = QtWidgets.QPushButton(label)
        button.clicked.connect(callback)
        return button

    def divider(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("color: #555;")
        return line

    def build_panel(self):
        panel = QtWidgets.QVBoxLayout()
        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("font-family: monospace; font-size: 12px;")
        self.status.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        panel.addWidget(self.status)

        self.energy_plot = pg.PlotWidget()
        self.energy_plot.setLabel("left", "energy / atom")
        self.energy_plot.setLabel("bottom", "time")
        self.energy_curve = self.energy_plot.plot(pen="#4a90c4")
        self.energy_plot.setMaximumHeight(180)
        panel.addWidget(self.energy_plot)
        panel.addWidget(self.divider())

        self.pause_button = self.button("Pause", self.toggle_pause)
        panel.addWidget(self.pause_button)
        panel.addWidget(self.button("Reset", self.reset_simulation))
        panel.addWidget(self.button("Thermostat on/off", self.toggle_thermostat))
        panel.addWidget(self.button("Rebuild neighbours", self.rebuild_neighbours))
        panel.addWidget(self.divider())
        panel.addWidget(self.button("Auto-orbit", self.toggle_orbit))
        panel.addWidget(self.button("Recentre on box", self.recentre))
        panel.addWidget(self.divider())

        panel.addWidget(QtWidgets.QLabel("temperature"))
        self.temperature_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.temperature_slider.setRange(60, 200)
        self.temperature_slider.setValue(int(self.simulation.target_temperature_kelvin))
        self.temperature_slider.valueChanged.connect(self.on_temperature)
        panel.addWidget(self.temperature_slider)

        panel.addWidget(QtWidgets.QLabel("steps per frame"))
        self.steps_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.steps_slider.setRange(1, 400)
        self.steps_slider.setValue(STEPS_PER_FRAME)
        self.steps_slider.valueChanged.connect(self.on_steps)
        panel.addWidget(self.steps_slider)
        panel.addStretch(1)
        return panel

    def add_box_outline(self):
        s = self.box_size
        corners = np.array([
            [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
            [0, 0, s], [s, 0, s], [s, s, s], [0, s, s]
        ], dtype=np.float32)
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for a, b in edges:
            self.view.addItem(gl.GLLinePlotItem(
                pos=np.array([corners[a], corners[b]]),
                color=(0.45, 0.45, 0.45, 0.5), width=1.0, antialias=True
            ))

    def atom_colours(self):
        return np.tile(np.array([[0.30, 0.65, 0.95, 0.90]], dtype=np.float32),
                       (self.simulation.particle_count, 1))

    def refresh_appearance(self):
        self.scatter.setData(
            pos=self.simulation.positions_numpy % self.box_size,
            color=self.atom_colours(),
            size=0.34,
            pxMode=False,
        )

    def centre_camera(self, point, distance=None):
        self.view.opts["center"] = pg.Vector(float(point[0]), float(point[1]), float(point[2]))
        if distance is not None:
            self.view.setCameraPosition(distance=float(distance))
        self.view.update()

    def recentre(self):
        self.centre_camera(np.full(3, self.box_size / 2.0), self.box_size * 2.0)

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.setText("Resume" if self.paused else "Pause")

    def toggle_thermostat(self):
        self.simulation.set_thermostat(not self.simulation.thermostat_is_on)

    def rebuild_neighbours(self):
        self.simulation.force_rebuild_neighbours()

    def toggle_orbit(self):
        self.auto_orbit = not self.auto_orbit

    def on_temperature(self, value):
        self.simulation.target_temperature_kelvin = value

    def on_steps(self, value):
        self.steps_per_frame = int(value)

    def reset_simulation(self):
        self.simulation.reset()
        self.box_size = float(self.simulation.box_size)
        self.time_history.clear()
        self.energy_history.clear()
        self.frame_times.clear()
        self.frame_index = 0
        self.last_time = time.perf_counter()
        self.refresh_appearance()
        self.recentre()

    def update_frame(self):
        if not self.paused:
            self.simulation.step(self.steps_per_frame)

        self.frame_index += 1
        if self.auto_orbit:
            self.view.opts["azimuth"] = (self.view.opts["azimuth"] + 0.35) % 360.0

        self.refresh_appearance()

        now = time.perf_counter()
        elapsed = max(now - self.last_time, 1e-9)
        self.last_time = now
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        fps = 1.0 / np.mean(self.frame_times)

        self.time_history.append(self.simulation.elapsed_picoseconds)
        self.energy_history.append(self.simulation.total_energy / max(self.simulation.particle_count, 1))
        if len(self.time_history) > 500:
            self.time_history.pop(0)
            self.energy_history.pop(0)
        if self.frame_index % 5 == 0:
            self.energy_curve.setData(self.time_history, self.energy_history)

        self.status.setText(
            f"backend       {self.simulation.device}\n"
            f"atoms         {self.simulation.particle_count:9d}\n"
            f"time          {self.simulation.elapsed_picoseconds:9.3f} ps\n"
            f"temperature   {self.simulation.temperature_kelvin:9.2f} K\n"
            f"target        {self.simulation.target_temperature_kelvin:9.2f} K\n"
            f"kinetic       {self.simulation.kinetic_energy:9.3f}\n"
            f"potential     {self.simulation.potential_energy:9.3f}\n"
            f"total         {self.simulation.total_energy:9.3f}\n"
            f"neighbour pairs {self.simulation.neighbour_pair_count:7d}\n"
            f"list rebuilds {self.simulation.neighbour_rebuild_count:9d}\n"
            f"steps/frame   {self.steps_per_frame:9d}\n"
            f"fps           {fps:9.1f}"
        )


def main():
    pg.setConfigOptions(antialias=True)
    application = QtWidgets.QApplication(sys.argv)
    viewer = Viewer()
    viewer.show()
    sys.exit(application.exec())


if __name__ == "__main__":
    main()
