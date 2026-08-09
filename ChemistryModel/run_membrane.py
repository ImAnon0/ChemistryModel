import sys

import numpy as np

import pyqtgraph as pg

from pyqtgraph.Qt import QtCore, QtWidgets

import pyqtgraph.opengl as gl

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from membrane import BEADS_PER_LIPID

import structures


# ============================================================
# Settings
# ============================================================

NUMBER_OF_LIPIDS = 3000
BOX_SIZE = 40

TEMPERATURE = 1.1

STEPS_PER_FRAME = 40

START = "vesicle"

# "torch" for the GPU force loop, "numpy" for the CPU one.
# Puncture and detergent are only implemented on numpy.

BACKEND = "torch"

CLUSTER_REPORT_EVERY = 20


HEAD_COLOUR = np.array([0.91, 0.25, 0.16, 1.0], dtype=np.float32)
TAIL_COLOUR = np.array([0.29, 0.56, 0.77, 1.0], dtype=np.float32)

DIMMED_ALPHA = 0.06


def make_simulation(structure=START, lipids=NUMBER_OF_LIPIDS,
                    box=BOX_SIZE):
    if BACKEND == "torch":
        from membrane_torch import TorchMembraneSimulation

        return TorchMembraneSimulation(
            number_of_lipids=lipids,
            box_size=box,
            target_temperature=TEMPERATURE,
            start=structure
        )

    from membrane_sim import MembraneSimulation

    return MembraneSimulation(
        number_of_lipids=lipids,
        box_size=box,
        target_temperature=TEMPERATURE,
        start=structure
    )


def positions_of(simulation):
    if hasattr(simulation, "positions_numpy"):
        return simulation.positions_numpy

    return simulation.positions


def types_of(simulation):
    types = simulation.model.bead_types

    if hasattr(types, "detach"):
        types = types.detach().cpu().numpy()

    return types


def lipid_cluster_labels(positions, box_size, contact=1.6):
    tails = positions[1::BEADS_PER_LIPID] % box_size

    tree = cKDTree(tails, boxsize=box_size)

    pairs = tree.query_pairs(r=contact, output_type="ndarray")

    count = len(tails)

    if len(pairs) == 0:
        return np.arange(count), count

    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
        shape=(count, count)
    )

    number_of_clusters, labels = connected_components(
        graph,
        directed=False
    )

    return labels, number_of_clusters


class PickableView(gl.GLViewWidget):
    # Double-click selects a structure; ordinary drag still
    # orbits the camera as usual.

    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def mouseDoubleClickEvent(self, event):
        point = event.position()
        self.owner.select_at(point.x(), point.y())


class Viewer(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.simulation = make_simulation()
        self.box_size = float(BOX_SIZE)
        self.steps_per_frame = STEPS_PER_FRAME
        self.frame_index = 0
        self.paused = False
        self.isolating = False
        self.auto_orbit = False
        self.selected_label = None
        self.cluster_labels = None
        self.cluster_text = (
            "clusters          ...\nlargest           ..."
        )

        self.setWindowTitle("Self-assembling membrane")
        self.resize(1340, 860)

        layout = QtWidgets.QHBoxLayout(self)

        self.view = PickableView(self)
        self.view.setCameraPosition(distance=self.box_size * 2.0)

        layout.addWidget(self.view, stretch=4)

        self.box_lines = []
        self.add_box_outline()

        self.scatter = gl.GLScatterPlotItem(
            pos=positions_of(self.simulation),
            color=self.current_colours(),
            size=self.current_sizes(),
            pxMode=False
        )

        self.scatter.setGLOptions("translucent")
        self.view.addItem(self.scatter)

        self.centre_camera(np.full(3, self.box_size / 2.0))

        layout.addLayout(self.build_panel(), stretch=1)

        self.time_history = []
        self.energy_history = []

        self.frame_times = []
        self.last_time = QtCore.QTime.currentTime()

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
        self.status.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop
        )

        panel.addWidget(self.status)

        self.energy_plot = pg.PlotWidget()
        self.energy_plot.setLabel("left", "energy / lipid")
        self.energy_plot.setLabel("bottom", "time")
        self.energy_curve = self.energy_plot.plot(pen="#4a90c4")
        self.energy_plot.setMaximumHeight(180)

        panel.addWidget(self.energy_plot)

        panel.addWidget(QtWidgets.QLabel("starting structure"))

        self.structure_box = QtWidgets.QComboBox()
        self.structure_box.addItems(sorted(structures.STRUCTURES))
        self.structure_box.setCurrentText(START)

        panel.addWidget(self.structure_box)
        panel.addWidget(
            self.button("Load structure", self.on_load_structure)
        )

        panel.addWidget(self.divider())

        self.pause_button = self.button("Pause", self.toggle_pause)
        panel.addWidget(self.pause_button)

        panel.addWidget(self.button("Puncture", self.on_puncture))
        panel.addWidget(
            self.button("Detergent", self.on_detergent)
        )
        panel.addWidget(
            self.button("Thermostat on/off", self.on_thermostat)
        )

        panel.addWidget(self.divider())

        self.isolate_button = self.button(
            "Isolate selection",
            self.toggle_isolate
        )
        panel.addWidget(self.isolate_button)

        self.orbit_button = self.button(
            "Auto-orbit",
            self.toggle_orbit
        )
        panel.addWidget(self.orbit_button)

        panel.addWidget(
            self.button("Clear selection", self.clear_selection)
        )
        panel.addWidget(
            self.button("Recentre on box", self.on_recentre)
        )

        hint = QtWidgets.QLabel(
            "double-click a bead to select its whole\n"
            "structure and centre the camera on it"
        )
        hint.setStyleSheet("color: #888; font-size: 11px;")
        panel.addWidget(hint)

        panel.addWidget(self.divider())

        panel.addWidget(QtWidgets.QLabel("temperature"))

        self.temperature_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.temperature_slider.setRange(60, 200)
        self.temperature_slider.setValue(int(TEMPERATURE * 100))
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

    def centre_camera(self, point, distance=None):
        self.view.opts["center"] = pg.Vector(
            float(point[0]),
            float(point[1]),
            float(point[2])
        )

        if distance is not None:
            self.view.setCameraPosition(distance=float(distance))

        self.view.update()

    # --------------------------------------------------------

    def current_colours(self):
        types = types_of(self.simulation)

        colours = np.empty((len(types), 4), dtype=np.float32)
        colours[types == 0] = HEAD_COLOUR
        colours[types == 1] = TAIL_COLOUR

        if (
            self.isolating
            and self.selected_label is not None
            and self.cluster_labels is not None
        ):
            mask = np.repeat(
                self.cluster_labels == self.selected_label,
                BEADS_PER_LIPID
            )

            if len(mask) == len(colours):
                colours[~mask, 3] = DIMMED_ALPHA

        return colours

    def current_sizes(self):
        types = types_of(self.simulation)
        return np.where(types == 0, 0.62, 0.46).astype(np.float32)

    def refresh_appearance(self):
        self.scatter.setData(
            pos=positions_of(self.simulation) % self.box_size,
            color=self.current_colours(),
            size=self.current_sizes()
        )

    # --------------------------------------------------------

    def pick(self, screen_x, screen_y):
        positions = positions_of(self.simulation) % self.box_size

        projection = np.array(
            self.view.projectionMatrix().data()
        ).reshape(4, 4).T

        modelview = np.array(
            self.view.viewMatrix().data()
        ).reshape(4, 4).T

        transform = projection @ modelview

        homogeneous = np.column_stack([
            positions,
            np.ones(len(positions))
        ])

        clip = homogeneous @ transform.T

        in_front = clip[:, 3] > 1e-6

        if not np.any(in_front):
            return None

        with np.errstate(invalid="ignore", divide="ignore"):
            normalised = clip[:, :3] / clip[:, 3][:, None]

        width = self.view.width()
        height = self.view.height()

        pixel_x = (normalised[:, 0] + 1.0) * 0.5 * width
        pixel_y = (1.0 - normalised[:, 1]) * 0.5 * height

        distance = np.hypot(pixel_x - screen_x, pixel_y - screen_y)
        distance[~in_front] = np.inf

        nearest = int(np.argmin(distance))

        if not np.isfinite(distance[nearest]):
            return None

        if distance[nearest] > 25.0:
            return None

        return nearest

    def select_at(self, screen_x, screen_y):
        bead = self.pick(screen_x, screen_y)

        if bead is None:
            print("nothing under the cursor")
            return

        positions = positions_of(self.simulation)

        labels, _ = lipid_cluster_labels(positions, self.box_size)

        self.cluster_labels = labels

        lipid = bead // BEADS_PER_LIPID

        self.selected_label = int(labels[lipid])

        member_beads = np.repeat(
            labels == self.selected_label,
            BEADS_PER_LIPID
        )

        chosen = positions[member_beads] % self.box_size

        # Unwrap around the first bead, so a structure sitting
        # across the periodic boundary does not average out to
        # somewhere in the middle of the box.

        anchor = chosen[0]

        offsets = chosen - anchor
        offsets -= self.box_size * np.round(
            offsets / self.box_size
        )

        centre = anchor + offsets.mean(axis=0)

        extent = float(np.linalg.norm(offsets, axis=1).max())

        self.centre_camera(centre, distance=max(extent * 3.2, 8.0))

        size = int(
            np.count_nonzero(labels == self.selected_label)
        )

        print(f"selected structure: {size} lipids")

        self.refresh_appearance()

    def clear_selection(self):
        self.selected_label = None
        self.isolating = False
        self.isolate_button.setText("Isolate selection")
        self.refresh_appearance()

    def toggle_isolate(self):
        if self.selected_label is None:
            print("nothing selected - double-click a bead first")
            return

        self.isolating = not self.isolating

        self.isolate_button.setText(
            "Show all" if self.isolating else "Isolate selection"
        )

        self.refresh_appearance()

    def toggle_orbit(self):
        self.auto_orbit = not self.auto_orbit

        self.orbit_button.setText(
            "Stop orbit" if self.auto_orbit else "Auto-orbit"
        )

    def on_recentre(self):
        self.centre_camera(
            np.full(3, self.box_size / 2.0),
            distance=self.box_size * 2.0
        )

    # --------------------------------------------------------

    def toggle_pause(self):
        self.paused = not self.paused

        self.pause_button.setText(
            "Resume" if self.paused else "Pause"
        )

    def on_load_structure(self):
        name = self.structure_box.currentText()

        self.simulation = make_simulation(structure=name)

        self.time_history.clear()
        self.energy_history.clear()

        self.frame_index = 0
        self.cluster_labels = None

        self.clear_selection()
        self.refresh_appearance()
        self.on_recentre()

        print(f"loaded: {name}")

    def on_puncture(self):
        if not hasattr(self.simulation, "puncture"):
            print(
                "puncture is only implemented on the numpy "
                "backend - set BACKEND = 'numpy'"
            )
            return

        removed = self.simulation.puncture(radius=4.0)

        print(f"punctured: {removed} lipids removed")

        self.cluster_labels = None
        self.clear_selection()

    def on_detergent(self):
        if not hasattr(self.simulation, "add_detergent"):
            print(
                "detergent is only implemented on the numpy "
                "backend - set BACKEND = 'numpy'"
            )
            return

        self.simulation.add_detergent(fraction=0.3)
        self.refresh_appearance()

    def on_thermostat(self):
        self.simulation.thermostat_is_on = (
            not self.simulation.thermostat_is_on
        )

    def on_temperature(self, value):
        self.simulation.target_temperature_kelvin = value / 100.0

    def on_steps(self, value):
        self.steps_per_frame = int(value)

    # --------------------------------------------------------

    def update_frame(self):
        if not self.paused:
            self.simulation.step(self.steps_per_frame)

        self.frame_index += 1

        if self.auto_orbit:
            self.view.opts["azimuth"] = (
                self.view.opts["azimuth"] + 0.35
            ) % 360.0

        self.scatter.setData(
            pos=positions_of(self.simulation) % self.box_size
        )

        now = QtCore.QTime.currentTime()
        elapsed = max(self.last_time.msecsTo(now), 1)
        self.last_time = now

        self.frame_times.append(elapsed)

        if len(self.frame_times) > 30:
            self.frame_times.pop(0)

        frames_per_second = 1000.0 / float(
            np.mean(self.frame_times)
        )

        lipid_count = self.simulation.model.number_of_lipids

        energy_per_lipid = (
            self.simulation.potential_energy
            / max(lipid_count, 1)
        )

        self.time_history.append(
            self.simulation.elapsed_picoseconds
        )
        self.energy_history.append(energy_per_lipid)

        if self.frame_index % 5 == 0:
            self.energy_curve.setData(
                self.time_history,
                self.energy_history
            )

        if self.frame_index % CLUSTER_REPORT_EVERY == 0:
            labels, cluster_count = lipid_cluster_labels(
                positions_of(self.simulation),
                self.box_size
            )

            self.cluster_labels = labels

            largest = int(np.max(np.bincount(labels)))

            self.cluster_text = (
                f"clusters    {cluster_count:9d}\n"
                f"largest     {largest:9d}"
            )

            if self.isolating:
                self.refresh_appearance()

        if (
            self.selected_label is None
            or self.cluster_labels is None
        ):
            selection_text = "none"
        else:
            selection_text = str(
                int(
                    np.count_nonzero(
                        self.cluster_labels == self.selected_label
                    )
                )
            ) + " lipids"

        self.status.setText(
            f"backend     {BACKEND:>9}\n"
            f"lipids      {lipid_count:9d}\n"
            f"time        "
            f"{self.simulation.elapsed_picoseconds:9.1f}\n"
            f"temperature "
            f"{self.simulation.temperature_kelvin:9.3f}\n"
            f"energy      {energy_per_lipid:9.3f}\n"
            f"{self.cluster_text}\n"
            f"selected    {selection_text:>9}\n"
            f"steps/frame {self.steps_per_frame:9d}\n"
            f"fps         {frames_per_second:9.1f}"
        )


def main():
    pg.setConfigOptions(antialias=True)

    application = QtWidgets.QApplication(sys.argv)

    viewer = Viewer()
    viewer.show()

    sys.exit(application.exec())


if __name__ == "__main__":
    main()