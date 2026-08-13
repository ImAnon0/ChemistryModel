import sys
import time

import numpy as np

import pyqtgraph as pg

from pyqtgraph.Qt import QtCore, QtWidgets

import pyqtgraph.opengl as gl

import reactive as R
import build_box

from reactive_torch import ReactiveSimulation
from recorder import Recorder
from lab_renderer import AtomView, ELEMENT_COLOUR, ELEMENT_SIZE


# ============================================================
# Settings
# ============================================================

BOX_SIZE = 19.0

TEMPERATURE = 500.0

# Femtoseconds. Hydrogen vibrates fastest and sets the limit;
# 0.25 fs is safe, 0.5 fs is borderline.

TIME_STEP = 0.25

FRICTION = 0.01

STEPS_PER_FRAME = 4

BOND_REFRESH_EVERY = 3

# One captured frame every this many drawn frames, and how many
# frames to keep before the oldest start dropping off.

# Recording is measured in simulation steps, not drawn frames.
# That keeps playback smooth no matter how fast the simulation is
# being pushed: steps-per-frame controls speed, this controls
# how finely the run is sampled, and the two no longer fight.

RECORD_EVERY_STEPS = 8
MAXIMUM_FRAMES = 40000

# Playback is measured in recorded frames per real second.
#
# Counting drawn frames instead does not work: during playback
# there is no simulation to do, so the timer runs as fast as Qt
# can manage, which can be hundreds of ticks a second. Tying
# playback to the wall clock makes the speed mean the same thing
# whatever the machine is doing.

PLAYBACK_FPS = 12.0

# Lightning. A real discharge is a conducting channel of ionised
# gas, centimetres across and tens of thousands of kelvin, that
# heats everything it passes through at once. Kicking a single
# atom is a poor model of that: the energy mostly goes back into
# the bond it came from. A channel tears a whole column of
# molecules apart simultaneously, so the fragments are left in
# contact with each other and can recombine into something new.

CHANNEL_RADIUS = 2.2

# Channel temperature in kelvin. A real lightning channel runs at
# 20,000 to 30,000 K, so this is a measured quantity rather than
# a number picked to make something happen.

CHANNEL_TEMPERATURE = 25000.0

# Fraction of the bonds inside the channel broken outright. This
# stands in for electron-impact dissociation, which is how a real
# spark actually breaks things and which a model without
# electrons cannot produce by heating alone.

CHANNEL_DISSOCIATION = 0.6


from mixtures import STARTS


def make_simulation(name="loose H + O", box=BOX_SIZE,
                    temperature=TEMPERATURE, seed=0):
    kind, contents = STARTS[name]

    if kind == "atoms":
        symbols, positions = build_box.loose_atoms(
            contents, box, minimum_separation=1.25, random_seed=seed
        )
    else:
        symbols, positions = build_box.build(
            contents, box, random_seed=seed
        )

    return ReactiveSimulation(
        symbols=symbols,
        positions=positions,
        box_size=box,
        time_step=TIME_STEP,
        target_temperature=temperature,
        friction=FRICTION,
    )


class Viewer(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.start_name = "loose H + O"
        self.simulation = make_simulation(self.start_name)
        self.box_size = float(self.simulation.box_size)

        self.steps_per_frame = STEPS_PER_FRAME
        self.paused = False
        self.auto_orbit = False
        self.show_bonds = True
        self.frame_index = 0
        self.seed = 0

        self.recording = True
        self.replaying = False
        self.playing_back = False
        self.record_every_steps = RECORD_EVERY_STEPS
        self.channel_temperature = CHANNEL_TEMPERATURE
        self.channel_dissociation = CHANNEL_DISSOCIATION
        self.atom_scale = 1.0

        self.playback_fps = PLAYBACK_FPS
        self.last_advance = time.perf_counter()

        self.molecule_entries = []
        self.molecule_labels = None
        self.selected_molecule = None
        self.isolating = False

        self.recorder = Recorder(
            self.simulation.symbols,
            self.box_size,
            maximum_frames=MAXIMUM_FRAMES
        )

        self.frame_times = []
        self.time_history = []
        self.energy_history = []
        self.last_time = time.perf_counter()

        self.setWindowTitle("Reactive chemistry")
        self.resize(1720, 900)

        layout = QtWidgets.QHBoxLayout(self)

        self.view = AtomView()
        self.view.setCameraPosition(distance=self.box_size * 2.2)

        layout.addWidget(self.view, stretch=4)

        self.box_lines = []
        self.add_box_outline()

        self.bond_item = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)),
            color=(0.6, 0.6, 0.6, 0.85),
            width=3.0,
            mode="lines",
            antialias=True
        )

        self.view.addItem(self.bond_item)

        self.scatter = gl.GLScatterPlotItem(
            pos=self.simulation.positions_numpy,
            color=self.atom_colours(),
            size=self.atom_sizes(),
            pxMode=False
        )

        self.scatter.setGLOptions("opaque")
        self.view.addItem(self.scatter)

        self.recentre()

        left_panel, right_panel = self.build_panels()

        # The panels are built after the 3D view because they
        # reference it, but the left one belongs at the far edge
        # of the window, so it gets inserted in front rather than
        # appended.

        layout.insertLayout(0, left_panel, stretch=1)
        layout.addLayout(right_panel, stretch=1)

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

    def build_panels(self):
        # Two columns rather than one. Everything to do with
        # running the simulation goes on the left, everything to
        # do with looking at it goes on the right, and the energy
        # plot finally gets enough height to be readable.

        left = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()

        # ---- left: state and controls ----

        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet(
            "font-family: monospace; font-size: 12px;"
        )
        self.status.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        left.addWidget(self.status)

        left.addWidget(QtWidgets.QLabel("starting mixture"))

        self.start_box = QtWidgets.QComboBox()
        self.start_box.addItems(list(STARTS))

        left.addWidget(self.start_box)
        left.addWidget(self.button("Load mixture", self.on_load))
        left.addWidget(
            self.button("New random seed", self.on_reseed)
        )

        left.addWidget(self.divider())

        self.pause_button = self.button("Pause", self.toggle_pause)
        left.addWidget(self.pause_button)

        left.addWidget(self.button("Lightning strike", self.on_spark))

        self.channel_label = QtWidgets.QLabel(
            f"channel {CHANNEL_TEMPERATURE:.0f} K"
        )
        left.addWidget(self.channel_label)

        self.channel_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.channel_slider.setRange(5, 60)
        self.channel_slider.setValue(
            int(CHANNEL_TEMPERATURE / 1000)
        )
        self.channel_slider.valueChanged.connect(
            self.on_channel_temperature
        )
        left.addWidget(self.channel_slider)

        self.dissociation_label = QtWidgets.QLabel(
            f"bonds broken in channel  {CHANNEL_DISSOCIATION:.0%}"
        )
        left.addWidget(self.dissociation_label)

        self.dissociation_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.dissociation_slider.setRange(0, 100)
        self.dissociation_slider.setValue(
            int(CHANNEL_DISSOCIATION * 100)
        )
        self.dissociation_slider.valueChanged.connect(
            self.on_channel_dissociation
        )
        left.addWidget(self.dissociation_slider)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.button("Heat x1.5", self.on_heat))
        row.addWidget(self.button("Cool x0.6", self.on_cool))
        left.addLayout(row)

        left.addWidget(
            self.button("Thermostat on/off", self.toggle_thermostat)
        )

        left.addWidget(self.divider())

        self.record_button = self.button(
            "Recording: on", self.toggle_recording
        )
        left.addWidget(self.record_button)

        self.frame_label = QtWidgets.QLabel("live")
        self.frame_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; color: #999;"
        )
        left.addWidget(self.frame_label)

        self.scrubber = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.scrubber.setRange(0, 0)
        self.scrubber.valueChanged.connect(self.on_scrub)
        left.addWidget(self.scrubber)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.button("<<", lambda: self.nudge(-10)))
        row.addWidget(self.button("<", lambda: self.nudge(-1)))
        row.addWidget(self.button(">", lambda: self.nudge(1)))
        row.addWidget(self.button(">>", lambda: self.nudge(10)))
        left.addLayout(row)

        self.play_button = self.button(
            "Play recording", self.toggle_playback
        )
        left.addWidget(self.play_button)

        left.addWidget(
            self.button(
                "Continue from this frame",
                self.continue_from_here
            )
        )

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.button("Back to live", self.go_live))
        row.addWidget(
            self.button("Clear rec", self.clear_recording)
        )
        left.addLayout(row)

        self.playback_label = QtWidgets.QLabel(
            f"playback  {PLAYBACK_FPS:.1f} frames/s"
        )
        left.addWidget(self.playback_label)

        self.playback_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.playback_slider.setRange(1, 600)
        self.playback_slider.setValue(int(PLAYBACK_FPS * 10))
        self.playback_slider.valueChanged.connect(
            self.on_playback_speed
        )
        left.addWidget(self.playback_slider)

        self.record_label = QtWidgets.QLabel(
            f"capture every {RECORD_EVERY_STEPS} steps"
            f"  ({RECORD_EVERY_STEPS * TIME_STEP:.2f} fs)"
        )
        left.addWidget(self.record_label)

        self.record_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.record_slider.setRange(1, 100)
        self.record_slider.setValue(RECORD_EVERY_STEPS)
        self.record_slider.valueChanged.connect(
            self.on_record_interval
        )
        left.addWidget(self.record_slider)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.button("Save", self.on_save))
        row.addWidget(self.button("Load", self.on_load_recording))
        row.addWidget(self.button("XYZ", self.on_export))
        left.addLayout(row)

        left.addStretch(1)

        # ---- right: measurement and viewing ----

        self.energy_plot = pg.PlotWidget()
        self.energy_plot.setLabel("left", "potential energy (eV)")
        self.energy_plot.setLabel("bottom", "fs")
        self.energy_plot.setMinimumHeight(260)
        self.energy_curve = self.energy_plot.plot(pen="#4a90c4")

        right.addWidget(self.energy_plot)

        right.addWidget(QtWidgets.QLabel("molecules present"))

        self.inventory = QtWidgets.QLabel("")
        self.inventory.setStyleSheet(
            "font-family: monospace; font-size: 12px; color: #6ab04c;"
        )
        self.inventory.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop
        )
        self.inventory.setWordWrap(True)
        self.inventory.setMinimumHeight(90)

        right.addWidget(self.inventory)

        right.addWidget(self.divider())

        right.addWidget(QtWidgets.QLabel("inspect a molecule"))

        self.molecule_box = QtWidgets.QComboBox()
        self.molecule_box.currentIndexChanged.connect(
            self.on_select_molecule
        )
        right.addWidget(self.molecule_box)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.button("Find", self.find_molecules))
        row.addWidget(
            self.button("Clear", self.clear_molecule_selection)
        )
        right.addLayout(row)

        self.isolate_button = self.button(
            "Isolate: off", self.toggle_isolate
        )
        right.addWidget(self.isolate_button)

        right.addWidget(self.divider())

        row = QtWidgets.QHBoxLayout()
        self.bond_button = self.button("Bonds: on", self.toggle_bonds)
        row.addWidget(self.bond_button)
        row.addWidget(self.button("Auto-orbit", self.toggle_orbit))
        row.addWidget(self.button("Recentre", self.recentre))
        right.addLayout(row)

        right.addWidget(self.divider())

        self.temperature_label = QtWidgets.QLabel(
            f"target temperature  {int(TEMPERATURE)} K"
        )
        right.addWidget(self.temperature_label)

        self.temperature_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.temperature_slider.setRange(10, 4000)
        self.temperature_slider.setValue(int(TEMPERATURE))
        self.temperature_slider.valueChanged.connect(
            self.on_temperature
        )

        right.addWidget(self.temperature_slider)

        self.scale_label = QtWidgets.QLabel("atom size  1.0x")
        right.addWidget(self.scale_label)

        self.scale_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.scale_slider.setRange(3, 30)
        self.scale_slider.setValue(10)
        self.scale_slider.valueChanged.connect(self.on_atom_scale)
        right.addWidget(self.scale_slider)

        self.box_label = QtWidgets.QLabel(
            f"next run box  {BOX_SIZE:.1f} A"
        )
        right.addWidget(self.box_label)

        self.box_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.box_slider.setRange(60, 600)
        self.box_slider.setValue(int(BOX_SIZE * 10))
        self.box_slider.valueChanged.connect(self.on_box_size)
        right.addWidget(self.box_slider)

        self.steps_label = QtWidgets.QLabel(
            f"steps per frame  {STEPS_PER_FRAME}"
            f"  ({STEPS_PER_FRAME * TIME_STEP:.1f} fs)"
        )
        right.addWidget(self.steps_label)

        self.steps_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.steps_slider.setRange(1, 200)
        self.steps_slider.setValue(STEPS_PER_FRAME)
        self.steps_slider.valueChanged.connect(self.on_steps)

        right.addWidget(self.steps_slider)

        legend = QtWidgets.QLabel(
            "white H    dark C    blue N    red O"
        )
        legend.setStyleSheet("color: #999; font-size: 11px;")

        right.addWidget(legend)
        right.addStretch(1)

        return left, right

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
                color=(0.45, 0.45, 0.45, 0.45),
                width=1.0,
                antialias=True
            )

            self.view.addItem(line)
            self.box_lines.append(line)

    def recentre(self):
        half = self.box_size / 2.0

        self.view.opts["center"] = pg.Vector(half, half, half)
        self.view.setCameraPosition(distance=self.box_size * 2.2)
        self.view.update()

    # --------------------------------------------------------

    def atom_colours(self):
        symbols = (
            self.recorder.symbols
            if self.replaying and len(self.recorder) > 0
            else self.simulation.symbols
        )

        return np.array([
            ELEMENT_COLOUR[symbol] for symbol in symbols
        ], dtype=np.float32)

    def atom_sizes(self):
        symbols = (
            self.recorder.symbols
            if self.replaying and len(self.recorder) > 0
            else self.simulation.symbols
        )

        return np.array([
            ELEMENT_SIZE[symbol] * self.atom_scale
            for symbol in symbols
        ], dtype=np.float32)

    def update_bonds(self):
        if not self.show_bonds:
            self.bond_item.setData(pos=np.zeros((2, 3)))
            return

        first, second = self.simulation.bond_list()

        if len(first) == 0:
            self.bond_item.setData(pos=np.zeros((2, 3)))
            return

        positions = self.simulation.positions_numpy

        start = positions[first]
        end = positions[second]

        # A bond that wraps around the box would otherwise be
        # drawn as a line straight across the whole picture.

        offset = end - start
        offset -= self.box_size * np.round(offset / self.box_size)

        keep = np.linalg.norm(offset, axis=1) < self.box_size / 2.0

        start = start[keep]
        end = end[keep]

        if len(start) == 0:
            self.bond_item.setData(pos=np.zeros((2, 3)))
            return

        segments = np.empty((2 * len(start), 3), dtype=np.float32)

        segments[0::2] = start
        segments[1::2] = start + offset[keep]

        self.bond_item.setData(pos=segments, mode="lines")

    # --------------------------------------------------------

    def toggle_pause(self):
        self.paused = not self.paused

        self.pause_button.setText(
            "Resume" if self.paused else "Pause"
        )

    def toggle_thermostat(self):
        self.simulation.thermostat_is_on = (
            not self.simulation.thermostat_is_on
        )

    def toggle_bonds(self):
        self.show_bonds = not self.show_bonds

        self.bond_button.setText(
            "Bonds: on" if self.show_bonds else "Bonds: off"
        )

    def toggle_orbit(self):
        self.auto_orbit = not self.auto_orbit

    def on_temperature(self, value):
        self.simulation.target_temperature = float(value)

        self.temperature_label.setText(
            f"target temperature  {int(value)} K"
        )

    def on_atom_scale(self, value):
        # Purely cosmetic. Atoms are drawn in world units so they
        # keep their true size relative to the cell, which is
        # correct but leaves them small in a large box.

        self.atom_scale = int(value) / 10.0

        self.scale_label.setText(
            f"atom size  {self.atom_scale:.1f}x"
        )

        self.scatter.setData(
            pos=self.positions_for_drawing(),
            color=self.atom_colours(),
            size=self.atom_sizes(),
            pxMode=False,
        )

    def on_box_size(self, value):
        self.update_box_label()

    def update_box_label(self):
        count = self.simulation.atom_count

        density = count / (self.box_size ** 3)

        requested = self.box_slider.value() / 10.0

        self.box_label.setText(
            f"next run box  {requested:.1f} A"
            f"   current {self.box_size:.1f} A"
            f" / {density:.3f} atoms/A^3"
        )

    def on_steps(self, value):
        self.steps_per_frame = int(value)

        self.steps_label.setText(
            f"steps per frame  {self.steps_per_frame}"
            f"  ({self.steps_per_frame * TIME_STEP:.1f} fs)"
        )

    def on_channel_temperature(self, value):
        self.channel_temperature = float(value) * 1000.0

        self.channel_label.setText(
            f"channel {self.channel_temperature:.0f} K"
        )

    def on_channel_dissociation(self, value):
        self.channel_dissociation = int(value) / 100.0

        self.dissociation_label.setText(
            f"bonds broken in channel  "
            f"{self.channel_dissociation:.0%}"
        )

    def on_heat(self):
        target = min(self.simulation.target_temperature * 1.5, 4000)

        self.simulation.target_temperature = target
        self.temperature_slider.setValue(int(target))

    def on_cool(self):
        target = max(self.simulation.target_temperature * 0.6, 10)

        self.simulation.target_temperature = target
        self.temperature_slider.setValue(int(target))

    def on_spark(self):
        # A discharge channel: a hot column plus direct
        # dissociation of bonds inside it.
        #
        # Heat alone barely breaks anything, because random
        # thermal velocities put most of their energy into a pair
        # moving off together rather than apart. Real sparks
        # dissociate by electron impact instead, which puts energy
        # straight into the bond, so that is modelled explicitly.

        import discharge

        report = discharge.apply_to(
            self.simulation,
            np.random.default_rng(),
            radius=CHANNEL_RADIUS,
            temperature=self.channel_temperature,
            dissociation=self.channel_dissociation,
        )

        if report["struck"] == 0:
            print("channel missed everything")
            return

        print(
            f"lightning at {self.channel_temperature:.0f} K: "
            f"{report['struck']} atoms in the channel, "
            f"{report['bonds_in_channel']} bonds caught, "
            f"{report['dissociated']} broken, "
            f"{report['deposited']:.1f} eV deposited"
        )


    def current_positions(self):
        if self.replaying and len(self.recorder) > 0:
            return self.recorder.positions[self.scrubber.value()]

        return self.simulation.positions_numpy

    def current_bonds(self):
        if self.replaying and len(self.recorder) > 0:
            return self.recorder.bonds_at(self.scrubber.value())

        return self.simulation.bond_list()

    def current_symbols(self):
        if self.replaying and len(self.recorder) > 0:
            return self.recorder.symbols

        return self.simulation.symbols

    def find_molecules(self):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        positions = self.current_positions()
        first, second = self.current_bonds()

        count = len(positions)

        if len(first) == 0:
            labels = np.arange(count)
        else:
            graph = coo_matrix(
                (np.ones(len(first)), (first, second)),
                shape=(count, count)
            )

            _, labels = connected_components(graph, directed=False)

        self.molecule_labels = labels

        symbols = self.current_symbols()

        entries = []

        for label in np.unique(labels):
            members = np.where(labels == label)[0]

            counts = {}

            for member in members:
                symbol = symbols[member]
                counts[symbol] = counts.get(symbol, 0) + 1

            formula = "".join(
                symbol
                + (str(counts[symbol]) if counts[symbol] > 1 else "")
                for symbol in ["C", "N", "O", "H"]
                if symbol in counts
            )

            entries.append((int(label), formula, members))

        entries.sort(key=lambda item: -len(item[2]))

        self.molecule_entries = entries

        self.molecule_box.blockSignals(True)
        self.molecule_box.clear()

        for label, formula, members in entries:
            self.molecule_box.addItem(
                f"{formula}   ({len(members)} atoms)"
            )

        self.molecule_box.blockSignals(False)

        print(f"found {len(entries)} molecules")

        if entries:
            self.molecule_box.setCurrentIndex(0)
            self.on_select_molecule(0)

    def on_select_molecule(self, index):
        if index < 0 or index >= len(self.molecule_entries):
            return

        label, formula, members = self.molecule_entries[index]

        self.selected_molecule = label

        positions = self.current_positions()
        symbols = self.current_symbols()

        chosen = positions[members] % self.box_size

        anchor = chosen[0]

        offsets = chosen - anchor
        offsets -= self.box_size * np.round(offsets / self.box_size)

        centre = anchor + offsets.mean(axis=0)

        extent = float(np.linalg.norm(offsets, axis=1).max())

        self.view.opts["center"] = pg.Vector(
            float(centre[0]), float(centre[1]), float(centre[2])
        )
        self.view.setCameraPosition(distance=max(extent * 4.0, 5.0))
        self.view.update()

        # Bond order, not neighbour count. A double bond is one
        # partner but two bonds, so counting partners would report
        # the carbon in CO2 as a radical when it is satisfied.

        import analysis
        import reactive

        counted = analysis.bond_counts(
            positions,
            reactive.types_from_symbols(symbols),
            self.box_size
        )

        counts = {
            int(index): int(value)
            for index, value in enumerate(counted)
        }

        expected = {"H": 1, "C": 4, "N": 3, "O": 2}

        report = []
        suspect = False

        for member in members:
            symbol = symbols[member]
            got = counts.get(int(member), 0)
            want = expected[symbol]

            flag = ""

            if got > want:
                flag = "  TOO MANY"
                suspect = True
            elif got < want:
                flag = "  radical"

            report.append(f"  {symbol}{member}: {got}/{want}{flag}")

        print(f"\n{formula}  ({len(members)} atoms)")
        print("\n".join(report))
        print(
            "  -> over-coordinated, probably a clump"
            if suspect else "  -> valences look sane"
        )

        self.refresh_colours()

    def clear_molecule_selection(self):
        self.selected_molecule = None
        self.isolating = False
        self.isolate_button.setText("Isolate: off")
        self.refresh_colours()

    def toggle_isolate(self):
        if self.selected_molecule is None:
            print("nothing selected - press Find first")
            return

        self.isolating = not self.isolating

        self.isolate_button.setText(
            "Isolate: on" if self.isolating else "Isolate: off"
        )

        self.refresh_colours()

    def positions_for_drawing(self):
        if self.replaying and len(self.recorder) > 0:
            index = min(
                self.scrubber.value(), len(self.recorder) - 1
            )

            return self.recorder.positions[index] % self.box_size

        return self.simulation.positions_numpy % self.box_size

    def refresh_colours(self):
        colours = self.atom_colours()

        if (
            self.isolating
            and self.selected_molecule is not None
            and self.molecule_labels is not None
            and len(self.molecule_labels) == len(colours)
        ):
            others = self.molecule_labels != self.selected_molecule
            colours[others, 3] = 0.05

        self.scatter.setData(color=colours)

    # --------------------------------------------------------
    # Recording and replay

    def toggle_recording(self):
        self.recording = not self.recording

        self.record_button.setText(
            "Recording: on" if self.recording else "Recording: off"
        )

    def clear_recording(self):
        self.recorder.clear()
        self.scrubber.setRange(0, 0)
        self.go_live()

    def go_live(self):
        self.replaying = False
        self.playing_back = False
        self.play_button.setText("Play recording")
        self.frame_label.setText("live")

    def toggle_playback(self):
        if len(self.recorder) == 0:
            print("nothing recorded yet")
            return

        self.replaying = True
        self.playing_back = not self.playing_back
        self.last_advance = time.perf_counter()

        self.play_button.setText(
            "Stop playback" if self.playing_back
            else "Play recording"
        )

    def on_playback_speed(self, value):
        self.playback_fps = int(value) / 10.0

        if self.playback_fps < 1.0:
            description = (
                f"1 frame every {1.0 / self.playback_fps:.1f} s"
            )
        else:
            description = f"{self.playback_fps:.1f} frames/s"

        self.playback_label.setText(f"playback  {description}")

    def on_record_interval(self, value):
        self.record_every_steps = int(value)

        self.record_label.setText(
            f"capture every {self.record_every_steps} steps"
            f"  ({self.record_every_steps * TIME_STEP:.2f} fs)"
        )

    def nudge(self, amount):
        if len(self.recorder) == 0:
            return

        self.replaying = True

        target = int(
            np.clip(
                self.scrubber.value() + amount,
                0,
                len(self.recorder) - 1
            )
        )

        self.scrubber.setValue(target)

    def on_scrub(self, value):
        if len(self.recorder) == 0:
            return

        self.replaying = True

        self.show_recorded_frame(int(value))

    def show_recorded_frame(self, index):
        index = int(np.clip(index, 0, len(self.recorder) - 1))

        positions = self.recorder.positions[index]

        self.scatter.setData(pos=positions % self.box_size)

        first, second = self.recorder.bonds_at(index)

        if len(first) == 0:
            self.bond_item.setData(pos=np.zeros((2, 3)))
        else:
            start = positions[first]
            end = positions[second]

            offset = end - start
            offset -= self.box_size * np.round(
                offset / self.box_size
            )

            keep = np.linalg.norm(offset, axis=1) < self.box_size / 2.0

            start = start[keep]
            offset = offset[keep]

            if len(start) == 0:
                self.bond_item.setData(pos=np.zeros((2, 3)))
            else:
                segments = np.empty(
                    (2 * len(start), 3), dtype=np.float32
                )
                segments[0::2] = start
                segments[1::2] = start + offset

                self.bond_item.setData(pos=segments, mode="lines")

        formulas = self.recorder.formulas_at(index)

        ordered = sorted(
            formulas.items(),
            key=lambda item: (-len(item[0]), -item[1])
        )

        self.inventory.setText(
            "replay\n"
            + "   ".join(
                f"{name} x{number}" for name, number in ordered[:18]
            )
        )

        self.frame_label.setText(
            f"replay  frame {index + 1}/{len(self.recorder)}   "
            f"{self.recorder.times[index]:.0f} fs   "
            f"T {self.recorder.temperature[index]:.0f} K"
        )

    def continue_from_here(self):
        # Rebuild the simulation from whichever frame is on screen
        # and carry on from there.

        if len(self.recorder) == 0:
            print("nothing recorded to continue from")
            return

        index = (
            self.scrubber.value() if self.replaying
            else len(self.recorder) - 1
        )

        positions = self.recorder.positions[index]
        symbols = self.recorder.symbols

        from reactive_torch import ReactiveSimulation

        self.simulation = ReactiveSimulation(
            symbols=symbols,
            positions=positions.astype(float),
            box_size=self.recorder.box_size,
            time_step=TIME_STEP,
            target_temperature=(
                self.simulation.target_temperature
            ),
            friction=FRICTION,
        )

        if self.recorder.has_velocities:
            import torch

            self.simulation.velocities = torch.tensor(
                self.recorder.velocities[index].astype(float),
                device=self.simulation.device,
                dtype=self.simulation.dtype
            )

            note = "velocities restored"
        else:
            note = (
                "no velocities in this recording, "
                "redrawn thermally"
            )

        self.simulation.elapsed_femtoseconds = float(
            self.recorder.times[index]
        )

        self.simulation.forces, self.simulation._potential_energy = (
            self.simulation.compute_forces()
        )

        self.recorder.positions = self.recorder.positions[:index + 1]
        self.recorder.velocities = (
            self.recorder.velocities[:index + 1]
            if self.recorder.velocities else []
        )
        self.recorder.times = self.recorder.times[:index + 1]
        self.recorder.potential = self.recorder.potential[:index + 1]
        self.recorder.kinetic = self.recorder.kinetic[:index + 1]
        self.recorder.temperature = (
            self.recorder.temperature[:index + 1]
        )

        self.box_size = float(self.recorder.box_size)

        self.scatter.setData(
            pos=self.simulation.positions_numpy % self.box_size,
            color=self.atom_colours(),
            size=self.atom_sizes(),
            pxMode=False
        )

        self.scrubber.blockSignals(True)
        self.scrubber.setRange(0, max(len(self.recorder) - 1, 0))
        self.scrubber.setValue(len(self.recorder) - 1)
        self.scrubber.blockSignals(False)

        self.go_live()
        self.paused = False
        self.pause_button.setText("Pause")

        self.add_box_outline()
        self.recentre()

        print(
            f"continuing from frame {index + 1} "
            f"at {self.recorder.times[index]:.0f} fs ({note})"
        )

    def on_save(self):
        if len(self.recorder) == 0:
            print("nothing recorded yet")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save recording", "run.npz", "Recordings (*.npz)"
        )

        if path:
            self.recorder.save(path)
            print(f"saved {len(self.recorder)} frames to {path}")

    def on_load_recording(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load recording", "", "Recordings (*.npz)"
        )

        if not path:
            return

        self.load_recording_from(path)

    def load_recording_from(self, path):
        self.recorder = Recorder.load(path)
        self.box_size = self.recorder.box_size

        self.scatter.setData(
            pos=self.recorder.positions[0],
            color=np.array([
                ELEMENT_COLOUR[symbol]
                for symbol in self.recorder.symbols
            ], dtype=np.float32),
            size=np.array([
                ELEMENT_SIZE[symbol]
                for symbol in self.recorder.symbols
            ], dtype=np.float32),
            pxMode=False
        )

        self.add_box_outline()
        self.recentre()

        self.scrubber.setRange(0, len(self.recorder) - 1)
        self.scrubber.setValue(0)

        self.replaying = True
        self.show_recorded_frame(0)

        print(f"loaded {len(self.recorder)} frames from {path}")

    def on_export(self):
        if len(self.recorder) == 0:
            print("nothing recorded yet")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export XYZ", "run.xyz", "XYZ files (*.xyz)"
        )

        if path:
            self.recorder.export_xyz(path)
            print(f"exported {len(self.recorder)} frames to {path}")

    # --------------------------------------------------------

    def on_load(self):
        self.start_name = self.start_box.currentText()
        self.rebuild()

    def on_reseed(self):
        self.seed += 1
        self.rebuild()

    def rebuild(self):
        # The new box is built at whatever size the slider is
        # showing, rather than the constant at the top of the
        # file. Set the size first, then load: the two settings
        # belong together, and having Load quietly override the
        # slider is how a box ends up four times denser than
        # intended.

        requested = self.box_slider.value() / 10.0

        self.simulation = make_simulation(
            self.start_name,
            box=requested,
            temperature=self.simulation.target_temperature,
            seed=self.seed
        )

        self.box_size = float(self.simulation.box_size)
        self.update_box_label()

        self.time_history.clear()
        self.energy_history.clear()
        self.frame_index = 0

        self.recorder = Recorder(
            self.simulation.symbols,
            self.box_size,
            maximum_frames=MAXIMUM_FRAMES
        )

        self.scrubber.setRange(0, 0)
        self.go_live()

        self.scatter.setData(
            pos=self.simulation.positions_numpy,
            color=self.atom_colours(),
            size=self.atom_sizes(),
            pxMode=False
        )

        self.update_bonds()
        self.recentre()

        print(f"loaded {self.start_name}: "
              f"{self.simulation.atom_count} atoms")

    # --------------------------------------------------------

    def update_frame(self):
        if self.replaying:
            # Scrubbing or playing back a recording. The
            # simulation is left exactly where it was, so Back to
            # live picks up from the same place.

            if self.playing_back and len(self.recorder) > 0:
                now = time.perf_counter()

                interval = 1.0 / max(self.playback_fps, 0.01)

                if now - self.last_advance >= interval:
                    # Advance by however many intervals have
                    # actually elapsed, so playback keeps real
                    # time even if a frame took a while to draw.

                    steps = int(
                        (now - self.last_advance) / interval
                    )

                    self.last_advance = now

                    target = self.scrubber.value() + steps

                    if target >= len(self.recorder):
                        target = 0

                    self.scrubber.setValue(int(target))

            return

        if not self.paused:
            # The frame's worth of stepping is broken into chunks
            # so the recorder can sample partway through. Copying
            # 82 positions off the GPU costs about a kilobyte and
            # nothing in time, so this is essentially free and it
            # decouples playback smoothness from run speed.

            remaining = self.steps_per_frame
            captured = False

            while remaining > 0:
                chunk = (
                    min(remaining, self.record_every_steps)
                    if self.recording
                    else remaining
                )

                self.simulation.step(chunk)

                remaining -= chunk

                if self.recording:
                    self.recorder.capture(
                        self.simulation.positions_numpy,
                        self.simulation.elapsed_femtoseconds,
                        self.simulation.potential_energy,
                        self.simulation.kinetic_energy,
                        self.simulation.temperature,
                        velocities=(
                            self.simulation.velocities
                            .detach().cpu().numpy()
                        ),
                        box_size=self.simulation.box_size,
                    )

                    captured = True

            if captured:
                self.scrubber.blockSignals(True)
                self.scrubber.setRange(
                    0, max(len(self.recorder) - 1, 0)
                )
                self.scrubber.setValue(len(self.recorder) - 1)
                self.scrubber.blockSignals(False)

                self.frame_label.setText(
                    f"live  {len(self.recorder)} frames  "
                    f"every {self.recorder.stride * self.record_every_steps}"
                    f" steps"
                )

        self.frame_index += 1

        if self.auto_orbit:
            self.view.opts["azimuth"] = (
                self.view.opts["azimuth"] + 0.3
            ) % 360.0

        self.scatter.setData(
            pos=self.simulation.positions_numpy % self.box_size
        )

        if self.frame_index % BOND_REFRESH_EVERY == 0:
            self.update_bonds()

        now = time.perf_counter()
        elapsed = max(now - self.last_time, 1e-9)
        self.last_time = now

        self.frame_times.append(elapsed)

        if len(self.frame_times) > 30:
            self.frame_times.pop(0)

        frames_per_second = 1.0 / float(np.mean(self.frame_times))

        self.time_history.append(
            self.simulation.elapsed_femtoseconds
        )
        self.energy_history.append(self.simulation.potential_energy)

        if len(self.time_history) > 800:
            self.time_history.pop(0)
            self.energy_history.pop(0)

        if self.frame_index % 5 == 0:
            self.energy_curve.setData(
                self.time_history,
                self.energy_history
            )

        if self.frame_index % 10 == 0:
            formulas = self.simulation.molecule_formulas()

            ordered = sorted(
                formulas.items(),
                key=lambda item: (-len(item[0]), -item[1])
            )

            text = "   ".join(
                f"{name} x{number}" for name, number in ordered[:18]
            )

            self.inventory.setText(text)

        self.status.setText(
            f"device       {str(self.simulation.device):>10}\n"
            f"atoms        {self.simulation.atom_count:10d}\n"
            f"time         "
            f"{self.simulation.elapsed_femtoseconds:10.1f} fs\n"
            f"temperature  {self.simulation.temperature:10.0f} K\n"
            f"target       "
            f"{self.simulation.target_temperature:10.0f} K\n"
            f"potential    "
            f"{self.simulation.potential_energy:10.2f} eV\n"
            f"kinetic      {self.simulation.kinetic_energy:10.2f} eV\n"
            f"steps/frame  {self.steps_per_frame:10d}\n"
            f"fps          {frames_per_second:10.1f}"
        )


def main():
    pg.setConfigOptions(antialias=True)

    # Optional: py run_reactive_gl.py --load runs/run_003.npz
    # The browser uses this to open a recording directly.

    load_path = None

    if "--load" in sys.argv:
        position = sys.argv.index("--load")

        if position + 1 < len(sys.argv):
            load_path = sys.argv[position + 1]

    application = QtWidgets.QApplication(sys.argv[:1])

    viewer = Viewer()

    if load_path:
        viewer.load_recording_from(load_path)

    viewer.show()

    sys.exit(application.exec())


if __name__ == "__main__":
    main()
