"""Embedded trajectory Replay Mode for Chemistry Lab."""

from __future__ import annotations

import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from lab_renderer import MolecularScene
from recorder import Recorder
from replay import BASE_FEMTOSECONDS_PER_SECOND, ReplayClock


def scan_reconstructed_reactions(recorder):
    """Find recorded-frame bond changes without blocking the Replay UI."""
    events = []
    previous = None
    for frame in range(len(recorder)):
        ids = np.asarray(recorder.atom_ids_at(frame), dtype=int)
        first, second = recorder.bonds_at(frame)
        current = {
            tuple(sorted((int(ids[left]), int(ids[right]))))
            for left, right in zip(first, second)
        }
        if previous is not None:
            formed = current - previous
            broken = previous - current
            if formed or broken:
                events.append({
                    "frame": frame,
                    "time_fs": float(recorder.times[frame]),
                    "formed": formed,
                    "broken": broken,
                })
        previous = current
    if not events:
        return []
    grouped = [events[0]]
    for event in events[1:]:
        prior = grouped[-1]
        if event["frame"] <= prior["frame"] + 1:
            prior["formed"] |= event["formed"]
            prior["broken"] |= event["broken"]
        else:
            grouped.append(event)
    return grouped


class ReplayWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.clock = ReplayClock()
        self.path = None
        self.selected_ids = []
        self._bond_cache_index = None
        self._bond_cache = (np.array([], dtype=int), np.array([], dtype=int))
        self.failure_events = []
        self.failure_lines = []
        self.reaction_events = []
        self.reaction_lines = []
        self._active_reaction_event = None
        self._reaction_change_index = 0
        self._reaction_executor = ThreadPoolExecutor(max_workers=1)
        self._reaction_future = None
        self._reaction_path = None
        self._distance_signature = None
        self.distance_curves = []
        self._atom_menu_signature = None
        self._components_by_species = {}
        self._focused_molecule_ids = []
        self._active_species = None
        self._active_component_index = 0
        self._camera_animation = None
        self._last_tick = time.perf_counter()
        self._scrubbing = False
        self._build_ui()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

    def _button(self, text, callback):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(callback)
        return button

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self._button("Load trajectory", self.choose_file))
        controls.addWidget(self._button("|<", lambda: self.seek_frame(0)))
        controls.addWidget(self._button("< frame", lambda: self.step_frame(-1)))
        self.play_button = self._button("Play", self.toggle_play)
        controls.addWidget(self.play_button)
        controls.addWidget(self._button("frame >", lambda: self.step_frame(1)))
        controls.addWidget(self._button(">|", self.seek_end))
        controls.addWidget(
            self._button("< reaction", lambda: self.jump_reaction(-1))
        )
        controls.addWidget(
            self._button("reaction >", lambda: self.jump_reaction(1))
        )
        self.reaction_change_button = self._button(
            "Next change", self.next_reaction_change
        )
        self.reaction_change_button.setEnabled(False)
        controls.addWidget(self.reaction_change_button)
        controls.addWidget(
            self._button("< failure", lambda: self.jump_failure(-1))
        )
        controls.addWidget(
            self._button("failure >", lambda: self.jump_failure(1))
        )
        controls.addWidget(QtWidgets.QLabel("replay speed"))
        self.speed_box = QtWidgets.QComboBox()
        for speed in (0.25, 0.5, 1, 2, 5, 10):
            self.speed_box.addItem(f"{speed:g}x", speed)
        self.speed_box.setCurrentIndex(2)
        self.speed_box.currentIndexChanged.connect(self.on_speed)
        controls.addWidget(self.speed_box)
        self.follow_box = QtWidgets.QCheckBox("follow selected molecule")
        controls.addWidget(self.follow_box)
        self.dim_box = QtWidgets.QCheckBox("dim other atoms")
        self.dim_box.stateChanged.connect(self.on_dim)
        controls.addWidget(self.dim_box)
        self.highlight_box = QtWidgets.QCheckBox("highlight selected atoms")
        self.highlight_box.stateChanged.connect(self.on_highlight)
        controls.addWidget(self.highlight_box)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.timeline = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 10000)
        self.timeline.sliderPressed.connect(self.begin_scrub)
        self.timeline.sliderReleased.connect(self.end_scrub)
        self.timeline.valueChanged.connect(self.on_timeline)
        layout.addWidget(self.timeline)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.scene = MolecularScene()
        self.scene.setMinimumWidth(640)
        self.scene.atomClicked.connect(self.on_atom_clicked)
        splitter.addWidget(self.scene)

        side = QtWidgets.QWidget()
        side.setMinimumWidth(380)
        side_layout = QtWidgets.QVBoxLayout(side)
        self.file_label = QtWidgets.QLabel("No trajectory loaded")
        self.file_label.setWordWrap(True)
        side_layout.addWidget(self.file_label)
        self.time_label = QtWidgets.QLabel("time —    frame —")
        self.time_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        side_layout.addWidget(self.time_label)
        self.state_label = QtWidgets.QLabel("No recorded state")
        self.state_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        self.state_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        side_layout.addWidget(self.state_label)

        side_layout.addWidget(QtWidgets.QLabel("current species"))
        self.atom_box = QtWidgets.QComboBox()
        self.atom_box.setEditable(False)
        self.atom_box.setMaxVisibleItems(12)
        self.atom_box.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.atom_box.setMinimumContentsLength(24)
        self.atom_box.addItem("Choose a species...", None)
        self.atom_box.activated.connect(self.on_atom_chosen)
        side_layout.addWidget(self.atom_box)
        self.next_molecule_button = self._button(
            "Next molecule", self.next_molecule
        )
        self.next_molecule_button.setEnabled(False)
        side_layout.addWidget(self.next_molecule_button)
        self.inventory_label = QtWidgets.QLabel("No molecular inventory")
        self.inventory_label.setWordWrap(True)
        self.inventory_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.inventory_label.setMinimumHeight(76)
        self.inventory_label.setStyleSheet(
            "font-family: monospace; font-size: 12px; padding: 8px; "
            "background: #202426; border: 1px solid #454b4e; "
            "border-radius: 4px;"
        )
        side_layout.addWidget(self.inventory_label)
        self.event_label = QtWidgets.QLabel("No recorded failure selected")
        self.event_label.setWordWrap(True)
        self.event_label.setStyleSheet("color: #e09a55;")
        side_layout.addWidget(self.event_label)
        self.selection_label = QtWidgets.QLabel(
            "Click atoms in the 3D view. Up to three stable atom IDs are kept."
        )
        self.selection_label.setWordWrap(True)
        side_layout.addWidget(self.selection_label)
        side_layout.addWidget(self._button("Clear selection", self.clear_selection))

        self.distance_graph = pg.PlotWidget()
        self.distance_graph.setLabel("left", "selected distance (A)")
        self.distance_graph.setLabel("bottom", "simulation time (fs)")
        self.distance_graph.setMinimumHeight(135)
        self.distance_graph.setMaximumHeight(175)
        self.distance_graph.addLegend()
        self.distance_cursor = pg.InfiniteLine(
            angle=90, movable=False, pen="#eeeeee"
        )
        self.distance_graph.addItem(self.distance_cursor)
        self.distance_graph.scene().sigMouseClicked.connect(
            self.on_distance_graph_clicked
        )
        self.distance_graph.hide()
        side_layout.addWidget(self.distance_graph)

        self.graph = pg.PlotWidget()
        self.graph.setLabel("left", "energy / temperature")
        self.graph.setLabel("bottom", "simulation time (fs)")
        self.potential_curve = self.graph.plot(pen="#4a90c4", name="potential")
        self.kinetic_curve = self.graph.plot(pen="#d88c36", name="kinetic")
        self.total_curve = self.graph.plot(pen="#70b56c", name="total")
        self.temperature_curve = self.graph.plot(pen="#ad6ec9", name="temperature")
        self.cursor = pg.InfiniteLine(angle=90, movable=False, pen="#eeeeee")
        self.graph.addItem(self.cursor)
        self.graph.scene().sigMouseClicked.connect(self.on_graph_clicked)
        self.graph.setMinimumHeight(170)
        self.graph.setMaximumHeight(240)
        side_layout.addWidget(self.graph)
        side_layout.addWidget(QtWidgets.QLabel(
            f"At 1x, replay advances {BASE_FEMTOSECONDS_PER_SECOND:g} fs per "
            "real second. Motion is interpolated; bonds and scalar values are "
            "taken only from real recorded frames."
        ))
        side_layout.addStretch(1)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setSizes([1100, 650])
        layout.addWidget(splitter, stretch=1)

    def choose_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load trajectory", "runs", "Trajectory recordings (*.npz)"
        )
        if path:
            self.load_path(path)

    def load_path(self, path):
        recorder = Recorder.load(path)
        if not len(recorder):
            raise ValueError("trajectory contains no frames")
        self.path = os.path.abspath(path)
        self.clock.load(recorder)
        self.selected_ids = []
        self._focused_molecule_ids = []
        self._active_species = None
        self._active_reaction_event = None
        self._reaction_change_index = 0
        self._atom_menu_signature = None
        self._bond_cache_index = None
        self.file_label.setText(self.path)
        times = np.asarray(recorder.times, dtype=float)
        potential = np.asarray(recorder.potential, dtype=float)
        kinetic = np.asarray(recorder.kinetic, dtype=float)
        temperature = np.asarray(recorder.temperature, dtype=float)
        self.potential_curve.setData(times, potential)
        self.kinetic_curve.setData(times, kinetic)
        self.total_curve.setData(times, potential + kinetic)
        self.temperature_curve.setData(times, temperature)
        self.build_failure_events(times, potential, kinetic)
        if getattr(recorder, "events", None):
            self.load_authoritative_events(recorder)
        else:
            self.start_reaction_scan(recorder)
        self.render()
        self.scene.recentre()

    def build_failure_events(self, times, potential, kinetic):
        for line in self.failure_lines:
            self.graph.removeItem(line)
        self.failure_lines = []
        total = potential + kinetic
        rises = np.diff(total)
        threshold = max(80.0, 0.08 * abs(float(potential[-1])))
        self.failure_events = [
            {
                "frame": int(index + 1),
                "time_fs": float(times[index + 1]),
                "jump_eV": float(rises[index]),
            }
            for index in np.where(rises > threshold)[0]
        ]
        for event in self.failure_events:
            line = pg.InfiniteLine(
                pos=event["time_fs"], angle=90,
                pen=pg.mkPen("#e06b48", width=2),
                label=f"+{event['jump_eV']:.0f} eV",
                labelOpts={"color": "#e6a087", "position": 0.9},
            )
            self.graph.addItem(line)
            self.failure_lines.append(line)
        if self.failure_events:
            self.event_label.setText(
                f"{len(self.failure_events)} integration-energy jump"
                + ("s" if len(self.failure_events) != 1 else "")
                + " detected. Use the failure buttons to inspect them."
            )
        else:
            self.event_label.setText("No integration-energy jumps detected")

    def start_reaction_scan(self, recorder):
        for line in self.reaction_lines:
            self.graph.removeItem(line)
        self.reaction_lines = []
        self.reaction_events = []
        self._reaction_path = self.path
        self._reaction_future = self._reaction_executor.submit(
            scan_reconstructed_reactions, recorder
        )
        self.event_label.setText(
            "Reconstructing recorded-frame bond changes in the background..."
        )

    def load_authoritative_events(self, recorder):
        for line in self.reaction_lines:
            self.graph.removeItem(line)
        self.reaction_lines = []
        times = np.asarray(recorder.times, dtype=float)
        grouped = {}
        for stored in recorder.events:
            formed = {tuple(map(int, pair)) for pair in stored.get("formed", [])}
            broken = {tuple(map(int, pair)) for pair in stored.get("broken", [])}
            if not formed and not broken:
                continue
            time_fs = float(stored["time_fs"])
            window_id = stored.get("window_id")
            key = ("window", int(window_id)) if window_id is not None else (
                "event", time_fs
            )
            if key not in grouped:
                grouped[key] = {
                    "frame": int(np.argmin(np.abs(times - time_fs))),
                    "time_fs": time_fs,
                    "formed": set(),
                    "broken": set(),
                    "authoritative": True,
                    "reason": stored.get("reason", stored.get("type", "event")),
                    "event_count": 0,
                }
            grouped[key]["formed"] |= formed
            grouped[key]["broken"] |= broken
            grouped[key]["event_count"] += 1
        events = sorted(grouped.values(), key=lambda item: item["time_fs"])
        self.reaction_events = events
        for event in events[:200]:
            line = pg.InfiniteLine(
                pos=event["time_fs"], angle=90,
                pen=pg.mkPen("#45c07a", width=2),
            )
            self.graph.addItem(line)
            self.reaction_lines.append(line)
        self._reaction_future = None
        dropped = int(getattr(recorder, "adaptive_dropped_frames", 0))
        self.event_label.setText(
            f"{len(events)} authoritative recorded bond-change event"
            + ("s" if len(events) != 1 else "")
            + ". Green markers come directly from the recorder."
            + (f" {dropped} lower-priority frames were thinned." if dropped else "")
        )

    def finish_reaction_scan(self):
        future = self._reaction_future
        if future is None or not future.done():
            return
        self._reaction_future = None
        events = future.result()
        if self._reaction_path != self.path:
            return
        self.reaction_events = events
        # Keep the energy plot readable on extremely reactive recordings.
        stride = max(1, int(np.ceil(len(events) / 200)))
        for event in events[::stride]:
            line = pg.InfiniteLine(
                pos=event["time_fs"], angle=90,
                pen=pg.mkPen("#5b9bd5", width=1),
            )
            self.graph.addItem(line)
            self.reaction_lines.append(line)
        self.event_label.setText(
            f"{len(events)} reconstructed bond-change event"
            + ("s" if len(events) != 1 else "")
            + ". Blue markers are reconstructed from recorded frames."
        )

    def jump_reaction(self, direction):
        if not self.reaction_events:
            return
        current = self.clock.time_fs
        if direction > 0:
            event = next(
                (item for item in self.reaction_events if item["time_fs"] > current + 1e-9),
                self.reaction_events[0],
            )
        else:
            event = next(
                (item for item in reversed(self.reaction_events) if item["time_fs"] < current - 1e-9),
                self.reaction_events[-1],
            )
        self.clock.set_playing(False)
        self.clock.seek_frame(event["frame"])
        self._active_reaction_event = event
        self._reaction_change_index = 0
        self.focus_reaction_change()
        self.render()

    def reaction_changes(self, event):
        changes = [
            ("formed", pair) for pair in sorted(event["formed"])
        ]
        changes += [
            ("broken", pair) for pair in sorted(event["broken"])
        ]
        return changes

    def next_reaction_change(self):
        event = self._active_reaction_event
        if event is None:
            return
        changes = self.reaction_changes(event)
        if not changes:
            return
        self._reaction_change_index = (
            self._reaction_change_index + 1
        ) % len(changes)
        self.focus_reaction_change()
        self.render()

    def focus_reaction_change(self):
        event = self._active_reaction_event
        changes = self.reaction_changes(event) if event is not None else []
        if not changes:
            self.reaction_change_button.setEnabled(False)
            return
        self._reaction_change_index %= len(changes)
        kind, pair = changes[self._reaction_change_index]
        self._focused_molecule_ids = list(pair)
        self.selected_ids = []
        self._active_species = None
        self.focus_ids(pair)
        recorder = self.clock.recorder
        frame = event["frame"]
        ids = np.asarray(recorder.atom_ids_at(frame), dtype=int)
        symbols = recorder.symbols_at(frame)
        names = []
        for atom_id in pair:
            match = np.where(ids == atom_id)[0]
            names.append(str(symbols[int(match[0])]) if len(match) else "atom")
        self.reaction_change_button.setEnabled(len(changes) > 1)
        self.reaction_change_button.setText(
            f"Next change ({self._reaction_change_index + 1}/{len(changes)})"
        )
        self.event_label.setText(
            f"Reaction at {event['time_fs']:.1f} fs — change "
            f"{self._reaction_change_index + 1}/{len(changes)}: "
            f"{names[0]}-{names[1]} bond {kind}. Camera centred on those atoms."
        )

    def jump_failure(self, direction):
        if not self.failure_events:
            return
        current = self.clock.time_fs
        if direction > 0:
            event = next(
                (
                    item for item in self.failure_events
                    if item["time_fs"] > current + 1e-9
                ),
                self.failure_events[0],
            )
        else:
            event = next(
                (
                    item for item in reversed(self.failure_events)
                    if item["time_fs"] < current - 1e-9
                ),
                self.failure_events[-1],
            )
        self.clock.set_playing(False)
        self.clock.seek_time(event["time_fs"])
        self.speed_box.setCurrentIndex(0)
        self.event_label.setText(
            f"Integration-energy jump at {event['time_fs']:.1f} fs: "
            f"+{event['jump_eV']:.1f} eV between recorded frames. "
            "Replay slowed to 0.25x."
        )
        self.render()

    def on_speed(self):
        self.clock.speed = float(self.speed_box.currentData())

    def toggle_play(self):
        if not self.clock.frame_count:
            return
        if self.clock.time_fs >= self.clock.end_time:
            self.clock.seek_time(self.clock.start_time)
        self.clock.set_playing(not self.clock.playing)
        self.play_button.setText("Pause" if self.clock.playing else "Play")

    def seek_frame(self, index):
        self.clock.set_playing(False)
        self.clock.seek_frame(index)
        self.render()

    def seek_end(self):
        if self.clock.frame_count:
            self.seek_frame(self.clock.frame_count - 1)

    def step_frame(self, amount):
        self.clock.set_playing(False)
        self.clock.step_frame(amount)
        self.render()

    def begin_scrub(self):
        self._scrubbing = True
        self.clock.set_playing(False)

    def end_scrub(self):
        self._scrubbing = False
        self.on_timeline(self.timeline.value())

    def on_timeline(self, value):
        if self._scrubbing or self.timeline.isSliderDown():
            self.clock.seek_fraction(value / 10000.0)
            self.render(update_slider=False)

    def on_graph_clicked(self, event):
        if not self.clock.frame_count or not self.graph.sceneBoundingRect().contains(event.scenePos()):
            return
        point = self.graph.getPlotItem().vb.mapSceneToView(event.scenePos())
        self.clock.set_playing(False)
        self.clock.seek_time(point.x())
        self.render()

    def on_distance_graph_clicked(self, event):
        if (
            not self.clock.frame_count
            or not self.distance_graph.sceneBoundingRect().contains(event.scenePos())
        ):
            return
        point = self.distance_graph.getPlotItem().vb.mapSceneToView(event.scenePos())
        self.clock.set_playing(False)
        self.clock.seek_time(point.x())
        self.render()

    def on_dim(self):
        self.scene.dim_unselected = self.dim_box.isChecked()
        self.scene.refresh_atoms()

    def on_highlight(self):
        self.scene.highlight_selected = self.highlight_box.isChecked()
        self.scene.refresh_atoms()

    def on_atom_clicked(self, slot, _modifiers):
        if not self.clock.frame_count:
            return
        frame = self.clock.current_frame()
        ids = np.asarray(self.clock.recorder.atom_ids_at(frame), dtype=int)
        component = self.component_containing_slot(slot)
        chosen_ids = [int(ids[index]) for index in component]
        self._focused_molecule_ids = chosen_ids
        self.selected_ids = chosen_ids if len(chosen_ids) == 1 else []
        symbols = self.clock.recorder.symbols_at(frame)
        self._active_species = (
            None if len(component) == 1 else
            self.molecular_formula(symbols[index] for index in component)
        )
        if self._active_species:
            components = self._components_by_species.get(self._active_species, [])
            chosen = tuple(chosen_ids)
            self._active_component_index = next(
                (index for index, item in enumerate(components) if tuple(item) == chosen),
                0,
            )
        self.render()

    def component_containing_slot(self, slot):
        first, second = self._bond_cache
        adjacency = defaultdict(list)
        for left, right in zip(first, second):
            adjacency[int(left)].append(int(right))
            adjacency[int(right)].append(int(left))
        found = {int(slot)}
        pending = [int(slot)]
        while pending:
            current = pending.pop()
            for neighbour in adjacency[current]:
                if neighbour not in found:
                    found.add(neighbour)
                    pending.append(neighbour)
        return sorted(found)

    def on_atom_chosen(self, index):
        entry = self.atom_box.itemData(index)
        if entry is None or not self.clock.frame_count:
            return
        kind, value = entry
        if kind == "atom":
            atom_id = int(value)
            self._focused_molecule_ids = []
            self._active_species = None
            self.selected_ids = [atom_id]
            self.focus_ids([atom_id])
        else:
            self._active_species = str(value)
            self._active_component_index = 0
            self.focus_active_molecule()
        self.render()

    def next_molecule(self):
        components = self._components_by_species.get(self._active_species, [])
        if not components:
            return
        self._active_component_index = (
            self._active_component_index + 1
        ) % len(components)
        self.focus_active_molecule()
        self.render()

    def focus_active_molecule(self):
        components = self._components_by_species.get(self._active_species, [])
        if not components:
            return
        self._active_component_index %= len(components)
        self._focused_molecule_ids = list(components[self._active_component_index])
        self.selected_ids = []
        self.next_molecule_button.setEnabled(True)
        self.next_molecule_button.setText(
            f"Next {self._active_species} molecule  "
            f"({self._active_component_index + 1}/{len(components)})"
        )
        self.focus_ids(self._focused_molecule_ids)

    def focus_ids(self, atom_ids):
        frame = self.clock.current_frame()
        ids = np.asarray(self.clock.recorder.atom_ids_at(frame), dtype=int)
        positions = self.clock.interpolated_positions()
        slots = [
            int(np.where(ids == int(atom_id))[0][0])
            for atom_id in atom_ids if np.any(ids == int(atom_id))
        ]
        if not slots:
            return
        box = float(self.clock.recorder.box_at(frame))
        chosen = positions[slots]
        anchor = chosen[0]
        offsets = chosen - anchor
        offsets -= box * np.round(offsets / box)
        self.smooth_center(anchor + offsets.mean(axis=0))

    def smooth_center(self, point, duration_ms=280):
        """Ease the camera target to an atom without changing its angle."""
        start_vector = self.scene.view.opts["center"]
        start = np.array([start_vector.x(), start_vector.y(), start_vector.z()])
        target = np.asarray(point, dtype=float)
        animation = QtCore.QVariantAnimation(self)
        animation.setDuration(duration_ms)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)

        def move(value):
            position = start + float(value) * (target - start)
            self.scene.view.opts["center"] = pg.Vector(*map(float, position))
            self.scene.view.update()

        animation.valueChanged.connect(move)
        animation.finished.connect(lambda: setattr(self, "_camera_animation", None))
        self._camera_animation = animation
        animation.start()

    @staticmethod
    def molecular_formula(symbols):
        counts = Counter(map(str, symbols))
        order = [item for item in ("C", "N", "O", "H") if item in counts]
        order += sorted(set(counts) - set(order))
        return "".join(
            item + (str(counts[item]) if counts[item] > 1 else "")
            for item in order
        )

    def update_atom_menu(self, frame, bonds):
        recorder = self.clock.recorder
        ids = np.asarray(recorder.atom_ids_at(frame), dtype=int)
        symbols = list(map(str, recorder.symbols_at(frame)))
        first, second = (np.asarray(values, dtype=int) for values in bonds)
        signature = (
            tuple(zip(ids.tolist(), symbols)),
            tuple(zip(first.tolist(), second.tolist())),
        )
        if signature == self._atom_menu_signature:
            return

        parent = list(range(len(ids)))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def join(left, right):
            left, right = find(left), find(right)
            if left != right:
                parent[right] = left

        for left, right in zip(first, second):
            if 0 <= left < len(ids) and 0 <= right < len(ids):
                join(int(left), int(right))
        components = defaultdict(list)
        for slot in range(len(ids)):
            components[find(slot)].append(slot)

        species = defaultdict(list)
        free_atoms = []
        for slots in components.values():
            if len(slots) == 1:
                slot = slots[0]
                free_atoms.append((int(ids[slot]), symbols[slot], slot))
            else:
                formula = self.molecular_formula(symbols[slot] for slot in slots)
                species[formula].append(tuple(int(ids[slot]) for slot in slots))
        self._components_by_species = dict(species)

        # Put chemically larger species first. For species of the same size,
        # put the one with more current instances first, then alphabetically.
        species_order = sorted(
            species,
            key=lambda formula: (
                -len(species[formula][0]), -len(species[formula]), formula
            ),
        )
        free_counts = Counter(symbol for _, symbol, _ in free_atoms)
        free_atoms.sort(key=lambda item: (-free_counts[item[1]], item[1], item[0]))

        self.atom_box.blockSignals(True)
        self.atom_box.clear()
        self.atom_box.addItem("Choose a species...", None)
        for formula in species_order:
            self.atom_box.addItem(
                f"{formula} molecule  x{len(species[formula])}",
                ("molecule", formula),
            )
        # Free atoms remain individually addressable by stable ID, but follow
        # the molecules and put the most abundant free element first.
        for atom_id, symbol, slot in free_atoms:
            self.atom_box.addItem(
                f"free {symbol}  ID {atom_id}  (slot {slot})",
                ("atom", atom_id),
            )
        self.atom_box.setCurrentIndex(0)
        self.atom_box.blockSignals(False)

        inventory_entries = [
            (formula, len(species[formula])) for formula in species_order
        ]
        inventory_entries += sorted(
            free_counts.items(), key=lambda item: (-item[1], item[0])
        )
        inventory_text = "   ".join(
            f"{name} x{count}" for name, count in inventory_entries
        ) or "none"
        self.inventory_label.setText(
            f"molecular inventory\n{inventory_text}\n"
            f"{sum(count for _, count in inventory_entries)} objects   "
            f"{len(ids)} atoms"
        )
        self.next_molecule_button.setEnabled(
            self._active_species in self._components_by_species
        )
        self._atom_menu_signature = signature

    def clear_selection(self):
        self.selected_ids = []
        self._focused_molecule_ids = []
        self._active_species = None
        self.render()

    def current_selected_slots(self, frame):
        ids = self.clock.recorder.atom_ids_at(frame)
        return [
            int(np.where(ids == atom_id)[0][0])
            for atom_id in self.selected_ids
            if np.any(ids == atom_id)
        ]

    def update_selection_text(self, frame, positions, box):
        recorder = self.clock.recorder
        ids = recorder.atom_ids_at(frame)
        symbols = recorder.symbols_at(frame)
        slots = self.current_selected_slots(frame)
        lines = [
            f"{symbols[slot]} atom ID {int(ids[slot])} (slot {slot})"
            for slot in slots
        ]
        if self._active_species and self._focused_molecule_ids:
            components = self._components_by_species.get(self._active_species, [])
            lines.insert(
                0,
                f"{self._active_species} molecule "
                f"{self._active_component_index + 1}/{len(components)}   "
                f"atom IDs {', '.join(map(str, self._focused_molecule_ids))}",
            )
        if len(slots) >= 2:
            lines.append("pair distances")
            for left in range(len(slots)):
                for right in range(left + 1, len(slots)):
                    offset = positions[slots[right]] - positions[slots[left]]
                    offset -= box * np.round(offset / box)
                    lines.append(
                        f"  ID {int(ids[slots[left]])}–{int(ids[slots[right]])}: "
                        f"{np.linalg.norm(offset):.3f} Å"
                    )
        self.selection_label.setText(
            "\n".join(lines) if lines else
            "Click atoms in the 3D view. Up to three stable atom IDs are kept."
        )

    def update_selection_text(self, frame, positions, box):
        """Describe chemistry without exposing recorder-internal atom IDs."""
        recorder = self.clock.recorder
        ids = np.asarray(recorder.atom_ids_at(frame), dtype=int)
        symbols = recorder.symbols_at(frame)
        focus_ids = self._focused_molecule_ids or self.selected_ids
        slots = [
            int(np.where(ids == int(atom_id))[0][0])
            for atom_id in focus_ids if np.any(ids == int(atom_id))
        ]
        if not slots:
            self.selection_label.setText(
                "Click an atom to inspect its whole molecule and current bonds."
            )
            return
        formula = self.molecular_formula(symbols[slot] for slot in slots)
        if len(slots) == 1:
            self.selection_label.setText(
                f"selected: free {formula} atom\nbonds: none"
            )
            return

        components = self._components_by_species.get(formula, [])
        suffix = (
            f"  {self._active_component_index + 1}/{len(components)}"
            if components else ""
        )
        lines = [
            f"selected: {formula} molecule{suffix}",
            f"atoms: {len(slots)}   bonds:",
        ]
        slot_set = set(slots)
        element_numbers = defaultdict(int)
        labels = {}
        for slot in slots:
            element_numbers[str(symbols[slot])] += 1
            labels[slot] = f"{symbols[slot]}{element_numbers[str(symbols[slot])]}"
        first, second = self._bond_cache
        for left, right in zip(first, second):
            left, right = int(left), int(right)
            if left not in slot_set or right not in slot_set:
                continue
            offset = positions[right] - positions[left]
            offset -= box * np.round(offset / box)
            lines.append(
                f"  {labels[left]}-{labels[right]}  "
                f"{np.linalg.norm(offset):.3f} A"
            )
        self.selection_label.setText("\n".join(lines))

    def update_distance_graph(self, frame):
        tracked = tuple(self._focused_molecule_ids or self.selected_ids)
        signature = tracked if 2 <= len(tracked) <= 3 else ()
        if signature == self._distance_signature:
            return
        self._distance_signature = signature
        for curve in self.distance_curves:
            self.distance_graph.removeItem(curve)
        self.distance_curves = []
        if not signature:
            self.distance_graph.hide()
            return

        recorder = self.clock.recorder
        current_ids = np.asarray(recorder.atom_ids_at(frame), dtype=int)
        current_symbols = recorder.symbols_at(frame)
        labels = {}
        counts = defaultdict(int)
        for atom_id in signature:
            matches = np.where(current_ids == atom_id)[0]
            symbol = str(current_symbols[int(matches[0])]) if len(matches) else "?"
            counts[symbol] += 1
            labels[atom_id] = f"{symbol}{counts[symbol]}"
        times = np.asarray(recorder.times, dtype=float)
        colours = ("#59a9e8", "#e38c45", "#75bd72")
        colour_index = 0
        for left_index in range(len(signature)):
            for right_index in range(left_index + 1, len(signature)):
                left_id = signature[left_index]
                right_id = signature[right_index]
                values = np.full(len(recorder), np.nan, dtype=float)
                for index in range(len(recorder)):
                    ids = np.asarray(recorder.atom_ids_at(index), dtype=int)
                    left = np.where(ids == left_id)[0]
                    right = np.where(ids == right_id)[0]
                    if not len(left) or not len(right):
                        continue
                    offset = (
                        recorder.positions[index][int(right[0])]
                        - recorder.positions[index][int(left[0])]
                    )
                    box = float(recorder.box_at(index))
                    offset -= box * np.round(offset / box)
                    values[index] = np.linalg.norm(offset)
                curve = self.distance_graph.plot(
                    times, values,
                    pen=pg.mkPen(colours[colour_index], width=2),
                    name=f"{labels[left_id]}-{labels[right_id]}",
                )
                self.distance_curves.append(curve)
                colour_index += 1
        self.distance_graph.show()

    def render(self, update_slider=True):
        if not self.clock.frame_count:
            return
        recorder = self.clock.recorder
        lower, upper, alpha = self.clock.frame_interval()
        positions = self.clock.interpolated_positions()
        box = (
            recorder.box_at(lower)
            + alpha * (recorder.box_at(upper) - recorder.box_at(lower))
        )
        symbols = recorder.symbols_at(lower)
        if self._bond_cache_index != lower:
            self._bond_cache = recorder.bonds_at(lower)
            self._bond_cache_index = lower
        bonds = self._bond_cache
        slots = self.current_selected_slots(lower)
        ids = np.asarray(recorder.atom_ids_at(lower), dtype=int)
        focus_slots = []
        for atom_id in self._focused_molecule_ids:
            matches = np.where(ids == int(atom_id))[0]
            if len(matches):
                focus_slots.append(int(matches[0]))
        self.scene.selected = set(slots + focus_slots)
        self.scene.set_state(positions, symbols, box, bonds)
        self.update_atom_menu(lower, bonds)
        self.update_selection_text(lower, positions, box)
        self.update_distance_graph(lower)
        tracked_slots = focus_slots or slots
        if self.follow_box.isChecked() and tracked_slots:
            chosen = positions[tracked_slots]
            anchor = chosen[0]
            offsets = chosen - anchor
            offsets -= box * np.round(offsets / box)
            point = anchor + offsets.mean(axis=0)
            self.scene.view.opts["center"] = pg.Vector(*map(float, point))
        self.time_label.setText(
            f"time {self.clock.time_fs:10.2f} fs    "
            f"recorded frame {lower + 1}/{self.clock.frame_count}    "
            f"between {lower + 1}→{upper + 1} ({alpha:.0%})    "
            f"T {recorder.temperature[lower]:.0f} K"
        )
        potential = float(recorder.potential[lower])
        kinetic = float(recorder.kinetic[lower])
        self.state_label.setText(
            f"atoms       {len(symbols):10d}\n"
            f"box         {box:10.2f} A\n"
            f"potential   {potential:10.2f} eV\n"
            f"kinetic     {kinetic:10.2f} eV\n"
            f"total       {potential + kinetic:10.2f} eV"
        )
        self.cursor.setValue(self.clock.time_fs)
        self.distance_cursor.setValue(self.clock.time_fs)
        if update_slider and self.clock.end_time > self.clock.start_time:
            fraction = (
                (self.clock.time_fs - self.clock.start_time)
                / (self.clock.end_time - self.clock.start_time)
            )
            self.timeline.blockSignals(True)
            self.timeline.setValue(int(round(fraction * 10000)))
            self.timeline.blockSignals(False)
        self.play_button.setText("Pause" if self.clock.playing else "Play")

    def tick(self):
        self.finish_reaction_scan()
        if not self.clock.frame_count:
            return
        before = self.clock.time_fs
        self.clock.advance()
        if self.clock.time_fs != before or self.clock.playing:
            self.render()

    def closeEvent(self, event):
        self._reaction_executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)
