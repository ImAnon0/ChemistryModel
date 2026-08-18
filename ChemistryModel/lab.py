import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

import pyqtgraph as pg

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import mixtures
import running
import molecule_library as molecule_store
import molecule_scanner
import qm_structure_validator as qm_validator
from high_fidelity_torch import HF_MODEL_REVISION
import characterisation_results as character_results


# ============================================================
# One place to run everything from
# ============================================================
#
#   py lab.py
#
# Four tabs. Run builds a job and queues it. Batches shows what
# is running, what is waiting and what has finished. Results
# reads the reports. Molecules captures structures discovered in
# recorded trajectories for later controlled characterisation.
#
# Batches are launched as separate processes rather than run
# inside this one. That keeps batch_runner.py usable from the
# command line, gives each batch its own GPU context, and means
# a crash costs one batch instead of the whole session. Progress
# comes from reading the index each batch writes after every run,
# which is more reliable than trying to parse its output.


QUEUE_FILE = "lab_queue.json"
TEMPLATE_FILE = "lab_templates.json"

POLL_MILLISECONDS = 1500

DEFAULT_CONCURRENCY = 3
GROUP_SIZE = 16
CHARACTERISATION_ROOT = "characterisation"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


STATES = ("queued", "running", "done", "stopped", "failed")


def read_index(path):
    target = os.path.join(path, "index.json")

    if not os.path.exists(target):
        return []

    try:
        with open(target) as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []


def _heartbeat_matches_seeds(payload, seeds):
    if not seeds:
        return True

    label = payload.get("seed")
    if label is None:
        return False

    wanted = {int(seed) for seed in seeds}
    text = str(label)

    try:
        if "-" in text:
            first, last = text.split("-", 1)
            return int(first) in wanted and int(last) in wanted
        return int(text) in wanted
    except (TypeError, ValueError):
        return False


def read_heartbeat(folder, pid, seeds=None):
    # Prefer the exact child-process heartbeat. If Lab has been restarted
    # from an older queue file whose PID was never persisted, fall back to
    # the newest heartbeat whose seed label belongs to this job. The runner
    # writes heartbeat files atomically, so a successful JSON load is a
    # complete update rather than a half-written one.

    candidates = []
    if pid:
        exact = os.path.join(folder, f".progress_{int(pid)}.json")
        if os.path.exists(exact):
            candidates.append(exact)

    if not candidates:
        candidates = glob.glob(os.path.join(folder, ".progress_*.json"))

    best = None
    best_updated = -1.0

    for path in candidates:
        try:
            with open(path) as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue

        if pid and int(payload.get("pid", -1)) == int(pid):
            return payload

        if not _heartbeat_matches_seeds(payload, seeds):
            continue

        try:
            updated = float(payload.get("updated", 0.0))
        except (TypeError, ValueError):
            updated = 0.0

        if updated >= best_updated:
            best = payload
            best_updated = updated

    return best


def entry_seeds(folder):
    # Which seeds have finished, read from the entry files rather
    # than the index. Entries appear the moment a run completes
    # and cannot be clobbered by another process, so they are the
    # honest source for progress.

    seeds = set()

    directory = os.path.join(folder, "entries")

    if os.path.isdir(directory):
        for name in os.listdir(directory):
            if name.startswith("seed_") and name.endswith(".json"):
                try:
                    path = os.path.join(directory, name)
                    with open(path, encoding="utf-8") as handle:
                        entry = json.load(handle)
                    if entry.get("finished", True):
                        seeds.add(int(name[5:-5]))
                except (ValueError, json.JSONDecodeError, OSError):
                    continue

    if not seeds:
        for entry in read_index(folder):
            if (
                entry.get("seed") is not None
                and entry.get("finished", True)
            ):
                seeds.add(int(entry["seed"]))

    return seeds


def find_batches(root):
    found = []

    if os.path.exists(os.path.join(root, "index.json")):
        found.append((os.path.basename(os.path.abspath(root)), root))

    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)

            if os.path.isdir(path) and os.path.exists(
                os.path.join(path, "index.json")
            ):
                found.append((name, path))

    return found


def matching_folder(root, wanted):
    # Mirror batch_runner's experiment identity. Adaptive-v2 fields that
    # were not written by older Lab batches are interpreted as the recorder
    # defaults that Lab used at the time; new runner entries record them
    # explicitly.

    for label, path in find_batches(root):
        index = read_index(path)
        if not index:
            continue

        first = index[0]
        strikes = int(first.get("strikes", 0))
        adaptive = bool(first.get("adaptive_recording", False))

        found = {
            "physics": first.get("physics", "reactive"),
            "mixture": first.get("mixture"),
            "box": round(float(first.get("box", 0)), 2),
            "picoseconds": round(float(
                first.get("requested_picoseconds", first.get("picoseconds", 0))
            ), 3),
            "strikes": strikes,
            "strike_temperature": (
                round(float(first.get("strike_temperature", 0) or 0), 0)
                if strikes else 0.0
            ),
            "strike_dissociation": (
                round(float(first.get("strike_dissociation", 0) or 0), 3)
                if strikes else 0.0
            ),
            "expand_to": round(float(first.get("expand_to", 0) or 0), 2),
            "hot_temperature": round(
                float(first.get("hot_temperature", 500) or 500), 0
            ),
            "cool_temperature": round(
                float(first.get("cool_temperature", 250) or 250), 0
            ),
            "adaptive_recording": adaptive,
            "adaptive_candidate_fs": round(float(
                first.get("adaptive_candidate_fs", 2.0 if adaptive else 0.0)
                or 0.0
            ), 3),
            "adaptive_pre_event_fs": round(float(
                first.get("adaptive_pre_event_fs", 100.0 if adaptive else 0.0)
                or 0.0
            ), 3),
            "adaptive_post_event_fs": round(float(
                first.get("adaptive_post_event_fs", 100.0 if adaptive else 0.0)
                or 0.0
            ), 3),
            "adaptive_energy_jump_ev": round(float(
                first.get("adaptive_energy_jump_ev", 20.0 if adaptive else 0.0)
                or 0.0
            ), 6),
            "adaptive_close_contact_scale": round(float(
                first.get("adaptive_close_contact_scale", 0.35 if adaptive else 0.0)
                or 0.0
            ), 4),
            "adaptive_reaction_window_fs": round(float(
                first.get("adaptive_reaction_window_fs", 20.0 if adaptive else 0.0)
                or 0.0
            ), 3),
            "adaptive_chemical_context_fs": round(float(
                first.get("adaptive_chemical_context_fs", 10.0 if adaptive else 0.0)
                or 0.0
            ), 3),
            "compiled_forces": bool(first.get("compiled_forces", False)),
        }

        differs = False
        for key, requested in wanted.items():
            existing = found.get(key)
            if key == "picoseconds":
                differs |= abs(float(existing) - float(requested)) > 0.011
            else:
                differs |= existing != requested

        if not differs:
            return label, path, index

    return None, None, []


class Choice(QtWidgets.QComboBox):
    # A dropdown you can also type into.
    #
    # Spin boxes are miserable for anything with a wide range:
    # getting from 250 to 30000 by arrow is absurd, and typing
    # into one still leaves the arrows there to be caught by a
    # stray scroll. This offers the values actually worth using
    # and accepts anything else typed in.

    def __init__(self, options, value=None, decimals=1):
        super().__init__()

        self.decimals = decimals

        self.setEditable(True)
        self.setInsertPolicy(
            QtWidgets.QComboBox.InsertPolicy.NoInsert
        )

        for option in options:
            self.addItem(self.format(option))

        self.setValue(value if value is not None else options[0])

        # Scrolling over a control while reading a form should not
        # silently change the experiment.

        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def format(self, value):
        if self.decimals == 0:
            return f"{float(value):.0f}"

        text = f"{float(value):.{self.decimals}f}"

        return text.rstrip("0").rstrip(".") or "0"

    def wheelEvent(self, event):
        event.ignore()

    def value(self):
        try:
            return float(self.currentText().strip())
        except ValueError:
            return 0.0

    def setValue(self, value):
        self.setCurrentText(self.format(value))

    @property
    def valueChanged(self):
        return self.currentTextChanged


class SectionCard(QtWidgets.QFrame):
    """Shared titled surface used throughout the workbench."""

    def __init__(self, title, subtitle=""):
        super().__init__()
        self.setObjectName("sectionCard")
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(18, 16, 18, 18)
        self.layout.setSpacing(10)

        heading = QtWidgets.QLabel(title)
        heading.setObjectName("sectionTitle")
        self.layout.addWidget(heading)

        if subtitle:
            note = QtWidgets.QLabel(subtitle)
            note.setObjectName("sectionSubtitle")
            note.setWordWrap(True)
            self.layout.addWidget(note)

    def addWidget(self, widget, stretch=0):
        self.layout.addWidget(widget, stretch)

    def addLayout(self, layout, stretch=0):
        self.layout.addLayout(layout, stretch)


class CollapsibleCard(SectionCard):
    """A card that keeps uncommon scientific controls out of the way."""

    def __init__(self, title, subtitle="", expanded=False):
        QtWidgets.QFrame.__init__(self)
        self.setObjectName("sectionCard")
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 12, 18, 14)
        outer.setSpacing(8)

        self.toggle = QtWidgets.QToolButton()
        self.toggle.setObjectName("sectionToggle")
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        outer.addWidget(self.toggle)

        if subtitle:
            note = QtWidgets.QLabel(subtitle)
            note.setObjectName("sectionSubtitle")
            note.setWordWrap(True)
            outer.addWidget(note)

        self.body = QtWidgets.QWidget()
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 6, 0, 0)
        self.body_layout.setSpacing(10)
        self.body.setVisible(expanded)
        outer.addWidget(self.body)
        self.toggle.toggled.connect(self.body.setVisible)
        self.toggle.toggled.connect(self._update_arrow)
        self._update_arrow(expanded)

    def _update_arrow(self, expanded):
        self.toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow
            if expanded else QtCore.Qt.ArrowType.RightArrow
        )

    def addWidget(self, widget, stretch=0):
        self.body_layout.addWidget(widget, stretch)

    def addLayout(self, layout, stretch=0):
        self.body_layout.addLayout(layout, stretch)


class Job:

    def __init__(self, name, arguments, out, runs, seeds=None,
                 runner="batch_runner.py"):
        self.name = name
        self.arguments = arguments
        self.out = out
        self.runs = runs
        self.runner = runner

        # Which seeds belong to this job. Several jobs can share
        # one folder, so counting everything in that folder would
        # have every part reporting the same progress.

        self.seeds = list(seeds) if seeds else []

        self.state = "queued"
        self.process = None
        self.pid = None
        self.reattached = False

        # How far through the run currently being computed.

        self.run_fraction = 0.0
        self.run_seed = None
        self.inflight_runs = 1
        self.started = None
        self.finished = None
        self.completed = 0
        self.headlines = []
        self.live_chemistry = None
        self.run_phase = "queued"
        self.results_done = 0
        self.results_total = 0
        self.log_path = None

    def as_dict(self):
        return {
            "name": self.name,
            "arguments": self.arguments,
            "out": self.out,
            "runs": self.runs,
            "state": self.state,
            "pid": self.pid,
            "seeds": self.seeds,
            "runner": self.runner,
        }

    @classmethod
    def from_dict(cls, stored):
        job = cls(
            stored["name"], stored["arguments"],
            stored["out"], stored["runs"],
            stored.get("seeds"),
            stored.get("runner", "batch_runner.py"),
        )

        job.state = stored.get("state", "queued")
        job.pid = stored.get("pid")

        return job

    def refresh(self):
        # Progress comes from the entry files each run writes,
        # which appear one at a time and can be attributed to a
        # particular job by seed.

        done = entry_seeds(self.out)

        if self.seeds:
            mine = [seed for seed in self.seeds if seed in done]

            self.completed = len(mine)
        else:
            self.completed = len(done)

            mine = sorted(done)

        beat = read_heartbeat(self.out, self.pid, self.seeds)

        if beat and beat.get("steps_total"):
            heartbeat_pid = beat.get("pid")
            if heartbeat_pid:
                try:
                    self.pid = int(heartbeat_pid)
                except (TypeError, ValueError):
                    pass

            self.run_phase = str(beat.get("phase", "running"))
            self.results_done = int(beat.get("results_done", 0) or 0)
            self.results_total = int(beat.get("results_total", 0) or 0)
            self.run_fraction = min(
                beat["steps_done"] / beat["steps_total"], 1.0
            )
            self.live_chemistry = beat.get("live")

            reported_group = beat.get("boxes_in_group")
            if reported_group is None:
                # Compatibility with runners started before grouped
                # heartbeats carried their size. New Lab soup jobs use the
                # Lab group default unless they explicitly request another.
                reported_group = GROUP_SIZE
                if "--group" in self.arguments:
                    position = self.arguments.index("--group") + 1
                    if position < len(self.arguments):
                        try:
                            reported_group = int(self.arguments[position])
                        except ValueError:
                            pass
            remaining = max(self.runs - self.completed, 1)
            self.inflight_runs = max(
                1, min(int(reported_group or 1), remaining)
            )

            label = beat.get("seed")
            if self.inflight_runs > 1 and self.seeds:
                unfinished = [seed for seed in self.seeds if seed not in done]
                try:
                    first = int(str(label).split("-", 1)[0])
                except (TypeError, ValueError):
                    first = unfinished[0] if unfinished else self.seeds[0]
                if first in done and unfinished:
                    first = unfinished[0]
                try:
                    start = self.seeds.index(first)
                except ValueError:
                    start = 0
                active = self.seeds[start:start + self.inflight_runs]
                if active:
                    label = (
                        str(active[0]) if len(active) == 1
                        else f"{active[0]}-{active[-1]}"
                    )
            self.run_seed = label
        else:
            self.run_fraction = 0.0
            self.run_seed = None
            self.inflight_runs = 1
            self.live_chemistry = None
            self.run_phase = "waiting"
            self.results_done = 0
            self.results_total = 0

        index = read_index(self.out)

        wanted = set(mine)

        self.headlines = [
            f"{entry.get('seed', '?')}: {entry.get('headline', '')}"
            for entry in index
            if not self.seeds or entry.get("seed") in wanted
        ][-4:]

    @property
    def fraction(self):
        if not self.runs:
            return 0.0

        # The run in progress counts as its own fraction of one,
        # so the bar creeps rather than jumping once every few
        # minutes.

        return min(
            (self.completed + self.run_fraction * self.inflight_runs)
            / self.runs,
            1.0,
        )

    @property
    def elapsed(self):
        if self.started is None:
            return 0.0

        end = self.finished or time.time()

        return end - self.started


def clock(seconds):
    seconds = int(max(seconds, 0))

    if seconds < 60:
        return f"{seconds}s"

    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"

    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class Lab(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.root = "runs"
        self.jobs = []
        self.queue_paused = False
        self.concurrency = DEFAULT_CONCURRENCY

        self.setWindowTitle("Chemistry lab")
        self.resize(1500, 900)
        self.setObjectName("chemistryWorkbench")
        self.apply_workbench_style()

        layout = QtWidgets.QVBoxLayout(self)

        self.tabs = QtWidgets.QTabWidget()

        self.tabs.addTab(self.build_run_tab(), "Run")
        self.tabs.addTab(self.build_batches_tab(), "Batches")
        self.tabs.addTab(self.build_results_tab(), "Results")
        self.replay_tab_index = self.tabs.addTab(
            self.build_replay_tab(), "Replay"
        )
        self.tabs.addTab(self.build_molecules_tab(), "Molecules")
        self.tabs.currentChanged.connect(self.on_tab_changed)

        layout.addWidget(self.tabs)

        self.load_queue()
        self.on_mode()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(POLL_MILLISECONDS)

    def apply_workbench_style(self):
        self.setStyleSheet("""
            QWidget#chemistryWorkbench {
                background: #15181d;
                color: #e6eaf0;
                font-family: "Segoe UI";
                font-size: 10pt;
            }
            QTabWidget::pane {
                border: 0;
                border-top: 1px solid #303640;
                background: #15181d;
            }
            QTabBar::tab {
                background: transparent;
                color: #929cab;
                padding: 12px 18px;
                margin-right: 2px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                color: #f4f7fb;
                border-bottom: 3px solid #4ca6ff;
            }
            QFrame#sectionCard {
                background: #20242b;
                border: 1px solid #303640;
                border-radius: 9px;
            }
            QLabel#sectionTitle {
                color: #f6f8fb;
                font-size: 12pt;
                font-weight: 650;
            }
            QLabel#sectionSubtitle, QLabel#muted {
                color: #98a2b1;
            }
            QLabel#eyebrow {
                color: #67b4ff;
                font-size: 9pt;
                font-weight: 700;
            }
            QLabel#heroTitle {
                color: #ffffff;
                font-size: 20pt;
                font-weight: 700;
            }
            QLabel#metricValue {
                color: #ffffff;
                font-size: 15pt;
                font-weight: 650;
            }
            QLabel#statusGood { color: #67d79b; }
            QLabel#statusWarn { color: #f2bd62; }
            QLabel#statusBad { color: #ff7373; }
            QToolButton#sectionToggle {
                border: 0;
                color: #f6f8fb;
                font-size: 11pt;
                font-weight: 650;
                text-align: left;
                padding: 2px;
            }
            QPushButton {
                background: #2a3039;
                border: 1px solid #3a424e;
                border-radius: 6px;
                padding: 7px 12px;
            }
            QPushButton:hover { background: #343c47; }
            QPushButton:disabled { color: #69717d; background: #22262c; }
            QPushButton#primaryAction {
                background: #1685e5;
                border-color: #2898f5;
                color: white;
                font-size: 11pt;
                font-weight: 700;
                padding: 12px 18px;
            }
            QPushButton#primaryAction:hover { background: #2695f3; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit,
            QListWidget, QTableWidget, QTreeWidget {
                background: #191d22;
                border: 1px solid #353c46;
                border-radius: 5px;
                selection-background-color: #176cae;
                padding: 5px;
            }
            QComboBox::drop-down { border: 0; width: 24px; }
            QHeaderView::section {
                background: #242a32;
                color: #aeb7c4;
                border: 0;
                border-bottom: 1px solid #3a414c;
                padding: 7px;
                font-weight: 600;
            }
            QScrollArea, QScrollArea > QWidget,
            QScrollArea > QWidget > QWidget {
                border: 0;
                background: #15181d;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background: #171b20;
                border: 0;
                margin: 0;
            }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background: #414a57;
                border-radius: 5px;
                min-height: 28px;
                min-width: 28px;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                width: 0;
                height: 0;
            }
            QProgressBar {
                background: #171a1f;
                border: 0;
                border-radius: 5px;
                min-height: 10px;
                text-align: center;
            }
            QProgressBar::chunk { background: #3b9cf0; border-radius: 5px; }
            QSplitter::handle { background: #15181d; width: 5px; height: 5px; }
            QToolTip {
                background: #2b313a;
                color: #f0f3f7;
                border: 1px solid #505967;
                padding: 6px;
            }
        """)

    # --------------------------------------------------------
    # Replay tab

    def build_replay_tab(self):
        from lab_replay import ReplayWidget

        self.replay_widget = ReplayWidget()
        return self.replay_widget

    def open_replay(self, path):
        try:
            self.replay_widget.load_path(os.path.abspath(path))
        except Exception as problem:
            QtWidgets.QMessageBox.warning(
                self, "Cannot load trajectory", str(problem)
            )
            return
        self.tabs.setCurrentIndex(self.replay_tab_index)

    # --------------------------------------------------------
    # Run tab

    def build_run_tab(self):
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(12, 12, 12, 12)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        columns = QtWidgets.QHBoxLayout(content)
        columns.setContentsMargins(6, 6, 6, 18)
        columns.setSpacing(14)
        scroll.setWidget(content)
        page_layout.addWidget(scroll)

        designer = QtWidgets.QVBoxLayout()
        designer.setSpacing(12)
        review = QtWidgets.QVBoxLayout()
        review.setSpacing(12)
        columns.addLayout(designer, 3)
        columns.addLayout(review, 2)

        hero = SectionCard("Design an experiment")
        eyebrow = QtWidgets.QLabel("NEW EXPERIMENT")
        eyebrow.setObjectName("eyebrow")
        hero.layout.insertWidget(0, eyebrow)
        self.run_hero_title = QtWidgets.QLabel("Configure a chemistry run")
        self.run_hero_title.setObjectName("heroTitle")
        hero.addWidget(self.run_hero_title)
        self.run_hero_summary = QtWidgets.QLabel("")
        self.run_hero_summary.setObjectName("sectionSubtitle")
        self.run_hero_summary.setWordWrap(True)
        hero.addWidget(self.run_hero_summary)
        designer.addWidget(hero)

        mode = SectionCard(
            "Experiment mode",
            "Start independent seeds, or extend every usable trajectory in an existing batch.",
        )
        mode_buttons = QtWidgets.QHBoxLayout()
        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItems(["New runs", "Continue existing batch"])
        self.mode_box.currentIndexChanged.connect(self.on_mode)
        mode_buttons.addWidget(self.mode_box)
        mode.addLayout(mode_buttons)
        self.source_box = QtWidgets.QComboBox()
        self.source_box.currentTextChanged.connect(self.refresh_existing)
        source_form = QtWidgets.QFormLayout()
        source_form.addRow("Source batch", self.source_box)
        self.source_panel = self.wrap(source_form)
        mode.addWidget(self.source_panel)
        designer.addWidget(mode)

        mixture = SectionCard(
            "Mixture",
            "Choose the starting composition. Atom count and density update with the selected box.",
        )
        mix_row = QtWidgets.QHBoxLayout()
        self.mixture_box = QtWidgets.QComboBox()
        self.reload_mixtures()
        self.mixture_box.currentTextChanged.connect(self.refresh_existing)
        mix_row.addWidget(self.mixture_box, 1)
        self.new_mixture_button = self.button(
            "New mixture", self.on_new_mixture
        )
        self.edit_mixture_button = self.button(
            "Edit selected", self.on_edit_mixture
        )
        mix_row.addWidget(self.new_mixture_button)
        mix_row.addWidget(self.edit_mixture_button)
        mix_row.addWidget(self.button("Reload", self.reload_mixtures))
        mixture.addLayout(mix_row)
        self.atom_note = QtWidgets.QLabel("")
        self.atom_note.setObjectName("sectionSubtitle")
        self.atom_note.setWordWrap(True)
        mixture.addWidget(self.atom_note)
        self.mixture_panel = mixture
        designer.addWidget(mixture)

        core = SectionCard("Core simulation conditions")
        core_form = QtWidgets.QFormLayout()
        core_form.setHorizontalSpacing(18)
        self.box_size = self.choice([12, 15, 17, 19, 21, 24, 28, 34], 19, 1)
        self.box_size.setToolTip(
            "Fixed periodic-cell width selected before the simulation starts. The box does not resize during a run."
        )
        self.picoseconds = self.choice([5, 10, 20, 40, 60, 100], 20, 1)
        self.picoseconds.setToolTip("Physical duration simulated independently for every seed.")
        self.seeds = self.choice([1, 3, 5, 10, 15, 20, 30, 50], 10, 0)
        self.seeds.setToolTip("Number of independent initial random seeds to simulate.")
        self.first_seed = QtWidgets.QComboBox()
        self.first_seed.setEditable(True)
        self.first_seed.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.first_seed.addItems(["continue automatically", "0", "100", "700", "800"])
        self.first_seed.setToolTip(
            "Automatic allocation skips seeds already present in the matching batch. Enter a number to choose the sequence explicitly."
        )
        core_form.addRow("Fixed box size (Å)", self.box_size)
        core_form.addRow("Duration per run (ps)", self.picoseconds)
        core_form.addRow("Independent runs", self.seeds)
        core_form.addRow("First seed", self.first_seed)
        core.addLayout(core_form)
        self.core_panel = core
        designer.addWidget(core)

        thermal = SectionCard(
            "Thermal schedule",
            "Temperature targets guide the thermostat; they do not change the integration timestep.",
        )
        thermal_form = QtWidgets.QFormLayout()
        self.hot_temperature = self.choice([250, 350, 500, 700, 1000, 1500], 500, 0)
        self.hot_until = self.choice([0, 500, 1000, 2000, 4000, 8000], 2000, 0)
        self.cool_temperature = self.choice([100, 250, 350, 500, 700, 1000], 250, 0)
        self.hot_temperature.setToolTip("Thermostat target at the beginning of the simulation.")
        self.hot_until.setToolTip("Time in femtoseconds before switching to the trapping temperature.")
        self.cool_temperature.setToolTip("Thermostat target after the initial warm period, used to trap products.")
        thermal_form.addRow("Starting temperature (K)", self.hot_temperature)
        thermal_form.addRow("Hold until (fs)", self.hot_until)
        thermal_form.addRow("Trap temperature (K)", self.cool_temperature)
        thermal.addLayout(thermal_form)
        self.thermal_preview = pg.PlotWidget()
        self.thermal_preview.setMaximumHeight(145)
        self.thermal_preview.setMouseEnabled(x=False, y=False)
        self.thermal_preview.hideButtons()
        self.thermal_preview.setLabel("left", "target K")
        self.thermal_preview.setLabel("bottom", "simulation time", units="ps")
        self.thermal_curve = self.thermal_preview.plot(
            pen=pg.mkPen("#67b4ff", width=2)
        )
        self.thermal_points = pg.ScatterPlotItem(
            size=7, brush=pg.mkBrush("#67b4ff"), pen=pg.mkPen(None)
        )
        self.thermal_preview.addItem(self.thermal_points)
        thermal.addWidget(self.thermal_preview)
        self.thermal_panel = thermal
        designer.addWidget(thermal)

        lightning = CollapsibleCard(
            "Lightning / energy events",
            "Optional local heating and dissociation events. Set strikes to zero to disable.",
            expanded=False,
        )
        lightning_form = QtWidgets.QFormLayout()
        self.strikes = self.choice([0, 1, 2, 3, 5, 8, 10, 20], 0, 0)
        self.strike_temperature = self.choice([5000, 10000, 20000, 25000, 30000, 50000], 30000, 0)
        self.strike_dissociation = self.choice([0, 0.2, 0.4, 0.6, 1.0, 1.5], 0.6, 2)
        self.first_strike = self.choice([500, 1000, 2500, 5000], 2500, 0)
        self.strike_interval = self.choice([1000, 2000, 3500, 5000, 10000], 3500, 0)
        self.strike_dissociation.setToolTip(
            "Electron-impact proxy controlling direct bond disruption inside the channel; this is separate from temperature."
        )
        lightning_form.addRow("Number of strikes", self.strikes)
        lightning_form.addRow("First strike (fs)", self.first_strike)
        lightning_form.addRow("Interval (fs)", self.strike_interval)
        lightning_form.addRow("Channel temperature (K)", self.strike_temperature)
        lightning_form.addRow("Dissociation setting", self.strike_dissociation)
        lightning.addLayout(lightning_form)
        self.lightning_preview = QtWidgets.QLabel("")
        self.lightning_preview.setObjectName("sectionSubtitle")
        lightning.addWidget(self.lightning_preview)
        designer.addWidget(lightning)

        recording = CollapsibleCard(
            "Recording and crash protection",
            "Recorder v2 observes the simulation without changing its physics.",
            expanded=True,
        )
        recording_form = QtWidgets.QFormLayout()
        self.capture_every = self.choice([10, 20, 40, 80, 200], 40, 0)
        self.capture_every.setToolTip(
            "Simulation steps between ordinary trajectory frames. Smaller values improve temporal resolution and replay smoothness but increase storage and recording work."
        )
        self.save_every = self.choice([0, 1, 2, 5, 10, 20], 5, 0)
        self.save_every.setToolTip(
            "Checkpoint the recording and index this often in picoseconds. This affects crash recovery and disk writes, not trajectory sampling or chemistry. Zero saves only at completion."
        )
        recording_form.addRow("Ordinary capture interval (steps)", self.capture_every)
        recording_form.addRow("Checkpoint interval (ps)", self.save_every)
        recording.addLayout(recording_form)
        self.recording_note = QtWidgets.QLabel("")
        self.recording_note.setObjectName("sectionSubtitle")
        self.recording_note.setWordWrap(True)
        recording.addWidget(self.recording_note)
        designer.addWidget(recording)

        execution = SectionCard("Execution")
        self.grouped = QtWidgets.QCheckBox(f"Grouped GPU — up to {GROUP_SIZE} simulations together")
        self.grouped.setChecked(True)
        self.grouped.setToolTip(
            "Advance several independent periodic boxes together to use the GPU efficiently. Each seed keeps independent state and physics."
        )
        self.grouped.stateChanged.connect(self.refresh_existing)
        execution.addWidget(self.grouped)

        self.physics_box = QtWidgets.QComboBox()
        self.physics_box.addItem(
            "Reactive base (current default)",
            "reactive",
        )
        self.physics_box.addItem(
            "Optimised valence state (experimental)",
            "optimised-valence",
        )
        self.physics_box.setToolTip(
            "Uses the validated factorisable H-state and heavy-valence "
            "engine. This is opt-in; the historical reactive engine remains "
            "the default."
        )
        self.physics_box.currentIndexChanged.connect(
            self.on_physics_changed
        )
        execution.addWidget(self.physics_box)

        self.execution_preview = QtWidgets.QLabel("")
        self.execution_preview.setObjectName("sectionSubtitle")
        execution.addWidget(self.execution_preview)
        designer.addWidget(execution)

        overview = SectionCard("Experiment overview")
        metrics = QtWidgets.QGridLayout()
        self.run_metric_mixture = QtWidgets.QLabel("—")
        self.run_metric_runs = QtWidgets.QLabel("—")
        self.run_metric_duration = QtWidgets.QLabel("—")
        self.run_metric_box = QtWidgets.QLabel("—")
        for column, (label, widget) in enumerate((
            ("MIXTURE", self.run_metric_mixture), ("RUNS", self.run_metric_runs),
            ("DURATION", self.run_metric_duration), ("BOX", self.run_metric_box),
        )):
            caption = QtWidgets.QLabel(label)
            caption.setObjectName("eyebrow")
            widget.setObjectName("metricValue")
            metrics.addWidget(caption, 0, column)
            metrics.addWidget(widget, 1, column)
        overview.addLayout(metrics)
        review.addWidget(overview)

        status = SectionCard("Validation and destination")
        self.validation_label = QtWidgets.QLabel("")
        self.validation_label.setWordWrap(True)
        status.addWidget(self.validation_label)
        self.existing_note = QtWidgets.QLabel("")
        self.existing_note.setWordWrap(True)
        self.existing_note.setObjectName("sectionSubtitle")
        status.addWidget(self.existing_note)
        review.addWidget(status)

        templates = CollapsibleCard("Experiment templates", expanded=True)
        self.template_box = QtWidgets.QComboBox()
        templates.addWidget(self.template_box)
        template_actions = QtWidgets.QHBoxLayout()
        template_actions.addWidget(self.button("Load", self.on_load_template))
        template_actions.addWidget(self.button("Save current…", self.on_save_template))
        template_actions.addWidget(self.button("Delete", self.on_delete_template))
        templates.addLayout(template_actions)
        self.reload_templates()
        review.addWidget(templates)

        advanced = CollapsibleCard("Advanced output and command", expanded=False)
        output_form = QtWidgets.QFormLayout()
        self.folder_name = QtWidgets.QLineEdit()
        self.folder_name.setPlaceholderText("Automatic, based on experiment conditions")
        output_form.addRow("Custom output folder", self.folder_name)
        advanced.addLayout(output_form)
        self.preview = QtWidgets.QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(110)
        self.preview.setStyleSheet("font-family: Consolas, monospace; font-size: 9pt;")
        advanced.addWidget(self.preview)
        review.addWidget(advanced)

        final = SectionCard("Ready to queue")
        self.final_summary = QtWidgets.QLabel("")
        self.final_summary.setWordWrap(True)
        final.addWidget(self.final_summary)
        self.queue_button = self.button("ADD EXPERIMENT TO QUEUE", self.on_queue)
        self.queue_button.setObjectName("primaryAction")
        self.queue_button.setMinimumHeight(46)
        final.addWidget(self.queue_button)
        review.addWidget(final)
        review.addStretch(1)

        for widget in (
            self.box_size, self.picoseconds, self.seeds, self.hot_temperature,
            self.hot_until, self.cool_temperature, self.strikes,
            self.strike_temperature, self.strike_dissociation,
            self.first_strike, self.strike_interval, self.capture_every,
            self.save_every,
        ):
            widget.valueChanged.connect(self.refresh_existing)
        self.first_seed.currentTextChanged.connect(self.refresh_existing)
        self.folder_name.textChanged.connect(self.refresh_existing)

        return page

    # --------------------------------------------------------

    def button(self, label, callback):
        widget = QtWidgets.QPushButton(label)
        widget.clicked.connect(callback)
        return widget

    def divider(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("color: #bbb;")
        return line

    def wrap(self, layout):
        holder = QtWidgets.QWidget()
        holder.setLayout(layout)
        return holder

    def choice(self, options, value, decimals=1):
        return Choice(options, value, decimals)

    # --------------------------------------------------------

    def reload_mixtures(self):
        current = self.mixture_box.currentText()

        self.available = mixtures.all_mixtures()

        self.mixture_box.blockSignals(True)
        self.mixture_box.clear()
        self.mixture_box.addItems(sorted(self.available))

        if current:
            position = self.mixture_box.findText(current)

            if position >= 0:
                self.mixture_box.setCurrentIndex(position)

        self.mixture_box.blockSignals(False)

    def on_physics_changed(self):
        if self.physics_box.currentData() == "optimised-valence":
            self.grouped.setChecked(True)

        self.refresh_existing()

    def on_mode(self):
        continuing = self.mode_box.currentIndex() == 1

        # Everything that describes a fresh box is meaningless
        # when extending one that already exists: the mixture, the
        # size and the starting temperature are already fixed by
        # the runs being continued.

        for widget in (
            self.mixture_box, self.box_size, self.seeds,
            self.first_seed, self.hot_temperature, self.hot_until,
            self.capture_every,
            self.grouped, self.physics_box,
        ):
            widget.setEnabled(not continuing)

        self.source_box.setEnabled(continuing)
        self.source_panel.setVisible(continuing)
        self.mixture_panel.setVisible(not continuing)
        self.core_panel.setVisible(not continuing)
        self.thermal_panel.setVisible(not continuing)

        if continuing:
            self.reload_sources()

        self.refresh_existing()

    def reload_sources(self):
        current = self.source_box.currentText()

        self.source_box.blockSignals(True)
        self.source_box.clear()

        for label, path in find_batches(self.root):
            self.source_box.addItem(label)

        if current:
            position = self.source_box.findText(current)

            if position >= 0:
                self.source_box.setCurrentIndex(position)

        self.source_box.blockSignals(False)

    def conditions(self):
        strikes = int(self.strikes.value())
        adaptive = strikes == 0
        return {
            "physics": self.physics_box.currentData(),
            "mixture": self.mixture_box.currentText(),
            "box": round(self.box_size.value(), 2),
            "picoseconds": round(self.picoseconds.value(), 3),
            "strikes": strikes,
            "strike_temperature": (
                round(self.strike_temperature.value(), 0) if strikes else 0.0
            ),
            "strike_dissociation": (
                round(self.strike_dissociation.value(), 3) if strikes else 0.0
            ),
            "expand_to": 0.0,
            "hot_temperature": round(self.hot_temperature.value(), 0),
            "cool_temperature": round(self.cool_temperature.value(), 0),
            "adaptive_recording": adaptive,
            "adaptive_candidate_fs": 2.0 if adaptive else 0.0,
            "adaptive_pre_event_fs": 100.0 if adaptive else 0.0,
            "adaptive_post_event_fs": 100.0 if adaptive else 0.0,
            "adaptive_energy_jump_ev": 20.0 if adaptive else 0.0,
            "adaptive_close_contact_scale": 0.35 if adaptive else 0.0,
            "adaptive_reaction_window_fs": 20.0 if adaptive else 0.0,
            "adaptive_chemical_context_fs": 10.0 if adaptive else 0.0,
            "compiled_forces": False,
        }

    def refresh_existing(self):
        # Every control on this tab can be typed into, so any of
        # them can be momentarily blank or zero. Nothing here
        # should be able to take the window down mid-edit.

        try:
            self.describe_settings()
        except Exception as problem:
            self.existing_note.setText(
                f"waiting for a valid value\n\n{problem}"
            )
        self.update_experiment_overview()

    def estimated_atom_count(self):
        entry = self.available.get(self.mixture_box.currentText())
        if not entry:
            return 0
        kind, contents = entry
        try:
            return mixtures.atom_count(kind, contents)
        except ValueError:
            # Old manually-authored files remain visible even if they contain
            # a species the current runtime cannot launch.
            return 0

    def update_experiment_overview(self):
        continuing = self.mode_box.currentIndex() == 1
        mixture = self.mixture_box.currentText() or "No mixture"
        runs = max(int(self.seeds.value()), 0)
        duration = max(float(self.picoseconds.value()), 0.0)
        box = max(float(self.box_size.value()), 0.0)
        atoms = self.estimated_atom_count()

        self.run_metric_mixture.setText(
            self.source_box.currentText() if continuing else mixture
        )
        self.run_metric_runs.setText("batch" if continuing else str(runs))
        self.run_metric_duration.setText(f"+{duration:g} ps" if continuing else f"{duration:g} ps")
        self.run_metric_box.setText("existing" if continuing else f"{box:g} Å")

        density = atoms / box ** 3 if atoms and box > 0 else 0.0
        self.run_hero_title.setText(
            "Continue an existing experiment" if continuing
            else (mixture or "Configure a chemistry run")
        )
        self.run_hero_summary.setText(
            f"{runs} independent runs  •  {duration:g} ps each  •  "
            f"{box:g} Å fixed box  •  approximately {atoms} atoms  •  "
            f"density {density:.4f} atoms/Å³"
            if not continuing else
            f"Extend the usable runs in {self.source_box.currentText() or 'the selected batch'} by {duration:g} ps."
        )

        hot = self.hot_temperature.value()
        cool = self.cool_temperature.value()
        hold_ps = self.hot_until.value() / 1000.0
        middle_ps = min(hold_ps * 2.0, duration)
        middle_temperature = (hot + cool) / 2.0
        thermal_x = np.array([
            0.0, min(hold_ps, duration), min(hold_ps, duration),
            middle_ps, middle_ps, duration,
        ])
        thermal_y = np.array([
            hot, hot, middle_temperature,
            middle_temperature, cool, cool,
        ])
        self.thermal_curve.setData(thermal_x, thermal_y)
        self.thermal_points.setData(
            [0.0, min(hold_ps, duration), middle_ps, duration],
            [hot, middle_temperature, cool, cool],
        )

        strike_count = max(int(self.strikes.value()), 0)
        if strike_count:
            moments = [
                (self.first_strike.value() + index * self.strike_interval.value()) / 1000.0
                for index in range(strike_count)
            ]
            visible = ", ".join(f"{value:g}" for value in moments[:6])
            if len(moments) > 6:
                visible += ", …"
            self.lightning_preview.setText(
                f"⚡ at {visible} ps  •  {self.strike_temperature.value():g} K channel"
            )
        else:
            self.lightning_preview.setText("Disabled — no external energy events")

        capture = max(int(self.capture_every.value()), 0)
        if capture <= 20:
            quality = "Detailed"
        elif capture <= 80:
            quality = "Balanced"
        else:
            quality = "Compact"
        self.recording_note.setText(
            f"{quality} ordinary capture ({capture} steps). Recorder v2 also protects reaction and failure context. "
            + (f"Crash-recovery checkpoint every {self.save_every.value():g} ps."
               if self.save_every.value() else "Recording is written only when a run finishes.")
        )

        group = self.group_size() if self.grouped.isChecked() else 1
        pieces = []
        left = runs
        while left > 0:
            take = min(group, left)
            pieces.append(str(take))
            left -= take
        self.execution_preview.setText(
            ("Grouped GPU: " if self.grouped.isChecked() else "Standard: ")
            + (" + ".join(pieces) if pieces else "no runs")
            + " simulations, groups run sequentially"
        )

        checks = []
        valid = True
        if continuing:
            source_ok = bool(self.source_box.currentText())
            checks.append((source_ok, "source batch selected"))
            valid &= source_ok
        else:
            mixture_ok = mixture in self.available
            checks.append((mixture_ok, "valid mixture"))
            checks.append((box > 0, "fixed box size is valid"))
            checks.append((runs > 0, "at least one run requested"))
            checks.append((duration > 0, "simulation duration is valid"))
            valid &= mixture_ok and box > 0 and runs > 0 and duration > 0
            if density > 0.12:
                checks.append((None, f"unusually high starting density ({density:.3f})"))
            if capture >= 200:
                checks.append((None, "very sparse ordinary recording"))

        lines = []
        for state, text in checks:
            lines.append(("✓ " if state else "✕ " if state is False else "⚠ ") + text)
        self.validation_label.setText("\n".join(lines))
        self.queue_button.setEnabled(bool(valid))

        thermal_text = f"{hot:g} K → {cool:g} K"
        self.final_summary.setText(
            f"Extend {self.source_box.currentText()} by {duration:g} ps"
            if continuing else
            f"{runs} × {duration:g} ps  •  {mixture}  •  ~{atoms} atoms  •  "
            f"{thermal_text}  •  "
            + ("Grouped GPU " + (" + ".join(pieces)) if self.grouped.isChecked() else "Standard execution")
        )

    def describe_settings(self):
        name = self.mixture_box.currentText()

        entry = self.available.get(name)

        if entry:
            kind, contents = entry
            try:
                total = mixtures.atom_count(kind, contents)
            except ValueError as problem:
                self.atom_note.setText(f"Cannot launch this mixture: {problem}")
                return

            box = self.box_size.value()

            # The box comes from a box you can type into, so it is
            # briefly empty or zero while being edited. Dividing
            # by it then takes the whole window down.

            if box > 0:
                self.atom_note.setText(
                    f"{total} atoms, density "
                    f"{total / (box ** 3):.4f} atoms per cubic "
                    f"angstrom"
                )
            else:
                self.atom_note.setText(
                    f"{total} atoms, waiting for a box size"
                )

        if self.mode_box.currentIndex() == 1:
            self.describe_continuation()
            return

        label, path, index = matching_folder(
            self.root, self.conditions()
        )

        if index:
            seeds = sorted(
                entry.get("seed", -1) for entry in index
            )

            unstable = sum(
                1 for entry in index
                if entry.get("stable") is False
            )

            self.existing_note.setText(
                f"{len(index)} runs already exist under these\n"
                f"conditions, in {label}.\n\n"
                f"seeds {min(seeds)} to {max(seeds)}\n"
                f"{unstable} of them unstable\n\n"
                f"Next runs will take seeds "
                f"{self.planned_seeds_text(path)}\n"
                f"and write to the same folder."
                + self.group_note()
            )
        else:
            self.existing_note.setText(
                "No runs exist under these conditions yet.\n\n"
                "A new folder will be created, named from the\n"
                "settings unless you give one."
                + self.group_note()
            )

        self.preview.setPlainText(" ".join(self.build_arguments()))

    def describe_continuation(self):
        label = self.source_box.currentText()

        source = os.path.join(self.root, label)

        index = read_index(source)

        if not index:
            self.existing_note.setText(
                "Pick a batch to continue."
            )

            self.preview.setPlainText("")
            return

        unstable = [
            entry for entry in index
            if entry.get("stable") is False
        ]

        target = self.target_folder()

        done = {
            os.path.basename(path)
            for path in glob.glob(os.path.join(target, "run_*.npz"))
        }

        already = sum(
            1 for entry in index
            if entry.get("file") in done
        )

        duration = index[0].get("picoseconds", 0)

        self.existing_note.setText(
            f"{len(index)} runs in {label}, currently\n"
            f"{duration:g} ps each.\n\n"
            f"{len(unstable)} will be skipped as unstable, since\n"
            f"extending a run that blew up only makes more\n"
            f"of a contaminated trajectory.\n\n"
            f"{already} already extended and will be skipped.\n\n"
            f"{len(index) - len(unstable) - already} to run, each\n"
            f"gaining {self.picoseconds.value():g} ps to reach "
            f"{duration + self.picoseconds.value():g} ps.\n\n"
            f"Writing to {os.path.basename(target)}"
        )

        self.preview.setPlainText(" ".join(self.build_arguments()))

    def planned_seeds_text(self, out):
        wanted = int(self.seeds.value())

        start, taken = self.next_free_seed(out)

        planned = []

        cursor = start

        while len(planned) < max(wanted, 1) and len(planned) < 12:
            if cursor not in taken:
                planned.append(cursor)

            cursor += 1

        text = ", ".join(str(value) for value in planned)

        if wanted > len(planned):
            text += ", ..."

        return text

    def group_size(self):
        # One grouped process advances at most GROUP_SIZE boxes.
        # batch_runner.py already processes its groups sequentially, so a
        # A remainder becomes a smaller final group, never a concurrent one.
        return GROUP_SIZE

    def group_note(self):
        if not self.grouped.isChecked():
            return ""

        wanted = int(self.seeds.value())
        size = self.group_size()
        groups = (wanted + size - 1) // size

        return (
            f"\n\nGrouped GPU mode: up to {size} runs are advanced "
            f"together; {groups} group"
            + ("s" if groups != 1 else "")
            + " will run one after another."
        )

    def build_arguments(self):
        if self.mode_box.currentIndex() == 1:
            return self.build_continue_arguments()

        arguments = [
            "--physics", str(self.physics_box.currentData()),
            "--mixture", self.mixture_box.currentText(),
            "--box", f"{self.box_size.value():g}",
            "--ps", f"{self.picoseconds.value():g}",
            "--seeds", str(int(self.seeds.value())),
            "--capture-every", str(int(self.capture_every.value())),
            "--save-every-ps", f"{self.save_every.value():g}",
            "--hot-temperature", f"{self.hot_temperature.value():g}",
            "--hot-until-fs", f"{self.hot_until.value():g}",
            "--cool-temperature",
            f"{self.cool_temperature.value():g}",
        ]

        arguments.append(
            "--adaptive-recording"
            if self.strikes.value() == 0
            else "--legacy-recording"
        )

        arguments += [
            "--group",
            str(self.group_size() if self.grouped.isChecked() else 1),
        ]

        seed_text = self.first_seed.currentText().strip()

        if seed_text and not seed_text[0].isalpha():
            try:
                arguments += ["--first-seed", str(int(float(seed_text)))]
            except ValueError:
                pass

        if self.strikes.value() > 0:
            arguments += [
                "--strikes", str(int(self.strikes.value())),
                "--strike-temperature",
                f"{self.strike_temperature.value():g}",
                "--strike-dissociation",
                f"{self.strike_dissociation.value():g}",
                "--first-strike-fs",
                f"{self.first_strike.value():g}",
                "--strike-interval-fs",
                f"{self.strike_interval.value():g}",
            ]

        if self.folder_name.text().strip():
            arguments += [
                "--out",
                os.path.join(
                    self.root, self.folder_name.text().strip()
                ),
            ]

        return arguments

    def build_continue_arguments(self):
        label = self.source_box.currentText()

        source = os.path.join(self.root, label)

        arguments = [
            "--continue-from", source,
            "--ps", f"{self.picoseconds.value():g}",
            "--cool-temperature",
            f"{self.cool_temperature.value():g}",
            "--capture-every", str(int(self.capture_every.value())),
            "--save-every-ps", f"{self.save_every.value():g}",
            "--legacy-recording",
        ]

        if self.strikes.value() > 0:
            arguments += [
                "--strikes", str(int(self.strikes.value())),
                "--strike-temperature",
                f"{self.strike_temperature.value():g}",
                "--strike-dissociation",
                f"{self.strike_dissociation.value():g}",
                "--first-strike-fs",
                f"{self.first_strike.value():g}",
                "--strike-interval-fs",
                f"{self.strike_interval.value():g}",
            ]

        if self.folder_name.text().strip():
            arguments += [
                "--out",
                os.path.join(
                    self.root, self.folder_name.text().strip()
                ),
            ]

        return arguments

    def target_folder(self):
        if self.mode_box.currentIndex() == 1:
            if self.folder_name.text().strip():
                return os.path.join(
                    self.root, self.folder_name.text().strip()
                )

            source = os.path.join(
                self.root, self.source_box.currentText()
            )

            return source + f"_plus{self.picoseconds.value():g}ps"


        if self.folder_name.text().strip():
            return os.path.join(
                self.root, self.folder_name.text().strip()
            )

        label, path, index = matching_folder(
            self.root, self.conditions()
        )

        if path:
            return path

        # Mirrors the naming in batch_runner so the panel can
        # watch the right folder before the batch has made it.

        safe = self.mixture_box.currentText().strip().replace("+", "plus")
        if self.physics_box.currentData() != "reactive":
            safe += "_optimised_valence"
        safe = "-".join(safe.replace("_", " ").split())

        parts = [
            safe,
            f"{self.box_size.value():g}A",
            f"{self.picoseconds.value():g}ps",
        ]

        if self.strikes.value() > 0:
            parts.append(
                f"lightning{int(self.strikes.value())}x"
                f"{self.strike_temperature.value() / 1000:g}kK"
            )

        if self.cool_temperature.value() != 250.0:
            parts.append(f"cool{self.cool_temperature.value():g}K")

        if self.strikes.value() == 0:
            parts.append("v2")
        else:
            parts.append("v1")

        return os.path.join(self.root, "_".join(parts))

    def next_free_seed(self, out):
        seeds = set()

        for entry in read_index(out):
            if entry.get("seed") is not None:
                seeds.add(int(entry["seed"]))

        entries = os.path.join(out, "entries")

        if os.path.isdir(entries):
            for name in os.listdir(entries):
                if name.startswith("seed_") and name.endswith(".json"):
                    try:
                        seeds.add(int(name[5:-5]))
                    except ValueError:
                        continue

        text = self.first_seed.currentText().strip()

        if text and not text[0].isalpha():
            try:
                start = int(float(text))
            except ValueError:
                start = 0
        else:
            # From the lowest seed not present, so gaps left by
            # crashed or interrupted runs get filled rather than
            # skipped over.

            start = min(seeds) if seeds else 0

        while start in seeds:
            start += 1

        return start, seeds

    def on_queue(self):
        out = self.target_folder()

        if self.mode_box.currentIndex() == 1:
            source = read_index(
                os.path.join(self.root, self.source_box.currentText())
            )

            total = sum(
                1 for entry in source
                if entry.get("stable") is not False
            )

            job = Job(
                name=os.path.basename(out),
                arguments=self.build_arguments(),
                out=out,
                runs=total,
            )
        else:
            wanted = int(self.seeds.value())

            start, taken = self.next_free_seed(out)

            planned = []

            cursor = start

            while len(planned) < wanted:
                if cursor not in taken:
                    planned.append(cursor)

                cursor += 1

            job = Job(
                name=os.path.basename(out),
                arguments=self.build_arguments(),
                out=out,
                runs=wanted,
                seeds=planned,
            )

        self.jobs.append(job)

        self.save_queue()
        self.draw_jobs()

        self.tabs.setCurrentIndex(1)

    # --------------------------------------------------------
    # Molecules tab

    def build_molecules_tab(self):
        page = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(page)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(10)

        eyebrow = QtWidgets.QLabel("REACTION EXPERIMENTS")
        eyebrow.setObjectName("eyebrow")
        outer.addWidget(eyebrow)
        title = QtWidgets.QLabel("Build, run, and reuse chemistry")
        title.setObjectName("heroTitle")
        outer.addWidget(title)
        subtitle = QtWidgets.QLabel(
            "Choose what collides, how encounters are sampled, then inspect or reuse products."
        )
        subtitle.setObjectName("sectionSubtitle")
        outer.addWidget(subtitle)

        font = QtGui.QFont("Consolas")
        font.setPointSize(10)

        # The Molecules page is deliberately built from splitters rather than
        # fixed nested layouts. Characterisation output grows much faster than
        # the controls above it, so the useful amount of screen space depends
        # on what is being inspected. Every major pane can now be resized by
        # dragging its separator.

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        outer.addWidget(main_splitter, 1)

        # ----------------------------------------------------
        # Left: selected molecule + controlled experiments

        research_widget = QtWidgets.QWidget()
        research_widget.setObjectName("sectionCard")
        research_layout = QtWidgets.QVBoxLayout(research_widget)
        research_layout.setContentsMargins(14, 14, 14, 14)

        research_splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Vertical
        )
        research_splitter.setChildrenCollapsible(False)
        research_layout.addWidget(research_splitter)

        # Selected molecule. This was a full text panel of the same height as
        # the others, and most of what it said -- atoms, appearances, longest
        # observed life -- is repeated by the library detail on the right. So
        # it is a dropdown and one line now, and the height it was using goes
        # to the results.
        #
        # What it had that the library detail does not: the controlled trial
        # totals and the natural formation examples. Those are folded into
        # the line and its tooltip rather than dropped.
        selected_panel = QtWidgets.QWidget()
        selected_layout = QtWidgets.QVBoxLayout(selected_panel)
        selected_layout.setContentsMargins(0, 0, 0, 0)
        selected_layout.setSpacing(4)

        picker = QtWidgets.QHBoxLayout()

        title = QtWidgets.QLabel("reactant A")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        picker.addWidget(title)

        # Editable with a completer rather than a separate search field.
        # Nearly four hundred species is not something to scroll, and typing
        # into the box itself is one control rather than two: "CH3" or "041"
        # both narrow it, since the completer matches anywhere in the entry
        # rather than only at the start.
        self.character_molecule = QtWidgets.QComboBox()
        self.character_molecule.setEditable(True)
        self.character_molecule.setInsertPolicy(
            QtWidgets.QComboBox.InsertPolicy.NoInsert
        )

        completer = self.character_molecule.completer()
        completer.setCompletionMode(
            QtWidgets.QCompleter.CompletionMode.PopupCompletion
        )
        completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        completer.setCaseSensitivity(
            QtCore.Qt.CaseSensitivity.CaseInsensitive
        )

        self.character_molecule.lineEdit().setPlaceholderText(
            "atom or saved molecule"
        )
        self.character_molecule.currentIndexChanged.connect(
            self.on_character_molecule_changed
        )
        picker.addWidget(self.character_molecule, stretch=1)
        selected_layout.addLayout(picker)

        self.character_selected = QtWidgets.QLabel()
        self.character_selected.setTextFormat(
            QtCore.Qt.TextFormat.PlainText
        )
        self.character_selected.setStyleSheet("color: #999;")
        self.character_selected.setWordWrap(False)
        selected_layout.addWidget(self.character_selected)

        research_splitter.addWidget(selected_panel)

        # New controlled experiment controls.
        #
        # The form has grown past the height the splitter usually gives it,
        # so the run buttons could end up below the fold with no way to reach
        # them. The controls live inside a scroll area; the panel itself stays
        # a plain widget so the splitter keeps behaving as before.
        test_panel = QtWidgets.QWidget()
        test_panel_layout = QtWidgets.QVBoxLayout(test_panel)
        test_panel_layout.setContentsMargins(0, 0, 0, 0)

        test_scroll = QtWidgets.QScrollArea()
        test_scroll.setWidgetResizable(True)
        test_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        test_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        test_panel_layout.addWidget(test_scroll)

        test_contents = QtWidgets.QWidget()
        test_scroll.setWidget(test_contents)

        test_layout = QtWidgets.QVBoxLayout(test_contents)
        test_layout.setContentsMargins(0, 4, 8, 4)

        new_title = QtWidgets.QLabel("Experiment definition")
        new_title.setStyleSheet("font-weight: bold;")
        test_layout.addWidget(new_title)

        # Two columns rather than one. Eleven full width rows of dropdowns
        # took a third of the tab for values that are mostly a few characters
        # wide, and pushed the results pane -- the thing the tab exists to
        # show -- into a strip. Split so that what defines the experiment is
        # on the left and what defines the run is on the right.
        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(18)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        run_form = QtWidgets.QFormLayout()
        run_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        run_form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.character_test = QtWidgets.QComboBox()
        self.character_test.addItems(["A only", "A + B collision"])
        self.character_test.currentIndexChanged.connect(
            self.on_character_test_mode
        )
        form.addRow("encounter", self.character_test)

        self.character_physics = QtWidgets.QComboBox()
        self.character_physics.addItem("standard", "standard")
        self.character_physics.addItem(
            "high fidelity (experimental)", "high_fidelity"
        )
        self.character_physics.setToolTip(
            "Characterisation only. Standard uses the same reactive potential "
            "as discovery. High fidelity adds experimental competitive "
            "valence-state mixing for transferring hydrogen; normal soup runs "
            "are not changed."
        )
        form.addRow("physics", self.character_physics)

        self.character_partner = QtWidgets.QComboBox()
        self.character_partner.setEnabled(False)
        form.addRow("reactant B", self.character_partner)

        self.character_sampling = QtWidgets.QComboBox()
        self.character_sampling.addItem("controlled / targeted", "targeted")
        self.character_sampling.addItem("random orientations", "random_orientation")
        self.character_sampling.addItem("targeted + randomized", "targeted_random")
        self.character_sampling.setEnabled(False)
        form.addRow("sampling", self.character_sampling)

        self.character_target_atom = QtWidgets.QComboBox()
        self.character_target_atom.setEnabled(False)
        self.character_target_atom.setToolTip(
            "Controls initial geometry only; reactive physics decides the outcome."
        )
        form.addRow("target on A", self.character_target_atom)

        self.character_impact_target = QtWidgets.QComboBox()
        self.character_impact_target.addItem("random / COM", "com")
        self.character_impact_target.addItem("carbon", "carbon")
        self.character_impact_target.addItem("oxygen", "oxygen")
        self.character_impact_target.addItem("hydrogen", "hydrogen")
        self.character_impact_target.setEnabled(False)
        self.character_impact_target.setToolTip(
            "Choose where the incoming partner trajectory is aimed. Random / "
            "COM preserves the existing baseline. Element choices aim the "
            "partner through a randomly selected atom of that element. Targeted "
            "mode now chooses a clear line-of-sight attack direction so another "
            "atom is not deliberately sitting in the beam; reactive physics still "
            "decides what happens."
        )
        self.character_impact_target.hide()

        self.character_approach = self.choice(
            [0.5, 1, 1.5, 2, 3, 5], 2, 1
        )
        self.character_approach.setEnabled(False)
        self.character_approach.setToolTip(
            "Directed centre-of-mass approach speed relative to the normal "
            "thermal RMS atomic speed already generated for that box. This "
            "changes the collision energy without prescribing any reaction."
        )
        form.addRow("approach (thermal x)", self.character_approach)

        self.character_start_gap = self.choice(
            [1.5, 2, 2.5, 3, 4], 2.5, 1
        )
        self.character_start_gap.setEnabled(False)
        self.character_start_gap.setToolTip(
            "Initial surface-to-surface clearance before the two randomly "
            "oriented reactants are aimed directly at one another."
        )
        form.addRow("start gap (A)", self.character_start_gap)

        self.character_temperature = self.choice(
            [100, 200, 250, 300, 500, 750, 1000, 1500, 2000], 250, 0
        )
        run_form.addRow("temperature (K)", self.character_temperature)

        self.character_duration = self.choice(
            [1, 2, 5, 10, 20, 40], 10, 1
        )
        run_form.addRow("duration (ps)", self.character_duration)

        # A collision is over in tens of femtoseconds, so the default soup
        # cadence of 40 steps (10 fs a frame) captures the whole reaction in
        # about four frames. 4 steps is 1 fs a frame at the 0.25 fs time step,
        # which resolves the C-H stretch that gates the transfer. Larger
        # values are still useful for long runs where only the outcome
        # matters and the trajectory file would otherwise be enormous.
        self.character_capture = self.choice(
            [2, 4, 10, 20, 40, 80], 4, 1
        )
        self.character_capture.setToolTip(
            "Frames are written every N integration steps. At the 0.25 fs "
            "time step, 4 gives a frame every femtosecond. The encounter "
            "itself lasts roughly 20 to 40 fs, so coarse values record the "
            "outcome but not the mechanism."
        )
        run_form.addRow("capture every N steps", self.character_capture)

        self.character_box = self.choice(
            [10, 12, 15, 19, 24, 30], 12, 1
        )
        run_form.addRow("box (A)", self.character_box)

        self.character_repeats = self.choice(
            [1, 8, 16, 24, 32, 40, 48, 64, 80, 96, 128], 8, 0
        )
        run_form.addRow("runs", self.character_repeats)

        group_rule = QtWidgets.QLabel(
            "1, or exact multiples of 8; one 8-box group at a time"
        )
        group_rule.setStyleSheet("color: #555;")
        run_form.addRow("grouping", group_rule)

        columns.addLayout(form, 1)
        columns.addLayout(run_form, 1)
        test_layout.addLayout(columns)

        row = QtWidgets.QHBoxLayout()
        self.character_run_button = self.button(
            "Run experiment", self.on_character_run
        )
        self.character_all_button = self.button(
            "Test all", self.on_character_not_ready
        )
        self.character_run_button.setEnabled(False)
        self.character_all_button.setEnabled(False)
        self.character_run_button.setToolTip(
            "Queue controlled isolated or partner-collision repeats through the "
            "standard discovery physics or the optional characterisation-only "
            "high-fidelity mode, using the same Recorder path. Repeat count must "
            "be 1 or a multiple of 8. Multi-repeat groups "
            "contain exactly eight boxes and groups run strictly one after another."
        )
        row.addWidget(self.character_run_button)
        row.addWidget(self.character_all_button)
        test_layout.addLayout(row)
        test_layout.addStretch(1)

        research_splitter.addWidget(test_panel)

        # Characterisation results. This is the pane that benefits most from
        # extra room, so it receives the remaining height by default.
        results_panel = QtWidgets.QWidget()
        results_layout = QtWidgets.QVBoxLayout(results_panel)
        results_layout.setContentsMargins(0, 4, 0, 0)

        results_header = QtWidgets.QHBoxLayout()
        results_title = QtWidgets.QLabel("experiment results")
        results_title.setStyleSheet("font-weight: bold;")
        results_header.addWidget(results_title)
        results_header.addStretch(1)
        results_header.addWidget(
            self.button("Refresh tests", self.on_refresh_character_results)
        )
        results_layout.addLayout(results_header)

        self.character_experiment_list = QtWidgets.QListWidget()
        self.character_experiment_list.setFont(font)
        self.character_experiment_list.currentRowChanged.connect(
            self.on_character_experiment_changed
        )
        results_layout.addWidget(self.character_experiment_list, stretch=2)

        result_splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Horizontal
        )
        result_splitter.setChildrenCollapsible(False)

        self.character_result_summary = QtWidgets.QPlainTextEdit()
        self.character_result_summary.setReadOnly(True)
        self.character_result_summary.setLineWrapMode(
            QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap
        )
        self.character_result_summary.setFont(font)
        self.character_result_summary.setMinimumWidth(220)
        result_splitter.addWidget(self.character_result_summary)

        run_widget = QtWidgets.QWidget()
        run_side = QtWidgets.QVBoxLayout(run_widget)
        run_side.setContentsMargins(0, 0, 0, 0)

        self.character_runs = QtWidgets.QTableWidget()
        self.character_runs.setColumnCount(7)
        self.character_runs.setHorizontalHeaderLabels([
            "seed", "outcome", "contact fs", "closest A",
            "stable", "final K", "final species"
        ])
        self.character_runs.horizontalHeader().setStretchLastSection(True)
        self.character_runs.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.character_runs.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.character_runs.verticalHeader().setVisible(False)
        self.character_runs.itemSelectionChanged.connect(
            self.on_character_run_selection_changed
        )
        self.character_runs.cellDoubleClicked.connect(
            self.on_open_character_run_replay
        )
        run_side.addWidget(self.character_runs)

        self.character_open_result = self.button(
            "Open run in Replay",
            self.on_open_character_run_replay,
        )
        self.character_open_result.setEnabled(False)
        run_side.addWidget(self.character_open_result)

        self.character_save_product = self.button(
            "Save final product to library...",
            self.on_save_character_product,
        )
        self.character_save_product.setEnabled(False)
        self.character_save_product.setToolTip(
            "Choose a connected component actually present in the final recorded frame. No bond order, charge, radical label, or reaction is invented."
        )
        run_side.addWidget(self.character_save_product)

        result_splitter.addWidget(run_widget)
        result_splitter.setStretchFactor(0, 2)
        result_splitter.setStretchFactor(1, 5)
        result_splitter.setSizes([330, 830])
        results_layout.addWidget(result_splitter, stretch=5)

        research_splitter.addWidget(results_panel)
        research_splitter.setStretchFactor(0, 0)
        research_splitter.setStretchFactor(1, 1)
        research_splitter.setStretchFactor(2, 6)

        # The molecule panel is a dropdown and a line now rather than a text
        # pane, and the form is two columns rather than eleven rows, so both
        # give their height to the results -- which is the thing the tab
        # exists to show and was previously a four row strip.
        research_splitter.setSizes([60, 210, 575])

        main_splitter.addWidget(research_widget)

        # ----------------------------------------------------
        # Right: automatic discovery/library panel

        library_widget = QtWidgets.QWidget()
        library_widget.setObjectName("sectionCard")
        library = QtWidgets.QVBoxLayout(library_widget)
        library.setContentsMargins(14, 14, 14, 14)

        title = QtWidgets.QLabel("Discovered structures")
        title.setObjectName("sectionTitle")
        library.addWidget(title)

        self.molecule_search = QtWidgets.QLineEdit()
        self.molecule_search.setPlaceholderText(
            "Search by formula, structure ID, or elements…"
        )
        self.molecule_search.textChanged.connect(
            self.filter_molecule_library
        )
        library.addWidget(self.molecule_search)

        self.molecule_scan_all_button = self.button(
            "Scan recordings", self.on_scan_recordings
        )
        self.molecule_scan_all_button.setToolTip(
            "Scan only recordings/tails not already logged in the molecule "
            "scan manifest. Legacy identity-less and unstable runs are ignored."
        )
        library.addWidget(self.molecule_scan_all_button)

        self.molecule_export_button = self.button(
            "Export library...", self.on_export_molecule_library
        )
        self.molecule_export_button.setToolTip(
            "Create one zip containing every stored molecule geometry, species "
            "record, formation event and the scan manifest. Source trajectories "
            "are referenced but are not copied into the zip."
        )
        library.addWidget(self.molecule_export_button)

        self.molecule_scan_status = QtWidgets.QLabel("")
        self.molecule_scan_status.setWordWrap(True)
        self.molecule_scan_status.setStyleSheet(
            "font-family: Consolas, monospace; color: #555;"
        )
        library.addWidget(self.molecule_scan_status)

        library_splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Vertical
        )
        library_splitter.setChildrenCollapsible(False)

        self.molecule_library_list = QtWidgets.QListWidget()
        self.molecule_library_list.setFont(font)
        self.molecule_library_list.currentRowChanged.connect(
            self.on_library_molecule_changed
        )
        library_splitter.addWidget(self.molecule_library_list)

        from lab_renderer import MolecularScene
        self.molecule_preview = MolecularScene()
        self.molecule_preview.setMinimumHeight(190)
        library_splitter.addWidget(self.molecule_preview)

        self.molecule_details = QtWidgets.QPlainTextEdit()
        self.molecule_details.setReadOnly(True)
        self.molecule_details.setLineWrapMode(
            QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth
        )
        self.molecule_details.setFont(font)
        library_splitter.addWidget(self.molecule_details)

        library_splitter.setStretchFactor(0, 2)
        library_splitter.setStretchFactor(1, 3)
        library_splitter.setStretchFactor(2, 2)
        library_splitter.setSizes([230, 390, 220])
        library.addWidget(library_splitter, stretch=1)

        qm_panel = QtWidgets.QGroupBox("QM structure check")
        qm_layout = QtWidgets.QVBoxLayout(qm_panel)
        qm_layout.setSpacing(5)

        qm_state = QtWidgets.QHBoxLayout()
        self.qm_charge = QtWidgets.QLineEdit()
        self.qm_charge.setPlaceholderText("required")
        self.qm_charge.setMaximumWidth(75)
        self.qm_multiplicity = QtWidgets.QLineEdit()
        self.qm_multiplicity.setPlaceholderText("required")
        self.qm_multiplicity.setMaximumWidth(75)
        self.qm_method = QtWidgets.QLineEdit(qm_validator.DEFAULT_METHOD)
        self.qm_method.setMaximumWidth(110)
        self.qm_basis = QtWidgets.QLineEdit(qm_validator.DEFAULT_BASIS)
        self.qm_basis.setMaximumWidth(130)
        for label, widget in (
            ("charge", self.qm_charge),
            ("multiplicity", self.qm_multiplicity),
            ("method", self.qm_method),
            ("basis", self.qm_basis),
        ):
            qm_state.addWidget(QtWidgets.QLabel(label))
            qm_state.addWidget(widget)
        qm_state.addStretch(1)
        qm_layout.addLayout(qm_state)

        qm_actions = QtWidgets.QHBoxLayout()
        self.qm_run_button = self.button(
            "Run QM Structure Check", self.on_run_qm_structure_check
        )
        self.qm_run_button.setEnabled(False)
        self.qm_geometry_choice = QtWidgets.QComboBox()
        self.qm_geometry_choice.addItems([
            "ChemistryModel geometry", "QM-optimised geometry"
        ])
        self.qm_geometry_choice.setEnabled(False)
        self.qm_geometry_choice.currentIndexChanged.connect(
            self.on_qm_geometry_choice_changed
        )
        qm_actions.addWidget(self.qm_run_button)
        qm_actions.addWidget(self.qm_geometry_choice)
        qm_actions.addStretch(1)
        qm_layout.addLayout(qm_actions)

        self.qm_status = QtWidgets.QLabel(
            "UNTESTED — charge and multiplicity are not inferred."
        )
        self.qm_status.setWordWrap(True)
        self.qm_status.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.qm_status.setStyleSheet(
            "font-family: Consolas, monospace; color: #9aa7b8;"
        )
        qm_layout.addWidget(self.qm_status)
        library.addWidget(qm_panel)

        self.molecule_open_source = self.button(
            "Open formation source in Replay", self.on_molecule_open_source
        )
        self.molecule_open_source.setEnabled(False)
        library.addWidget(self.molecule_open_source)

        main_splitter.addWidget(library_widget)
        main_splitter.setStretchFactor(0, 5)
        main_splitter.setStretchFactor(1, 2)
        main_splitter.setSizes([980, 660])

        # Remember the user's splitter positions. The defaults above make the
        # results pane larger than before, but after the first drag the page
        # comes back exactly as the user left it on the next Lab launch.
        settings = QtCore.QSettings("ChemistryModel", "Lab")
        splitters = [
            (main_splitter, "molecules/main_splitter"),
            (research_splitter, "molecules/research_splitter"),
            (result_splitter, "molecules/result_splitter"),
            (library_splitter, "molecules/library_splitter"),
        ]

        for splitter, key in splitters:
            state = settings.value(key)
            if state is not None:
                splitter.restoreState(state)

            splitter.splitterMoved.connect(
                lambda position, index, s=splitter, k=key:
                    settings.setValue(k, s.saveState())
            )

        self.library_molecules = []
        self.qm_process = None
        self.qm_selected_record = None
        self.character_experiments_data = []
        self.character_run_entries = []
        self.reload_molecule_library()
        self.refresh_molecule_scan_status()

        return page

    def on_tab_changed(self, index):
        if self.tabs.tabText(index) != "Molecules":
            return

        if not hasattr(self, "character_molecule"):
            return

        self.reload_characterisation_results(
            self.character_molecule.currentData()
        )

    def refresh_molecule_scan_status(self):
        try:
            status = molecule_scanner.manifest_summary()
            species = len(molecule_store.list_molecules())
        except Exception as problem:
            self.molecule_scan_status.setText(
                f"scan history unavailable: {problem}"
            )
            return

        self.molecule_scan_status.setText(
            f"{species} stored species\n"
            f"{status['scanned']} recordings scanned, "
            f"{status['legacy']} legacy blocked, "
            f"{status['unstable']} unstable skipped"
            + (f", {status['errors']} errors" if status["errors"] else "")
        )

    def on_scan_recordings(self):
        self.molecule_scan_all_button.setEnabled(False)
        self.molecule_scan_status.setText("finding recordings...")
        QtWidgets.QApplication.processEvents()

        def progress(update):
            stage = update.get("stage")

            if stage == "recording":
                self.molecule_scan_status.setText(
                    f"recording {update.get('number', '?')}/"
                    f"{update.get('total', '?')}\n"
                    f"{update.get('recording', '')}"
                )
            elif stage == "frames":
                self.molecule_scan_status.setText(
                    f"scanning {update.get('recording', '')}\n"
                    f"frame {update.get('frame', '?')}/"
                    f"{update.get('frames', '?')}"
                )

            QtWidgets.QApplication.processEvents()

        try:
            summary = molecule_scanner.scan_recordings(
                runs_root=self.root,
                progress=progress,
            )
        except Exception as problem:
            QtWidgets.QMessageBox.warning(
                self, "Molecule scan failed", str(problem)
            )
            self.molecule_scan_all_button.setEnabled(True)
            self.refresh_molecule_scan_status()
            return

        self.reload_molecule_library()
        self.molecule_scan_all_button.setEnabled(True)

        lines = [
            f"{summary['recordings_found']} recordings found",
            f"{summary['scanned']} scanned, "
            f"{summary['unchanged']} already current",
            f"{summary['frames_counted']} new frames read",
            f"{summary['new_species']} new species, "
            f"{summary['formation_events']} formation/reaction events",
        ]

        skipped = []
        if summary["legacy"]:
            skipped.append(f"{summary['legacy']} legacy")
        if summary["unstable"]:
            skipped.append(f"{summary['unstable']} unstable")
        if summary["empty"]:
            skipped.append(f"{summary['empty']} empty")

        if skipped:
            lines.append("skipped: " + ", ".join(skipped))

        if summary["errors"]:
            lines.append(f"errors: {len(summary['errors'])}")

        self.molecule_scan_status.setText("\n".join(lines))

        if summary["errors"]:
            QtWidgets.QMessageBox.warning(
                self,
                "Scan completed with errors",
                "\n".join(summary["errors"][:8])
                + ("\n..." if len(summary["errors"]) > 8 else ""),
            )

    def on_export_molecule_library(self):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        default_name = os.path.join(
            self.root, f"molecule_library_{stamp}.zip"
        )

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export molecule library",
            default_name,
            "Zip archive (*.zip)",
        )

        if not path:
            return

        self.molecule_export_button.setEnabled(False)
        QtWidgets.QApplication.processEvents()

        try:
            result = molecule_store.export_library(path)
        except Exception as problem:
            QtWidgets.QMessageBox.warning(
                self, "Molecule export failed", str(problem)
            )
            self.molecule_export_button.setEnabled(True)
            return

        self.molecule_export_button.setEnabled(True)

        QtWidgets.QMessageBox.information(
            self,
            "Molecule library exported",
            f"{result['species']} species and {result['events']} formation/reaction "
            f"events exported to:\n\n{result['path']}",
        )

    def reload_molecule_library(self, select_id=None):
        current_id = select_id

        if current_id is None and hasattr(self, "character_molecule"):
            current_id = self.character_molecule.currentData()

        if current_id is None and hasattr(self, "molecule_library_list"):
            row = self.molecule_library_list.currentRow()
            if 0 <= row < len(self.library_molecules):
                current_id = self.library_molecules[row].get("id")

        self.library_molecules = molecule_store.list_molecules()

        self.molecule_library_list.blockSignals(True)
        self.molecule_library_list.clear()

        selected_row = -1

        for row, item in enumerate(self.library_molecules):
            stats = item.get("stats", {})
            self.molecule_library_list.addItem(
                f"{item.get('id', '?')}  "
                f"{item.get('formula', '?'):<12} "
                f"instances {stats.get('appearances', 0):>4}  "
                f"formed {stats.get('formations', 0):>4}"
            )

            if item.get("id") == current_id:
                selected_row = row

        self.molecule_library_list.blockSignals(False)

        # Main test dropdown. Editable boxes keep whatever the user typed
        # when the model is replaced, so the text is restored from the
        # selection afterwards rather than left as a stale fragment.
        self.character_molecule.blockSignals(True)
        self.character_molecule.clear()

        for symbol in ("H", "C", "N", "O"):
            self.character_molecule.addItem(f"{symbol} atom", f"atom:{symbol}")

        for item in self.library_molecules:
            self.character_molecule.addItem(
                f"{item.get('id', '?')} - {item.get('formula', '?')}",
                item.get("id"),
            )

        if current_id:
            for index in range(self.character_molecule.count()):
                if self.character_molecule.itemData(index) == current_id:
                    self.character_molecule.setCurrentIndex(index)
                    break

        # Put the selected entry's text back, since an editable box keeps
        # whatever was typed when its contents are replaced and would
        # otherwise show a half-finished filter next to a different molecule.
        if self.character_molecule.currentIndex() >= 0:
            self.character_molecule.setEditText(
                self.character_molecule.currentText()
            )

        self.character_molecule.blockSignals(False)

        # Partner choices keep elemental feedstock available even before it
        # happens to exist as a stored connected species, then add every SP.
        partner_id = self.character_partner.currentData()
        self.character_partner.clear()
        self.character_partner.addItem("H atom", "atom:H")
        self.character_partner.addItem("O atom", "atom:O")
        self.character_partner.addItem("N atom", "atom:N")
        self.character_partner.addItem("C atom", "atom:C")

        for item in self.library_molecules:
            self.character_partner.addItem(
                f"{item.get('id', '?')} - {item.get('formula', '?')}",
                item.get("id"),
            )

        if partner_id:
            for index in range(self.character_partner.count()):
                if self.character_partner.itemData(index) == partner_id:
                    self.character_partner.setCurrentIndex(index)
                    break

        self.character_molecule.setEnabled(True)

        if selected_row >= 0:
            self.molecule_library_list.setCurrentRow(selected_row)
        elif self.library_molecules:
            self.molecule_library_list.setCurrentRow(0)
        else:
            self.molecule_details.setPlainText(
                "No stored molecules yet.\n\n"
                "Press Scan recordings. Only stable recordings with verified "
                "per-frame atom identity are allowed to contribute."
            )
            self.character_selected.setText(
                "Library empty; elemental H, C, N and O are ready to use."
            )
            self.molecule_open_source.setEnabled(False)

        self.on_character_molecule_changed(
            self.character_molecule.currentIndex()
        )

        if hasattr(self, "molecule_search"):
            self.filter_molecule_library(self.molecule_search.text())

    def filter_molecule_library(self, text):
        wanted = "".join(text.lower().split())
        for row, item in enumerate(self.library_molecules):
            searchable = " ".join((
                str(item.get("id", "")),
                str(item.get("formula", "")),
                " ".join(item.get("elements", []) or []),
            )).lower()
            match = not wanted or wanted in "".join(searchable.split())
            self.molecule_library_list.item(row).setHidden(not match)

    def on_character_test_mode(self, index):
        partner_mode = index == 1
        self.character_partner.setEnabled(partner_mode)
        self.character_sampling.setEnabled(partner_mode)
        self.character_target_atom.setEnabled(partner_mode)
        self.character_approach.setEnabled(partner_mode)
        self.character_start_gap.setEnabled(partner_mode)
        self.character_run_button.setEnabled(
            self.character_molecule.currentIndex() >= 0
        )

    def characterisation_folder(self, molecule_id, test, partner_id,
                               temperature, duration, box,
                               approach=None, start_gap=None,
                               impact_target="com", physics_mode="standard",
                               sampling_mode="random_orientation",
                               target_atom=None):
        physics_suffix = (
            (
                f"_hf_htransfer_v{HF_MODEL_REVISION}"
                if str(physics_mode) == "high_fidelity" else ""
            )
        )

        if test == "with_partner":
            safe_partner = str(partner_id or "unknown").replace(":", "-")
            safe_molecule = str(molecule_id).replace(":", "-")
            target_suffix = (
                f"_{str(sampling_mode)}"
                + ("" if target_atom is None else f"_target{int(target_atom)+1}")
            )
            return os.path.join(
                CHARACTERISATION_ROOT,
                f"{safe_molecule}_with_{safe_partner}_{temperature:g}K_"
                f"{duration:g}ps_box{box:g}_a{float(approach):g}_g{float(start_gap):g}"
                f"{target_suffix}{physics_suffix}",
            )

        return os.path.join(
            CHARACTERISATION_ROOT,
            f"{str(molecule_id).replace(':', '-')}_isolated_{temperature:g}K_"
            f"{duration:g}ps_box{box:g}{physics_suffix}",
        )

    def characterisation_seeds(self, out, wanted):
        taken = entry_seeds(out)
        planned = []
        cursor = 0

        while len(planned) < wanted:
            if cursor not in taken:
                planned.append(cursor)
            cursor += 1

        return planned

    def _character_partner_payload(self, partner_id):
        if not partner_id:
            raise ValueError("pick a collision partner")

        if str(partner_id).startswith("atom:"):
            symbol = str(partner_id).split(":", 1)[1]
            return {
                "id": partner_id,
                "formula": symbol,
                "symbols": [symbol],
                "positions": np.zeros((1, 3), dtype=float),
            }

        return molecule_store.load_molecule(partner_id)

    def _character_reactant_payload(self, reactant_id):
        return self._character_partner_payload(reactant_id)

    def _reload_character_targets(self, reactant):
        self.character_target_atom.clear()
        for index, symbol in enumerate(reactant.get("symbols", [])):
            self.character_target_atom.addItem(f"{symbol} #{index + 1}", index)

    def _required_character_box(self, molecule, partner=None, start_gap=2.5):
        first = np.asarray(molecule.get("positions", []), dtype=float)
        if len(first):
            first = first - np.mean(first, axis=0)
        first_radius = (
            float(np.max(np.linalg.norm(first, axis=1))) if len(first) else 0.0
        )

        if partner is None:
            if len(first) > 1:
                return float(np.max(np.ptp(first, axis=0)) + 6.0)
            return 6.0

        second = np.asarray(partner.get("positions", []), dtype=float)
        if len(second):
            second = second - np.mean(second, axis=0)
        second_radius = (
            float(np.max(np.linalg.norm(second, axis=1))) if len(second) else 0.0
        )

        centre_distance = first_radius + second_radius + float(start_gap)
        return float(
            centre_distance + 2.0 * max(first_radius, second_radius) + 4.0
        )

    def on_character_run(self):
        molecule_id = self.character_molecule.currentData()

        if not molecule_id:
            return

        partner_mode = self.character_test.currentIndex() == 1
        test = "with_partner" if partner_mode else "isolated"
        partner_id = self.character_partner.currentData() if partner_mode else None
        physics_mode = str(self.character_physics.currentData() or "standard")

        temperature = float(self.character_temperature.value())
        duration = float(self.character_duration.value())
        capture_every = int(self.character_capture.value())
        box = float(self.character_box.value())
        repeats = max(1, int(self.character_repeats.value()))
        approach = float(self.character_approach.value())
        start_gap = float(self.character_start_gap.value())
        sampling_mode = str(
            self.character_sampling.currentData() or "random_orientation"
        )
        target_atom = self.character_target_atom.currentData() if partner_mode else None
        impact_target = "com"

        try:
            molecule = self._character_reactant_payload(molecule_id)
            partner = (
                self._character_partner_payload(partner_id)
                if partner_mode else None
            )
        except Exception as problem:
            QtWidgets.QMessageBox.warning(
                self, "Cannot load characterisation input", str(problem)
            )
            return

        if partner_mode and impact_target != "com":
            target_symbol = {
                "carbon": "C",
                "oxygen": "O",
                "hydrogen": "H",
            }.get(impact_target)
            if target_symbol not in list(molecule.get("symbols", [])):
                QtWidgets.QMessageBox.warning(
                    self,
                    "No target atom",
                    f"{molecule_id} contains no {target_symbol or impact_target} "
                    "atom to aim at.",
                )
                return

        required = self._required_character_box(
            molecule, partner=partner, start_gap=start_gap
        )

        if box + 1e-9 < required:
            description = molecule_id
            if partner is not None:
                description += f" + {partner.get('id', partner_id)}"
            QtWidgets.QMessageBox.warning(
                self,
                "Box too small",
                f"{description} needs about {required:.1f} A or more for this "
                f"test. Pick a larger box than {box:g} A.",
            )
            return

        out = self.characterisation_folder(
            molecule_id,
            test,
            partner_id,
            temperature,
            duration,
            box,
            approach=approach,
            start_gap=start_gap,
            impact_target=impact_target,
            physics_mode=physics_mode,
            sampling_mode=sampling_mode,
            target_atom=target_atom,
        )
        planned = self.characterisation_seeds(out, repeats)

        arguments = [
            "--molecule", str(molecule_id),
            "--test", test,
            "--physics", physics_mode,
            "--temperature", f"{temperature:g}",
            "--ps", f"{duration:g}",
            "--capture-every", str(int(capture_every)),
            "--box", f"{box:g}",
            "--repeats", str(repeats),
            "--group", str(GROUP_SIZE),
            "--seed-list", ",".join(str(seed) for seed in planned),
            "--out", out,
        ]

        if partner_mode:
            arguments += [
                "--partner", str(partner_id),
                "--approach-factor", f"{approach:g}",
                "--start-gap", f"{start_gap:g}",
                "--impact-target", impact_target,
                "--sampling-mode", sampling_mode,
            ]
            if target_atom is not None and sampling_mode != "random_orientation":
                arguments += ["--target-atom-a", str(int(target_atom))]

        partner_text = ""
        if partner is not None:
            partner_text = f" + {partner.get('formula', partner_id)}"
        aim_text = f" {sampling_mode.replace('_', '-')}" if partner_mode else ""
        physics_text = " HF" if physics_mode == "high_fidelity" else ""

        job = Job(
            name=(
                f"CHAR {molecule_id} {molecule.get('formula', '')}"
                f"{partner_text}{aim_text}{physics_text} {temperature:g}K"
            ),
            arguments=arguments,
            out=out,
            runs=repeats,
            seeds=planned,
            runner="characterisation_runner.py",
        )

        self.jobs.append(job)
        self.save_queue()
        self.draw_jobs()
        self.tabs.setCurrentIndex(1)

    def on_character_molecule_changed(self, index):
        if index < 0 or index >= self.character_molecule.count():
            self.character_selected.setText(
                "Scan recordings to populate the test dropdown."
            )
            self.character_run_button.setEnabled(False)
            self.reload_characterisation_results(None)
            return

        self.character_run_button.setEnabled(True)

        molecule_id = self.character_molecule.itemData(index)
        try:
            reactant = self._character_reactant_payload(molecule_id)
            self._reload_character_targets(reactant)
        except Exception as problem:
            self.character_selected.setText(str(problem))
            self.character_run_button.setEnabled(False)
            return
        selected = next(
            (item for item in self.library_molecules
             if item.get("id") == molecule_id),
            None,
        )

        if selected is None and str(molecule_id).startswith("atom:"):
            symbol = str(molecule_id).split(":", 1)[1]
            self.character_selected.setText(
                f"Single {symbol} atom · no saved-library entry required"
            )
            self.reload_characterisation_results(molecule_id)
            return
        if selected is None:
            return

        totals = self.reload_characterisation_results(molecule_id)
        stats = selected.get("stats", {})

        # One line, the things that change as you move through the library:
        # how much natural evidence there is and how much testing has been
        # done. Everything static about the molecule is in the library detail
        # on the right and does not need saying twice.
        summary = (
            f"{selected.get('atoms', '?')} atoms "
            f"({selected.get('heavy_atoms', '?')} heavy)"
            f"   ·   natural: {stats.get('appearances', 0)} episodes, "
            f"{stats.get('formations', 0)} formations, "
            f"{stats.get('runs_seen', 0)} runs"
            f"   ·   tested: {totals.get('trials', 0)} trials across "
            f"{totals.get('experiments', 0)} settings"
        )

        outcomes = totals.get("outcomes", {})
        if outcomes:
            summary += "   ·   " + ", ".join(
                f"{name} {count}" for name, count in outcomes.items()
            )

        self.character_selected.setText(summary)

        # The formation examples are worth keeping but not worth a panel:
        # they are read once when a molecule is new to you, not watched.
        events = molecule_store.formation_events_for_species(molecule_id, limit=3)
        if events:
            detail = ["natural formation examples"]
            for event in events:
                before = self._format_event_side(event.get("reactants", []))
                after = self._format_event_side(event.get("products", []))
                detail.append(
                    f"  {event.get('temperature_K', 0):.0f} K  "
                    f"{before} -> {after}"
                )
            self.character_selected.setToolTip("\n".join(detail))
        else:
            self.character_selected.setToolTip("")

        for row, item in enumerate(self.library_molecules):
            if item.get("id") == molecule_id:
                if self.molecule_library_list.currentRow() != row:
                    self.molecule_library_list.setCurrentRow(row)
                break

    def on_refresh_character_results(self):
        molecule_id = self.character_molecule.currentData()
        totals = self.reload_characterisation_results(molecule_id)

        # Rebuild the selected molecule summary too, but block the combo signal
        # so refreshing results cannot unexpectedly bounce the library row.
        index = self.character_molecule.currentIndex()
        if index >= 0:
            self.on_character_molecule_changed(index)

        return totals

    def reload_characterisation_results(self, molecule_id=None):
        if not hasattr(self, "character_experiment_list"):
            return {"experiments": 0, "trials": 0, "stable_trials": 0, "outcomes": {}}

        current_folder = None
        current_row = self.character_experiment_list.currentRow()
        if 0 <= current_row < len(self.character_experiments_data):
            current_folder = self.character_experiments_data[current_row].get("folder")

        if not molecule_id:
            self.character_experiments_data = []
        else:
            self.character_experiments_data = character_results.list_experiments(
                molecule_id=molecule_id,
                root=CHARACTERISATION_ROOT,
            )

        totals = character_results.aggregate(self.character_experiments_data)

        self.character_experiment_list.blockSignals(True)
        self.character_experiment_list.clear()
        selected_row = -1

        for row, experiment in enumerate(self.character_experiments_data):
            self.character_experiment_list.addItem(
                character_results.experiment_label(experiment)
            )
            if experiment.get("folder") == current_folder:
                selected_row = row

        self.character_experiment_list.blockSignals(False)

        if self.character_experiments_data:
            self.character_experiment_list.setCurrentRow(
                selected_row if selected_row >= 0 else 0
            )
        else:
            self.character_result_summary.setPlainText(
                "No controlled tests recorded for this molecule yet."
                if molecule_id else
                "Select a molecule to see its controlled tests."
            )
            self.character_runs.setRowCount(0)
            self.character_run_entries = []
            self.character_open_result.setEnabled(False)

        return totals

    def on_character_experiment_changed(self, row):
        if row < 0 or row >= len(self.character_experiments_data):
            self.character_result_summary.setPlainText(
                "No characterisation experiment selected."
            )
            self.character_runs.setRowCount(0)
            self.character_run_entries = []
            self.character_open_result.setEnabled(False)
            return

        experiment = self.character_experiments_data[row]
        self.character_result_summary.setPlainText(
            "\n".join(character_results.experiment_summary_lines(experiment))
        )

        entries = list(experiment.get("entries", []))
        self.character_run_entries = entries
        self.character_runs.setSortingEnabled(False)
        self.character_runs.setRowCount(len(entries))

        for result_row, entry in enumerate(entries):
            data = character_results.run_row(entry)
            contact = data.get("contact_fs")
            if contact is None:
                contact_text = "-"
            else:
                contact_text = f"{float(contact):.1f}"
                if data.get("confirmed_contact"):
                    contact_text += " *"

            closest = data.get("closest_A")
            closest_text = (
                "-" if closest is None else f"{float(closest):.3f}"
            )

            values = [
                data.get("seed", "?"),
                data.get("outcome", "?"),
                contact_text,
                closest_text,
                "yes" if data.get("stable") else "NO",
                (
                    f"{float(data['final_temperature']):.0f}"
                    if data.get("final_temperature") is not None else "?"
                ),
                data.get("final_species", "?"),
            ]

            tooltip = data.get("contact_tooltip", "")
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                if column in (0, 1, 6):
                    item.setFont(QtGui.QFont("Consolas", 9))
                if tooltip and column in (2, 3):
                    item.setToolTip(tooltip)
                self.character_runs.setItem(result_row, column, item)

        self.character_runs.resizeColumnsToContents()
        self.character_runs.horizontalHeader().setStretchLastSection(True)

        if entries:
            self.character_runs.selectRow(0)
        else:
            self.character_open_result.setEnabled(False)

    def on_character_run_selection_changed(self):
        experiment_row = self.character_experiment_list.currentRow()
        run_row = self.character_runs.currentRow()

        if (
            experiment_row < 0
            or experiment_row >= len(self.character_experiments_data)
            or run_row < 0
            or run_row >= len(self.character_run_entries)
        ):
            self.character_open_result.setEnabled(False)
            self.character_save_product.setEnabled(False)
            return

        path = character_results.recording_path(
            self.character_experiments_data[experiment_row],
            self.character_run_entries[run_row],
        )
        self.character_open_result.setEnabled(bool(path))
        self.character_save_product.setEnabled(bool(path))

    def on_save_character_product(self):
        experiment_row = self.character_experiment_list.currentRow()
        run_row = self.character_runs.currentRow()
        if not (0 <= experiment_row < len(self.character_experiments_data)
                and 0 <= run_row < len(self.character_run_entries)):
            return
        path = character_results.recording_path(
            self.character_experiments_data[experiment_row],
            self.character_run_entries[run_row],
        )
        if not path:
            return
        try:
            from recorder import Recorder
            recorder = Recorder.load(path)
            components = molecule_store.molecules_at(recorder, -1)
            if not components:
                raise ValueError("the final recorded frame has no connected component")
            choices = [
                f"{item['formula']} · {item['atoms']} atoms · component {item['component'] + 1}"
                for item in components
            ]
            selected_index = 0
            if len(choices) > 1:
                choice, accepted = QtWidgets.QInputDialog.getItem(
                    self, "Choose product", "Connected product:", choices, 0, False
                )
                if not accepted:
                    return
                selected_index = choices.index(choice)
            saved = molecule_store.save_component(
                path, -1, components[selected_index], recorder=recorder,
                note="Saved from a Molecules-tab reaction experiment",
            )
            self.reload_molecule_library(select_id=saved["id"])
            QtWidgets.QMessageBox.information(
                self, "Product saved",
                f"Saved {saved['formula']} as {saved['id']}. It is now available as reactant A or B.",
            )
        except Exception as problem:
            QtWidgets.QMessageBox.warning(self, "Cannot save product", str(problem))

    def on_open_character_run_replay(self, *unused):
        experiment_row = self.character_experiment_list.currentRow()
        run_row = self.character_runs.currentRow()

        if (
            experiment_row < 0
            or experiment_row >= len(self.character_experiments_data)
            or run_row < 0
            or run_row >= len(self.character_run_entries)
        ):
            return

        path = character_results.recording_path(
            self.character_experiments_data[experiment_row],
            self.character_run_entries[run_row],
        )

        if not path:
            return

        self.open_replay(path)

    def on_character_not_ready(self):
        QtWidgets.QMessageBox.information(
            self,
            "Test all",
            "Run test is now wired for isolated stored molecules. Test all stays "
            "locked until that reconstruction path has been checked on a real "
            "species, then partner/collision sweeps can be layered on top."
        )

    def _format_event_side(self, items):
        parts = []

        for item in items:
            name = item.get("id") or item.get("formula", "?")
            count = int(item.get("count", 1))
            parts.append(name + (f" x{count}" if count > 1 else ""))

        return " + ".join(parts) if parts else "?"

    def on_library_molecule_changed(self, row):
        if row < 0 or row >= len(self.library_molecules):
            self.molecule_details.setPlainText("")
            self.molecule_open_source.setEnabled(False)
            if hasattr(self, "qm_run_button"):
                self.qm_run_button.setEnabled(False)
                self.qm_geometry_choice.setEnabled(False)
                self.qm_status.setText("UNTESTED — select a recorded molecule.")
            return

        item = self.library_molecules[row]
        source = item.get("source", {})
        stats = item.get("stats", {})
        sources = item.get("sources", {})

        try:
            molecule = molecule_store.load_molecule(item)
            positions = np.asarray(molecule.get("positions", []), dtype=float)
            extent = np.ptp(positions, axis=0) if len(positions) else np.ones(3)
            box_size = max(float(np.max(extent)) + 6.0, 8.0)
            shifted = positions - np.min(positions, axis=0) + (
                box_size - extent
            ) / 2.0
            bonds = np.asarray(molecule.get("bonds", []), dtype=int).reshape(-1, 2)
            first = bonds[:, 0] if len(bonds) else np.array([], dtype=int)
            second = bonds[:, 1] if len(bonds) else np.array([], dtype=int)
            self.molecule_preview.set_state(
                shifted, molecule.get("symbols", []), box_size,
                (first, second),
            )
            self.molecule_preview.recentre()
        except Exception:
            pass

        lines = [
            f"{item.get('id', '?')}  {item.get('formula', '?')}",
            "=" * 52,
            f"atoms                  {item.get('atoms', '?')}",
            f"heavy atoms            {item.get('heavy_atoms', '?')}",
            f"appearance episodes    {stats.get('appearances', 0)}",
            f"formation products     {stats.get('formations', 0)}",
            f"molecule-frame samples {stats.get('observations', 0)}",
            f"recordings seen in     {stats.get('runs_seen', 0)}",
            f"longest observed life  {stats.get('longest_observed_lifetime_fs', 0):.0f} fs",
            "",
            "first stored geometry",
            "-" * 52,
            f"recording              {source.get('recording', '?')}",
            f"seed                   {source.get('seed', '?')}",
            f"mixture                {source.get('mixture', '?')}",
            f"time                   {source.get('time_fs', '?')} fs",
            f"temperature            {source.get('temperature_K', '?')} K",
        ]

        if sources:
            lines += ["", "recording sightings", "-" * 52]

            ordered = sorted(
                sources.values(),
                key=lambda value: (
                    str(value.get("batch", "")),
                    value.get("seed") if value.get("seed") is not None else -1,
                )
            )

            for seen in ordered[:8]:
                lines.append(
                    f"seed {str(seen.get('seed', '?')):<6} "
                    f"instances {seen.get('appearances', 0):>4}  "
                    f"formed {seen.get('formations', 0):>4}  "
                    f"{seen.get('batch', '')}"
                )

            if len(ordered) > 8:
                lines.append(f"... and {len(ordered) - 8} more recordings")

        events = molecule_store.formation_events_for_species(
            item.get("id"), limit=6
        )

        if events:
            lines += ["", "recent formation examples", "-" * 52]

            for event in events:
                before = self._format_event_side(event.get("reactants", []))
                after = self._format_event_side(event.get("products", []))
                lines.append(
                    f"{event.get('time_fs', 0):>8.0f} fs  "
                    f"{event.get('temperature_K', 0):>6.0f} K  "
                    f"{before} -> {after}"
                )

                formed = event.get("formed_bonds", [])
                broken = event.get("broken_bonds", [])
                if formed or broken:
                    lines.append(
                        f"           bonds +{len(formed)} / -{len(broken)}  "
                        f"local {event.get('local_environment_elements', {})}"
                    )

        lines += [
            "",
            "identity note",
            "-" * 52,
            "SP identity currently uses the confirmed element-labelled bond",
            "graph. Bond order/charge/radical/stereo are not yet part of it.",
        ]

        self.molecule_details.setPlainText("\n".join(lines))

        recording = source.get("recording")
        self.molecule_open_source.setEnabled(
            bool(recording and os.path.exists(recording))
        )
        self.refresh_qm_structure_check(item.get("id"))

        molecule_id = item.get("id")
        for index in range(self.character_molecule.count()):
            if self.character_molecule.itemData(index) == molecule_id:
                if self.character_molecule.currentIndex() != index:
                    self.character_molecule.blockSignals(True)
                    self.character_molecule.setCurrentIndex(index)
                    self.character_molecule.blockSignals(False)
                    self.on_character_molecule_changed(index)
                break

    def _selected_library_molecule_id(self):
        row = self.molecule_library_list.currentRow()
        if 0 <= row < len(self.library_molecules):
            return self.library_molecules[row].get("id")
        return None

    def refresh_qm_structure_check(self, molecule_id=None):
        molecule_id = molecule_id or self._selected_library_molecule_id()
        running = self.qm_process is not None
        self.qm_run_button.setEnabled(bool(molecule_id) and not running)
        self.qm_selected_record = None
        self.qm_geometry_choice.setEnabled(False)
        if not molecule_id:
            self.qm_status.setText("UNTESTED — select a recorded molecule.")
            return
        records = qm_validator.list_validations(molecule_id)
        if not records:
            self.qm_status.setText(
                "UNTESTED — enter charge and multiplicity; these are not guessed "
                "from the current graph-only species identity."
            )
            return
        record = records[0]
        self.qm_selected_record = record
        status = str(record.get("status", "untested")).upper()
        if status == "RUNNING":
            self.qm_status.setText(
                f"QM RUNNING  {record.get('method')}/{record.get('basis')}  "
                f"charge {record.get('charge')}  multiplicity {record.get('multiplicity')}"
            )
            return
        if status == "FAILED":
            self.qm_status.setText(
                f"QM FAILED  {record.get('method')}/{record.get('basis')}\n"
                f"{record.get('error', 'No error detail was recorded.')}"
            )
            return
        single = record.get("single_point", {})
        optimisation = record.get("optimisation", {})
        comparison = record.get("comparison", {})
        connectivity = (
            "preserved" if comparison.get("connectivity_preserved")
            else "CHANGED"
        )
        self.qm_status.setText(
            f"QM COMPLETE  {record.get('method')}/{record.get('basis')}  "
            f"q={record.get('charge')} mult={record.get('multiplicity')}\n"
            f"exact-geometry force: max {single.get('max_force_eV_per_A', float('nan')):.4g}, "
            f"RMS {single.get('rms_force_eV_per_A', float('nan')):.4g} eV/A  |  "
            f"relaxation {optimisation.get('relaxation_energy_eV', float('nan')):.5g} eV\n"
            f"RMSD: all {comparison.get('all_atom_rmsd_A', float('nan')):.4g} A, "
            f"heavy {comparison.get('heavy_atom_rmsd_A') if comparison.get('heavy_atom_rmsd_A') is not None else 'n/a'} A  |  "
            f"connectivity {connectivity}"
        )
        self.qm_geometry_choice.setEnabled(True)
        self.on_qm_geometry_choice_changed(self.qm_geometry_choice.currentIndex())

    def on_run_qm_structure_check(self):
        molecule_id = self._selected_library_molecule_id()
        if not molecule_id or self.qm_process is not None:
            return
        try:
            charge = int(self.qm_charge.text().strip())
            multiplicity = int(self.qm_multiplicity.text().strip())
            if multiplicity < 1:
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Electronic state required",
                "Enter an integer charge and a positive integer multiplicity. "
                "Molecule Lab will not guess an electronic state from the current graph-only identity."
            )
            return
        method = self.qm_method.text().strip()
        basis = self.qm_basis.text().strip()
        if not method or not basis:
            QtWidgets.QMessageBox.warning(
                self, "QM method required", "Enter both a method and basis set."
            )
            return
        process = QtCore.QProcess(self)
        process.setWorkingDirectory(os.path.abspath("."))
        process.setProcessChannelMode(
            QtCore.QProcess.ProcessChannelMode.MergedChannels
        )
        process.finished.connect(self.on_qm_structure_check_finished)
        process.errorOccurred.connect(self.on_qm_structure_check_process_error)
        self.qm_process = process
        self.qm_run_button.setEnabled(False)
        self.qm_geometry_choice.setEnabled(False)
        self.qm_status.setText(
            f"QM RUNNING  {method}/{basis}  q={charge} mult={multiplicity}\n"
            "Evaluating the exact ChemistryModel geometry, then optimising from it…"
        )
        script = os.path.abspath(qm_validator.__file__)
        try:
            worker_python = qm_validator.psi4_worker_python()
        except ValueError as problem:
            self.qm_process = None
            process.deleteLater()
            self.qm_run_button.setEnabled(True)
            self.qm_status.setText(f"QM FAILED — {problem}")
            return
        process.start(worker_python, [
            script, molecule_id,
            "--charge", str(charge),
            "--multiplicity", str(multiplicity),
            "--method", method,
            "--basis", basis,
        ])

    def on_qm_structure_check_finished(self, exit_code, exit_status):
        process = self.qm_process
        output = ""
        if process is not None:
            output = bytes(process.readAllStandardOutput()).decode(
                "utf-8", errors="replace"
            ).strip()
            process.deleteLater()
        self.qm_process = None
        self.refresh_qm_structure_check()
        if int(exit_code) != 0 and not qm_validator.list_validations(
            self._selected_library_molecule_id() or ""
        ):
            self.qm_status.setText(
                "QM FAILED before a validation record could be created.\n" +
                (output[-1000:] or "No process output was available.")
            )

    def on_qm_structure_check_process_error(self, error):
        if self.qm_process is None:
            return
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            self.qm_status.setText(
                "QM FAILED — the Python/Psi4 worker could not be started."
            )
            self.qm_process.deleteLater()
            self.qm_process = None
            self.qm_run_button.setEnabled(
                bool(self._selected_library_molecule_id())
            )

    def on_qm_geometry_choice_changed(self, index):
        record = self.qm_selected_record
        if not record or record.get("status") != "complete":
            return
        try:
            payload = qm_validator.load_geometries(record)
            positions = np.asarray(
                payload[
                    "optimised_coordinates_A" if index == 1
                    else "original_coordinates_A"
                ], dtype=float,
            )
            symbols = [str(value) for value in payload["symbols"]]
            if index == 1:
                bonds = qm_validator.inferred_bonds(symbols, positions)
            else:
                bonds = np.asarray(payload["original_bonds"], dtype=int).reshape(-1, 2)
            extent = np.ptp(positions, axis=0) if len(positions) else np.ones(3)
            box_size = max(float(np.max(extent)) + 6.0, 8.0)
            shifted = positions - np.min(positions, axis=0) + (box_size - extent) / 2.0
            first = bonds[:, 0] if len(bonds) else np.asarray([], dtype=int)
            second = bonds[:, 1] if len(bonds) else np.asarray([], dtype=int)
            self.molecule_preview.set_state(
                shifted, symbols, box_size, (first, second)
            )
            self.molecule_preview.recentre()
        except Exception as problem:
            self.qm_status.setText(
                self.qm_status.text() + f"\nCannot load stored QM geometry: {problem}"
            )

    def on_molecule_open_source(self):
        row = self.molecule_library_list.currentRow()

        if row < 0 or row >= len(self.library_molecules):
            return

        source = self.library_molecules[row].get("source", {})
        path = source.get("recording")

        if not path or not os.path.exists(path):
            return

        self.open_replay(path)

    # --------------------------------------------------------
    # Batches tab

    def build_batches_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        header = QtWidgets.QHBoxLayout()
        title_stack = QtWidgets.QVBoxLayout()
        eyebrow = QtWidgets.QLabel("SIMULATION CONTROL ROOM")
        eyebrow.setObjectName("eyebrow")
        title_stack.addWidget(eyebrow)
        title = QtWidgets.QLabel("Batches and live runs")
        title.setObjectName("heroTitle")
        title_stack.addWidget(title)
        header.addLayout(title_stack)
        header.addStretch(1)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Concurrent jobs"))

        self.concurrency_box = QtWidgets.QSpinBox()
        self.concurrency_box.setRange(1, 8)
        self.concurrency_box.setValue(DEFAULT_CONCURRENCY)
        self.concurrency_box.valueChanged.connect(
            self.on_concurrency
        )
        controls.addWidget(self.concurrency_box)

        self.pause_button = self.button(
            "Pause queue", self.on_pause
        )
        controls.addWidget(self.pause_button)

        controls.addWidget(
            self.button("Stop selected", self.on_stop_selected)
        )
        controls.addWidget(self.button("Stop all", self.on_stop_all))
        controls.addWidget(
            self.button("Clear finished", self.on_clear_finished)
        )

        controls.addStretch(1)
        header.addLayout(controls)
        layout.addLayout(header)

        summary = SectionCard("Queue overview")
        summary_grid = QtWidgets.QGridLayout()
        self.batch_summary_labels = {}
        for column, key in enumerate(("RUNNING", "QUEUED", "COMPLETED", "FAILED")):
            caption = QtWidgets.QLabel(key)
            caption.setObjectName("eyebrow")
            value = QtWidgets.QLabel("0")
            value.setObjectName("metricValue")
            summary_grid.addWidget(caption, 0, column)
            summary_grid.addWidget(value, 1, column)
            self.batch_summary_labels[key.lower()] = value
        summary.addLayout(summary_grid)
        self.queue_state_label = QtWidgets.QLabel("Queue running")
        self.queue_state_label.setObjectName("statusGood")
        summary.addWidget(self.queue_state_label)
        layout.addWidget(summary)

        self.jobs_table = QtWidgets.QTableWidget()
        self.jobs_table.setColumnCount(7)
        self.jobs_table.setHorizontalHeaderLabels([
            "batch", "state", "active group", "overall",
            "finished", "elapsed", "estimated left",
        ])
        jobs_header = self.jobs_table.horizontalHeader()
        jobs_header.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        for column in range(1, 7):
            jobs_header.setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
        self.jobs_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.jobs_table.verticalHeader().setVisible(False)
        self.jobs_table.itemSelectionChanged.connect(
            self.draw_live_batch_details
        )

        splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Vertical
        )
        jobs_card = SectionCard(
            "Experiment queue",
            "Select a batch to inspect its active seed group and current chemistry below.",
        )
        jobs_card.addWidget(self.jobs_table)
        jobs_card.setMaximumHeight(270)
        splitter.addWidget(jobs_card)

        live_panel = SectionCard(
            "Selected batch",
            "Live chemistry snapshots are intentionally sparse and observational.",
        )
        live_layout = live_panel.layout

        self.live_batch_title = QtWidgets.QLabel(
            "Live chemistry — select a running batch"
        )
        title_font = self.live_batch_title.font()
        title_font.setPointSize(title_font.pointSize() + 2)
        title_font.setBold(True)
        self.live_batch_title.setFont(title_font)
        live_layout.addWidget(self.live_batch_title)

        self.live_batch_summary = QtWidgets.QLabel(
            "Live atom information will appear at the next progress update."
        )
        self.live_batch_summary.setWordWrap(True)
        live_layout.addWidget(self.live_batch_summary)

        actions = QtWidgets.QHBoxLayout()
        self.batch_view_results = self.button(
            "View results", self.on_batch_view_results
        )
        self.batch_open_replay = self.button(
            "Open latest replay", self.on_batch_open_replay
        )
        self.batch_continue = self.button(
            "Continue batch", self.on_batch_continue
        )
        for action in (
            self.batch_view_results, self.batch_open_replay,
            self.batch_continue,
        ):
            action.setEnabled(False)
            actions.addWidget(action)
        actions.addStretch(1)
        live_layout.addLayout(actions)

        self.live_seed_table = QtWidgets.QTableWidget()
        self.live_seed_table.setColumnCount(8)
        self.live_seed_table.setHorizontalHeaderLabels([
            "seed", "largest structure", "atoms", "heavy atoms",
            "carbon", "heavy molecules", "C-C bonds", "temperature K",
        ])
        live_header = self.live_seed_table.horizontalHeader()
        live_header.setSectionsClickable(True)
        live_header.sectionClicked.connect(
            self.on_live_seed_header_clicked
        )
        live_header.setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        for column in (0, 2, 3, 4, 5, 6, 7):
            live_header.setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
        self.live_seed_table.verticalHeader().setVisible(False)
        self.live_seed_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.live_seed_display_rows = []
        self.live_seed_sort_column = None
        self.live_seed_sort_stage = 0
        live_layout.addWidget(self.live_seed_table, stretch=1)

        splitter.addWidget(live_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([250, 650])

        layout.addWidget(splitter, stretch=1)

        return page

    def selected_batch_job(self):
        selected = self.jobs_table.selectionModel().selectedRows()
        if not selected:
            return None
        row = selected[0].row()
        return self.jobs[row] if 0 <= row < len(self.jobs) else None

    def on_batch_view_results(self):
        job = self.selected_batch_job()
        if not job:
            return
        self.tabs.setCurrentIndex(2)
        self.reload_results()
        label = os.path.basename(os.path.normpath(job.out))
        position = self.results_batch.findText(label)
        if position >= 0:
            self.results_batch.setCurrentIndex(position)

    def on_batch_open_replay(self):
        job = self.selected_batch_job()
        if not job:
            return
        index = read_index(job.out)
        candidates = []
        for entry in index:
            name = entry.get("file")
            if name:
                path = name if os.path.isabs(name) else os.path.join(job.out, name)
                if os.path.exists(path):
                    candidates.append(path)
        if candidates:
            self.open_replay(candidates[-1])

    def on_batch_continue(self):
        job = self.selected_batch_job()
        if not job:
            return
        self.tabs.setCurrentIndex(0)
        self.mode_box.setCurrentIndex(1)
        label = os.path.basename(os.path.normpath(job.out))
        position = self.source_box.findText(label)
        if position >= 0:
            self.source_box.setCurrentIndex(position)

    def on_concurrency(self, value):
        self.concurrency = int(value)

    def on_pause(self):
        self.queue_paused = not self.queue_paused

        self.pause_button.setText(
            "Resume queue" if self.queue_paused else "Pause queue"
        )

    def stop_job(self, job):
        if job.process is None and job.pid:
            # Reattached from a previous session: no handle, but
            # the process id is enough.

            running.stop(job.pid)
            running.remove_lock(job.out)

            job.state = "stopped"
            job.finished = time.time()
            return

        if job.process is None or job.process.poll() is not None:
            job.state = "stopped"
            return

        # Terminate first so the batch can finish writing whatever
        # index it may be partway through, and only force it if it
        # ignores that.

        job.process.terminate()

        try:
            job.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            job.process.kill()

        job.state = "stopped"
        job.finished = time.time()

    def on_stop_selected(self):
        for index in {
            item.row() for item in self.jobs_table.selectedItems()
        }:
            if 0 <= index < len(self.jobs):
                job = self.jobs[index]

                if job.state in ("running", "queued"):
                    self.stop_job(job)

        self.save_queue()
        self.draw_jobs()

    def on_stop_all(self):
        running = [
            job for job in self.jobs
            if job.state in ("running", "queued")
        ]

        if not running:
            return

        first = QtWidgets.QMessageBox.question(
            self, "Stop everything",
            f"Stop {len(running)} running or queued batches?\n\n"
            f"Completed runs are kept. Whatever each batch is "
            f"partway through will be lost.",
        )

        if first != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        second = QtWidgets.QMessageBox.question(
            self, "Really stop everything",
            "Last chance. Stop all batches and clear the queue?",
        )

        if second != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        for job in running:
            self.stop_job(job)

        self.save_queue()
        self.draw_jobs()

    def on_clear_finished(self):
        self.jobs = [
            job for job in self.jobs
            if job.state in ("queued", "running")
        ]

        self.save_queue()
        self.draw_jobs()

    def start_job(self, job):
        os.makedirs(job.out, exist_ok=True)

        runner = (
            job.runner if os.path.isabs(job.runner)
            else os.path.join(PROJECT_ROOT, job.runner)
        )
        command = [sys.executable, runner] + job.arguments

        if "--out" not in job.arguments:
            command += ["--out", job.out]

        # Keep the child's output instead of discarding the only useful error
        # message when argparse/condition validation rejects a job.
        job.log_path = os.path.join(job.out, "lab_runner.log")
        with open(job.log_path, "a", encoding="utf-8") as log:
            log.write("\n=== " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===\n")
            log.write(" ".join(str(piece) for piece in command) + "\n")
            log.flush()
            job.process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
            )

        job.pid = job.process.pid
        job.reattached = False
        job.state = "running"
        job.run_phase = "initialising"
        job.started = time.time()

        # Persist the PID/state immediately. Previously lab_queue.json still
        # said queued/pid=null for an already-running child.
        self.save_queue()

    @staticmethod
    def job_uses_grouped_gpu(job):
        """Whether a job already fills one GPU with a tensor seed group."""
        if job.runner not in ("batch_runner.py", "characterisation_runner.py"):
            return False
        arguments = list(job.arguments)
        device = None
        if "--device" in arguments:
            position = arguments.index("--device") + 1
            if position < len(arguments):
                device = str(arguments[position]).lower()
        # No explicit device means the runners choose CUDA when available.
        if device == "cpu":
            return False
        group = 1
        if "--group" in arguments:
            position = arguments.index("--group") + 1
            if position < len(arguments):
                try:
                    group = int(arguments[position])
                except ValueError:
                    pass
        return group > 1

    def tick(self):
        queue_dirty = False

        for job in self.jobs:
            if job.state != "running":
                continue

            job.refresh()

            if job.process is not None:
                code = job.process.poll()

                if code is not None:
                    job.finished = time.time()

                    job.state = "done" if code == 0 else "failed"
                    queue_dirty = True

            elif job.reattached:
                # Started by an earlier session, so there is no
                # handle to poll. The lock file it wrote says
                # whether it is still alive.

                state, lock = running.state_of(job.out, job.pid)

                if state != "running":
                    job.finished = time.time()

                    job.state = (
                        "done" if job.completed >= job.runs
                        else "stopped"
                    )
                    queue_dirty = True

        if not self.queue_paused:
            # Named 'active' rather than 'running': a local called
            # running would shadow the module of that name for the
            # whole function, and the reattach check above would
            # fail with it.

            active = sum(
                1 for job in self.jobs if job.state == "running"
            )
            characterisation_active = any(
                job.state == "running"
                and job.runner == "characterisation_runner.py"
                for job in self.jobs
            )
            grouped_gpu_active = any(
                job.state == "running" and self.job_uses_grouped_gpu(job)
                for job in self.jobs
            )

            for job in self.jobs:
                if active >= self.concurrency:
                    break

                if job.state != "queued":
                    continue

                # Characterisation has its own hard GPU rule: one process,
                # one group of at most eight boxes, then the next group. Do
                # not allow two queued characterisation jobs to overlap even
                # when the general Lab concurrency is larger than one.
                if (
                    job.runner == "characterisation_runner.py"
                    and characterisation_active
                ):
                    continue

                # A width-16 tensor batch is already faster than the best
                # multi-process CUDA result on this machine. Starting another
                # grouped process creates a second CUDA context and contention,
                # so grouped GPU jobs remain sequential even when general queue
                # concurrency is higher. CPU and legacy group-1 jobs retain the
                # existing concurrency setting.
                if self.job_uses_grouped_gpu(job) and grouped_gpu_active:
                    continue

                self.start_job(job)
                active += 1

                if job.runner == "characterisation_runner.py":
                    characterisation_active = True
                if self.job_uses_grouped_gpu(job):
                    grouped_gpu_active = True

        if queue_dirty:
            self.save_queue()

        self.draw_jobs()

    def draw_jobs(self):
        counts = {
            "running": sum(job.state == "running" for job in self.jobs),
            "queued": sum(job.state == "queued" for job in self.jobs),
            "completed": sum(job.state == "done" for job in self.jobs),
            "failed": sum(
                job.state in ("failed", "stopped") for job in self.jobs
            ),
        }
        if hasattr(self, "batch_summary_labels"):
            for key, value in counts.items():
                self.batch_summary_labels[key].setText(str(value))
            self.queue_state_label.setText(
                "Queue paused" if self.queue_paused else
                f"Queue running  •  concurrency {self.concurrency}"
            )
            self.queue_state_label.setObjectName(
                "statusWarn" if self.queue_paused else "statusGood"
            )
            self.queue_state_label.style().unpolish(self.queue_state_label)
            self.queue_state_label.style().polish(self.queue_state_label)

        self.jobs_table.setRowCount(len(self.jobs))

        for row, job in enumerate(self.jobs):
            if job.state == "running" and job.fraction > 0.01:
                remaining = clock(
                    job.elapsed / job.fraction - job.elapsed
                )
            else:
                remaining = "-"

            def bar(fraction, width=16):
                filled = int(fraction * width)

                return "#" * filled + "-" * (width - filled)

            if job.state == "running":
                if job.run_phase == "initialising":
                    run_bar = "initialising engine"
                elif job.run_phase == "saving_results":
                    if job.results_total:
                        run_bar = (
                            f"saving results {job.results_done}/"
                            f"{job.results_total}"
                        )
                    else:
                        run_bar = "saving results"
                else:
                    run_bar = (
                        f"[{bar(job.run_fraction)}] "
                        f"{job.run_fraction:3.0%}"
                    )

                    if job.run_seed is not None:
                        label = str(job.run_seed)
                        run_bar += (
                            f"  seeds {label}"
                            if job.inflight_runs > 1 or "-" in label
                            else f"  seed {label}"
                        )
            else:
                run_bar = "-"

            live_text = "   ".join(job.headlines[-2:])
            live = job.live_chemistry
            if job.state == "running" and live:
                largest = live.get("largest")
                if largest:
                    largest_text = (
                        f"largest: seed {largest.get('seed', '?')} "
                        f"{largest.get('formula', '?')} "
                        f"({largest.get('atoms', 0)} atoms, "
                        f"{largest.get('heavy', 0)} heavy, "
                        f"{largest.get('carbon', 0)} C)"
                    )
                else:
                    largest_text = "largest: none with 2+ heavy atoms"
                live_text = (
                    f"{largest_text}   |   "
                    f"heavy molecules: {live.get('heavy_molecules', 0)}   |   "
                    f"C-C bonds: {live.get('cc_bonds', 0)}   |   "
                    f"hottest: seed {live.get('hottest_seed', '?')} "
                    f"{live.get('hottest_K', 0)} K   |   "
                    f"time: {float(live.get('time_fs', 0)) / 1000.0:.2f} ps"
                )

            values = [
                job.name,
                job.state,
                run_bar,
                f"[{bar(job.fraction)}] {job.fraction:3.0%}",
                f"{job.completed}/{job.runs}",
                clock(job.elapsed) if job.started else "-",
                remaining,
            ]

            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))

                if column in (2, 3):
                    item.setFont(QtGui.QFont("Consolas", 9))

                if job.state == "failed":
                    item.setForeground(QtGui.QColor("#b4471f"))
                elif job.state == "stopped":
                    item.setForeground(QtGui.QColor("#9a6b1f"))
                elif job.state == "done":
                    item.setForeground(QtGui.QColor("#1d7a55"))

                self.jobs_table.setItem(row, column, item)

        self.draw_live_batch_details()

    def draw_live_batch_details(self):
        if not hasattr(self, "live_seed_table"):
            return

        selected = self.jobs_table.selectionModel().selectedRows()
        if selected:
            row = selected[0].row()
        else:
            row = next(
                (
                    index for index, job in enumerate(self.jobs)
                    if job.state == "running"
                ),
                -1,
            )

        if row < 0 or row >= len(self.jobs):
            self.live_batch_title.setText(
                "Live chemistry — select a running batch"
            )
            self.live_batch_summary.setText(
                "Live atom information will appear at the next progress update."
            )
            self.live_seed_table.setRowCount(0)
            return

        job = self.jobs[row]
        batch_index = read_index(job.out)
        has_recording = any(
            entry.get("file") and os.path.exists(
                entry["file"] if os.path.isabs(entry["file"])
                else os.path.join(job.out, entry["file"])
            )
            for entry in batch_index
        )
        self.batch_view_results.setEnabled(bool(batch_index))
        self.batch_open_replay.setEnabled(has_recording)
        self.batch_continue.setEnabled(
            bool(batch_index) and job.state != "running"
        )
        self.live_batch_title.setText(f"Live chemistry — {job.name}")
        live = job.live_chemistry

        if not live:
            unstable = sum(
                entry.get("stable") is False for entry in batch_index
            )
            message = (
                "Waiting for the next live chemistry update. Updates are "
                "sparse so this panel does not add meaningful simulation load."
                if job.state == "running"
                else f"{len(batch_index)} recorded runs • "
                     f"{len(batch_index) - unstable} usable • "
                     f"{unstable} numerically unstable"
            )
            self.live_batch_summary.setText(message)
            self.live_seed_display_rows = []
            for table_row, entry in enumerate(batch_index):
                values = [
                    entry.get("seed", "?"),
                    ("✓ " if entry.get("stable") is not False else "⚠ ")
                    + str(entry.get("headline", "")),
                    entry.get("atoms", 0),
                    entry.get("largest_any_heavy", 0),
                    entry.get("most_carbon", 0),
                    entry.get("species_count", 0),
                    entry.get("heavy_bonds_formed", 0),
                    f"{float(entry.get('final_temperature', 0) or 0):.0f}",
                ]
                self.live_seed_display_rows.append(values)
            self.populate_live_seed_table()
            return

        largest = live.get("largest")
        if largest:
            largest_text = (
                f"Largest structure: {largest.get('formula', '?')} in seed "
                f"{largest.get('seed', '?')} — {largest.get('atoms', 0)} atoms, "
                f"{largest.get('heavy', 0)} heavy, {largest.get('carbon', 0)} carbon"
            )
        else:
            largest_text = "Largest structure: none with two or more heavy atoms"

        self.live_batch_summary.setText(
            f"{largest_text}     •     "
            f"Heavy molecules: {live.get('heavy_molecules', 0)}     •     "
            f"C–C bonds: {live.get('cc_bonds', 0)}     •     "
            f"Hottest: seed {live.get('hottest_seed', '?')} at "
            f"{live.get('hottest_K', 0)} K     •     "
            f"Simulation time: {float(live.get('time_fs', 0)) / 1000.0:.2f} ps"
        )

        rows = list(live.get("per_seed", []))
        self.live_seed_display_rows = []
        for table_row, record in enumerate(rows):
            structure = record.get("largest") or {}
            values = [
                record.get("seed", "?"),
                structure.get("formula", "—"),
                structure.get("atoms", 0),
                structure.get("heavy", 0),
                structure.get("carbon", 0),
                record.get("heavy_molecules", 0),
                record.get("cc_bonds", 0),
                record.get("temperature_K", 0),
            ]
            self.live_seed_display_rows.append(values)
        self.populate_live_seed_table()

    def on_live_seed_header_clicked(self, column):
        if self.live_seed_sort_column != column:
            self.live_seed_sort_column = column
            self.live_seed_sort_stage = 1
        else:
            self.live_seed_sort_stage = (self.live_seed_sort_stage + 1) % 3
            if self.live_seed_sort_stage == 0:
                self.live_seed_sort_column = None
        self.populate_live_seed_table()

    @staticmethod
    def live_sort_value(value):
        try:
            return 0, float(value)
        except (TypeError, ValueError):
            return 1, str(value).lower()

    def populate_live_seed_table(self):
        rows = list(getattr(self, "live_seed_display_rows", []))
        column = self.live_seed_sort_column
        if column is not None and self.live_seed_sort_stage:
            rows.sort(
                key=lambda row: self.live_sort_value(row[column]),
                reverse=self.live_seed_sort_stage == 1,
            )
        self.live_seed_table.setRowCount(len(rows))
        for table_row, values in enumerate(rows):
            for cell_column, value in enumerate(values):
                self.live_seed_table.setItem(
                    table_row, cell_column,
                    QtWidgets.QTableWidgetItem(str(value)),
                )

    # --------------------------------------------------------
    # Results tab

    def build_results_tab(self):
        page = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(page)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(12)

        eyebrow = QtWidgets.QLabel("SCIENTIFIC RESULTS")
        eyebrow.setObjectName("eyebrow")
        outer.addWidget(eyebrow)
        title = QtWidgets.QLabel("Compare runs and discover chemistry")
        title.setObjectName("heroTitle")
        outer.addWidget(title)

        headline = SectionCard("Selected batch")
        headline_grid = QtWidgets.QGridLayout()
        self.result_metric_labels = {}
        for index, (key, caption) in enumerate((
            ("usable", "USABLE RUNS"), ("unstable", "UNSTABLE"),
            ("species", "MAX SPECIES"), ("largest", "LARGEST STRUCTURE"),
        )):
            label = QtWidgets.QLabel(caption)
            label.setObjectName("eyebrow")
            value = QtWidgets.QLabel("—")
            value.setObjectName("metricValue")
            grid_row = (index // 2) * 2
            grid_column = index % 2
            headline_grid.addWidget(label, grid_row, grid_column)
            headline_grid.addWidget(value, grid_row + 1, grid_column)
            self.result_metric_labels[key] = value
        headline.addLayout(headline_grid)
        self.results_batch_context = QtWidgets.QLabel(
            "Choose a batch to inspect its conditions and runs."
        )
        self.results_batch_context.setObjectName("sectionSubtitle")
        headline.addWidget(self.results_batch_context)
        plot_controls = QtWidgets.QHBoxLayout()
        plot_controls.addWidget(QtWidgets.QLabel("Compare"))
        self.results_x_metric = QtWidgets.QComboBox()
        self.results_y_metric = QtWidgets.QComboBox()
        result_metrics = [
            ("Heavy bonds formed", "heavy_bonds_formed"),
            ("Late bonds formed", "late_formed"),
            ("Bond turnovers", "turnovers"),
            ("Largest closed shell", "largest_closed"),
            ("Largest structure", "largest_any"),
            ("Largest carbon count", "most_carbon"),
            ("Species count", "species_count"),
            ("Final potential energy", "final_potential"),
            ("Final temperature", "final_temperature"),
        ]
        for label, key in result_metrics:
            self.results_x_metric.addItem(label, key)
            self.results_y_metric.addItem(label, key)
        self.results_y_metric.setCurrentIndex(4)
        self.results_x_metric.currentIndexChanged.connect(
            self.draw_result_scatter
        )
        self.results_y_metric.currentIndexChanged.connect(
            self.draw_result_scatter
        )
        plot_controls.addWidget(self.results_x_metric)
        plot_controls.addWidget(QtWidgets.QLabel("against"))
        plot_controls.addWidget(self.results_y_metric)
        plot_controls.addStretch(1)
        headline.addLayout(plot_controls)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)

        left_column = QtWidgets.QWidget()
        left_column_layout = QtWidgets.QVBoxLayout(left_column)
        left_column_layout.setContentsMargins(0, 0, 0, 0)
        left_column_layout.setSpacing(10)
        left_column_layout.addWidget(headline)

        browser = SectionCard(
            "Runs",
            "Unstable runs are marked clearly. Select a run to analyse it; open it in Replay for event-level inspection.",
        )
        left = browser.layout

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Batch"))

        self.results_batch = QtWidgets.QComboBox()
        self.results_batch.currentIndexChanged.connect(
            self.on_pick_batch
        )
        row.addWidget(self.results_batch, stretch=1)

        left.addLayout(row)

        self.results_search = QtWidgets.QLineEdit()
        self.results_search.setPlaceholderText("Filter by seed, headline, or status…")
        self.results_search.textChanged.connect(self.filter_result_runs)
        left.addWidget(self.results_search)

        self.results_list = QtWidgets.QListWidget()
        self.results_list.currentRowChanged.connect(
            self.on_pick_run
        )

        font = QtGui.QFont("Consolas")
        font.setPointSize(10)
        self.results_list.setFont(font)

        left.addWidget(self.results_list, stretch=1)

        selected_label = QtWidgets.QLabel("SELECTED RUN")
        selected_label.setObjectName("eyebrow")
        left.addWidget(selected_label)
        self.results_open_replay = self.button(
            "Open selected run in Replay", self.on_open_viewer
        )
        self.results_open_replay.setObjectName("primaryAction")
        self.results_open_replay.setEnabled(False)
        left.addWidget(self.results_open_replay)

        analysis_label = QtWidgets.QLabel("BATCH ANALYSIS")
        analysis_label.setObjectName("eyebrow")
        left.addWidget(analysis_label)
        analysis_grid = QtWidgets.QGridLayout()
        analysis_grid.setSpacing(6)
        analysis_grid.addWidget(
            self.button("Summarise batch", self.on_summarise), 0, 0
        )
        analysis_grid.addWidget(
            self.button("Species table", self.on_species), 0, 1
        )
        analysis_grid.addWidget(
            self.button("Compare batches", self.on_compare), 1, 0
        )
        analysis_grid.addWidget(
            self.button("Dashboard", self.on_dashboard), 1, 1
        )
        left.addLayout(analysis_grid)

        utility_label = QtWidgets.QLabel("UTILITIES & DISPLAY")
        utility_label.setObjectName("eyebrow")
        left.addWidget(utility_label)
        utility_row = QtWidgets.QHBoxLayout()
        utility_row.addWidget(self.button("Export CSV", self.on_export))
        utility_row.addWidget(self.button("Refresh", self.reload_results))
        self.structures_button = self.button(
            "Structures: off", self.on_toggle_structures
        )
        utility_row.addWidget(self.structures_button)
        left.addLayout(utility_row)

        left_column_layout.addWidget(browser, 1)
        splitter.addWidget(left_column)

        analysis_card = SectionCard("Run analysis")
        right = analysis_card.layout

        self.results_title = QtWidgets.QLabel("select a run")
        right.addWidget(self.results_title)

        self.results_report = QtWidgets.QPlainTextEdit()
        self.results_report.setReadOnly(True)
        self.results_report.setLineWrapMode(
            QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap
        )
        self.results_report.setFont(font)
        self.results_report.setMinimumHeight(280)

        right.addWidget(self.results_report, stretch=2)

        self.results_plot = pg.PlotWidget()
        self.results_plot.setMinimumHeight(240)
        self.results_scatter = self.results_plot
        self.results_scatter_item = pg.ScatterPlotItem(size=10)
        self.results_scatter_item.sigClicked.connect(
            self.on_result_scatter_clicked
        )
        self.results_plot.addItem(self.results_scatter_item)

        right.addWidget(self.results_plot, stretch=3)

        splitter.addWidget(analysis_card)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([460, 860])

        self.want_structures = False
        self.results_paths = []

        self.reload_results()

        return page

    def filter_result_runs(self, text):
        wanted = text.strip().lower()
        for row in range(self.results_list.count()):
            item = self.results_list.item(row)
            item.setHidden(bool(wanted and wanted not in item.text().lower()))

    def draw_result_scatter(self):
        if not hasattr(self, "results_scatter_item"):
            return
        entries = getattr(self, "current_results_index", [])
        x_key = self.results_x_metric.currentData()
        y_key = self.results_y_metric.currentData()
        spots = []
        for row, entry in enumerate(entries):
            x = entry.get(x_key)
            y = entry.get(y_key)
            if x is None or y is None:
                continue
            stable = entry.get("stable") is not False
            spots.append({
                "pos": (float(x), float(y)),
                "data": row,
                "brush": pg.mkBrush("#4ca6ff" if stable else "#ff7373"),
                "pen": pg.mkPen(None),
            })
        self.results_scatter_item.setData(spots)
        self.results_scatter.setLabel(
            "bottom", self.results_x_metric.currentText()
        )
        self.results_scatter.setLabel(
            "left", self.results_y_metric.currentText()
        )

    def on_result_scatter_clicked(self, _plot, points, _event=None):
        if points:
            self.results_list.setCurrentRow(int(points[0].data()))

    def on_toggle_structures(self):
        self.want_structures = not self.want_structures

        self.structures_button.setText(
            "Structures: on" if self.want_structures
            else "Structures: off"
        )

        self.on_pick_run(self.results_list.currentRow())

    def reload_results(self):
        current = self.results_batch.currentText()

        self.batches = find_batches(self.root)

        self.results_batch.blockSignals(True)
        self.results_batch.clear()
        self.results_batch.addItems(
            [label for label, path in self.batches]
        )

        if current:
            position = self.results_batch.findText(current)

            if position >= 0:
                self.results_batch.setCurrentIndex(position)

        self.results_batch.blockSignals(False)

        self.on_pick_batch(self.results_batch.currentIndex())

    def on_pick_batch(self, position):
        self.results_list.clear()
        self.results_paths = []
        self.results_open_replay.setEnabled(False)

        if position < 0 or position >= len(self.batches):
            return

        label, path = self.batches[position]
        batch_index = read_index(path)
        self.current_results_index = batch_index
        usable = [
            entry for entry in batch_index
            if entry.get("stable") is not False
        ]
        unstable = len(batch_index) - len(usable)
        self.result_metric_labels["usable"].setText(
            f"{len(usable)} / {len(batch_index)}"
        )
        self.result_metric_labels["unstable"].setText(str(unstable))
        self.result_metric_labels["species"].setText(str(max(
            (int(entry.get("species_count", 0) or 0) for entry in usable),
            default=0,
        )))
        self.result_metric_labels["largest"].setText(str(max(
            (int(entry.get("largest_any", 0) or 0) for entry in usable),
            default=0,
        )))
        if batch_index:
            first = batch_index[0]
            self.results_batch_context.setText(
                f"{first.get('mixture', label)}  •  {first.get('box', '?')} Å  •  "
                f"{first.get('picoseconds', '?')} ps  •  "
                f"{first.get('hot_temperature', '?')} → "
                f"{first.get('cool_temperature', '?')} K  •  "
                f"recorder v{first.get('recording_format', 1)}"
            )

        for entry in batch_index:
            mark = "✓" if entry.get("stable", True) else "⚠"

            self.results_list.addItem(
                f"{mark} {entry.get('number', 0):03d}  "
                f"seed {entry.get('seed', '?'):<5} "
                f"{entry.get('headline', '')}"
            )

            self.results_paths.append(
                os.path.join(path, entry.get("file", ""))
            )

        if self.results_paths:
            self.results_list.setCurrentRow(0)
        self.draw_result_scatter()

    def on_pick_run(self, row):
        self.results_open_replay.setEnabled(
            0 <= row < len(self.results_paths)
            and os.path.exists(self.results_paths[row])
        )
        if row < 0 or row >= len(self.results_paths):
            return

        path = self.results_paths[row]

        if not os.path.exists(path):
            self.results_report.setPlainText(f"missing: {path}")
            return

        from recorder import Recorder

        import analysis
        import analysis_cache

        self.results_title.setText(os.path.basename(path))
        self.results_report.setPlainText("analysing...")
        QtWidgets.QApplication.processEvents()

        recorder = Recorder.load(path)

        result = analysis_cache.analyse_cached(
            recorder, path, analysis.analyse,
            stride=4, structures=self.want_structures,
        )

        lines = analysis.summary_lines(result)

        if not self.want_structures:
            lines.insert(
                0, "  structures off - turn them on to name isomers"
            )

        self.results_report.setPlainText("\n".join(lines))

        self.draw_result_scatter()

    def on_open_viewer(self):
        row = self.results_list.currentRow()

        if row < 0 or row >= len(self.results_paths):
            return

        self.open_replay(self.results_paths[row])

    def batch_numbers(self, index, key):
        values = [
            float(entry[key]) for entry in index
            if entry.get(key) is not None
            and entry.get("stable") is not False
        ]

        return np.array(values) if values else np.array([])

    def on_summarise(self):
        # Everything about one folder in one place: how the runs
        # were set up, the spread of every measure across them,
        # and which molecules turned up how often.

        position = self.results_batch.currentIndex()

        if position < 0 or position >= len(self.batches):
            return

        label, path = self.batches[position]

        index = read_index(path)

        if not index:
            self.results_report.setPlainText("nothing in this batch")
            return

        first = index[0]

        unstable = [
            entry for entry in index
            if entry.get("stable") is False
        ]

        usable = len(index) - len(unstable)

        lines = []

        lines.append("=" * 62)
        lines.append(f"  {label}")
        lines.append("=" * 62)
        lines.append("")
        lines.append(
            f"  {len(index)} runs, {usable} usable"
            + (f", {len(unstable)} excluded as unstable"
               if unstable else "")
        )

        seeds = sorted(
            entry.get("seed", -1) for entry in index
        )

        lines.append(f"  seeds {min(seeds)} to {max(seeds)}")
        lines.append("")

        lines.append("  how these were run")
        lines.append("  " + "-" * 46)

        def stated(key, suffix="", template="{:g}"):
            # A missing field is reported as unknown rather than
            # filled in with a default. A summary that quietly
            # shows 250 K for a run that was actually held at 500
            # is worse than one that admits it does not know.

            value = first.get(key)

            if value is None:
                return "not recorded"

            try:
                return template.format(float(value)) + suffix
            except (TypeError, ValueError):
                return str(value) + suffix

        settings = [
            ("mixture", first.get("mixture") or "not recorded"),
            ("atoms", first.get("atoms", "not recorded")),
            ("box", stated("box", " A")),
            ("density", (
                f"{first.get('atoms', 0) / (first.get('box', 1) ** 3):.4f}"
                " atoms per cubic angstrom"
                if first.get("atoms") and first.get("box")
                else "not recorded"
            )),
            ("duration", stated("picoseconds", " ps")),
            ("starting temperature", stated("hot_temperature", " K")),
            ("held until", stated("hot_until_fs", " fs")),
            ("trap temperature", stated("cool_temperature", " K")),
        ]

        if first.get("continued_from"):
            settings.append(
                ("continued from", first["continued_from"])
            )

            settings.append(
                ("added", stated("added_picoseconds", " ps"))
            )

        if first.get("expand_to"):
            settings.append(
                ("box expands to", f"{first['expand_to']:g} A")
            )

        if first.get("strikes"):
            settings += [
                ("strikes", first.get("strikes")),
                ("channel temperature",
                 f"{first.get('strike_temperature', 0):g} K"),
                ("bonds broken per strike",
                 first.get("strike_dissociation")),
            ]

        for name, value in settings:
            lines.append(f"    {name:<24} {value}")

        lines.append("")
        lines.append("")
        lines.append("  measures across the usable runs")
        lines.append("  " + "-" * 58)
        lines.append(
            f"    {'':<22}{'mean':>8}{'sd':>8}"
            f"{'lowest':>9}{'highest':>9}"
        )

        measures = [
            ("heavy_bonds_formed", "bonds formed"),
            ("late_formed", "formed after 2500 fs"),
            ("late_broke", "broke after 2500 fs"),
            ("turnovers", "pairs changed twice"),
            ("largest_closed", "largest closed shell"),
            ("largest_any", "largest of any kind"),
            ("most_carbon", "longest carbon count"),
            ("best_chain", "best carbon chain"),
            ("best_tail", "best clean tail"),
            ("species_count", "distinct species"),
            ("final_temperature", "ended at (K)"),
            ("final_potential", "final energy (eV)"),
            ("wall_seconds", "seconds per run"),
        ]

        for key, name in measures:
            values = self.batch_numbers(index, key)

            if not len(values):
                continue

            spread = (
                values.std(ddof=1) if len(values) > 1 else 0.0
            )

            lines.append(
                f"    {name:<22}{values.mean():>8.1f}"
                f"{spread:>8.1f}{values.min():>9.1f}"
                f"{values.max():>9.1f}"
            )

        # A trap that does not match where the runs ended is
        # worth pointing at: either the schedule did not do what
        # was asked, or the metadata is describing a different
        # run than the one that happened.

        trap = first.get("cool_temperature")

        ended = self.batch_numbers(index, "final_temperature")

        if trap is not None and len(ended):
            drift = abs(ended.mean() - float(trap))

            if drift > 0.25 * float(trap) + 40:
                lines.append("")
                lines.append(
                    f"    note: these were set to trap at "
                    f"{float(trap):g} K but ended around "
                    f"{ended.mean():.0f} K."
                )
                lines.append(
                    "    Either the schedule was overridden or "
                    "the settings above are not the ones used."
                )

        # Which molecules, and in how many runs.

        counts = {}
        closed = {}

        for entry in index:
            if entry.get("stable") is False:
                continue

            for name in entry.get("final_species", []):
                counts[name] = counts.get(name, 0) + 1

            for name in entry.get("closed_shell", []):
                closed[name] = closed.get(name, 0) + 1

        lines.append("")
        lines.append("")
        lines.append("  molecules surviving to the end")
        lines.append("  " + "-" * 58)
        lines.append(
            f"    {'formula':<14}{'in runs':>9}{'closed shell':>14}"
        )

        ordered = sorted(
            counts.items(),
            key=lambda item: (-item[1], -len(item[0])),
        )

        for name, number in ordered[:26]:
            lines.append(
                f"    {name:<14}{number:>4}/{usable:<4}"
                f"{closed.get(name, 0):>9}/{usable:<4}"
            )

        if len(ordered) > 26:
            lines.append(
                f"    ... and {len(ordered) - 26} more"
            )

        # The best single result in the folder.

        biggest = max(
            (
                entry for entry in index
                if entry.get("stable") is not False
            ),
            key=lambda entry: entry.get("largest_any", 0),
            default=None,
        )

        if biggest:
            lines.append("")
            lines.append("")
            lines.append("  biggest single product")
            lines.append("  " + "-" * 58)
            lines.append(
                f"    run {biggest.get('number', 0):03d}, "
                f"seed {biggest.get('seed', '?')}: "
                f"{biggest.get('headline', '')}"
            )

        if unstable:
            lines.append("")
            lines.append("")
            lines.append("  excluded")
            lines.append("  " + "-" * 58)

            for entry in unstable:
                lines.append(
                    f"    seed {entry.get('seed', '?')}: unexplained energy "
                    f"rise "
                    f"{entry.get('largest_energy_jump', 0):.0f} eV"
                )

        injected = [
            entry for entry in index
            if entry.get("declared_external_energy_events", 0)
        ]
        if injected:
            lines.append("")
            lines.append("")
            lines.append("  intentional strike stress")
            lines.append("  " + "-" * 58)
            lines.append(
                f"    {sum(entry.get('declared_external_energy_events', 0) for entry in injected)} "
                "declared energy events across "
                f"{len(injected)} runs; these do not by themselves mark a run unstable."
            )

        self.results_title.setText(f"summary of {label}")
        self.results_report.setPlainText("\n".join(lines))

    def on_species(self):
        # A table of every molecule against every batch, written
        # to results/species.csv and shown here.

        import export

        batches = export.find_batches(self.root)

        if not batches:
            self.results_report.setPlainText("no batches found")
            return

        rows = export.load(batches)

        os.makedirs("results", exist_ok=True)

        path = os.path.join("results", "species.csv")

        export.write_species(rows, path)

        with open(path) as handle:
            text = handle.read()

        self.results_title.setText(f"species table -> {path}")
        self.results_report.setPlainText(text)

    def on_compare(self):
        # Which products appear in which conditions, and how the
        # averages differ. Reuses the comparison already written
        # for the browser rather than keeping a second copy.

        from run_browser import compare_batches, discover_batches

        batches = discover_batches(self.root)

        if len(batches) < 2:
            self.results_report.setPlainText(
                "Only one batch found. A comparison needs at "
                "least two."
            )
            return

        self.results_title.setText(
            f"comparing {len(batches)} batches"
        )

        self.results_report.setPlainText(
            "\n".join(compare_batches(batches))
        )

    def on_export(self):
        # Writes runs.csv, species.csv and summary.txt, then shows
        # the summary here.

        import export

        try:
            rows, _ = export.find_batches(self.root), None
        except Exception:
            rows = None

        self.results_report.setPlainText("exporting...")
        QtWidgets.QApplication.processEvents()

        batches = export.find_batches(self.root)

        if not batches:
            self.results_report.setPlainText("no batches found")
            return

        data = export.load(batches)

        os.makedirs("results", exist_ok=True)

        export.write_runs(data, os.path.join("results", "runs.csv"))
        export.write_species(
            data, os.path.join("results", "species.csv")
        )

        lines = export.summarise(
            data, os.path.join("results", "summary.txt")
        )

        self.results_title.setText(
            f"{len(data)} runs exported to results/"
        )

        self.results_report.setPlainText("\n".join(lines))

    def on_dashboard(self):
        subprocess.Popen([
            sys.executable, "dashboard.py", self.root, "--open"
        ])

    # --------------------------------------------------------
    # Mixtures, templates, and keeping the queue on disk

    def on_new_mixture(self):
        current = self.available.get(self.mixture_box.currentText())
        dialog = MixtureDialog(
            self, box_size=self.box_size.value(),
            copy_source=(self.mixture_box.currentText(), current),
        )

        self._save_mixture_dialog(dialog)

    def on_edit_mixture(self):
        name = self.mixture_box.currentText()
        entry = self.available.get(name)
        if not entry:
            return
        custom = mixtures.load_custom()
        editing_name = name if name in custom else None
        initial_name = name if editing_name else f"{name} copy"
        title = "Edit mixture" if editing_name else "Copy built-in preset"
        dialog = MixtureDialog(
            self, name=initial_name, kind=entry[0], contents=entry[1],
            box_size=self.box_size.value(), editing_name=editing_name,
            window_title=title,
        )
        self._save_mixture_dialog(dialog)

    def _save_mixture_dialog(self, dialog):

        if dialog.exec():
            name, kind, contents = dialog.result()

            if not name or not contents:
                return

            custom = mixtures.load_custom()
            if dialog.editing_name and dialog.editing_name != name:
                custom.pop(dialog.editing_name, None)
            custom[name] = (kind, contents)

            mixtures.save_custom(custom)

            self.reload_mixtures()

            position = self.mixture_box.findText(name)

            if position >= 0:
                self.mixture_box.setCurrentIndex(position)

    def reload_templates(self):
        self.templates = {}

        if os.path.exists(TEMPLATE_FILE):
            try:
                with open(TEMPLATE_FILE) as handle:
                    self.templates = json.load(handle)
            except (json.JSONDecodeError, OSError):
                self.templates = {}

        self.template_box.clear()
        self.template_box.addItems(sorted(self.templates))

    def current_settings(self):
        return {
            "mixture": self.mixture_box.currentText(),
            "box": self.box_size.value(),
            "picoseconds": self.picoseconds.value(),
            "seeds": self.seeds.value(),
            "first_seed": self.first_seed.currentText(),
            "hot_temperature": self.hot_temperature.value(),
            "hot_until": self.hot_until.value(),
            "cool_temperature": self.cool_temperature.value(),
            "strikes": self.strikes.value(),
            "strike_temperature": self.strike_temperature.value(),
            "strike_dissociation": self.strike_dissociation.value(),
            "first_strike": self.first_strike.value(),
            "strike_interval": self.strike_interval.value(),
            "capture_every": self.capture_every.value(),
            "save_every": self.save_every.value(),
            "character_capture": self.character_capture.value(),
            "grouped": self.grouped.isChecked(),
            "physics": self.physics_box.currentData(),
            "folder_name": self.folder_name.text(),
        }

    def apply_settings(self, stored):
        position = self.mixture_box.findText(
            stored.get("mixture", "")
        )

        if position >= 0:
            self.mixture_box.setCurrentIndex(position)

        pairs = [
            (self.box_size, "box"),
            (self.picoseconds, "picoseconds"),
            (self.seeds, "seeds"),
            (self.hot_temperature, "hot_temperature"),
            (self.hot_until, "hot_until"),
            (self.cool_temperature, "cool_temperature"),
            (self.strikes, "strikes"),
            (self.strike_temperature, "strike_temperature"),
            (self.strike_dissociation, "strike_dissociation"),
            (self.first_strike, "first_strike"),
            (self.strike_interval, "strike_interval"),
            (self.capture_every, "capture_every"),
            (self.save_every, "save_every"),
            (self.character_capture, "character_capture"),
        ]

        for widget, key in pairs:
            if key in stored:
                widget.setValue(stored[key])

        self.first_seed.setCurrentText(
            str(stored.get("first_seed", "continue automatically"))
        )

        self.grouped.setChecked(bool(stored.get("grouped", False)))

        physics = stored.get("physics", "reactive")
        physics_index = self.physics_box.findData(physics)
        if physics_index < 0:
            physics_index = self.physics_box.findData("reactive")
        if physics_index >= 0:
            self.physics_box.setCurrentIndex(physics_index)

        self.folder_name.setText(stored.get("folder_name", ""))

        self.refresh_existing()

    def on_save_template(self):
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save template", "name"
        )

        if not ok or not name.strip():
            return

        self.templates[name.strip()] = self.current_settings()

        with open(TEMPLATE_FILE, "w") as handle:
            json.dump(self.templates, handle, indent=1)

        self.reload_templates()

        position = self.template_box.findText(name.strip())

        if position >= 0:
            self.template_box.setCurrentIndex(position)

    def on_load_template(self):
        name = self.template_box.currentText()

        if name in self.templates:
            self.apply_settings(self.templates[name])

    def on_delete_template(self):
        name = self.template_box.currentText()

        if name in self.templates:
            del self.templates[name]

            with open(TEMPLATE_FILE, "w") as handle:
                json.dump(self.templates, handle, indent=1)

            self.reload_templates()

    def save_queue(self):
        with open(QUEUE_FILE, "w") as handle:
            json.dump(
                [job.as_dict() for job in self.jobs],
                handle, indent=1,
            )

    def load_queue(self):
        if not os.path.exists(QUEUE_FILE):
            return

        try:
            with open(QUEUE_FILE) as handle:
                stored = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return

        self.jobs = [Job.from_dict(entry) for entry in stored]

        self.reattach()

        self.draw_jobs()

    def reattach(self):
        # Batches outlive the panel, so a job remembered as
        # running might still be going. Starting a second copy
        # would put two processes on the same index and destroy
        # it, so every remembered job is checked before the queue
        # is allowed to touch it.

        alive = []
        finished = []
        abandoned = []

        for job in self.jobs:
            # Trust the live process/lock over the persisted queue state.
            # An older Lab session could write "stopped" even while the
            # detached batch was still running; skipping terminal-looking
            # queue entries here made that mistake permanent on every reopen.
            state, lock = running.state_of(job.out, job.pid)

            if state == "running":
                job.state = "running"
                job.reattached = True
                job.pid = (lock or {}).get("pid", job.pid)
                job.started = (lock or {}).get("started", time.time())
                job.finished = None
                job.refresh()

                alive.append(job.name)
            elif job.state not in ("running", "queued"):
                # No live process backs this completed/stopped/failed entry,
                # so leave its persisted terminal state alone.
                continue
            elif state == "stale":
                # A lock left behind by something that died.

                running.remove_lock(job.out, job.pid)

                job.refresh()

                job.state = (
                    "done" if job.completed >= job.runs
                    else "stopped"
                )

                abandoned.append(job.name)
            elif job.state == "running":
                job.refresh()

                job.state = (
                    "done" if job.completed >= job.runs
                    else "stopped"
                )

                finished.append(job.name)

        if alive or abandoned:
            message = []

            if alive:
                message.append(
                    f"Still running from an earlier session, now "
                    f"being watched again:\n  "
                    + "\n  ".join(alive)
                )

            if abandoned:
                message.append(
                    f"Left a lock behind but is no longer running, "
                    f"so marked stopped:\n  "
                    + "\n  ".join(abandoned)
                )

            QtWidgets.QMessageBox.information(
                self, "Picking up where you left off",
                "\n\n".join(message),
            )

        # Keep recovered PIDs and reconciled states on disk too.
        if alive or finished or abandoned:
            self.save_queue()

    def closeEvent(self, event):
        running = [
            job for job in self.jobs if job.state == "running"
        ]

        if running:
            answer = QtWidgets.QMessageBox.question(
                self, "Batches are running",
                f"{len(running)} batches are still going. Close "
                f"anyway and leave them running in the background?",
            )

            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        self.save_queue()

        event.accept()


class MixtureRow(QtWidgets.QWidget):

    changed = QtCore.Signal()
    removeRequested = QtCore.Signal(object)

    def __init__(self, options, species=None, amount=1, parent=None):
        super().__init__(parent)
        self._amount_sync = False
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.species = QtWidgets.QComboBox()
        self.species.setEditable(True)
        self.species.setInsertPolicy(
            QtWidgets.QComboBox.InsertPolicy.NoInsert
        )
        completer = self.species.completer()
        completer.setCompletionMode(
            QtWidgets.QCompleter.CompletionMode.PopupCompletion
        )
        completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        completer.setCaseSensitivity(
            QtCore.Qt.CaseSensitivity.CaseInsensitive
        )
        for label, value in options:
            self.species.addItem(label, value)
        if species is not None:
            index = self.species.findData(species)
            if index < 0:
                self.species.addItem(f"{species}  (unsupported legacy entry)", species)
                index = self.species.count() - 1
            self.species.setCurrentIndex(index)
        self.amount = QtWidgets.QSpinBox()
        self.amount.setRange(1, 2_000_000_000)
        self.amount.setValue(max(1, int(amount)))
        self.amount.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.amount.setMinimumWidth(112)
        self.amount.setStyleSheet(
            "QSpinBox { padding-left: 8px; padding-right: 30px; }"
        )
        self.amount.setToolTip(
            "Exact integer amount. Values above the normal slider range remain valid."
        )
        self.amount_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.amount_slider.setRange(1, max(500, int(amount)))
        self.amount_slider.setValue(max(1, int(amount)))
        self.amount_slider.setMinimumWidth(130)
        self.amount_slider.setToolTip(
            "Absolute species amount; changing this does not alter other species."
        )
        remove = QtWidgets.QToolButton()
        remove.setText("Remove")
        remove.setToolTip("Remove this species from the composition")
        remove.clicked.connect(lambda: self.removeRequested.emit(self))
        self.species.currentIndexChanged.connect(self.changed.emit)
        self.amount.valueChanged.connect(self.on_amount_changed)
        self.amount_slider.valueChanged.connect(self.on_slider_changed)
        layout.addWidget(self.species, 2)
        layout.addWidget(self.amount_slider, 4)
        layout.addWidget(self.amount, 1)
        layout.addWidget(remove)

    def value(self):
        return self.species.currentData(), int(self.amount.value())

    def set_amount(self, amount):
        amount = max(1, int(amount))
        if amount > self.amount_slider.maximum():
            self.amount_slider.setMaximum(amount)
        self._amount_sync = True
        self.amount.setValue(amount)
        self.amount_slider.setValue(amount)
        self._amount_sync = False
        self.changed.emit()

    def on_amount_changed(self, amount):
        if self._amount_sync:
            return
        if amount > self.amount_slider.maximum():
            self.amount_slider.setMaximum(int(amount))
        self._amount_sync = True
        self.amount_slider.setValue(int(amount))
        self._amount_sync = False
        self.changed.emit()

    def on_slider_changed(self, amount):
        if self._amount_sync:
            return
        self._amount_sync = True
        self.amount.setValue(int(amount))
        self._amount_sync = False
        self.changed.emit()


class MixtureDialog(QtWidgets.QDialog):

    def __init__(self, parent=None, *, name="", kind="atoms", contents=None,
                 box_size=19.0, editing_name=None, window_title="New mixture",
                 copy_source=None):
        super().__init__(parent)
        self.editing_name = editing_name
        self.box_size = float(box_size)
        self.copy_source = copy_source
        self.rows = []
        self._raw_sync = False
        self._raw_dirty = False
        self._final_result = None
        self._known_custom = set(mixtures.load_custom())
        self._species_options_cache = {}

        self.setWindowTitle(window_title)
        self.resize(680, 700)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(10)

        heading = QtWidgets.QLabel("Mixture Builder")
        heading.setObjectName("heroTitle")
        outer.addWidget(heading)
        help_text = QtWidgets.QLabel(
            "Build the same atom or molecule composition used by the existing "
            "runtime without editing its text format."
        )
        help_text.setWordWrap(True)
        help_text.setObjectName("sectionSubtitle")
        outer.addWidget(help_text)

        identity = QtWidgets.QFormLayout()
        identity.setHorizontalSpacing(16)
        self.name = QtWidgets.QLineEdit(name)
        self.name.setPlaceholderText("e.g. Carbon-rich experiment")
        self.name.textChanged.connect(self.update_feedback)
        identity.addRow("Name", self.name)

        self.kind = QtWidgets.QComboBox()
        self.kind.addItem("Loose atoms", "atoms")
        self.kind.addItem("Molecules", "molecules")
        self.kind.setCurrentIndex(0 if kind == "atoms" else 1)
        self.kind.currentIndexChanged.connect(self.on_kind_changed)
        identity.addRow("Composition type", self.kind)

        if copy_source and copy_source[1]:
            self.start_from = QtWidgets.QComboBox()
            self.start_from.addItem("Empty", None)
            self.start_from.addItem(
                f"Copy current: {copy_source[0]}", copy_source[0]
            )
            self.start_from.currentIndexChanged.connect(self.on_start_from)
            identity.addRow("Start from", self.start_from)
        else:
            self.start_from = None
        outer.addLayout(identity)

        self.runtime_note = QtWidgets.QLabel()
        self.runtime_note.setWordWrap(True)
        self.runtime_note.setObjectName("sectionSubtitle")
        outer.addWidget(self.runtime_note)

        composition_group = QtWidgets.QGroupBox("Composition")
        composition_layout = QtWidgets.QVBoxLayout(composition_group)
        labels = QtWidgets.QHBoxLayout()
        species_label = QtWidgets.QLabel("Species")
        species_label.setStyleSheet("font-weight: bold;")
        slider_label = QtWidgets.QLabel("Amount slider")
        slider_label.setStyleSheet("font-weight: bold;")
        amount_label = QtWidgets.QLabel("Amount")
        amount_label.setStyleSheet("font-weight: bold;")
        labels.addWidget(species_label, 2)
        labels.addWidget(slider_label, 4)
        labels.addWidget(amount_label, 1)
        labels.addSpacing(72)
        composition_layout.addLayout(labels)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(185)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.rows_widget = QtWidgets.QWidget()
        self.rows_layout = QtWidgets.QVBoxLayout(self.rows_widget)
        self.rows_layout.setContentsMargins(0, 0, 4, 0)
        self.rows_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self.rows_widget)
        composition_layout.addWidget(scroll)
        add_button = QtWidgets.QPushButton("+ Add species")
        add_button.clicked.connect(self.add_row)
        composition_layout.addWidget(add_button, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        outer.addWidget(composition_group, 2)

        feedback_group = QtWidgets.QGroupBox("Live composition")
        feedback_layout = QtWidgets.QVBoxLayout(feedback_group)
        self.feedback = QtWidgets.QLabel()
        self.feedback.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.feedback.setWordWrap(True)
        self.feedback.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        feedback_layout.addWidget(self.feedback)

        density_line = QtWidgets.QFrame()
        density_line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        feedback_layout.addWidget(density_line)
        target_row = QtWidgets.QHBoxLayout()
        target_row.addWidget(QtWidgets.QLabel("Target density"))
        self.density_preset = QtWidgets.QComboBox()
        for label, value in (
            ("Dilute soup", 0.010),
            ("Light soup", 0.020),
            ("Standard soup", 0.040),
            ("Dense soup", 0.050),
            ("Custom", None),
        ):
            self.density_preset.addItem(label, value)
        self.density_preset.setCurrentIndex(2)
        self.custom_density_index = self.density_preset.count() - 1
        self.target_density = QtWidgets.QDoubleSpinBox()
        self.target_density.setDecimals(6)
        self.target_density.setRange(0.000001, 100.0)
        self.target_density.setSingleStep(0.001)
        self.target_density.setValue(0.040)
        self.target_density.setMinimumWidth(120)
        self.target_density.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.target_density.setStyleSheet(
            "QDoubleSpinBox { padding-left: 8px; padding-right: 30px; }"
        )
        target_row.addWidget(self.density_preset)
        target_row.addWidget(self.target_density)
        target_row.addWidget(QtWidgets.QLabel("atoms/Å³"))
        target_row.addStretch(1)
        feedback_layout.addLayout(target_row)
        self.density_preview = QtWidgets.QLabel()
        self.density_preview.setWordWrap(True)
        self.density_preview.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        feedback_layout.addWidget(self.density_preview)
        self.apply_density_button = QtWidgets.QPushButton()
        self.apply_density_button.setToolTip(
            "Changes particle loading while preserving composition. Density is "
            "atoms per Å³; this is not a direct pressure calculation."
        )
        self.apply_density_button.clicked.connect(self.apply_target_density)
        feedback_layout.addWidget(
            self.apply_density_button,
            alignment=QtCore.Qt.AlignmentFlag.AlignLeft,
        )
        density_help = QtWidgets.QLabel(
            "Convenience loading targets for ps-scale soup discovery. The fixed "
            "box does not resize, and these values are not atmospheric pressure."
        )
        density_help.setWordWrap(True)
        density_help.setObjectName("sectionSubtitle")
        feedback_layout.addWidget(density_help)
        self.density_preset.currentIndexChanged.connect(
            self.on_density_preset_changed
        )
        self.target_density.valueChanged.connect(
            self.on_target_density_changed
        )
        outer.addWidget(feedback_group)

        self.advanced = QtWidgets.QGroupBox("Advanced — raw mixture definition")
        self.advanced.setCheckable(True)
        self.advanced.setChecked(False)
        advanced_layout = QtWidgets.QVBoxLayout(self.advanced)
        advanced_help = QtWidgets.QLabel(
            "Legacy one-entry-per-line form. Closing Advanced reparses it into "
            "the structured rows; invalid text is never silently discarded."
        )
        advanced_help.setWordWrap(True)
        self.raw_contents = QtWidgets.QPlainTextEdit()
        self.raw_contents.setMinimumHeight(110)
        self.raw_contents.textChanged.connect(self.on_raw_changed)
        advanced_layout.addWidget(advanced_help)
        advanced_layout.addWidget(self.raw_contents)
        self.advanced.toggled.connect(self.on_advanced_toggled)
        self.raw_contents.setVisible(False)
        advanced_help.setVisible(False)
        self.advanced_help = advanced_help
        outer.addWidget(self.advanced)

        self.error = QtWidgets.QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #ff8a80;")
        outer.addWidget(self.error)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.save_button = buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Save
        )
        outer.addWidget(buttons)

        self.set_contents(kind, contents or {})
        self.update_runtime_note()
        self.update_feedback()
        if parent is not None and hasattr(parent, "box_size"):
            try:
                changed = parent.box_size.valueChanged
                if callable(changed) and not hasattr(changed, "connect"):
                    changed = changed()
                changed.connect(self.on_parent_box_changed)
            except Exception:
                pass

    def current_kind(self):
        return str(self.kind.currentData())

    def species_options(self):
        kind = self.current_kind()
        if kind in self._species_options_cache:
            return self._species_options_cache[kind]
        species = mixtures.supported_species(kind)
        if kind == "atoms":
            options = [(symbol, symbol) for symbol in species]
            self._species_options_cache[kind] = options
            return options
        sightings = {}
        try:
            for item in molecule_store.list_molecules():
                formula = str(item.get("formula", ""))
                sightings[formula] = sightings.get(formula, 0) + 1
        except Exception:
            sightings = {}
        options = [
            (
                f"{formula}  ({sightings[formula]} saved structure"
                f"{'s' if sightings[formula] != 1 else ''})"
                if sightings.get(formula) else formula,
                formula,
            )
            for formula in species
        ]
        self._species_options_cache[kind] = options
        return options

    def update_runtime_note(self):
        if self.current_kind() == "atoms":
            self.runtime_note.setText(
                "Supported loose elements come directly from the reactive engine."
            )
            return
        try:
            discovered = len(molecule_store.list_molecules())
        except Exception:
            discovered = 0
        self.runtime_note.setText(
            f"The runtime currently has {len(mixtures.supported_species('molecules'))} "
            f"launchable molecule geometries. Molecule Lab contains {discovered} "
            "discovered structures; unsupported discoveries are not offered because "
            "the current soup schema cannot launch them safely yet."
        )

    def add_row(self, checked=False, species=None, amount=1):
        options = self.species_options()
        if not options:
            return
        if species is None:
            used = {row.value()[0] for row in self.rows}
            species = next((value for _, value in options if value not in used), options[0][1])
        row = MixtureRow(options, species, amount, self.rows_widget)
        row.changed.connect(self.on_rows_changed)
        row.removeRequested.connect(self.remove_row)
        self.rows.append(row)
        self.rows_layout.addWidget(row)
        self.on_rows_changed()

    def remove_row(self, row):
        if row not in self.rows:
            return
        self.rows.remove(row)
        self.rows_layout.removeWidget(row)
        row.deleteLater()
        self.on_rows_changed()

    def row_contents(self):
        contents = {}
        for row in self.rows:
            species, amount = row.value()
            if species:
                contents[species] = contents.get(species, 0) + int(amount)
        return contents

    def clear_rows(self):
        for row in self.rows:
            self.rows_layout.removeWidget(row)
            row.deleteLater()
        self.rows = []

    def set_contents(self, kind, contents):
        index = self.kind.findData(kind)
        if index >= 0 and self.kind.currentIndex() != index:
            self.kind.blockSignals(True)
            self.kind.setCurrentIndex(index)
            self.kind.blockSignals(False)
        self.clear_rows()
        for species, amount in contents.items():
            self.add_row(species=species, amount=amount)
        if not self.rows:
            self.add_row()
        self._raw_dirty = False
        self.sync_raw_from_rows()
        self.update_runtime_note()
        self.update_feedback()

    def on_kind_changed(self, unused=None):
        self.set_contents(self.current_kind(), {})

    def on_start_from(self, index):
        if index == 1 and self.copy_source and self.copy_source[1]:
            kind, contents = self.copy_source[1]
            self.set_contents(kind, contents)
        elif index == 0:
            self.set_contents(self.current_kind(), {})

    def on_rows_changed(self):
        if not self._raw_dirty:
            self.sync_raw_from_rows()
        self.update_feedback()

    def sync_raw_from_rows(self):
        self._raw_sync = True
        self.raw_contents.setPlainText(
            mixtures.format_definition(self.row_contents())
        )
        self._raw_sync = False

    def on_raw_changed(self):
        if not self._raw_sync:
            self._raw_dirty = True
            self.update_feedback()

    def apply_raw(self):
        try:
            contents = mixtures.parse_definition(
                self.raw_contents.toPlainText(), self.current_kind()
            )
        except ValueError as problem:
            self.error.setText(f"Advanced definition: {problem}")
            return False
        self.set_contents(self.current_kind(), contents)
        self.error.setText("")
        return True

    def on_advanced_toggled(self, visible):
        self.raw_contents.setVisible(visible)
        self.advanced_help.setVisible(visible)
        if visible:
            if not self._raw_dirty:
                self.sync_raw_from_rows()
        elif self._raw_dirty and not self.apply_raw():
            self.advanced.blockSignals(True)
            self.advanced.setChecked(True)
            self.advanced.blockSignals(False)
            self.raw_contents.setVisible(True)
            self.advanced_help.setVisible(True)
        self.adjustSize()

    def validated_result(self):
        name = self.name.text().strip()
        if not name:
            raise ValueError("Give the mixture a name.")
        if name in mixtures.BUILT_IN:
            raise ValueError(
                "Built-in presets are read-only. Choose a new name for this copy."
            )
        if name in self._known_custom and name != self.editing_name:
            raise ValueError(
                "A custom mixture already uses that name. Edit it directly or choose another name."
            )
        if self._raw_dirty:
            contents = mixtures.parse_definition(
                self.raw_contents.toPlainText(), self.current_kind()
            )
        else:
            contents = mixtures.validate_contents(
                self.current_kind(), self.row_contents()
            )
        return name, self.current_kind(), contents

    def actual_box_size(self):
        parent = self.parent()
        if parent is not None and hasattr(parent, "box_size"):
            try:
                value = float(parent.box_size.value())
                if value > 0:
                    self.box_size = value
            except Exception:
                pass
        return self.box_size

    def on_parent_box_changed(self, unused=None):
        self.update_feedback()

    def on_density_preset_changed(self, index):
        value = self.density_preset.itemData(index)
        if value is None:
            return
        self.target_density.blockSignals(True)
        self.target_density.setValue(float(value))
        self.target_density.blockSignals(False)
        self.update_density_preview()

    def on_target_density_changed(self, value):
        matching = self.custom_density_index
        for index in range(self.density_preset.count()):
            preset = self.density_preset.itemData(index)
            if preset is not None and abs(float(preset) - float(value)) < 5e-7:
                matching = index
                break
        self.density_preset.blockSignals(True)
        self.density_preset.setCurrentIndex(matching)
        self.density_preset.blockSignals(False)
        self.update_density_preview()

    def density_contents(self):
        return (
            mixtures.parse_definition(
                self.raw_contents.toPlainText(), self.current_kind()
            ) if self._raw_dirty else
            mixtures.validate_contents(self.current_kind(), self.row_contents())
        )

    def update_density_preview(self):
        try:
            contents = self.density_contents()
            preview = mixtures.scale_to_density(
                self.current_kind(), contents, self.actual_box_size(),
                self.target_density.value(),
            )
        except ValueError as problem:
            self.density_preview.setText(f"Cannot scale composition: {problem}")
            self.apply_density_button.setText("Adjust composition first")
            self.apply_density_button.setEnabled(False)
            return
        same = preview["contents"] == contents
        direction = (
            "Pressurise" if preview["result_atoms"] > preview["current_atoms"]
            else "Depressurise"
        )
        self.density_preview.setText(
            f"Current atoms <b>{preview['current_atoms']:,}</b>&nbsp;&nbsp; "
            f"Target atoms <b>{preview['target_atoms']:,}</b>&nbsp;&nbsp; "
            f"Scale <b>×{preview['scale']:.3g}</b><br>"
            f"Result after integer apportionment: <b>{preview['result_atoms']:,} atoms</b>, "
            f"<b>{preview['result_density_atoms_per_A3']:.6f} atoms/Å³</b>"
        )
        if same:
            self.apply_density_button.setText("✓ At target density")
            self.apply_density_button.setEnabled(False)
        else:
            self.apply_density_button.setText(f"{direction} to target")
            self.apply_density_button.setEnabled(True)

    def apply_target_density(self):
        try:
            preview = mixtures.scale_to_density(
                self.current_kind(), self.density_contents(),
                self.actual_box_size(), self.target_density.value(),
            )
        except ValueError as problem:
            self.error.setText(str(problem))
            return
        # Rebuild unique rows from the apportioned ordinary integer counts.
        # The target field is deliberately untouched.
        self.set_contents(self.current_kind(), preview["contents"])
        self.error.setText("")

    def update_feedback(self):
        try:
            contents = (
                mixtures.parse_definition(
                    self.raw_contents.toPlainText(), self.current_kind()
                ) if self._raw_dirty else
                mixtures.validate_contents(self.current_kind(), self.row_contents())
            )
            metrics = mixtures.composition_metrics(
                self.current_kind(), contents, self.actual_box_size()
            )
            elements = metrics["elements"]
            element_lines = []
            for symbol, amount in sorted(
                elements.items(), key=lambda item: (-item[1], item[0])
            ):
                percent = 100.0 * amount / metrics["atoms"]
                blocks = "▰" * max(1, int(round(percent / 5.0)))
                element_lines.append(
                    f"<b>{symbol}</b>&nbsp;&nbsp;{amount:,}&nbsp;&nbsp;"
                    f"{percent:5.1f}%&nbsp;&nbsp;<span style='color:#58a6ff'>{blocks}</span>"
                )
            molecule_text = (
                f"&nbsp;&nbsp; Molecules <b>{metrics['molecules']:,}</b>"
                if metrics["molecules"] is not None else ""
            )
            self.feedback.setText(
                f"Total atoms <b>{metrics['atoms']:,}</b>{molecule_text}"
                f"&nbsp;&nbsp; Box <b>{metrics['box_A']:g} Å</b>"
                f"&nbsp;&nbsp; Density <b>{metrics['density_atoms_per_A3']:.4f} atoms/Å³</b>"
                "<br><br>" + "<br>".join(element_lines)
            )
            self.error.setText("")
            valid = bool(self.name.text().strip())
        except ValueError as problem:
            self.feedback.setText("Composition incomplete.")
            self.error.setText(str(problem))
            valid = False
        self.save_button.setEnabled(valid)
        self.update_density_preview()

    def accept(self):
        try:
            self._final_result = self.validated_result()
        except ValueError as problem:
            self.error.setText(str(problem))
            self.save_button.setEnabled(False)
            return
        super().accept()

    def result(self):
        if self._final_result is not None:
            return self._final_result
        return self.validated_result()


def main():
    os.chdir(PROJECT_ROOT)
    pg.setConfigOptions(antialias=True)

    application = QtWidgets.QApplication(sys.argv)

    window = Lab()
    window.show()

    sys.exit(application.exec())


if __name__ == "__main__":
    main()
