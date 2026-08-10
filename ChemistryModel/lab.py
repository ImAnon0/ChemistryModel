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


# ============================================================
# One place to run everything from
# ============================================================
#
#   py lab.py
#
# Three tabs. Run builds a job and queues it. Batches shows what
# is running, what is waiting and what has finished. Results
# reads the reports.
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
    # Any existing folder whose runs were made under the same
    # conditions. Used to tell the person which seeds are already
    # taken before they queue more.

    for label, path in find_batches(root):
        index = read_index(path)

        if not index:
            continue

        first = index[0]

        strikes = int(first.get("strikes", 0))

        found = {
            "mixture": first.get("mixture"),
            "box": round(float(first.get("box", 0)), 2),
            "picoseconds": round(
                float(first.get("picoseconds", 0)), 3
            ),
            "strikes": strikes,
            "cool_temperature": round(
                float(first.get("cool_temperature", 250) or 250), 0
            ),
            "expand_to": round(
                float(first.get("expand_to", 0) or 0), 2
            ),
        }

        if found == wanted:
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


class Job:

    def __init__(self, name, arguments, out, runs):
        self.name = name
        self.arguments = arguments
        self.out = out
        self.runs = runs

        self.state = "queued"
        self.process = None
        self.pid = None
        self.reattached = False
        self.started = None
        self.finished = None
        self.completed = 0
        self.headlines = []

    def as_dict(self):
        return {
            "name": self.name,
            "arguments": self.arguments,
            "out": self.out,
            "runs": self.runs,
            "state": self.state,
            "pid": self.pid,
        }

    @classmethod
    def from_dict(cls, stored):
        job = cls(
            stored["name"], stored["arguments"],
            stored["out"], stored["runs"],
        )

        job.state = stored.get("state", "queued")
        job.pid = stored.get("pid")

        return job

    def refresh(self):
        # Progress and recent results come from the index the
        # batch writes after each run.

        index = read_index(self.out)

        self.completed = len(index)

        self.headlines = [
            f"{entry.get('seed', '?')}: {entry.get('headline', '')}"
            for entry in index[-4:]
        ]

    @property
    def fraction(self):
        if not self.runs:
            return 0.0

        return min(self.completed / self.runs, 1.0)

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

        layout = QtWidgets.QVBoxLayout(self)

        self.tabs = QtWidgets.QTabWidget()

        self.tabs.addTab(self.build_run_tab(), "Run")
        self.tabs.addTab(self.build_batches_tab(), "Batches")
        self.tabs.addTab(self.build_results_tab(), "Results")

        layout.addWidget(self.tabs)

        self.load_queue()
        self.on_mode()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(POLL_MILLISECONDS)

    # --------------------------------------------------------
    # Run tab

    def build_run_tab(self):
        page = QtWidgets.QWidget()
        columns = QtWidgets.QHBoxLayout(page)

        left = QtWidgets.QFormLayout()

        # Two ways to make runs: start fresh ones, or take an
        # existing batch and carry every run in it further. The
        # second reuses everything already computed, which matters
        # when a twenty picosecond run is still producing
        # molecules at nineteen.

        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItems([
            "new runs",
            "continue an existing batch",
        ])
        self.mode_box.currentIndexChanged.connect(self.on_mode)
        left.addRow("what to do", self.mode_box)

        self.source_box = QtWidgets.QComboBox()
        self.source_row = left.rowCount()
        left.addRow("batch to continue", self.source_box)
        self.source_box.currentTextChanged.connect(
            self.refresh_existing
        )

        self.mixture_box = QtWidgets.QComboBox()
        self.reload_mixtures()
        self.mixture_box.currentTextChanged.connect(
            self.refresh_existing
        )
        left.addRow("mixture", self.mixture_box)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(
            self.button("New mixture...", self.on_new_mixture)
        )
        row.addWidget(
            self.button("Reload list", self.reload_mixtures)
        )
        left.addRow("", self.wrap(row))

        self.box_size = self.choice(
            [12, 15, 17, 19, 21, 24, 28, 34], 19, 1
        )
        left.addRow("box size (A)", self.box_size)

        self.atom_note = QtWidgets.QLabel("")
        self.atom_note.setStyleSheet("color: #666;")
        left.addRow("", self.atom_note)

        self.picoseconds = self.choice(
            [5, 10, 20, 40, 60, 100], 20, 1
        )
        left.addRow("duration (ps)", self.picoseconds)

        self.seeds = self.choice([1, 3, 5, 10, 15, 20, 30, 50], 10, 0)
        left.addRow("how many runs", self.seeds)

        self.first_seed = QtWidgets.QComboBox()
        self.first_seed.setEditable(True)
        self.first_seed.setInsertPolicy(
            QtWidgets.QComboBox.InsertPolicy.NoInsert
        )
        self.first_seed.addItems(
            ["continue automatically", "0", "100", "700", "800"]
        )
        left.addRow("first seed", self.first_seed)

        self.hot_temperature = self.choice(
            [250, 350, 500, 700, 1000, 1500], 500, 0
        )
        left.addRow("starting temp (K)", self.hot_temperature)

        self.hot_until = self.choice(
            [0, 500, 1000, 2000, 4000, 8000], 2000, 0
        )
        left.addRow("hold until (fs)", self.hot_until)

        self.cool_temperature = self.choice(
            [100, 250, 350, 500, 700, 1000], 250, 0
        )
        left.addRow("trap temp (K)", self.cool_temperature)

        self.expand_to = self.choice(
            [0, 15, 17, 19, 21, 24, 28], 0, 1
        )
        left.addRow("expand box to (A)", self.expand_to)

        self.expand_at = self.choice(
            [500, 1000, 2000, 4000], 2000, 0
        )
        left.addRow("expand at (fs)", self.expand_at)

        self.strikes = self.choice([0, 1, 2, 3, 5, 8, 10, 20], 0, 0)
        left.addRow("lightning strikes", self.strikes)

        self.strike_temperature = self.choice(
            [5000, 10000, 20000, 25000, 30000, 50000], 30000, 0
        )
        left.addRow("channel temp (K)", self.strike_temperature)

        self.strike_dissociation = self.choice(
            [0, 0.2, 0.4, 0.6, 1.0, 1.5], 0.6, 2
        )
        left.addRow("bonds broken", self.strike_dissociation)

        self.first_strike = self.choice(
            [500, 1000, 2500, 5000], 2500, 0
        )
        left.addRow("first strike (fs)", self.first_strike)

        self.strike_interval = self.choice(
            [1000, 2000, 3500, 5000, 10000], 3500, 0
        )
        left.addRow("strike every (fs)", self.strike_interval)

        self.capture_every = self.choice([10, 20, 40, 80, 200], 40, 0)
        left.addRow("capture every N steps", self.capture_every)

        self.parallel = QtWidgets.QCheckBox(
            "split across the running slots"
        )
        self.parallel.setToolTip(
            "Break the runs into as many jobs as can run at once, "
            "each taking its own seeds, all writing to the same "
            "folder."
        )
        self.parallel.stateChanged.connect(self.refresh_existing)
        left.addRow("in parallel", self.parallel)

        self.folder_name = QtWidgets.QLineEdit()
        self.folder_name.setPlaceholderText(
            "leave blank to name it from the conditions"
        )
        left.addRow("output folder", self.folder_name)

        for widget in (
            self.box_size, self.picoseconds, self.cool_temperature,
            self.strikes, self.expand_to,
        ):
            widget.valueChanged.connect(self.refresh_existing)

        columns.addLayout(left, stretch=3)

        right = QtWidgets.QVBoxLayout()

        self.existing_note = QtWidgets.QLabel("")
        self.existing_note.setWordWrap(True)
        self.existing_note.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop
        )
        self.existing_note.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 12px;"
        )
        right.addWidget(self.existing_note)

        right.addWidget(self.divider())

        right.addWidget(QtWidgets.QLabel("templates"))

        self.template_box = QtWidgets.QComboBox()
        right.addWidget(self.template_box)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.button("Load", self.on_load_template))
        row.addWidget(self.button("Save as...", self.on_save_template))
        row.addWidget(self.button("Delete", self.on_delete_template))
        right.addLayout(row)

        self.reload_templates()

        right.addStretch(1)

        self.queue_button = self.button(
            "Add to queue", self.on_queue
        )
        self.queue_button.setMinimumHeight(38)
        right.addWidget(self.queue_button)

        self.preview = QtWidgets.QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(120)
        self.preview.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px;"
        )
        right.addWidget(QtWidgets.QLabel("command"))
        right.addWidget(self.preview)

        columns.addLayout(right, stretch=2)

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

    def on_mode(self):
        continuing = self.mode_box.currentIndex() == 1

        # Everything that describes a fresh box is meaningless
        # when extending one that already exists: the mixture, the
        # size and the starting temperature are already fixed by
        # the runs being continued.

        for widget in (
            self.mixture_box, self.box_size, self.seeds,
            self.first_seed, self.hot_temperature, self.hot_until,
            self.expand_to, self.expand_at, self.capture_every,
        ):
            widget.setEnabled(not continuing)

        self.source_box.setEnabled(continuing)

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
        return {
            "mixture": self.mixture_box.currentText(),
            "box": round(self.box_size.value(), 2),
            "picoseconds": round(self.picoseconds.value(), 3),
            "strikes": int(self.strikes.value()),
            "cool_temperature": round(
                self.cool_temperature.value(), 0
            ),
            "expand_to": round(self.expand_to.value(), 2),
        }

    def refresh_existing(self):
        name = self.mixture_box.currentText()

        entry = self.available.get(name)

        if entry:
            kind, contents = entry

            total = sum(contents.values())

            if kind == "molecules":
                # Rough: the atom count depends on which molecules.
                sizes = {"H2": 2, "H2O": 3, "NH3": 4, "CH4": 5}

                total = sum(
                    sizes.get(molecule, 3) * number
                    for molecule, number in contents.items()
                )

            box = self.box_size.value()

            self.atom_note.setText(
                f"{total} atoms, density "
                f"{total / (box ** 3):.4f} atoms per cubic angstrom"
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
                f"Queueing more will continue from seed "
                f"{max(seeds) + 1} and write to the same folder."
                + self.parallel_note()
            )
        else:
            self.existing_note.setText(
                "No runs exist under these conditions yet.\n\n"
                "A new folder will be created, named from the\n"
                "settings unless you give one."
                + self.parallel_note()
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

    def parallel_note(self):
        if not self.parallel.isChecked():
            return ""

        wanted = int(self.seeds.value())

        parts = max(1, min(self.concurrency, wanted))

        share = wanted // parts
        extra = wanted % parts

        counts = [
            share + (1 if part < extra else 0)
            for part in range(parts)
        ]

        return (
            f"\n\nSplit into {parts} jobs of "
            + ", ".join(str(count) for count in counts)
            + " runs, each with its own seeds,\nall writing to "
            "the same folder."
        )

    def build_arguments(self):
        if self.mode_box.currentIndex() == 1:
            return self.build_continue_arguments()

        arguments = [
            "--mixture", self.mixture_box.currentText(),
            "--box", f"{self.box_size.value():g}",
            "--ps", f"{self.picoseconds.value():g}",
            "--seeds", str(int(self.seeds.value())),
            "--capture-every", str(int(self.capture_every.value())),
            "--hot-temperature", f"{self.hot_temperature.value():g}",
            "--hot-until-fs", f"{self.hot_until.value():g}",
            "--cool-temperature",
            f"{self.cool_temperature.value():g}",
        ]

        seed_text = self.first_seed.currentText().strip()

        if seed_text and not seed_text[0].isalpha():
            try:
                arguments += ["--first-seed", str(int(float(seed_text)))]
            except ValueError:
                pass

        if self.expand_to.value() > 0:
            arguments += [
                "--expand-to", f"{self.expand_to.value():g}",
                "--expand-at-fs", f"{self.expand_at.value():g}",
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

    def build_continue_arguments(self):
        label = self.source_box.currentText()

        source = os.path.join(self.root, label)

        arguments = [
            "--continue-from", source,
            "--ps", f"{self.picoseconds.value():g}",
            "--cool-temperature",
            f"{self.cool_temperature.value():g}",
            "--capture-every", str(int(self.capture_every.value())),
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

        safe = self.mixture_box.currentText().replace(
            " ", "_"
        ).replace("+", "plus")

        parts = [
            safe,
            f"box{self.box_size.value():g}",
            f"{self.picoseconds.value():g}ps",
        ]

        if self.expand_to.value() > 0:
            parts.append(f"to{self.expand_to.value():g}")

        if self.strikes.value() > 0:
            parts.append(
                f"{int(self.strikes.value())}strikes"
                f"{self.strike_temperature.value() / 1000:g}k"
            )
        else:
            parts.append("quiet")

        if self.cool_temperature.value() != 250.0:
            parts.append(f"cool{self.cool_temperature.value():g}")

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
            start = (max(seeds) + 1) if seeds else 0

        while start in seeds:
            start += 1

        return start, seeds

    def on_queue(self):
        out = self.target_folder()

        if (
            self.parallel.isChecked()
            and self.mode_box.currentIndex() == 0
        ):
            self.queue_in_parts(out)
            return

        if self.mode_box.currentIndex() == 1:
            source = read_index(
                os.path.join(self.root, self.source_box.currentText())
            )

            total = sum(
                1 for entry in source
                if entry.get("stable") is not False
            )
        else:
            total = len(read_index(out)) + int(self.seeds.value())

        job = Job(
            name=os.path.basename(out),
            arguments=self.build_arguments(),
            out=out,
            runs=total,
        )

        self.jobs.append(job)

        self.save_queue()
        self.draw_jobs()

        self.tabs.setCurrentIndex(1)

    def queue_in_parts(self, out):
        # One condition, several processes, one folder.
        #
        # Each part is given its own block of seeds so nothing is
        # repeated, and the runs write separate entry files rather
        # than a shared index, so they cannot overwrite each
        # other's results.

        wanted = int(self.seeds.value())

        parts = max(1, min(self.concurrency, wanted))

        start, seeds = self.next_free_seed(out)

        existing = len(seeds)

        share = wanted // parts
        extra = wanted % parts

        cursor = start

        for part in range(parts):
            count = share + (1 if part < extra else 0)

            if count <= 0:
                continue

            arguments = self.build_arguments()

            # Replace whatever seed count and starting seed the
            # form produced with this part's own block.

            arguments = self.replaced(arguments, "--seeds", str(count))
            arguments = self.replaced(
                arguments, "--first-seed", str(cursor)
            )

            if "--out" not in arguments:
                arguments += ["--out", out]

            job = Job(
                name=f"{os.path.basename(out)}  part {part + 1}",
                arguments=arguments,
                out=out,
                runs=existing + wanted,
            )

            self.jobs.append(job)

            cursor += count

        self.save_queue()
        self.draw_jobs()

        self.tabs.setCurrentIndex(1)

    def replaced(self, arguments, flag, value):
        arguments = list(arguments)

        if flag in arguments:
            arguments[arguments.index(flag) + 1] = value
        else:
            arguments += [flag, value]

        return arguments

    # --------------------------------------------------------
    # Batches tab

    def build_batches_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        controls = QtWidgets.QHBoxLayout()

        controls.addWidget(QtWidgets.QLabel("run at once"))

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

        layout.addLayout(controls)

        self.jobs_table = QtWidgets.QTableWidget()
        self.jobs_table.setColumnCount(7)
        self.jobs_table.setHorizontalHeaderLabels([
            "batch", "state", "progress", "runs",
            "elapsed", "left", "recent",
        ])
        self.jobs_table.horizontalHeader().setStretchLastSection(True)
        self.jobs_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.jobs_table.verticalHeader().setVisible(False)

        layout.addWidget(self.jobs_table)

        return page

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

        command = [
            sys.executable, "batch_runner.py"
        ] + job.arguments

        if "--out" not in job.arguments:
            command += ["--out", job.out]

        job.process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        job.pid = job.process.pid
        job.reattached = False

        job.state = "running"
        job.started = time.time()

    def tick(self):
        for job in self.jobs:
            if job.state != "running":
                continue

            job.refresh()

            if job.process is not None:
                code = job.process.poll()

                if code is not None:
                    job.finished = time.time()

                    job.state = "done" if code == 0 else "failed"

            elif job.reattached:
                # Started by an earlier session, so there is no
                # handle to poll. The lock file it wrote says
                # whether it is still alive.

                state, lock = running.state_of(job.out)

                if state != "running":
                    job.finished = time.time()

                    job.state = (
                        "done" if job.completed >= job.runs
                        else "stopped"
                    )

        if not self.queue_paused:
            running = sum(
                1 for job in self.jobs if job.state == "running"
            )

            for job in self.jobs:
                if running >= self.concurrency:
                    break

                if job.state == "queued":
                    self.start_job(job)
                    running += 1

        self.draw_jobs()

    def draw_jobs(self):
        self.jobs_table.setRowCount(len(self.jobs))

        for row, job in enumerate(self.jobs):
            if job.state == "running" and job.fraction > 0.01:
                remaining = clock(
                    job.elapsed / job.fraction - job.elapsed
                )
            else:
                remaining = "-"

            bar = "#" * int(job.fraction * 18)
            bar += "-" * (18 - len(bar))

            values = [
                job.name,
                job.state,
                f"[{bar}] {job.fraction:3.0%}",
                f"{job.completed}/{job.runs}",
                clock(job.elapsed) if job.started else "-",
                remaining,
                "   ".join(job.headlines[-2:]),
            ]

            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))

                if column in (2, 6):
                    item.setFont(QtGui.QFont("Consolas", 9))

                if job.state == "failed":
                    item.setForeground(QtGui.QColor("#b4471f"))
                elif job.state == "stopped":
                    item.setForeground(QtGui.QColor("#9a6b1f"))
                elif job.state == "done":
                    item.setForeground(QtGui.QColor("#1d7a55"))

                self.jobs_table.setItem(row, column, item)

        self.jobs_table.resizeColumnsToContents()

    # --------------------------------------------------------
    # Results tab

    def build_results_tab(self):
        page = QtWidgets.QWidget()
        columns = QtWidgets.QHBoxLayout(page)

        left = QtWidgets.QVBoxLayout()

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("batch"))

        self.results_batch = QtWidgets.QComboBox()
        self.results_batch.currentIndexChanged.connect(
            self.on_pick_batch
        )
        row.addWidget(self.results_batch, stretch=1)

        left.addLayout(row)

        self.results_list = QtWidgets.QListWidget()
        self.results_list.currentRowChanged.connect(
            self.on_pick_run
        )

        font = QtGui.QFont("Consolas")
        font.setPointSize(10)
        self.results_list.setFont(font)

        left.addWidget(self.results_list, stretch=1)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.button("Refresh", self.reload_results))

        self.structures_button = self.button(
            "Structures: off", self.on_toggle_structures
        )
        row.addWidget(self.structures_button)

        left.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(
            self.button("Open in viewer", self.on_open_viewer)
        )
        row.addWidget(
            self.button("Dashboard", self.on_dashboard)
        )
        left.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(
            self.button("Compare batches", self.on_compare)
        )
        row.addWidget(
            self.button("Export CSV", self.on_export)
        )
        left.addLayout(row)

        columns.addLayout(left, stretch=2)

        right = QtWidgets.QVBoxLayout()

        self.results_title = QtWidgets.QLabel("select a run")
        right.addWidget(self.results_title)

        self.results_report = QtWidgets.QPlainTextEdit()
        self.results_report.setReadOnly(True)
        self.results_report.setLineWrapMode(
            QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap
        )
        self.results_report.setFont(font)

        right.addWidget(self.results_report, stretch=1)

        self.results_plot = pg.PlotWidget()
        self.results_plot.setLabel("left", "potential energy (eV)")
        self.results_plot.setLabel("bottom", "fs")
        self.results_plot.setMaximumHeight(180)
        self.results_curve = self.results_plot.plot(pen="#2f6f9f")

        right.addWidget(self.results_plot)

        columns.addLayout(right, stretch=3)

        self.want_structures = False
        self.results_paths = []

        self.reload_results()

        return page

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

        if position < 0 or position >= len(self.batches):
            return

        label, path = self.batches[position]

        for entry in read_index(path):
            mark = " " if entry.get("stable", True) else "!"

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

    def on_pick_run(self, row):
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

        self.results_curve.setData(
            np.array(recorder.times), np.array(recorder.potential)
        )

    def on_open_viewer(self):
        row = self.results_list.currentRow()

        if row < 0 or row >= len(self.results_paths):
            return

        subprocess.Popen([
            sys.executable, "run_reactive_gl.py",
            "--load", os.path.abspath(self.results_paths[row]),
        ])

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
        dialog = MixtureDialog(self)

        if dialog.exec():
            name, kind, contents = dialog.result()

            if not name or not contents:
                return

            custom = mixtures.load_custom()
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
            "expand_to": self.expand_to.value(),
            "expand_at": self.expand_at.value(),
            "strikes": self.strikes.value(),
            "strike_temperature": self.strike_temperature.value(),
            "strike_dissociation": self.strike_dissociation.value(),
            "first_strike": self.first_strike.value(),
            "strike_interval": self.strike_interval.value(),
            "capture_every": self.capture_every.value(),
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
            (self.expand_to, "expand_to"),
            (self.expand_at, "expand_at"),
            (self.strikes, "strikes"),
            (self.strike_temperature, "strike_temperature"),
            (self.strike_dissociation, "strike_dissociation"),
            (self.first_strike, "first_strike"),
            (self.strike_interval, "strike_interval"),
            (self.capture_every, "capture_every"),
        ]

        for widget, key in pairs:
            if key in stored:
                widget.setValue(stored[key])

        self.first_seed.setCurrentText(
            str(stored.get("first_seed", "continue automatically"))
        )

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
            if job.state not in ("running", "queued"):
                continue

            state, lock = running.state_of(job.out)

            if state == "running":
                job.state = "running"
                job.reattached = True
                job.pid = (lock or {}).get("pid", job.pid)
                job.started = (lock or {}).get("started", time.time())
                job.refresh()

                alive.append(job.name)
            elif state == "stale":
                # A lock left behind by something that died.

                running.remove_lock(job.out)

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


class MixtureDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("New mixture")
        self.resize(420, 300)

        layout = QtWidgets.QFormLayout(self)

        self.name = QtWidgets.QLineEdit()
        layout.addRow("name", self.name)

        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["atoms", "molecules"])
        layout.addRow("kind", self.kind)

        self.contents = QtWidgets.QPlainTextEdit()
        self.contents.setPlaceholderText(
            "one per line, for example\n\n"
            "C 80\nH 200\nN 20\nO 30\n\n"
            "or for molecules\n\nCH4 6\nNH3 4\nH2O 6\nH2 8"
        )
        layout.addRow("contents", self.contents)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addRow(buttons)

    def result(self):
        contents = {}

        for line in self.contents.toPlainText().splitlines():
            parts = line.split()

            if len(parts) != 2:
                continue

            try:
                contents[parts[0]] = int(parts[1])
            except ValueError:
                continue

        return (
            self.name.text().strip(),
            self.kind.currentText(),
            contents,
        )


def main():
    pg.setConfigOptions(antialias=True)

    application = QtWidgets.QApplication(sys.argv)

    window = Lab()
    window.show()

    sys.exit(application.exec())


if __name__ == "__main__":
    main()