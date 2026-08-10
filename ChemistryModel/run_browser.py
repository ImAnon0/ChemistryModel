import json
import os
import shutil
import subprocess
import sys

import numpy as np

import pyqtgraph as pg

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from recorder import Recorder

import analysis
import analysis_cache


# ============================================================
# Browsing a batch of runs
# ============================================================
#
#   py run_browser.py            reads ./runs
#   py run_browser.py other_dir  reads somewhere else
#
# The list on the left is every run in the index. Selecting one
# analyses its recording and shows what was made at the end, what
# appeared at any point during the run, and every heavy-atom bond
# that formed or broke.


RUNS_DIRECTORY = "runs"

# Frames are strided when analysing, since a full pass over tens
# of thousands of frames is wasted effort for a summary.

ANALYSIS_STRIDE = 4



def discover_batches(root):
    # Any folder containing an index.json counts as a batch. The
    # root itself is included, plus one level of subfolders, so
    # runs/, runs/quiet and runs/lightning all show up together.

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


def describe_batch(index):
    if not index:
        return "empty"

    first = index[0]

    strikes = {entry.get("strikes", 0) for entry in index}

    strike_text = (
        f"{min(strikes)} strikes" if len(strikes) == 1
        else f"{min(strikes)}-{max(strikes)} strikes"
    )

    return (
        f"{len(index)} runs   {first.get('mixture', '?')}   "
        f"{first.get('atoms', '?')} atoms   "
        f"{first.get('picoseconds', '?')} ps   {strike_text}"
    )


def compare_batches(batches):
    # Which closed-shell products appear, and in how many runs of
    # each batch. This is the whole reason for running more than
    # one box: it turns "it happened once" into a frequency.

    loaded = []

    for label, path in batches:
        with open(os.path.join(path, "index.json")) as handle:
            loaded.append((label, json.load(handle)))

    names = set()

    for label, index in loaded:
        for entry in index:
            names.update(entry.get("closed_shell", []))
            names.update(entry.get("final_species", []))

    lines = []

    lines.append("=" * 66)
    lines.append("  COMPARING BATCHES")
    lines.append("=" * 66)
    lines.append("")

    for label, index in loaded:
        lines.append(f"  {label:<16} {describe_batch(index)}")

    lines.append("")
    lines.append("")
    lines.append("-" * 66)
    lines.append("  CLOSED-SHELL PRODUCTS, runs containing each")
    lines.append("-" * 66)
    lines.append("")

    header = f"    {'formula':<12}"

    for label, index in loaded:
        header += f"{label[:12]:>14}"

    lines.append(header)
    lines.append("    " + "-" * (12 + 14 * len(loaded)))

    rows = []

    for name in names:
        counts = []

        for label, index in loaded:
            counts.append(sum(
                1 for entry in index
                if name in entry.get("closed_shell", [])
            ))

        if any(counts):
            rows.append((name, counts))

    rows.sort(key=lambda row: (-sum(row[1]), -len(row[0])))

    for name, counts in rows[:30]:
        line = f"    {name:<12}"

        for count, (label, index) in zip(counts, loaded):
            line += f"{count:>8d}/{len(index):<5d}"

        lines.append(line)

    if not rows:
        lines.append("    nothing closed-shell in any batch")

    lines.append("")
    lines.append("")
    lines.append("-" * 66)
    lines.append("  AVERAGES PER RUN")
    lines.append("-" * 66)
    lines.append("")
    lines.append(
        "  Matched conditions share the same box until the first"
    )
    lines.append(
        "  discharge, so only the 'late' columns can separate them."
    )
    lines.append("")

    columns = [
        ("heavy_bonds_formed", "bonds"),
        ("late_formed", "late formed"),
        ("late_broke", "late broke"),
        ("turnovers", "turnovers"),
        ("largest_closed", "closed"),
        ("largest_any", "any"),
        ("most_carbon", "carbons"),
        ("species_count", "species"),
    ]

    header = f"    {'batch':<20}"

    for key, title in columns:
        header += f"{title:>13}"

    lines.append(header)
    lines.append("    " + "-" * (20 + 13 * len(columns)))

    for label, index in loaded:
        row = f"    {label[:20]:<20}"

        for key, title in columns:
            values = [
                entry.get(key, 0) for entry in index
                if key in entry
            ]

            if values:
                row += f"{sum(values) / len(values):>13.1f}"
            else:
                row += f"{'-':>13}"

        lines.append(row)

    lines.append("")
    lines.append(
        "  A dash means the batch predates these measurements."
    )
    lines.append("  Run reindex.py to fill them in.")

    return lines


class Browser(QtWidgets.QWidget):

    def __init__(self, root=RUNS_DIRECTORY):
        super().__init__()

        self.root = root
        self.batches = discover_batches(root)

        self.directory = (
            self.batches[0][1] if self.batches else root
        )
        self.index = []
        self.current = None
        self.current_recorder = None
        self.current_result = None

        # Structure fingerprinting is worth about three and a half
        # times everything else put together, and it is only
        # needed when looking closely at one run rather than
        # clicking through many. Off by default, on when wanted.

        self.want_structures = False

        self.setWindowTitle(f"Runs - {os.path.abspath(root)}")
        self.resize(1500, 900)

        layout = QtWidgets.QHBoxLayout(self)

        # ---- left: the list ----

        left = QtWidgets.QVBoxLayout()

        self.batch_box = QtWidgets.QComboBox()

        for label, path in self.batches:
            self.batch_box.addItem(label)

        self.batch_box.currentIndexChanged.connect(
            self.on_switch_batch
        )

        left.addWidget(QtWidgets.QLabel("batch"))
        left.addWidget(self.batch_box)

        self.summary_label = QtWidgets.QLabel("")
        self.summary_label.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 12px;"
        )
        self.summary_label.setWordWrap(True)
        left.addWidget(self.summary_label)

        self.list_widget = QtWidgets.QListWidget()

        list_font = QtGui.QFont("Consolas")
        list_font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        list_font.setPointSize(10)
        self.list_widget.setFont(list_font)
        self.list_widget.currentRowChanged.connect(self.on_select)

        left.addWidget(self.list_widget, stretch=1)

        left.addWidget(
            self.button("Compare all batches", self.on_compare)
        )

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.button("Refresh", self.rescan))
        row.addWidget(
            self.button("Sort by size", self.sort_by_size)
        )
        row.addWidget(
            self.button("Sort by number", self.reload_index)
        )
        left.addLayout(row)

        self.open_button = self.button(
            "Open in viewer", self.open_in_viewer
        )
        left.addWidget(self.open_button)

        self.keep_button = self.button(
            "Save a copy...", self.save_copy
        )
        left.addWidget(self.keep_button)

        self.report_button = self.button(
            "Write report to file", self.write_report
        )
        left.addWidget(self.report_button)

        self.structures_button = self.button(
            "Structures: off", self.toggle_structures
        )
        left.addWidget(self.structures_button)

        left.addWidget(
            self.button("Clear cached results", self.on_clear_cache)
        )

        layout.addLayout(left, stretch=2)

        # ---- right: the report ----

        right = QtWidgets.QVBoxLayout()

        self.title = QtWidgets.QLabel("select a run")
        self.title.setStyleSheet("font-size: 15px;")
        right.addWidget(self.title)

        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setLineWrapMode(
            QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap
        )

        font = QtGui.QFont("Consolas")
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        font.setPointSize(10)
        self.report.setFont(font)
        right.addWidget(self.report, stretch=1)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "potential energy (eV)")
        self.plot.setLabel("bottom", "fs")
        self.plot.setMaximumHeight(200)
        self.energy_curve = self.plot.plot(pen="#4a90c4")

        right.addWidget(self.plot)

        layout.addLayout(right, stretch=4)

        self.reload_index()

    # --------------------------------------------------------

    def button(self, label, callback):
        widget = QtWidgets.QPushButton(label)
        widget.clicked.connect(callback)
        return widget

    def rescan(self):
        self.batches = discover_batches(self.root)

        current = self.batch_box.currentText()

        self.batch_box.blockSignals(True)
        self.batch_box.clear()

        for label, path in self.batches:
            self.batch_box.addItem(label)

        if current:
            position = self.batch_box.findText(current)

            if position >= 0:
                self.batch_box.setCurrentIndex(position)

        self.batch_box.blockSignals(False)

        if self.batches:
            index = max(self.batch_box.currentIndex(), 0)
            self.directory = self.batches[index][1]

        self.reload_index()

    def on_switch_batch(self, position):
        if 0 <= position < len(self.batches):
            self.directory = self.batches[position][1]
            self.reload_index()

    def on_compare(self):
        if len(self.batches) < 1:
            return

        self.title.setText(
            f"comparing {len(self.batches)} batches"
        )

        self.report.setPlainText(
            "\n".join(compare_batches(self.batches))
        )

    def index_path(self):
        return os.path.join(self.directory, "index.json")

    def reload_index(self):
        path = self.index_path()

        if not os.path.exists(path):
            self.summary_label.setText(
                f"no index.json in {os.path.abspath(self.directory)}\n"
                f"run batch_runner.py first"
            )
            self.list_widget.clear()
            return

        with open(path) as handle:
            self.index = json.load(handle)

        self.index.sort(key=lambda entry: entry["number"])

        self.populate()

    def sort_by_size(self):
        def biggest(entry):
            names = entry.get("closed_shell") or entry.get(
                "final_species", []
            )

            return -max(
                (len(name) for name in names), default=0
            )

        self.index.sort(key=biggest)
        self.populate()

    def populate(self):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()

        tally = {}

        for entry in self.index:
            for name in entry.get("closed_shell", []):
                tally[name] = tally.get(name, 0) + 1

            marker = "*" if entry.get("closed_shell") else " "

            self.list_widget.addItem(
                f"{marker} {entry['number']:03d}  "
                f"seed {entry['seed']:<4d} "
                f"{entry['headline']}"
            )

        self.list_widget.blockSignals(False)

        top = sorted(tally.items(), key=lambda item: -item[1])[:6]

        text = describe_batch(self.index) + "\n"

        if top:
            text += "closed-shell products:  " + "   ".join(
                f"{name} {number}/{len(self.index)}"
                for name, number in top
            )

        self.summary_label.setText(text)

        if self.index:
            self.list_widget.setCurrentRow(0)

    # --------------------------------------------------------

    def on_select(self, row):
        if row < 0 or row >= len(self.index):
            return

        entry = self.index[row]

        self.current = entry

        path = os.path.join(self.directory, entry["file"])

        if not os.path.exists(path):
            self.report.setPlainText(f"missing file: {path}")
            return

        self.title.setText(
            f"run {entry['number']:03d}   seed {entry['seed']}   "
            f"{entry['mixture']}   {entry['atoms']} atoms   "
            f"{entry.get('strikes', 0)} strikes"
        )

        self.report.setPlainText("analysing...")
        QtWidgets.QApplication.processEvents()

        recorder = Recorder.load(path)

        result = analysis_cache.analyse_cached(
            recorder,
            path,
            analysis.analyse,
            stride=ANALYSIS_STRIDE,
            structures=self.want_structures,
        )

        self.current_recorder = recorder
        self.current_result = result

        lines = analysis.summary_lines(result)

        if not self.want_structures:
            lines.insert(
                0,
                "  structures off - press the button on the left "
                "to identify isomers"
            )

        self.report.setPlainText("\n".join(lines))

        self.energy_curve.setData(
            np.array(recorder.times),
            np.array(recorder.potential)
        )

    # --------------------------------------------------------

    def toggle_structures(self):
        self.want_structures = not self.want_structures

        self.structures_button.setText(
            "Structures: on" if self.want_structures
            else "Structures: off"
        )

        row = self.list_widget.currentRow()

        if row >= 0:
            self.on_select(row)

    def on_clear_cache(self):
        removed = analysis_cache.clear(self.root)

        print(f"cleared {removed} cached results")

        row = self.list_widget.currentRow()

        if row >= 0:
            self.on_select(row)

    def open_in_viewer(self):
        if self.current is None:
            return

        path = os.path.abspath(
            os.path.join(self.directory, self.current["file"])
        )

        # A separate process, so the browser stays usable and a
        # crash in one does not take the other down.

        subprocess.Popen(
            [sys.executable, "run_reactive_gl.py", "--load", path]
        )

        print(f"opening {path} in the viewer")

    def save_copy(self):
        if self.current is None:
            return

        source = os.path.join(
            self.directory, self.current["file"]
        )

        suggested = (
            f"run_{self.current['number']:03d}_"
            f"{self.current['headline'].split()[0]}.npz"
        )

        target, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save a copy", suggested, "Recordings (*.npz)"
        )

        if target:
            shutil.copyfile(source, target)
            print(f"copied to {target}")

    def write_report(self):
        if self.current_result is None:
            return

        suggested = f"run_{self.current['number']:03d}_report.txt"

        target, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Write report", suggested, "Text (*.txt)"
        )

        if target:
            with open(target, "w") as handle:
                handle.write(self.title.text() + "\n\n")
                handle.write(
                    "\n".join(
                        analysis.summary_lines(self.current_result)
                    )
                )

            print(f"report written to {target}")


def main():
    pg.setConfigOptions(antialias=True)

    root = sys.argv[1] if len(sys.argv) > 1 else RUNS_DIRECTORY

    application = QtWidgets.QApplication(sys.argv[:1])

    browser = Browser(root)
    browser.show()

    sys.exit(application.exec())


if __name__ == "__main__":
    main()