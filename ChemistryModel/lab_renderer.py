"""Shared OpenGL atom, bond, camera, and picking scene for Chemistry Lab."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


ELEMENT_COLOUR = {
    "H": (0.92, 0.92, 0.92, 1.0),
    "C": (0.28, 0.28, 0.30, 1.0),
    "N": (0.19, 0.31, 0.97, 1.0),
    "O": (0.90, 0.16, 0.13, 1.0),
}
ELEMENT_SIZE = {"H": 0.42, "C": 0.72, "N": 0.68, "O": 0.64}


class AtomView(gl.GLViewWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atom_positions = np.empty((0, 3))
        self.atom_clicked = None
        self._press_position = None

    def mousePressEvent(self, event):
        # GLViewWidget must always see the press so it can initialise the
        # camera drag position. Consuming it for atom picking made the next
        # move use a stale coordinate and snap the camera to an extreme.
        self._press_position = event.position()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        press = self._press_position
        self._press_position = None
        moved = (
            press is None
            or np.hypot(
                event.position().x() - press.x(),
                event.position().y() - press.y(),
            ) > 4.0
        )
        super().mouseReleaseEvent(event)
        if not moved:
            selected = self.atom_at_screen_position(event.position())
            if selected is not None and self.atom_clicked is not None:
                self.atom_clicked(selected, event.modifiers())

    def atom_at_screen_position(self, cursor):
        if not len(self.atom_positions):
            return None
        # pyqtgraph 0.14 requires both the projected region and OpenGL
        # viewport. For a click in the full widget those are the same.
        viewport = self.getViewport()
        matrix = self.projectionMatrix(viewport, viewport) * self.viewMatrix()
        screen = []
        visible = []
        for index, point in enumerate(self.atom_positions):
            vector = matrix.map(QtGui.QVector4D(
                float(point[0]), float(point[1]), float(point[2]), 1.0
            ))
            if vector.w() <= 0:
                continue
            x = (vector.x() / vector.w() + 1.0) * self.width() * 0.5
            y = (1.0 - vector.y() / vector.w()) * self.height() * 0.5
            screen.append((x, y))
            visible.append(index)
        if not screen:
            return None
        distances = np.linalg.norm(
            np.asarray(screen) - np.array([cursor.x(), cursor.y()]), axis=1
        )
        closest = int(np.argmin(distances))
        # A slightly generous target is intentional: atoms are small in world
        # units and exact pixel picking is frustrating on high-DPI displays.
        return int(visible[closest]) if distances[closest] <= 30.0 else None


class MolecularScene(QtWidgets.QWidget):
    atomClicked = QtCore.Signal(int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = AtomView()
        self.view.atom_clicked = self.atomClicked.emit
        layout.addWidget(self.view)
        self.bond_item = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), color=(0.65, 0.65, 0.65, 0.85),
            width=3.0, mode="lines", antialias=True,
        )
        self.view.addItem(self.bond_item)
        self.scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), color=np.ones((1, 4)),
            size=np.ones(1), pxMode=False,
        )
        # Match the standalone viewer: bonds are submitted first and opaque
        # atoms second, so atom discs cover the ends of their bond sticks.
        self.scatter.setGLOptions("opaque")
        self.view.addItem(self.scatter)
        self.box_lines = []
        self.box_size = 1.0
        self.positions = np.empty((0, 3))
        self.symbols = []
        self.selected = set()
        self.dim_unselected = False
        self.highlight_selected = False

    def set_state(self, positions, symbols, box_size, bonds=((), ())):
        self.positions = np.asarray(positions, dtype=float) % float(box_size)
        self.symbols = [str(symbol) for symbol in symbols]
        box_changed = abs(float(box_size) - self.box_size) > 1e-6
        self.box_size = float(box_size)
        self.view.atom_positions = self.positions
        self.refresh_atoms()
        self.set_bonds(*bonds)
        if box_changed:
            self.add_box_outline()

    def refresh_atoms(self):
        colours = np.array([
            ELEMENT_COLOUR.get(symbol, (0.7, 0.7, 0.7, 1.0))
            for symbol in self.symbols
        ], dtype=np.float32)
        sizes = np.array([
            ELEMENT_SIZE.get(symbol, 0.6) for symbol in self.symbols
        ], dtype=np.float32)
        if self.dim_unselected and self.selected:
            # The atom pass is deliberately opaque for correct bond depth.
            # Dim using colour intensity rather than transparency.
            original = colours.copy()
            colours[:, :3] *= 0.12
            for index in self.selected:
                if 0 <= index < len(colours):
                    colours[index] = original[index]
        if self.highlight_selected:
            for index in self.selected:
                if 0 <= index < len(colours):
                    colours[index] = (1.0, 0.82, 0.08, 1.0)
                    sizes[index] *= 1.45
        self.scatter.setData(
            pos=self.positions, color=colours, size=sizes, pxMode=False
        )

    def set_bonds(self, first, second):
        first = np.asarray(first, dtype=int)
        second = np.asarray(second, dtype=int)
        if not len(first):
            self.bond_item.setData(pos=np.zeros((2, 3)))
            return
        start = self.positions[first]
        offset = self.positions[second] - start
        offset -= self.box_size * np.round(offset / self.box_size)
        keep = np.linalg.norm(offset, axis=1) < self.box_size / 2.0
        kept_first = first[keep]
        kept_second = second[keep]
        start = start[keep]
        offset = offset[keep]
        lengths = np.linalg.norm(offset, axis=1)
        direction = offset / np.maximum(lengths[:, None], 1e-12)
        start_radius = np.array([
            ELEMENT_SIZE.get(self.symbols[index], 0.6) * 0.42
            for index in kept_first
        ])
        end_radius = np.array([
            ELEMENT_SIZE.get(self.symbols[index], 0.6) * 0.42
            for index in kept_second
        ])
        # Very short/overlapping contacts still need a visible segment.
        scale = np.minimum(
            1.0,
            lengths / np.maximum(start_radius + end_radius + 1e-12, 1e-12),
        )
        start_radius *= scale * 0.92
        end_radius *= scale * 0.92
        segments = np.empty((2 * len(start), 3), dtype=np.float32)
        segments[0::2] = start + direction * start_radius[:, None]
        segments[1::2] = start + offset - direction * end_radius[:, None]
        self.bond_item.setData(pos=segments, mode="lines")

    def add_box_outline(self):
        for item in self.box_lines:
            self.view.removeItem(item)
        self.box_lines = []
        size = self.box_size
        corners = np.array([
            [x, y, z]
            for x in (0, size) for y in (0, size) for z in (0, size)
        ], dtype=float)
        edges = [
            (a, b) for a in range(8) for b in range(a + 1, 8)
            if np.count_nonzero(corners[a] != corners[b]) == 1
        ]
        for a, b in edges:
            line = gl.GLLinePlotItem(
                pos=np.array([corners[a], corners[b]]),
                color=(0.45, 0.45, 0.45, 0.45), width=1.0,
                antialias=True,
            )
            self.view.addItem(line)
            self.box_lines.append(line)

    def recentre(self):
        half = self.box_size / 2.0
        self.view.opts["center"] = pg.Vector(half, half, half)
        self.view.setCameraPosition(distance=self.box_size * 2.2)
        self.view.update()
