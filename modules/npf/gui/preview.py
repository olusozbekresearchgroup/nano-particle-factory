from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ase.data import atomic_numbers
from ase.data import colors as ase_colors
from ase.io import read
from ase.neighborlist import natural_cutoffs, neighbor_list

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget


def build_cluster_preview(structure_path: str | Path) -> dict[str, Any]:
    path = Path(structure_path).expanduser().resolve()
    if path.suffix.lower() == ".data":
        atoms_obj = read(str(path), format="lammps-data")
    else:
        atoms_obj = read(str(path))
    return build_cluster_preview_from_atoms(atoms_obj, path=str(path))


def build_cluster_preview_from_atoms(atoms_obj, path: str | None = None) -> dict[str, Any]:
    positions = atoms_obj.get_positions().tolist()
    symbols = atoms_obj.get_chemical_symbols()
    return {
        "positions": positions,
        "symbols": symbols,
        "bonds": build_bonds(atoms_obj),
        "center": atoms_obj.get_positions().mean(axis=0).tolist(),
        "path": str(path) if path is not None else "",
    }


def build_bonds(atoms_obj):
    if len(atoms_obj) < 2:
        return []
    try:
        cutoffs = natural_cutoffs(atoms_obj, mult=1.12)
        first, second, shifts = neighbor_list("ijS", atoms_obj, cutoffs)
    except (ValueError, RuntimeError):
        return []
    return [
        (int(i), int(j))
        for i, j, shift in zip(first, second, shifts, strict=False)
        if int(i) < int(j) and not shift.any()
    ]


class ClusterPreviewView(QWidget):
    def __init__(self, parent=None, empty_message: str = "Geometry preview appears here."):
        super().__init__(parent)
        self.setMinimumHeight(520)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAutoFillBackground(False)
        self._preview: dict[str, Any] | None = None
        self._rot_x = -0.45
        self._rot_y = 0.65
        self._zoom = 1.0
        self._atom_scale = 1.0
        self._show_bonds = True
        self._bond_radius = 1.25
        self._depth_cue = 1.0
        self._projection = "orthographic"
        self._surface_indices: set[int] = set()
        self._marked_atoms: dict[str, set[int]] = {}
        self._mark_colors: dict[str, QColor] = {}
        self._movie_frames: list[dict[str, Any]] = []
        self._movie_index = 0
        self._dragging = False
        self._last_pos = None
        self._empty_message = empty_message

    def _atom_color(self, symbol: str) -> QColor:
        try:
            z = int(atomic_numbers.get(symbol, 0))
        except Exception:
            z = 0
        color = ase_colors.jmol_colors[z]
        r = max(0, min(255, int(float(color[0]) * 255)))
        g = max(0, min(255, int(float(color[1]) * 255)))
        b = max(0, min(255, int(float(color[2]) * 255)))
        if r == 0 and g == 0 and b == 0:
            return QColor("#b6bec9")
        return QColor(r, g, b)

    def _mix(self, a: QColor, b: QColor, t: float) -> QColor:
        t = max(0.0, min(1.0, t))
        return QColor(
            int(a.red() + (b.red() - a.red()) * t),
            int(a.green() + (b.green() - a.green()) * t),
            int(a.blue() + (b.blue() - a.blue()) * t),
        )

    def _clamp_alpha(self, value: float | int) -> int:
        return max(0, min(255, int(round(value))))

    def _rotate_point(self, pos):
        import math

        x, y, z = pos
        sx, cx = math.sin(self._rot_x), math.cos(self._rot_x)
        sy, cy = math.sin(self._rot_y), math.cos(self._rot_y)
        y1 = y * cx - z * sx
        z1 = y * sx + z * cx
        x2 = x * cy + z1 * sy
        z2 = -x * sy + z1 * cy
        return x2, y1, z2

    def _project(self, pos, scale: float):
        x, y, z = self._rotate_point(pos)
        if self._projection == "perspective":
            perspective = 1.0 / (1.0 + max(-0.85, z / (scale * 8.5)))
            return x * scale * perspective, y * scale * perspective, z, perspective
        if self._projection == "oblique":
            skew = 0.42
            px = x + skew * z
            py = y - skew * 0.55 * z
            depth = 1.0 / (1.0 + max(-0.85, z / (scale * 8.5)))
            return px * scale, py * scale, z, depth
        return x * scale, y * scale, z, 1.0

    def _centered_positions(self):
        preview = self._preview or {}
        positions = preview.get("positions") or []
        center = preview.get("center") or [0, 0, 0]
        return [
            (
                pos[0] - center[0],
                pos[1] - center[1],
                pos[2] - center[2],
            )
            for pos in positions
        ]

    def clear_preview(self):
        self._preview = None
        self._movie_frames = []
        self._movie_index = 0
        self._surface_indices = set()
        self._marked_atoms = {}
        self._rot_x = -0.45
        self._rot_y = 0.65
        self._zoom = 1.0
        self.update()

    def set_atom_scale(self, value: float):
        self._atom_scale = max(0.2, float(value))
        self.update()

    def set_projection(self, projection: str):
        self._projection = projection if projection in {"orthographic", "perspective", "oblique"} else "orthographic"
        self.update()

    def set_show_bonds(self, enabled: bool):
        self._show_bonds = bool(enabled)
        self.update()

    def set_bond_radius(self, value: float):
        self._bond_radius = max(0.5, float(value))
        self.update()

    def set_depth_cue(self, value: float):
        self._depth_cue = max(0.0, min(3.0, float(value)))
        self.update()

    def set_cluster(self, preview: dict[str, Any]):
        self._preview = preview
        if not self._movie_frames:
            self._movie_index = 0
        self.update()

    def set_movie_frames(self, frames: list[dict[str, Any]] | None):
        self._movie_frames = list(frames or [])
        self._movie_index = 0
        if self._movie_frames:
            self._preview = self._movie_frames[0]
        self.update()

    def movie_frame_count(self) -> int:
        return len(self._movie_frames)

    def set_movie_frame_index(self, index: int):
        if not self._movie_frames:
            return
        self._movie_index = max(0, min(int(index), len(self._movie_frames) - 1))
        self._preview = self._movie_frames[self._movie_index]
        self.update()

    def movie_frame_index(self) -> int:
        return self._movie_index

    def set_surface_indices(self, indices):
        self._surface_indices = {int(i) for i in (indices or [])}
        self._marked_atoms["surface atoms"] = set(self._surface_indices)
        self.update()

    def set_marked_atoms(self, marks: dict[str, list[int] | set[int] | tuple[int, ...]] | None):
        self._marked_atoms = {
            str(name): {int(i) for i in (indices or [])}
            for name, indices in (marks or {}).items()
        }
        self._surface_indices = set(self._marked_atoms.get("surface atoms", set()))
        self.update()

    def set_mark_colors(self, colors: dict[str, QColor | str | tuple[int, int, int]] | None):
        self._mark_colors = {}
        for name, value in (colors or {}).items():
            if isinstance(value, QColor):
                color = QColor(value)
            elif isinstance(value, tuple) and len(value) == 3:
                color = QColor(*value)
            else:
                color = QColor(str(value))
            if color.isValid():
                self._mark_colors[str(name)] = color
        self.update()

    def load_structure_file(self, structure_path: str | Path):
        self.set_cluster(build_cluster_preview(structure_path))

    def paintEvent(self, event):  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing, True)
        rect = self.rect().adjusted(0, 0, -1, -1)

        background = QRadialGradient(rect.center(), max(rect.width(), rect.height()) * 0.8)
        background.setColorAt(0.0, QColor("#101725"))
        background.setColorAt(0.72, QColor("#080b12"))
        background.setColorAt(1.0, QColor("#05070c"))
        painter.fillRect(rect, background)

        if not self._preview:
            painter.setPen(QColor("#9aa6bb"))
            painter.drawText(rect.adjusted(18, 18, -18, -18), Qt.AlignTop | Qt.AlignLeft | Qt.TextWordWrap, self._empty_message)
            painter.end()
            return

        positions = self._centered_positions()
        if not positions:
            painter.setPen(QColor("#9aa6bb"))
            painter.drawText(rect.adjusted(18, 18, -18, -18), Qt.AlignTop | Qt.AlignLeft | Qt.TextWordWrap, self._empty_message)
            painter.end()
            return

        symbols = self._preview.get("symbols") or []
        bonds = self._preview.get("bonds") or []
        cell_edges = self._preview.get("cell_edges") or []
        max_r = max(max((x * x + y * y + z * z) ** 0.5 for x, y, z in positions), 1e-6)
        scale = min(rect.width(), rect.height()) * 0.34 / max_r * self._zoom
        cx = rect.center().x()
        cy = rect.center().y()
        center = self._preview.get("center") or [0, 0, 0]
        projected = []
        for i, pos in enumerate(positions):
            x, y, z, depth = self._project(pos, scale)
            projected.append(
                {
                    "i": i,
                    "x": cx + x,
                    "y": cy + y,
                    "z": z,
                    "depth": depth,
                    "symbol": symbols[i] if i < len(symbols) else "Ru",
                }
            )

        if cell_edges:
            pen_back = QPen(QColor(120, 136, 156, 120))
            pen_back.setWidthF(1.2)
            pen_front = QPen(QColor(182, 196, 214, 165))
            pen_front.setWidthF(1.8)
            for p1, p2 in cell_edges:
                rp1 = (p1[0] - center[0], p1[1] - center[1], p1[2] - center[2])
                rp2 = (p2[0] - center[0], p2[1] - center[1], p2[2] - center[2])
                x1, y1, z1, _d1 = self._project(rp1, scale)
                x2, y2, z2, _d2 = self._project(rp2, scale)
                depth = (z1 + z2) * 0.5
                painter.setPen(pen_back if depth < 0 else pen_front)
                painter.drawLine(QPointF(cx + x1, cy + y1), QPointF(cx + x2, cy + y2))

        if self._show_bonds:
            projected_by_index = {atom["i"]: atom for atom in projected}
            bond_depth_rank = {atom["i"]: idx / max(1, len(projected) - 1) for idx, atom in enumerate(projected)}
            for i, j in bonds:
                a = projected_by_index[i]
                b = projected_by_index[j]
                cue = max(0.0, self._depth_cue)
                backness = (bond_depth_rank.get(i, 0.0) + bond_depth_rank.get(j, 0.0)) * 0.5
                depth = max(0.10, min(1.0, 1.0 - cue * backness))
                dx = b["x"] - a["x"]
                dy = b["y"] - a["y"]
                length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
                px, py = -dy / length, dx / length
                width = max(1.8, 2.15 * self._bond_radius * (0.55 + 0.55 * depth))

                shadow_pen = QPen(QColor(22, 27, 36, int(34 + 55 * depth)))
                shadow_pen.setCapStyle(Qt.RoundCap)
                shadow_pen.setJoinStyle(Qt.RoundJoin)
                shadow_pen.setWidthF(width * 1.25)
                painter.setPen(shadow_pen)
                painter.drawLine(
                    QPointF(a["x"] + px * width * 0.18, a["y"] + py * width * 0.18),
                    QPointF(b["x"] + px * width * 0.18, b["y"] + py * width * 0.18),
                )

                body_pen = QPen(QColor(205, 219, 236, int(80 + 135 * depth)))
                body_pen.setCapStyle(Qt.RoundCap)
                body_pen.setJoinStyle(Qt.RoundJoin)
                body_pen.setWidthF(width)
                painter.setPen(body_pen)
                painter.drawLine(QPointF(a["x"], a["y"]), QPointF(b["x"], b["y"]))

                highlight_pen = QPen(QColor(248, 251, 255, int(45 + 75 * depth)))
                highlight_pen.setCapStyle(Qt.RoundCap)
                highlight_pen.setJoinStyle(Qt.RoundJoin)
                highlight_pen.setWidthF(max(1.0, width * 0.42))
                painter.setPen(highlight_pen)
                painter.drawLine(
                    QPointF(a["x"] - px * width * 0.14, a["y"] - py * width * 0.14),
                    QPointF(b["x"] - px * width * 0.14, b["y"] - py * width * 0.14),
                )

        projected.sort(key=lambda atom: atom["z"])
        radii_values = self._preview.get("radii") or []
        radius_scale_by_atom: dict[int, float] = {}
        has_physical_radii = isinstance(radii_values, list) and len(radii_values) == len(projected)
        if has_physical_radii:
            valid = [float(v) for v in radii_values if float(v) > 1e-8]
            ref = sum(valid) / len(valid) if valid else 1.0
            for atom_i, v in enumerate(radii_values):
                scale = float(v) / ref if ref > 1e-8 else 1.0
                radius_scale_by_atom[atom_i] = max(0.6, min(1.8, scale))
        else:
            for atom_i in range(len(projected)):
                radius_scale_by_atom[atom_i] = 1.0

        mark_palette = [
            # Specific terrace interiors take priority over overlapping marks.
            ("3-fold interior atoms", QColor("#69d1b0"), "3i"),
            ("4-fold square interior atoms", QColor("#69d1b0"), "4i"),
            ("4-fold rectangular interior atoms", QColor("#69d1b0"), "4i"),
            ("surface atoms", QColor("#ffb347"), "X"),
            ("on-top (B1) sites", QColor("#8ad8ff"), "1"),
            ("bridge (B2) sites", QColor("#7ef0c6"), "2"),
            ("3-fold (B3) sites", QColor("#b58cff"), "3"),
            ("4-fold (B4) square sites", QColor("#ff8f8f"), "4"),
            ("4-fold (B4) rectangular sites", QColor("#ffca73"), "4"),
            ("B5 sites", QColor("#f2c94c"), "5"),
        ]
        for idx, atom in enumerate(projected):
            cue = max(0.0, self._depth_cue)
            backness = idx / max(1, len(projected) - 1)
            depth = max(0.02, min(1.0, 1.0 - cue * backness))
            depth_factor = 1.0 if has_physical_radii else (0.78 + 0.26 * depth)
            atom_scale = radius_scale_by_atom.get(atom["i"], 1.0)
            radius = max(8.0, 13.5 * self._atom_scale * atom_scale * depth_factor)
            fill = self._atom_color(atom["symbol"])
            label = None
            marked = False
            for name, default_color, short_label in mark_palette:
                if atom["i"] in self._marked_atoms.get(name, set()):
                    fill = self._mark_colors.get(name, default_color)
                    label = short_label
                    marked = True
                    break

            shell = fill.darker(int(200 + 30 * cue * backness) if not marked else int(165 + 14 * cue * backness))
            mid = self._mix(fill, QColor("#ffffff"), min(0.30, (0.08 if not marked else 0.22) + 0.08 * cue * (1.0 - backness)))
            highlight = self._mix(fill, QColor("#ffffff"), min(0.98, (0.94 if marked else 0.82) - 0.14 * backness + 0.05 * cue))
            shadow = QColor(0, 0, 0, int(18 + (24 + 18 * cue) * backness))
            fill_alpha = QColor(fill)
            fill_alpha.setAlpha(self._clamp_alpha(255 if marked else 150 + (60 + 70 * cue) * depth))
            painter.setPen(Qt.NoPen)
            painter.setBrush(shadow)
            painter.drawEllipse(QPointF(atom["x"] + radius * 0.18, atom["y"] + radius * 0.22), radius * 1.0, radius * 0.9)

            body = QRadialGradient(QPointF(atom["x"] - radius * 0.34, atom["y"] - radius * 0.34), radius * 1.18)
            body.setColorAt(0.0, QColor(255, 255, 255, 170))
            body.setColorAt(0.08, QColor(highlight.red(), highlight.green(), highlight.blue(), 240 if marked else 205))
            body.setColorAt(0.30, QColor(mid.red(), mid.green(), mid.blue(), 250 if marked else 220))
            body.setColorAt(0.78, QColor(shell.red(), shell.green(), shell.blue(), 245))
            body.setColorAt(1.0, QColor(shell.red(), shell.green(), shell.blue(), 255))
            painter.setBrush(body)
            painter.setPen(QPen(QColor("#dfe7f2"), max(0.9, radius * 0.075)))
            painter.drawEllipse(QPointF(atom["x"], atom["y"]), radius, radius)

            painter.setBrush(QColor(255, 255, 255, 135 if marked else int(50 + 40 * depth)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(atom["x"] - radius * 0.24, atom["y"] - radius * 0.25), radius * 0.58, radius * 0.58)

            painter.setBrush(fill_alpha)
            painter.drawEllipse(QPointF(atom["x"] + radius * 0.10, atom["y"] + radius * 0.08), radius * 0.80, radius * 0.80)
            if label is not None:
                painter.setPen(QColor("#fff7ea"))
                painter.drawText(
                    QRectF(atom["x"] - radius, atom["y"] - radius, radius * 2, radius * 2),
                    Qt.AlignCenter,
                    label,
                )

        painter.setPen(QColor("#9aa6bb"))
        painter.drawText(
            rect.adjusted(18, 18, -18, -18),
            Qt.AlignTop | Qt.AlignLeft,
            f"atoms {len(positions)} | radius {self._atom_scale:.2f}x | bond radius {self._bond_radius:.2f}x | projection {self._projection}",
        )

        symbols = self._preview.get("symbols") or []
        counts = Counter(symbols)
        if counts:
            legend_x = rect.right() - 190
            legend_y = rect.top() + 16
            legend_w = 170
            line_h = 18
            legend_h = 26 + line_h * len(counts)
            painter.setPen(QPen(QColor(44, 56, 74), 1.0))
            painter.setBrush(QColor(10, 14, 22, 170))
            painter.drawRoundedRect(QRectF(legend_x, legend_y, legend_w, legend_h), 6, 6)
            painter.setPen(QColor("#c9d1e3"))
            painter.drawText(QRectF(legend_x + 8, legend_y + 6, legend_w - 16, 16), Qt.AlignLeft | Qt.AlignVCenter, "Species legend")
            row = 0
            for sym in sorted(counts):
                y = legend_y + 24 + row * line_h
                color = self._atom_color(sym)
                painter.setPen(QPen(QColor(26, 34, 47), 1.0))
                painter.setBrush(color)
                painter.drawEllipse(QPointF(legend_x + 14, y + 7), 5.0, 5.0)
                painter.setPen(QColor("#c9d1e3"))
                painter.drawText(QRectF(legend_x + 26, y, legend_w - 34, line_h), Qt.AlignLeft | Qt.AlignVCenter, f"{sym} x {counts[sym]}")
                row += 1
        painter.end()

    def wheelEvent(self, event):  # type: ignore[override]
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom *= 1.08
        else:
            self._zoom *= 0.92
        self._zoom = max(0.35, min(self._zoom, 3.0))
        self.update()

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._last_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if self._dragging and self._last_pos is not None:
            delta = event.position() - self._last_pos
            self._last_pos = event.position()
            self._rot_y += float(delta.x()) * 0.006
            self._rot_x += float(delta.y()) * 0.006
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self._last_pos = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def reset_view(self):
        self._rot_x = -0.45
        self._rot_y = 0.65
        self._zoom = 1.0
        self._atom_scale = 1.0
        self._projection = "orthographic"
        self.update()
