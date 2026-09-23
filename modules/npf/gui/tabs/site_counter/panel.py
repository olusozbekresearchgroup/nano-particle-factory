# Copyright (c) 2026 Olus Ozbek Research Group
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from ase.io import read
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import QDir, QThread, QUrl, Signal, Qt
from PySide6.QtGui import QCursor, QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QMenu,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QSlider,
    QSizePolicy,
    QListWidget,
    QListWidgetItem,
    QFileSystemModel,
    QTreeView,
    QToolTip,
    QVBoxLayout,
    QWidget,
    QDialog,
    QColorDialog,
)

from ...preview import ClusterPreviewView
from ....common import _average_nn_distance, _write_xyz_file, _xyz_output_path
from ....count.utils import average_nearest_distance, build_neighbor_graph, distances_within
from ....count.site_count import TERRACE_TYPES


_TERRACE_MARKS = {
    "3-fold interior atoms": "hollow3",
    "4-fold square interior atoms": "hollow4_square",
    "4-fold rectangular interior atoms": "hollow4_rect",
}


def _settings_from_config(config_path: str, overrides: dict[str, Any]) -> dict[str, Any]:
    from ....count.config_loader import load_settings

    cli = SimpleNamespace(
        config=config_path,
        structure=overrides.get("structure"),
        mode=overrides.get("mode"),
        cutoff=overrides.get("cutoff"),
        hist_rmin=overrides.get("hist_rmin"),
        hist_rmax=overrides.get("hist_rmax"),
        surface_threshold=overrides.get("surface_threshold"),
        edge_tol=overrides.get("edge_tol"),
        planarity=overrides.get("planarity"),
        tag=overrides.get("tag"),
    )
    cfg, _ = load_settings(cli)
    return asdict(cfg)


def _extract_structure_metadata(atoms, structure_path: Path) -> dict[str, Any]:
    info = dict(getattr(atoms, "info", {}) or {})
    crystal_structure = info.get("npf_crystal_structure") or info.get("crystal_structure")
    crystal_variant = info.get("npf_crystal_variant") or info.get("crystal_variant")
    shape = info.get("npf_shape") or info.get("shape")
    tag = info.get("npf_tag") or info.get("tag")
    return {
        "shape": shape,
        "tag": tag,
        "crystal_structure": crystal_structure,
        "crystal_variant": crystal_variant,
        "source_file": str(structure_path),
    }


def _parse_cutoff_value(cutoff_info: str) -> float | None:
    import re

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", cutoff_info or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _site_atom_indices(sites: dict[str, Any]) -> dict[str, list[int]]:
    def flatten(items):
        out: set[int] = set()
        for item in items or []:
            if isinstance(item, (list, tuple)):
                for idx in item:
                    out.add(int(idx))
            else:
                out.add(int(item))
        return sorted(out)

    return {
        "on-top (B1) sites": flatten(sites.get("atop", [])),
        "bridge (B2) sites": flatten(sites.get("bridge", [])),
        "3-fold (B3) sites": flatten(sites.get("hollow3", [])),
        "4-fold (B4) square sites": flatten(sites.get("hollow4_square", [])),
        "4-fold (B4) rectangular sites": flatten(sites.get("hollow4_rect", [])),
        "B5 sites": flatten(sites.get("B5", [])),
        **{name: flatten(sites.get(f"{kind}_interior_atoms", []))
           for name, kind in _TERRACE_MARKS.items()},
    }


def _read_structure(path: Path):
    if path.suffix.lower() == ".data":
        return read(str(path), format="lammps-data")
    return read(str(path))


class HistogramPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5.6, 4.8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.distance_ax = self.figure.add_subplot(211)
        self.cn_ax = self.figure.add_subplot(212)
        self.figure.subplots_adjust(hspace=0.42, left=0.10, right=0.98, top=0.96, bottom=0.10)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self._draw_empty()

    def _style_axes(self, ax, title: str, xlabel: str, ylabel: str = "Count"):
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.7)
        ax.tick_params(labelsize=8)

    def _draw_empty(self):
        self.figure.clear()
        self.distance_ax = self.figure.add_subplot(211)
        self.cn_ax = self.figure.add_subplot(212)
        self.distance_ax.text(0.5, 0.5, "Distance histogram appears here", ha="center", va="center", transform=self.distance_ax.transAxes, color="#8b97aa")
        self.cn_ax.text(0.5, 0.5, "CN histogram appears here", ha="center", va="center", transform=self.cn_ax.transAxes, color="#8b97aa")
        self.distance_ax.set_axis_off()
        self.cn_ax.set_axis_off()
        self.canvas.draw_idle()

    def set_histograms(
        self,
        *,
        distance_values: np.ndarray | list[float] | None,
        cn_values: np.ndarray | list[int] | None,
        hist_rmin: float,
        hist_rmax: float,
        title: str = "",
    ):
        self.figure.clear()
        self.distance_ax = self.figure.add_subplot(211)
        self.cn_ax = self.figure.add_subplot(212)

        dist = np.asarray(distance_values if distance_values is not None else [], dtype=float)
        cn = np.asarray(cn_values if cn_values is not None else [], dtype=float)

        if dist.size:
            bins = np.linspace(float(hist_rmin), float(hist_rmax), 40)
            self.distance_ax.hist(dist, bins=bins, color="#72b7ff", edgecolor="#bfdcff", linewidth=0.6)
        else:
            self.distance_ax.text(0.5, 0.5, "No pair distances in range", ha="center", va="center", transform=self.distance_ax.transAxes, color="#8b97aa")
        self._style_axes(self.distance_ax, f"{title} Atom-Atom Distance", "Distance (Å)")

        if cn.size:
            cn_min = int(np.min(cn))
            cn_max = int(np.max(cn))
            bins = np.arange(cn_min, cn_max + 2) - 0.5
            self.cn_ax.hist(cn, bins=bins, color="#6fe3c0", edgecolor="#c6f5e7", linewidth=0.6)
            self.cn_ax.set_xticks(np.arange(cn_min, cn_max + 1))
        else:
            self.cn_ax.text(0.5, 0.5, "No CN values available", ha="center", va="center", transform=self.cn_ax.transAxes, color="#8b97aa")
        self._style_axes(self.cn_ax, f"{title} Coordination Number", "CN")

        self.canvas.draw_idle()


class HistogramDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Histograms")
        self.resize(860, 780)
        self.panel = HistogramPanel(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self.panel)

    def set_histograms(self, **kwargs):
        self.panel.set_histograms(**kwargs)


class WorkingFolderDialog(QDialog):
    def __init__(self, start_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Working Folder")
        self.resize(820, 560)
        self._selected_folder: str = ""
        self._current_dir = Path(start_dir).expanduser().resolve()

        self._model = QFileSystemModel(self)
        self._model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Drives | QDir.Hidden)
        self._model.setRootPath(str(self._current_dir))

        self._view = QTreeView(self)
        self._view.setModel(self._model)
        self._view.setRootIndex(self._model.index(str(self._current_dir)))
        self._view.setSelectionMode(QAbstractItemView.SingleSelection)
        self._view.setSortingEnabled(True)
        self._view.sortByColumn(0, Qt.AscendingOrder)
        for col in range(1, 4):
            self._view.hideColumn(col)
        self._view.doubleClicked.connect(self._on_double_clicked)

        self._path_label = QLabel(str(self._current_dir), self)
        self._path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._path_label.setStyleSheet("color: #9aa6bb;")

        up_button = QPushButton("Up")
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        up_button.clicked.connect(self._go_up)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Double-click a folder to choose it."))
        button_row = QHBoxLayout()
        button_row.addWidget(up_button)
        button_row.addStretch(1)
        button_row.addWidget(buttons)
        layout.addLayout(button_row)
        layout.addWidget(self._path_label)
        layout.addWidget(self._view, 1)

    def _on_double_clicked(self, index):
        path = Path(self._model.filePath(index))
        if path.is_dir():
            self._selected_folder = str(path)
            self.accept()

    def _go_up(self):
        parent = self._current_dir.parent
        if parent == self._current_dir:
            return
        self._current_dir = parent
        self._path_label.setText(str(self._current_dir))
        self._view.setRootIndex(self._model.index(str(self._current_dir)))

    def selected_folder(self) -> str:
        if self._selected_folder:
            return self._selected_folder
        indexes = self._view.selectedIndexes()
        if indexes:
            path = Path(self._model.filePath(indexes[0]))
            if path.is_dir():
                return str(path)
        return ""


class SiteCounterWorker(QThread):
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, structure_path: str, config_path: str, overrides: dict[str, Any], surface_only: bool = False):
        super().__init__()
        self.structure_path = structure_path
        self.config_path = config_path
        self.overrides = overrides
        self.surface_only = surface_only

    def run(self) -> None:
        from ....count.main import write_single_csv
        from ....count.cn_count import compute_cn, cn_fixed
        from ....count.site_count import detect_sites_by_cn
        from ....count.main import count_surface_atoms, _default_out_path

        structure_path = Path(self.structure_path).expanduser().resolve()
        if not structure_path.exists():
            self.failed.emit(f"Structure file not found: {structure_path}")
            return

        try:
            cfg = _settings_from_config(self.config_path, self.overrides)
            atoms = _read_structure(structure_path)
            atoms.pbc = False
            hist_rmin = float(cfg["hist_rmin"])
            hist_rmax = float(cfg["hist_rmax"])
            graph_cutoff = max(
                4.0,
                hist_rmax,
                float(cfg["cutoff"] or 0.0) if str(cfg["mode"]) == "fixed" else 0.0,
            )
            neighbor_graph = build_neighbor_graph(atoms, graph_cutoff)
            average_nn_distance = average_nearest_distance(neighbor_graph)
            pair_distances = distances_within(atoms, hist_rmax, graph=neighbor_graph)
            effective_cutoff = None
            if self.surface_only:
                result = count_surface_atoms(
                    atoms=atoms,
                    mode=str(cfg["mode"]),
                    cutoff=cfg["cutoff"],
                    hist_rmin=hist_rmin,
                    hist_rmax=hist_rmax,
                    surface_threshold=int(cfg["surface_threshold"]),
                    neighbor_graph=neighbor_graph,
                )
                if str(cfg["mode"]) == "auto":
                    effective_cutoff = _parse_cutoff_value(result["cutoff_info"])
                    min_cutoff = average_nn_distance * 1.15 if average_nn_distance > 0 else None
                    if effective_cutoff is not None and min_cutoff is not None and effective_cutoff < min_cutoff:
                        effective_cutoff = float(min_cutoff)
                        cn_arr = cn_fixed(atoms, effective_cutoff, neighbor_graph=neighbor_graph)
                        result["surface_atoms"] = int(np.sum(cn_arr < int(cfg["surface_threshold"])))
                        result["surface_indices"] = np.where(cn_arr < int(cfg["surface_threshold"]))[0].tolist()
                        hist, edges = np.histogram(cn_arr, bins=np.arange(int(np.min(cn_arr)), int(np.max(cn_arr)) + 2) - 0.5)
                        result["cn_hist"] = [
                            (int(c), int(h))
                            for c, h in zip(((edges[:-1] + edges[1:]) * 0.5).astype(int), hist)
                        ]
                        result["cutoff_info"] = f"auto cutoff adjusted = {effective_cutoff:.3f} Å"
                report = {
                    "structure": str(structure_path),
                    "config": self.config_path,
                    "output_csv": "",
                    "atoms": len(atoms),
                    "mode": str(cfg["mode"]),
                    "cutoff_info": result["cutoff_info"],
                    "cutoff_value": effective_cutoff if effective_cutoff is not None else _parse_cutoff_value(result["cutoff_info"]),
                    "hist_rmin": hist_rmin,
                    "hist_rmax": hist_rmax,
                    "surface_threshold": int(cfg["surface_threshold"]),
                    "surface_atoms": int(result["surface_atoms"]),
                    "surface_indices": result["surface_indices"],
                    "metadata": {
                        **_extract_structure_metadata(atoms, structure_path),
                        "average_nn_distance": average_nn_distance,
                    },
                    "counts": {"surface_atoms": int(result["surface_atoms"])},
                    "cn_hist": result["cn_hist"],
                    "cn_values": result["cn_arr"],
                    "distance_values": pair_distances,
                    "site_marks": {"surface atoms": result["surface_indices"]},
                    "extras": {},
                }
                self.finished_ok.emit(report)
                return
            cn_arr, cutoff_info = compute_cn(
                atoms,
                mode=str(cfg["mode"]),
                cutoff=cfg["cutoff"],
                hist_rmin=hist_rmin,
                hist_rmax=hist_rmax,
                neighbor_graph=neighbor_graph,
            )
            cutoff_value = _parse_cutoff_value(cutoff_info)
            if str(cfg["mode"]) == "auto":
                min_cutoff = average_nn_distance * 1.15 if average_nn_distance > 0 else None
                if cutoff_value is not None and min_cutoff is not None and cutoff_value < min_cutoff:
                    cutoff_value = float(min_cutoff)
                    cn_arr = cn_fixed(atoms, cutoff_value, neighbor_graph=neighbor_graph)
                    cutoff_info = f"auto cutoff adjusted = {cutoff_value:.3f} Å"
            sites, labels, codes, extras = detect_sites_by_cn(
                atoms,
                cn_arr,
                surface_threshold=int(cfg["surface_threshold"]),
                edge_tol=float(cfg["edge_tol"]),
                planarity=float(cfg["planarity"]),
                r_cap=4.0,
                cn_surface_thr=int(cfg["surface_threshold"]),
                same_cn_tolerance=1,
                min_shared_neighbors=1,
                neighbor_graph=neighbor_graph,
            )
            out_path = Path(_default_out_path(str(structure_path), prefer_csv=True))
            write_single_csv(
                str(out_path),
                input_file=str(structure_path),
                n_atoms=len(atoms),
                mode=str(cfg["mode"]),
                cutoff_info=cutoff_info,
                surface_threshold=int(cfg["surface_threshold"]),
                cn_arr=cn_arr,
                atoms=atoms,
                sites=sites,
            )
            hist, edges = np.histogram(cn_arr, bins=np.arange(int(np.min(cn_arr)), int(np.max(cn_arr)) + 2) - 0.5)
            report = {
                "structure": str(structure_path),
                "config": self.config_path,
                "output_csv": str(out_path),
                "atoms": len(atoms),
                "mode": str(cfg["mode"]),
                "cutoff_info": cutoff_info,
                "cutoff_value": cutoff_value,
                "hist_rmin": hist_rmin,
                "hist_rmax": hist_rmax,
                "surface_threshold": int(cfg["surface_threshold"]),
                "surface_atoms": int(np.sum(cn_arr < int(cfg["surface_threshold"]))),
                "surface_indices": np.where(cn_arr < int(cfg["surface_threshold"]))[0].tolist(),
                "metadata": {
                    **_extract_structure_metadata(atoms, structure_path),
                    "average_nn_distance": average_nn_distance,
                },
                "counts": {k: len(v) for k, v in sites.items()},
                "cn_hist": [(int(c), int(h)) for c, h in zip(((edges[:-1] + edges[1:]) * 0.5).astype(int), hist)],
                "cn_values": cn_arr,
                "distance_values": pair_distances,
                "site_marks": _site_atom_indices(sites),
                "extras": extras,
            }
            self.finished_ok.emit(report)
        except Exception as exc:
            self.failed.emit(str(exc))


class SiteCounterPanel(QWidget):
    def __init__(self, parent=None, default_config: str = "npf.count.in", default_structure: str = "POSCAR"):
        super().__init__(parent)
        self._worker: SiteCounterWorker | None = None
        self._last_report_surface_indices: list[int] = []
        self._last_report_site_marks: dict[str, list[int]] = {}
        self._mark_color_defaults: dict[str, str] = {
            "surface atoms": "#ffb347",
            "on-top (B1) sites": "#8ad8ff",
            "bridge (B2) sites": "#7ef0c6",
            "3-fold (B3) sites": "#b58cff",
            "4-fold (B4) square sites": "#ff8f8f",
            "4-fold (B4) rectangular sites": "#ffca73",
            "B5 sites": "#f2c94c",
        }
        self._mark_export_slugs: dict[str, str] = {
            "surface atoms": "surface",
            "on-top (B1) sites": "ontop",
            "bridge (B2) sites": "bridge",
            "3-fold (B3) sites": "threefold",
            "4-fold (B4) square sites": "fourfold_square",
            "4-fold (B4) rectangular sites": "fourfold_rect",
            "B5 sites": "b5",
        }
        for name, kind in _TERRACE_MARKS.items():
            self._mark_color_defaults[name] = "#69d1b0"
            self._mark_export_slugs[name] = f"{kind}_interior_atoms"
        self._mark_checkboxes: dict[str, QCheckBox] = {}
        self._mark_color_buttons: dict[str, QPushButton] = {}
        self._mark_export_buttons: dict[str, QPushButton] = {}
        self._mark_colors: dict[str, QColor] = {}
        self._build_ui(default_config=default_config, default_structure=default_structure)

    def _build_ui(self, default_config: str, default_structure: str):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        header = QLabel("Count CN and surface sites for an ASE-readable structure.")
        header.setStyleSheet("font-size: 16px; font-weight: 700;")
        header.setWordWrap(True)
        root_layout.addWidget(header)

        self.preview = ClusterPreviewView(self, empty_message="Open a geometry file to preview the site-counter structure.")
        self.preview.setMinimumHeight(420)
        self.total_atoms_value = QLabel("0")
        self.surface_atoms_value = QLabel("0")
        self.crystal_structure_value = QLabel("unknown")
        self.average_distance_value = QLabel("unknown")
        self.mark_surface_atoms = QCheckBox("surface atoms")
        self.b1_sites = QCheckBox("on-top (B1) sites")
        self.b2_sites = QCheckBox("bridge (B2) sites")
        self.b3_sites = QCheckBox("3-fold (B3) sites")
        self.b4_square_sites = QCheckBox("4-fold (B4) square sites")
        self.b4_rect_sites = QCheckBox("4-fold (B4) rectangular sites")
        self.b5_sites = QCheckBox("B5 sites")
        self._mark_checkboxes = {
            "surface atoms": self.mark_surface_atoms,
            "on-top (B1) sites": self.b1_sites,
            "bridge (B2) sites": self.b2_sites,
            "3-fold (B3) sites": self.b3_sites,
            "4-fold (B4) square sites": self.b4_square_sites,
            "4-fold (B4) rectangular sites": self.b4_rect_sites,
            "B5 sites": self.b5_sites,
        }
        for name in _TERRACE_MARKS:
            self._mark_checkboxes[name] = QCheckBox(name)
        self._mark_colors = {name: QColor(color) for name, color in self._mark_color_defaults.items()}
        self.hist_dialog = HistogramDialog(self)
        self.show_histograms = QPushButton("Show Histograms")
        self.show_histograms.setEnabled(False)
        self.atom_radius = QSlider(Qt.Horizontal)
        self.atom_radius.setRange(10, 500)
        self.atom_radius.setSingleStep(1)
        self.atom_radius.setPageStep(10)
        self.atom_radius.setTickPosition(QSlider.TicksBelow)
        self.atom_radius.setTickInterval(25)
        self.atom_radius.setValue(300)
        self.atom_radius.setMinimumWidth(260)
        self.atom_radius_min = QLabel("0.10x")
        self.atom_radius_max = QLabel("5.00x")
        self.atom_radius_min.setStyleSheet("color: #9aa6bb; font-size: 11px;")
        self.atom_radius_max.setStyleSheet("color: #9aa6bb; font-size: 11px;")
        self.atom_radius_value = QLabel("3.00x")
        self.atom_radius_value.setMinimumWidth(48)
        self.bond_radius = QSlider(Qt.Horizontal)
        self.bond_radius.setRange(50, 400)
        self.bond_radius.setSingleStep(1)
        self.bond_radius.setPageStep(10)
        self.bond_radius.setTickPosition(QSlider.TicksBelow)
        self.bond_radius.setTickInterval(25)
        self.bond_radius.setValue(125)
        self.bond_radius.setMinimumWidth(260)
        self.bond_radius_min = QLabel("0.50x")
        self.bond_radius_max = QLabel("4.00x")
        self.bond_radius_min.setStyleSheet("color: #9aa6bb; font-size: 11px;")
        self.bond_radius_max.setStyleSheet("color: #9aa6bb; font-size: 11px;")
        self.bond_radius_value = QLabel("1.25x")
        self.bond_radius_value.setMinimumWidth(48)
        self.depth_cue = QSlider(Qt.Horizontal)
        self.depth_cue.setRange(0, 300)
        self.depth_cue.setSingleStep(1)
        self.depth_cue.setPageStep(10)
        self.depth_cue.setTickPosition(QSlider.TicksBelow)
        self.depth_cue.setTickInterval(25)
        self.depth_cue.setValue(100)
        self.depth_cue.setMinimumWidth(260)
        self.depth_cue_min = QLabel("0.0x")
        self.depth_cue_max = QLabel("3.0x")
        self.depth_cue_min.setStyleSheet("color: #9aa6bb; font-size: 11px;")
        self.depth_cue_max.setStyleSheet("color: #9aa6bb; font-size: 11px;")
        self.depth_cue_value = QLabel("1.0x")
        self.depth_cue_value.setMinimumWidth(48)
        self.show_bonds = QCheckBox("bonds")
        self.show_bonds.setChecked(True)
        self.projection_mode = QComboBox()
        self.projection_mode.addItems(["orthographic", "perspective", "oblique"])
        self.reset_preview_button = QPushButton("Reset View")
        self.structure_path = QLineEdit(default_structure)
        self.config_path = QLineEdit(default_config)
        self.mode = QComboBox()
        self.mode.addItems(["auto", "fixed", "adaptive12"])
        self.cutoff = QDoubleSpinBox()
        self.cutoff.setRange(0.0, 100.0)
        self.cutoff.setDecimals(3)
        self.cutoff.setValue(3.90)
        self.hist_rmin = QDoubleSpinBox()
        self.hist_rmin.setRange(0.0, 100.0)
        self.hist_rmin.setDecimals(3)
        self.hist_rmin.setValue(1.8)
        self.hist_rmax = QDoubleSpinBox()
        self.hist_rmax.setRange(0.0, 100.0)
        self.hist_rmax.setDecimals(3)
        self.hist_rmax.setValue(4.2)
        self.surface_threshold = QSpinBox()
        self.surface_threshold.setRange(1, 999)
        self.surface_threshold.setValue(12)
        self.edge_tol = QDoubleSpinBox()
        self.edge_tol.setRange(0.0, 1.0)
        self.edge_tol.setDecimals(3)
        self.edge_tol.setSingleStep(0.01)
        self.edge_tol.setValue(0.10)
        self.planarity = QDoubleSpinBox()
        self.planarity.setRange(0.0, 1.0)
        self.planarity.setDecimals(3)
        self.planarity.setSingleStep(0.01)
        self.planarity.setValue(0.02)
        self.tag = QLineEdit("NP")
        self.working_folder = QLineEdit()
        self.working_folder.setPlaceholderText("Select a working folder")

        self.output_csv = QLineEdit()
        self.output_csv.setReadOnly(True)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText("Analysis output will appear here.")

        browse_config = QPushButton("Browse")
        select_working_folder = QPushButton("Select Working Folder")
        refresh_working_folder = QPushButton("Refresh")
        load_config = QPushButton("Load Config")
        run_button = QPushButton("Analyze")
        self.working_file_list = QListWidget(self)
        self.working_file_list.setMaximumHeight(165)
        self.working_file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.working_file_list.setToolTip("Files in the selected working folder")
        self.working_file_list.setContextMenuPolicy(Qt.CustomContextMenu)

        form = QGroupBox("Analysis Setup")
        form_layout = QGridLayout(form)
        form_layout.setColumnStretch(1, 1)
        folder_row = QWidget(self)
        folder_row_layout = QHBoxLayout(folder_row)
        folder_row_layout.setContentsMargins(0, 0, 0, 0)
        folder_row_layout.setSpacing(8)
        folder_row_layout.addWidget(select_working_folder)
        folder_row_layout.addWidget(refresh_working_folder)
        folder_row_layout.addStretch(1)
        form_layout.addWidget(folder_row, 0, 0, 1, 4)
        form_layout.addWidget(self.working_folder, 1, 0, 1, 4)
        form_layout.addWidget(self.working_file_list, 2, 0, 1, 4)

        form_layout.addWidget(QLabel("Config file"), 3, 0)
        form_layout.addWidget(self.config_path, 3, 1)
        form_layout.addWidget(load_config, 3, 2)
        form_layout.addWidget(browse_config, 3, 3)

        form_layout.addWidget(QLabel("Mode"), 4, 0)
        form_layout.addWidget(self.mode, 4, 1)
        form_layout.addWidget(QLabel("Cutoff"), 4, 2)
        form_layout.addWidget(self.cutoff, 4, 3)

        form_layout.addWidget(QLabel("Hist rmin"), 5, 0)
        form_layout.addWidget(self.hist_rmin, 5, 1)
        form_layout.addWidget(QLabel("Hist rmax"), 5, 2)
        form_layout.addWidget(self.hist_rmax, 5, 3)

        form_layout.addWidget(QLabel("Surface threshold"), 6, 0)
        form_layout.addWidget(self.surface_threshold, 6, 1)
        form_layout.addWidget(QLabel("Edge tol"), 6, 2)
        form_layout.addWidget(self.edge_tol, 6, 3)

        form_layout.addWidget(QLabel("Planarity"), 7, 0)
        form_layout.addWidget(self.planarity, 7, 1)
        form_layout.addWidget(QLabel("Tag"), 7, 2)
        form_layout.addWidget(self.tag, 7, 3)

        form_layout.addWidget(QLabel("Output CSV"), 8, 0)
        form_layout.addWidget(self.output_csv, 8, 1, 1, 3)

        action_row = QVBoxLayout()
        action_row.setSpacing(8)
        action_row.addWidget(run_button)
        action_row.addWidget(self.show_histograms)
        action_row.addStretch(1)

        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        left_layout.addWidget(form)
        left_layout.addLayout(action_row)
        left_layout.addWidget(self.report, 1)

        middle_panel = QWidget(self)
        middle_layout = QVBoxLayout(middle_panel)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(12)
        preview_label = QLabel("Geometry Preview")
        preview_label.setStyleSheet("font-size: 14px; font-weight: 600;")
        middle_layout.addWidget(preview_label)
        middle_layout.addWidget(self.preview, 1)

        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        controls_label = QLabel("Viewer Controls")
        controls_label.setStyleSheet("font-size: 14px; font-weight: 600;")
        right_layout.addWidget(controls_label)

        stats_box = QGroupBox("Counts")
        stats_layout = QGridLayout(stats_box)
        stats_layout.addWidget(QLabel("Total num. atoms"), 0, 0)
        stats_layout.addWidget(self.total_atoms_value, 0, 1)
        stats_layout.addWidget(QLabel("Number of surface atoms"), 1, 0)
        stats_layout.addWidget(self.surface_atoms_value, 1, 1)
        stats_layout.addWidget(QLabel("Crystal structure"), 2, 0)
        stats_layout.addWidget(self.crystal_structure_value, 2, 1)
        stats_layout.addWidget(QLabel("Avg. atom distance"), 3, 0)
        stats_layout.addWidget(self.average_distance_value, 3, 1)
        right_layout.addWidget(stats_box)

        slider_row = QWidget(self)
        slider_row_layout = QHBoxLayout(slider_row)
        slider_row_layout.setContentsMargins(0, 0, 0, 0)
        slider_row_layout.setSpacing(10)
        slider_row_layout.addWidget(QLabel("Atom radius"))
        slider_block = QWidget(self)
        slider_block_layout = QVBoxLayout(slider_block)
        slider_block_layout.setContentsMargins(0, 0, 0, 0)
        slider_block_layout.setSpacing(2)
        slider_block_layout.addWidget(self.atom_radius)
        slider_labels = QWidget(self)
        slider_labels_layout = QHBoxLayout(slider_labels)
        slider_labels_layout.setContentsMargins(2, 0, 2, 0)
        slider_labels_layout.setSpacing(0)
        slider_labels_layout.addWidget(self.atom_radius_min)
        slider_labels_layout.addStretch(1)
        slider_labels_layout.addWidget(self.atom_radius_max)
        slider_block_layout.addWidget(slider_labels)
        slider_row_layout.addWidget(slider_block, 1)
        slider_row_layout.addWidget(self.atom_radius_value)
        right_layout.addWidget(slider_row)

        bond_row = QWidget(self)
        bond_row_layout = QHBoxLayout(bond_row)
        bond_row_layout.setContentsMargins(0, 0, 0, 0)
        bond_row_layout.setSpacing(10)
        bond_row_layout.addWidget(QLabel("Bond radius"))
        bond_block = QWidget(self)
        bond_block_layout = QVBoxLayout(bond_block)
        bond_block_layout.setContentsMargins(0, 0, 0, 0)
        bond_block_layout.setSpacing(2)
        bond_block_layout.addWidget(self.bond_radius)
        bond_labels = QWidget(self)
        bond_labels_layout = QHBoxLayout(bond_labels)
        bond_labels_layout.setContentsMargins(2, 0, 2, 0)
        bond_labels_layout.setSpacing(0)
        bond_labels_layout.addWidget(self.bond_radius_min)
        bond_labels_layout.addStretch(1)
        bond_labels_layout.addWidget(self.bond_radius_max)
        bond_block_layout.addWidget(bond_labels)
        bond_row_layout.addWidget(bond_block, 1)
        bond_row_layout.addWidget(self.bond_radius_value)
        right_layout.addWidget(bond_row)

        depth_row = QWidget(self)
        depth_row_layout = QHBoxLayout(depth_row)
        depth_row_layout.setContentsMargins(0, 0, 0, 0)
        depth_row_layout.setSpacing(10)
        depth_row_layout.addWidget(QLabel("Depth cueing"))
        depth_block = QWidget(self)
        depth_block_layout = QVBoxLayout(depth_block)
        depth_block_layout.setContentsMargins(0, 0, 0, 0)
        depth_block_layout.setSpacing(2)
        depth_block_layout.addWidget(self.depth_cue)
        depth_labels = QWidget(self)
        depth_labels_layout = QHBoxLayout(depth_labels)
        depth_labels_layout.setContentsMargins(2, 0, 2, 0)
        depth_labels_layout.setSpacing(0)
        depth_labels_layout.addWidget(self.depth_cue_min)
        depth_labels_layout.addStretch(1)
        depth_labels_layout.addWidget(self.depth_cue_max)
        depth_block_layout.addWidget(depth_labels)
        depth_row_layout.addWidget(depth_block, 1)
        depth_row_layout.addWidget(self.depth_cue_value)
        right_layout.addWidget(depth_row)

        projection_row = QWidget(self)
        projection_row_layout = QHBoxLayout(projection_row)
        projection_row_layout.setContentsMargins(0, 0, 0, 0)
        projection_row_layout.setSpacing(10)
        projection_row_layout.addWidget(QLabel("Projection"))
        projection_row_layout.addWidget(self.projection_mode, 1)
        projection_row_layout.addWidget(self.reset_preview_button)
        right_layout.addWidget(projection_row)

        bonds_toggle_row = QWidget(self)
        bonds_toggle_layout = QHBoxLayout(bonds_toggle_row)
        bonds_toggle_layout.setContentsMargins(0, 0, 0, 0)
        bonds_toggle_layout.setSpacing(10)
        bonds_toggle_layout.addWidget(self.show_bonds)
        bonds_toggle_layout.addStretch(1)
        right_layout.addWidget(bonds_toggle_row)

        surface_box = QGroupBox("Surface Marking")
        surface_box_layout = QVBoxLayout(surface_box)
        surface_box_layout.setContentsMargins(8, 8, 8, 8)
        for name in [
            "surface atoms",
            "on-top (B1) sites",
            "bridge (B2) sites",
            "3-fold (B3) sites",
            "4-fold (B4) square sites",
            "4-fold (B4) rectangular sites",
            "B5 sites",
            *_TERRACE_MARKS,
        ]:
            surface_box_layout.addLayout(self._make_mark_row(name))
        right_layout.addWidget(surface_box)

        right_layout.addStretch(1)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(left_panel)
        splitter.addWidget(middle_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([480, 760, 320])
        root_layout.addWidget(splitter, 1)

        browse_config.clicked.connect(self._browse_config)
        select_working_folder.clicked.connect(self._browse_working_folder)
        refresh_working_folder.clicked.connect(self._refresh_working_folder)
        load_config.clicked.connect(self._load_config_into_fields)
        run_button.clicked.connect(self.analyze)
        self.working_folder.editingFinished.connect(self._working_folder_edited)
        self.working_file_list.itemDoubleClicked.connect(self._working_file_double_clicked)
        self.working_file_list.customContextMenuRequested.connect(self._working_file_context_menu)
        self.atom_radius.valueChanged.connect(self._sync_atom_radius)
        self.bond_radius.valueChanged.connect(self._sync_bond_radius)
        self.depth_cue.valueChanged.connect(self._sync_depth_cue)
        self.projection_mode.currentTextChanged.connect(self.preview.set_projection)
        self.show_bonds.toggled.connect(self.preview.set_show_bonds)
        self.reset_preview_button.clicked.connect(self._reset_preview_controls)
        for widget in self._mark_checkboxes.values():
            widget.toggled.connect(self._apply_surface_marking)
        self.show_histograms.clicked.connect(self._show_histograms)
        self._set_mark_controls_enabled(False)
        self._load_current_structure()
        self._sync_atom_radius(self.atom_radius.value())
        self._sync_bond_radius(self.bond_radius.value())
        self._sync_depth_cue(self.depth_cue.value())

    def _browse_structure(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open geometry file",
            str(Path(self.structure_path.text()).expanduser() if self.structure_path.text().strip() else Path.cwd()),
            "Geometry files (*.xyz *.cif *.vasp *.POSCAR *.CONTCAR *.traj *.db *.*)",
        )
        if path:
            self.set_structure_path(path)

    def _browse_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open config", str(Path(self.config_path.text()).expanduser()), "Config files (*.in *.cn *.*)")
        if path:
            self.config_path.setText(path)

    def _browse_working_folder(self):
        start_dir = self.working_folder.text().strip() or self._structure_folder()
        dialog = WorkingFolderDialog(start_dir, self)
        if dialog.exec():
            folder = dialog.selected_folder()
            if folder:
                self.set_working_folder(folder, auto_select_structure=True)

    def _load_config_into_fields(self):
        try:
            cfg = _settings_from_config(self.config_path.text().strip() or "npf.count.in", {})
        except Exception as exc:
            QMessageBox.critical(self, "Config load failed", str(exc))
            return
        self.mode.setCurrentText(str(cfg.get("mode", "auto")))
        if cfg.get("cutoff") is not None:
            self.cutoff.setValue(float(cfg["cutoff"]))
        self.hist_rmin.setValue(float(cfg.get("hist_rmin", 1.8)))
        self.hist_rmax.setValue(float(cfg.get("hist_rmax", 4.2)))
        self.surface_threshold.setValue(int(cfg.get("surface_threshold", 12)))
        self.edge_tol.setValue(float(cfg.get("edge_tol", 0.10)))
        self.planarity.setValue(float(cfg.get("planarity", 0.02)))
        self.tag.setText(str(cfg.get("tag", "NP")))

    def _use_last_generated(self):
        current = Path(self.structure_path.text().strip()).expanduser()
        candidates = [current, Path("POSCAR"), Path("CONTCAR"), Path("np_gen.xyz"), Path("lammps.relaxed.data"), Path("lammps.start.data")]
        for candidate in candidates:
            if candidate.exists():
                self.set_structure_path(str(candidate))
                return
        QMessageBox.information(self, "No generated structure", "No recent generated geometry was found in the current folder.")

    def _working_folder_edited(self):
        folder = self.working_folder.text().strip()
        if folder:
            self.set_working_folder(folder, auto_select_structure=True)

    def _refresh_working_folder(self):
        folder = self.working_folder.text().strip()
        if folder:
            self.set_working_folder(folder, auto_select_structure=False)

    def _structure_folder(self) -> str:
        path_text = self.structure_path.text().strip() or "POSCAR"
        return str(Path(path_text).expanduser().resolve().parent)

    def set_working_folder(self, folder: str, *, auto_select_structure: bool = True):
        path = Path(folder).expanduser()
        self.working_folder.setText(str(path))
        if not path.exists():
            self.working_file_list.clear()
            return
        self._refresh_working_folder_files(path)
        if auto_select_structure:
            self._maybe_load_structure_from_folder(path)

    def _refresh_working_folder_files(self, folder: Path):
        self.working_file_list.blockSignals(True)
        self.working_file_list.clear()
        try:
            entries = sorted([p for p in folder.iterdir() if p.is_file()], key=lambda p: (p.suffix.lower(), p.name.lower()))
        except Exception:
            entries = []
        for path in entries:
            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
            item.setData(Qt.UserRole, str(path))
            self.working_file_list.addItem(item)
        self.working_file_list.blockSignals(False)

    def _maybe_load_structure_from_folder(self, folder: Path):
        for name in ["lammps.relaxed.POSCAR", "relaxed.POSCAR", "POSCAR", "CONTCAR", "lammps.relaxed.xyz", "relaxed.xyz", "np_gen.xyz", "lammps.relaxed.data", "lammps.start.data"]:
            candidate = folder / name
            if candidate.exists():
                self.set_structure_path(str(candidate))
                return

    def _selected_working_file_path(self, item: QListWidgetItem | None = None) -> Path | None:
        if item is None:
            item = self.working_file_list.currentItem()
        if item is None:
            return
        path = Path(str(item.data(Qt.UserRole) or item.text())).expanduser()
        if not path.exists():
            return None
        return path

    def _is_geometry_file(self, path: Path) -> bool:
        suffix = path.suffix.lower()
        name = path.name.lower()
        return suffix in {".xyz", ".vasp", ".poscar", ".contcar", ".data"} or name in {"poscar", "contcar"}

    def _open_in_editor(self, path: Path):
        if not path.exists():
            QMessageBox.information(self, "File not found", f"File not found:\n{path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _save_geometry_as_xyz(self, path: Path):
        if not self._is_geometry_file(path):
            QMessageBox.information(self, "Not a geometry file", f"This file is not a geometry file:\n{path.name}")
            return
        try:
            atoms = _read_structure(path)
            out_path = _xyz_output_path(path)
            _write_xyz_file(out_path, atoms)
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        QMessageBox.information(self, "Saved", f"Saved geometry as XYZ:\n{out_path}")

    def _load_action_label(self, path: Path) -> str:
        if self._is_geometry_file(path):
            return "Load geometry file"
        return "Load file"

    def _load_working_file(self, path: Path):
        if not self._is_geometry_file(path):
            QMessageBox.information(self, "Not a geometry file", f"This file is not a geometry file:\n{path.name}")
            return
        self.set_structure_path(str(path))

    def _working_file_double_clicked(self, item: QListWidgetItem):
        path = self._selected_working_file_path(item)
        if path is None:
            return
        if self._is_geometry_file(path):
            self._load_working_file(path)
        else:
            self._open_in_editor(path)

    def _working_file_context_menu(self, pos):
        item = self.working_file_list.itemAt(pos) or self.working_file_list.currentItem()
        path = self._selected_working_file_path(item)
        if path is None:
            return
        menu = QMenu(self)
        load_action = menu.addAction(self._load_action_label(path))
        save_action = menu.addAction("Save file in XYZ format") if self._is_geometry_file(path) else None
        editor_action = menu.addAction("Open in Editor")
        chosen = menu.exec(self.working_file_list.viewport().mapToGlobal(pos))
        if chosen == load_action:
            self._load_working_file(path)
        elif save_action is not None and chosen == save_action:
            self._save_geometry_as_xyz(path)
        elif chosen == editor_action:
            self._open_in_editor(path)

    def _load_current_structure(self):
        path_text = self.structure_path.text().strip()
        if not path_text:
            self.total_atoms_value.setText("0")
            self.crystal_structure_value.setText("unknown")
            self.average_distance_value.setText("unknown")
            self._last_report_surface_indices = []
            self._last_report_site_marks = {}
            self.preview.set_marked_atoms({})
            self.preview.set_mark_colors(self._current_mark_colors())
            self._set_mark_controls_enabled(False)
            return
        path = Path(path_text).expanduser()
        if not path.exists():
            self.preview.clear_preview()
            self.total_atoms_value.setText("0")
            self.crystal_structure_value.setText("unknown")
            self.average_distance_value.setText("unknown")
            self._last_report_surface_indices = []
            self._last_report_site_marks = {}
            self.preview.set_marked_atoms({})
            self.preview.set_mark_colors(self._current_mark_colors())
            self._set_mark_controls_enabled(False)
            return
        try:
            atoms = _read_structure(path)
            self.total_atoms_value.setText(str(len(atoms)))
            meta = _extract_structure_metadata(atoms, path)
            self.crystal_structure_value.setText(str(meta.get("crystal_structure") or "unknown"))
            avg_nn = _average_nn_distance(atoms)
            self.average_distance_value.setText(f"{float(avg_nn):.3f} Å" if avg_nn not in {None, ""} else "unknown")
            self._last_report_surface_indices = []
            self._last_report_site_marks = {}
            self.preview.set_marked_atoms({})
            self.preview.set_mark_colors(self._current_mark_colors())
            self._set_mark_controls_enabled(False)
            self.preview.load_structure_file(path)
        except Exception as exc:
            self.preview.clear_preview()
            self.report.setPlainText(f"Could not load geometry preview:\n{exc}")
            self.total_atoms_value.setText("0")
            self.crystal_structure_value.setText("unknown")
            self.average_distance_value.setText("unknown")
            self._last_report_surface_indices = []
            self._last_report_site_marks = {}
            self.preview.set_marked_atoms({})
            self.preview.set_mark_colors(self._current_mark_colors())
            self._set_mark_controls_enabled(False)

    def _sync_atom_radius(self, value: int):
        radius_scale = max(0.1, min(5.0, value / 100.0))
        self.atom_radius_value.setText(f"{radius_scale:.2f}x")
        QToolTip.showText(QCursor.pos(), f"Atom radius {radius_scale:.2f}x", self.atom_radius)
        self.preview.set_atom_scale(radius_scale)

    def _sync_bond_radius(self, value: int):
        radius_scale = max(0.5, min(4.0, value / 100.0))
        self.bond_radius_value.setText(f"{radius_scale:.2f}x")
        QToolTip.showText(QCursor.pos(), f"Bond radius {radius_scale:.2f}x", self.bond_radius)
        self.preview.set_bond_radius(radius_scale)

    def _sync_depth_cue(self, value: int):
        cue_scale = max(0.0, min(3.0, value / 100.0))
        self.depth_cue_value.setText(f"{cue_scale:.1f}x")
        QToolTip.showText(QCursor.pos(), f"Depth cueing {cue_scale:.1f}x", self.depth_cue)
        self.preview.set_depth_cue(cue_scale)

    def _reset_preview_controls(self):
        self.atom_radius.blockSignals(True)
        self.bond_radius.blockSignals(True)
        self.depth_cue.blockSignals(True)
        self.projection_mode.blockSignals(True)
        self.atom_radius.setValue(300)
        self.bond_radius.setValue(125)
        self.depth_cue.setValue(100)
        self.projection_mode.setCurrentText("orthographic")
        self.atom_radius.blockSignals(False)
        self.bond_radius.blockSignals(False)
        self.depth_cue.blockSignals(False)
        self.projection_mode.blockSignals(False)
        self._sync_atom_radius(self.atom_radius.value())
        self._sync_bond_radius(self.bond_radius.value())
        self._sync_depth_cue(self.depth_cue.value())
        self.preview.reset_view()

    def _apply_surface_marking(self, checked: bool):
        self.preview.set_marked_atoms(self._current_marked_atoms())
        self.preview.set_mark_colors(self._current_mark_colors())

    def _make_mark_row(self, name: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        checkbox = self._mark_checkboxes[name]
        swatch = QPushButton()
        swatch.setFixedSize(22, 22)
        swatch.setCursor(Qt.PointingHandCursor)
        swatch.setToolTip(f"Choose color for {name}")
        swatch.clicked.connect(lambda _=False, mark=name: self._pick_mark_color(mark))
        self._mark_color_buttons[name] = swatch
        export_button = QPushButton("Write marked xyz")
        export_button.setEnabled(False)
        export_button.setToolTip(f"Export {name} as an XYZ file with marked atoms labeled Cu.")
        export_button.setCursor(Qt.PointingHandCursor)
        export_button.clicked.connect(lambda _=False, mark=name: self._export_marked_xyz(mark))
        self._mark_export_buttons[name] = export_button
        self._apply_swatch_style(name)
        row.addWidget(checkbox, 1)
        row.addWidget(swatch, 0)
        row.addWidget(export_button, 0)
        return row

    def _export_output_path(self, source: Path, name: str) -> Path:
        slug = self._mark_export_slugs.get(name, name.lower().replace(" ", "_"))
        return _xyz_output_path(source, f"_{slug}_marked")

    def _mark_indices_for_export(self, name: str) -> list[int]:
        if name == "surface atoms":
            return list(self._last_report_surface_indices)
        return list(self._last_report_site_marks.get(name, []))

    def _export_marked_xyz(self, name: str):
        structure_text = self.structure_path.text().strip()
        if not structure_text:
            QMessageBox.information(self, "Missing structure", "Select a geometry file first.")
            return
        source = Path(structure_text).expanduser()
        if not source.exists():
            QMessageBox.information(self, "Missing structure", f"Structure file not found:\n{source}")
            return
        indices = self._mark_indices_for_export(name)
        if not indices:
            QMessageBox.information(self, "No marked atoms", f"No atoms are available for {name}. Run the analysis first.")
            return
        try:
            atoms = _read_structure(source)
            marked = atoms.copy()
            for idx in sorted(set(int(i) for i in indices)):
                marked[idx].symbol = "Cu"
            out_path = self._export_output_path(source, name)
            _write_xyz_file(out_path, marked)
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        QMessageBox.information(self, "Exported", f"Saved marked XYZ file:\n{out_path}")

    def _pick_mark_color(self, name: str):
        current = self._mark_colors.get(name, QColor(self._mark_color_defaults.get(name, "#ffffff")))
        color = QColorDialog.getColor(current, self, f"Select color for {name}")
        if not color.isValid():
            return
        self._mark_colors[name] = color
        self._apply_swatch_style(name)
        self._update_surface_marking()

    def _apply_swatch_style(self, name: str):
        button = self._mark_color_buttons.get(name)
        if button is None:
            return
        color = self._mark_colors.get(name, QColor(self._mark_color_defaults.get(name, "#ffffff")))
        border = "#ffffff" if color.lightness() < 150 else "#1e2430"
        button.setStyleSheet(
            f"QPushButton {{ background-color: {color.name()}; border: 1px solid {border}; border-radius: 4px; }}"
            f"QPushButton:hover {{ border: 1px solid #e6edf7; }}"
        )

    def _current_mark_colors(self) -> dict[str, QColor]:
        return {name: QColor(color) for name, color in self._mark_colors.items()}

    def _current_marked_atoms(self) -> dict[str, list[int]]:
        marks: dict[str, list[int]] = {}
        if self._mark_checkboxes["surface atoms"].isChecked():
            marks["surface atoms"] = list(self._last_report_surface_indices)
        for name, widget in self._mark_checkboxes.items():
            if name == "surface atoms":
                continue
            if widget.isChecked():
                marks[name] = list(self._last_report_site_marks.get(name, []))
        return marks

    def _set_mark_controls_enabled(self, enabled: bool):
        for widget in self._mark_checkboxes.values():
            widget.setEnabled(enabled)
        for widget in self._mark_color_buttons.values():
            widget.setEnabled(enabled)
        for widget in self._mark_export_buttons.values():
            widget.setEnabled(False)

    def _update_mark_export_buttons(self):
        for name, widget in self._mark_export_buttons.items():
            widget.setEnabled(bool(self._mark_indices_for_export(name)))

    def _update_surface_marking(self):
        self.preview.set_marked_atoms(self._current_marked_atoms())
        self.preview.set_mark_colors(self._current_mark_colors())

    def _show_histograms(self):
        self.hist_dialog.show()
        self.hist_dialog.raise_()
        self.hist_dialog.activateWindow()

    def _collect_overrides(self) -> dict[str, Any]:
        mode = self.mode.currentText()
        return {
            "mode": mode,
            "cutoff": float(self.cutoff.value()) if mode == "fixed" else None,
            "hist_rmin": float(self.hist_rmin.value()),
            "hist_rmax": float(self.hist_rmax.value()),
            "surface_threshold": int(self.surface_threshold.value()),
            "edge_tol": float(self.edge_tol.value()),
            "planarity": float(self.planarity.value()),
            "tag": self.tag.text().strip() or "NP",
        }

    def analyze(self):
        structure_path = self.structure_path.text().strip()
        if not structure_path:
            QMessageBox.warning(self, "Missing structure", "Select a structure file to analyze.")
            return
        if not Path(structure_path).expanduser().exists():
            QMessageBox.warning(self, "Missing structure", f"Structure file not found: {structure_path}")
            return
        if self.mode.currentText() == "fixed" and self.cutoff.value() <= 0:
            QMessageBox.warning(self, "Missing cutoff", "Fixed mode requires a positive cutoff.")
            return

        self.report.setPlainText("Analyzing...\n")
        self.output_csv.clear()
        self.show_histograms.setEnabled(False)
        self._set_mark_controls_enabled(False)
        self.setEnabled(False)
        self._worker = SiteCounterWorker(
            structure_path=structure_path,
            config_path=self.config_path.text().strip() or "npf.count.in",
            overrides=self._collect_overrides(),
        )
        self._worker.finished_ok.connect(self._analysis_finished)
        self._worker.failed.connect(self._analysis_failed)
        self._worker.finished.connect(self._analysis_cleanup)
        self._worker.start()

    def count_surface_atoms(self):
        structure_path = self.structure_path.text().strip()
        if not structure_path:
            QMessageBox.warning(self, "Missing structure", "Select a structure file to count.")
            return
        if not Path(structure_path).expanduser().exists():
            QMessageBox.warning(self, "Missing structure", f"Structure file not found: {structure_path}")
            return

        self.report.setPlainText("Counting surface atoms...\n")
        self.output_csv.clear()
        self.setEnabled(False)
        self._worker = SiteCounterWorker(
            structure_path=structure_path,
            config_path=self.config_path.text().strip() or "npf.count.in",
            overrides=self._collect_overrides(),
            surface_only=True,
        )
        self._worker.finished_ok.connect(self._analysis_finished)
        self._worker.failed.connect(self._analysis_failed)
        self._worker.finished.connect(self._analysis_cleanup)
        self._worker.start()

    def _analysis_finished(self, report: dict):
        counts = report["counts"]
        metadata = report.get("metadata", {})
        self._last_report_surface_indices = report.get("surface_indices", [])
        self._last_report_site_marks = report.get("site_marks", {})
        self.total_atoms_value.setText(str(report.get("atoms", 0)))
        self.surface_atoms_value.setText(str(report.get("surface_atoms", counts.get("surface_atoms", 0))))
        self.crystal_structure_value.setText(str(metadata.get("crystal_structure") or "unknown"))
        avg_nn = metadata.get("average_nn_distance")
        self.average_distance_value.setText(f"{float(avg_nn):.3f} Å" if avg_nn not in {None, ""} else "unknown")
        if report.get("hist_rmin") is not None:
            self.hist_rmin.setValue(float(report["hist_rmin"]))
        if report.get("hist_rmax") is not None:
            self.hist_rmax.setValue(float(report["hist_rmax"]))
        if report.get("cutoff_value") is not None:
            self.cutoff.setValue(float(report["cutoff_value"]))
        self.hist_dialog.set_histograms(
            distance_values=report.get("distance_values"),
            cn_values=report.get("cn_values"),
            hist_rmin=float(report.get("hist_rmin", self.hist_rmin.value())),
            hist_rmax=float(report.get("hist_rmax", self.hist_rmax.value())),
            title="Analyze - ",
        )
        self.show_histograms.setEnabled(True)
        self._set_mark_controls_enabled(True)
        self._update_mark_export_buttons()
        self._update_surface_marking()
        lines = [
            f"Structure: {report['structure']}",
            f"Config: {report['config']}",
            f"Atoms: {report['atoms']}",
            f"Crystal structure: {metadata.get('crystal_structure') or 'unknown'}",
            f"Avg. atom distance: {float(avg_nn):.3f} Å" if avg_nn not in {None, ""} else "Avg. atom distance: unknown",
            f"Hist rmin: {report.get('hist_rmin', 'unknown')}",
            f"Hist rmax: {report.get('hist_rmax', 'unknown')}",
            f"Cutoff: {report.get('cutoff_value', report['cutoff_info'])}",
            f"Mode: {report['mode']} ({report['cutoff_info']})",
            f"Surface threshold: CN < {report['surface_threshold']}",
            f"Surface atoms: {report.get('surface_atoms', 0)}",
        ]
        if report.get("output_csv"):
            lines.insert(3, f"Output CSV: {report['output_csv']}")
        motif_keys = ("atop", "bridge", "hollow3", "hollow4_square", "hollow4_rect", "kink", "B5")
        if any(k in counts for k in motif_keys):
            lines.extend([
                "",
                "Site counts:",
                f"  atop: {counts.get('atop', 0)}",
                f"  bridge: {counts.get('bridge', 0)}",
                f"  hollow3: {counts.get('hollow3', 0)}",
                f"  hollow4_square: {counts.get('hollow4_square', 0)}",
                f"  hollow4_rect: {counts.get('hollow4_rect', 0)}",
                f"  kink: {counts.get('kink', 0)}",
                f"  B5: {counts.get('B5', 0)}",
            ])
            lines.extend(["", "Terrace interiors (excluding patch-boundary atoms):"])
            for kind in TERRACE_TYPES:
                lines.append(
                    f"  {kind}: {counts.get(kind + '_interior_atoms', 0)} atoms; "
                    f"{counts.get(kind + '_interior_sites', 0)} all-interior hollow sites"
                )
        self.output_csv.setText(report["output_csv"])
        self.report.setPlainText("\n".join(lines))

    def _analysis_failed(self, message: str):
        self.report.setPlainText(message)
        self.show_histograms.setEnabled(False)
        self._set_mark_controls_enabled(False)
        QMessageBox.critical(self, "Site counter failed", message)

    def _analysis_cleanup(self):
        self.setEnabled(True)
        self._worker = None

    def set_structure_path(self, path: str):
        self.structure_path.setText(path)
        self._load_current_structure()
