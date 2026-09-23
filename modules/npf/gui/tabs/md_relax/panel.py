from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read, write

from PySide6.QtCore import QDir, QThread, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QCursor, QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialogButtonBox,
    QComboBox,
    QFileDialog,
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
    QScrollArea,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QFileSystemModel,
    QTreeView,
    QAbstractItemView,
    QToolTip,
    QVBoxLayout,
    QWidget,
    QDialog,
    QListWidget,
    QListWidgetItem,
    QApplication,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ....common import _average_nn_distance, _write_xyz_file, _xyz_output_path
from ....relax.main import _parse_thermo_history, run_thermal_relaxation, write_thermal_relaxation_inputs
from ...preview import ClusterPreviewView, build_cluster_preview, build_cluster_preview_from_atoms


class ThermalRelaxationWorker(QThread):
    finished_ok = Signal(dict)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, structure_path: str, settings: dict[str, Any]):
        super().__init__()
        self.structure_path = structure_path
        self.settings = settings

    def run(self) -> None:
        try:
            self.status.emit("Preparing relaxation inputs...")
            result = run_thermal_relaxation(
                structure_path=self.structure_path,
                lammps_command=str(self.settings.get("lammps_command", "lmp")),
                potential_file=str(self.settings.get("potential_file", "")),
                pair_style=str(self.settings.get("pair_style", "eam/alloy")),
                pair_coeff=str(self.settings.get("pair_coeff", "")),
                temperature=float(self.settings.get("temperature", 1200.0)),
                thermal_steps=int(self.settings.get("thermal_steps", 5000)),
                quench_temperature=float(self.settings.get("quench_temperature", 50.0)),
                quench_steps=int(self.settings.get("quench_steps", 5000)),
                timestep=float(self.settings.get("timestep", 0.001)),
                damping=float(self.settings.get("damping", 0.1)),
                seed=int(self.settings.get("seed", 12345)),
                minimize_etol=float(self.settings.get("minimize_etol", 1.0e-8)),
                minimize_ftol=float(self.settings.get("minimize_ftol", 1.0e-10)),
                minimize_maxiter=int(self.settings.get("minimize_maxiter", 1000)),
                minimize_maxeval=int(self.settings.get("minimize_maxeval", 10000)),
                vacuum=float(self.settings.get("vacuum", 10.0)),
                remove_drift=bool(self.settings.get("remove_drift", True)),
                freeze_core=bool(self.settings.get("freeze_core", False)),
                surface_threshold=int(self.settings.get("surface_threshold", 12)),
                hist_rmin=float(self.settings.get("hist_rmin", 1.8)),
                hist_rmax=float(self.settings.get("hist_rmax", 4.2)),
                dump_interval=int(self.settings.get("dump_interval", 100)),
                workdir=self.settings.get("workdir"),
                status_callback=self.status.emit,
            )
            self.finished_ok.emit(result.to_dict())
        except Exception as exc:
            self.failed.emit(str(exc))


class ThermalRelaxationWriteWorker(QThread):
    finished_ok = Signal(dict)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, structure_path: str, settings: dict[str, Any]):
        super().__init__()
        self.structure_path = structure_path
        self.settings = settings

    def run(self) -> None:
        try:
            result = write_thermal_relaxation_inputs(
                structure_path=self.structure_path,
                lammps_command=str(self.settings.get("lammps_command", "lmp")),
                potential_file=str(self.settings.get("potential_file", "")),
                pair_style=str(self.settings.get("pair_style", "eam/alloy")),
                pair_coeff=str(self.settings.get("pair_coeff", "")),
                temperature=float(self.settings.get("temperature", 1200.0)),
                thermal_steps=int(self.settings.get("thermal_steps", 5000)),
                quench_temperature=float(self.settings.get("quench_temperature", 50.0)),
                quench_steps=int(self.settings.get("quench_steps", 5000)),
                timestep=float(self.settings.get("timestep", 0.001)),
                damping=float(self.settings.get("damping", 0.1)),
                seed=int(self.settings.get("seed", 12345)),
                minimize_etol=float(self.settings.get("minimize_etol", 1.0e-8)),
                minimize_ftol=float(self.settings.get("minimize_ftol", 1.0e-10)),
                minimize_maxiter=int(self.settings.get("minimize_maxiter", 1000)),
                minimize_maxeval=int(self.settings.get("minimize_maxeval", 10000)),
                vacuum=float(self.settings.get("vacuum", 10.0)),
                remove_drift=bool(self.settings.get("remove_drift", True)),
                freeze_core=bool(self.settings.get("freeze_core", False)),
                surface_threshold=int(self.settings.get("surface_threshold", 12)),
                hist_rmin=float(self.settings.get("hist_rmin", 1.8)),
                hist_rmax=float(self.settings.get("hist_rmax", 4.2)),
                dump_interval=int(self.settings.get("dump_interval", 100)),
                workdir=self.settings.get("workdir"),
                status_callback=self.status.emit,
            )
            self.finished_ok.emit(result.to_dict())
        except Exception as exc:
            self.failed.emit(str(exc))


class ThermoPlotPanel(QWidget):
    plot_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(7.2, 6.4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self._axis_keys: dict[int, str] = {}
        self._plot_specs: dict[str, dict[str, Any]] = {}
        self._click_cid = self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self._draw_empty()

    def _draw_empty(self):
        self.figure.clear()
        self._axis_keys.clear()
        self._plot_specs.clear()
        axes = [
            self.figure.add_subplot(231),
            self.figure.add_subplot(232),
            self.figure.add_subplot(233),
            self.figure.add_subplot(234),
            self.figure.add_subplot(235),
            self.figure.add_subplot(236),
        ]
        labels = [
            "Energy vs step appears here",
            "Temperature vs step appears here",
            "Temperature vs energy appears here",
            "PE vs step appears here",
            "Etotal vs temperature appears here",
            "Temperature vs time appears here",
        ]
        for ax, label in zip(axes, labels):
            ax.text(0.5, 0.5, label, ha="center", va="center", transform=ax.transAxes, color="#8b97aa")
            ax.set_axis_off()
        self.figure.subplots_adjust(hspace=0.55, wspace=0.35, left=0.08, right=0.98, top=0.96, bottom=0.08)
        self.canvas.draw_idle()

    def _build_plot_specs(self, thermo: dict[str, list[float]] | None, title: str, timestep: float) -> list[dict[str, Any]]:
        thermo = thermo or {}
        step = np.asarray(thermo.get("step", []), dtype=float)
        temp = np.asarray(thermo.get("temp", []), dtype=float)
        pe = np.asarray(thermo.get("pe", []), dtype=float)
        etotal = np.asarray(thermo.get("etotal", []), dtype=float)
        time_axis = step * float(timestep)

        return [
            {
                "key": "energy_vs_step",
                "title": f"{title}Energy vs step".strip(),
                "xlabel": "Step",
                "ylabel": "E total",
                "kind": "line",
                "x": step,
                "y": etotal,
                "color": "#72b7ff",
                "message": "No thermo data",
            },
            {
                "key": "temperature_vs_step",
                "title": f"{title}Temperature vs step".strip(),
                "xlabel": "Step",
                "ylabel": "Temp (K)",
                "kind": "line",
                "x": step,
                "y": temp,
                "color": "#ffbf66",
                "message": "No thermo data",
            },
            {
                "key": "temperature_vs_energy",
                "title": f"{title}Temperature vs energy".strip(),
                "xlabel": "Temp (K)",
                "ylabel": "PE",
                "kind": "scatter",
                "x": temp,
                "y": pe,
                "color": "#7ef0c6",
                "message": "No thermo data",
            },
            {
                "key": "pe_vs_step",
                "title": f"{title}PE vs step".strip(),
                "xlabel": "Step",
                "ylabel": "PE",
                "kind": "line",
                "x": step,
                "y": pe,
                "color": "#ff8f8f",
                "message": "No thermo data",
            },
            {
                "key": "etotal_vs_temperature",
                "title": f"{title}Etotal vs temperature".strip(),
                "xlabel": "Temp (K)",
                "ylabel": "E total",
                "kind": "scatter",
                "x": temp,
                "y": etotal,
                "color": "#d5b3ff",
                "message": "No thermo data",
            },
            {
                "key": "temperature_vs_time",
                "title": f"{title}Temperature vs time".strip(),
                "xlabel": "Time (ps)",
                "ylabel": "Temp (K)",
                "kind": "line",
                "x": time_axis,
                "y": temp,
                "color": "#8ad8ff",
                "message": "No thermo data",
            },
        ]

    def _render_plot(self, ax, spec: dict[str, Any]):
        x = np.asarray(spec.get("x", []), dtype=float)
        y = np.asarray(spec.get("y", []), dtype=float)
        if x.size:
            if spec.get("kind") == "scatter":
                ax.scatter(x, y, s=8, color=spec.get("color", "#72b7ff"), alpha=0.75)
            else:
                ax.plot(x, y, color=spec.get("color", "#72b7ff"), linewidth=1.4)
        else:
            ax.text(0.5, 0.5, spec.get("message", "No thermo data"), ha="center", va="center", transform=ax.transAxes, color="#8b97aa")
        ax.set_title(spec.get("title", ""), fontsize=10, pad=8)
        ax.set_xlabel(spec.get("xlabel", ""))
        ax.set_ylabel(spec.get("ylabel", ""))
        ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.7)

    def set_thermo_data(self, thermo: dict[str, list[float]] | None, title: str = "", timestep: float = 1.0):
        self.figure.clear()
        self._axis_keys.clear()
        specs = self._build_plot_specs(thermo, title, timestep)
        axes = [
            self.figure.add_subplot(231),
            self.figure.add_subplot(232),
            self.figure.add_subplot(233),
            self.figure.add_subplot(234),
            self.figure.add_subplot(235),
            self.figure.add_subplot(236),
        ]
        self._plot_specs = {spec["key"]: spec for spec in specs}
        for ax, spec in zip(axes, specs):
            self._axis_keys[id(ax)] = str(spec["key"])
            self._render_plot(ax, spec)
        self.figure.subplots_adjust(hspace=0.55, wspace=0.35, left=0.08, right=0.98, top=0.95, bottom=0.08)
        self.canvas.draw_idle()

    def _on_canvas_click(self, event):
        if event.button != 1 or event.inaxes is None:
            return
        key = self._axis_keys.get(id(event.inaxes))
        if key:
            self.plot_clicked.emit(key)


class ThermoPlotDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Thermal Relaxation Plots")
        self.resize(980, 840)
        self.panel = ThermoPlotPanel(self)
        self.panel.plot_clicked.connect(self._open_plot_popup)
        self._open_dialogs: list[QDialog] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self.panel)

    def set_thermo_data(self, **kwargs):
        self.panel.set_thermo_data(**kwargs)

    def _open_plot_popup(self, plot_key: str):
        spec = self.panel._plot_specs.get(plot_key)
        if not spec:
            return
        dialog = ThermoPlotDetailDialog(spec, self)
        self._open_dialogs.append(dialog)
        dialog.finished.connect(lambda _result, d=dialog: self._open_dialogs.remove(d) if d in self._open_dialogs else None)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()


class ThermoPlotDetailPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8.4, 6.2), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self._ax = None
        self._draw_empty()

    def _draw_empty(self):
        self.figure.clear()
        self._ax = self.figure.add_subplot(111)
        self._ax.text(0.5, 0.5, "No plot selected", ha="center", va="center", transform=self._ax.transAxes, color="#8b97aa")
        self._ax.set_axis_off()
        self.canvas.draw_idle()

    def set_plot(self, spec: dict[str, Any]):
        self.figure.clear()
        self._ax = self.figure.add_subplot(111)
        x = np.asarray(spec.get("x", []), dtype=float)
        y = np.asarray(spec.get("y", []), dtype=float)
        if x.size:
            if spec.get("kind") == "scatter":
                self._ax.scatter(x, y, s=18, color=spec.get("color", "#72b7ff"), alpha=0.85)
            else:
                self._ax.plot(x, y, color=spec.get("color", "#72b7ff"), linewidth=1.8)
        else:
            self._ax.text(0.5, 0.5, spec.get("message", "No thermo data"), ha="center", va="center", transform=self._ax.transAxes, color="#8b97aa")
        self._ax.set_title(spec.get("title", ""), fontsize=12, pad=12)
        self._ax.set_xlabel(spec.get("xlabel", ""))
        self._ax.set_ylabel(spec.get("ylabel", ""))
        self._ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.7)
        self.figure.tight_layout()
        self.canvas.draw_idle()


class ThermoPlotDetailDialog(QDialog):
    def __init__(self, spec: dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle(spec.get("title", "Thermal Plot"))
        self.resize(1100, 760)
        self.panel = ThermoPlotDetailPanel(self)
        self.panel.set_plot(spec)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self.panel)


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
        self._view.setHeaderHidden(False)
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


class ThermalRelaxationPanel(QWidget):
    def __init__(self, parent=None, default_structure: str = "POSCAR"):
        super().__init__(parent)
        self._worker: ThermalRelaxationWorker | None = None
        self._write_worker: ThermalRelaxationWriteWorker | None = None
        self._last_relaxation_result: dict[str, Any] | None = None
        self._last_write_result: dict[str, Any] | None = None
        self._working_folder_explicit = False
        self._working_folder_dialog_open = False
        self._build_ui(default_structure=default_structure)
        self._wire_signals()
        self._load_current_structure()
        self._sync_atom_radius(self.atom_radius.value())
        self._sync_bond_radius(self.bond_radius.value())
        self._sync_depth_cue(self.depth_cue.value())

    def _build_ui(self, default_structure: str):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        header = QLabel("Thermal relax a generated nanoparticle with LAMMPS, then quench and minimize.")
        header.setWordWrap(True)
        header.setStyleSheet("font-size: 16px; font-weight: 700;")
        root_layout.addWidget(header)

        self.preview = ClusterPreviewView(self, empty_message="Open a geometry file to preview the relaxation input.")
        self.preview.setMinimumHeight(420)

        self.structure_path = QLineEdit(default_structure)
        self.working_folder = QLineEdit()
        self.working_folder.setPlaceholderText("Select a working folder for LAMMPS results")
        self.output_prefix = QLineEdit("relaxed_np")
        self.lammps_command = QLineEdit("lmp")
        self.potential_file = QLineEdit()
        self.pair_style = QLineEdit("eam/alloy")
        self.pair_coeff = QLineEdit()
        self.pair_coeff.setPlaceholderText("* * {potential_file} {elements}")
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 10000.0)
        self.temperature.setDecimals(2)
        self.temperature.setValue(1200.0)
        self.thermal_steps = QSpinBox()
        self.thermal_steps.setRange(0, 1_000_000)
        self.thermal_steps.setValue(5000)
        self.quench_temperature = QDoubleSpinBox()
        self.quench_temperature.setRange(0.0, 10000.0)
        self.quench_temperature.setDecimals(2)
        self.quench_temperature.setValue(50.0)
        self.quench_steps = QSpinBox()
        self.quench_steps.setRange(0, 1_000_000)
        self.quench_steps.setValue(5000)
        self.timestep = QDoubleSpinBox()
        self.timestep.setRange(0.000001, 0.1)
        self.timestep.setDecimals(6)
        self.timestep.setSingleStep(0.0001)
        self.timestep.setValue(0.001)
        self.damping = QDoubleSpinBox()
        self.damping.setRange(0.001, 10.0)
        self.damping.setDecimals(3)
        self.damping.setSingleStep(0.01)
        self.damping.setValue(0.1)
        self.vacuum = QDoubleSpinBox()
        self.vacuum.setRange(0.0, 50.0)
        self.vacuum.setDecimals(2)
        self.vacuum.setValue(10.0)
        self.seed = QSpinBox()
        self.seed.setRange(1, 2_000_000_000)
        self.seed.setValue(12345)
        self.minimize_etol = QDoubleSpinBox()
        self.minimize_etol.setRange(1.0e-14, 1.0)
        self.minimize_etol.setDecimals(10)
        self.minimize_etol.setSingleStep(1.0e-8)
        self.minimize_etol.setValue(1.0e-8)
        self.minimize_ftol = QDoubleSpinBox()
        self.minimize_ftol.setRange(1.0e-14, 1.0)
        self.minimize_ftol.setDecimals(10)
        self.minimize_ftol.setSingleStep(1.0e-10)
        self.minimize_ftol.setValue(1.0e-10)
        self.minimize_maxiter = QSpinBox()
        self.minimize_maxiter.setRange(1, 1_000_000)
        self.minimize_maxiter.setValue(1000)
        self.minimize_maxeval = QSpinBox()
        self.minimize_maxeval.setRange(1, 10_000_000)
        self.minimize_maxeval.setValue(10000)
        self.remove_drift = QCheckBox("Remove center-of-mass drift")
        self.remove_drift.setChecked(True)
        self.freeze_core = QCheckBox("Relax surface atoms only")
        self.freeze_core.setChecked(False)
        self.surface_threshold = QSpinBox()
        self.surface_threshold.setRange(1, 999)
        self.surface_threshold.setValue(12)
        self.dump_interval = QSpinBox()
        self.dump_interval.setRange(1, 1000000)
        self.dump_interval.setValue(100)
        self.output_xyz = QLineEdit()
        self.output_xyz.setReadOnly(True)
        self.output_poscar = QLineEdit()
        self.output_poscar.setReadOnly(True)
        self.output_data = QLineEdit()
        self.output_data.setReadOnly(True)
        self.trajectory_file = QLineEdit()
        self.trajectory_file.setReadOnly(True)
        self.relaxation_folder = QLineEdit()
        self.relaxation_folder.setReadOnly(True)
        self.initial_atoms_value = QLabel("0")
        self.final_atoms_value = QLabel("0")
        self.initial_nn_value = QLabel("unknown")
        self.final_nn_value = QLabel("unknown")
        self.initial_pe_value = QLabel("unknown")
        self.final_pe_value = QLabel("unknown")

        browse_structure = QPushButton("Browse")
        use_poscar = QPushButton("Use Last Generated")
        browse_working_folder = QPushButton("Select Working Folder")
        refresh_working_folder = QPushButton("Refresh")
        browse_potential = QPushButton("Browse")
        run_button = QPushButton("Run LAMMPS Locally")
        write_button = QPushButton("Write LAMMPS Files")
        self.working_file_list = QListWidget(self)
        self.working_file_list.setMaximumHeight(165)
        self.working_file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.working_file_list.setToolTip("Files in the selected working folder")

        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        setup_box = QGroupBox("Relaxation Setup")
        setup_layout = QGridLayout(setup_box)
        setup_layout.setColumnStretch(1, 1)
        folder_row = QWidget(self)
        folder_row_layout = QHBoxLayout(folder_row)
        folder_row_layout.setContentsMargins(0, 0, 0, 0)
        folder_row_layout.setSpacing(8)
        folder_row_layout.addWidget(browse_working_folder)
        folder_row_layout.addWidget(refresh_working_folder)
        folder_row_layout.addStretch(1)
        setup_layout.addWidget(folder_row, 0, 0, 1, 4)
        setup_layout.addWidget(self.working_folder, 1, 0, 1, 4)
        setup_layout.addWidget(self.working_file_list, 2, 0, 1, 4)

        setup_layout.addWidget(QLabel("LAMMPS command"), 3, 0)
        setup_layout.addWidget(self.lammps_command, 3, 1, 1, 3)
        setup_layout.addWidget(QLabel("Potential file"), 4, 0)
        setup_layout.addWidget(self.potential_file, 4, 1)
        setup_layout.addWidget(browse_potential, 4, 2)
        setup_layout.addWidget(QLabel("Pair style"), 5, 0)
        setup_layout.addWidget(self.pair_style, 5, 1, 1, 3)
        setup_layout.addWidget(QLabel("Pair coeff"), 6, 0)
        setup_layout.addWidget(self.pair_coeff, 6, 1, 1, 3)
        setup_layout.addWidget(QLabel("Output prefix"), 7, 0)
        setup_layout.addWidget(self.output_prefix, 7, 1, 1, 3)
        setup_layout.addWidget(self.freeze_core, 8, 0, 1, 4)

        run_row = QWidget(self)
        run_row_layout = QHBoxLayout(run_row)
        run_row_layout.setContentsMargins(0, 0, 0, 0)
        run_row_layout.setSpacing(8)
        run_row_layout.addWidget(run_button)
        run_row_layout.addWidget(write_button)
        run_row_layout.addStretch(1)

        params_box = QGroupBox("MD Parameters")
        params_layout = QGridLayout(params_box)
        params_layout.addWidget(QLabel("Temperature (K)"), 0, 0)
        params_layout.addWidget(self.temperature, 0, 1)
        params_layout.addWidget(QLabel("Thermal steps"), 0, 2)
        params_layout.addWidget(self.thermal_steps, 0, 3)
        params_layout.addWidget(QLabel("Quench temp (K)"), 1, 0)
        params_layout.addWidget(self.quench_temperature, 1, 1)
        params_layout.addWidget(QLabel("Quench steps"), 1, 2)
        params_layout.addWidget(self.quench_steps, 1, 3)
        params_layout.addWidget(QLabel("Time step (ps)"), 2, 0)
        params_layout.addWidget(self.timestep, 2, 1)
        params_layout.addWidget(QLabel("Langevin damp (ps)"), 2, 2)
        params_layout.addWidget(self.damping, 2, 3)
        params_layout.addWidget(QLabel("Vacuum (Å)"), 3, 0)
        params_layout.addWidget(self.vacuum, 3, 1)
        params_layout.addWidget(QLabel("Seed"), 3, 2)
        params_layout.addWidget(self.seed, 3, 3)
        params_layout.addWidget(QLabel("Minimize etol"), 4, 0)
        params_layout.addWidget(self.minimize_etol, 4, 1)
        params_layout.addWidget(QLabel("Minimize ftol"), 4, 2)
        params_layout.addWidget(self.minimize_ftol, 4, 3)
        params_layout.addWidget(QLabel("Minimize maxiter"), 5, 0)
        params_layout.addWidget(self.minimize_maxiter, 5, 1)
        params_layout.addWidget(QLabel("Minimize maxeval"), 5, 2)
        params_layout.addWidget(self.minimize_maxeval, 5, 3)
        params_layout.addWidget(QLabel("Surface threshold"), 6, 0)
        params_layout.addWidget(self.surface_threshold, 6, 1)
        params_layout.addWidget(QLabel("Dump interval"), 6, 2)
        params_layout.addWidget(self.dump_interval, 6, 3)
        params_layout.addWidget(self.remove_drift, 7, 0, 1, 4)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText("Thermal relaxation output will appear here.")

        left_layout.addWidget(setup_box)
        left_layout.addWidget(params_box)
        left_layout.addWidget(run_row)
        left_layout.addWidget(self.report, 1)

        middle_panel = QWidget(self)
        middle_layout = QVBoxLayout(middle_panel)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(12)
        preview_label = QLabel("Relaxation Preview")
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

        stats_box = QGroupBox("Results")
        stats_layout = QGridLayout(stats_box)
        stats_layout.addWidget(QLabel("Initial atoms"), 0, 0)
        stats_layout.addWidget(self.initial_atoms_value, 0, 1)
        stats_layout.addWidget(QLabel("Final atoms"), 1, 0)
        stats_layout.addWidget(self.final_atoms_value, 1, 1)
        stats_layout.addWidget(QLabel("Initial avg. NN"), 2, 0)
        stats_layout.addWidget(self.initial_nn_value, 2, 1)
        stats_layout.addWidget(QLabel("Final avg. NN"), 3, 0)
        stats_layout.addWidget(self.final_nn_value, 3, 1)
        stats_layout.addWidget(QLabel("Initial PE"), 4, 0)
        stats_layout.addWidget(self.initial_pe_value, 4, 1)
        stats_layout.addWidget(QLabel("Final PE"), 5, 0)
        stats_layout.addWidget(self.final_pe_value, 5, 1)
        stats_layout.addWidget(QLabel("Relaxed XYZ"), 6, 0)
        stats_layout.addWidget(self.output_xyz, 6, 1)
        stats_layout.addWidget(QLabel("Relaxed POSCAR"), 7, 0)
        stats_layout.addWidget(self.output_poscar, 7, 1)
        stats_layout.addWidget(QLabel("Relaxed DATA"), 8, 0)
        stats_layout.addWidget(self.output_data, 8, 1)
        stats_layout.addWidget(QLabel("Trajectory"), 9, 0)
        stats_layout.addWidget(self.trajectory_file, 9, 1)
        stats_layout.addWidget(QLabel("Results folder"), 10, 0)
        stats_layout.addWidget(self.relaxation_folder, 10, 1)
        right_layout.addWidget(stats_box)

        self.atom_radius = QSlider(Qt.Horizontal)
        self.atom_radius.setRange(10, 500)
        self.atom_radius.setSingleStep(1)
        self.atom_radius.setPageStep(10)
        self.atom_radius.setTickPosition(QSlider.TicksBelow)
        self.atom_radius.setTickInterval(25)
        self.atom_radius.setValue(300)
        self.atom_radius_min = QLabel("0.10x")
        self.atom_radius_max = QLabel("5.00x")
        self.atom_radius_value = QLabel("3.00x")
        self.atom_radius_min.setStyleSheet("color: #9aa6bb; font-size: 11px;")
        self.atom_radius_max.setStyleSheet("color: #9aa6bb; font-size: 11px;")

        self.bond_radius = QSlider(Qt.Horizontal)
        self.bond_radius.setRange(50, 400)
        self.bond_radius.setSingleStep(1)
        self.bond_radius.setPageStep(10)
        self.bond_radius.setTickPosition(QSlider.TicksBelow)
        self.bond_radius.setTickInterval(25)
        self.bond_radius.setValue(125)
        self.bond_radius_min = QLabel("0.50x")
        self.bond_radius_max = QLabel("4.00x")
        self.bond_radius_value = QLabel("1.25x")
        self.bond_radius_min.setStyleSheet("color: #9aa6bb; font-size: 11px;")
        self.bond_radius_max.setStyleSheet("color: #9aa6bb; font-size: 11px;")

        self.depth_cue = QSlider(Qt.Horizontal)
        self.depth_cue.setRange(0, 300)
        self.depth_cue.setSingleStep(1)
        self.depth_cue.setPageStep(10)
        self.depth_cue.setTickPosition(QSlider.TicksBelow)
        self.depth_cue.setTickInterval(25)
        self.depth_cue.setValue(100)
        self.depth_cue_min = QLabel("0.0x")
        self.depth_cue_max = QLabel("3.0x")
        self.depth_cue_value = QLabel("1.0x")
        self.depth_cue_min.setStyleSheet("color: #9aa6bb; font-size: 11px;")
        self.depth_cue_max.setStyleSheet("color: #9aa6bb; font-size: 11px;")

        self.show_bonds = QCheckBox("bonds")
        self.show_bonds.setChecked(True)
        self.projection_mode = QComboBox()
        self.projection_mode.addItems(["orthographic", "perspective", "oblique"])
        self.reset_preview_button = QPushButton("Reset View")
        self.movie_play = QPushButton("Play Movie")
        self.movie_play.setEnabled(False)
        self.show_plots = QPushButton("Show Plots")
        self.show_plots.setEnabled(False)
        self.open_results_folder = QPushButton("Open Results Folder")
        self.open_relaxed_xyz = QPushButton("Open Relaxed XYZ")
        self.open_relaxed_data = QPushButton("Open Relaxed DATA")
        self.open_trajectory = QPushButton("Open Trajectory")
        self.open_results_folder.setEnabled(False)
        self.open_relaxed_xyz.setEnabled(False)
        self.open_relaxed_data.setEnabled(False)
        self.open_trajectory.setEnabled(False)
        self.movie_frame = QSlider(Qt.Horizontal)
        self.movie_frame.setRange(0, 0)
        self.movie_frame.setEnabled(False)
        self.movie_label = QLabel("No trajectory loaded")
        self.movie_label.setStyleSheet("color: #9aa6bb;")
        self.movie_timer = QTimer(self)
        self.movie_timer.setInterval(120)
        self.plot_dialog = ThermoPlotDialog(self)

        result_open_row = QWidget(self)
        result_open_layout = QHBoxLayout(result_open_row)
        result_open_layout.setContentsMargins(0, 0, 0, 0)
        result_open_layout.setSpacing(8)
        result_open_layout.addWidget(self.open_results_folder)
        result_open_layout.addWidget(self.open_relaxed_xyz)
        result_open_layout.addWidget(self.open_relaxed_data)
        result_open_layout.addWidget(self.open_trajectory)
        right_layout.addWidget(result_open_row)

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
        atom_labels = QWidget(self)
        atom_labels_layout = QHBoxLayout(atom_labels)
        atom_labels_layout.setContentsMargins(2, 0, 2, 0)
        atom_labels_layout.setSpacing(0)
        atom_labels_layout.addWidget(self.atom_radius_min)
        atom_labels_layout.addStretch(1)
        atom_labels_layout.addWidget(self.atom_radius_max)
        slider_block_layout.addWidget(atom_labels)
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

        movie_box = QGroupBox("Trajectory Movie")
        movie_box_layout = QVBoxLayout(movie_box)
        movie_box_layout.setContentsMargins(8, 8, 8, 8)
        movie_box_layout.setSpacing(8)
        movie_box_layout.addWidget(self.movie_label)
        movie_box_layout.addWidget(self.movie_frame)
        movie_box_layout.addWidget(self.movie_play)
        movie_box_layout.addWidget(self.show_plots)
        right_layout.addWidget(movie_box)
        right_layout.addStretch(1)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(left_panel)
        splitter.addWidget(middle_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([520, 760, 320])
        root_layout.addWidget(splitter, 1)

        self._browse_structure_button = browse_structure
        self._use_poscar_button = use_poscar
        self._browse_working_folder_button = browse_working_folder
        self._refresh_working_folder_button = refresh_working_folder
        self._browse_potential_button = browse_potential
        self._run_button = run_button
        self._write_button = write_button

    def _wire_signals(self):
        self._browse_structure_button.clicked.connect(self._browse_structure)
        self._use_poscar_button.clicked.connect(self._use_last_generated)
        self._browse_working_folder_button.clicked.connect(self._browse_working_folder)
        self._refresh_working_folder_button.clicked.connect(self._refresh_working_folder)
        self._browse_potential_button.clicked.connect(self._browse_potential)
        self._run_button.clicked.connect(self.run_relaxation)
        self._write_button.clicked.connect(self.write_lammps_files)
        self.structure_path.editingFinished.connect(self._load_current_structure)
        self.working_folder.editingFinished.connect(self._working_folder_edited)
        self.atom_radius.valueChanged.connect(self._sync_atom_radius)
        self.bond_radius.valueChanged.connect(self._sync_bond_radius)
        self.depth_cue.valueChanged.connect(self._sync_depth_cue)
        self.projection_mode.currentTextChanged.connect(self.preview.set_projection)
        self.show_bonds.toggled.connect(self.preview.set_show_bonds)
        self.reset_preview_button.clicked.connect(self._reset_preview_controls)
        self.movie_play.clicked.connect(self._toggle_movie)
        self.movie_frame.valueChanged.connect(self._show_movie_frame)
        self.movie_timer.timeout.connect(self._advance_movie_frame)
        self.show_plots.clicked.connect(self._show_plots)
        self.open_results_folder.clicked.connect(self._open_results_folder)
        self.open_relaxed_xyz.clicked.connect(self._open_relaxed_xyz)
        self.open_relaxed_data.clicked.connect(self._open_relaxed_data)
        self.open_trajectory.clicked.connect(self._open_trajectory)
        self.working_file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.working_file_list.itemDoubleClicked.connect(self._working_file_double_clicked)
        self.working_file_list.customContextMenuRequested.connect(self._working_file_context_menu)

    def _browse_structure(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open geometry file",
            str(Path(self.structure_path.text()).expanduser() if self.structure_path.text().strip() else Path.cwd()),
            "Geometry files (*.xyz *.cif *.vasp *.POSCAR *.CONTCAR *.traj *.data *.db *.*)",
        )
        if path:
            self.set_structure_path(path)

    def _browse_potential(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open LAMMPS potential file",
            str(Path(self.potential_file.text()).expanduser() if self.potential_file.text().strip() else Path.cwd()),
            "Potential files (*.*)",
        )
        if path:
            self.potential_file.setText(path)

    def _use_last_generated(self):
        current = Path(self.structure_path.text().strip()).expanduser()
        candidates = [current, Path("POSCAR"), Path("CONTCAR"), Path("np_gen.xyz"), Path("lammps.relaxed.data"), Path("lammps.start.data")]
        for candidate in candidates:
            if candidate.exists():
                self.set_structure_path(str(candidate))
                return
        QMessageBox.information(self, "No generated structure", "No recent generated geometry was found in the current folder.")

    def _read_atoms_from_path(self, path: Path):
        suffix = path.suffix.lower()
        if suffix == ".data":
            return read(str(path), format="lammps-data")
        return read(str(path))

    def _browse_working_folder(self):
        if self._working_folder_dialog_open:
            return
        self._working_folder_dialog_open = True
        self._browse_working_folder_button.setEnabled(False)
        start_dir = self.working_folder.text().strip() or self._structure_workdir()
        try:
            dialog = WorkingFolderDialog(start_dir, self)
            if dialog.exec():
                selected = dialog.selected_folder()
                if selected:
                    QTimer.singleShot(0, lambda p=selected: self.set_working_folder(p, explicit=True, auto_load=False))
        finally:
            self._working_folder_dialog_open = False
            self._browse_working_folder_button.setEnabled(True)

    def _working_folder_edited(self):
        folder = self.working_folder.text().strip()
        if folder:
            self.set_working_folder(folder, explicit=True, auto_load=False)

    def _refresh_working_folder(self):
        folder = self.working_folder.text().strip()
        if folder:
            self.set_working_folder(folder, explicit=True, auto_load=False)

    def set_working_folder(self, folder: str, *, explicit: bool = False, auto_load: bool = True):
        path = Path(folder).expanduser()
        self.working_folder.setText(str(path))
        self._working_folder_explicit = bool(explicit)
        if not path.exists():
            self.working_file_list.clear()
            self.output_xyz.clear()
            self.output_poscar.clear()
            self.trajectory_file.clear()
            self.relaxation_folder.clear()
            self._clear_movie_controls()
            self.movie_label.setText("Working folder not found")
            return
        self.relaxation_folder.setText(str(path))
        self._refresh_working_folder_files(path)
        if auto_load:
            self._load_working_folder_outputs(path)

    def _working_folder_path(self) -> Path | None:
        folder_text = self.working_folder.text().strip()
        if not folder_text:
            return None
        return Path(folder_text).expanduser()

    def _structure_workdir(self, structure_path: str | None = None) -> str:
        folder = self._working_folder_path()
        if folder is not None:
            return str(folder)
        path_text = (structure_path or self.structure_path.text().strip() or "POSCAR").strip()
        return str(Path(path_text).expanduser().resolve().parent)

    def _refresh_working_folder_files(self, folder: Path):
        self.working_file_list.blockSignals(True)
        self.working_file_list.clear()
        if folder.exists():
            try:
                entries = sorted(
                    [p for p in folder.iterdir() if p.is_file()],
                    key=lambda p: (p.suffix.lower(), p.name.lower()),
                )
            except Exception:
                entries = []
            for path in entries:
                item = QListWidgetItem(path.name)
                item.setToolTip(str(path))
                item.setData(Qt.UserRole, str(path))
                self.working_file_list.addItem(item)
        self.working_file_list.blockSignals(False)

    def _pick_existing_file(self, folder: Path, names: list[str], suffixes: tuple[str, ...] = ()) -> Path | None:
        for name in names:
            candidate = folder / name
            if candidate.exists():
                return candidate
        for suffix in suffixes:
            matches = sorted(
                [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == suffix],
                key=lambda p: p.name.lower(),
            )
            if matches:
                return matches[0]
        return None

    def _load_text_file_into_report(self, path: Path):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            self.report.setPlainText(f"Could not load text file:\n{path}\n\n{exc}")
            return
        self.report.setPlainText(f"Working folder file: {path}\n\n{text}")

    def _load_thermo_from_log_file(self, path: Path):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return
        thermo = _parse_thermo_history(text)
        if thermo.get("step"):
            self.plot_dialog.set_thermo_data(
                thermo=thermo,
                title=f"{path.parent.name} - ",
                timestep=float(self.timestep.value()),
            )

    def _load_working_folder_outputs(self, folder: Path):
        self.relaxation_folder.setText(str(folder))
        self.output_xyz.clear()
        self.output_poscar.clear()
        self.output_data.clear()
        relaxed_xyz = self._pick_existing_file(folder, ["lammps.relaxed.xyz", "relaxed.xyz"], suffixes=(".xyz",))
        if relaxed_xyz is not None:
            self.output_xyz.setText(str(relaxed_xyz))
            try:
                self.preview.set_cluster(build_cluster_preview(relaxed_xyz))
            except Exception:
                pass

        relaxed_poscar = self._pick_existing_file(folder, ["lammps.relaxed.POSCAR", "relaxed.POSCAR"], suffixes=(".vasp", ".poscar"))
        if relaxed_poscar is not None:
            self.output_poscar.setText(str(relaxed_poscar))

        relaxed_data = self._pick_existing_file(folder, ["lammps.relaxed.data", "relaxed.data", "lammps.start.data"], suffixes=(".data",))
        if relaxed_data is not None:
            self.output_data.setText(str(relaxed_data))
            try:
                self.preview.set_cluster(build_cluster_preview_from_atoms(self._read_atoms_from_path(relaxed_data), path=str(relaxed_data)))
            except Exception:
                pass
        trajectory = self._pick_existing_file(
            folder,
            ["lammps.relax.traj", "lammps.traj", "traj.traj", "trajectory.traj"],
            suffixes=(".traj",),
        )
        if trajectory is not None:
            self.movie_label.setText("Loading selected MD trajectory data...")
            QApplication.processEvents()
            self.trajectory_file.setText(str(trajectory))
            self._load_movie_from_trajectory(str(trajectory))
        else:
            self._clear_movie_controls()

        output = self._pick_existing_file(
            folder,
            ["lammps.relax.out", "lammps.relax.log", "lammps.out", "lammps.log", "slurm.out"],
            suffixes=(".out", ".log"),
        )
        if output is not None:
            self._load_text_file_into_report(output)
            self._load_thermo_from_log_file(output)

    def _selected_working_file_path(self, item: QListWidgetItem | None = None) -> Path | None:
        if item is None:
            item = self.working_file_list.currentItem()
        if item is None:
            return None
        path_text = item.data(Qt.UserRole) or item.text()
        path = Path(str(path_text)).expanduser()
        if not path.exists():
            return None
        return path

    def _is_geometry_file(self, path: Path) -> bool:
        suffix = path.suffix.lower()
        name = path.name.lower()
        return suffix in {".xyz", ".vasp", ".poscar", ".contcar", ".data"} or name in {"poscar", "contcar"}

    def _is_text_file(self, path: Path) -> bool:
        return path.suffix.lower() in {".out", ".log", ".txt"}

    def _load_working_file(self, path: Path):
        if path.suffix.lower() == ".traj":
            self.movie_label.setText("Loading selected MD trajectory data...")
            QApplication.processEvents()
            self.trajectory_file.setText(str(path))
            self._load_movie_from_trajectory(str(path))
            return
        if self._is_text_file(path):
            self._load_text_file_into_report(path)
            self._load_thermo_from_log_file(path)
            return
        if self._is_geometry_file(path):
            try:
                if path.suffix.lower() == ".data" or path.name.lower() in {"poscar", "contcar"}:
                    self.preview.set_cluster(build_cluster_preview_from_atoms(self._read_atoms_from_path(path), path=str(path)))
                else:
                    self.preview.set_cluster(build_cluster_preview(path))
            except Exception as exc:
                QMessageBox.warning(self, "Preview failed", str(exc))
            return
        self._open_in_editor(path)

    def _open_in_editor(self, path: Path):
        if not path.exists():
            QMessageBox.information(self, "Path not found", f"Path not found:\n{path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _save_geometry_as_xyz(self, path: Path):
        if not self._is_geometry_file(path):
            QMessageBox.information(self, "Not a geometry file", f"This file is not a geometry file:\n{path.name}")
            return
        try:
            atoms = self._read_atoms_from_path(path)
            out_path = _xyz_output_path(path)
            _write_xyz_file(out_path, atoms)
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        QMessageBox.information(self, "Saved", f"Saved geometry as XYZ:\n{out_path}")

    def _load_action_label(self, path: Path) -> str:
        suffix = path.suffix.lower()
        name = path.name.lower()
        if suffix == ".traj":
            return "Load LAMMPS trajectory file"
        if suffix in {".out", ".log", ".txt"}:
            return "Load LAMMPS output file"
        if suffix == ".data" or suffix in {".xyz", ".vasp", ".poscar", ".contcar"} or name in {"poscar", "contcar"}:
            return "Load geometry file"
        return "Load file"

    def _working_file_double_clicked(self, item: QListWidgetItem):
        path = self._selected_working_file_path(item)
        if path is None:
            return
        self._load_working_file(path)

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
            self.preview.clear_preview()
            self.initial_atoms_value.setText("0")
            self.final_atoms_value.setText("0")
            self.initial_nn_value.setText("unknown")
            self.final_nn_value.setText("unknown")
            self.initial_pe_value.setText("unknown")
            self.final_pe_value.setText("unknown")
            return
        path = Path(path_text).expanduser()
        if not path.exists():
            self.preview.clear_preview()
            self.report.setPlainText(f"Structure file not found: {path}")
            return
        try:
            atoms = self._read_atoms_from_path(path)
            self.initial_atoms_value.setText(str(len(atoms)))
            self.final_atoms_value.setText(str(len(atoms)))
            avg_nn = _average_nn_distance(atoms)
            self.initial_nn_value.setText(f"{avg_nn:.3f} Å" if avg_nn > 0 else "unknown")
            self.final_nn_value.setText(f"{avg_nn:.3f} Å" if avg_nn > 0 else "unknown")
            if path.suffix.lower() == ".data":
                self.preview.set_cluster(build_cluster_preview_from_atoms(atoms, path=str(path)))
            else:
                self.preview.load_structure_file(path)
            if not self.working_folder.text().strip():
                self.set_working_folder(str(path.parent), explicit=False, auto_load=False)
        except Exception as exc:
            self.preview.clear_preview()
            self.report.setPlainText(f"Could not load geometry preview:\n{exc}")

    def _clear_movie_controls(self):
        self.movie_timer.stop()
        self.movie_play.setText("Play Movie")
        self.movie_play.setEnabled(False)
        self.show_plots.setEnabled(False)
        self.open_results_folder.setEnabled(False)
        self.open_relaxed_xyz.setEnabled(False)
        self.open_relaxed_data.setEnabled(False)
        self.open_trajectory.setEnabled(False)
        self.movie_frame.blockSignals(True)
        self.movie_frame.setRange(0, 0)
        self.movie_frame.setValue(0)
        self.movie_frame.blockSignals(False)
        self.movie_frame.setEnabled(False)
        self.movie_label.setText("No trajectory loaded")
        self.preview.set_movie_frames([])
        self.plot_dialog.set_thermo_data(thermo={}, title="")

    def _append_report_line(self, text: str):
        existing = self.report.toPlainText().rstrip()
        if existing:
            self.report.setPlainText(existing + "\n" + text)
        else:
            self.report.setPlainText(text)
        cursor = self.report.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.report.setTextCursor(cursor)

    def _load_movie_from_trajectory(self, trajectory_path: str):
        path = Path(trajectory_path).expanduser()
        if not path.exists():
            self._clear_movie_controls()
            self.movie_label.setText("Trajectory file not found")
            return
        try:
            frames = read(str(path), index=":")
        except Exception as exc:
            self._clear_movie_controls()
            self.movie_label.setText(f"Could not load trajectory: {exc}")
            return
        if frames is None:
            frames = []
        if not isinstance(frames, list):
            frames = [frames]
        previews = [build_cluster_preview_from_atoms(frame) for frame in frames] if frames else []
        if not previews:
            self._clear_movie_controls()
            self.movie_label.setText("Trajectory is empty")
            return
        self.preview.set_movie_frames(previews)
        self.movie_frame.blockSignals(True)
        self.movie_frame.setRange(0, len(previews) - 1)
        self.movie_frame.setValue(0)
        self.movie_frame.blockSignals(False)
        self.movie_frame.setEnabled(True)
        self.movie_play.setEnabled(True)
        self.show_plots.setEnabled(True)
        self.movie_play.setText("Play Movie")
        self.movie_label.setText(f"{len(previews)} frames loaded")
        self.preview.set_cluster(previews[0])

    def _toggle_movie(self):
        if not self.preview.movie_frame_count():
            return
        if self.movie_timer.isActive():
            self.movie_timer.stop()
            self.movie_play.setText("Play Movie")
        else:
            self.movie_timer.start()
            self.movie_play.setText("Pause Movie")

    def _advance_movie_frame(self):
        if not self.preview.movie_frame_count():
            self.movie_timer.stop()
            return
        frame = self.preview.movie_frame_index() + 1
        if frame >= self.preview.movie_frame_count():
            frame = 0
        self.movie_frame.blockSignals(True)
        self.movie_frame.setValue(frame)
        self.movie_frame.blockSignals(False)
        self._show_movie_frame(frame)

    def _show_movie_frame(self, frame: int):
        if not self.preview.movie_frame_count():
            return
        self.preview.set_movie_frame_index(int(frame))
        self.movie_label.setText(f"Frame {int(frame) + 1}/{self.preview.movie_frame_count()}")

    def _show_plots(self):
        self.plot_dialog.show()
        self.plot_dialog.raise_()
        self.plot_dialog.activateWindow()

    def _open_path(self, path_text: str):
        path = Path(path_text).expanduser()
        if not path.exists():
            QMessageBox.information(self, "Path not found", f"Path not found:\n{path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_results_folder(self):
        result = self._last_relaxation_result or self._last_write_result or {}
        folder = result.get("inputs_dir") or result.get("workdir") or self.relaxation_folder.text().strip()
        if not folder:
            QMessageBox.information(self, "No results", "No relaxation results folder is available yet.")
            return
        self._open_path(str(folder))

    def _open_relaxed_xyz(self):
        result = self._last_relaxation_result or {}
        path_text = result.get("relaxed_xyz") or self.output_xyz.text().strip()
        if not path_text:
            QMessageBox.information(self, "No results", "No relaxed XYZ file is available yet.")
            return
        self._open_path(str(path_text))

    def _open_relaxed_data(self):
        result = self._last_relaxation_result or {}
        path_text = result.get("relaxed_data") or self.output_data.text().strip()
        if not path_text:
            QMessageBox.information(self, "No results", "No relaxed DATA file is available yet.")
            return
        self._open_path(str(path_text))

    def _open_trajectory(self):
        result = self._last_relaxation_result or {}
        path_text = result.get("trajectory_file") or self.trajectory_file.text().strip()
        if not path_text:
            QMessageBox.information(self, "No results", "No trajectory file is available yet.")
            return
        self._open_path(str(path_text))

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

    def _collect_settings(self) -> dict[str, Any]:
        return {
            "lammps_command": self.lammps_command.text().strip() or "lmp",
            "potential_file": self.potential_file.text().strip(),
            "pair_style": self.pair_style.text().strip() or "eam/alloy",
            "pair_coeff": self.pair_coeff.text().strip(),
            "temperature": float(self.temperature.value()),
            "thermal_steps": int(self.thermal_steps.value()),
            "quench_temperature": float(self.quench_temperature.value()),
            "quench_steps": int(self.quench_steps.value()),
            "timestep": float(self.timestep.value()),
            "damping": float(self.damping.value()),
            "seed": int(self.seed.value()),
            "minimize_etol": float(self.minimize_etol.value()),
            "minimize_ftol": float(self.minimize_ftol.value()),
            "minimize_maxiter": int(self.minimize_maxiter.value()),
            "minimize_maxeval": int(self.minimize_maxeval.value()),
            "vacuum": float(self.vacuum.value()),
            "remove_drift": self.remove_drift.isChecked(),
            "freeze_core": self.freeze_core.isChecked(),
            "surface_threshold": int(self.surface_threshold.value()),
            "dump_interval": int(self.dump_interval.value()),
            "hist_rmin": 1.8,
            "hist_rmax": 4.2,
            "workdir": self._structure_workdir(),
        }

    def run_relaxation(self):
        self.run_lammps_locally()

    def run_lammps_locally(self):
        structure_path = self.structure_path.text().strip()
        if not structure_path:
            QMessageBox.warning(self, "Missing structure", "Select a structure file to relax.")
            return
        if not Path(structure_path).expanduser().exists():
            QMessageBox.warning(self, "Missing structure", f"Structure file not found: {structure_path}")
            return
        if not self.pair_style.text().strip():
            QMessageBox.warning(self, "Missing potential", "Provide a LAMMPS pair style.")
            return
        if not self.pair_coeff.text().strip() and not self.potential_file.text().strip():
            QMessageBox.warning(
                self,
                "Missing potential",
                "Provide a potential file or a custom pair_coeff line for LAMMPS.",
            )
            return

        self.report.setPlainText("Running thermal relaxation...\n")
        self.output_xyz.clear()
        self.output_poscar.clear()
        self.output_data.clear()
        self.trajectory_file.clear()
        self.relaxation_folder.clear()
        self._last_relaxation_result = None
        self._clear_movie_controls()
        self.setEnabled(False)
        self._worker = ThermalRelaxationWorker(structure_path, self._collect_settings())
        self._worker.status.connect(self._append_report_line)
        self._worker.finished_ok.connect(self._relaxation_finished)
        self._worker.failed.connect(self._relaxation_failed)
        self._worker.finished.connect(self._relaxation_cleanup)
        self._worker.start()

    def write_lammps_files(self):
        structure_path = self.structure_path.text().strip()
        if not structure_path:
            QMessageBox.warning(self, "Missing structure", "Select a structure file first.")
            return
        if not Path(structure_path).expanduser().exists():
            QMessageBox.warning(self, "Missing structure", f"Structure file not found: {structure_path}")
            return
        if not self.pair_style.text().strip():
            QMessageBox.warning(self, "Missing potential", "Provide a LAMMPS pair style.")
            return
        if not self.pair_coeff.text().strip() and not self.potential_file.text().strip():
            QMessageBox.warning(
                self,
                "Missing potential",
                "Provide a potential file or a custom pair_coeff line for LAMMPS.",
            )
            return

        self.report.setPlainText("Writing LAMMPS input files...\n")
        self._last_write_result = None
        self.output_data.clear()
        self.setEnabled(False)
        self._write_worker = ThermalRelaxationWriteWorker(structure_path, self._collect_settings())
        self._write_worker.status.connect(self._append_report_line)
        self._write_worker.finished_ok.connect(self._write_files_finished)
        self._write_worker.failed.connect(self._relaxation_failed)
        self._write_worker.finished.connect(self._write_files_cleanup)
        self._write_worker.start()

    def _write_files_finished(self, result: dict):
        self._last_write_result = dict(result)
        self.relaxation_folder.setText(str(result.get("inputs_dir", result.get("workdir", ""))))
        self.open_results_folder.setEnabled(True)
        data_path = str(result.get("data_path", ""))
        if data_path:
            self.output_data.setText(data_path)
            self.open_relaxed_data.setEnabled(True)
        lines = [
            "LAMMPS input files written.",
            f"Structure: {result.get('structure_path', '')}",
            f"Workdir: {result.get('workdir', '')}",
            f"LAMMPS inputs: {result.get('inputs_dir', '')}",
            f"Data file: {result.get('data_path', '')}",
            f"Script file: {result.get('script_path', '')}",
            f"Site count CSV: {result.get('site_count_before_csv', '')}",
            f"Potential file: {result.get('potential_file', '') or 'none'}",
            f"Copied potential: {result.get('copied_potential_file', '') or 'none'}",
            "",
            "Remote run hint:",
            f"  cd {result.get('inputs_dir', '')} && {result.get('lammps_command', '')} -in {Path(result.get('script_path', '')).name}",
        ]
        self.report.setPlainText("\n".join(lines))

    def _write_files_cleanup(self):
        self.setEnabled(True)
        self._write_worker = None

    def _relaxation_finished(self, result: dict):
        self._last_relaxation_result = dict(result)
        self.initial_atoms_value.setText(str(result.get("initial_atoms", 0)))
        self.final_atoms_value.setText(str(result.get("final_atoms", 0)))
        self.initial_nn_value.setText(f"{float(result.get('initial_average_nn_distance', 0.0)):.3f} Å")
        self.final_nn_value.setText(f"{float(result.get('final_average_nn_distance', 0.0)):.3f} Å")
        init_pe = result.get("initial_potential_energy")
        final_pe = result.get("final_potential_energy")
        self.initial_pe_value.setText(f"{float(init_pe):.6f}" if init_pe is not None else "unknown")
        self.final_pe_value.setText(f"{float(final_pe):.6f}" if final_pe is not None else "unknown")
        self.output_xyz.setText(str(result.get("relaxed_xyz", "")))
        self.output_poscar.setText(str(result.get("relaxed_poscar", "")))
        self.output_data.setText(str(result.get("relaxed_data", "")))
        self.trajectory_file.setText(str(result.get("trajectory_file", "")))
        self.relaxation_folder.setText(str(result.get("inputs_dir", result.get("workdir", ""))))
        self.open_results_folder.setEnabled(True)
        self.open_relaxed_xyz.setEnabled(True)
        self.open_relaxed_data.setEnabled(True)
        self.open_trajectory.setEnabled(True)
        relaxed_xyz = str(result.get("relaxed_xyz", ""))
        if relaxed_xyz and getattr(self.parent(), "site_counter_panel", None) is not None:
            self.parent().site_counter_panel.set_structure_path(relaxed_xyz)
        try:
            self.preview.set_cluster(build_cluster_preview(relaxed_xyz))
        except Exception:
            pass
        self._load_movie_from_trajectory(str(result.get("trajectory_file", "")))
        self.plot_dialog.set_thermo_data(
            thermo=result.get("thermo_history", {}),
            title="Relaxation - ",
            timestep=float(result.get("timestep", self.timestep.value())),
        )
        lines = [
            f"Input: {result.get('input_file', '')}",
            f"LAMMPS: {result.get('lammps_command', '')}",
            f"Potential file: {self.potential_file.text().strip() or 'none'}",
            f"LAMMPS inputs: {result.get('inputs_dir', '')}",
            f"Pair style: {result.get('pair_style', '')}",
            f"Pair coeff: {result.get('pair_coeff', '')}",
            f"Output directory: {result.get('workdir', '')}",
            f"Site count before: {result.get('site_count_before_csv', '')}",
            f"Surface-only motion: {'yes' if self.freeze_core.isChecked() else 'no'}",
            f"Temperature: {result.get('temperature', 0.0)} K",
            f"Thermal steps: {result.get('thermal_steps', 0)}",
            f"Quench temperature: {result.get('quench_temperature', 0.0)} K",
            f"Quench steps: {result.get('quench_steps', 0)}",
            f"Timestep: {result.get('timestep', 0.0)} ps",
            f"Damping: {result.get('damping', 0.0)} ps",
            f"Vacuum: {self.vacuum.value():.2f} Å",
            f"Initial atoms: {result.get('initial_atoms', 0)}",
            f"Final atoms: {result.get('final_atoms', 0)}",
            f"Surface atoms: {result.get('surface_atoms', 0)}",
            f"Mobile atoms: {result.get('mobile_atoms', 0)}",
            f"Initial avg. NN: {result.get('initial_average_nn_distance', 0.0):.3f} Å",
            f"Final avg. NN: {result.get('final_average_nn_distance', 0.0):.3f} Å",
            f"Trajectory: {result.get('trajectory_file', '')}",
        ]
        if result.get("initial_potential_energy") is not None:
            lines.append(f"Initial PE: {float(result['initial_potential_energy']):.6f}")
        if result.get("final_potential_energy") is not None:
            lines.append(f"Final PE: {float(result['final_potential_energy']):.6f}")
        lines.append("")
        lines.append("LAMMPS log:")
        lines.append(result.get("log_text", ""))
        self.report.setPlainText("\n".join(lines))

    def _relaxation_failed(self, message: str):
        self.report.setPlainText(message)
        QMessageBox.critical(self, "Thermal relaxation failed", message)

    def _relaxation_cleanup(self):
        self.setEnabled(True)
        self._worker = None

    def set_structure_path(self, path: str):
        self.structure_path.setText(path)
        self._load_current_structure()
