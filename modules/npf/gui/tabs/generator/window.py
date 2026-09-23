from __future__ import annotations

import contextlib
import io
import os
import threading
from pathlib import Path
from typing import Any

from ase.data import covalent_radii

from ....common import DEFAULT_A, DEFAULT_C
from ....crystal import CrystalSource, load_crystal
from ....geometry_map import normalize_shape_key, normalize_structure_class, shape_label
from ....input import build_from_spec, load_input_spec
from ....common import cluster_output_dir
from ...preview import ClusterPreviewView, build_cluster_preview, build_cluster_preview_from_atoms
from ...server import ROOT, spec_to_input_text


_CWD_LOCK = threading.Lock()
_NM_TO_ANGSTROM = 10.0
_CRYSTAL_DRIVEN_SHAPES = [
    ("sphere", "sphere"),
    ("wulff", "wulff"),
    ("cube", "cube"),
    ("octahedron", "octahedron"),
    ("icosahedron", "ico"),
    ("decahedron", "decahedron"),
    ("dodecahedron", "dodecahedron"),
    ("morphed spherical", "morphed_spherical"),
    ("cuboctahedron", "cuboct"),
    ("hexagonal prism", "hexagonal_prism"),
    ("truncated hexagonal prism", "truncated_hexagonal_prism"),
    ("hexagonal bipyramid", "hexagonal_bipyramid"),
    ("truncated hexagonal bipyramid", "truncated_hexagonal_bipyramid"),
    ("nanorod", "nanorod"),
    ("hexagonal platelet", "hexagonal_platelet"),
]


def _normalize_structure(value: str | None) -> str:
    structure = (value or "hexagonal").strip().lower()
    return {
        "hcp": "hexagonal",
        "fcc": "cubic",
        "bcc": "cubic",
        "trigonal": "rhombohedral",
        "custom": "triclinic",
    }.get(structure, structure)


def _normalize_variant(value: str | None) -> str | None:
    if value is None:
        return None
    variant = value.strip().lower()
    return variant or None


def _build_cell_edges(atoms, repeat: tuple[int, int, int]):
    rx, ry, rz = repeat
    cell = atoms.cell.array
    a = [float(v) for v in cell[0]]
    b = [float(v) for v in cell[1]]
    c = [float(v) for v in cell[2]]
    va = [rx * v for v in a]
    vb = [ry * v for v in b]
    vc = [rz * v for v in c]

    def add(u, v):
        return [u[0] + v[0], u[1] + v[1], u[2] + v[2]]

    o = [0.0, 0.0, 0.0]
    p1 = va
    p2 = vb
    p3 = vc
    p4 = add(va, vb)
    p5 = add(va, vc)
    p6 = add(vb, vc)
    p7 = add(add(va, vb), vc)
    corners = [o, p1, p2, p3, p4, p5, p6, p7]
    edge_idx = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]
    return [[corners[i], corners[j]] for i, j in edge_idx]


def _shape_default_output(element: str, shape: str, size_mode: str, atoms: int | None, diameter: float | None) -> str:
    if size_mode == "atoms":
        size_part = f"{atoms}" if atoms is not None else "NA"
    else:
        size_part = f"d{diameter:g}nm" if diameter is not None else "dNAnm"
    return f"{element}_{shape}_{size_part}.xyz"


def _build_spec_from_ui(values: dict[str, Any]) -> dict[str, Any]:
    shape = values["shape"]
    element = values["element"] or "Ru"
    crystal_structure = _normalize_structure(values.get("crystal_structure"))
    size_mode = values.get("size_mode") or "diameter"
    atoms = int(values["atoms"]) if size_mode == "atoms" and values.get("atoms") not in {None, ""} else None
    diameter = (
        float(values["diameter"])
        if size_mode == "diameter" and values.get("diameter") not in {None, ""}
        else None
    )
    radius = diameter / 2.0 if diameter is not None else None
    output_file_name = values.get("output_file_name") or _shape_default_output(element, shape, size_mode, atoms, diameter)

    spec: dict[str, Any] = {
        "shape": shape,
        "crystal_file": values.get("crystal_file"),
        "input_file_name": values.get("input_file_name") or "npf_np_gen.in",
        "output_file_name": output_file_name,
        "tag": Path(output_file_name).stem,
        "element": element,
        "crystal_structure": crystal_structure,
        "crystal_variant": _normalize_variant(values.get("crystal_variant")),
        "size_mode": size_mode,
        "atoms": atoms,
        "radius": radius,
        "a": values.get("a"),
        "b": values.get("b"),
        "c": values.get("c"),
        "alpha": values.get("alpha"),
        "beta": values.get("beta"),
        "gamma": values.get("gamma"),
        "verbose": bool(values.get("verbose", False)),
    }

    if shape == "wulff":
        spec.update(
            {
                "surfaces": values.get("surfaces") or "",
                "facet_mode": values.get("facet_mode") or "top",
                "extra_facets": values.get("extra_facets") or "none",
                "sphericity": float(values.get("sphericity") or 0.0),
                "radius_tol": float(values.get("radius_tol") or 0.3),
                "max_refine": int(values.get("max_refine") or 12),
                "radius_policy": values.get("radius_policy") or "closest",
                "radial_trim": values.get("radial_trim"),
            }
        )
    elif shape == "ico":
        spec["ico_shells"] = values.get("ico_shells")
    elif shape == "cube":
        spec["cube_layers"] = values.get("cube_layers")
    elif shape == "octahedron":
        spec["octa_length"] = values.get("octa_length")
        spec["octa_cutoff"] = int(values.get("octa_cutoff") or 0)
    elif shape == "decahedron":
        spec["p"] = values.get("p")
        spec["q"] = values.get("q")
        spec["r"] = values.get("r")
    elif shape == "morphed_spherical":
        spec["morph"] = float(values.get("morph") or 0.0)
    elif shape == "cuboct":
        spec["cuboct_layers"] = values.get("cuboct_layers")

    return spec


def run_qt_gui(host: str = "127.0.0.1", port: int = 8765) -> None:
    try:
        from PySide6.QtCore import Qt, QThread, Signal
        from PySide6.QtGui import QAction, QCursor, QDoubleValidator, QIntValidator, QSurfaceFormat
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QFileDialog,
            QPlainTextEdit,
            QScrollArea,
            QSplitter,
            QSpinBox,
            QStatusBar,
            QTabWidget,
            QSlider,
            QSizePolicy,
            QToolButton,
            QToolTip,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:  # pragma: no cover - runtime dependency gate
        raise RuntimeError("PySide6 is not installed. Install PySide6 to use the native GUI.") from exc

    from ..md_relax.panel import ThermalRelaxationPanel
    from ..site_counter.panel import SiteCounterPanel

    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setVersion(2, 1)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setAlphaBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)

    class GenerationWorker(QThread):
        finished_ok = Signal(object, object, str, object)
        failed = Signal(str)

        def __init__(self, spec: dict[str, Any], working_path: str):
            super().__init__()
            self.spec = spec
            self.working_path = working_path

        def run(self) -> None:
            working_dir = Path(self.working_path).expanduser().resolve()
            try:
                working_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self.failed.emit(f"Could not create working path '{working_dir}': {exc}")
                return

            stdout = io.StringIO()
            original_cwd = Path.cwd()
            try:
                with _CWD_LOCK:
                    os.chdir(working_dir)
                    try:
                        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
                            result = build_from_spec(self.spec)
                    finally:
                        os.chdir(original_cwd)
                tag = self.spec.get("tag") or Path(self.spec.get("output_file_name") or "Ru_NP").stem
                xyz_path = cluster_output_dir(working_dir, tag) / f"{tag}.xyz"
                preview = build_cluster_preview(xyz_path)
                self.finished_ok.emit(result, preview, stdout.getvalue(), self.spec)
            except Exception as exc:
                try:
                    os.chdir(original_cwd)
                except Exception:
                    pass
                output = stdout.getvalue()
                message = str(exc)
                if output.strip():
                    message = f"{message}\n\n{output}"
                self.failed.emit(message)

    class NPFNativeWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("NPF Generator")
            self.resize(1500, 980)
            self.setStatusBar(QStatusBar(self))
            self._worker: GenerationWorker | None = None
            self._last_spec: dict[str, Any] | None = None
            self._current_preview: dict[str, Any] | None = None
            self._crystal_source: CrystalSource | None = None

            self._build_ui()
            self._wire_signals()
            self.apply_spec({
                "shape": "sphere",
                "element": "Ru",
                "crystal_structure": "hexagonal",
                "crystal_variant": "",
                "size_mode": "diameter",
                "diameter": 20.0,
                "atoms": 147,
                "a": DEFAULT_A,
                "c": DEFAULT_C,
                "facet_mode": "top",
                "extra_facets": "none",
                "sphericity": 0.0,
                "radius_tol": 0.3,
                "max_refine": 12,
                "radius_policy": "closest",
                "verbose": True,
                "input_file_name": "npf_np_gen.in",
                "output_file_name": "",
            })

        def _build_ui(self):
            self._preview = ClusterPreviewView(self)
            self._summary = QLabel("Ready.")
            self._summary.setWordWrap(True)
            self._summary.setStyleSheet("font-size: 14px;")
            self._input_preview = QPlainTextEdit(self)
            self._input_preview.setReadOnly(True)
            self._input_preview.setPlaceholderText("Exact npf_np_gen.in will appear here.")
            self._log_view = QPlainTextEdit(self)
            self._log_view.setReadOnly(True)
            self._log_view.setPlaceholderText("Build log will appear here.")

            self.shape = QComboBox()
            self.size_mode = QComboBox()
            self.size_mode.addItems(["atoms", "diameter"])
            self.size_value_label = QLabel("Number of atoms")
            self.size_value_atoms = QLineEdit()
            self.size_value_atoms.setValidator(QIntValidator(1, 10_000_000, self))
            self.size_value_atoms.setText("147")
            self.size_value_diameter = QDoubleSpinBox()
            self.size_value_diameter.setDecimals(3)
            self.size_value_diameter.setRange(0.001, 1_000_000.0)
            self.size_value_diameter.setSingleStep(0.1)

            self.input_file_name = QLineEdit("npf_np_gen.in")
            self.output_file_name = QLineEdit()
            self.working_path = QLineEdit(str(ROOT))
            self.input_file_name.setClearButtonEnabled(True)
            self.output_file_name.setClearButtonEnabled(True)
            self.working_path.setClearButtonEnabled(True)

            self.surfaces = QPlainTextEdit("0001=0.162,10-10=0.181,10-11=0.180")
            self.facet_mode = QComboBox()
            self.facet_mode.addItems(["top", "minimal", "full"])
            self.extra_facets = QComboBox()
            self.extra_facets.addItems(["none", "auto"])
            self.sphericity = QDoubleSpinBox()
            self.sphericity.setRange(0.0, 1.0)
            self.sphericity.setSingleStep(0.05)
            self.radius_tol = QDoubleSpinBox()
            self.radius_tol.setRange(0.0, 100.0)
            self.radius_tol.setSingleStep(0.05)
            self.radius_tol.setValue(0.3)
            self.max_refine = QComboBox()
            self.max_refine.addItems([str(i) for i in range(1, 31)])
            self.radius_policy = QComboBox()
            self.radius_policy.addItems(["closest", "at_least", "at_most"])
            self.radial_trim = QLineEdit()
            self.radial_trim.setValidator(QDoubleValidator(0.0, 1_000_000.0, 4, self))

            self.ico_shells = QLineEdit()
            self.ico_shells.setValidator(QIntValidator(1, 10000, self))
            self.cube_layers = QLineEdit()
            self.cube_layers.setValidator(QIntValidator(1, 10000, self))
            self.octa_length = QLineEdit()
            self.octa_length.setValidator(QIntValidator(1, 10000, self))
            self.octa_cutoff = QLineEdit("0")
            self.octa_cutoff.setValidator(QIntValidator(0, 10000, self))
            self.p = QLineEdit()
            self.p.setValidator(QIntValidator(1, 10000, self))
            self.q = QLineEdit()
            self.q.setValidator(QIntValidator(1, 10000, self))
            self.r = QLineEdit()
            self.r.setValidator(QIntValidator(0, 10000, self))
            self.morph = QDoubleSpinBox()
            self.morph.setRange(0.0, 1.0)
            self.morph.setSingleStep(0.05)
            self.cuboct_layers = QLineEdit()
            self.cuboct_layers.setValidator(QIntValidator(1, 10000, self))

            self.verbose = QCheckBox("Verbose")
            self.verbose.setChecked(True)

            self._field_groups: dict[str, QWidget] = {}
            self._structure_fields: dict[str, QWidget] = {}
            self._shape_groups: dict[str, QWidget] = {}

            form_host = QWidget(self)
            form_layout = QVBoxLayout(form_host)
            form_layout.setContentsMargins(16, 16, 16, 16)
            form_layout.setSpacing(14)

            self.write_button = QPushButton("Write Input")
            self.load_button = QPushButton("Load Input")
            self.open_folder_button = QPushButton("Open Folder")
            file_box = QGroupBox("File Operations")
            file_layout = QVBoxLayout(file_box)
            file_layout.setContentsMargins(10, 10, 10, 10)
            file_layout.setSpacing(8)
            file_layout.addWidget(QLabel("Input file"))
            file_layout.addWidget(self.input_file_name)
            file_layout.addWidget(QLabel("Output file"))
            file_layout.addWidget(self.output_file_name)
            file_layout.addWidget(QLabel("Base path"))
            file_layout.addWidget(self.working_path)
            file_actions = QWidget(self)
            file_actions_layout = QHBoxLayout(file_actions)
            file_actions_layout.setContentsMargins(0, 8, 0, 0)
            file_actions_layout.setSpacing(10)
            file_actions_layout.addWidget(self.write_button)
            file_actions_layout.addWidget(self.load_button)
            file_actions_layout.addWidget(self.open_folder_button)
            file_actions_layout.addStretch(1)
            file_layout.addWidget(file_actions)
            form_layout.addWidget(file_box)

            np_box = QGroupBox("NP Properties")
            np_layout = QVBoxLayout(np_box)
            np_layout.setContentsMargins(10, 10, 10, 10)
            np_layout.setSpacing(10)
            np_basic = QWidget(self)
            np_basic_layout = QGridLayout(np_basic)
            np_basic_layout.setContentsMargins(0, 0, 0, 0)
            np_basic_layout.addWidget(QLabel("Shape"), 0, 0)
            np_basic_layout.addWidget(self.shape, 0, 1)
            np_basic_layout.addWidget(QLabel("Size mode"), 1, 0)
            np_basic_layout.addWidget(self.size_mode, 1, 1)
            size_widget = QWidget(self)
            size_layout = QVBoxLayout(size_widget)
            size_layout.setContentsMargins(0, 0, 0, 0)
            size_layout.addWidget(self.size_value_label)
            size_layout.addWidget(self.size_value_atoms)
            size_layout.addWidget(self.size_value_diameter)
            np_basic_layout.addWidget(size_widget, 2, 1)
            np_layout.addWidget(np_basic)

            geom_shell = QWidget(self)
            geom_shell_layout = QVBoxLayout(geom_shell)
            geom_shell_layout.setContentsMargins(0, 0, 0, 0)
            geom_shell_layout.setSpacing(8)
            geom_header = QWidget(self)
            geom_header_layout = QHBoxLayout(geom_header)
            geom_header_layout.setContentsMargins(0, 0, 0, 0)
            geom_header_layout.setSpacing(10)
            self.geom_toggle = QToolButton(self)
            self.geom_toggle.setCheckable(True)
            self.geom_toggle.setChecked(True)
            self.geom_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            self.geom_toggle.setArrowType(Qt.DownArrow)
            self.geom_toggle.setText("Geometry Details")
            self.geom_toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.geom_hint = QLabel("ASE options for the selected nanoparticle shape")
            self.geom_hint.setStyleSheet("color: #9aa6bb; font-size: 12px;")
            geom_header_layout.addWidget(self.geom_toggle)
            geom_header_layout.addWidget(self.geom_hint, 1)

            self.geom_content = QWidget(self)
            geom_layout = QVBoxLayout(self.geom_content)
            geom_layout.setContentsMargins(10, 8, 10, 10)
            geom_layout.setSpacing(10)

            wulff = QGroupBox("Wulff")
            self._group_wulff = wulff
            wulff_layout = QFormLayout(wulff)
            wulff_layout.addRow("Surfaces", self.surfaces)
            wulff_layout.addRow("Radial trim", self.radial_trim)
            wulff_layout.addRow(self.verbose)
            geom_layout.addWidget(wulff)

            ico = QGroupBox("Icosahedron")
            self._group_ico = ico
            ico_layout = QFormLayout(ico)
            ico_layout.addRow("Shells", self.ico_shells)
            geom_layout.addWidget(ico)

            cube = QGroupBox("Cube")
            self._group_cube = cube
            cube_layout = QFormLayout(cube)
            cube_layout.addRow("Layers", self.cube_layers)
            geom_layout.addWidget(cube)

            octa = QGroupBox("Octahedron")
            self._group_octa = octa
            octa_layout = QFormLayout(octa)
            octa_layout.addRow("Length", self.octa_length)
            octa_layout.addRow("Cutoff", self.octa_cutoff)
            geom_layout.addWidget(octa)

            deca = QGroupBox("Decahedron")
            self._group_deca = deca
            deca_layout = QGridLayout(deca)
            deca_layout.addWidget(QLabel("p"), 0, 0)
            deca_layout.addWidget(self.p, 0, 1)
            deca_layout.addWidget(QLabel("q"), 0, 2)
            deca_layout.addWidget(self.q, 0, 3)
            deca_layout.addWidget(QLabel("r"), 1, 0)
            deca_layout.addWidget(self.r, 1, 1)
            geom_layout.addWidget(deca)

            morph = QGroupBox("Morphed spherical")
            self._group_morph = morph
            morph_layout = QFormLayout(morph)
            morph_layout.addRow("Morph", self.morph)
            geom_layout.addWidget(morph)

            cuboct = QGroupBox("Cuboctahedron")
            self._group_cuboct = cuboct
            cuboct_layout = QFormLayout(cuboct)
            cuboct_layout.addRow("Layers", self.cuboct_layers)
            geom_layout.addWidget(cuboct)

            geom_shell_layout.addWidget(geom_header)
            geom_shell_layout.addWidget(self.geom_content)
            np_layout.addWidget(geom_shell)
            self.generate_button = QPushButton("Generate")
            np_layout.addWidget(self.generate_button)
            form_layout.addWidget(np_box)

            form_layout.addStretch(1)

            scroll = QScrollArea(self)
            scroll.setWidgetResizable(True)
            scroll.setWidget(form_host)

            self.tabs = QTabWidget(self)
            overview = QWidget(self)
            overview_layout = QVBoxLayout(overview)
            overview_layout.setContentsMargins(12, 12, 12, 12)
            overview_layout.setSpacing(12)
            preview_controls = QWidget(self)
            preview_controls_layout = QHBoxLayout(preview_controls)
            preview_controls_layout.setContentsMargins(0, 0, 0, 0)
            preview_controls_layout.setSpacing(10)
            preview_controls_layout.addWidget(QLabel("Atom radius"))
            slider_block = QWidget(self)
            slider_block_layout = QVBoxLayout(slider_block)
            slider_block_layout.setContentsMargins(0, 0, 0, 0)
            slider_block_layout.setSpacing(2)
            self.atom_radius = QSlider(Qt.Horizontal)
            self.atom_radius.setRange(10, 500)
            self.atom_radius.setSingleStep(1)
            self.atom_radius.setPageStep(10)
            self.atom_radius.setTickPosition(QSlider.TicksBelow)
            self.atom_radius.setTickInterval(25)
            self.atom_radius.setValue(100)
            self.atom_radius.setMinimumWidth(260)
            slider_block_layout.addWidget(self.atom_radius)
            slider_labels = QWidget(self)
            slider_labels_layout = QHBoxLayout(slider_labels)
            slider_labels_layout.setContentsMargins(2, 0, 2, 0)
            slider_labels_layout.setSpacing(0)
            self.atom_radius_min = QLabel("0.10x")
            self.atom_radius_max = QLabel("5.00x")
            self.atom_radius_min.setStyleSheet("color: #9aa6bb; font-size: 11px;")
            self.atom_radius_max.setStyleSheet("color: #9aa6bb; font-size: 11px;")
            slider_labels_layout.addWidget(self.atom_radius_min)
            slider_labels_layout.addStretch(1)
            slider_labels_layout.addWidget(self.atom_radius_max)
            slider_block_layout.addWidget(slider_labels)
            preview_controls_layout.addWidget(slider_block, 1)
            self.atom_radius_value = QLabel("1.00x")
            self.atom_radius_value.setMinimumWidth(48)
            preview_controls_layout.addWidget(self.atom_radius_value)
            preview_controls_layout.addWidget(QLabel("Projection"))
            self.projection_mode = QComboBox()
            self.projection_mode.addItems(["orthographic", "perspective", "oblique"])
            preview_controls_layout.addWidget(self.projection_mode)
            self.reset_preview_button = QPushButton("Reset View")
            preview_controls_layout.addWidget(self.reset_preview_button)
            preview_controls_layout.addStretch(1)
            overview_layout.addWidget(preview_controls)
            overview_layout.addWidget(self._summary)
            overview_layout.addWidget(self._preview, 1)
            self.tabs.addTab(overview, "Overview")

            input_tab = QWidget(self)
            input_layout = QVBoxLayout(input_tab)
            input_layout.setContentsMargins(12, 12, 12, 12)
            input_layout.addWidget(self._input_preview, 1)
            self.tabs.addTab(input_tab, "Input")

            log_tab = QWidget(self)
            log_layout = QVBoxLayout(log_tab)
            log_layout.setContentsMargins(12, 12, 12, 12)
            log_layout.addWidget(self._log_view, 1)
            self.tabs.addTab(log_tab, "Log")

            right_host = QWidget(self)
            right_layout = QVBoxLayout(right_host)
            right_layout.setContentsMargins(0, 0, 0, 0)
            right_layout.addWidget(self.tabs, 1)

            splitter = QSplitter(self)
            splitter.addWidget(scroll)
            splitter.addWidget(right_host)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)

            generator_page = QWidget(self)
            root_layout = QVBoxLayout(generator_page)
            root_layout.setContentsMargins(0, 0, 0, 0)
            root_layout.addWidget(splitter, 1)

            self.site_counter_panel = SiteCounterPanel(parent=self, default_config="npf.count.in", default_structure="POSCAR")
            self.md_panel = ThermalRelaxationPanel(parent=self, default_structure="POSCAR")
            self.crystal_panel = self._build_crystal_tab_placeholder()

            self.main_tabs = QTabWidget(self)
            self.main_tabs.addTab(self.crystal_panel, "Crystal")
            self.main_tabs.addTab(generator_page, "NP Generator")
            self.main_tabs.addTab(self.site_counter_panel, "Site Counter")
            self.main_tabs.addTab(self.md_panel, "MD Relaxation")
            self.setCentralWidget(self.main_tabs)

            toolbar = self.addToolBar("Main")
            toolbar.setMovable(False)
            open_folder_action = QAction("Open Folder", self)
            open_folder_action.triggered.connect(self.open_working_folder)
            toolbar.addAction(open_folder_action)
            reset_action = QAction("Reset Preview", self)
            reset_action.triggered.connect(self._preview.reset_view)
            toolbar.addAction(reset_action)

            self.statusBar().showMessage(f"Base path: {ROOT}")

        def _build_crystal_tab_placeholder(self):
            page = QWidget(self)
            page_layout = QHBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(10)

            left_panel = QWidget(self)
            left_layout = QVBoxLayout(left_panel)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(10)

            crystal_file_box = QGroupBox("Crystal File", left_panel)
            crystal_file_layout = QVBoxLayout(crystal_file_box)
            crystal_file_row = QWidget(crystal_file_box)
            crystal_file_row_layout = QHBoxLayout(crystal_file_row)
            crystal_file_row_layout.setContentsMargins(0, 0, 0, 0)
            self.crystal_file_edit = QLineEdit("")
            self.crystal_file_edit.setReadOnly(True)
            self.crystal_file_browse = QPushButton("Browse")
            crystal_file_row_layout.addWidget(self.crystal_file_edit, 1)
            crystal_file_row_layout.addWidget(self.crystal_file_browse, 0)
            crystal_file_layout.addWidget(crystal_file_row)
            left_layout.addWidget(crystal_file_box)

            crystal_box = QGroupBox("Crystal", left_panel)
            crystal_form = QFormLayout(crystal_box)
            self.crystal_element_edit = QLineEdit("")
            self.crystal_element_edit.setReadOnly(True)
            self.crystal_type_edit = QComboBox()
            self.crystal_type_edit.addItems(
                ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "rhombohedral", "hexagonal", "cubic"]
            )
            self.crystal_type_edit.setEnabled(False)
            self.crystal_subtype_edit = QComboBox()
            self.crystal_a_edit = QDoubleSpinBox()
            self.crystal_b_edit = QDoubleSpinBox()
            self.crystal_c_edit = QDoubleSpinBox()
            self.crystal_alpha_edit = QDoubleSpinBox()
            self.crystal_beta_edit = QDoubleSpinBox()
            self.crystal_gamma_edit = QDoubleSpinBox()
            for spin in [self.crystal_a_edit, self.crystal_b_edit, self.crystal_c_edit]:
                spin.setRange(0.0, 999.0)
                spin.setDecimals(6)
                spin.setReadOnly(True)
            for spin in [self.crystal_alpha_edit, self.crystal_beta_edit, self.crystal_gamma_edit]:
                spin.setRange(0.0, 180.0)
                spin.setDecimals(3)
                spin.setReadOnly(True)
            crystal_form.addRow("Formula", self.crystal_element_edit)
            crystal_form.addRow("Crystal system", self.crystal_type_edit)
            crystal_form.addRow("Crystal subtype", self.crystal_subtype_edit)
            crystal_form.addRow("a", self.crystal_a_edit)
            crystal_form.addRow("b", self.crystal_b_edit)
            crystal_form.addRow("c", self.crystal_c_edit)
            crystal_form.addRow("alpha", self.crystal_alpha_edit)
            crystal_form.addRow("beta", self.crystal_beta_edit)
            crystal_form.addRow("gamma", self.crystal_gamma_edit)
            left_layout.addWidget(crystal_box)

            repeat_box = QGroupBox("Repeat", left_panel)
            repeat_form = QFormLayout(repeat_box)
            self.crystal_repeat_x = QSpinBox()
            self.crystal_repeat_y = QSpinBox()
            self.crystal_repeat_z = QSpinBox()
            for spin in [self.crystal_repeat_x, self.crystal_repeat_y, self.crystal_repeat_z]:
                spin.setRange(1, 12)
                spin.setValue(1)
            repeat_form.addRow("repeat_x", self.crystal_repeat_x)
            repeat_form.addRow("repeat_y", self.crystal_repeat_y)
            repeat_form.addRow("repeat_z", self.crystal_repeat_z)
            left_layout.addWidget(repeat_box)
            left_layout.addStretch(1)

            left_scroll = QScrollArea(self)
            left_scroll.setWidgetResizable(True)
            left_scroll.setWidget(left_panel)

            right_panel = QWidget(self)
            right_layout = QVBoxLayout(right_panel)
            right_layout.setContentsMargins(12, 12, 12, 12)
            right_layout.setSpacing(10)
            self.crystal_tab_status = QLabel("Load a crystal file to define the nanoparticle source basis.")
            self.crystal_tab_status.setWordWrap(True)
            self.crystal_tab_view = ClusterPreviewView(self, empty_message="Load a crystal file to preview.")
            self.crystal_tab_info = QPlainTextEdit(self)
            self.crystal_tab_info.setReadOnly(True)
            self.crystal_tab_info.setPlaceholderText("Crystal metadata")
            right_layout.addWidget(self.crystal_tab_status)
            right_layout.addWidget(self.crystal_tab_view, 1)
            right_layout.addWidget(self.crystal_tab_info, 0)

            page_layout.addWidget(left_scroll, 0)
            page_layout.addWidget(right_panel, 1)
            return page

        def _wire_signals(self):
            self.crystal_file_browse.clicked.connect(self.pick_crystal_file)
            self.crystal_repeat_x.valueChanged.connect(self._render_loaded_crystal_preview)
            self.crystal_repeat_y.valueChanged.connect(self._render_loaded_crystal_preview)
            self.crystal_repeat_z.valueChanged.connect(self._render_loaded_crystal_preview)
            self.crystal_type_edit.currentTextChanged.connect(self._on_crystal_type_changed)
            self.crystal_subtype_edit.currentTextChanged.connect(self._sync_visibility)
            self.shape.currentTextChanged.connect(self._sync_visibility)
            self.size_mode.currentTextChanged.connect(self._sync_visibility)
            self.geom_toggle.toggled.connect(self._sync_geom_visibility)
            self.atom_radius.valueChanged.connect(self._sync_atom_radius)
            self.projection_mode.currentTextChanged.connect(lambda v: self._preview.set_projection(v))
            self.reset_preview_button.clicked.connect(self._reset_preview_controls)
            self.generate_button.clicked.connect(self.generate_cluster)
            self.write_button.clicked.connect(self.write_input_only)
            self.load_button.clicked.connect(self.load_input_dialog)
            self.open_folder_button.clicked.connect(self.open_working_folder)

            for widget in [
                self.shape,
                self.size_mode,
                self.atom_radius,
                self.atom_radius_min,
                self.atom_radius_max,
                self.atom_radius_value,
                self.projection_mode,
                self.size_value_atoms,
                self.size_value_diameter,
                self.input_file_name,
                self.output_file_name,
                self.working_path,
                self.surfaces,
                self.facet_mode,
                self.extra_facets,
                self.sphericity,
                self.radius_tol,
                self.max_refine,
                self.radius_policy,
                self.radial_trim,
                self.ico_shells,
                self.cube_layers,
                self.octa_length,
                self.octa_cutoff,
                self.p,
                self.q,
                self.r,
                self.morph,
                self.cuboct_layers,
                self.verbose,
            ]:
                if hasattr(widget, "textChanged"):
                    widget.textChanged.connect(self.update_preview)
                elif hasattr(widget, "valueChanged"):
                    widget.valueChanged.connect(self.update_preview)
                elif hasattr(widget, "currentTextChanged"):
                    widget.currentTextChanged.connect(self.update_preview)
                elif hasattr(widget, "currentIndexChanged"):
                    widget.currentIndexChanged.connect(self.update_preview)
                elif hasattr(widget, "stateChanged"):
                    widget.stateChanged.connect(self.update_preview)

            self.size_value_diameter.valueChanged.connect(self.update_preview)
            self._sync_atom_radius(self.atom_radius.value())
            self._sync_geom_visibility(self.geom_toggle.isChecked())
            self._sync_crystal_subtype_options()

        def _crystal_tab_structure(self) -> str:
            if self._crystal_source is not None:
                return self._crystal_source.crystal_system
            return normalize_structure_class(self.crystal_type_edit.currentText() or "hexagonal")

        def _sync_crystal_subtype_options(self):
            structure = self._crystal_tab_structure()
            current = self.crystal_subtype_edit.currentData() or self.crystal_subtype_edit.currentText()
            if self._crystal_source is not None and self._crystal_source.subtype:
                current = self._crystal_source.subtype
            items = [("Auto / loaded basis", "")] + self._structure_variants(structure)
            self._set_combo_items(self.crystal_subtype_edit, items, current)

        def _on_crystal_type_changed(self, _text: str):
            self._sync_crystal_subtype_options()
            self._sync_visibility()

        def pick_crystal_file(self):
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Open crystal file",
                str(Path.cwd()),
                "Crystal files (*.cif *POSCAR* *CONTCAR*);;All files (*)",
            )
            if filename:
                self.crystal_file_edit.setText(filename)
                self.load_crystal_file(filename)

        def load_crystal_file(self, path_text: str):
            path = Path(path_text).expanduser().resolve()
            if not path.exists():
                QMessageBox.critical(self, "Crystal load error", f"Crystal file does not exist: {path}")
                return
            try:
                source = load_crystal(path)
            except Exception as exc:
                QMessageBox.critical(self, "Crystal load error", f"Could not read crystal file:\n{path}\n\n{exc}")
                return

            try:
                self._crystal_source = source
                atoms = source.atoms
                cellpar = atoms.cell.cellpar()
                if source.species != ("Ru",):
                    self.surfaces.clear()
                self.crystal_element_edit.setText(source.formula)
                self.crystal_type_edit.setCurrentText(source.crystal_system)
                self._sync_crystal_subtype_options()
                self.crystal_a_edit.setValue(float(cellpar[0]))
                self.crystal_b_edit.setValue(float(cellpar[1]))
                self.crystal_c_edit.setValue(float(cellpar[2]))
                self.crystal_alpha_edit.setValue(float(cellpar[3]))
                self.crystal_beta_edit.setValue(float(cellpar[4]))
                self.crystal_gamma_edit.setValue(float(cellpar[5]))
                self._render_loaded_crystal_preview(path=path)
                self.crystal_tab_status.setText(f"Loaded crystal: {path.name} ({len(atoms)} atoms)")
                self.crystal_tab_info.setPlainText(
                    f"Loaded file: {path}\n"
                    f"Formula: {source.formula or 'N/A'}\n"
                    f"Species: {', '.join(source.species)}\n"
                    f"Atoms: {len(atoms)}\n"
                    f"Crystal system: {source.crystal_system}\n"
                    f"Space group: {source.spacegroup_symbol or 'unknown'} ({source.spacegroup_number or 'unknown'})\n"
                    f"Suggested subtype: {source.subtype or 'not determined'}\n"
                    f"Cell: a={cellpar[0]:.6f}, b={cellpar[1]:.6f}, c={cellpar[2]:.6f}\n"
                    f"Angles: alpha={cellpar[3]:.3f}, beta={cellpar[4]:.3f}, gamma={cellpar[5]:.3f}"
                )
                self._sync_visibility()
            except Exception as exc:
                QMessageBox.critical(self, "Crystal load error", f"Loaded but failed to apply to UI:\n{exc}")

        def _render_loaded_crystal_preview(self, _value=None, path: Path | None = None):
            if self._crystal_source is None:
                return
            rx = int(self.crystal_repeat_x.value())
            ry = int(self.crystal_repeat_y.value())
            rz = int(self.crystal_repeat_z.value())
            atoms = self._crystal_source.atoms
            repeated = atoms.repeat((rx, ry, rz))
            preview = build_cluster_preview_from_atoms(repeated, path=str(path) if path else "")
            preview["cell_edges"] = _build_cell_edges(atoms, (rx, ry, rz))
            preview["radii"] = [float(covalent_radii[int(z)]) for z in repeated.numbers]
            self.crystal_tab_view.set_cluster(preview)

        def _sync_atom_radius(self, value: int):
            radius_scale = max(0.1, min(5.0, value / 100.0))
            self.atom_radius_value.setText(f"{radius_scale:.2f}x")
            QToolTip.showText(QCursor.pos(), f"Atom radius {radius_scale:.2f}x", self.atom_radius)
            self._preview.set_atom_scale(radius_scale)

        def _reset_preview_controls(self):
            self.atom_radius.setValue(100)
            self.projection_mode.setCurrentText("orthographic")
            self._preview.reset_view()

        def _sync_geom_visibility(self, checked: bool):
            self.geom_content.setVisible(bool(checked))
            self.geom_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

        def _structure_variants(self, structure: str):
            return {
                "cubic": [
                    ("simple cubic / sc", "simple_cubic"),
                    ("bcc", "bcc"),
                    ("fcc", "fcc"),
                    ("diamond cubic", "diamond_cubic"),
                    ("zincblende", "zincblende"),
                    ("rocksalt / NaCl", "rocksalt"),
                    ("CsCl", "cscl"),
                    ("fluorite / CaF2", "fluorite"),
                    ("perovskite / ABO3", "perovskite"),
                    ("spinel", "spinel"),
                ],
                "hexagonal": [
                    ("hcp", "hcp"),
                    ("wurtzite", "wurtzite"),
                    ("graphite / layered hexagonal", "graphite"),
                    ("AlB2-type", "alb2"),
                    ("NiAs-type", "nias"),
                ],
                "tetragonal": [
                    ("simple tetragonal", "simple_tetragonal"),
                    ("body-centered tetragonal / bct", "bct"),
                    ("rutile / TiO2", "rutile"),
                    ("anatase / TiO2", "anatase"),
                    ("zircon / ZrSiO4", "zircon"),
                ],
                "orthorhombic": [
                    ("simple orthorhombic", "simple_orthorhombic"),
                    ("base-centered orthorhombic", "base_centered_orthorhombic"),
                    ("body-centered orthorhombic", "body_centered_orthorhombic"),
                    ("face-centered orthorhombic", "face_centered_orthorhombic"),
                    ("olivine-type", "olivine"),
                    ("distorted perovskite", "perovskite_distorted_orthorhombic"),
                ],
                "rhombohedral": [
                    ("rhombohedral", "rhombohedral"),
                    ("corundum / Al2O3", "corundum"),
                    ("calcite / CaCO3", "calcite"),
                    ("ilmenite", "ilmenite"),
                    ("Bi/Sb-type A7", "a7"),
                ],
                "monoclinic": [
                    ("simple monoclinic", "simple_monoclinic"),
                    ("base-centered monoclinic", "base_centered_monoclinic"),
                    ("baddeleyite / ZrO2", "baddeleyite"),
                    ("general monoclinic", "general_monoclinic"),
                ],
                "triclinic": [
                    ("simple triclinic", "simple_triclinic"),
                    ("molecular triclinic", "molecular_triclinic"),
                    ("framework triclinic", "framework_triclinic"),
                ],
            }.get(structure, [])

        def _set_combo_items(self, combo: QComboBox, items: list[tuple[str, str]], selected: str | None = None):
            combo.blockSignals(True)
            combo.clear()
            if not items:
                combo.addItem("No subtypes available", "")
                combo.setEnabled(False)
                combo.blockSignals(False)
                return
            combo.setEnabled(True)
            selected_value = selected if selected and any(value == selected for _, value in items) else items[0][1]
            for label, value in items:
                combo.addItem(label, value)
            index = combo.findData(selected_value)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

        def _sync_size_mode(self):
            mode = self.size_mode.currentText()
            is_atoms = mode == "atoms"
            self.size_value_label.setText("Number of atoms" if is_atoms else "Diameter (nm)")
            self.size_value_atoms.setVisible(is_atoms)
            self.size_value_diameter.setVisible(not is_atoms)

        def _sync_visibility(self):
            current_shape = self.shape.currentData() or self.shape.currentText()
            self._set_combo_items(self.shape, _CRYSTAL_DRIVEN_SHAPES, current_shape)
            shape = self.shape.currentData() or self.shape.currentText()
            self.geom_hint.setText(f"Loaded-crystal carving options: {shape_label(shape)}")

            # Shape-specific controls
            self._group_wulff.setVisible(shape == "wulff")
            self._group_ico.setVisible(False)
            self._group_cube.setVisible(False)
            self._group_octa.setVisible(False)
            self._group_deca.setVisible(False)
            self._group_morph.setVisible(shape == "morphed_spherical")
            self._group_cuboct.setVisible(False)

            self._sync_size_mode()
            self.update_preview()

        def _collect_values(self) -> dict[str, Any]:
            if self._crystal_source is None:
                raise ValueError("Load crystal data in the Crystal tab first.")
            cellpar = self._crystal_source.atoms.cell.cellpar()

            values: dict[str, Any] = {
                "element": self._crystal_source.formula,
                "crystal_file": str(self._crystal_source.path),
                "shape": normalize_shape_key(self.shape.currentData() or self.shape.currentText()),
                "crystal_structure": self._crystal_source.crystal_system,
                "crystal_variant": self.crystal_subtype_edit.currentData() or self.crystal_subtype_edit.currentText(),
                "size_mode": self.size_mode.currentText(),
                "atoms": int(self.size_value_atoms.text()) if self.size_value_atoms.text().strip() else None,
                "diameter": float(self.size_value_diameter.value()) * _NM_TO_ANGSTROM if self.size_mode.currentText() == "diameter" else None,
                "a": float(cellpar[0]),
                "b": float(cellpar[1]),
                "c": float(cellpar[2]),
                "alpha": float(cellpar[3]),
                "beta": float(cellpar[4]),
                "gamma": float(cellpar[5]),
                "input_file_name": self.input_file_name.text().strip() or "npf_np_gen.in",
                "output_file_name": self.output_file_name.text().strip() or "",
                "working_path": self.working_path.text().strip() or str(ROOT),
                "verbose": self.verbose.isChecked(),
                "surfaces": self.surfaces.toPlainText().strip(),
                "facet_mode": self.facet_mode.currentText(),
                "extra_facets": self.extra_facets.currentText(),
                "sphericity": float(self.sphericity.value()),
                "radius_tol": float(self.radius_tol.value()),
                "max_refine": int(self.max_refine.currentText()),
                "radius_policy": self.radius_policy.currentText(),
                "radial_trim": float(self.radial_trim.text()) if self.radial_trim.text().strip() else None,
                "ico_shells": int(self.ico_shells.text()) if self.ico_shells.text().strip() else None,
                "cube_layers": int(self.cube_layers.text()) if self.cube_layers.text().strip() else None,
                "octa_length": int(self.octa_length.text()) if self.octa_length.text().strip() else None,
                "octa_cutoff": int(self.octa_cutoff.text()) if self.octa_cutoff.text().strip() else 0,
                "p": int(self.p.text()) if self.p.text().strip() else None,
                "q": int(self.q.text()) if self.q.text().strip() else None,
                "r": int(self.r.text()) if self.r.text().strip() else None,
                "morph": float(self.morph.value()),
                "cuboct_layers": int(self.cuboct_layers.text()) if self.cuboct_layers.text().strip() else None,
            }
            return values

        def build_spec(self) -> dict[str, Any]:
            values = self._collect_values()
            spec = _build_spec_from_ui(values)
            if spec["size_mode"] == "diameter":
                spec["radius"] = float(self.size_value_diameter.value()) * _NM_TO_ANGSTROM / 2.0
                spec["atoms"] = None
            else:
                spec["atoms"] = int(self.size_value_atoms.text()) if self.size_value_atoms.text().strip() else None
                spec["radius"] = None

            if not spec["output_file_name"]:
                spec["output_file_name"] = _shape_default_output(
                    spec["element"],
                    spec["shape"],
                    spec["size_mode"],
                    spec["atoms"],
                    float(self.size_value_diameter.value()) if spec["size_mode"] == "diameter" else None,
                )
                spec["tag"] = Path(spec["output_file_name"]).stem
            return spec

        def spec_to_text(self, spec: dict[str, Any]) -> str:
            return spec_to_input_text(spec)

        def update_preview(self):
            try:
                spec = self.build_spec()
            except Exception:
                return
            self._last_spec = spec
            self._input_preview.setPlainText(self.spec_to_text(spec))

        def apply_spec(self, spec: dict[str, Any]):
            shape = normalize_shape_key(spec.get("shape", "sphere"))
            self._set_combo_items(self.shape, _CRYSTAL_DRIVEN_SHAPES, shape)

            if spec.get("atoms") not in {None, ""}:
                self.size_mode.setCurrentText("atoms")
                self.size_value_atoms.setText(str(spec.get("atoms")))
            elif spec.get("radius") not in {None, ""}:
                self.size_mode.setCurrentText("diameter")
                self.size_value_diameter.setValue(float(spec.get("radius")) * 2.0 / _NM_TO_ANGSTROM)
            elif spec.get("diameter") not in {None, ""}:
                self.size_mode.setCurrentText("diameter")
                self.size_value_diameter.setValue(float(spec.get("diameter")) / _NM_TO_ANGSTROM)
            else:
                self.size_mode.setCurrentText(spec.get("size_mode", "diameter"))

            self.input_file_name.setText(str(spec.get("input_file_name", "npf_np_gen.in")))
            self.output_file_name.setText(str(spec.get("output_file_name", "")) if spec.get("output_file_name") else "")
            self.working_path.setText(str(spec.get("working_path", ROOT)))
            self.verbose.setChecked(bool(spec.get("verbose", False)))
            self.surfaces.setPlainText(str(spec.get("surfaces", "")))
            self.facet_mode.setCurrentText(str(spec.get("facet_mode", "top")))
            self.extra_facets.setCurrentText(str(spec.get("extra_facets", "none")))
            self.sphericity.setValue(float(spec.get("sphericity", 0.0) or 0.0))
            self.radius_tol.setValue(float(spec.get("radius_tol", 0.3) or 0.3))
            self.max_refine.setCurrentText(str(spec.get("max_refine", 12)))
            self.radius_policy.setCurrentText(str(spec.get("radius_policy", "closest")))
            self.radial_trim.setText("" if spec.get("radial_trim") in {None, ""} else str(spec.get("radial_trim")))
            self.ico_shells.setText("" if spec.get("ico_shells") in {None, ""} else str(spec.get("ico_shells")))
            self.cube_layers.setText("" if spec.get("cube_layers") in {None, ""} else str(spec.get("cube_layers")))
            self.octa_length.setText("" if spec.get("octa_length") in {None, ""} else str(spec.get("octa_length")))
            self.octa_cutoff.setText(str(spec.get("octa_cutoff", 0) or 0))
            self.p.setText("" if spec.get("p") in {None, ""} else str(spec.get("p")))
            self.q.setText("" if spec.get("q") in {None, ""} else str(spec.get("q")))
            self.r.setText("" if spec.get("r") in {None, ""} else str(spec.get("r")))
            self.morph.setValue(float(spec.get("morph", 0.0) or 0.0))
            self.cuboct_layers.setText("" if spec.get("cuboct_layers") in {None, ""} else str(spec.get("cuboct_layers")))
            self._sync_visibility()
            self.update_preview()

        def load_input_dialog(self):
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open npf_np_gen.in",
                str(Path(self.working_path.text().strip() or ROOT)),
                "Input files (*.in *.txt *.json);;All files (*.*)",
            )
            if not path:
                return
            try:
                spec = load_input_spec(path)
            except Exception as exc:
                QMessageBox.critical(self, "Load failed", str(exc))
                return
            self.apply_spec(spec)
            self.statusBar().showMessage(f"Loaded {path}", 4000)

        def open_working_folder(self):
            base_dir = Path(self.working_path.text().strip() or ROOT).expanduser()
            tag = None
            if self._last_spec:
                tag = self._last_spec.get("tag") or Path(self._last_spec.get("output_file_name") or "").stem or None
            folder = cluster_output_dir(base_dir, tag)
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl

            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

        def _run_generation(self, write_only: bool):
            try:
                spec = self.build_spec()
            except Exception as exc:
                QMessageBox.warning(self, "Invalid input", str(exc))
                return

            working_path = str(Path(self.working_path.text().strip() or ROOT).expanduser())
            output_dir = cluster_output_dir(working_path, spec["tag"])
            if write_only:
                try:
                    input_path = output_dir / spec.get("input_file_name", "npf_np_gen.in")
                    input_path.write_text(self.spec_to_text(spec), encoding="utf-8")
                except Exception as exc:
                    QMessageBox.critical(self, "Write failed", str(exc))
                    return
                self._log_view.setPlainText(f"Wrote input file:\n{input_path}")
                self.statusBar().showMessage(f"Wrote {input_path}", 5000)
                return

            self._input_preview.setPlainText(self.spec_to_text(spec))
            self._log_view.setPlainText("Generating cluster...\n")
            self.statusBar().showMessage("Generating...", 0)
            self.setEnabled(False)
            self._worker = GenerationWorker(spec, working_path)
            self._worker.finished_ok.connect(self._generation_finished)
            self._worker.failed.connect(self._generation_failed)
            self._worker.finished.connect(self._generation_cleanup)
            self._worker.start()

        def generate_cluster(self):
            self._run_generation(write_only=False)

        def write_input_only(self):
            self._run_generation(write_only=True)

        def _generation_finished(self, result, preview, log_text: str, spec: dict[str, Any]):
            tag = spec.get("tag") or Path(spec.get("output_file_name") or "Ru_NP").stem
            working_path = Path(self.working_path.text().strip() or ROOT).expanduser().resolve()
            output_dir = cluster_output_dir(working_path, tag)
            input_path = output_dir / str(spec.get("input_file_name", "npf_np_gen.in"))
            input_write_error = None
            try:
                input_path.write_text(self.spec_to_text(spec), encoding="utf-8")
            except Exception as exc:
                input_write_error = str(exc)
            meta_suffix = {
                "wulff": "wulff",
                "ico": "ico",
                "sphere": "sphere",
                "cube": "cube",
                "octahedron": "octahedron",
                "decahedron": "decahedron",
                "dodecahedron": "dodecahedron",
                "morphed_spherical": "morphed_spherical",
                "cuboct": "cuboct",
                "hexagonal_prism": "hexagonal_prism",
                "truncated_hexagonal_prism": "truncated_hexagonal_prism",
                "hexagonal_bipyramid": "hexagonal_bipyramid",
                "truncated_hexagonal_bipyramid": "truncated_hexagonal_bipyramid",
                "nanorod": "nanorod",
                "hexagonal_platelet": "hexagonal_platelet",
            }.get(spec.get("shape", "wulff"), "wulff")
            xyz_path = output_dir / f"{tag}.xyz"
            poscar_path = output_dir / "POSCAR"
            meta_path = output_dir / f"{tag}_{meta_suffix}_meta.json"
            file_lines = [str(path) for path in [input_path, xyz_path, poscar_path, meta_path] if path.exists()]
            summary = [
                f"Shape: {spec.get('shape')}",
                f"Formula: {getattr(result, 'formula', '')}",
                f"Atoms: {getattr(result, 'atoms_count', '')}",
                f"Radius: {getattr(result, 'actual_radius', 0.0):.3f} Å",
                f"Output: {spec.get('output_file_name')}",
                f"Output folder: {output_dir}",
            ]
            if input_write_error:
                summary.append("")
                summary.append(f"Input write failed: {input_write_error}")
            if file_lines:
                summary.append("")
                summary.append("Generated files:")
                summary.extend(f"  {line}" for line in file_lines)
            self._summary.setText("\n".join(summary))
            self._current_preview = preview
            self._preview.set_cluster(preview)
            generated_geometry = None
            for candidate in [output_dir / "POSCAR", output_dir / f"{tag}.xyz"]:
                if candidate.exists():
                    generated_geometry = candidate
                    break
            if getattr(self, "site_counter_panel", None) is not None and generated_geometry is not None:
                self.site_counter_panel.set_structure_path(str(generated_geometry))
            if getattr(self, "md_panel", None) is not None and generated_geometry is not None:
                self.md_panel.set_structure_path(str(generated_geometry))
                self.md_panel.set_working_folder(str(output_dir), explicit=True, auto_load=False)
            self._log_view.setPlainText(log_text or "Build completed.")
            self.statusBar().showMessage(f"Generated {getattr(result, 'formula', '')} with {getattr(result, 'atoms_count', '')} atoms", 5000)

        def _generation_failed(self, message: str):
            self._log_view.setPlainText(message)
            QMessageBox.critical(self, "Generation failed", message)
            self.statusBar().showMessage("Generation failed", 5000)

        def _generation_cleanup(self):
            self.setEnabled(True)
            self._worker = None

    app = QApplication.instance() or QApplication([])
    window = NPFNativeWindow()
    window.show()
    app.exec()
