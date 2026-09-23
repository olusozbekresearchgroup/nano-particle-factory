# Nano Particle Factory (npf)

`npf` (Nano Particle Factory) is a crystal-driven nanoparticle generator and
surface-site analysis toolkit. It loads a periodic crystal structure and carves
nanoparticles directly from that atomic basis, preserving all species in
single- and multi-component crystals.

This README is also the installation and user manual. The project supports
Linux and Windows through Python 3.10 or newer.

NPF is released under the [Apache License 2.0](LICENSE). Please use
[CITATION.cff](CITATION.cff) when citing the software in research or other
publications.

## Installation

Install Python 3.10+ first. On Windows, enable the Python installer option
`Add python.exe to PATH`. On Linux, use the distribution Python package or a
Python installation from python.org/conda.

Automatic installers are included. On Linux/macOS run:

```bash
./install_npf.sh
```

On Windows PowerShell run:

```powershell
.\install_npf.ps1
```

You can also double-click `install_npf.bat`. These installers create `.venv`,
upgrade pip/build tools, install NPF and its runtime dependencies, and verify
the main scientific and GUI imports. They require internet access unless all
packages are already available in the local pip cache.

For development dependencies, use `./install_npf.sh` followed by
`python -m pip install -e ".[development]"` on Linux/macOS, or
`.\install_npf.ps1 -Development` on Windows.

Create an isolated environment in the project directory:

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Windows Command Prompt uses the same environment with
`.venv\Scripts\activate.bat`. If PowerShell blocks activation, either run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once or invoke
`.venv\Scripts\python.exe` directly.

The editable install provides the commands `npf`, `npf-gui`, `npf-generate`,
`npf-count`, and `npf-relax`. A dependency-only installation is also available:

```bash
python -m pip install -r requirements.txt
```

For tests and development:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The MD Relaxation tab additionally requires a working LAMMPS installation and
the appropriate potential files. LAMMPS is not required for generation or site
counting.

## Quick Start

Generate a nanoparticle from an input file:

```bash
npf-generate np_gen.in
```

Start the GUI:

```bash
npf-gui
```

Count coordination and surface sites:

```bash
npf-count POSCAR --config npf.count.in
```

Run the relaxation backend:

```bash
npf-relax POSCAR
```

The repository includes starter files under `examples/`. For example:

```bash
cp examples/np_gen.in ./np_gen.in
cp examples/npf.count.in ./npf.count.in
npf-generate np_gen.in
npf-count POSCAR --config npf.count.in
```

On Windows PowerShell, use `Copy-Item` instead of `cp`.

The repository launchers remain available on Linux:

```bash
python3 bin/npf.generate.py np_gen.in
python3 bin/npf.site_count.py POSCAR --config npf.count.in
```

On Windows, use the included `run_npf_*.bat` files or the module commands:

```powershell
run_npf_gui.bat
run_npf_generate.bat np_gen.in
run_npf_count.bat POSCAR --config npf.count.in
```

For batch counting on Windows PowerShell:

```powershell
.\batch_count_sites.ps1 C:\path\to\nano_part
```

The GUI launcher supports `--backend qt`, `--backend web`, and `--backend auto`.
In `auto` mode it prefers a native PySide6 window when `PySide6` is installed, and falls back to the existing browser-based UI otherwise.
Both front ends write `np_gen.in`, run the same backend, and preview the generated cluster.
The web UI also supports loading an existing `np_gen.in` file, previewing the exact written input before generation, and rendering bonds in the embedded view.
For the embedded Qt browser view, `PySide6.QtWebEngineWidgets` is recommended.

The native Qt GUI has four tabs:

- `Crystal`
- `NP Generator`
- `Site Counter`
- `MD Relaxation`

The `Crystal` tab is the canonical source of composition, lattice, crystal system, and atomic basis for the NP Generator. Symmetry is detected with `spglib` when available. Generated input files record the absolute `crystal_file` path for reproducibility. Legacy input files without `crystal_file` continue to use the older Ru-specific builders.

Crystal-driven builds generate periodic lattice sites directly inside shape-aware Cartesian bounds instead of constructing a full rectangular ASE supercell. Atom-count builds use a density estimate with adaptive expansion, and small candidate sets are cached for repeated builds from the same crystal.

The MD tab performs a thermal anneal, then quenching, then a final minimization through LAMMPS.
It expects a working LAMMPS executable and a valid pair style / pair coefficient setup for the chosen material.

The CN tool keeps its own modular namespace under `modules/npf/count` and reads
`npf.count.in` by default, while still accepting the older `config.cn` path for
compatibility.
CN and active-site analysis reuse one neighbor graph for cutoff detection, coordination numbers, nearest-neighbor statistics, and surface topology. Surface motifs are enumerated from local graph connectivity rather than all surface-atom combinations.

For nonperiodic particles with at least 10,000 atoms, the graph uses an exact
SciPy `cKDTree` search in chunks of 2,048 atoms. Only unique `i < j` pairs are
stored, with the same strict distance cutoff as ASE. Temporary neighbor lists
are bounded by the chunk size; the final graph still scales with the number
of pairs. SciPy is required for this large-particle path. Small structures and
all periodic structures retain the ASE implementation. This changes the search
implementation, not the CN cutoffs or site definitions. Periodic counting is
not enabled by this change: nanoparticle CLI/GUI analysis still sets PBC off.
The GUI preview's separate bond-building/rendering path is not changed.

### Batch Site Counting

Keep `batch_count_sites.sh` in this NPF installation, and pass the directory
containing the shape/size subdirectories. Only original files named `POSCAR`
are counted, not duplicate or marked XYZs:

```bash
/path/to/npf_v1/batch_count_sites.sh ~/work_vasp/Ni/nano_part
```

Each particle gets `site_count_before_relax.csv` and `site_count_batch.log`.
Existing reports/logs for processed particles are replaced; structures are not
modified. A uniquely named status TSV is written in the root directory. The
batch is sequential, continues after individual failures, and returns nonzero
if any failed. To rerun only failures without touching successful results:

```bash
cd ~/work_vasp/Ni/nano_part
/path/to/npf_v1/batch_count_sites.sh --retry-failed site_count_batch_PREVIOUS.tsv .
```

Add `--dry-run` before the root argument to list selected inputs without
counting. Counter options such as `--mode fixed --cutoff 2.83` go after the
root argument. Retry uses the current configuration; supply the same options
as the original run when reproducing its settings. Set `PYTHON` if a specific
Python environment is needed.

The input file uses a plain `key = value` format, not JSON.

## Supported Shapes

- `wulff`
- `ico`
- `sphere`
- `cube`
- `octahedron`
- `decahedron`
- `dodecahedron`
- `morphed_spherical`
- `cuboct`
- `hexagonal_prism`
- `truncated_hexagonal_prism`
- `hexagonal_bipyramid`
- `truncated_hexagonal_bipyramid`
- `nanorod`
- `hexagonal_platelet`

## Input Schema

| Key | Type | Default | Applies To | Meaning |
|---|---:|---:|---|---|
| `crystal_file` | path | none | crystal-driven builds | Periodic source structure. Its complete atomic basis and species are preserved. |
| `shape` | string | `wulff` | all | Geometry to build. |
| `tag` | string | `Ru_NP` | all | Output file prefix. |
| `element` | string | source formula | all | Composition label used in the default output name. Legacy builds default to Ru. |
| `output_file_name` | string | derived | all | Default output filename. If omitted, the code uses `element_shape_atomcount.xyz`. |
| `atoms` | int | none | all | Target atom count for approximate size matching. |
| `radius` | float | none | all | Target circumscribed radius in Angstrom. |
| `a` | float | `2.706` | all | Lattice reference length in Angstrom. Used as `hcp a` for Wulff and as the reference length for other shapes. |
| `c` | float | `4.282` | `wulff` | HCP `c` lattice parameter in Angstrom. |
| `surfaces` | string | none | `wulff` | Surface-energy overrides in `hkil=gamma,...` form. |
| `facet_mode` | string | `top` | `wulff` | Facet set selection: `minimal`, `top`, `full`. |
| `extra_facets` | string | `none` | `wulff` | Add heuristic extra facets: `none`, `auto`. |
| `sphericity` | float | `0.0` | `wulff` | Mix toward isotropic surface energy in `[0, 1]`. |
| `radius_tol` | float | `0.3` | `wulff` | Radius tolerance for refinement in Angstrom. |
| `max_refine` | int | `12` | `wulff` | Maximum refinement iterations. |
| `radius_policy` | string | `closest` | `wulff` | Fallback policy: `closest`, `at_least`, `at_most`. |
| `radial_trim` | float or `null` | `null` | `wulff` | Remove atoms beyond this radius after build. |
| `ico_shells` | int | none | `ico` | Explicit Mackay shell count. |
| `cube_layers` | int | none | `cube` | Explicit cubic layer count. |
| `octa_length` | int | none | `octahedron` | Explicit octahedron length. |
| `octa_cutoff` | int | `0` | `octahedron` | Octahedron cutoff parameter. |
| `p` | int | none | `decahedron` | Decahedron `p` parameter. Must be greater than 0. |
| `q` | int | none | `decahedron` | Decahedron `q` parameter. Must be greater than 0. |
| `r` | int | none | `decahedron` | Decahedron `r` parameter. Must be greater than or equal to 0. |
| `morph` | float | `0.0` | `morphed_spherical` | Shape factor in `[0, 1]`. Larger values push the cluster toward a more faceted superellipsoid. |
| `cuboct_layers` | int | none | `cuboct` | Explicit cuboctahedron layer count. |
| `verbose` | bool | `false` | all | Print builder selection details. |

## Example

```text
shape = wulff
tag = Ru_demo
atoms = 147
a = 2.706
c = 4.282
surfaces = 0001=0.162,10-10=0.181,10-11=0.180
facet_mode = top
extra_facets = none
sphericity = 0.0
radius_tol = 0.3
max_refine = 12
radius_policy = closest
radial_trim = null
verbose = true
```

## Notes

### Terrace Interior Counts

Site counting additionally reports `hollow3`, `hollow4_square`, and
`hollow4_rect` terrace interiors without changing the original inventories:

- `<type>_interior_atoms`: unique atoms excluding each terrace patch's boundary.
- `<type>_interior_sites`: hollows whose vertices are all interior to the same patch.

Patches are edge-connected motifs within 10 degrees and 0.10 nearest-neighbor
distances of a seed plane. Patch perimeter, hole, and nonmanifold-edge atoms
are excluded. These are geometric motif types, not crystallographic Miller-index
assignments. Strongly curved or reconstructed surfaces may split into patches.
This additional classification inherits the existing hollow candidates; it does
not redefine them or add a new threefold exposure test.

Both counts appear in CLI/GUI reports and CSV files (with zero-based membership
indices). GUI Surface Marking also exports interior atoms separately, marked Cu.
Counts are totals over the particle, deduplicated within each terrace type.

### Generation Notes

- `dodecahedron` is implemented as a practical FCC rhombic-dodecahedron-like approximation using a Wulff construction biased toward `{110}`.
- `morphed_spherical` is implemented as an FCC superellipsoid cut.
- `sphere` is a spherical cut built by shaving a sphere from a seed cluster that follows the selected structure subtype when ASE supports it, with a cubic fallback for unsupported subtypes. It is available for every crystal system in the GUI.
- The `cube`, `octahedron`, `decahedron`, `dodecahedron`, `cuboct`, and `morphed_spherical` presets are cubic/FCC-style shapes. Use `wulff` when you want hcp Ru facets such as `0001`, `10-10`, and `10-11`.
- The package writes `.xyz` by default, plus `POSCAR` and metadata JSON files for each build.
- The GUI is a starter front end and can be extended without changing the generator core.

## Project Layout

- `modules/npf`: nanoparticle builders, crystal loading, GUI, and relaxation backend.
- `modules/npf/count`: coordination-number and active-site analysis.
- `bin`: Linux-compatible source-tree launchers.
- `examples`: sample crystal, generator, and counting inputs.
- `tests`: regression tests when included in a development checkout.
- `pyproject.toml`: install metadata and command-line entry points.

## Troubleshooting

If `npf` or `npf-gui` is not found after installation, activate the virtual
environment again and use `python -m pip show nano-particle-factory` to confirm
that the package is installed in that environment. The module equivalents are
always available:

```bash
python -m npf.gui
python -m npf.cli examples/np_gen.in
python -m npf.count POSCAR --config examples/npf.count.in
```

If the native Qt GUI cannot start, try `npf-gui --backend web`. The web backend
uses the standard library server and does not require Qt WebEngine. For very
large nonperiodic particles, SciPy is required because the counter switches to
its memory-efficient spatial-tree neighbor search automatically.

## Development

Install the development dependencies and run the tests from the repository
root:

```bash
python -m pip install -e ".[development]"
python -m pytest tests
```

The optional `[md]` extra adds the Python LAMMPS package, but a system LAMMPS
executable and compatible potential files may still be required by the chosen
relaxation workflow.
