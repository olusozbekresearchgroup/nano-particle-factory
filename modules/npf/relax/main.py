from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from ase.io import read, write

from ..common import _average_nn_distance, _ensure_cell
from ..count.cn_count import compute_cn
from ..count.main import write_single_csv
from ..count.site_count import detect_sites_by_cn


@dataclass
class ThermalRelaxationResult:
    input_file: str
    relaxed_xyz: str
    relaxed_poscar: str
    relaxed_data: str
    trajectory_file: str
    workdir: str
    site_count_before_csv: str
    lammps_command: str
    pair_style: str
    pair_coeff: str
    temperature: float
    thermal_steps: int
    quench_temperature: float
    quench_steps: int
    timestep: float
    damping: float
    seed: int
    initial_atoms: int
    final_atoms: int
    surface_atoms: int
    mobile_atoms: int
    initial_average_nn_distance: float
    final_average_nn_distance: float
    initial_potential_energy: float | None
    final_potential_energy: float | None
    thermo_history: dict[str, list[float]]
    log_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThermalRelaxationInputs:
    structure_path: str
    workdir: str
    inputs_dir: str
    data_path: str
    script_path: str
    site_count_before_csv: str
    lammps_command: str
    pair_style: str
    pair_coeff: str
    potential_file: str
    copied_potential_file: str
    temperature: float
    thermal_steps: int
    quench_temperature: float
    quench_steps: int
    timestep: float
    damping: float
    seed: int
    surface_atoms: int
    mobile_atoms: int
    log_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unique_specorder(atoms) -> list[str]:
    seen: list[str] = []
    for symbol in atoms.get_chemical_symbols():
        if symbol not in seen:
            seen.append(symbol)
    return seen


def _format_pair_coeff(pair_coeff: str, potential_file: str, elements: list[str]) -> str:
    text = (pair_coeff or "").strip()
    if not text:
        if not potential_file:
            raise ValueError("A pair_coeff line or potential file is required for LAMMPS relaxation.")
        return f"pair_coeff * * {potential_file} {' '.join(elements)}"
    return text.format(potential_file=potential_file, elements=" ".join(elements))


def _parse_energy(log_text: str) -> float | None:
    lines = [line.strip() for line in (log_text or "").splitlines() if line.strip()]
    for line in reversed(lines):
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            floats = [float(p) for p in parts[:4]]
        except Exception:
            continue
        return float(floats[2])
    return None


def _parse_thermo_history(log_text: str) -> dict[str, list[float]]:
    steps: list[float] = []
    temps: list[float] = []
    pes: list[float] = []
    etotals: list[float] = []
    for raw in (log_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            step = float(parts[0])
            temp = float(parts[1])
            pe = float(parts[2])
            etotal = float(parts[3])
        except Exception:
            continue
        steps.append(step)
        temps.append(temp)
        pes.append(pe)
        etotals.append(etotal)
    return {"step": steps, "temp": temps, "pe": pes, "etotal": etotals}


def _parse_cutoff_value(cutoff_info: str) -> float | None:
    import re

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", cutoff_info or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _surface_atom_indices(
    atoms,
    *,
    surface_threshold: int,
    hist_rmin: float = 1.8,
    hist_rmax: float = 4.2,
    mode: str = "auto",
    cutoff: float | None = None,
) -> tuple[list[int], str]:
    cn_arr, cutoff_info = compute_cn(
        atoms,
        mode=str(mode),
        cutoff=cutoff,
        hist_rmin=float(hist_rmin),
        hist_rmax=float(hist_rmax),
    )
    avg_nn = _average_nn_distance(atoms)
    if str(mode) == "auto":
        min_cutoff = avg_nn * 1.15 if avg_nn > 0 else None
        parsed_cutoff = None
        for token in cutoff_info.split():
            try:
                parsed_cutoff = float(token)
                break
            except Exception:
                continue
        if parsed_cutoff is not None and min_cutoff is not None and parsed_cutoff < min_cutoff:
            cn_arr, _ = compute_cn(
                atoms,
                mode="fixed",
                cutoff=float(min_cutoff),
                hist_rmin=float(hist_rmin),
                hist_rmax=float(hist_rmax),
            )
            cutoff_info = f"auto cutoff adjusted = {float(min_cutoff):.3f} Å"
    surface_indices = np.where(cn_arr < int(surface_threshold))[0].tolist()
    return surface_indices, cutoff_info


def _site_count_output_path(workdir_path: Path, stage: str) -> Path:
    return workdir_path / f"site_count.{stage}.csv"


def _write_site_count_csv(
    *,
    atoms,
    structure_path: Path,
    out_path: Path,
    mode: str,
    cutoff,
    hist_rmin: float,
    hist_rmax: float,
    surface_threshold: int,
    edge_tol: float = 0.10,
    planarity: float = 0.02,
) -> dict[str, Any]:
    cn_arr, cutoff_info = compute_cn(
        atoms,
        mode=str(mode),
        cutoff=cutoff,
        hist_rmin=float(hist_rmin),
        hist_rmax=float(hist_rmax),
    )
    sites, labels, codes, extras = detect_sites_by_cn(
        atoms,
        cn_arr,
        surface_threshold=int(surface_threshold),
        edge_tol=float(edge_tol),
        planarity=float(planarity),
        r_cap=4.0,
        cn_surface_thr=int(surface_threshold),
        same_cn_tolerance=1,
        min_shared_neighbors=1,
    )
    write_single_csv(
        str(out_path),
        input_file=str(structure_path),
        n_atoms=len(atoms),
        mode=str(mode),
        cutoff_info=cutoff_info,
        surface_threshold=int(surface_threshold),
        cn_arr=cn_arr,
        atoms=atoms,
        sites=sites,
    )
    hist, edges = np.histogram(cn_arr, bins=np.arange(int(np.min(cn_arr)), int(np.max(cn_arr)) + 2) - 0.5)
    return {
        "output_csv": str(out_path),
        "mode": str(mode),
        "cutoff_info": cutoff_info,
        "cutoff_value": _parse_cutoff_value(cutoff_info),
        "hist_rmin": float(hist_rmin),
        "hist_rmax": float(hist_rmax),
        "surface_threshold": int(surface_threshold),
        "surface_atoms": int(np.sum(cn_arr < int(surface_threshold))),
        "surface_indices": np.where(cn_arr < int(surface_threshold))[0].tolist(),
        "counts": {k: len(v) for k, v in sites.items()},
        "cn_hist": [(int(c), int(h)) for c, h in zip(((edges[:-1] + edges[1:]) * 0.5).astype(int), hist)],
        "cn_values": cn_arr.tolist(),
        "site_marks": {},
        "extras": extras,
        "site_codes": codes,
        "site_labels": labels,
    }


def _build_input_script(
    *,
    data_file: str,
    pair_style: str,
    pair_coeff: str,
    temperature: float,
    thermal_steps: int,
    quench_temperature: float,
    quench_steps: int,
    timestep: float,
    damping: float,
    seed: int,
    minimize_etol: float,
    minimize_ftol: float,
    minimize_maxiter: int,
    minimize_maxeval: int,
    remove_drift: bool,
    freeze_core: bool,
    surface_indices: list[int] | None = None,
    dump_interval: int = 100,
    relax_temp: float | None = None,
) -> str:
    t0 = float(temperature if relax_temp is None else relax_temp)
    surface_indices = [int(i) for i in (surface_indices or [])]
    group_lines: list[str] = []
    if freeze_core:
        if not surface_indices:
            raise ValueError("Surface-only relaxation requested, but no surface atoms were identified.")
        lammps_ids = " ".join(str(i + 1) for i in surface_indices)
        group_lines = [
            f"group mobile id {lammps_ids}",
            "group core subtract all mobile",
        ]
    lines = [
        "units metal",
        "atom_style atomic",
        "boundary f f f",
        "log lammps.log",
        f"read_data {data_file}",
        f"pair_style {pair_style}",
        pair_coeff,
        "neighbor 2.0 bin",
        "neigh_modify delay 0 every 1 check yes",
        f"timestep {timestep:.6f}",
        "thermo 100",
        "thermo_style custom step temp pe etotal",
        f"dump traj all custom {int(max(1, dump_interval))} lammps.relax.traj id type x y z",
        "dump_modify traj sort id",
    ]
    lines.extend(group_lines)
    if freeze_core:
        lines.extend(
            [
                f"velocity mobile create {t0:.6f} {seed} mom yes rot yes dist gaussian",
                "velocity core set 0.0 0.0 0.0",
                "fix md mobile nve",
                f"fix bath mobile langevin {t0:.6f} {t0:.6f} {damping:.6f} {seed + 1} zero yes tally yes",
                "fix freeze core setforce 0.0 0.0 0.0",
            ]
        )
        if remove_drift:
            lines.append("fix drift mobile momentum 50 linear 1 1 1")
    else:
        lines.extend(
            [
                f"velocity all create {t0:.6f} {seed} mom yes rot yes dist gaussian",
                "fix md all nve",
                f"fix bath all langevin {t0:.6f} {t0:.6f} {damping:.6f} {seed + 1} zero yes tally yes",
            ]
        )
        if remove_drift:
            lines.append("fix drift all momentum 50 linear 1 1 1")
    lines.extend(
        [
            f"run {int(thermal_steps)}",
            "unfix bath",
            (
                f"fix quench mobile langevin {t0:.6f} {float(quench_temperature):.6f} {damping:.6f} {seed + 2} zero yes tally yes"
                if freeze_core
                else f"fix quench all langevin {t0:.6f} {float(quench_temperature):.6f} {damping:.6f} {seed + 2} zero yes tally yes"
            ),
            f"run {int(quench_steps)}",
            "unfix quench",
            "unfix md",
            "unfix drift" if remove_drift else "",
            "undump traj",
            "min_style fire",
            f"minimize {minimize_etol:.6e} {minimize_ftol:.6e} {int(minimize_maxiter)} {int(minimize_maxeval)}",
            "write_data lammps.relaxed.data",
        ]
    )
    return "\n".join(line for line in lines if line) + "\n"


def run_thermal_relaxation(
    *,
    structure_path: str | Path,
    lammps_command: str = "lmp",
    potential_file: str = "",
    pair_style: str = "eam/alloy",
    pair_coeff: str = "",
    temperature: float = 1200.0,
    thermal_steps: int = 5000,
    quench_temperature: float = 50.0,
    quench_steps: int = 5000,
    timestep: float = 0.001,
    damping: float = 0.1,
    seed: int = 12345,
    minimize_etol: float = 1.0e-8,
    minimize_ftol: float = 1.0e-10,
    minimize_maxiter: int = 1000,
    minimize_maxeval: int = 10000,
    vacuum: float = 10.0,
    remove_drift: bool = True,
    freeze_core: bool = False,
    surface_threshold: int = 12,
    surface_mode: str = "auto",
    surface_cutoff: float | None = None,
    hist_rmin: float = 1.8,
    hist_rmax: float = 4.2,
    workdir: str | Path | None = None,
    dump_interval: int = 100,
    status_callback: Callable[[str], None] | None = None,
) -> ThermalRelaxationResult:
    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    structure_path = Path(structure_path).expanduser().resolve()
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")
    if not (lammps_command or "").strip():
        raise ValueError("LAMMPS command is empty.")

    atoms = read(str(structure_path))
    atoms.pbc = False
    _ensure_cell(atoms)
    if vacuum > 0:
        try:
            atoms.center(vacuum=float(vacuum))
        except Exception:
            pass
    initial_atoms = len(atoms)
    initial_average_nn_distance = _average_nn_distance(atoms)
    initial_potential_energy = None
    try:
        initial_potential_energy = float(getattr(atoms, "get_potential_energy")())  # type: ignore[misc]
    except Exception:
        initial_potential_energy = None

    surface_indices: list[int] = []
    surface_cutoff_info = ""
    if freeze_core:
        surface_indices, surface_cutoff_info = _surface_atom_indices(
            atoms,
            surface_threshold=int(surface_threshold),
            hist_rmin=float(hist_rmin),
            hist_rmax=float(hist_rmax),
            mode=str(surface_mode),
            cutoff=surface_cutoff,
        )

    if workdir is None:
        workdir_path = structure_path.parent
    else:
        workdir_path = Path(workdir).expanduser().resolve()
    workdir_path.mkdir(parents=True, exist_ok=True)

    prepared = prepare_thermal_relaxation_inputs(
        structure_path=structure_path,
        atoms=atoms,
        workdir_path=workdir_path,
        lammps_command=lammps_command,
        potential_file=potential_file,
        pair_style=pair_style,
        pair_coeff=pair_coeff,
        temperature=temperature,
        thermal_steps=thermal_steps,
        quench_temperature=quench_temperature,
        quench_steps=quench_steps,
        timestep=timestep,
        damping=damping,
        seed=seed,
        minimize_etol=minimize_etol,
        minimize_ftol=minimize_ftol,
        minimize_maxiter=minimize_maxiter,
        minimize_maxeval=minimize_maxeval,
        remove_drift=remove_drift,
        freeze_core=freeze_core,
        surface_threshold=surface_threshold,
        surface_mode=surface_mode,
        surface_cutoff=surface_cutoff,
        hist_rmin=hist_rmin,
        hist_rmax=hist_rmax,
        surface_indices=surface_indices,
        dump_interval=dump_interval,
        status_callback=_status,
    )

    cmd = list(lammps_exe) + ["-in", Path(prepared.script_path).name]
    _status(f"Launching LAMMPS: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(prepared.inputs_dir)),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _status(f"LAMMPS process started (pid {proc.pid}). Waiting for completion...")
    stdout, stderr = proc.communicate()
    log_text = (stdout or "") + ("\n" if stdout and stderr else "") + (stderr or "")
    _status("LAMMPS finished. Reading relaxed output...")
    if proc.returncode != 0:
        raise RuntimeError(f"LAMMPS failed with exit code {proc.returncode}.\n{log_text}")

    inputs_dir = Path(prepared.inputs_dir)
    relaxed_data = inputs_dir / "lammps.relaxed.data"
    if not relaxed_data.exists():
        raise RuntimeError(f"LAMMPS completed but did not write {relaxed_data}")

    relaxed_atoms = read(str(relaxed_data), format="lammps-data")
    relaxed_atoms.pbc = False
    relaxed_xyz = inputs_dir / "lammps.relaxed.xyz"
    relaxed_poscar = inputs_dir / "lammps.relaxed.POSCAR"
    write(relaxed_xyz, relaxed_atoms, format="extxyz")
    try:
        write(relaxed_poscar, relaxed_atoms, format="vasp")
    except Exception:
        write(relaxed_poscar, relaxed_atoms)

    final_average_nn_distance = _average_nn_distance(relaxed_atoms)
    final_potential_energy = None
    try:
        final_potential_energy = _parse_energy(log_text)
    except Exception:
        final_potential_energy = None
    thermo_history = _parse_thermo_history(log_text)

    result = ThermalRelaxationResult(
        input_file=str(structure_path),
        relaxed_xyz=str(relaxed_xyz),
        relaxed_poscar=str(relaxed_poscar),
        relaxed_data=str(relaxed_data),
        trajectory_file=str(inputs_dir / "lammps.relax.traj"),
        workdir=str(workdir_path),
        inputs_dir=prepared.inputs_dir,
        site_count_before_csv=prepared.site_count_before_csv,
        lammps_command=" ".join(lammps_exe),
        pair_style=pair_style.strip(),
        pair_coeff=prepared.pair_coeff,
        potential_file=prepared.potential_file,
        copied_potential_file=prepared.copied_potential_file,
        temperature=float(temperature),
        thermal_steps=int(thermal_steps),
        quench_temperature=float(quench_temperature),
        quench_steps=int(quench_steps),
        timestep=float(timestep),
        damping=float(damping),
        seed=int(seed),
        initial_atoms=initial_atoms,
        final_atoms=len(relaxed_atoms),
        surface_atoms=len(surface_indices),
        mobile_atoms=len(surface_indices) if freeze_core else len(relaxed_atoms),
        initial_average_nn_distance=float(initial_average_nn_distance),
        final_average_nn_distance=float(final_average_nn_distance),
        initial_potential_energy=initial_potential_energy,
        final_potential_energy=final_potential_energy,
        thermo_history=thermo_history,
        log_text=(f"{surface_cutoff_info}\n{log_text}" if surface_cutoff_info else log_text),
    )
    return result


def write_thermal_relaxation_inputs(
    *,
    structure_path: str | Path,
    lammps_command: str = "lmp",
    potential_file: str = "",
    pair_style: str = "eam/alloy",
    pair_coeff: str = "",
    temperature: float = 1200.0,
    thermal_steps: int = 5000,
    quench_temperature: float = 50.0,
    quench_steps: int = 5000,
    timestep: float = 0.001,
    damping: float = 0.1,
    seed: int = 12345,
    minimize_etol: float = 1.0e-8,
    minimize_ftol: float = 1.0e-10,
    minimize_maxiter: int = 1000,
    minimize_maxeval: int = 10000,
    vacuum: float = 10.0,
    remove_drift: bool = True,
    freeze_core: bool = False,
    surface_threshold: int = 12,
    surface_mode: str = "auto",
    surface_cutoff: float | None = None,
    hist_rmin: float = 1.8,
    hist_rmax: float = 4.2,
    workdir: str | Path | None = None,
    dump_interval: int = 100,
    status_callback: Callable[[str], None] | None = None,
) -> ThermalRelaxationInputs:
    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    structure_path = Path(structure_path).expanduser().resolve()
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    if not (lammps_command or "").strip():
        raise ValueError("LAMMPS command is empty.")

    atoms = read(str(structure_path))
    atoms.pbc = False
    _ensure_cell(atoms)
    if vacuum > 0:
        try:
            atoms.center(vacuum=float(vacuum))
        except Exception:
            pass

    if not pair_style.strip():
        raise ValueError("LAMMPS pair_style must be provided.")
    specorder = _unique_specorder(atoms)
    resolved_pair_coeff = _format_pair_coeff(pair_coeff, potential_file.strip(), specorder)

    surface_indices: list[int] = []
    if freeze_core:
        surface_indices, _ = _surface_atom_indices(
            atoms,
            surface_threshold=int(surface_threshold),
            hist_rmin=float(hist_rmin),
            hist_rmax=float(hist_rmax),
            mode=str(surface_mode),
            cutoff=surface_cutoff,
        )

    if workdir is None:
        workdir_path = structure_path.parent
    else:
        workdir_path = Path(workdir).expanduser().resolve()
    workdir_path.mkdir(parents=True, exist_ok=True)

    return prepare_thermal_relaxation_inputs(
        structure_path=structure_path,
        atoms=atoms,
        workdir_path=workdir_path,
        lammps_command=lammps_command,
        potential_file=potential_file,
        pair_style=pair_style,
        pair_coeff=pair_coeff,
        temperature=temperature,
        thermal_steps=thermal_steps,
        quench_temperature=quench_temperature,
        quench_steps=quench_steps,
        timestep=timestep,
        damping=damping,
        seed=seed,
        minimize_etol=minimize_etol,
        minimize_ftol=minimize_ftol,
        minimize_maxiter=minimize_maxiter,
        minimize_maxeval=minimize_maxeval,
        remove_drift=remove_drift,
        freeze_core=freeze_core,
        surface_threshold=surface_threshold,
        surface_mode=surface_mode,
        surface_cutoff=surface_cutoff,
        hist_rmin=hist_rmin,
        hist_rmax=hist_rmax,
        surface_indices=surface_indices,
        dump_interval=dump_interval,
        include_site_count=False,
        status_callback=_status,
    )


def prepare_thermal_relaxation_inputs(
    *,
    structure_path: str | Path,
    atoms,
    workdir_path: Path,
    lammps_command: str,
    potential_file: str,
    pair_style: str,
    pair_coeff: str,
    temperature: float,
    thermal_steps: int,
    quench_temperature: float,
    quench_steps: int,
    timestep: float,
    damping: float,
    seed: int,
    minimize_etol: float,
    minimize_ftol: float,
    minimize_maxiter: int,
    minimize_maxeval: int,
    remove_drift: bool,
    freeze_core: bool,
    surface_threshold: int,
    surface_mode: str,
    surface_cutoff: float | None,
    hist_rmin: float,
    hist_rmax: float,
    surface_indices: list[int],
    dump_interval: int,
    include_site_count: bool = True,
    status_callback: Callable[[str], None] | None = None,
) -> ThermalRelaxationInputs:
    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    structure_path = Path(structure_path).expanduser().resolve()
    workdir_path.mkdir(parents=True, exist_ok=True)

    inputs_dir = workdir_path / "lammps_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    _status(f"Created LAMMPS inputs folder at {inputs_dir}")

    before_site_count_csv = ""
    if include_site_count:
        _status("Writing pre-relaxation site-count data...")
        before_site_count = _write_site_count_csv(
            atoms=atoms,
            structure_path=structure_path,
            out_path=_site_count_output_path(workdir_path, "before_relax"),
            mode=str(surface_mode),
            cutoff=surface_cutoff,
            hist_rmin=float(hist_rmin),
            hist_rmax=float(hist_rmax),
            surface_threshold=int(surface_threshold),
        )
        before_site_count_csv = str(before_site_count["output_csv"])
    else:
        _status("Skipping site-count generation for write-only export.")

    copied_potential_file = ""
    potential_source = Path(potential_file).expanduser()
    if potential_file.strip() and potential_source.exists():
        copied_potential_file = potential_source.name
        shutil.copy2(str(potential_source), str(inputs_dir / copied_potential_file))
    elif potential_file.strip():
        copied_potential_file = potential_source.name or potential_file.strip()

    specorder = _unique_specorder(atoms)
    resolved_pair_coeff = _format_pair_coeff(pair_coeff, copied_potential_file or potential_file.strip(), specorder)

    data_path = inputs_dir / "lammps.start.data"
    script_path = inputs_dir / "lammps.relax.in"
    write(data_path, atoms, format="lammps-data", atom_style="atomic", specorder=specorder)
    script_path.write_text(
        _build_input_script(
            data_file=str(data_path.name),
            pair_style=pair_style.strip(),
            pair_coeff=resolved_pair_coeff,
            temperature=float(temperature),
            thermal_steps=int(thermal_steps),
            quench_temperature=float(quench_temperature),
            quench_steps=int(quench_steps),
            timestep=float(timestep),
            damping=float(damping),
            seed=int(seed),
            minimize_etol=float(minimize_etol),
            minimize_ftol=float(minimize_ftol),
            minimize_maxiter=int(minimize_maxiter),
            minimize_maxeval=int(minimize_maxeval),
            remove_drift=bool(remove_drift),
            freeze_core=bool(freeze_core),
            surface_indices=surface_indices,
            dump_interval=int(dump_interval),
        ),
        encoding="utf-8",
    )

    _status(f"Wrote LAMMPS input files to {inputs_dir}")
    return ThermalRelaxationInputs(
        structure_path=str(structure_path),
        workdir=str(workdir_path),
        inputs_dir=str(inputs_dir),
        data_path=str(data_path),
        script_path=str(script_path),
        site_count_before_csv=before_site_count_csv,
        lammps_command=str(lammps_command),
        pair_style=pair_style.strip(),
        pair_coeff=resolved_pair_coeff,
        potential_file=potential_file.strip(),
        copied_potential_file=copied_potential_file,
        temperature=float(temperature),
        thermal_steps=int(thermal_steps),
        quench_temperature=float(quench_temperature),
        quench_steps=int(quench_steps),
        timestep=float(timestep),
        damping=float(damping),
        seed=int(seed),
        surface_atoms=len(surface_indices),
        mobile_atoms=len(surface_indices) if freeze_core else len(atoms),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Thermal relaxation and quench via LAMMPS.")
    p.add_argument("structure", help="Input geometry file (ASE-readable).")
    p.add_argument("--lammps-command", default="lmp")
    p.add_argument("--potential-file", default="")
    p.add_argument("--pair-style", default="eam/alloy")
    p.add_argument("--pair-coeff", default="")
    p.add_argument("--temperature", type=float, default=1200.0)
    p.add_argument("--thermal-steps", type=int, default=5000)
    p.add_argument("--quench-temperature", type=float, default=50.0)
    p.add_argument("--quench-steps", type=int, default=5000)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--damping", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--minimize-etol", type=float, default=1.0e-8)
    p.add_argument("--minimize-ftol", type=float, default=1.0e-10)
    p.add_argument("--minimize-maxiter", type=int, default=1000)
    p.add_argument("--minimize-maxeval", type=int, default=10000)
    p.add_argument("--vacuum", type=float, default=10.0)
    p.add_argument("--no-remove-drift", action="store_true")
    p.add_argument("--freeze-core", action="store_true")
    p.add_argument("--surface-threshold", type=int, default=12)
    p.add_argument("--surface-mode", default="auto")
    p.add_argument("--surface-cutoff", type=float, default=None)
    p.add_argument("--hist-rmin", type=float, default=1.8)
    p.add_argument("--hist-rmax", type=float, default=4.2)
    p.add_argument("--dump-interval", type=int, default=100)
    p.add_argument("--workdir", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_thermal_relaxation(
        structure_path=args.structure,
        lammps_command=args.lammps_command,
        potential_file=args.potential_file,
        pair_style=args.pair_style,
        pair_coeff=args.pair_coeff,
        temperature=args.temperature,
        thermal_steps=args.thermal_steps,
        quench_temperature=args.quench_temperature,
        quench_steps=args.quench_steps,
        timestep=args.timestep,
        damping=args.damping,
        seed=args.seed,
        minimize_etol=args.minimize_etol,
        minimize_ftol=args.minimize_ftol,
        minimize_maxiter=args.minimize_maxiter,
        minimize_maxeval=args.minimize_maxeval,
        vacuum=args.vacuum,
        remove_drift=not args.no_remove_drift,
        freeze_core=args.freeze_core,
        surface_threshold=args.surface_threshold,
        surface_mode=args.surface_mode,
        surface_cutoff=args.surface_cutoff,
        hist_rmin=args.hist_rmin,
        hist_rmax=args.hist_rmax,
        dump_interval=args.dump_interval,
        workdir=args.workdir,
    )
    print(f"Input: {result.input_file}")
    print(f"Relaxed XYZ: {result.relaxed_xyz}")
    print(f"Relaxed POSCAR: {result.relaxed_poscar}")
    print(f"Trajectory: {result.trajectory_file}")
    print(f"Initial atoms: {result.initial_atoms}")
    print(f"Final atoms: {result.final_atoms}")
    if result.surface_atoms:
        print(f"Surface atoms: {result.surface_atoms}")
        print(f"Mobile atoms: {result.mobile_atoms}")
    print(f"Initial avg NN: {result.initial_average_nn_distance:.3f} Å")
    print(f"Final avg NN: {result.final_average_nn_distance:.3f} Å")
    if result.initial_potential_energy is not None:
        print(f"Initial PE: {result.initial_potential_energy:.6f}")
    if result.final_potential_energy is not None:
        print(f"Final PE: {result.final_potential_energy:.6f}")
    print(result.log_text)


if __name__ == "__main__":
    main()
