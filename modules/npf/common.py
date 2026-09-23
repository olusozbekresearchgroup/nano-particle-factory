from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from ase.io import write
from ase.cluster.wulff import wulff_construction

# ---------- Lattice + Ru γ(hkil) ----------
DEFAULT_A = 2.706  # Å (hcp Ru)
DEFAULT_C = 4.282  # Å

CRYSTALIUM_RU_EGAMMA: Dict[str, float] = {
    "0001": 0.162,
    "1011": 0.180,
    "1010": 0.181,
    "1012": 0.190,
    "2112": 0.194,
    "2132": 0.199,
    "2131": 0.203,
    "1121": 0.203,
    "2021": 0.205,
    "2130": 0.205,
    "2241": 0.207,
    "1120": 0.208,
}

HKIL_RE = re.compile(r"^\s*([+-]?\d+)\s*([+-]?\d+)\s*-?\s*([+-]?\d+)\s*([+-]?\d+)\s*$")
H_K_L_RE = re.compile(r"^\s*([+-]?\d+)\s*([+-]?\d+)\s*([+-]?\d+)\s*$")


def parse_hkil_token(tok: str) -> Tuple[int, int, int, int]:
    s = tok.strip().replace(" ", "")
    if re.match(r"^[+-]?\d+[+-]?\d+-[+-]?\d+[+-]?\d+$", s):
        nums, sign = [], 1
        for ch in s:
            if ch == "+":
                sign = 1
            elif ch == "-":
                sign = -1
            else:
                if not ch.isdigit():
                    raise ValueError(f"Cannot parse facet '{tok}'.")
                nums.append(sign * int(ch))
                sign = 1
        if len(nums) != 4:
            raise ValueError(f"Cannot parse facet '{tok}'.")
        h, k, i_, l = nums
        if i_ != -(h + k):
            i_ = -(h + k)
        return h, k, i_, l
    m4 = HKIL_RE.match(s)
    if m4:
        h, k, i_, l = map(int, m4.groups())
        if i_ != -(h + k):
            i_ = -(h + k)
        return h, k, i_, l
    m3 = H_K_L_RE.match(tok.strip())
    if m3:
        h, k, l = map(int, m3.groups())
        return h, k, -(h + k), l
    d = tok.strip()
    if len(d) == 4 and d.isdigit():
        h, k, i_, l = [int(ch) for ch in d]
        if i_ != -(h + k):
            i_ = -(h + k)
        return h, k, i_, l
    raise ValueError(f"Could not parse HKIL facet token: '{tok}'")


def hkil_to_hkl(h: int, k: int, i_: int, l: int) -> Tuple[int, int, int]:
    return int(h), int(k), int(l)


def hkl_to_hkil(h: int, k: int, l: int) -> Tuple[int, int, int, int]:
    return int(h), int(k), int(-(h + k)), int(l)


def format_hkil(h: int, k: int, l: int) -> str:
    hh, kk, ii, ll = hkl_to_hkil(h, k, l)
    return f"({hh} {kk} {ii} {ll})"


def crystalium_ru_gammas_hkl() -> Dict[Tuple[int, int, int], float]:
    out = {}
    for facet, g in CRYSTALIUM_RU_EGAMMA.items():
        h, k, i_, l = parse_hkil_token(facet)
        out[hkil_to_hkl(h, k, i_, l)] = float(g)
    return out


def parse_surfaces_expr(expr: str) -> Dict[Tuple[int, int, int], float]:
    if not expr:
        return {}
    out = {}
    for item in expr.split(","):
        item = item.strip()
        if not item:
            continue
        facet, val = [x.strip() for x in item.split("=")]
        h, k, i_, l = parse_hkil_token(facet)
        out[hkil_to_hkl(h, k, i_, l)] = float(val)
    return out


@dataclass
class BuildResult:
    atoms_count: int
    formula: str
    surfaces_hkl: List[Tuple[int, int, int]]
    gammas: List[float]
    actual_radius: float


def _try_build(
    surfaces: Sequence[Tuple[int, int, int]],
    energies: Sequence[float],
    size_value: float | int,
    a: float,
    c: float,
):
    surfaces = [(int(h), int(k), int(l)) for (h, k, l) in surfaces]
    energies = [float(g) for g in energies]
    return wulff_construction(
        symbol="Ru",
        surfaces=list(surfaces),
        energies=list(energies),
        size=size_value,
        structure="hcp",
        latticeconstant=(a, c),
        rounding="above",
    )


def _max_radius(atoms) -> float:
    pos = atoms.get_positions()
    com = pos.mean(axis=0)
    return float(np.sqrt(((pos - com) ** 2).sum(axis=1)).max())


def _average_nn_distance(atoms) -> float:
    pos = atoms.get_positions()
    if len(pos) < 2:
        return 0.0
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(pos).query(pos, k=2, workers=-1)
        return float(np.mean(distances[:, 1]))
    except ImportError:
        from ase.neighborlist import neighbor_list

        cutoff = 4.0
        for _ in range(8):
            i, distances = neighbor_list("id", atoms, cutoff)
            nearest = np.full(len(pos), np.inf, dtype=float)
            np.minimum.at(nearest, i, distances)
            if np.all(np.isfinite(nearest)):
                return float(nearest.mean())
            cutoff *= 2.0
    return 0.0


def _ensure_cell(atoms):
    try:
        atoms.center(vacuum=10.0)
    except Exception:
        pos = atoms.get_positions()
        mins, maxs = pos.min(axis=0), pos.max(axis=0)
        lengths = (maxs - mins) + 20.0
        atoms.set_cell([[lengths[0], 0, 0], [0, lengths[1], 0], [0, 0, lengths[2]]], scale_atoms=False)
        atoms.center()


def _apply_radial_trim(atoms, rtrim: float):
    """Remove atoms with radius > rtrim (keeps COM)."""
    pos = atoms.get_positions()
    com = pos.mean(axis=0)
    r = np.sqrt(((pos - com) ** 2).sum(axis=1))
    keep = r <= float(rtrim) + 1e-9
    sub = atoms[keep]
    if len(sub) == 0:
        raise ValueError("Radial trim removed all atoms; choose a larger --radial-trim.")
    return sub


def stamp_cluster_metadata(
    atoms,
    *,
    tag: str,
    shape: str,
    crystal_structure: str | None = None,
    crystal_variant: str | None = None,
    **extra,
):
    info = dict(getattr(atoms, "info", {}) or {})
    info["npf_tag"] = str(tag)
    info["npf_shape"] = str(shape)
    if crystal_structure is not None:
        info["npf_crystal_structure"] = str(crystal_structure)
    if crystal_variant is not None:
        info["npf_crystal_variant"] = str(crystal_variant)
    for key, value in extra.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            info[f"npf_{key}"] = value
        else:
            info[f"npf_{key}"] = json.dumps(value)
    atoms.info = info
    return atoms


def write_cluster_files(tag: str, atoms):
    output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    write(output_dir / f"{tag}.xyz", atoms, format="extxyz")
    try:
        write(output_dir / "POSCAR", atoms, format="vasp")
    except Exception:
        write(output_dir / "POSCAR", atoms)


def _write_xyz_file(path: str | Path, atoms) -> None:
    """Write extxyz explicitly; ASE otherwise treats POSCAR*.xyz names as VASP."""
    write(Path(path), atoms, format="extxyz")


def _xyz_output_path(source: str | Path, suffix: str = "") -> Path:
    """Name XYZ exports after the structure, without POSCAR/CONTCAR labels."""
    path = Path(source)
    stem = path.stem
    folder = re.sub(r"poscar|contcar", "", path.parent.name, flags=re.IGNORECASE).strip("._-") or "structure"
    stem = re.sub(
        r"^(?:poscar|contcar)(?=$|[._-])",
        lambda _match: folder,
        stem,
        flags=re.IGNORECASE,
    )
    stem = re.sub(r"poscar|contcar", "", stem, flags=re.IGNORECASE).strip("._-") or folder
    return path.with_name(f"{stem}{suffix}.xyz")


def cluster_output_dir(base_dir: str | Path | None = None, tag: str | None = None) -> Path:
    root = Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd().resolve()
    output_dir = root / tag if tag else root
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
