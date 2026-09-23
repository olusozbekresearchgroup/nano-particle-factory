from __future__ import annotations

import json
import math
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.io import write

from ..common import BuildResult, _average_nn_distance, _ensure_cell, _max_radius, cluster_output_dir, stamp_cluster_metadata, write_cluster_files


FCC_BASIS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ],
    dtype=float,
)


def _superellipsoid_mask(pos: np.ndarray, radius: float, exponent: float) -> np.ndarray:
    ax = ay = az = float(radius)
    p = float(exponent)
    x = np.abs(pos[:, 0]) / ax
    y = np.abs(pos[:, 1]) / ay
    z = np.abs(pos[:, 2]) / az
    return (x**p + y**p + z**p) <= 1.0


def _fcc_superellipsoid(radius: float, a_fcc: float, exponent: float) -> Atoms:
    n_cells = max(2, int(math.ceil(radius / a_fcc)) + 3)
    points = []
    for i in range(-n_cells, n_cells + 1):
        for j in range(-n_cells, n_cells + 1):
            for k in range(-n_cells, n_cells + 1):
                cell_origin = np.array([i, j, k], dtype=float)
                for basis in FCC_BASIS:
                    points.append((cell_origin + basis) * a_fcc)
    pos = np.asarray(points, dtype=float)
    mask = _superellipsoid_mask(pos, radius=radius, exponent=exponent)
    pos = pos[mask]
    if len(pos) == 0:
        raise ValueError("Generated morphed spherical cluster is empty.")
    atoms = Atoms(symbols=["Ru"] * len(pos), positions=pos)
    atoms.center(vacuum=10.0)
    return atoms


def build_ru_morphed_spherical(
    a_ref: float = 2.706,
    tag: str = "Ru_morphed_spherical",
    target_atoms: int | None = None,
    target_radius: float | None = None,
    morph: float = 0.0,
    verbose: bool = False,
) -> BuildResult:
    """
    Build a spherical-to-faceted Ru cluster using an FCC superellipsoid cut.

    morph = 0.0 gives a sphere-like cluster.
    morph = 1.0 gives a more cube-like superellipsoid.
    """
    if not (0.0 <= float(morph) <= 1.0):
        raise ValueError("morph must be in [0, 1].")

    a_fcc = float(a_ref) * math.sqrt(2.0)
    exponent = 2.0 + 8.0 * float(morph)

    def build_from_radius(radius_value: float):
        return _fcc_superellipsoid(radius_value, a_fcc=a_fcc, exponent=exponent)

    if target_radius is not None:
        atoms = build_from_radius(float(target_radius))
        chosen_radius = float(target_radius)
    else:
        rho = 4.0 / (a_fcc**3)
        guess = max(1.0, (3.0 * float(target_atoms) / (4.0 * math.pi * rho)) ** (1.0 / 3.0))
        best = None
        radius_value = guess
        for _ in range(12):
            atoms = build_from_radius(radius_value)
            n = len(atoms)
            score = abs(n - int(target_atoms))
            actual_r = _max_radius(atoms)
            if best is None or score < best[0]:
                best = (score, radius_value, atoms, actual_r, n)
            if score == 0:
                break
            if n == 0:
                break
            radius_value *= (float(target_atoms) / float(n)) ** (1.0 / 3.0)
        atoms = best[2]
        chosen_radius = best[1]
        if verbose:
            print(
                f"[morphed_spherical] selected radius={chosen_radius:.3f} "
                f"n={best[4]} r={best[3]:.3f} Å exponent={exponent:.2f}"
            )

    _ensure_cell(atoms)
    stamp_cluster_metadata(
        atoms,
        tag=tag,
        shape="morphed_spherical",
        crystal_structure="fcc",
        average_nn_distance=_average_nn_distance(atoms),
    )
    write_cluster_files(tag, atoms)
    output_dir = cluster_output_dir()

    r_final = _max_radius(atoms)
    meta = {
        "tag": tag,
        "shape": "morphed_spherical",
        "a_fcc_used": a_fcc,
        "morph": float(morph),
        "superellipsoid_exponent": exponent,
        "radius_seed": chosen_radius,
        "n_atoms": len(atoms),
        "actual_radius": r_final,
        "formula": atoms.get_chemical_formula(empirical=True),
    }
    with open(output_dir / f"{tag}_morphed_spherical_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return BuildResult(
        len(atoms),
        atoms.get_chemical_formula(empirical=True),
        surfaces_hkl=[],
        gammas=[],
        actual_radius=r_final,
    )
