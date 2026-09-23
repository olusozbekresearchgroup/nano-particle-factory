from __future__ import annotations

import json
import math

from ase.cluster.cubic import SimpleCubic
from ase.io import write

from ..common import BuildResult, _average_nn_distance, _ensure_cell, _max_radius, stamp_cluster_metadata, write_cluster_files


def build_ru_cube(
    a_ref: float = 2.706,
    tag: str = "Ru_cube",
    target_atoms: int | None = None,
    target_radius: float | None = None,
    cube_layers: int | None = None,
    verbose: bool = False,
) -> BuildResult:
    """
    Build a cubic Ru cluster using ASE's SimpleCubic cluster constructor.

    The search uses layer count as the size parameter when atoms/radius are requested.
    """
    a_sc = float(a_ref)
    surfaces = [(1, 0, 0)]

    def mk(n: int):
        atoms = SimpleCubic("Ru", surfaces=surfaces, layers=[(n, n)], latticeconstant=a_sc)
        atoms.center(vacuum=10.0)
        return atoms

    if cube_layers is not None:
        atoms = mk(int(cube_layers))
    else:
        best = None
        for n in range(1, 25):
            cand = mk(n)
            score = abs(len(cand) - int(target_atoms)) if target_atoms is not None else abs(_max_radius(cand) - float(target_radius))
            if best is None or score < best[0]:
                best = (score, n, cand)
        atoms = best[2]
        if verbose:
            print(f"[cube] selected layers={best[1]} n={len(atoms)} r={_max_radius(atoms):.3f} Å")

    _ensure_cell(atoms)
    stamp_cluster_metadata(
        atoms,
        tag=tag,
        shape="cube",
        crystal_structure="fcc",
        average_nn_distance=_average_nn_distance(atoms),
    )
    write_cluster_files(tag, atoms)

    r_final = _max_radius(atoms)
    meta = {
        "tag": tag,
        "shape": "cube",
        "a_sc_used": a_sc,
        "n_atoms": len(atoms),
        "actual_radius": r_final,
        "formula": atoms.get_chemical_formula(empirical=True),
    }
    with open(f"{tag}_cube_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return BuildResult(len(atoms), atoms.get_chemical_formula(empirical=True), surfaces_hkl=[], gammas=[], actual_radius=r_final)
