from __future__ import annotations

import json
import math

from ase.cluster.cubic import FaceCenteredCubic
from ase.io import write

from ..common import BuildResult, _average_nn_distance, _ensure_cell, _max_radius, cluster_output_dir, stamp_cluster_metadata, write_cluster_files


def build_ru_cuboctahedron(
    a_ref: float = 2.706,
    tag: str = "Ru_cuboct",
    target_atoms: int | None = None,
    target_radius: float | None = None,
    cuboct_layers: int | None = None,
    verbose: bool = False,
) -> BuildResult:
    """
    Build an fcc cuboctahedral Ru cluster exposing (111) and (100) facets.

    Works with older/newer ASE: use 'surfaces' + 'layers' (not 'directions').
    For a symmetric cuboctahedron, set equal up/down layers for each surface family.
    """
    a_fcc = float(a_ref) * math.sqrt(2.0)
    surfaces = [(1, 0, 0), (1, 1, 1)]

    def mk(n: int):
        layers = [(n, n), (n, n)]
        atoms = FaceCenteredCubic("Ru", surfaces=surfaces, layers=layers, latticeconstant=a_fcc)
        atoms.center(vacuum=10.0)
        return atoms

    if cuboct_layers is not None:
        atoms = mk(int(cuboct_layers))
    else:
        best = None
        for n in range(1, 25):
            cand = mk(n)
            score = abs(len(cand) - int(target_atoms)) if target_atoms is not None else abs(_max_radius(cand) - float(target_radius))
            if best is None or score < best[0]:
                best = (score, n, cand)
        atoms = best[2]
        if verbose:
            print(f"[cuboct] selected layers={best[1]} n={len(atoms)} r={_max_radius(atoms):.3f} Å")

    _ensure_cell(atoms)
    stamp_cluster_metadata(atoms, tag=tag, shape="cuboct", crystal_structure="fcc", average_nn_distance=_average_nn_distance(atoms))
    write_cluster_files(tag, atoms)
    output_dir = cluster_output_dir()

    r_final = _max_radius(atoms)
    meta = {
        "tag": tag,
        "shape": "cuboctahedron",
        "a_fcc_used": a_fcc,
        "n_atoms": len(atoms),
        "actual_radius": r_final,
        "formula": atoms.get_chemical_formula(empirical=True),
    }
    with open(output_dir / f"{tag}_cuboct_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return BuildResult(len(atoms), atoms.get_chemical_formula(empirical=True), surfaces_hkl=[], gammas=[], actual_radius=r_final)
