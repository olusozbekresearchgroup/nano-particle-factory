from __future__ import annotations

import json
import math

from ase.cluster.octahedron import Octahedron
from ase.io import write

from ..common import BuildResult, _average_nn_distance, _ensure_cell, _max_radius, cluster_output_dir, stamp_cluster_metadata, write_cluster_files


def build_ru_octahedron(
    a_ref: float = 2.706,
    tag: str = "Ru_octahedron",
    target_atoms: int | None = None,
    target_radius: float | None = None,
    octa_length: int | None = None,
    octa_cutoff: int = 0,
    verbose: bool = False,
) -> BuildResult:
    """
    Build an octahedral Ru cluster using ASE's Octahedron constructor.

    The search uses the `length` parameter when atoms/radius are requested.
    """
    a_fcc = float(a_ref) * math.sqrt(2.0)

    def mk(n: int):
        atoms = Octahedron("Ru", length=int(n), cutoff=int(octa_cutoff), latticeconstant=a_fcc)
        atoms.center(vacuum=10.0)
        return atoms

    if octa_length is not None:
        atoms = mk(int(octa_length))
    else:
        best = None
        for n in range(1, 25):
            cand = mk(n)
            score = abs(len(cand) - int(target_atoms)) if target_atoms is not None else abs(_max_radius(cand) - float(target_radius))
            if best is None or score < best[0]:
                best = (score, n, cand)
        atoms = best[2]
        if verbose:
            print(f"[octahedron] selected length={best[1]} n={len(atoms)} r={_max_radius(atoms):.3f} Å")

    _ensure_cell(atoms)
    stamp_cluster_metadata(
        atoms,
        tag=tag,
        shape="octahedron",
        crystal_structure="fcc",
        average_nn_distance=_average_nn_distance(atoms),
    )
    write_cluster_files(tag, atoms)
    output_dir = cluster_output_dir()

    r_final = _max_radius(atoms)
    meta = {
        "tag": tag,
        "shape": "octahedron",
        "a_fcc_used": a_fcc,
        "cutoff": int(octa_cutoff),
        "n_atoms": len(atoms),
        "actual_radius": r_final,
        "formula": atoms.get_chemical_formula(empirical=True),
    }
    with open(output_dir / f"{tag}_octahedron_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return BuildResult(len(atoms), atoms.get_chemical_formula(empirical=True), surfaces_hkl=[], gammas=[], actual_radius=r_final)
