from __future__ import annotations

import json
import math

from ase.io import write

from ..common import BuildResult, _average_nn_distance, _ensure_cell, _max_radius, cluster_output_dir, stamp_cluster_metadata, write_cluster_files


def build_ru_icosahedron(
    a_hcp: float,
    tag: str,
    target_atoms: int | None = None,
    target_radius: float | None = None,
    ico_shells: int | None = None,
    verbose: bool = False,
) -> BuildResult:
    """
    Build a Mackay icosahedron for Ru. ASE's Icosahedron uses an fcc lattice constant 'a_fcc';
    nearest-neighbor distance is a_fcc / sqrt(2). We choose a_fcc so that NN distance equals Ru's a_hcp.
    """
    from ase.cluster.icosahedron import Icosahedron

    a_fcc = float(a_hcp) * math.sqrt(2.0)

    def mk(nshells: int):
        atoms = Icosahedron("Ru", noshells=int(nshells), latticeconstant=a_fcc)
        atoms.center(vacuum=10.0)
        return atoms

    if ico_shells is not None:
        atoms = mk(int(ico_shells))
    else:
        best = None
        for ns in range(1, 25):
            cand = mk(ns)
            if target_atoms is not None:
                score = abs(len(cand) - int(target_atoms))
            else:
                score = abs(_max_radius(cand) - float(target_radius))
            if (best is None) or (score < best[0]):
                best = (score, ns, cand)
        atoms = best[2]
        if verbose:
            print(f"[ico] selected shells={best[1]}  n={len(atoms)}  r={_max_radius(atoms):.3f} Å")

    _ensure_cell(atoms)
    stamp_cluster_metadata(
        atoms,
        tag=tag,
        shape="icosahedron",
        crystal_structure="fcc",
        average_nn_distance=_average_nn_distance(atoms),
    )
    write_cluster_files(tag, atoms)
    output_dir = cluster_output_dir()

    r_final = _max_radius(atoms)
    meta = {
        "tag": tag,
        "shape": "icosahedron",
        "a_fcc_used": a_fcc,
        "n_atoms": len(atoms),
        "actual_radius": r_final,
        "formula": atoms.get_chemical_formula(empirical=True),
    }
    with open(output_dir / f"{tag}_ico_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return BuildResult(len(atoms), atoms.get_chemical_formula(empirical=True), surfaces_hkl=[], gammas=[], actual_radius=r_final)
