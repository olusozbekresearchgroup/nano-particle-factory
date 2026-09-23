from __future__ import annotations

import json
import math

from ase.cluster.decahedron import Decahedron
from ase.io import write

from ..common import BuildResult, _average_nn_distance, _ensure_cell, _max_radius, cluster_output_dir, stamp_cluster_metadata, write_cluster_files


def build_ru_decahedron(
    a_ref: float = 2.706,
    tag: str = "Ru_decahedron",
    target_atoms: int | None = None,
    target_radius: float | None = None,
    p: int | None = None,
    q: int | None = None,
    r: int | None = None,
    verbose: bool = False,
) -> BuildResult:
    """
    Build a decahedral Ru cluster using ASE's Decahedron constructor.

    If p, q, r are not provided, the builder brute-forces a small parameter grid
    and picks the closest match by atom count or radius.
    """
    a_fcc = float(a_ref) * math.sqrt(2.0)

    def mk(pp: int, qq: int, rr: int):
        if int(pp) <= 0 or int(qq) <= 0:
            raise ValueError("Decahedron requires p and q to be greater than 0.")
        atoms = Decahedron("Ru", p=int(pp), q=int(qq), r=int(rr), latticeconstant=a_fcc)
        atoms.center(vacuum=10.0)
        return atoms

    if p is not None and q is not None and r is not None:
        atoms = mk(int(p), int(q), int(r))
        chosen = (int(p), int(q), int(r))
    else:
        best = None
        for pp in range(1, 11):
            for qq in range(1, 11):
                for rr in range(0, 11):
                    try:
                        cand = mk(pp, qq, rr)
                    except Exception:
                        continue
                    score = abs(len(cand) - int(target_atoms)) if target_atoms is not None else abs(_max_radius(cand) - float(target_radius))
                    if best is None or score < best[0]:
                        best = (score, (pp, qq, rr), cand)
        if best is None:
            raise RuntimeError("Could not build any decahedron candidate with the current parameter grid.")
        atoms = best[2]
        chosen = best[1]
        if verbose:
            print(f"[decahedron] selected p,q,r={chosen} n={len(atoms)} r={_max_radius(atoms):.3f} Å")

    _ensure_cell(atoms)
    stamp_cluster_metadata(
        atoms,
        tag=tag,
        shape="decahedron",
        crystal_structure="fcc",
        average_nn_distance=_average_nn_distance(atoms),
    )
    write_cluster_files(tag, atoms)
    output_dir = cluster_output_dir()

    r_final = _max_radius(atoms)
    meta = {
        "tag": tag,
        "shape": "decahedron",
        "a_fcc_used": a_fcc,
        "pqr": list(chosen),
        "n_atoms": len(atoms),
        "actual_radius": r_final,
        "formula": atoms.get_chemical_formula(empirical=True),
    }
    with open(output_dir / f"{tag}_decahedron_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return BuildResult(len(atoms), atoms.get_chemical_formula(empirical=True), surfaces_hkl=[], gammas=[], actual_radius=r_final)
