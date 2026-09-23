from __future__ import annotations

import json
import math

from ase.cluster.wulff import wulff_construction
from ase.io import write

from ..common import BuildResult, _average_nn_distance, _ensure_cell, _max_radius, cluster_output_dir, stamp_cluster_metadata, write_cluster_files


def build_ru_dodecahedron(
    a_ref: float = 2.706,
    tag: str = "Ru_dodecahedron",
    target_atoms: int | None = None,
    target_radius: float | None = None,
    verbose: bool = False,
) -> BuildResult:
    """
    Build a rhombic-dodecahedron-like FCC cluster using a one-facet Wulff construction.

    ASE does not expose a dedicated dodecahedron constructor, so this uses the FCC
    Wulff shape with the {110} family as a practical approximation.
    """
    a_fcc = float(a_ref) * math.sqrt(2.0)
    surfaces = [(1, 1, 0)]
    energies = [1.0]

    def build_from_size(size_value: int):
        atoms = wulff_construction(
            symbol="Ru",
            surfaces=surfaces,
            energies=energies,
            size=int(size_value),
            structure="fcc",
            latticeconstant=a_fcc,
            rounding="above",
        )
        atoms.center(vacuum=10.0)
        return atoms

    if target_atoms is not None:
        atoms = build_from_size(int(target_atoms))
    else:
        target = float(target_radius)
        rho = 4.0 / (a_fcc**3)
        guess = max(1, int(round((4.0 / 3.0 * math.pi * target**3) * rho)))
        best = None
        size_value = guess
        for _ in range(10):
            atoms = build_from_size(size_value)
            r_act = _max_radius(atoms)
            score = abs(r_act - target)
            if best is None or score < best[0]:
                best = (score, size_value, atoms, r_act)
            if score < 0.25:
                break
            if len(atoms) == 0:
                break
            size_value = max(1, int(round(size_value * (target / max(r_act, 1e-9)) ** 3)))
        atoms = best[2]
        if verbose:
            print(f"[dodecahedron] selected size={best[1]} n={len(atoms)} r={best[3]:.3f} Å")

    _ensure_cell(atoms)
    stamp_cluster_metadata(
        atoms,
        tag=tag,
        shape="dodecahedron",
        crystal_structure="fcc",
        average_nn_distance=_average_nn_distance(atoms),
    )
    write_cluster_files(tag, atoms)
    output_dir = cluster_output_dir()

    r_final = _max_radius(atoms)
    meta = {
        "tag": tag,
        "shape": "dodecahedron",
        "note": "FCC rhombic-dodecahedron-like approximation via Wulff {110}",
        "a_fcc_used": a_fcc,
        "n_atoms": len(atoms),
        "actual_radius": r_final,
        "formula": atoms.get_chemical_formula(empirical=True),
    }
    with open(output_dir / f"{tag}_dodecahedron_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return BuildResult(
        len(atoms),
        atoms.get_chemical_formula(empirical=True),
        surfaces_hkl=surfaces,
        gammas=energies,
        actual_radius=r_final,
    )
