from __future__ import annotations

import json

from ase.cluster.hexagonal import HexagonalClosedPacked

from ..common import BuildResult, DEFAULT_A, DEFAULT_C, _average_nn_distance, _ensure_cell, _max_radius, cluster_output_dir, stamp_cluster_metadata, write_cluster_files


_HEX_SURFACES = {
    "hexagonal_prism": [
        (0, 0, 0, 1),
        (1, 0, -1, 0),
    ],
    "truncated_hexagonal_prism": [
        (0, 0, 0, 1),
        (1, 0, -1, 0),
        (1, 0, -1, 1),
    ],
    "hexagonal_bipyramid": [
        (1, 0, -1, 1),
    ],
    "truncated_hexagonal_bipyramid": [
        (1, 0, -1, 1),
        (0, 0, 0, 1),
    ],
    "nanorod": [
        (0, 0, 0, 1),
        (1, 0, -1, 0),
    ],
    "hexagonal_platelet": [
        (0, 0, 0, 1),
        (1, 0, -1, 0),
    ],
}


def _shape_layers(shape: str, n: int) -> list[int]:
    n = max(1, int(n))
    if shape == "hexagonal_prism":
        return [n, n]
    if shape == "truncated_hexagonal_prism":
        return [n, n, max(1, n // 3)]
    if shape == "hexagonal_bipyramid":
        return [n]
    if shape == "truncated_hexagonal_bipyramid":
        return [n, max(1, n // 3)]
    if shape == "nanorod":
        return [max(2, n * 2), n]
    if shape == "hexagonal_platelet":
        return [max(1, n // 2), n]
    raise ValueError(f"Unsupported hexagonal shape: {shape}")


def _build_candidate(shape: str, n: int, a_ref: float, c_ref: float):
    surfaces = _HEX_SURFACES[shape]
    layers = _shape_layers(shape, n)
    atoms = HexagonalClosedPacked("Ru", surfaces=surfaces, layers=layers, latticeconstant=(float(a_ref), float(c_ref)))
    _ensure_cell(atoms)
    return atoms


def _score_candidate(atoms, target_atoms: int | None, target_radius: float | None) -> float:
    if target_atoms is not None:
        return float(abs(len(atoms) - int(target_atoms)))
    return float(abs(_max_radius(atoms) - float(target_radius)))


def build_ru_hexagonal_shape(
    shape: str,
    a_ref: float = DEFAULT_A,
    c_ref: float = DEFAULT_C,
    tag: str = "Ru_hexagonal",
    target_atoms: int | None = None,
    target_radius: float | None = None,
    verbose: bool = False,
) -> BuildResult:
    if (target_atoms is None) == (target_radius is None):
        raise ValueError("Provide exactly one of --atoms or --radius.")
    if shape not in _HEX_SURFACES:
        raise ValueError(f"Unsupported hexagonal shape: {shape}")

    best = None
    for n in range(1, 31):
        try:
            cand = _build_candidate(shape, n, a_ref, c_ref)
        except Exception:
            continue
        score = _score_candidate(cand, target_atoms, target_radius)
        if best is None or score < best[0]:
            best = (score, n, cand)

    if best is None:
        raise RuntimeError(f"Could not build any candidate for hexagonal shape '{shape}'.")

    _, chosen_n, atoms = best
    if verbose:
        print(f"[{shape}] selected layers={chosen_n} n={len(atoms)} r={_max_radius(atoms):.3f} Å")

    _ensure_cell(atoms)
    stamp_cluster_metadata(
        atoms,
        tag=tag,
        shape=shape,
        crystal_structure="hcp",
        average_nn_distance=_average_nn_distance(atoms),
    )
    write_cluster_files(tag, atoms)
    output_dir = cluster_output_dir()

    r_final = _max_radius(atoms)
    meta = {
        "tag": tag,
        "shape": shape,
        "lattice": {"a": float(a_ref), "c": float(c_ref)},
        "selected_layers": chosen_n,
        "layer_profile": _shape_layers(shape, chosen_n),
        "n_atoms": len(atoms),
        "actual_radius": r_final,
        "formula": atoms.get_chemical_formula(empirical=True),
    }
    with open(output_dir / f"{tag}_{shape}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return BuildResult(len(atoms), atoms.get_chemical_formula(empirical=True), surfaces_hkl=[], gammas=[], actual_radius=r_final)
