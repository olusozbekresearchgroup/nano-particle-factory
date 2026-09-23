from __future__ import annotations

import json
from typing import Dict, List, Tuple

from ase.io import write

from ..common import (
    BuildResult,
    _average_nn_distance,
    DEFAULT_A,
    DEFAULT_C,
    _apply_radial_trim,
    _ensure_cell,
    _max_radius,
    cluster_output_dir,
    format_hkil,
    _try_build,
    crystalium_ru_gammas_hkl,
    stamp_cluster_metadata,
    write_cluster_files,
)


def build_ru_wulff(
    target_atoms: int | None = None,
    target_radius: float | None = None,
    a: float = DEFAULT_A,
    c: float = DEFAULT_C,
    overrides: Dict[Tuple[int, int, int], float] | None = None,
    tag: str = "Ru_Wulff",
    facet_mode: str = "top",
    radius_tol: float = 0.3,
    max_refine: int = 12,
    verbose: bool = False,
    radius_policy: str = "closest",
    sphericity: float = 0.0,
    extra_facets: str = "none",
    radial_trim: float | None = None,
) -> BuildResult:
    if (target_atoms is None) == (target_radius is None):
        raise ValueError("Provide exactly one of --atoms or --radius.")

    base = crystalium_ru_gammas_hkl()
    if overrides:
        base.update(overrides)

    if extra_facets == "auto":
        candidate = set(base.keys())
        for h in [0, 1, 2]:
            for k in [0, 1, 2]:
                for l in [1, 2]:
                    candidate.add((h, k, l))
        known = list(base.items())

        def nearest_gamma(hkl):
            h, k, l = hkl
            bestg, bestd = None, 1e9
            for (hh, kk, ll), gg in known:
                d = abs(h - hh) + abs(k - kk) + abs(l - ll)
                if d < bestd:
                    bestd, bestg = d, gg
            return bestg

        for hkl in candidate:
            if hkl not in base:
                base[hkl] = nearest_gamma(hkl)

    if not (0.0 <= sphericity <= 1.0):
        raise ValueError("--sphericity must be in [0,1]")
    if sphericity > 0:
        gmean = float(sum(base.values()) / len(base))
        alpha = sphericity
        for k in list(base.keys()):
            base[k] = (1.0 - alpha) * base[k] + alpha * gmean

    facet_sets = {
        "top": [(0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 0, 1)],
        "full": sorted(base.keys()),
        "minimal": [(0, 0, 1), (1, 0, 0)],
    }
    if facet_mode not in facet_sets:
        raise ValueError("facet_mode must be one of: top, full, minimal")

    try_order = [facet_mode] + (
        ["minimal"] if facet_mode == "top" else (["top", "minimal"] if facet_mode == "full" else [])
    )

    last_err = None
    atoms = None
    used_surfaces: List[Tuple[int, int, int]] = []
    used_energies: List[float] = []

    def build_with_surfaces(slist, size_value_local):
        return _try_build(slist, [base[s] for s in slist], size_value_local, a, c)

    if target_atoms is not None:
        size_value = int(target_atoms)
        for mode in try_order:
            slist = [s for s in facet_sets[mode] if s in base]
            try:
                atoms = build_with_surfaces(slist, size_value)
                used_surfaces, used_energies = slist, [base[s] for s in slist]
                break
            except Exception as e:
                last_err = e
                continue
        if atoms is None:
            raise RuntimeError(f"ASE failed to build Wulff cluster. Last error: {last_err}")
    else:
        target = float(target_radius)
        size_value = max(1.0, target)

        def build_with_mode_list(size_param):
            nonlocal used_surfaces, used_energies
            for mode in try_order:
                slist = [s for s in facet_sets[mode] if s in base]
                try:
                    atoms_local = _try_build(slist, [base[s] for s in slist], size_param, a, c)
                    used_surfaces, used_energies = slist, [base[s] for s in slist]
                    return atoms_local
                except Exception:
                    continue
            return None

        atoms = build_with_mode_list(size_value)
        if atoms is None:
            raise RuntimeError("ASE failed to build Wulff at initial guess.")
        r_act = _max_radius(atoms)
        if verbose:
            print(f"[refine] init size={size_value:.3f} -> r_act={r_act:.3f}Å (target={target:.3f}Å)")

        best_atoms, best_r, best_size = atoms, r_act, size_value
        lo_s, lo_r, lo_atoms = size_value, r_act, atoms
        hi_s = hi_r = None
        hi_atoms = None

        grows = 0
        while lo_r < target and grows < max_refine:
            size_value *= 2.0
            atoms = build_with_mode_list(size_value)
            if atoms is None:
                break
            r_new = _max_radius(atoms)
            if verbose:
                print(f"[refine] grow ×2 size={size_value:.3f} -> r_act={r_new:.3f}Å")
            if r_new > best_r:
                best_r, best_atoms, best_size = r_new, atoms, size_value
            lo_s, lo_r, lo_atoms = size_value, r_new, atoms
            grows += 1
            if r_new >= target:
                hi_s, hi_r, hi_atoms = size_value, r_new, atoms
                break

        built, last_r = False, best_r
        if hi_s is None:
            if verbose:
                print(f"[refine] unable to bracket; best radius={best_r:.3f}Å at size={best_size:.3f}")
            atoms, last_r, built = best_atoms, best_r, (abs(best_r - target) <= radius_tol or radius_policy in ("closest", "at_most"))
        else:
            for i in range(max_refine):
                mid = 0.5 * (lo_s + hi_s)
                atoms_mid = build_with_mode_list(mid)
                if atoms_mid is None:
                    break
                r_mid = _max_radius(atoms_mid)
                if verbose:
                    print(f"[refine] bisect i={i} size={mid:.3f} -> r_act={r_mid:.3f}Å")
                if abs(r_mid - target) <= radius_tol:
                    atoms, last_r, built = atoms_mid, r_mid, True
                    break
                if abs(r_mid - target) < abs(best_r - target):
                    best_r, best_atoms, best_size = r_mid, atoms_mid, mid
                if r_mid < target:
                    lo_s, lo_r, lo_atoms = mid, r_mid, atoms_mid
                else:
                    hi_s, hi_r, hi_atoms = mid, r_mid, atoms_mid
            if not built:
                if radius_policy == "closest":
                    if hi_r is None or abs(lo_r - target) <= abs(hi_r - target):
                        atoms, last_r = lo_atoms, lo_r
                    else:
                        atoms, last_r = hi_atoms, hi_r
                elif radius_policy == "at_least":
                    atoms, last_r = hi_atoms, hi_r
                else:
                    atoms, last_r = lo_atoms, lo_r
                built = True
        if not built:
            raise RuntimeError(f"Could not refine to target radius (last {last_r:.3f} Å vs {target:.3f} Å).")

    if radial_trim is not None:
        atoms = _apply_radial_trim(atoms, float(radial_trim))

    _ensure_cell(atoms)
    stamp_cluster_metadata(
        atoms,
        tag=tag,
        shape="wulff",
        crystal_structure="hcp",
        average_nn_distance=_average_nn_distance(atoms),
    )
    write_cluster_files(tag, atoms)
    output_dir = cluster_output_dir()

    r_final = _max_radius(atoms)
    meta = {
        "tag": tag,
        "lattice": {"a": a, "c": c},
        "mode": "atoms" if target_atoms is not None else "radius",
        "target_value": int(target_atoms) if target_atoms is not None else float(target_radius),
        "actual_radius": r_final,
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(empirical=True),
        "surfaces_hkl": [list(hkl) for hkl in used_surfaces],
        "surfaces_hkil": [list((h, k, -(h + k), l)) for (h, k, l) in used_surfaces],
        "facet_labels_hkil": [format_hkil(h, k, l) for (h, k, l) in used_surfaces],
        "gammas_eV_per_A2": used_energies,
        "sphericity": sphericity,
        "extra_facets": extra_facets,
        "radial_trim": radial_trim,
    }
    with open(output_dir / f"{tag}_wulff_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return BuildResult(len(atoms), atoms.get_chemical_formula(empirical=True), used_surfaces, used_energies, r_final)
