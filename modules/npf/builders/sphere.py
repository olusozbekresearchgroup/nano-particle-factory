from __future__ import annotations

import json
import math

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write

from ..common import BuildResult, _average_nn_distance, _ensure_cell, _max_radius, cluster_output_dir, stamp_cluster_metadata, write_cluster_files


def _normalize_structure(value: str | None) -> str:
    structure = (value or "hexagonal").strip().lower()
    return {
        "hcp": "hexagonal",
        "fcc": "cubic",
        "bcc": "cubic",
        "trigonal": "rhombohedral",
        "custom": "triclinic",
    }.get(structure, structure)


def _normalize_variant(value: str | None) -> str | None:
    if value is None:
        return None
    variant = value.strip().lower()
    return variant or None


def _bulk_spec(
    crystal_structure: str | None,
    crystal_variant: str | None,
    a: float,
    c: float | None,
):
    """
    Map the GUI's structure/variant selection to an ASE bulk constructor.

    The sphere geometry needs an actual lattice-filled cube, so we build a
    conventional bulk cell and repeat it rather than using cluster surface
    constructors.
    """

    structure = _normalize_structure(crystal_structure)
    variant = _normalize_variant(crystal_variant)
    a = float(a)
    c = float(c if c is not None else a)

    if variant in {"simple_cubic", "sc"}:
        return "sc", {"a": a, "cubic": True}, variant
    if variant == "bcc":
        return "bcc", {"a": a, "cubic": True}, variant
    if variant == "fcc":
        return "fcc", {"a": a, "cubic": True}, variant

    if structure == "hexagonal":
        return "hcp", {"a": a, "c": c, "orthorhombic": True}, variant or "hcp"

    if structure == "cubic":
        # The GUI can expose many cubic subtypes, but the sphere builder only
        # needs a lattice-preserving bulk seed. Simple cubic is the safest
        # default when no explicit cubic subtype is provided.
        return "sc", {"a": a, "cubic": True}, variant or "simple_cubic"

    # Fallback for low-symmetry selections.
    return "hcp", {"a": a, "c": c, "orthorhombic": True}, variant or structure


def _build_bulk_seed(
    crystal_structure: str | None,
    crystal_variant: str | None,
    a: float,
    c: float | None,
):
    crystal_kind, kwargs, label = _bulk_spec(crystal_structure, crystal_variant, a, c)
    try:
        return bulk("Ru", crystal_kind, **kwargs), label, crystal_kind
    except Exception:
        # If ASE does not support the requested style for this element, fall
        # back to the closest structure family rather than failing the build.
        if crystal_kind == "hcp":
            return bulk("Ru", "sc", a=float(a), cubic=True), "simple_cubic", "sc"
        return bulk("Ru", "hcp", a=float(a), c=float(c if c is not None else a), orthorhombic=True), "hcp", "hcp"


def _center_positions_about_bbox(atoms: Atoms) -> Atoms:
    pos = atoms.get_positions()
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    shift = (mins + maxs) / 2.0
    centered = atoms.copy()
    centered.set_positions(pos - shift)
    return centered


def _repeat_counts_for_side(seed: Atoms, side: float) -> tuple[int, int, int]:
    lengths = np.linalg.norm(seed.cell.array, axis=1)
    counts = []
    for length in lengths:
        if length <= 0:
            counts.append(1)
        else:
            counts.append(max(1, int(math.ceil(float(side) / float(length))) + 2))
    return tuple(counts)  # type: ignore[return-value]


def _build_cube_seed_from_bulk(
    side: float,
    crystal_structure: str | None,
    crystal_variant: str | None,
    a: float,
    c: float | None,
):
    base, seed_label, bulk_kind = _build_bulk_seed(crystal_structure, crystal_variant, a, c)
    repeats = _repeat_counts_for_side(base, side)
    supercell = base.repeat(repeats)
    supercell = _center_positions_about_bbox(supercell)

    half = float(side) / 2.0
    pos = supercell.get_positions()
    keep = np.all(np.abs(pos) <= half + 1e-9, axis=1)
    cube = supercell[keep]
    if len(cube) == 0:
        raise ValueError("Cube seeding removed all atoms; choose a larger side length.")

    cube = Atoms(
        symbols=cube.get_chemical_symbols(),
        positions=cube.get_positions(),
        cell=cube.cell,
        pbc=cube.pbc,
    )
    return cube, seed_label, bulk_kind, repeats


def _shave_sphere_from_seed(seed: Atoms, radius: float) -> Atoms:
    pos = seed.get_positions()
    r = np.sqrt((pos**2).sum(axis=1))
    keep = r <= float(radius) + 1e-9
    cut = seed[keep]
    if len(cut) == 0:
        raise ValueError("Sphere shave removed all atoms; choose a larger seed cube or radius.")
    cut = Atoms(
        symbols=cut.get_chemical_symbols(),
        positions=cut.get_positions(),
        cell=cut.cell,
        pbc=cut.pbc,
    )
    cut.center(vacuum=10.0)
    return cut


def _spherical_cut_counts(seed: Atoms):
    pos = seed.get_positions()
    radii = np.sqrt((pos**2).sum(axis=1))
    unique_radii = np.unique(np.round(radii, 12))
    counts = []
    for radius in unique_radii:
        counts.append((int(np.count_nonzero(radii <= radius + 1e-9)), float(radius)))
    counts.sort(key=lambda item: (item[0], item[1]))
    return counts


def _best_radius_for_target_atoms(seed: Atoms, target_atoms: int) -> float:
    counts = _spherical_cut_counts(seed)
    target_atoms = int(target_atoms)
    exact = [radius for count, radius in counts if count == target_atoms]
    if exact:
        return float(exact[0])
    _, best_radius = min(counts, key=lambda item: (abs(item[0] - target_atoms), item[1]))
    return float(best_radius)


def _estimate_radius_for_target_atoms(
    target_atoms: int,
    crystal_structure: str | None,
    crystal_variant: str | None,
    a: float,
    c: float | None,
) -> float:
    base, _, _ = _build_bulk_seed(crystal_structure, crystal_variant, a, c)
    volume_per_atom = abs(float(base.cell.volume)) / max(len(base), 1)
    # Match the sphere volume to the target atom count as a coarse starting point.
    return float((3.0 * float(target_atoms) * volume_per_atom / (4.0 * math.pi)) ** (1.0 / 3.0))


def _build_sphere_from_cube_seed(
    side: float,
    radius: float,
    crystal_structure: str | None,
    crystal_variant: str | None,
    a: float,
    c: float | None,
):
    seed, seed_label, bulk_kind, repeats = _build_cube_seed_from_bulk(
        side=side,
        crystal_structure=crystal_structure,
        crystal_variant=crystal_variant,
        a=a,
        c=c,
    )
    atoms = _shave_sphere_from_seed(seed, radius)
    return atoms, seed, seed_label, bulk_kind, repeats


def build_ru_sphere(
    a: float = 2.706,
    b: float | None = None,
    c: float | None = 4.282,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    crystal_structure: str = "hexagonal",
    crystal_variant: str | None = None,
    tag: str = "Ru_sphere",
    target_atoms: int | None = None,
    target_radius: float | None = None,
    verbose: bool = False,
) -> BuildResult:
    """
    Build a sphere by:
    1. creating a cube with side length equal to the target diameter,
    2. filling the cube with a lattice that matches the chosen crystal structure,
    3. trimming atoms farther than radius from the cube center.
    """

    if (target_atoms is None) == (target_radius is None):
        raise ValueError("Provide exactly one of --atoms or --radius.")

    if target_radius is not None:
        radius = float(target_radius)
        side = 2.0 * radius
        atoms, seed, seed_label, bulk_kind, repeats = _build_sphere_from_cube_seed(
            side=side,
            radius=radius,
            crystal_structure=crystal_structure,
            crystal_variant=crystal_variant,
            a=a,
            c=c,
        )
    else:
        target_atoms = int(target_atoms)
        radius_est = _estimate_radius_for_target_atoms(target_atoms, crystal_structure, crystal_variant, a, c)
        # Try a few progressively larger cubes so the spherical trim can land on
        # a discrete atom count close to the target.
        best = None
        for scale in (1.4, 1.7, 2.0, 2.4, 2.8):
            side = max(2.0 * radius_est * scale, 2.0 * float(a))
            seed, seed_label, bulk_kind, repeats = _build_cube_seed_from_bulk(
                side=side,
                crystal_structure=crystal_structure,
                crystal_variant=crystal_variant,
                a=a,
                c=c,
            )
            radius = _best_radius_for_target_atoms(seed, target_atoms)
            atoms = _shave_sphere_from_seed(seed, radius)
            delta = abs(len(atoms) - target_atoms)
            candidate = (delta, side, radius, atoms, seed, seed_label, bulk_kind, repeats)
            if best is None or delta < best[0]:
                best = candidate
            if delta == 0:
                break
        assert best is not None
        _, side, radius, atoms, seed, seed_label, bulk_kind, repeats = best

    seed_atoms = len(seed)
    r_final = _max_radius(atoms)

    if verbose:
        print(
            f"[sphere] seed={seed_label} bulk={bulk_kind} repeats={repeats} "
            f"seed_atoms={seed_atoms} sphere_radius={radius:.3f} Å remaining_atoms={len(atoms)}"
        )

    _ensure_cell(atoms)
    stamp_cluster_metadata(
        atoms,
        tag=tag,
        shape="sphere",
        crystal_structure=_normalize_structure(crystal_structure),
        crystal_variant=_normalize_variant(crystal_variant),
        average_nn_distance=_average_nn_distance(atoms),
    )
    write_cluster_files(tag, atoms)
    output_dir = cluster_output_dir()

    meta = {
        "tag": tag,
        "shape": "sphere",
        "seed_structure": _normalize_structure(crystal_structure),
        "seed_variant": _normalize_variant(crystal_variant),
        "seed_factory": seed_label,
        "bulk_kind": bulk_kind,
        "bulk_repeats": repeats,
        "cube_side_length": 2.0 * float(target_radius) if target_radius is not None else side,
        "seed_atoms_before_shave": seed_atoms,
        "sphere_radius_seed": radius,
        "n_atoms": len(atoms),
        "actual_radius": r_final,
        "formula": atoms.get_chemical_formula(empirical=True),
        "crystal_structure": crystal_structure,
        "crystal_variant": crystal_variant,
        "lattice": {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma},
    }
    with open(output_dir / f"{tag}_sphere_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return BuildResult(
        len(atoms),
        atoms.get_chemical_formula(empirical=True),
        surfaces_hkl=[],
        gammas=[],
        actual_radius=r_final,
    )
