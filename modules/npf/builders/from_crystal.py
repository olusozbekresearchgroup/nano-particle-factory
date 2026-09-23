# Copyright (c) 2026 Olus Ozbek Research Group
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import numpy as np
from ase import Atoms

from ..common import (
    BuildResult,
    _ensure_cell,
    _max_radius,
    cluster_output_dir,
    stamp_cluster_metadata,
    write_cluster_files,
)


_CANDIDATE_CACHE: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
_MAX_CACHE_ENTRIES = 2
_MAX_CACHED_ATOMS = 250_000


def _validate_source(source: Atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell = np.asarray(source.cell.array, dtype=float)
    volume = abs(float(np.linalg.det(cell)))
    if volume <= 1e-12:
        raise ValueError("The source crystal must have a non-zero three-dimensional cell.")
    if len(source) == 0:
        raise ValueError("The source crystal contains no atoms.")

    scaled = np.asarray(source.get_scaled_positions(wrap=True), dtype=float)
    scaled -= scaled.mean(axis=0)
    return cell, scaled, np.asarray(source.numbers, dtype=int)


def _candidate_cache_key(
    cell: np.ndarray,
    scaled: np.ndarray,
    numbers: np.ndarray,
    bounds: np.ndarray,
) -> str:
    digest = hashlib.sha1()
    for values in (cell, scaled, numbers, bounds):
        digest.update(np.ascontiguousarray(np.round(values, 12)).tobytes())
    return digest.hexdigest()


def _periodic_candidates(source: Atoms, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate only periodic basis images inside a Cartesian bounding box."""
    cell, scaled, source_numbers = _validate_source(source)
    bounds = np.asarray(bounds, dtype=float)
    key = _candidate_cache_key(cell, scaled, source_numbers, bounds)
    cached = _CANDIDATE_CACHE.get(key)
    if cached is not None:
        _CANDIDATE_CACHE.move_to_end(key)
        return cached

    inverse_cell = np.linalg.inv(cell)
    fractional_limits = bounds @ np.abs(inverse_cell)
    translation_limits = np.ceil(fractional_limits + np.max(np.abs(scaled), axis=0)).astype(int) + 1
    ranges = [np.arange(-limit, limit + 1, dtype=int) for limit in translation_limits]

    # Iterate over the first lattice index to cap temporary array size.
    yz = np.stack(np.meshgrid(ranges[1], ranges[2], indexing="ij"), axis=-1).reshape(-1, 2)
    positions_parts: list[np.ndarray] = []
    number_parts: list[np.ndarray] = []
    chunk_numbers = np.tile(source_numbers, len(yz))
    tolerance = 1e-9
    for first in ranges[0]:
        translations = np.empty((len(yz), 3), dtype=float)
        translations[:, 0] = first
        translations[:, 1:] = yz
        positions = (translations[:, None, :] + scaled[None, :, :]).reshape(-1, 3) @ cell
        keep = np.all(np.abs(positions) <= bounds + tolerance, axis=1)
        if np.any(keep):
            positions_parts.append(positions[keep])
            number_parts.append(chunk_numbers[keep])

    if not positions_parts:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=int)
    positions = np.concatenate(positions_parts)
    numbers = np.concatenate(number_parts)

    if len(positions) <= _MAX_CACHED_ATOMS:
        _CANDIDATE_CACHE[key] = (positions, numbers)
        _CANDIDATE_CACHE.move_to_end(key)
        while len(_CANDIDATE_CACHE) > _MAX_CACHE_ENTRIES:
            _CANDIDATE_CACHE.popitem(last=False)
    return positions, numbers


def _hex_radius(positions: np.ndarray) -> np.ndarray:
    x = positions[:, 0]
    y = positions[:, 1]
    root3 = math.sqrt(3.0)
    across_flats = np.maximum.reduce(
        [np.abs(x), np.abs(0.5 * x + 0.5 * root3 * y), np.abs(0.5 * x - 0.5 * root3 * y)]
    )
    return across_flats * (2.0 / root3)


def _wulff_vectors(source: Atoms, surfaces: Mapping[tuple[int, int, int], float]) -> np.ndarray:
    if not surfaces:
        raise ValueError("Generic Wulff generation requires facet energies in 'surfaces'.")
    cell = np.asarray(source.cell.array, dtype=float)
    reciprocal = np.linalg.inv(cell).T
    energies = np.asarray([float(value) for value in surfaces.values()], dtype=float)
    if np.any(energies <= 0.0):
        raise ValueError("Wulff facet energies must be positive.")
    energy_scale = float(np.mean(energies))
    rotations = [np.eye(3)]
    try:
        import spglib

        symmetry = spglib.get_symmetry(
            (cell, source.get_scaled_positions(wrap=True), source.numbers),
            symprec=1e-3,
        )
        if symmetry is not None:
            inverse_cell = np.linalg.inv(cell)
            rotations = [
                inverse_cell @ np.asarray(rotation, dtype=float).T @ cell
                for rotation in symmetry["rotations"]
            ]
    except (ImportError, TypeError, ValueError):
        pass

    vectors = []
    unique_normals: set[tuple[float, float, float]] = set()
    for (h, k, l), energy in surfaces.items():
        normal = np.asarray([h, k, l], dtype=float) @ reciprocal
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            continue
        normal /= norm
        relative_energy = float(energy) / energy_scale
        for rotation in rotations:
            equivalent = normal @ rotation.T
            equivalent /= max(float(np.linalg.norm(equivalent)), 1e-12)
            key = tuple(float(value) for value in np.round(equivalent, 8))
            opposite = tuple(float(value) for value in np.round(-equivalent, 8))
            if key in unique_normals or opposite in unique_normals:
                continue
            unique_normals.add(key)
            vectors.append(equivalent / relative_energy)
    if not vectors:
        raise ValueError("No valid Wulff facet normals were supplied.")
    vector_matrix = np.asarray(vectors, dtype=float)
    if np.linalg.matrix_rank(vector_matrix) < 3:
        raise ValueError("Wulff facets and their symmetry equivalents do not bound a three-dimensional particle.")
    return vector_matrix


def _wulff_metric(positions: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return np.max(np.abs(positions @ vectors.T), axis=1)


def _shape_metric(
    positions: np.ndarray,
    shape: str,
    *,
    morph: float,
    source: Atoms,
    surfaces: Mapping[tuple[int, int, int], float] | None,
    wulff_vectors: np.ndarray | None = None,
) -> np.ndarray:
    absolute = np.abs(positions)
    radius = np.linalg.norm(positions, axis=1)
    max_axis = absolute.max(axis=1)
    l1 = absolute.sum(axis=1)

    if shape == "sphere":
        return radius
    if shape == "cube":
        return max_axis * math.sqrt(3.0)
    if shape == "octahedron":
        return l1
    if shape == "cuboct":
        return np.maximum(max_axis * math.sqrt(2.0), l1 / math.sqrt(2.0))
    if shape == "morphed_spherical":
        exponent = 2.0 + 8.0 * max(0.0, min(1.0, float(morph)))
        return np.power(np.power(absolute, exponent).sum(axis=1), 1.0 / exponent)
    if shape == "ico":
        return np.maximum(radius, l1 / math.sqrt(2.5))
    if shape == "dodecahedron":
        return np.maximum(radius, max_axis * 1.35)
    if shape == "decahedron":
        radial = np.linalg.norm(positions[:, :2], axis=1)
        return np.maximum(radial, absolute[:, 2] * 1.25)
    if shape == "wulff":
        vectors = wulff_vectors if wulff_vectors is not None else _wulff_vectors(source, surfaces or {})
        return _wulff_metric(positions, vectors)

    radial = _hex_radius(positions)
    z = absolute[:, 2]
    if shape == "hexagonal_prism":
        return np.maximum(radial, z)
    if shape == "truncated_hexagonal_prism":
        return np.maximum.reduce([radial, z, 0.72 * radial + 0.72 * z])
    if shape == "hexagonal_bipyramid":
        return radial + z
    if shape == "truncated_hexagonal_bipyramid":
        return np.maximum(radial + 0.72 * z, z)
    if shape == "nanorod":
        return np.maximum(radial, z / 2.0)
    if shape == "hexagonal_platelet":
        return np.maximum(radial, z * 3.0)
    raise ValueError(f"Unsupported crystal-driven shape: {shape}")


def _shape_bounds(shape: str, extent: float, wulff_vectors: np.ndarray | None) -> np.ndarray:
    if shape == "cube":
        factor = 1.0 / math.sqrt(3.0)
    elif shape == "cuboct":
        factor = 1.0 / math.sqrt(2.0)
    elif shape == "nanorod":
        return np.asarray([extent, extent, 2.0 * extent], dtype=float)
    elif shape == "hexagonal_platelet":
        return np.asarray([extent, extent, extent / 3.0], dtype=float)
    elif shape == "decahedron":
        return np.asarray([extent, extent, 0.8 * extent], dtype=float)
    elif shape == "wulff":
        if wulff_vectors is None:
            raise ValueError("Wulff vectors were not initialized.")
        # p = pinv(M) @ (M @ p) gives a conservative axis bound for metric <= extent.
        factor = float(np.max(np.sum(np.abs(np.linalg.pinv(wulff_vectors)), axis=1)))
    else:
        factor = 1.0
    return np.full(3, extent * factor, dtype=float)


def _estimate_extent(source: Atoms, target_atoms: int | None, target_radius: float | None) -> float:
    if target_radius is not None:
        return float(target_radius)
    volume_per_atom = abs(float(source.cell.volume)) / max(len(source), 1)
    sphere_radius = (3.0 * max(int(target_atoms or 1), 1) * volume_per_atom / (4.0 * math.pi)) ** (1.0 / 3.0)
    return max(sphere_radius * 1.5, max(source.cell.lengths()))


def _cut_by_target(
    numbers: np.ndarray,
    positions: np.ndarray,
    metric: np.ndarray,
    target_atoms: int | None,
    target_radius: float | None,
) -> Atoms:
    if target_radius is not None:
        radius = float(target_radius)
        radial_distance = np.linalg.norm(positions, axis=1)
        keep = (metric <= radius + 1e-9) & (radial_distance <= radius + 1e-9)
    else:
        count = max(1, int(target_atoms or 1))
        if count > len(positions):
            raise ValueError("The generated crystal seed is too small for the requested atom count.")
        threshold = float(np.partition(metric, count - 1)[count - 1])
        tolerance = max(1e-9, abs(threshold) * 1e-10)
        keep = metric <= threshold + tolerance
    cluster = Atoms(numbers=numbers[keep], positions=positions[keep], pbc=False)
    if len(cluster) == 0:
        raise ValueError("The requested size produced an empty nanoparticle.")
    cluster.set_pbc(False)
    if target_radius is not None:
        limit = float(target_radius) + 1e-9
        for _ in range(6):
            positions = cluster.get_positions()
            cluster.set_positions(positions - positions.mean(axis=0))
            radii = np.linalg.norm(cluster.get_positions(), axis=1)
            if float(radii.max()) <= limit:
                break
            cluster = cluster[radii <= limit]
            if len(cluster) == 0:
                raise ValueError("Radius centering removed all atoms from the nanoparticle.")
    return cluster


def build_from_crystal(
    source: Atoms,
    *,
    source_file: str,
    shape: str,
    tag: str,
    crystal_structure: str | None,
    crystal_variant: str | None,
    target_atoms: int | None,
    target_radius: float | None,
    surfaces: Mapping[tuple[int, int, int], float] | None = None,
    morph: float = 0.0,
    radial_trim: float | None = None,
    verbose: bool = False,
) -> BuildResult:
    if (target_atoms is None) == (target_radius is None):
        raise ValueError("Provide exactly one of atoms or radius.")
    if target_atoms is not None and int(target_atoms) < 1:
        raise ValueError("The requested atom count must be positive.")
    if target_radius is not None and float(target_radius) <= 0.0:
        raise ValueError("The requested radius must be positive.")
    started = time.perf_counter()
    wulff_vectors = _wulff_vectors(source, surfaces or {}) if shape == "wulff" else None
    extent = _estimate_extent(source, target_atoms, target_radius)
    candidate_count = 0
    for _attempt in range(8):
        bounds = (
            np.minimum(
                _shape_bounds(shape, float(target_radius), wulff_vectors),
                float(target_radius),
            )
            if target_radius is not None
            else _shape_bounds(shape, extent, wulff_vectors)
        )
        if verbose:
            dimensions = " x ".join(f"{2.0 * value:.1f}" for value in bounds)
            print(f"[{shape}] generating periodic sites in {dimensions} A bounds", flush=True)
        positions, numbers = _periodic_candidates(source, bounds)
        metric = _shape_metric(
            positions,
            shape,
            morph=morph,
            source=source,
            surfaces=surfaces,
            wulff_vectors=wulff_vectors,
        )
        keep = metric <= extent + 1e-9
        if target_radius is not None:
            keep &= np.linalg.norm(positions, axis=1) <= float(target_radius) + 1e-9
        positions = positions[keep]
        numbers = numbers[keep]
        metric = metric[keep]
        candidate_count = len(positions)
        if target_radius is not None or candidate_count >= max(1, int(target_atoms or 1)):
            break
        extent *= 1.5
        if verbose:
            print(f"[{shape}] expanding candidate metric bound to {extent:.3f} A")
    else:
        raise ValueError("Could not generate enough crystal sites for the requested atom count.")

    atoms = _cut_by_target(numbers, positions, metric, target_atoms, target_radius)

    if radial_trim is not None:
        positions = atoms.get_positions()
        keep = np.linalg.norm(positions - positions.mean(axis=0), axis=1) <= float(radial_trim) + 1e-9
        atoms = atoms[keep]
        if len(atoms) == 0:
            raise ValueError("Radial trim removed all atoms.")

    atoms.set_positions(atoms.get_positions() - atoms.get_positions().mean(axis=0))
    _ensure_cell(atoms)
    stamp_cluster_metadata(
        atoms,
        tag=tag,
        shape=shape,
        crystal_structure=crystal_structure,
        crystal_variant=crystal_variant,
        source_crystal=str(source_file),
    )
    write_cluster_files(tag, atoms)
    actual_radius = _max_radius(atoms)
    metadata = {
        "tag": tag,
        "shape": shape,
        "source_crystal": str(source_file),
        "source_formula": source.get_chemical_formula(empirical=True),
        "source_species": list(dict.fromkeys(source.get_chemical_symbols())),
        "crystal_structure": crystal_structure,
        "crystal_variant": crystal_variant,
        "target_atoms": target_atoms,
        "target_radius": target_radius,
        "n_atoms": len(atoms),
        "actual_radius": actual_radius,
        "formula": atoms.get_chemical_formula(empirical=True),
        "surfaces_hkl": [list(indices) for indices in (surfaces or {})],
        "surface_energies": [float(value) for value in (surfaces or {}).values()],
    }
    output_dir = cluster_output_dir()
    with open(output_dir / f"{tag}_{shape}_meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    if verbose:
        elapsed = time.perf_counter() - started
        print(
            f"[{shape}] source={Path(source_file).name} candidates={candidate_count} "
            f"cluster_atoms={len(atoms)} radius={actual_radius:.3f} A elapsed={elapsed:.3f}s"
        )
    return BuildResult(
        len(atoms),
        atoms.get_chemical_formula(empirical=True),
        list(surfaces or {}) if shape == "wulff" else [],
        [float(value) for value in (surfaces or {}).values()] if shape == "wulff" else [],
        actual_radius,
    )
