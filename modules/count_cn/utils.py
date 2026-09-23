# Copyright (c) 2026 Olus Ozbek Research Group
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.

import math
from dataclasses import dataclass

import numpy as np
from numpy.linalg import eig
from ase.neighborlist import neighbor_list


@dataclass(frozen=True)
class NeighborGraph:
    n_atoms: int
    r_max: float
    i: np.ndarray
    j: np.ndarray
    distances: np.ndarray

    def mask_within(self, cutoff: float) -> np.ndarray:
        if float(cutoff) > self.r_max + 1e-9:
            raise ValueError(
                f"Neighbor graph cutoff {self.r_max:.3f} A is smaller than "
                f"the requested cutoff {float(cutoff):.3f} A."
            )
        return self.distances <= float(cutoff) + 1e-12


_TREE_MIN_ATOMS = 10_000


def _build_tree_neighbor_graph(atoms, cutoff: float, chunk_size: int) -> NeighborGraph:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError(
            "Large nonperiodic neighbor searches require scipy. Install scipy "
            "in the Python environment used to run NPF."
        ) from exc

    positions = np.asarray(atoms.positions, dtype=float)
    tree = cKDTree(positions)
    # Slightly widen only the candidate search, then apply ASE's exact strict
    # cutoff using the same distance formula. This avoids boundary rounding loss.
    radius = cutoff + max(1.0, cutoff) * 1e-12
    lengths = tree.query_ball_point(positions, radius, return_length=True)
    capacity = (int(lengths.sum()) - len(positions)) // 2
    first = np.empty(capacity, dtype=int)
    second = np.empty(capacity, dtype=int)
    distances = np.empty(capacity, dtype=float)
    offset = 0
    # Count-only preallocation avoids retaining and concatenating pair chunks.
    # Python neighbor lists and displacement arrays exist for one chunk only.
    for start in range(0, len(positions), chunk_size):
        stop = min(start + chunk_size, len(positions))
        rows = tree.query_ball_point(positions[start:stop], radius, return_sorted=True)
        counts = np.fromiter((len(row) for row in rows), dtype=int, count=len(rows))
        i = np.repeat(np.arange(start, stop), counts)
        j = np.fromiter((index for row in rows for index in row), dtype=int,
                        count=int(counts.sum()))
        keep = i < j
        i, j = i[keep], j[keep]
        delta = positions[j] - positions[i]
        d = np.sqrt(np.sum(delta * delta, axis=1))
        keep = d < cutoff
        size = int(keep.sum())
        first[offset:offset + size] = i[keep]
        second[offset:offset + size] = j[keep]
        distances[offset:offset + size] = d[keep]
        offset += size
    return NeighborGraph(len(atoms), cutoff, first[:offset], second[:offset], distances[:offset])


def build_neighbor_graph(
    atoms, r_max: float, *, backend: str = "auto", chunk_size: int = 2048,
) -> NeighborGraph:
    """Build unique i<j pairs with distance < r_max.

    Auto retains ASE for small or periodic structures. Large nonperiodic
    particles use an exact, chunked spatial-tree search to avoid ASE's large
    temporary bin-pair arrays. Explicit backends support equivalence testing.
    Periodic image semantics are preserved by always using ASE for PBC.
    """
    cutoff = float(r_max)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("Neighbor cutoff must be finite and positive.")
    if backend not in {"auto", "ase", "kdtree"}:
        raise ValueError("Neighbor backend must be auto, ase, or kdtree.")
    periodic = bool(np.any(atoms.pbc))
    if backend == "kdtree" and periodic:
        raise ValueError("The kdtree backend requires nonperiodic atoms; use ase for PBC.")
    use_tree = backend == "kdtree" or (
        backend == "auto" and not periodic and len(atoms) >= _TREE_MIN_ATOMS
    )
    if use_tree:
        if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
            raise ValueError("Neighbor chunk_size must be a positive integer.")
        return _build_tree_neighbor_graph(atoms, cutoff, chunk_size)
    i, j, distances = neighbor_list("ijd", atoms, cutoff)
    unique = i < j
    return NeighborGraph(
        n_atoms=len(atoms),
        r_max=cutoff,
        i=np.asarray(i[unique], dtype=int),
        j=np.asarray(j[unique], dtype=int),
        distances=np.asarray(distances[unique], dtype=float),
    )


def coordination_numbers(graph: NeighborGraph, cutoff: float) -> np.ndarray:
    keep = graph.mask_within(cutoff)
    indices = np.concatenate((graph.i[keep], graph.j[keep]))
    return np.bincount(indices, minlength=graph.n_atoms).astype(int)


def nearest_neighbor_distances(graph: NeighborGraph) -> np.ndarray:
    nearest = np.full(graph.n_atoms, np.inf, dtype=float)
    np.minimum.at(nearest, graph.i, graph.distances)
    np.minimum.at(nearest, graph.j, graph.distances)
    return nearest


def average_nearest_distance(graph: NeighborGraph) -> float:
    nearest = nearest_neighbor_distances(graph)
    finite = np.isfinite(nearest)
    return float(nearest[finite].mean()) if np.any(finite) else 0.0


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def angle(u, v):
    c = float(np.clip(np.dot(unit(u), unit(v)), -1.0, 1.0))
    return math.degrees(math.acos(c))


def pca_normal(P):
    C = (P.T @ P) / max(1, len(P))
    vals, vecs = eig(C)
    n = vecs[:, np.argmin(vals.real)].real
    return unit(n)


def planarity_score(P):
    C = (P.T @ P) / max(1, len(P))
    w = np.sort(np.real(np.linalg.eigvalsh(C)))
    return float(w[0] / max(1e-12, w.sum()))


def unique_pairs_ij(atoms, r_max, with_dist=False, graph: NeighborGraph | None = None):
    graph = graph if graph is not None else build_neighbor_graph(atoms, r_max)
    keep = graph.mask_within(r_max)
    if with_dist:
        return graph.i[keep], graph.j[keep], graph.distances[keep]
    return graph.i[keep], graph.j[keep]


def distances_within(atoms, r_max, graph: NeighborGraph | None = None):
    _, _, d = unique_pairs_ij(atoms, r_max, with_dist=True, graph=graph)
    return np.asarray(d, float)


def per_atom_sorted_distances(atoms, r_max, graph: NeighborGraph | None = None):
    i, j, d = unique_pairs_ij(atoms, r_max, with_dist=True, graph=graph)
    n = len(atoms)
    buckets = [[] for _ in range(n)]
    for a, b, dist in zip(i, j, d):
        buckets[a].append(dist)
        buckets[b].append(dist)
    return [
        np.sort(np.asarray(lst, float)) if lst else np.zeros((0,), float)
        for lst in buckets
    ]
