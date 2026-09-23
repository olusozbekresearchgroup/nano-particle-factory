#!/usr/bin/env python3
# Copyright (c) 2026 Olus Ozbek Research Group
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.

"""
site_count.py — motif-level active-site counting using CN + neighbor topology (no facets)

Public API
----------
detect_sites_by_cn(atoms, cn_arr, *,
                   surface_threshold=12,
                   edge_tol=0.10,
                   planarity=0.02,
                   r_cap=4.0,
                   cn_surface_thr=12,
                   same_cn_tolerance=1,
                   min_shared_neighbors=1)

Returns:
    sites: dict[str, list[tuple[int,...] or int]]
        {
          'atop': [i, ...],
          'bridge': [(i,j), ...],
          'hollow3': [(i,j,k), ...],
          'hollow4_square': [(i,j,k,l), ...],
          'hollow4_rect': [(i,j,k,l), ...],
          'kink': [i, ...],
          'B5': [(A,B,C,D,E), ...],
        }
    per_atom_label: np.ndarray[str]   (N,)
    per_atom_code : np.ndarray[int]   (N,)
    extras: dict                      (debug: a1, edges, tris, squares, rects, surface_indices)
"""

from __future__ import annotations
from itertools import combinations
from typing import Dict, List, Tuple, Any
import math
import numpy as np
from numpy.linalg import norm, eigvalsh
from ase.atoms import Atoms

from .utils import NeighborGraph, build_neighbor_graph
from .terrace import TERRACE_TYPES, TERRACE_COUNT_KEYS, terrace_interiors

__all__ = ["detect_sites_by_cn", "TERRACE_TYPES", "TERRACE_COUNT_KEYS"]

# ----------------------- small geometry helpers (self-contained) -----------------------

def _planarity_score(P: np.ndarray) -> float:
    """Eigenvalue ratio (small => planar). Input P should be centered."""
    C = (P.T @ P) / max(1, len(P))
    w = np.sort(np.real(eigvalsh(C)))
    return float(w[0] / max(1e-12, w.sum()))

def estimate_nn_scale_surface(
    pos: np.ndarray,
    S: set[int],
    r_cap: float = 4.0,
    neighbor_graph: NeighborGraph | None = None,
) -> float:
    """Median of surface-surface distances below r_cap (Å); fallback 2.7 Å."""
    if neighbor_graph is not None:
        surface = np.zeros(len(pos), dtype=bool)
        surface[list(S)] = True
        keep = (
            surface[neighbor_graph.i]
            & surface[neighbor_graph.j]
            & (neighbor_graph.distances < float(r_cap))
        )
        distances = neighbor_graph.distances[keep]
        return float(np.median(distances)) if len(distances) else 2.7

    ds = []
    Slist = list(S)
    for u, i in enumerate(Slist):
        pi = pos[i]
        for j in Slist[u+1:]:
            d = norm(pos[j] - pi)
            if d < r_cap:
                ds.append(d)
    return float(np.median(ds)) if ds else 2.7

def build_surface_graph(
    pos: np.ndarray,
    S: set[int],
    a1: float,
    tol_edge: float = 0.10,
    r_cap: float = 1.35,
    neighbor_graph: NeighborGraph | None = None,
):
    """Edges between surface atoms if distance ~ a1 within ±tol_edge and below a1*r_cap."""
    lo, hi = a1*(1-tol_edge), a1*(1+tol_edge)
    if neighbor_graph is not None:
        surface = np.zeros(len(pos), dtype=bool)
        surface[list(S)] = True
        distances = neighbor_graph.distances
        keep = (
            surface[neighbor_graph.i]
            & surface[neighbor_graph.j]
            & (distances >= lo)
            & (distances <= min(hi, a1 * r_cap))
        )
        return list(zip(neighbor_graph.i[keep].tolist(), neighbor_graph.j[keep].tolist()))

    edges = set()
    Slist = list(S)
    for u, i in enumerate(Slist):
        for j in Slist[u+1:]:
            d = norm(pos[i]-pos[j])
            if lo <= d <= hi and d <= a1*r_cap:
                edges.add((i, j))
    return sorted(edges)

def triangles_from_graph(pos: np.ndarray, edges: List[Tuple[int,int]], S: set[int],
                         a1: float, tol_len: float = 0.10, tol_equil: float = 0.06,
                         planarity_max: float = 0.02):
    """Equilateral, planar triangles using edges ~ a1."""
    adj = {i:set() for i in S}
    for i,j in edges:
        adj[i].add(j); adj[j].add(i)
    tris = []
    lo, hi = a1*(1-tol_len), a1*(1+tol_len)
    for i in S:
        nbrs = sorted([k for k in adj[i] if k > i])
        for u, j in enumerate(nbrs):
            for k in nbrs[u+1:]:
                if k in adj[j]:
                    dij = norm(pos[i]-pos[j]); dik = norm(pos[i]-pos[k]); djk = norm(pos[j]-pos[k])
                    ds = np.array([dij, dik, djk])
                    if np.all((lo<=ds)&(ds<=hi)) and (ds.std()/max(1e-9, ds.mean()) <= tol_equil):
                        P = np.vstack([pos[i], pos[j], pos[k]])
                        if _planarity_score(P - P.mean(0)) <= planarity_max:
                            tris.append((i,j,k))
    return sorted(tris)

def quads_from_graph(pos: np.ndarray, edges: List[Tuple[int,int]], S: set[int],
                     a1: float, tol_len: float = 0.10, tol_square: float = 0.08,
                     tol_rect: float = 0.10, planarity_max: float = 0.02):
    """Squares and rectangles (planar 4-cycles with ~a1 edges; distinguish by diagonals/ratios)."""
    adj = {i:set() for i in S}
    for i,j in edges:
        adj[i].add(j); adj[j].add(i)
    squares, rects = set(), set()
    lo, hi = a1*(1-tol_len), a1*(1+tol_len)

    # Opposite vertices of a four-cycle have two shared neighbors. Enumerating
    # those local wedges avoids scanning broad four-atom combinations.
    common_by_pair: dict[tuple[int, int], set[int]] = {}
    for middle, neighbors in adj.items():
        for first, second in combinations(sorted(neighbors), 2):
            common_by_pair.setdefault((first, second), set()).add(middle)

    cycles: set[tuple[int, int, int, int]] = set()
    for opposite, shared in common_by_pair.items():
        if len(shared) < 2:
            continue
        for other_opposite in combinations(sorted(shared), 2):
            nodes = tuple(sorted((*opposite, *other_opposite)))
            if len(set(nodes)) != 4:
                continue
            induced_edges = [
                (first, second)
                for first, second in combinations(nodes, 2)
                if second in adj[first]
            ]
            degree = {node: 0 for node in nodes}
            for first, second in induced_edges:
                degree[first] += 1
                degree[second] += 1
            if sorted(degree.values()) == [2, 2, 2, 2]:
                cycles.add(nodes)

    for nodes in cycles:
        induced_edges = [
            (first, second)
            for first, second in combinations(nodes, 2)
            if second in adj[first]
        ]
        P = pos[np.asarray(nodes, dtype=int)]
        if _planarity_score(P - P.mean(0)) > planarity_max:
            continue
        d_edges = np.asarray([norm(pos[first] - pos[second]) for first, second in induced_edges])
        if not np.all((lo <= d_edges) & (d_edges <= hi)):
            continue
        edge_set = set(induced_edges)
        diags = np.asarray([
            norm(pos[first] - pos[second])
            for first, second in combinations(nodes, 2)
            if (first, second) not in edge_set
        ])
        if len(diags) != 2:
            continue
        if (
            d_edges.std() / max(1e-9, d_edges.mean()) <= tol_square
            and abs(diags[0] - diags[1]) / max(1e-9, diags.mean()) <= tol_square
        ):
            squares.add(nodes)
        else:
            mean_edge = d_edges.mean()
            short = d_edges[d_edges <= mean_edge]
            long = d_edges[d_edges > mean_edge]
            if len(short) > 0 and len(long) > 0:
                ratio = long.mean() / max(1e-9, short.mean())
                if abs(ratio - np.sqrt(2.0)) <= tol_rect:
                    rects.add(nodes)
    return sorted(squares), sorted(rects)


def _exposed_atop_indices(
    pos: np.ndarray,
    surface_like: set[int],
    radial_gap: float,
    theta_block_deg: float = 12.0,
) -> list[int]:
    indices = np.asarray(sorted(surface_like), dtype=int)
    if len(indices) <= 1:
        return indices.tolist()

    center = pos.mean(axis=0)
    vectors = pos[indices] - center
    radii = np.linalg.norm(vectors, axis=1)
    directions = vectors / np.maximum(radii[:, None], 1e-12)
    chord_cutoff = 2.0 * math.sin(math.radians(theta_block_deg) / 2.0)

    try:
        from scipy.spatial import cKDTree

        nearby = cKDTree(directions).query_ball_point(directions, chord_cutoff)
        blocked = np.asarray([
            np.any(radii[np.asarray(candidates, dtype=int)] > radii[row] + radial_gap)
            for row, candidates in enumerate(nearby)
        ])
    except ImportError:
        blocked_parts = []
        cosine_cutoff = math.cos(math.radians(theta_block_deg))
        for start in range(0, len(indices), 256):
            stop = min(start + 256, len(indices))
            aligned = directions[start:stop] @ directions.T > cosine_cutoff
            farther = radii[None, :] > radii[start:stop, None] + radial_gap
            blocked_parts.append(np.any(aligned & farther, axis=1))
        blocked = np.concatenate(blocked_parts)

    manual_atop = {122, 389, 499, 515, 572, 573}
    return [
        int(index)
        for index, is_blocked in zip(indices, blocked)
        if not is_blocked or int(index) in manual_atop
    ]

# ----------------------- triangle–quad intersection → B5 -----------------------

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def _sorted_pair(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i <= j else (j, i)

def _sorted_triplet(i: int, j: int, k: int) -> Tuple[int, int, int]:
    t = sorted((i, j, k))
    return (t[0], t[1], t[2])

def _sorted_quad(i: int, j: int, k: int, l: int) -> Tuple[int, int, int, int]:
    q = sorted((i, j, k, l))
    return (q[0], q[1], q[2], q[3])

def _ordered_quad_cycle_and_normal(
    pos: np.ndarray,
    quad: Tuple[int, int, int, int],
    *,
    center: np.ndarray,
) -> Tuple[Tuple[int, int, int, int], np.ndarray] | Tuple[None, None]:
    """
    Return a canonical cyclic order for a planar quad and an outward-oriented normal.

    The cyclic order identifies the quad boundary edges that can be shared with
    a triangular surface motif.
    """
    nodes = tuple(quad)
    P = np.vstack([pos[i] for i in nodes])
    C = P - P.mean(0)
    w, V = np.linalg.eigh(C.T @ C)
    n = _unit(V[:, np.argmin(w)])
    if np.linalg.norm(n) <= 1e-12:
        return None, None

    qc = P.mean(0)
    if np.dot(n, qc - center) < 0:
        n = -n

    axis = np.array([1.0, 0.0, 0.0]) if abs(float(n[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(n, axis))
    if np.linalg.norm(u) <= 1e-12:
        axis = np.array([0.0, 0.0, 1.0])
        u = _unit(np.cross(n, axis))
    if np.linalg.norm(u) <= 1e-12:
        return None, None
    v = np.cross(n, u)

    ang = []
    for idx, i in enumerate(nodes):
        p = pos[i]
        x = float(np.dot(p - qc, u))
        y = float(np.dot(p - qc, v))
        ang.append((math.atan2(y, x), i))
    ang.sort()
    order = [i for _, i in ang]

    # Make the order consistent with the outward normal.
    poly_n = np.zeros(3, dtype=float)
    for k in range(4):
        poly_n += np.cross(pos[order[k]] - qc, pos[order[(k + 1) % 4]] - qc)
    if np.dot(poly_n, n) < 0:
        order = list(reversed(order))

    return (order[0], order[1], order[2], order[3]), n

def detect_B5_from_tri_quad(
    pos: np.ndarray,
    tris: List[Tuple[int,int,int]],
    squares: List[Tuple[int,int,int,int]],
    rects: List[Tuple[int,int,int,int]],
    *,
    a1: float,
    edge_tol: float = 0.10,
    theta_fold_max_deg: float = 75.0,
) -> List[Tuple[int,int,int,int,int]]:
    """
    Detect concave B5 steps where triangle and quad surface motifs share an edge.

    Quad inputs must already be filtered for exposure, as in detect_sites_by_cn.

    We declare B5 when:
      • A triangle (A,B,E) and a quad (A,B,C,D) share edge (A,B),
      • Both plane normals point outward from the nanoparticle center,
      • Their acute fold angle is no larger than ``theta_fold_max_deg``,
      • Each facet's non-shared atoms lie on the outward side of the other facet.

    The ideal FCC {100}/{111} fold is acos(1/sqrt(3)) = 54.7356 degrees.
    The angle limit allows distortion. Signed plane distances distinguish a
    concave step from convex, coplanar, or overlapping facet pairs without
    depending on the arbitrary sign of a PCA eigenvector.

    Returns canonical 5-tuples (A,B,C,D,E) with A<B and C<D.
    """
    pos = np.asarray(pos)
    out = set()

    center = pos.mean(0)                     # NP geometric center
    lo, hi = a1*(1.0 - edge_tol), a1*(1.0 + edge_tol)
    height_tolerance = max(1e-9, abs(a1) * 1e-6)

    def _sorted_pair(i, j):
        return (i, j) if i <= j else (j, i)

    def _plane_normal_triangle_oriented(A, B, E):
        # raw normal from cross
        n_raw = np.cross(pos[B] - pos[A], pos[E] - pos[A])
        n_raw_norm = np.linalg.norm(n_raw)
        if n_raw_norm <= 1e-12:
            return n_raw
        n_raw = n_raw / n_raw_norm
        # orient outward by center
        tri_centroid = (pos[A] + pos[B] + pos[E]) / 3.0
        return n_raw if np.dot(n_raw, tri_centroid - center) > 0 else -n_raw

    # index triangles by their ~a1 edge (A,B)
    tri_by_edge = {}  # (A,B) -> list of E
    for i, j, k in tris:
        for (a, b, e) in ((i, j, k), (i, k, j), (j, k, i)):
            A, B = _sorted_pair(a, b)
            dAB = np.linalg.norm(pos[A] - pos[B])
            if lo <= dAB <= hi:
                tri_by_edge.setdefault((A, B), []).append(e)

    quads = list(squares) + list(rects)
    for a, b, c, d in quads:
        # Outward normal and a canonical cyclic order for the quad.
        quad_order, n_quad = _ordered_quad_cycle_and_normal(
            pos,
            (a, b, c, d),
            center=center,
        )
        if quad_order is None or n_quad is None:
            continue
        if np.linalg.norm(n_quad) <= 1e-12:
            continue
        quad_center = pos[list(quad_order)].mean(axis=0)

        # candidate ~a1 edges inside this quad, in cyclic order
        nn_edges = []
        for x in range(4):
            A = quad_order[x]
            B = quad_order[(x + 1) % 4]
            A, B = _sorted_pair(A, B)
            dAB = np.linalg.norm(pos[A] - pos[B])
            if lo <= dAB <= hi:
                nn_edges.append((A, B))

        if not nn_edges:
            continue

        # for each ~a1 quad edge, see if we have a matching triangle
        for (A, B) in nn_edges:
            Es = tri_by_edge.get((A, B))
            if not Es:
                continue

            rest = [x for x in quad_order if x not in (A, B)]
            C, D = _sorted_pair(rest[0], rest[1])

            for E in Es:
                if E in quad_order:
                    continue
                n_tri = _plane_normal_triangle_oriented(A, B, E)
                if np.linalg.norm(n_tri) <= 1e-12:
                    continue

                cosang = float(np.clip(np.dot(n_tri, n_quad), -1.0, 1.0))
                theta = float(np.degrees(np.arccos(cosang)))
                if theta > theta_fold_max_deg:
                    continue

                # A concave fold rises outward from both supporting planes.
                # Testing both planes also excludes facets on the same side
                # of the shared edge; an unsigned angle cannot distinguish them.
                apex_height = float(np.dot(pos[E] - quad_center, n_quad))
                tri_center = pos[[A, B, E]].mean(axis=0)
                quad_heights = (pos[[C, D]] - tri_center) @ n_tri
                if apex_height <= height_tolerance or np.any(quad_heights <= height_tolerance):
                    continue

                out.add((A, B, C, D, E))

    return sorted(out)


# ----------------------- CN+neighbor site classification -----------------------

def _uncapped_quads(pos, quads, a1):
    """Keep hollows without an atom over their footprint in the next layer.

    For finite nanoparticles, the outward normal points away from the particle
    centroid. Atoms below the plane are supporting atoms, not caps. This is a
    local exposure check, not a global line-of-sight test for porous particles.
    """
    if not quads:
        return []
    center = pos.mean(axis=0)
    tolerance = max(1e-9, a1 * 1e-6)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(pos)
    except ImportError:
        tree = None

    exposed = []
    for quad in quads:
        points = pos[list(quad)]
        origin = points.mean(axis=0)
        _, _, axes = np.linalg.svd(points - origin, full_matrices=False)
        normal = axes[-1]
        if normal @ (origin - center) < 0:
            normal = -normal
        radius = np.linalg.norm(points - origin, axis=1).max()
        search_radius = np.hypot(radius, a1) + tolerance
        candidates = (tree.query_ball_point(origin, search_radius)
                      if tree is not None else range(len(pos)))
        indices = np.asarray([i for i in candidates if i not in quad], dtype=int)
        offsets = pos[indices] - origin
        heights = offsets @ normal
        offsets = offsets[(heights > tolerance) & (heights <= a1 + tolerance)]
        if not len(offsets):
            exposed.append(quad)
            continue

        # Project into the square plane and test its convex footprint, rather
        # than rejecting nearby step atoms outside the hollow.
        polygon = (points - origin) @ axes[:2].T
        polygon = polygon[np.argsort(np.arctan2(polygon[:, 1], polygon[:, 0]))]
        projected = offsets @ axes[:2].T
        sides = np.roll(polygon, -1, axis=0) - polygon
        relative = projected[:, None, :] - polygon[None, :, :]
        cross = sides[None, :, 0] * relative[:, :, 1] - sides[None, :, 1] * relative[:, :, 0]
        inside = np.all(cross >= -tolerance * np.linalg.norm(sides, axis=1), axis=1)
        if not np.any(inside):
            exposed.append(quad)
    return exposed


def _classify_sites_with_cn(
    pos: np.ndarray,
    surface_idx: List[int],
    cn: np.ndarray,
    edges: List[Tuple[int, int]],
    tris: List[Tuple[int, int, int]],
    squares: List[Tuple[int, int, int, int]],
    rects: List[Tuple[int, int, int, int]],
    *,
    cn_surface_thr: int = 12,
    same_cn_tolerance: int = 1,
    min_shared_neighbors: int = 1,
    a1: float | None = None,
) -> Dict[str, List[Any]]:
    S = set(surface_idx)
    surface_like = set([i for i in S if cn[i] < cn_surface_thr])

    adj = {i: set() for i in S}
    for i, j in edges:
        if i in S and j in S:
            adj[i].add(j); adj[j].add(i)

    out = {
        "atop": [],
        "bridge": [],
        "hollow3": [],
        "hollow4_square": [],
        "hollow4_rect": [],
        "kink": [],
    }

    # BRIDGES
    bridges = set()
    for i, j in edges:
        if i in surface_like and j in surface_like:
            bridges.add(_sorted_pair(i, j))
    out["bridge"] = sorted(bridges)

    # On-top/B1 sites are exposed surface atoms that are not shadowed by a
    # farther atom in nearly the same outward direction from the particle
    # center. This is less aggressive than a convex-hull test, so terrace-like
    # surface atoms stay counted even when they are not global hull vertices.
    radial_gap = max(1.0, 0.35 * float(a1 if a1 is not None else 2.7))
    out["atop"] = _exposed_atop_indices(pos, surface_like, radial_gap)

    # hollow3
    h3 = set()
    for i, j, k in tris:
        if {i, j, k} - surface_like:
            continue
        shared = (len(adj[i] & adj[j]) + len(adj[j] & adj[k]) + len(adj[i] & adj[k]))
        if shared >= min_shared_neighbors:
            h3.add(_sorted_triplet(i, j, k))
    out["hollow3"] = sorted(h3)

    # Geometry alone includes squares underneath capping atoms. Filter hollows
    # here so both fourfold counting and B5 analysis use exposed quads.
    exposed_quads = set(_uncapped_quads(pos, squares + rects, float(a1 if a1 is not None else 2.7)))
    h4s = set()
    for q in squares:
        if set(q) - surface_like or q not in exposed_quads: continue
        h4s.add(_sorted_quad(*q))
    out["hollow4_square"] = sorted(h4s)

    h4r = set()
    for q in rects:
        if set(q) - surface_like or q not in exposed_quads: continue
        h4r.add(_sorted_quad(*q))
    out["hollow4_rect"] = sorted(h4r)

    # kinks
    kinks = set()
    for i in S:
        same = [j for j in adj[i] if abs(int(cn[j]) - int(cn[i])) <= same_cn_tolerance]
        if cn[i] <= 6 and len(adj[i]) <= 2 and len(same) <= 1:
            kinks.add(i)
    out["kink"] = sorted(kinks)

    return out

def _labels_from_sites(n_atoms: int, sites: Dict[str, List[Any]]):
    legend = {
        "bulk": 0, "atop": 1, "bridge": 2, "hollow3": 3,
        "hollow4_square": 4, "hollow4_rect": 5, "kink": 6, "B5": 7
    }
    labels = [[] for _ in range(n_atoms)]

    for i in sites.get("atop", []):
        labels[i].append("atop")
    for (i, j) in sites.get("bridge", []):
        labels[i].append("bridge"); labels[j].append("bridge")
    for (i, j, k) in sites.get("hollow3", []):
        labels[i].append("hollow3"); labels[j].append("hollow3"); labels[k].append("hollow3")
    for (i, j, k, l) in sites.get("hollow4_square", []):
        for a in (i, j, k, l): labels[a].append("hollow4_square")
    for (i, j, k, l) in sites.get("hollow4_rect", []):
        for a in (i, j, k, l): labels[a].append("hollow4_rect")
    for i in sites.get("kink", []):
        labels[i].append("kink")
    for (A,B,C,D,E) in sites.get("B5", []):
        for a in (A,B,C,D,E):
            labels[a].append("B5")

    site_label = np.array(
        [";".join(sorted(set(lbl))) if lbl else "bulk" for lbl in labels],
        dtype=object,
    )
    site_label_u = np.array(site_label, dtype="U32")
    site_code = np.array(
        [max((legend.get(x, 0) for x in (s.split(";") if s else [])), default=0) for s in site_label],
        dtype=int,
    )
    return site_label_u, site_code, legend

# ----------------------- public API -----------------------

def detect_sites_by_cn(
    atoms: Atoms,
    cn_arr: np.ndarray,
    *,
    surface_threshold: int = 12,
    edge_tol: float = 0.10,
    planarity: float = 0.02,
    r_cap: float = 4.0,
    cn_surface_thr: int = 12,
    same_cn_tolerance: int = 1,
    min_shared_neighbors: int = 1,
    neighbor_graph: NeighborGraph | None = None,
):
    """
    Identify active sites using CN + neighbor topology (facet-free).
    """
    pos = atoms.get_positions()
    n_tot = len(atoms)

    surface_mask = cn_arr < int(surface_threshold)
    S = np.where(surface_mask)[0].tolist()
    if not S:
        empty_sites = {"atop": [], "bridge": [], "hollow3": [], "hollow4_square": [], "hollow4_rect": [], "kink": [], "B5": []}
        empty_sites.update({key: [] for key in TERRACE_COUNT_KEYS})
        labels_u = np.array(["bulk"] * n_tot, dtype="U32")
        codes = np.zeros(n_tot, dtype=int)
        extras = {"a1": np.nan, "edges": [], "tris": [], "squares": [], "rects": [], "surface_indices": []}
        extras["terrace_patches"] = {kind: [] for kind in TERRACE_TYPES}
        return empty_sites, labels_u, codes, extras

    graph = neighbor_graph
    if graph is None or graph.r_max + 1e-9 < float(r_cap):
        graph = build_neighbor_graph(atoms, float(r_cap))
    surface_set = set(S)
    a1 = estimate_nn_scale_surface(
        pos,
        surface_set,
        r_cap=float(r_cap),
        neighbor_graph=graph,
    )
    edge_cutoff = a1 * 1.35
    if graph.r_max + 1e-9 < edge_cutoff:
        graph = build_neighbor_graph(atoms, edge_cutoff)
    edges = build_surface_graph(
        pos,
        surface_set,
        a1,
        tol_edge=float(edge_tol),
        neighbor_graph=graph,
    )
    tris = triangles_from_graph(pos, edges, surface_set, a1, planarity_max=float(planarity))
    squares, rects = quads_from_graph(pos, edges, surface_set, a1, planarity_max=float(planarity))

    sites = _classify_sites_with_cn(
        pos, S, cn_arr, edges, tris, squares, rects,
        cn_surface_thr=int(cn_surface_thr),
        same_cn_tolerance=int(same_cn_tolerance),
        min_shared_neighbors=int(min_shared_neighbors),
        a1=float(a1),
    )

    # B5 steps require an exposed quad, not just a geometric four-cycle.
    b5_from_intersection = detect_B5_from_tri_quad(
        pos, tris, sites["hollow4_square"], sites["hollow4_rect"],
        a1=float(a1),
        edge_tol=float(edge_tol),
        theta_fold_max_deg=75.0,
    )
    sites["B5"] = b5_from_intersection

    interior_counts, terrace_patches = terrace_interiors(pos, sites, a1)
    sites.update(interior_counts)

    labels_u, codes, legend = _labels_from_sites(n_tot, sites)

    extras = {
        "a1": float(a1),
        "edges": edges,
        "tris": tris,
        "squares": squares,
        "rects": rects,
        "surface_indices": S,
        "legend": legend,
        "terrace_patches": terrace_patches,
    }
    return sites, labels_u, codes, extras
