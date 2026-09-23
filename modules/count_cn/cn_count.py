# Copyright (c) 2026 Olus Ozbek Research Group
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.

import numpy as np
from .utils import (
    NeighborGraph,
    average_nearest_distance,
    build_neighbor_graph,
    coordination_numbers,
    distances_within,
    per_atom_sorted_distances,
    unique_pairs_ij,
)


def auto_cutoff_from_hist_dists(all_dists, bins=300, r_min=1.8, r_max=4.2):
    d = all_dists[(all_dists >= r_min) & (all_dists <= r_max)]
    if d.size < 50:
        return 3.2
    hist, edges = np.histogram(d, bins=bins, range=(r_min, r_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    k = 5
    sm = np.convolve(hist, np.ones(k) / k, mode="same")
    peak = int(np.argmax(sm))
    min_idx = None
    for ii in range(peak + 1, len(sm) - 1):
        if sm[ii] <= sm[ii - 1] and sm[ii] <= sm[ii + 1]:
            min_idx = ii
            break
    cutoff = centers[min_idx] if min_idx is not None else centers[peak] * 1.25
    cutoff = min(cutoff, 3.7, centers[peak] * 1.35)
    return float(cutoff)


def cn_fixed(atoms, cutoff, neighbor_graph: NeighborGraph | None = None):
    graph = neighbor_graph
    if graph is None or graph.r_max + 1e-9 < float(cutoff):
        graph = build_neighbor_graph(atoms, float(cutoff))
    return coordination_numbers(graph, float(cutoff))


def cn_adaptive12(atoms, r_buffer=4.0, neighbor_graph: NeighborGraph | None = None):
    r_max = float(max(3.8, r_buffer))
    graph = neighbor_graph
    if graph is None or graph.r_max + 1e-9 < r_max:
        graph = build_neighbor_graph(atoms, r_max)
    per_atom = per_atom_sorted_distances(atoms, r_max, graph=graph)
    n = len(atoms)
    cn = np.zeros(n, dtype=int)
    i, j, d = unique_pairs_ij(atoms, r_max, with_dist=True, graph=graph)
    neighbors = [[] for _ in range(n)]
    for a, b, dist in zip(i, j, d):
        neighbors[a].append(dist)
        neighbors[b].append(dist)
    for idx in range(n):
        di = per_atom[idx]
        if di.size >= 13:
            thr = 0.5 * (di[11] + di[12])
        elif di.size >= 2:
            k = min(12, di.size - 1)
            thr = 0.5 * (di[k - 1] + di[min(k, di.size - 1)])
        elif di.size == 1:
            thr = di[0] * 1.10
        else:
            thr = 0.0
        cn[idx] = int(np.sum(np.asarray(neighbors[idx]) < thr))
    return cn


def compute_cn(
    atoms,
    mode="auto",
    cutoff=None,
    hist_rmin=1.8,
    hist_rmax=4.2,
    neighbor_graph: NeighborGraph | None = None,
):
    """Unified CN computation interface."""
    if mode == "fixed":
        if cutoff is None:
            raise ValueError("--mode fixed requires --cutoff")
        cn_arr = cn_fixed(atoms, float(cutoff), neighbor_graph=neighbor_graph)
        info = f"fixed cutoff = {float(cutoff):.3f} Å"
    elif mode == "auto":
        graph = neighbor_graph
        if graph is None or graph.r_max + 1e-9 < float(hist_rmax):
            graph = build_neighbor_graph(atoms, float(hist_rmax))
        d_within = distances_within(atoms, hist_rmax, graph=graph)
        c = auto_cutoff_from_hist_dists(d_within, r_min=hist_rmin, r_max=hist_rmax)
        nearest_scale = average_nearest_distance(graph)
        if nearest_scale > 0.0:
            c = max(c, nearest_scale * 1.15)
        cn_arr = cn_fixed(atoms, c, neighbor_graph=graph)
        info = f"auto cutoff = {c:.3f} Å"
    elif mode == "adaptive12":
        cn_arr = cn_adaptive12(
            atoms,
            r_buffer=max(3.8, hist_rmax),
            neighbor_graph=neighbor_graph,
        )
        info = "adaptive12: per-atom cutoff = 0.5*(d12+d13)"
    else:
        raise ValueError("Unknown CN mode")
    return cn_arr, info
