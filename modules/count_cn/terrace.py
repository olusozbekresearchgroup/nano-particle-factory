"""Additional terrace-interior inventories; original site lists stay unchanged."""

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np


TERRACE_TYPES = ("hollow3", "hollow4_square", "hollow4_rect")
TERRACE_COUNT_KEYS = tuple(
    f"{kind}_interior_{unit}"
    for kind in TERRACE_TYPES
    for unit in ("atoms", "sites")
)


def terrace_interiors(pos, sites, a1, angle_tol_deg=10.0, plane_tol=0.10):
    """Find boundaries in edge-connected, approximately coplanar motif patches.

    Every patch is compared with its seed plane to avoid joining a curved
    particle surface through a chain of small angular differences. Boundary
    edges belong to only one motif (nonmanifold edges are boundaries too).
    An interior site must have all its vertices interior to the SAME patch.
    Atom and motif inventories are deduplicated across patches of each type.
    """
    additions = {key: [] for key in TERRACE_COUNT_KEYS}
    patches = {kind: [] for kind in TERRACE_TYPES}
    cosine = np.cos(np.deg2rad(angle_tol_deg))
    distance_tol = max(1e-9, float(a1) * plane_tol)

    for kind in TERRACE_TYPES:
        motifs = sorted({tuple(sorted(site)) for site in sites.get(kind, [])})
        if not motifs:
            continue
        centers, normals, boundaries = [], [], []
        by_edge = defaultdict(list)
        for index, motif in enumerate(motifs):
            points = pos[list(motif)]
            center = points.mean(axis=0)
            _, _, axes = np.linalg.svd(points - center, full_matrices=False)
            projected = (points - center) @ axes[:2].T
            order = np.argsort(np.arctan2(projected[:, 1], projected[:, 0]))
            cycle = [motif[i] for i in order]
            edges = [tuple(sorted((cycle[i], cycle[(i + 1) % len(cycle)])))
                     for i in range(len(cycle))]
            centers.append(center)
            normals.append(axes[-1])
            boundaries.append(edges)
            for edge in edges:
                by_edge[edge].append(index)

        adjacency = [set() for _ in motifs]
        for members in by_edge.values():
            for first, second in combinations(members, 2):
                adjacency[first].add(second)
                adjacency[second].add(first)

        remaining = set(range(len(motifs)))
        all_interior_atoms, all_interior_sites = set(), set()
        for seed in range(len(motifs)):
            if seed not in remaining:
                continue
            remaining.remove(seed)
            component, pending = [], [seed]
            while pending:
                current = pending.pop()
                component.append(current)
                for neighbor in sorted(adjacency[current] & remaining):
                    if abs(normals[seed] @ normals[neighbor]) < cosine:
                        continue
                    offsets = pos[list(motifs[neighbor])] - centers[seed]
                    if np.max(np.abs(offsets @ normals[seed])) > distance_tol:
                        continue
                    remaining.remove(neighbor)
                    pending.append(neighbor)

            edge_counts = Counter(edge for i in component for edge in boundaries[i])
            boundary_atoms = {a for edge, count in edge_counts.items() if count != 2 for a in edge}
            atoms = {a for i in component for a in motifs[i]}
            interior = atoms - boundary_atoms
            interior_sites = [motifs[i] for i in component if set(motifs[i]) <= interior]
            all_interior_atoms.update(interior)
            all_interior_sites.update(interior_sites)
            patches[kind].append({
                "atoms": sorted(atoms),
                "boundary_atoms": sorted(boundary_atoms),
                "interior_atoms": sorted(interior),
                "interior_sites": sorted(interior_sites),
            })
        additions[f"{kind}_interior_atoms"] = sorted(all_interior_atoms)
        additions[f"{kind}_interior_sites"] = sorted(all_interior_sites)
    return additions, patches
