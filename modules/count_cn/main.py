#!/usr/bin/env python3
"""
Main entry point — CN + active-site inventory (facet-free, single-file output).

- Reads structure from CLI positional arg or config (np_sites/general: input/structure)
- Computes CN using cn_count.compute_cn
- Identifies active sites using site_count.detect_sites_by_cn (CN + neighbor topology)
- Writes a single CSV:
    <input_basename>.csv

CSV schema:
  kind,key,value,index,element,x,y,z,CN,site_label,indices

Sections:
  - meta rows        : kind=meta,  key in {file, atoms, mode, cutoff_info, surface_threshold}
  - count rows       : kind=count, key in {atop,bridge,hollow3,hollow4_square,hollow4_rect,kink,B5,surface_atoms}
  - hist rows        : kind=hist,  key=CN (integer), value=count
  - atom rows        : kind=atom,  one per atom (index, element, x,y,z, CN, site_label)
  - motif rows       : kind in {'bridge','hollow3','hollow4_square','hollow4_rect','kink','B5'}, indices="[i,...]"
"""

import sys
import os
import csv
import argparse
import numpy as np
from ase.io import read

from .config_loader import load_settings
from .cn_count import compute_cn
from .site_count import detect_sites_by_cn
from .utils import build_neighbor_graph


def _resolve_input_path(args, cfg):
    """Pick the input file from CLI positional or config (np_sites/general)."""
    if args.structure is not None:
        return args.structure
    if hasattr(cfg, "input") and cfg.input:
        return cfg.input
    if hasattr(cfg, "structure") and cfg.structure:
        return cfg.structure
    try:
        if cfg.get("input"):
            return cfg.get("input")
        if cfg.get("structure"):
            return cfg.get("structure")
    except Exception:
        pass
    return None


def _default_out_path(input_file: str, prefer_csv: bool = True) -> str:
    base = os.path.splitext(os.path.basename(input_file))[0]
    return f"{base}.csv" if prefer_csv else f"{base}.txt"


def write_single_csv(
    out_path: str,
    *,
    input_file: str,
    n_atoms: int,
    mode: str,
    cutoff_info: str,
    surface_threshold: int,
    cn_arr: np.ndarray,
    atoms,
    sites: dict,
):
    """
    Single CSV with sections:
    - meta rows:   kind=meta, key,value
    - count rows:  kind=count, key=site_type, value=count
    - hist rows:   kind=hist,  key=CN integer, value=count
    - atom rows:   kind=atom, per-atom data
    - motif rows:  kind in {'bridge','hollow3','hollow4_square','hollow4_rect','kink','B5'}, indices as "[...]"
    """
    pos  = atoms.get_positions()
    elems = atoms.get_chemical_symbols()

    # counts
    counts = {k: len(v) for k, v in sites.items()}
    counts["surface_atoms"] = int(np.sum(cn_arr < surface_threshold))

    # CN histogram (integer bins across observed range)
    cn_min = int(np.min(cn_arr))
    cn_max = int(np.max(cn_arr))
    hist, edges_h = np.histogram(cn_arr, bins=np.arange(cn_min, cn_max + 2) - 0.5)
    cn_bins = (edges_h[:-1] + edges_h[1:]) * 0.5  # integer centers

    # per-atom labels from motif inventory
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
    for (i,j,k,l,m) in sites.get("B5", []):
        for a in (i,j,k,l,m): labels[a].append("B5")
    site_label = [";".join(sorted(set(lbl))) if lbl else "bulk" for lbl in labels]

    def _fmt_indices(x):
        if isinstance(x, (list, tuple)):
            return "[" + ",".join(str(int(i)) for i in x) + "]"
        return "[" + str(int(x)) + "]"

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        # header
        w.writerow(["kind","key","value","index","element","x","y","z","CN","site_label","indices"])

        # meta
        w.writerow(["meta","file", input_file, "", "", "", "", "", "", "", ""])
        w.writerow(["meta","atoms", n_atoms, "", "", "", "", "", "", "", ""])
        w.writerow(["meta","mode", mode, "", "", "", "", "", "", "", ""])
        w.writerow(["meta","cutoff_info", cutoff_info, "", "", "", "", "", "", "", ""])
        w.writerow(["meta","surface_threshold", surface_threshold, "", "", "", "", "", "", "", ""])

        # counts
        for k in ("atop","bridge","hollow3","hollow4_square","hollow4_rect","kink","B5","surface_atoms"):
            if k in counts:
                w.writerow(["count", k, counts[k], "", "", "", "", "", "", "", ""])

        # histogram rows
        for c, h in zip(cn_bins.astype(int), hist):
            w.writerow(["hist", int(c), int(h), "", "", "", "", "", "", "", ""])

        # per-atom
        for i, (e, p, cn) in enumerate(zip(elems, pos, cn_arr)):
            w.writerow([
                "atom","", "", i, e, f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}",
                int(cn), site_label[i], ""
            ])

        # motifs
        for typ in ("bridge","hollow3","hollow4_square","hollow4_rect","kink","B5"):
            for item in sites.get(typ, []):
                w.writerow([typ,"","", "", "", "", "", "", "", "", _fmt_indices(item)])


def main():
    ap = argparse.ArgumentParser(description="CN + active-site finder (facet-free, single file output).")
    ap.add_argument("structure", nargs="?", help="Input structure (ASE-readable)")
    ap.add_argument("--config", help="Config INI (default: ./config.cn or ~/.np_sites/config.cn)")
    # optional CLI overrides
    ap.add_argument("--mode", choices=["fixed", "auto", "adaptive12"], default=None)
    ap.add_argument("--cutoff", type=float, default=None)
    ap.add_argument("--hist-rmin", type=float, dest="hist_rmin", default=None)
    ap.add_argument("--hist-rmax", type=float, dest="hist_rmax", default=None)
    ap.add_argument("--surface-threshold", type=int, dest="surface_threshold", default=None)
    ap.add_argument("--edge-tol", type=float, dest="edge_tol", default=None)
    ap.add_argument("--planarity", type=float, default=None)
    ap.add_argument("--tag", default=None)  # kept for compatibility, unused in single-file mode
    args = ap.parse_args()

    # Load config (applies CLI overrides inside your loader)
    cfg, cfg_path = load_settings(args)

    # Resolve input structure
    input_file = _resolve_input_path(args, cfg)
    if not input_file:
        print("ERROR: no input structure (provide STRUCTURE or set input= / structure= in config.cn)", file=sys.stderr)
        sys.exit(2)

    # Read structure
    try:
        atoms = read(input_file)
    except Exception as e:
        print(f"ERROR: could not read structure '{input_file}': {e}", file=sys.stderr)
        sys.exit(1)
    atoms.pbc = False
    pos = atoms.get_positions()
    n_tot = len(atoms)

    # ---- 1) CN
    graph_cutoff = max(
        4.0,
        float(cfg.hist_rmax),
        float(cfg.cutoff or 0.0) if str(cfg.mode) == "fixed" else 0.0,
    )
    neighbor_graph = build_neighbor_graph(atoms, graph_cutoff)
    cn_arr, cutoff_info = compute_cn(
        atoms,
        mode=str(cfg.mode),
        cutoff=cfg.cutoff,
        hist_rmin=float(cfg.hist_rmin),
        hist_rmax=float(cfg.hist_rmax),
        neighbor_graph=neighbor_graph,
    )
    is_surface = cn_arr < int(cfg.surface_threshold)
    S = np.where(is_surface)[0]

    # ---- 2) Sites (facet-free; CN + neighbor topology)
    sites, site_label_u, site_code, extras = detect_sites_by_cn(
        atoms,
        cn_arr,
        surface_threshold=int(cfg.surface_threshold),
        edge_tol=float(cfg.edge_tol),
        planarity=float(cfg.planarity),
        r_cap=4.0,
        cn_surface_thr=int(cfg.surface_threshold),
        same_cn_tolerance=1,
        min_shared_neighbors=1,
        neighbor_graph=neighbor_graph,
    )

    # ---- 3) Single-file output
    out_path = _default_out_path(input_file, prefer_csv=True)
    write_single_csv(
        out_path,
        input_file=input_file,
        n_atoms=n_tot,
        mode=str(cfg.mode),
        cutoff_info=cutoff_info,
        surface_threshold=int(cfg.surface_threshold),
        cn_arr=cn_arr,
        atoms=atoms,
        sites=sites,
    )

    # ---- 4) Console summary
    def _n(k): return len(sites.get(k, []))
    print("=== CN & Active-site Inventory (facet-free) ===")
    print(f"File: {input_file}")
    print(f"Atoms: {n_tot}")
    print(f"Mode: {cfg.mode}  ({cutoff_info})")
    print(f"Surface threshold: CN < {cfg.surface_threshold}  → surface atoms = {len(S)}")
    hist, edges_h = np.histogram(cn_arr, bins=np.arange(cn_arr.min(), cn_arr.max() + 2) - 0.5)
    centers = (edges_h[:-1] + edges_h[1:]) * 0.5
    print("CN histogram (CN: count):")
    for c, h in zip(centers.astype(int), hist):
        print(f"  {c:2d}: {h}")

    print("\nSite counts:")
    print(f"  atop (exposed surface atoms): {_n('atop')}")
    print(f"  bridge (2-fold)           : {_n('bridge')}")
    print(f"  hollow3 (triangles)       : {_n('hollow3')}")
    print(f"  hollow4_square (100-like) : {_n('hollow4_square')}")
    print(f"  hollow4_rect (110-like)   : {_n('hollow4_rect')}")
    print(f"  kinks (atoms)             : {_n('kink')}")
    print(f"  B5 (5-atom step ensembles): {_n('B5')}")

    print(f"\nWrote: {out_path}")
    if cfg_path:
        print(f"(Config: {cfg_path})")


if __name__ == "__main__":
    main()
