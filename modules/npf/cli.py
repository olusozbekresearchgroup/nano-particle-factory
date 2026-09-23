# Copyright (c) 2026 Olus Ozbek Research Group
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import argparse
from pathlib import Path

from .builders import (
    build_ru_cube,
    build_ru_cuboctahedron,
    build_ru_decahedron,
    build_ru_hexagonal_shape,
    build_ru_dodecahedron,
    build_ru_icosahedron,
    build_ru_morphed_spherical,
    build_ru_octahedron,
    build_ru_sphere,
    build_ru_wulff,
)
from .common import DEFAULT_A, DEFAULT_C, cluster_output_dir, format_hkil, parse_surfaces_expr
from .input import load_input_spec, build_from_spec


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Ru nanoparticles.")
    p.add_argument("input_file", nargs="?", default=None, help="Input file describing the nanoparticle build.")
    p.add_argument("--input", dest="input_file_opt", type=str, default=None, help="Input file describing the nanoparticle build.")
    p.add_argument(
        "--shape",
        choices=[
            "wulff",
            "ico",
            "sphere",
            "cube",
            "octahedron",
            "decahedron",
            "dodecahedron",
            "morphed_spherical",
            "cuboct",
            "hexagonal_prism",
            "truncated_hexagonal_prism",
            "hexagonal_bipyramid",
            "truncated_hexagonal_bipyramid",
            "nanorod",
            "hexagonal_platelet",
        ],
        default=None,
        help="NP geometry: Wulff (hcp), Icosahedron, Sphere, Cube, Octahedron, Decahedron, Dodecahedron, Morphed spherical, Cuboctahedron, or one of the hexagonal presets.",
    )
    p.add_argument("--atoms", type=int, default=None, help="Target atom count (approx).")
    p.add_argument("--radius", type=float, default=None, help="Target circumscribed radius (Å).")
    p.add_argument("--a", type=float, default=None, help="hcp a (Å).")
    p.add_argument("--c", type=float, default=None, help="hcp c (Å).")
    p.add_argument("--tag", type=str, default=None, help="Output tag prefix.")
    p.add_argument(
        "--surfaces",
        type=str,
        default=None,
        help="Overrides 'hkil=gamma,...' (eV/Å²), e.g. '0001=0.180,10-10=0.182'",
    )
    p.add_argument("--facet-mode", choices=["minimal", "top", "full"], default=None, help="Facet set strategy.")
    p.add_argument("--extra-facets", choices=["none", "auto"], default=None, help="Add a denser facet set by heuristic.")
    p.add_argument("--sphericity", type=float, default=None, help="0..1 mix toward isotropic γ to round the Wulff (1 = most round).")
    p.add_argument("--radius-tol", type=float, default=None, help="Radius tolerance (Å).")
    p.add_argument("--max-refine", type=int, default=None, help="Max refinement iterations.")
    p.add_argument("--radius-policy", choices=["closest", "at_least", "at_most"], default=None)
    p.add_argument("--radial-trim", type=float, default=None, help="If set, remove atoms with r > value (Å) after Wulff build.")
    p.add_argument("--ico-shells", type=int, default=None, help="Explicit Mackay shells (overrides atoms/radius fit).")
    p.add_argument("--cube-layers", type=int, default=None, help="Explicit cube layer count (for --shape cube).")
    p.add_argument("--octa-length", type=int, default=None, help="Explicit octahedron length (for --shape octahedron).")
    p.add_argument("--octa-cutoff", type=int, default=0, help="Octahedron cutoff (for --shape octahedron).")
    p.add_argument("--p", type=int, default=None, help="Decahedron p parameter.")
    p.add_argument("--q", type=int, default=None, help="Decahedron q parameter.")
    p.add_argument("--r", type=int, default=None, help="Decahedron r parameter.")
    p.add_argument("--morph", type=float, default=None, help="Morphed spherical shape factor in [0, 1].")
    p.add_argument("--cuboct-layers", type=int, default=None, help="Number of fcc shells (for --shape cuboct).")
    p.add_argument("--verbose", action="store_true", default=None)
    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)

    input_path = args.input_file_opt or args.input_file
    if input_path:
        spec = load_input_spec(input_path)
        merged = dict(spec)
        for key, value in vars(args).items():
            if key in {"input_file", "input_file_opt"} or value is None:
                continue
            merged[key] = value
        res = build_from_spec(merged)
        print(f"[{merged.get('shape', 'wulff')}] Built {res.formula} with {res.atoms_count} atoms. R≈{res.actual_radius:.2f} Å")
        return

    shape = args.shape or "wulff"
    a = DEFAULT_A if args.a is None else args.a
    c = DEFAULT_C if args.c is None else args.c
    tag = args.tag or "Ru_NP"
    facet_mode = args.facet_mode or "top"
    extra_facets = args.extra_facets or "none"
    sphericity = 0.0 if args.sphericity is None else args.sphericity
    radius_tol = 0.3 if args.radius_tol is None else args.radius_tol
    max_refine = 12 if args.max_refine is None else args.max_refine
    radius_policy = args.radius_policy or "closest"

    if shape == "ico":
        if args.atoms is None and args.radius is None and args.ico_shells is None:
            p.error("For --shape ico, provide one of --atoms, --radius, or --ico-shells.")
        res = build_ru_icosahedron(
            a_hcp=a,
            tag=tag,
            target_atoms=args.atoms,
            target_radius=args.radius,
            ico_shells=args.ico_shells,
            verbose=bool(args.verbose),
        )
        print(f"[ico] Built {res.formula} with {res.atoms_count} atoms. R≈{res.actual_radius:.2f} Å")
        print(_output_file_list(tag, "ico"))
        return

    if shape == "sphere":
        if args.atoms is None and args.radius is None:
            p.error("For --shape sphere, provide --atoms or --radius.")
        res = build_ru_sphere(
            a=a,
            b=None,
            c=c,
            alpha=None,
            beta=None,
            gamma=None,
            crystal_structure="hexagonal",
            crystal_variant=None,
            tag=tag,
            target_atoms=args.atoms,
            target_radius=args.radius,
            verbose=bool(args.verbose),
        )
        print(f"[sphere] Built {res.formula} with {res.atoms_count} atoms after spherical shave. R≈{res.actual_radius:.2f} Å")
        print(_output_file_list(tag, "sphere"))
        return

    if shape == "cube":
        if args.atoms is None and args.radius is None and args.cube_layers is None:
            p.error("For --shape cube, provide one of --atoms, --radius, or --cube-layers.")
        res = build_ru_cube(
            a_ref=a,
            tag=tag,
            target_atoms=args.atoms,
            target_radius=args.radius,
            cube_layers=args.cube_layers,
            verbose=bool(args.verbose),
        )
        print(f"[cube] Built {res.formula} with {res.atoms_count} atoms. R≈{res.actual_radius:.2f} Å")
        print(_output_file_list(tag, "cube"))
        return

    if shape == "octahedron":
        if args.atoms is None and args.radius is None and args.octa_length is None:
            p.error("For --shape octahedron, provide one of --atoms, --radius, or --octa-length.")
        res = build_ru_octahedron(
            a_ref=a,
            tag=tag,
            target_atoms=args.atoms,
            target_radius=args.radius,
            octa_length=args.octa_length,
            octa_cutoff=args.octa_cutoff,
            verbose=bool(args.verbose),
        )
        print(f"[octahedron] Built {res.formula} with {res.atoms_count} atoms. R≈{res.actual_radius:.2f} Å")
        print(_output_file_list(tag, "octahedron"))
        return

    if shape == "decahedron":
        if args.atoms is None and args.radius is None and not all(v is not None for v in [args.p, args.q, args.r]):
            p.error("For --shape decahedron, provide one of --atoms, --radius, or --p/--q/--r.")
        res = build_ru_decahedron(
            a_ref=a,
            tag=tag,
            target_atoms=args.atoms,
            target_radius=args.radius,
            p=args.p,
            q=args.q,
            r=args.r,
            verbose=bool(args.verbose),
        )
        print(f"[decahedron] Built {res.formula} with {res.atoms_count} atoms. R≈{res.actual_radius:.2f} Å")
        print(_output_file_list(tag, "decahedron"))
        return

    if shape == "dodecahedron":
        if args.atoms is None and args.radius is None:
            p.error("For --shape dodecahedron, provide --atoms or --radius.")
        res = build_ru_dodecahedron(
            a_ref=a,
            tag=tag,
            target_atoms=args.atoms,
            target_radius=args.radius,
            verbose=bool(args.verbose),
        )
        print(f"[dodecahedron] Built {res.formula} with {res.atoms_count} atoms. R≈{res.actual_radius:.2f} Å")
        print(_output_file_list(tag, "dodecahedron"))
        return

    if shape == "morphed_spherical":
        if args.atoms is None and args.radius is None:
            p.error("For --shape morphed_spherical, provide --atoms or --radius.")
        res = build_ru_morphed_spherical(
            a_ref=a,
            tag=tag,
            target_atoms=args.atoms,
            target_radius=args.radius,
            morph=0.0 if args.morph is None else args.morph,
            verbose=bool(args.verbose),
        )
        print(f"[morphed_spherical] Built {res.formula} with {res.atoms_count} atoms. R≈{res.actual_radius:.2f} Å")
        print(_output_file_list(tag, "morphed_spherical"))
        return

    if shape == "cuboct":
        if args.atoms is None and args.radius is None and args.cuboct_layers is None:
            p.error("For --shape cuboct, provide one of --atoms, --radius, or --cuboct-layers.")
        res = build_ru_cuboctahedron(
            a_ref=a,
            tag=tag,
            target_atoms=args.atoms,
            target_radius=args.radius,
            cuboct_layers=args.cuboct_layers,
            verbose=bool(args.verbose),
        )
        print(f"[cuboct] Built {res.formula} with {res.atoms_count} atoms. R≈{res.actual_radius:.2f} Å")
        print(_output_file_list(tag, "cuboct"))
        return

    if shape in {
        "hexagonal_prism",
        "truncated_hexagonal_prism",
        "hexagonal_bipyramid",
        "truncated_hexagonal_bipyramid",
        "nanorod",
        "hexagonal_platelet",
    }:
        if args.atoms is None and args.radius is None:
            p.error(f"For --shape {shape}, provide --atoms or --radius.")
        res = build_ru_hexagonal_shape(
            shape=shape,
            a_ref=a,
            c_ref=c,
            tag=tag,
            target_atoms=args.atoms,
            target_radius=args.radius,
            verbose=bool(args.verbose),
        )
        print(f"[{shape}] Built {res.formula} with {res.atoms_count} atoms. R≈{res.actual_radius:.2f} Å")
        print(_output_file_list(tag, shape))
        return

    overrides = parse_surfaces_expr(args.surfaces) if args.surfaces else None
    res = build_ru_wulff(
        target_atoms=args.atoms,
        target_radius=args.radius,
        a=a,
        c=c,
        overrides=overrides,
        tag=tag,
        facet_mode=facet_mode,
        radius_tol=radius_tol,
        max_refine=max_refine,
        verbose=bool(args.verbose),
        radius_policy=radius_policy,
        sphericity=sphericity,
        extra_facets=extra_facets,
        radial_trim=args.radial_trim,
    )
    print(f"[wulff] Built {res.formula} with {res.atoms_count} atoms. R≈{res.actual_radius:.2f} Å")
    if res.surfaces_hkl:
        print("Facets (hkil) : gamma (eV/Å²)")
        for (h, k, l), g in zip(res.surfaces_hkl, res.gammas):
            print(f"  {format_hkil(h, k, l)} : {g:.3f}")
    print(_output_file_list(args.tag, "wulff"))


if __name__ == "__main__":
    main()
def _output_file_list(tag: str, suffix: str) -> str:
    output_dir = cluster_output_dir()
    return "Files: {0}, {1}, {2}".format(
        output_dir / "POSCAR",
        output_dir / f"{tag}.xyz",
        output_dir / f"{tag}_{suffix}_meta.json",
    )
