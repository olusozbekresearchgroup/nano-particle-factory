# Copyright (c) 2026 Olus Ozbek Research Group
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, Mapping

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
from .builders.from_crystal import build_from_crystal
from .common import DEFAULT_A, DEFAULT_C, cluster_output_dir, parse_surfaces_expr
from .crystal import load_crystal
from .geometry_map import is_shape_allowed, normalize_shape_key, normalize_structure_class, normalize_subtype


def load_input_spec(path: str | Path) -> dict[str, Any]:
    input_path = Path(path).expanduser().resolve()
    spec = parse_input_text(input_path.read_text(encoding="utf-8"))
    crystal_file = spec.get("crystal_file")
    if crystal_file:
        crystal_path = Path(str(crystal_file)).expanduser()
        if not crystal_path.is_absolute():
            crystal_path = (input_path.parent / crystal_path).resolve()
        spec["crystal_file"] = str(crystal_path)
    return spec


def resolve_output_file_name(spec: Mapping[str, Any]) -> str:
    output_file_name = spec.get("output_file_name")
    if output_file_name:
        return str(output_file_name)
    element = spec.get("element") or "Ru"
    shape = spec.get("shape", "wulff")
    atoms = spec.get("atoms")
    atom_part = f"{int(atoms)}" if atoms is not None else "NA"
    return f"{element}_{shape}_{atom_part}.xyz"


def parse_input_text(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Input file must contain a JSON object at the top level.")
        return data
    except json.JSONDecodeError:
        return _load_kv_input(text)


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _load_kv_input(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid input line: {raw_line!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid input line: {raw_line!r}")
        data[key] = _parse_scalar(value)
    if not data:
        raise ValueError("Input file is empty.")
    return data


def _pick(spec: Mapping[str, Any], key: str, fallback: Any):
    value = spec.get(key, fallback)
    return fallback if value is None else value


def build_from_spec(spec: Mapping[str, Any]):
    spec = dict(spec)
    shape = normalize_shape_key(spec.get("shape", "wulff"))
    crystal_structure = normalize_structure_class(spec.get("crystal_structure", "hexagonal"))
    crystal_variant = normalize_subtype(spec.get("crystal_variant"))
    output_file_name = resolve_output_file_name(spec)
    tag = spec.get("tag") or output_file_name or "Ru_NP"
    if "output_file_name" not in spec or spec.get("output_file_name") in {None, ""}:
        spec["output_file_name"] = output_file_name
        spec["tag"] = Path(output_file_name).stem
        tag = spec["tag"]
    a = float(_pick(spec, "a", DEFAULT_A))
    c = float(_pick(spec, "c", DEFAULT_C))
    working_root = spec.get("working_path") or Path.cwd()
    output_dir = cluster_output_dir(working_root, tag)
    crystal_file = spec.get("crystal_file")

    if not crystal_file and not is_shape_allowed(crystal_structure, crystal_variant, shape):
        raise ValueError(
            f"Shape '{shape}' is not available for crystal structure '{crystal_structure}'"
            + (f"/'{crystal_variant}'" if crystal_variant else "")
        )

    original_cwd = Path.cwd()
    try:
        os.chdir(output_dir)
        if crystal_file:
            crystal = load_crystal(str(crystal_file))
            overrides = spec.get("surfaces")
            if isinstance(overrides, str) and overrides:
                overrides = parse_surfaces_expr(overrides)
            elif not overrides:
                overrides = None
            return build_from_crystal(
                crystal.atoms,
                source_file=str(crystal.path),
                shape=shape,
                tag=tag,
                crystal_structure=crystal_structure,
                crystal_variant=crystal_variant,
                target_atoms=spec.get("atoms"),
                target_radius=spec.get("radius"),
                surfaces=overrides,
                morph=float(_pick(spec, "morph", 0.0)),
                radial_trim=spec.get("radial_trim"),
                verbose=bool(spec.get("verbose", False)),
            )

        if shape == "ico":
            surfaces = spec.get("surfaces", "")
            if isinstance(surfaces, str) and surfaces:
                raise ValueError("Icosahedron spec does not use 'surfaces'.")
            return build_ru_icosahedron(
                a_hcp=a,
                tag=tag,
                target_atoms=spec.get("atoms"),
                target_radius=spec.get("radius"),
                ico_shells=spec.get("ico_shells"),
                verbose=bool(spec.get("verbose", False)),
            )

        if shape == "cube":
            return build_ru_cube(
                a_ref=a,
                tag=tag,
                target_atoms=spec.get("atoms"),
                target_radius=spec.get("radius"),
                cube_layers=spec.get("cube_layers"),
                verbose=bool(spec.get("verbose", False)),
            )

        if shape == "octahedron":
            return build_ru_octahedron(
                a_ref=a,
                tag=tag,
                target_atoms=spec.get("atoms"),
                target_radius=spec.get("radius"),
                octa_length=spec.get("octa_length"),
                octa_cutoff=int(_pick(spec, "octa_cutoff", 0)),
                verbose=bool(spec.get("verbose", False)),
            )

        if shape == "decahedron":
            return build_ru_decahedron(
                a_ref=a,
                tag=tag,
                target_atoms=spec.get("atoms"),
                target_radius=spec.get("radius"),
                p=spec.get("p"),
                q=spec.get("q"),
                r=spec.get("r"),
                verbose=bool(spec.get("verbose", False)),
            )

        if shape == "dodecahedron":
            return build_ru_dodecahedron(
                a_ref=a,
                tag=tag,
                target_atoms=spec.get("atoms"),
                target_radius=spec.get("radius"),
                verbose=bool(spec.get("verbose", False)),
            )

        if shape == "morphed_spherical":
            return build_ru_morphed_spherical(
                a_ref=a,
                tag=tag,
                target_atoms=spec.get("atoms"),
                target_radius=spec.get("radius"),
                morph=float(_pick(spec, "morph", 0.0)),
                verbose=bool(spec.get("verbose", False)),
            )

        if shape == "cuboct":
            return build_ru_cuboctahedron(
                a_ref=a,
                tag=tag,
                target_atoms=spec.get("atoms"),
                target_radius=spec.get("radius"),
                cuboct_layers=spec.get("cuboct_layers"),
                verbose=bool(spec.get("verbose", False)),
            )

        if shape in {
            "hexagonal_prism",
            "truncated_hexagonal_prism",
            "hexagonal_bipyramid",
            "truncated_hexagonal_bipyramid",
            "nanorod",
            "hexagonal_platelet",
        }:
            return build_ru_hexagonal_shape(
                shape=shape,
                a_ref=a,
                c_ref=c,
                tag=tag,
                target_atoms=spec.get("atoms"),
                target_radius=spec.get("radius"),
                verbose=bool(spec.get("verbose", False)),
            )

        if shape == "sphere":
            return build_ru_sphere(
                a=a,
                b=spec.get("b"),
                c=c,
                alpha=spec.get("alpha"),
                beta=spec.get("beta"),
                gamma=spec.get("gamma"),
                crystal_structure=crystal_structure,
                crystal_variant=crystal_variant,
                tag=tag,
                target_atoms=spec.get("atoms"),
                target_radius=spec.get("radius"),
                verbose=bool(spec.get("verbose", False)),
            )

        overrides = spec.get("surfaces")
        if isinstance(overrides, str) and overrides:
            overrides = parse_surfaces_expr(overrides)
        elif not overrides:
            overrides = None

        return build_ru_wulff(
            target_atoms=spec.get("atoms"),
            target_radius=spec.get("radius"),
            a=a,
            c=c,
            overrides=overrides,
            tag=tag,
            facet_mode=_pick(spec, "facet_mode", "top"),
            radius_tol=float(_pick(spec, "radius_tol", 0.3)),
            max_refine=int(_pick(spec, "max_refine", 12)),
            verbose=bool(spec.get("verbose", False)),
            radius_policy=_pick(spec, "radius_policy", "closest"),
            sphericity=float(_pick(spec, "sphericity", 0.0)),
            extra_facets=_pick(spec, "extra_facets", "none"),
            radial_trim=spec.get("radial_trim"),
        )
    finally:
        os.chdir(original_cwd)


def run_input_file(path: str | Path):
    return build_from_spec(load_input_spec(path))
