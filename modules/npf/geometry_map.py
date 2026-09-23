from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_TABLE = ROOT / "crystal_type_vs_np_geom.dat"

CRYSTAL_CLASS_ALIASES = {
    "cubic": "cubic",
    "hexagonal": "hexagonal",
    "tetragonal": "tetragonal",
    "orthorhombic": "orthorhombic",
    "rhombohedral": "rhombohedral",
    "monoclinic": "monoclinic",
    "triclinic": "triclinic",
    "hcp": "hexagonal",
    "fcc": "cubic",
    "bcc": "cubic",
}

SUBTYPE_ALIASES = {
    "simple cubic": "simple_cubic",
    "sc": "simple_cubic",
    "simple_cubic": "simple_cubic",
    "bcc": "bcc",
    "fcc": "fcc",
    "diamond cubic": "diamond_cubic",
    "diamond_cubic": "diamond_cubic",
    "zincblende": "zincblende",
    "rocksalt": "rocksalt",
    "nacl": "rocksalt",
    "cscl": "cscl",
    "fluorite": "fluorite",
    "caf2": "fluorite",
    "perovskite": "perovskite",
    "abo3": "perovskite",
    "spinel": "spinel",
    "hcp": "hcp",
    "wurtzite": "wurtzite",
    "graphite": "graphite",
    "layered hexagonal": "graphite",
    "alb2-type": "alb2",
    "alb2": "alb2",
    "nias-type": "nias",
    "nias": "nias",
    "simple tetragonal": "simple_tetragonal",
    "body-centered tetragonal": "bct",
    "bct": "bct",
    "rutile": "rutile",
    "tio2": "rutile",
    "anatase": "anatase",
    "zircon": "zircon",
    "zrsi o4": "zircon",
    "simple orthorhombic": "simple_orthorhombic",
    "base-centered orthorhombic": "base_centered_orthorhombic",
    "body-centered orthorhombic": "body_centered_orthorhombic",
    "face-centered orthorhombic": "face_centered_orthorhombic",
    "olivine-type": "olivine",
    "olivine": "olivine",
    "distorted perovskite": "perovskite_distorted_orthorhombic",
    "rhombohedral": "rhombohedral",
    "corundum": "corundum",
    "al2o3": "corundum",
    "calcite": "calcite",
    "caco3": "calcite",
    "ilmenite": "ilmenite",
    "bi/sb-type a7": "a7",
    "a7": "a7",
    "simple monoclinic": "simple_monoclinic",
    "base-centered monoclinic": "base_centered_monoclinic",
    "baddeleyite": "baddeleyite",
    "zro2": "baddeleyite",
    "general monoclinic": "general_monoclinic",
    "simple triclinic": "simple_triclinic",
    "molecular triclinic": "molecular_triclinic",
    "framework triclinic": "framework_triclinic",
}

SHAPE_LABELS = {
    "sphere": "sphere",
    "wulff": "wulff",
    "cube": "cube",
    "octahedron": "octahedron",
    "ico": "icosahedron",
    "decahedron": "decahedron",
    "dodecahedron": "dodecahedron",
    "morphed_spherical": "morphed spherical",
    "cuboct": "cuboctahedron",
    "hexagonal_prism": "hexagonal prism",
    "truncated_hexagonal_prism": "truncated hexagonal prism",
    "hexagonal_bipyramid": "hexagonal bipyramid",
    "truncated_hexagonal_bipyramid": "truncated hexagonal bipyramid",
    "nanorod": "nanorod / elongated hexagonal prism",
    "hexagonal_platelet": "hexagonal platelet / nanodisc",
}

SHAPE_KEYS_BY_LABEL = {label.lower(): key for key, label in SHAPE_LABELS.items()}

GEOMETRY_NAME_ORDER = [
    ("cuboctahedron", "cuboct"),
    ("icosahedron", "ico"),
    ("dodecahedron", "dodecahedron"),
    ("octahedron", "octahedron"),
    ("decahedron", "decahedron"),
    ("morphed spherical", "morphed_spherical"),
    ("hexagonal prism", "hexagonal_prism"),
    ("truncated hexagonal prism", "truncated_hexagonal_prism"),
    ("hexagonal bipyramid", "hexagonal_bipyramid"),
    ("truncated hexagonal bipyramid", "truncated_hexagonal_bipyramid"),
    ("nanorod / elongated hexagonal prism", "nanorod"),
    ("hexagonal platelet / nanodisc", "hexagonal_platelet"),
    ("wulff", "wulff"),
    ("sphere", "sphere"),
    ("cube", "cube"),
]


def normalize_structure_class(value: str | None) -> str:
    text = (value or "").strip().lower()
    return CRYSTAL_CLASS_ALIASES.get(text, text or "hexagonal")


def normalize_subtype(value: str | None) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip().lower())
    text = text.replace(" / ", "/")
    aliases = {
        "simple cubic / sc": "simple_cubic",
        "body-centered tetragonal / bct": "bct",
        "simple orthorhombic": "simple_orthorhombic",
        "base-centered orthorhombic": "base_centered_orthorhombic",
        "body-centered orthorhombic": "body_centered_orthorhombic",
        "face-centered orthorhombic": "face_centered_orthorhombic",
        "graphite / layered hexagonal": "graphite",
        "alb2-type": "alb2",
        "nias-type": "nias",
        "rutile / tio2": "rutile",
        "anatase / tio2": "anatase",
        "zircon / zrsi o4": "zircon",
        "rocksalt / nacl": "rocksalt",
        "fluorite / caf2": "fluorite",
        "perovskite / abo3": "perovskite",
        "bi/sb-type a7": "a7",
        "baddeleyite / zro2": "baddeleyite",
        "molecular triclinic": "molecular_triclinic",
        "framework triclinic": "framework_triclinic",
    }
    if text in aliases:
        return aliases[text]
    return SUBTYPE_ALIASES.get(text, text.replace(" ", "_").replace("-", "_"))


def shape_label(shape_key: str) -> str:
    return SHAPE_LABELS.get(shape_key, shape_key)


def normalize_shape_key(value: str | None) -> str:
    text = (value or "").strip().lower()
    if not text:
        return "sphere"
    if text in SHAPE_LABELS:
        return text
    if text in SHAPE_KEYS_BY_LABEL:
        return SHAPE_KEYS_BY_LABEL[text]
    aliases = {
        "ico": "ico",
        "cuboct": "cuboct",
        "morphed spherical": "morphed_spherical",
        "morphed_spherical": "morphed_spherical",
        "hexagonal prism": "hexagonal_prism",
        "truncated hexagonal prism": "truncated_hexagonal_prism",
        "hexagonal bipyramid": "hexagonal_bipyramid",
        "truncated hexagonal bipyramid": "truncated_hexagonal_bipyramid",
        "nanorod / elongated hexagonal prism": "nanorod",
        "hexagonal platelet / nanodisc": "hexagonal_platelet",
        "icosahedron": "ico",
        "cuboctahedron": "cuboct",
        "sphere": "sphere",
        "wulff": "wulff",
        "cube": "cube",
        "octahedron": "octahedron",
        "decahedron": "decahedron",
        "dodecahedron": "dodecahedron",
    }
    return aliases.get(text, text)


def shape_items_for(structure_class: str, subtype: str | None = None) -> list[tuple[str, str]]:
    allowed = allowed_shape_keys(structure_class, subtype)
    return [(shape_label(key), key) for key in allowed]


@lru_cache(maxsize=1)
def _parsed_table() -> dict[tuple[str, str], list[str]]:
    if not GEOMETRY_TABLE.exists():
        return {}
    mapping: dict[tuple[str, str], list[str]] = {}
    for raw in GEOMETRY_TABLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("|") or line.startswith("| ---"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 3:
            continue
        crystal_class = normalize_structure_class(parts[0].replace("**", "").replace("*", ""))
        subtype = normalize_subtype(parts[1].replace("**", "").replace("*", ""))
        geom_cell = parts[2].lower()
        shapes: list[str] = []
        for geom_name, shape_key in GEOMETRY_NAME_ORDER:
            if geom_name in geom_cell and shape_key not in shapes:
                shapes.append(shape_key)
        mapping[(crystal_class, subtype)] = shapes
    return mapping


def allowed_shape_keys(structure_class: str, subtype: str | None = None) -> list[str]:
    cls = normalize_structure_class(structure_class)
    sub = normalize_subtype(subtype) if subtype else ""
    table = _parsed_table()
    if (cls, sub) in table:
        return table[(cls, sub)]
    # Fallback: union across all subtypes for the class.
    union: list[str] = []
    for (row_cls, _row_sub), shapes in table.items():
        if row_cls != cls:
            continue
        for shape in shapes:
            if shape not in union:
                union.append(shape)
    if union:
        return union
    return ["sphere"]


def is_shape_allowed(structure_class: str, subtype: str | None, shape_key: str) -> bool:
    return shape_key in allowed_shape_keys(structure_class, subtype)
