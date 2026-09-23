# Copyright (c) 2026 Olus Ozbek Research Group
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read


@dataclass(frozen=True)
class CrystalSource:
    path: Path
    atoms: Atoms
    formula: str
    species: tuple[str, ...]
    crystal_system: str
    subtype: str | None
    spacegroup_number: int | None
    spacegroup_symbol: str | None


def _crystal_system_from_spacegroup(number: int) -> str:
    if number <= 2:
        return "triclinic"
    if number <= 15:
        return "monoclinic"
    if number <= 74:
        return "orthorhombic"
    if number <= 142:
        return "tetragonal"
    if number <= 167:
        return "rhombohedral"
    if number <= 194:
        return "hexagonal"
    return "cubic"


def _fallback_crystal_system(atoms: Atoms) -> str:
    a, b, c, alpha, beta, gamma = [float(x) for x in atoms.cell.cellpar()]
    length_scale = max(a, b, c, 1.0)
    len_tol = length_scale * 1e-3
    ang_tol = 0.1

    def close(x: float, y: float, tol: float) -> bool:
        return abs(x - y) <= tol

    all90 = all(close(v, 90.0, ang_tol) for v in (alpha, beta, gamma))
    eq_ab = close(a, b, len_tol)
    eq_bc = close(b, c, len_tol)
    eq_ac = close(a, c, len_tol)
    if all90 and eq_ab and eq_bc:
        return "cubic"
    if eq_ab and close(alpha, 90.0, ang_tol) and close(beta, 90.0, ang_tol) and close(gamma, 120.0, ang_tol):
        return "hexagonal"
    if all90 and eq_ab:
        return "tetragonal"
    if all90:
        return "orthorhombic"
    if eq_ab and eq_bc and eq_ac and close(alpha, beta, ang_tol) and close(beta, gamma, ang_tol):
        return "rhombohedral"
    if sum(close(v, 90.0, ang_tol) for v in (alpha, beta, gamma)) == 2:
        return "monoclinic"
    return "triclinic"


def _composition_counts(atoms: Atoms) -> tuple[int, ...]:
    counts: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    values = list(counts.values())
    if not values:
        return ()
    from math import gcd
    from functools import reduce

    divisor = reduce(gcd, values)
    return tuple(sorted(value // divisor for value in values))


def _guess_subtype(atoms: Atoms, spacegroup_number: int | None) -> str | None:
    if spacegroup_number is None:
        return None
    species_count = len(set(atoms.get_chemical_symbols()))
    ratios = _composition_counts(atoms)

    if species_count == 1:
        return {
            194: "hcp",
            225: "fcc",
            229: "bcc",
            221: "simple_cubic",
            227: "diamond_cubic",
        }.get(spacegroup_number)
    if species_count == 2:
        if spacegroup_number == 186:
            return "wurtzite"
        if spacegroup_number == 216 and ratios == (1, 1):
            return "zincblende"
        if spacegroup_number == 225:
            return "rocksalt" if ratios == (1, 1) else "fluorite" if ratios == (1, 2) else None
        if spacegroup_number == 221 and ratios == (1, 1):
            return "cscl"
        if spacegroup_number == 191 and ratios == (1, 2):
            return "alb2"
        if spacegroup_number == 194 and ratios == (1, 1):
            return "nias"
        if spacegroup_number == 136:
            return "rutile"
    if species_count == 3 and spacegroup_number == 221 and ratios == (1, 1, 3):
        return "perovskite"
    return None


def analyze_crystal(atoms: Atoms, path: str | Path) -> CrystalSource:
    source = atoms.copy()
    source.set_pbc(True)
    number = None
    symbol = None
    crystal_system = _fallback_crystal_system(source)
    try:
        import spglib

        dataset = spglib.get_symmetry_dataset(
            (source.cell.array, source.get_scaled_positions(wrap=True), source.numbers),
            symprec=1e-3,
        )
        if dataset is not None:
            if isinstance(dataset, dict):
                number = int(dataset["number"])
                symbol = str(dataset["international"])
            else:
                number = int(dataset.number)
                symbol = str(dataset.international)
            crystal_system = _crystal_system_from_spacegroup(number)
    except (AttributeError, ImportError, KeyError, TypeError, ValueError):
        pass

    species = tuple(dict.fromkeys(source.get_chemical_symbols()))
    return CrystalSource(
        path=Path(path).expanduser().resolve(),
        atoms=source,
        formula=source.get_chemical_formula(empirical=True),
        species=species,
        crystal_system=crystal_system,
        subtype=_guess_subtype(source, number),
        spacegroup_number=number,
        spacegroup_symbol=symbol,
    )


def load_crystal(path: str | Path) -> CrystalSource:
    source_path = Path(path).expanduser().resolve()
    name = source_path.name.lower()
    if "poscar" in name or "contcar" in name:
        try:
            atoms = read(source_path, format="vasp")
        except (AssertionError, ValueError, IndexError):
            atoms = _read_vasp_fallback(source_path)
    else:
        atoms = read(source_path)
    return analyze_crystal(atoms, source_path)


def _symbols_from_vasp4_comment(comment: str, counts: list[int]) -> list[str]:
    from ase.formula import Formula

    candidates = [comment.strip(), comment.split("_", 1)[0].strip(), comment.split()[0].strip()]
    for candidate in candidates:
        try:
            symbols = list(Formula(candidate).count())
        except (ValueError, IndexError):
            continue
        if len(symbols) == len(counts):
            return symbols
    raise ValueError("VASP 4 POSCAR files require recognizable species in the comment line.")


def _read_vasp_fallback(path: Path) -> Atoms:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 8:
        raise ValueError(f"File is too short to be a POSCAR/CONTCAR file: {path}")

    comment = lines[0].strip()
    scale_values = [float(value) for value in lines[1].split()]
    raw_cell = np.asarray([[float(value) for value in lines[index].split()[:3]] for index in range(2, 5)])
    if len(scale_values) == 1:
        scale = scale_values[0]
        if scale < 0.0:
            scale_vector = np.full(3, (-scale / abs(float(np.linalg.det(raw_cell)))) ** (1.0 / 3.0))
        else:
            scale_vector = np.full(3, scale)
    elif len(scale_values) == 3 and all(value > 0.0 for value in scale_values):
        scale_vector = np.asarray(scale_values, dtype=float)
    else:
        raise ValueError("POSCAR scale must be one number or three positive numbers.")
    cell = raw_cell * scale_vector[np.newaxis, :]

    line_index = 5
    species_tokens = lines[line_index].split()
    try:
        counts = [int(value) for value in species_tokens]
    except ValueError:
        species = species_tokens
        line_index += 1
        counts = [int(value) for value in lines[line_index].split()]
    else:
        species = _symbols_from_vasp4_comment(comment, counts)
    if len(species) != len(counts):
        raise ValueError(f"Species/count mismatch in {path}")

    line_index += 1
    if lines[line_index].strip().lower().startswith("selective"):
        line_index += 1
    mode = lines[line_index].strip().lower()
    line_index += 1
    total = sum(counts)
    coordinate_lines = lines[line_index : line_index + total]
    if len(coordinate_lines) != total:
        raise ValueError(f"File does not contain {total} atomic coordinates: {path}")
    coordinates = np.asarray(
        [[float(value) for value in line.split()[:3]] for line in coordinate_lines],
        dtype=float,
    )

    symbols = [symbol for symbol, count in zip(species, counts, strict=False) for _ in range(count)]
    atoms = Atoms(symbols=symbols, cell=cell, pbc=True)
    if mode.startswith("d"):
        atoms.set_scaled_positions(coordinates)
    elif mode.startswith("c") or mode.startswith("k"):
        atoms.set_positions(coordinates * scale_vector[np.newaxis, :])
    else:
        raise ValueError(f"Unknown POSCAR coordinate mode: {lines[line_index - 1]!r}")
    return atoms
