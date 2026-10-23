"""Shared calculation helpers for the NMR shielding benchmarks."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read


def structure_dirs(root: Path, sets: tuple[str, ...]) -> list[tuple[str, Path]]:
    """
    List benchmark structure directories.

    Parameters
    ----------
    root
        Directory containing the ``H``, ``C`` and ``N`` folders.
    sets
        Folders to include.

    Returns
    -------
    list[tuple[str, Path]]
        (set name, structure directory) pairs, sorted.
    """
    return [
        (name, d)
        for name in sets
        for d in sorted(p for p in (root / name).iterdir() if p.is_dir())
    ]


def bare_copy(reference: Atoms) -> Atoms:
    """
    Copy a structure keeping only species, positions and cell.

    This is what reaches the model, so no reference data (e.g. DFT shieldings)
    can leak into a prediction.

    Parameters
    ----------
    reference
        Structure as read from the input data.

    Returns
    -------
    Atoms
        Periodic copy with species, positions and cell only.
    """
    return Atoms(
        reference.numbers,
        positions=reference.positions,
        cell=reference.cell,
        pbc=True,
    )


def read_structure(struct_dir: Path) -> tuple[Atoms, Atoms]:
    """
    Read one benchmark structure and its GIPAW-PBE reference.

    Parameters
    ----------
    struct_dir
        Structure directory containing exactly one ``.magres`` file.

    Returns
    -------
    tuple[Atoms, Atoms]
        The reference (with the ``ms`` shielding-tensor array) and a bare copy
        carrying only species, positions and cell, which is what reaches the
        model.
    """
    magres = sorted(struct_dir.glob("*.magres"))
    if len(magres) != 1:
        raise FileNotFoundError(
            f"Expected one .magres file in {struct_dir}, found {len(magres)}"
        )
    reference = read(magres[0])
    return reference, bare_copy(reference)
