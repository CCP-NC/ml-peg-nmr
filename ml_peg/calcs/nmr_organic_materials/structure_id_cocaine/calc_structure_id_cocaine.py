"""
Predict 1H shieldings for the cocaine structure-identification benchmark.

Based on the atomistic-cookbook recipe "NMR-shielding-driven structure determination
with ShiftML3" (shiftml-structure-match): 29 candidate crystal structures of cocaine
with GIPAW-PBE shieldings (``CS`` array), from ``ShiftML_poly.zip`` released with
ShiftML2 (Cordova et al., J. Phys. Chem. C 126, 16710 (2022); Zenodo record 7097427,
CC-BY-4.0; also on Materials Cloud, record j2fka-sda13).

For each model and candidate, the ensemble-mean and per-member isotropic shieldings
are written per atom, one extxyz file per candidate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.calcs.utils.nmr import bare_copy
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_nmr_models, load_nmr_predictor

OUT_PATH = Path(__file__).parent / "outputs"
MODELS = get_nmr_models(current_models)


def cocaine_candidates() -> list[Atoms]:
    """
    Download (once) and read the 29 cocaine candidates.

    Returns
    -------
    list[Atoms]
        Candidate structures with the GIPAW-PBE ``CS`` array.
    """
    data_path = (
        download_s3_data(
            filename="structure_id_cocaine.zip",
            key="inputs/nmr_organic_materials/structure_id_cocaine/structure_id_cocaine.zip",
        )
        / "structure_id_cocaine"
    )
    return read(data_path / "cocaine_QuantumEspresso.xyz", ":")


@pytest.mark.fast
@pytest.mark.parametrize("mlip", MODELS.items())
def test_structure_id_cocaine(mlip: tuple[str, dict[str, Any]]) -> None:
    """
    Predict isotropic shieldings for every cocaine candidate with one model.

    Parameters
    ----------
    mlip
        Name of the NMR model and its entry in ``nmr_models.yml``.
    """
    model_name, config = mlip
    predictor = load_nmr_predictor(config, precision="high")
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    for i, reference in enumerate(cocaine_candidates()):
        atoms = bare_copy(reference)
        try:
            members = np.asarray(predictor.get_cs_iso_ensemble(atoms), dtype=float)
        except Exception as exc:
            warn(f"{model_name} failed on candidate {i}: {exc}", stacklevel=2)
            members = np.full((len(atoms), 1), np.nan)
        atoms.arrays["sigma_iso_members"] = members
        atoms.arrays["sigma_iso_pred"] = members.mean(axis=1)
        atoms.info["candidate"] = i
        atoms.info["name"] = reference.info.get("NAME", "")
        write(write_dir / f"candidate_{i:02d}.extxyz", atoms)
