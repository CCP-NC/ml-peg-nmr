"""Analyse predicted NMR shieldings against experimental isotropic shifts."""

from __future__ import annotations

import json
from pathlib import Path

from ase.io import read
import numpy as np
import pandas as pd
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.nmr import (
    label_model_column,
    plot_shielding_fit,
    weighted_fit,
    weighted_mae,
    write_structures,
)
from ml_peg.analysis.utils.utils import load_metrics_config, write_struct_info
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK = Path("nmr_organic_materials") / "nmr_isotropic_expt"
CALC_PATH = CALCS_ROOT / BENCHMARK / "outputs"
# Experimental reference data, from the benchmark's input zip.
DATA_PATH = (
    download_s3_data(
        filename="nmr_isotropic_expt.zip",
        key="inputs/nmr_organic_materials/nmr_isotropic_expt/nmr_isotropic_expt.zip",
    )
    / "nmr_isotropic_expt"
)
SITES_PATH = DATA_PATH / "experimental_shifts.csv"
OUT_PATH = APP_ROOT / "data" / BENCHMARK
# Structures for the app's viewer, served by the app from its data folder.
STRUCT_URL = f"/assets/{BENCHMARK.as_posix()}/structures"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

NUCLEI = ("1H", "13C", "15N")
PRETTY = {"1H": "¹H", "13C": "¹³C", "15N": "¹⁵N"}
# Metric (table column) names; non-ASCII names are established in ml-peg (e.g. R²).
METRIC = {nucleus: f"{PRETTY[nucleus]} RMSE" for nucleus in PRETTY}
MAE_METRIC = {nucleus: f"{PRETTY[nucleus]} MAE" for nucleus in PRETTY}
SITES = pd.read_csv(SITES_PATH)
# Input structure (with its GIPAW-PBE tensors) per (set, structure folder).
INPUTS = {
    (path.parent.parent.name, path.parent.name): path
    for path in sorted((DATA_PATH / "structures").glob("*/*/*.magres"))
}


def reference_sites() -> pd.DataFrame:
    """
    Average the GIPAW-PBE isotropic shieldings of the input structures per site.

    Read from the benchmark's input data, so the reference does not depend on
    which models were run.

    Returns
    -------
    pd.DataFrame
        ``SITES`` with the GIPAW-PBE shielding in column ``pred``.
    """
    sigma = {
        key: np.trace(read(path).arrays["ms"], axis1=1, axis2=2) / 3
        for key, path in INPUTS.items()
    }
    pred = [
        float(
            np.mean(
                sigma[(row.set, row.structure)][
                    [int(i) for i in row.atom_indices.split(";")]
                ]
            )
        )
        for row in SITES.itertuples()
    ]
    return SITES.assign(pred=pred)


def site_means(model_name: str) -> pd.DataFrame:
    """
    Average one model's predicted shieldings over each site's atoms.

    Parameters
    ----------
    model_name
        Model whose calculation outputs are read.

    Returns
    -------
    pd.DataFrame
        ``SITES`` with the predicted shielding in column ``pred`` (NaN where
        outputs are missing).
    """
    pred = []
    cache: dict[tuple[str, str], np.ndarray | None] = {}
    for row in SITES.itertuples():
        key = (row.set, row.structure)
        if key not in cache:
            path = CALC_PATH / model_name / f"{row.set}__{row.structure}.extxyz"
            cache[key] = read(path).arrays["sigma_iso_pred"] if path.exists() else None
        values = cache[key]
        idx = [int(i) for i in row.atom_indices.split(";")]
        pred.append(np.nan if values is None else float(np.mean(values[idx])))
    return SITES.assign(pred=pred)


@pytest.fixture(scope="module")
def predictions() -> dict[str, pd.DataFrame]:
    """
    Load site-averaged predictions for every model, plus DFT.

    Returns
    -------
    dict[str, pd.DataFrame]
        Model name (and "DFT") to per-site shieldings in column ``pred``.
    """
    out = {model: site_means(model) for model in MODELS}
    out["DFT"] = reference_sites()
    return out


def parity_data(predictions: dict[str, pd.DataFrame], nucleus: str) -> dict[str, list]:
    """
    Build parity-plot data (predicted vs experimental shift) for one nucleus.

    Parameters
    ----------
    predictions
        Output of the ``predictions`` fixture.
    nucleus
        "1H", "13C" or "15N".

    Returns
    -------
    dict[str, list]
        ``ref`` experimental shifts and predicted shifts per model (and DFT),
        each converted with its own global line.
    """
    mask = (SITES.nucleus == nucleus).values
    results = {"ref": SITES.expt_shift_ppm[mask].tolist()}
    for name, df in predictions.items():
        d = df[mask]
        if d.pred.isna().any():
            continue
        a, b, _ = weighted_fit(d.pred.values, d.expt_shift_ppm.values, d.weight.values)
        results[name] = (a * d.pred + b).round(3).tolist()
    return results


def hover(nucleus: str) -> dict[str, list]:
    """
    Hover labels for one nucleus's parity plot.

    Parameters
    ----------
    nucleus
        "1H", "13C" or "15N".

    Returns
    -------
    dict[str, list]
        Structure and site labels per point.
    """
    d = SITES[SITES.nucleus == nucleus]
    return {
        "Structure": d.structure.str.replace("_pbe-d3bj", "").tolist(),
        "Site": d.site_label.tolist(),
    }


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_parity_1H.json",
    title="{} isotropic shifts".format(PRETTY["1H"]),
    x_label="Predicted shift / ppm",
    y_label="Experimental shift / ppm",
    hoverdata=hover("1H"),
)
def parity_1h(predictions: dict[str, pd.DataFrame]) -> dict[str, list]:
    """
    Get 1H parity data.

    Parameters
    ----------
    predictions
        Site-averaged predictions.

    Returns
    -------
    dict[str, list]
        Experimental and predicted shifts.
    """
    return parity_data(predictions, "1H")


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_parity_13C.json",
    title="{} isotropic shifts".format(PRETTY["13C"]),
    x_label="Predicted shift / ppm",
    y_label="Experimental shift / ppm",
    hoverdata=hover("13C"),
)
def parity_13c(predictions: dict[str, pd.DataFrame]) -> dict[str, list]:
    """
    Get 13C parity data.

    Parameters
    ----------
    predictions
        Site-averaged predictions.

    Returns
    -------
    dict[str, list]
        Experimental and predicted shifts.
    """
    return parity_data(predictions, "13C")


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_parity_15N.json",
    title="{} isotropic shifts".format(PRETTY["15N"]),
    x_label="Predicted shift / ppm",
    y_label="Experimental shift / ppm",
    hoverdata=hover("15N"),
)
def parity_15n(predictions: dict[str, pd.DataFrame]) -> dict[str, list]:
    """
    Get 15N parity data.

    Parameters
    ----------
    predictions
        Site-averaged predictions.

    Returns
    -------
    dict[str, list]
        Experimental and predicted shifts.
    """
    return parity_data(predictions, "15N")


@pytest.fixture
@build_table(
    filename=OUT_PATH / "nmr_isotropic_expt_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(predictions: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    """
    Get the RMSE and MAE per nucleus for each model.

    Both are residuals about the same weighted least-squares line.

    Parameters
    ----------
    predictions
        Site-averaged predictions.

    Returns
    -------
    dict[str, dict[str, float]]
        Metric name (RMSE columns, then MAE columns) to model name to value
        (ppm); NaN where outputs are missing.
    """
    results = {METRIC[n]: {} for n in NUCLEI} | {MAE_METRIC[n]: {} for n in NUCLEI}
    for model in MODELS:
        df = predictions[model]
        for nucleus in NUCLEI:
            d = df[df.nucleus == nucleus]
            args = (d.pred.values, d.expt_shift_ppm.values, d.weight.values)
            missing = d.pred.isna().any()
            results[METRIC[nucleus]][model] = (
                np.nan if missing else round(weighted_fit(*args)[2], 3)
            )
            results[MAE_METRIC[nucleus]][model] = (
                np.nan if missing else round(weighted_mae(*args), 3)
            )
    return results


def test_nmr_isotropic_expt(
    metrics: dict[str, dict],
    parity_1h: dict[str, list],
    parity_13c: dict[str, list],
    parity_15n: dict[str, list],
    predictions: dict[str, pd.DataFrame],
) -> None:
    """
    Build the table, plots and structure links for the app.

    The table and parity plots are written by their fixtures' decorators. Also
    written here: per nucleus, the shielding-vs-experiment plot with each
    method's shielding-to-shift line, slope and intercept (Kellner et al. 2025,
    Fig. 2), and ``structure_links.json``, which gives, in parity-plot point
    order, the structure file and the atoms (all symmetry copies of the site) to
    highlight when a point is clicked.

    Parameters
    ----------
    metrics
        Table metrics.
    parity_1h, parity_13c, parity_15n
        Parity-plot data.
    predictions
        Site-averaged predictions.
    """
    label_model_column(OUT_PATH / "nmr_isotropic_expt_metrics_table.json")

    # Elements present, for ml-peg's element filtering.
    write_struct_info(list(INPUTS.values()), OUT_PATH)

    links = {}
    urls = write_structures(
        {f"{key[0]}__{key[1]}": path for key, path in INPUTS.items()},
        OUT_PATH / "structures",
        STRUCT_URL,
    )
    for nucleus in NUCLEI:
        mask = (SITES.nucleus == nucleus).values
        d = SITES[mask]
        plot_shielding_fit(
            {name: df.pred.values[mask] for name, df in predictions.items()},
            d.expt_shift_ppm.values,
            d.weight.values,
            title=f"{PRETTY[nucleus]} predicted isotropic shielding vs experiment",
            x_label="Predicted isotropic shielding / ppm",
            hoverdata=hover(nucleus),
        ).write_json(OUT_PATH / f"figure_shielding_{nucleus}.json")
        links[nucleus] = {
            "structs": [urls[f"{r.set}__{r.structure}"] for r in d.itertuples()],
            "highlights": [
                [int(i) for i in r.atom_indices.split(";")] for r in d.itertuples()
            ],
        }
    (OUT_PATH / "structure_links.json").write_text(json.dumps(links))
