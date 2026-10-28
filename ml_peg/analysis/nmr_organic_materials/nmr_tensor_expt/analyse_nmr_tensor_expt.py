"""Analyse predicted 13C shielding tensors against experimental principal components."""

from __future__ import annotations

import json
from pathlib import Path

from ase.io import read
import numpy as np
import pandas as pd
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.nmr import (
    csa_params,
    label_model_column,
    plot_shielding_fit,
    principal,
    weighted_fit,
    write_structures,
)
from ml_peg.analysis.utils.utils import load_metrics_config, write_struct_info
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK = Path("nmr_organic_materials") / "nmr_tensor_expt"
CALC_PATH = CALCS_ROOT / BENCHMARK / "outputs"
# Experimental reference data, from the benchmark's input zip.
DATA_PATH = (
    download_s3_data(
        filename="nmr_tensor_expt.zip",
        key="inputs/nmr_organic_materials/nmr_tensor_expt/nmr_tensor_expt.zip",
    )
    / "nmr_tensor_expt"
)
SITES_PATH = DATA_PATH / "experimental_tensors.csv"
OUT_PATH = APP_ROOT / "data" / BENCHMARK
# Structures for the app's viewer, served by the app from its data folder.
STRUCT_URL = f"/assets/{BENCHMARK.as_posix()}/structures"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

METRIC = "¹³C tensor RMSE"
MAE_METRIC = "¹³C tensor MAE"
COMPONENTS = ("δ11", "δ22", "δ33")
# CSA parameter -> table column (weight 0; alternative views of the tensor).
# The isotropic shift comes first: the mean of the three predicted components,
# converted with the tensor fit, so it differs from the Isotropic shifts benchmark.
# Then both conventions for the tensor's shape, as pairs: Haeberlen (ζ, η) and
# Herzfeld-Berger (Ω, κ).
CSA_LABELS = {
    "iso": ("Isotropic shift δiso", "ppm"),
    "zeta": ("Reduced anisotropy ζ", "ppm"),
    "eta": ("Asymmetry η", None),
    "span": ("Span Ω", "ppm"),
    "skew": ("Skew κ", None),
}
CSA = {key: f"{label} RMSE" for key, (label, _) in CSA_LABELS.items()}
# Structures in the data file but left out of the scoring, with the reason.
EXCLUDED = {
    "HXACAN09_pbe-d3bj": (
        "left out for consistency with the ShiftML papers, which use 19 of these "
        "structures (Kellner et al. 2025, SI Section S7)"
    ),
}
SITES = pd.read_csv(SITES_PATH)
SITES = SITES[~SITES.structure.isin(EXCLUDED)].reset_index(drop=True)
# Input structure (with its GIPAW-PBE tensors) per structure folder.
INPUTS = {
    path.parent.name: path
    for path in sorted((DATA_PATH / "structures" / "C").glob("*/*.magres"))
}
# Experimental components, one row per site: (sites, 3).
EXPT = SITES[["expt_d11_ppm", "expt_d22_ppm", "expt_d33_ppm"]].to_numpy()
# Hover labels: one point per site (shape plots), or three per site, site by site
# (plots of all principal components).
SITE_HOVER = {
    "Structure": SITES.structure.str.replace("_pbe-d3bj", "").tolist(),
    "Site": SITES.site_label.astype(str).tolist(),
}
COMPONENT_HOVER = {
    key: [v for v in values for _ in COMPONENTS] for key, values in SITE_HOVER.items()
} | {"Component": list(COMPONENTS) * len(SITES)}


def reference_components() -> np.ndarray:
    """
    Average the GIPAW-PBE principal shieldings of the input structures per site.

    Read from the benchmark's input data, so the reference does not depend on
    which models were run.

    Returns
    -------
    np.ndarray
        Principal shieldings, shape (sites, 3).
    """
    sigma = {
        s: principal(read(INPUTS[s]).arrays["ms"].reshape(-1, 9))
        for s in SITES.structure.unique()
    }
    return np.array(
        [
            sigma[row.structure][[int(i) for i in row.atom_indices.split(";")]].mean(0)
            for row in SITES.itertuples()
        ]
    )


def site_components(model_name: str) -> np.ndarray:
    """
    Average one model's principal shieldings over each site's symmetry copies.

    Parameters
    ----------
    model_name
        Model whose calculation outputs are read.

    Returns
    -------
    np.ndarray
        Principal shieldings, shape (sites, 3); NaN where outputs are missing.
    """
    out = []
    cache: dict[str, np.ndarray | None] = {}
    for row in SITES.itertuples():
        if row.structure not in cache:
            path = CALC_PATH / model_name / f"C__{row.structure}.extxyz"
            cache[row.structure] = (
                principal(read(path).arrays["sigma_tensor_pred"])
                if path.exists()
                else None
            )
        values = cache[row.structure]
        idx = [int(i) for i in row.atom_indices.split(";")]
        out.append(np.full(3, np.nan) if values is None else values[idx].mean(0))
    return np.array(out)


def fit(sigma: np.ndarray):
    """
    Fit and score the shielding-to-shift line over all components of all sites.

    Parameters
    ----------
    sigma
        Principal shieldings, shape (sites, 3).

    Returns
    -------
    tuple[float, float, float]
        Slope, intercept and RMSE (ppm).
    """
    s = sigma.ravel()
    return weighted_fit(s, EXPT.ravel(), np.ones_like(s))


def tensor_errors(sigma: np.ndarray) -> tuple[float, float]:
    """
    Get the RMSE and MAE over all components, about the fitted line.

    Parameters
    ----------
    sigma
        Principal shieldings, shape (sites, 3).

    Returns
    -------
    tuple[float, float]
        RMSE and MAE (ppm).
    """
    a, b, rmse = fit(sigma)
    return rmse, float(np.abs(EXPT - (a * sigma + b)).mean())


EXPT_CSA = csa_params(EXPT)


def predicted_csa(sigma: np.ndarray) -> dict:
    """
    Get CSA parameters of a method's tensors, converted with its fitted line.

    Zeta uses the experiment's choice of dzz for each site (the same principal
    component on both sides). Eta keeps each tensor's own ordering, so it stays in
    [0, 1] (with the experiment's dzz it can exceed 1); it is unstable where zeta
    is small. The isotropic shift, span and skew do not depend on the ordering.

    Parameters
    ----------
    sigma
        Principal shieldings, shape (sites, 3).

    Returns
    -------
    dict
        As ``csa_params``; ``zz`` is the tensor's own choice of dzz.
    """
    a, b, _ = fit(sigma)
    delta = a * sigma + b
    return csa_params(delta) | {"zeta": csa_params(delta, EXPT_CSA["zz"])["zeta"]}


def csa_rmse(sigma: np.ndarray) -> dict[str, float]:
    """
    Get the RMSE of each CSA parameter against experiment.

    Parameters
    ----------
    sigma
        Principal shieldings, shape (sites, 3).

    Returns
    -------
    dict[str, float]
        ``CSA`` key to RMSE over sites.
    """
    pred = predicted_csa(sigma)
    return {
        key: float(np.sqrt(np.mean((pred[key] - EXPT_CSA[key]) ** 2))) for key in CSA
    }


@pytest.fixture(scope="module")
def predictions() -> dict[str, np.ndarray]:
    """
    Load site-averaged principal shieldings for every model, plus DFT.

    Returns
    -------
    dict[str, np.ndarray]
        Model name (and "DFT") to principal shieldings, shape (sites, 3).
    """
    out = {model: site_components(model) for model in MODELS}
    out["DFT"] = reference_components()
    return out


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_parity_13C_tensor.json",
    title="¹³C tensor principal components (all)",
    x_label="Predicted shift / ppm",
    y_label="Experimental shift / ppm",
    hoverdata=COMPONENT_HOVER,
)
def parity(predictions: dict[str, np.ndarray]) -> dict[str, list]:
    """
    Get parity-plot data: each method's components converted with its own line.

    Parameters
    ----------
    predictions
        Site-averaged principal shieldings.

    Returns
    -------
    dict[str, list]
        Experimental and predicted shifts, three per site.
    """
    results = {"ref": EXPT.ravel().tolist()}
    for name, sigma in predictions.items():
        if np.isnan(sigma).any():
            continue
        a, b, _ = fit(sigma)
        results[name] = (a * sigma + b).ravel().round(3).tolist()
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "nmr_tensor_expt_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(predictions: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """
    Get the tensor RMSE and MAE, and the shape-column RMSEs, for each model.

    Parameters
    ----------
    predictions
        Site-averaged principal shieldings.

    Returns
    -------
    dict[str, dict[str, float]]
        Metric name to model name to value; NaN where outputs are missing.
    """
    errors = {
        model: (
            (np.nan, np.nan)
            if np.isnan(predictions[model]).any()
            else tensor_errors(predictions[model])
        )
        for model in MODELS
    }
    out = {
        metric: {model: round(e[i], 3) for model, e in errors.items()}
        for i, metric in enumerate((METRIC, MAE_METRIC))
    }
    for key, metric in CSA.items():
        out[metric] = {
            model: (
                np.nan
                if np.isnan(predictions[model]).any()
                else round(csa_rmse(predictions[model])[key], 3)
            )
            for model in MODELS
        }
    return out


def test_nmr_tensor_expt(
    metrics: dict[str, dict],
    parity: dict[str, list],
    predictions: dict[str, np.ndarray],
) -> None:
    """
    Build the table, plots and structure links for the app.

    The table and the all-component parity plot are written by their fixtures'
    decorators. Also written here: one parity plot per CSA parameter (one point
    per site); the shielding-vs-experiment plot of all principal components with
    each method's fitted line (Kellner et al. 2025, Fig. 3); and, per plot and in
    point order, the structure file and the atoms (all symmetry copies of the
    site) to highlight when a point is clicked.

    Parameters
    ----------
    metrics
        Table metrics.
    parity
        Parity-plot data (all components).
    predictions
        Site-averaged principal shieldings.
    """
    label_model_column(OUT_PATH / "nmr_tensor_expt_metrics_table.json")

    # Elements present, for ml-peg's element filtering (scored structures only).
    write_struct_info([INPUTS[s] for s in SITES.structure.unique()], OUT_PATH)

    methods = {m: s for m, s in predictions.items() if not np.isnan(s).any()}
    for key, (label, unit) in CSA_LABELS.items():
        suffix = f" / {unit}" if unit else ""

        def csa_parity(key=key) -> dict[str, list]:
            """
            Get one CSA parameter's parity data.

            Parameters
            ----------
            key
                ``CSA`` key.

            Returns
            -------
            dict[str, list]
                Experimental and predicted values per site.
            """
            return {"ref": EXPT_CSA[key].round(4).tolist()} | {
                m: predicted_csa(s)[key].round(4).tolist() for m, s in methods.items()
            }

        plot_parity(
            filename=OUT_PATH / f"figure_parity_{key}.json",
            title=f"¹³C {label}",
            x_label=f"Predicted {label}{suffix}",
            y_label=f"Experimental {label}{suffix}",
            hoverdata=SITE_HOVER,
        )(csa_parity)()

    plot_shielding_fit(
        {m: s.ravel() for m, s in predictions.items()},
        EXPT.ravel(),
        np.ones(EXPT.size),
        title="¹³C predicted principal shieldings vs experiment",
        x_label="Predicted principal shielding σ11, σ22, σ33 / ppm",
        hoverdata=COMPONENT_HOVER,
    ).write_json(OUT_PATH / "figure_shielding_13C_tensor.json")

    urls = write_structures(
        {s: INPUTS[s] for s in SITES.structure.unique()},
        OUT_PATH / "structures",
        STRUCT_URL,
    )
    per_site = {
        "structs": [urls[s] for s in SITES.structure],
        "highlights": [
            [int(i) for i in a.split(";")] for a in SITES.atom_indices.astype(str)
        ],
    }
    links = {
        "site": per_site,
        # Three points per site (d11, d22, d33), site by site.
        "component": {
            k: [v for v in vals for _ in range(3)] for k, vals in per_site.items()
        },
    }
    (OUT_PATH / "structure_links.json").write_text(json.dumps(links))
