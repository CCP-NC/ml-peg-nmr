"""Analyse the cocaine structure-identification benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase import Atoms
from ase.io import read, write
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.nmr import (
    identification,
    label_model_column,
    plot_shielding_fit,
    trace_colour,
)
from ml_peg.analysis.utils.utils import load_metrics_config, write_struct_info
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK = Path("nmr_organic_materials") / "structure_id_cocaine"
CALC_PATH = CALCS_ROOT / BENCHMARK / "outputs"
# Experimental reference data, from the benchmark's input zip.
DATA_PATH = (
    download_s3_data(
        filename="structure_id_cocaine.zip",
        key="inputs/nmr_organic_materials/structure_id_cocaine/structure_id_cocaine.zip",
    )
    / "structure_id_cocaine"
)
SITES_PATH = DATA_PATH / "cocaine_experimental_shifts.csv"
CANDIDATES_PATH = DATA_PATH / "cocaine_QuantumEspresso.xyz"
OUT_PATH = APP_ROOT / "data" / BENCHMARK
STRUCT_URL = f"/assets/{BENCHMARK.as_posix()}/structures"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)
MARGIN = "Identification margin"
RANK = "Rank of correct structure"
RMSE_EXPT = "¹H RMSE (correct structure)"
DFT = "GIPAW-PBE"
H_PER_MOLECULE = 21
CORRECT_CANDIDATE = 0
NOISE_FLOOR = (0.33, 0.16)  # GIPAW-vs-experiment spread, Salager et al. 2010
SITES = pd.read_csv(SITES_PATH)
GROUPS = [[int(i) - 1 for i in p.split(";")] for p in SITES.protons.astype(str)]
Y = SITES.expt_shift_ppm.to_numpy()
SITE_HOVER = {"Site": SITES.site.astype(str).tolist()}


def site_shieldings(per_h: np.ndarray) -> np.ndarray:
    """
    Group one molecule's H shieldings into the assigned sites.

    As in the recipe: the H atoms are stored molecule-interleaved, so the
    first molecule's 21 protons are ``per_h.reshape(21, -1)[:, 0]``.

    Parameters
    ----------
    per_h
        Shieldings of all H atoms in the cell, in file order.

    Returns
    -------
    np.ndarray
        Mean shielding per site (17 values).
    """
    per_molecule = per_h.reshape(H_PER_MOLECULE, -1)[:, 0]
    return np.array([per_molecule[g].mean() for g in GROUPS])


def predicted_shifts(sigma: np.ndarray) -> np.ndarray:
    """
    Convert site shieldings to shifts: slope -1, intercept per candidate.

    Parameters
    ----------
    sigma
        Site shieldings for one candidate.

    Returns
    -------
    np.ndarray
        Predicted shifts (ppm), with the same mean as experiment.
    """
    return -sigma + (Y.mean() + sigma.mean())


def rmse_vs_expt(sigma: np.ndarray) -> float:
    """
    RMSE against experiment with slope -1 and a per-candidate intercept.

    Parameters
    ----------
    sigma
        Site shieldings for one candidate.

    Returns
    -------
    float
        RMSE (ppm).
    """
    return float(np.sqrt(np.mean((predicted_shifts(sigma) - Y) ** 2)))


@pytest.fixture(scope="module")
def candidates() -> dict:
    """
    Load site shieldings for every candidate, per model and for GIPAW-PBE.

    Models that were not run, or failed on a candidate, get NaN.

    Returns
    -------
    dict
        Site shieldings (candidates, sites) per method, and the
        candidate names.
    """
    # Reference and names from the input data, independent of the models run.
    inputs = read(CANDIDATES_PATH, ":")
    sigma = {
        DFT: np.array(
            [
                site_shieldings(np.asarray(a.arrays["CS"])[a.numbers == 1])
                for a in inputs
            ]
        )
    }
    names = [a.info.get("NAME", "") for a in inputs]
    nan = np.full((len(inputs), len(Y)), np.nan)
    for model in MODELS:
        sigma[model] = nan.copy()
        for path in sorted((CALC_PATH / model).glob("candidate_*.extxyz")):
            atoms = read(path)
            i = int(atoms.info["candidate"])
            is_h = atoms.numbers == 1
            sigma[model][i] = site_shieldings(atoms.arrays["sigma_iso_pred"][is_h])
    return {"sigma": sigma, "names": names}


@pytest.fixture(scope="module")
def results(candidates: dict) -> dict:
    """
    Score every candidate against experiment, for each method.

    Parameters
    ----------
    candidates
        Site shieldings per candidate.

    Returns
    -------
    dict
        Per-candidate RMSEs, rank and margin per method, and the candidate
        names.
    """
    out = {"names": candidates["names"], "methods": {}}
    for method, sigma in candidates["sigma"].items():
        rmse = np.array([rmse_vs_expt(s) for s in sigma])
        rank, margin = identification(rmse, CORRECT_CANDIDATE)
        out["methods"][method] = {
            "rmse_vs_expt": rmse.round(4).tolist(),
            "rank": rank,
            "margin": round(margin, 4),
        }
    return out


@pytest.fixture
@build_table(
    filename=OUT_PATH / "structure_id_cocaine_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(results: dict) -> dict[str, dict[str, float]]:
    """
    Get the benchmark table: margin (scored), rank and RMSE (information).

    Parameters
    ----------
    results
        Benchmark results.

    Returns
    -------
    dict[str, dict[str, float]]
        Metric names and values per model.
    """
    methods = results["methods"]
    return {
        MARGIN: {m: methods[m]["margin"] for m in MODELS},
        RANK: {m: methods[m]["rank"] for m in MODELS},
        RMSE_EXPT: {m: methods[m]["rmse_vs_expt"][CORRECT_CANDIDATE] for m in MODELS},
    }


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_parity_expt_structure.json",
    title="Correct cocaine structure: predicted vs experimental ¹H shifts",
    x_label="Predicted shift / ppm",
    y_label="Experimental shift / ppm",
    hoverdata=SITE_HOVER,
)
def parity(candidates: dict) -> dict[str, list]:
    """
    Get parity data for the correct structure, every model.

    Parameters
    ----------
    candidates
        Site shieldings per candidate.

    Returns
    -------
    dict[str, list]
        Experimental shifts and each model's predicted shifts.
    """
    sigma = candidates["sigma"]
    # Models with no prediction for this structure are left out, so trace
    # colours follow the same order as in the other plots.
    return {"ref": Y.round(3).tolist()} | {
        m: predicted_shifts(sigma[m][CORRECT_CANDIDATE]).round(3).tolist()
        for m in MODELS
        if not np.isnan(sigma[m][CORRECT_CANDIDATE]).any()
    }


def structure_links() -> dict:
    """
    Save the correct structure and the atoms behind each site.

    Returns
    -------
    dict
        ``{"structs": [...], "highlights": [[...], ...]}``, one entry per
        site, in the order of the parity and shielding plots.
    """
    atoms = read(CANDIDATES_PATH, CORRECT_CANDIDATE)
    # Same grouping as ``site_shieldings``: the first molecule's H atoms.
    per_molecule = np.flatnonzero(atoms.numbers == 1).reshape(H_PER_MOLECULE, -1)
    out_dir = OUT_PATH / "structures"
    out_dir.mkdir(parents=True, exist_ok=True)
    write(
        out_dir / "correct_structure.xyz",
        Atoms(atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=True),
        format="extxyz",
    )
    return {
        "structs": [f"{STRUCT_URL}/correct_structure.xyz"] * len(GROUPS),
        "highlights": [per_molecule[g, 0].tolist() for g in GROUPS],
    }


def lollipop_figure(results: dict) -> go.Figure:
    """
    Build the lollipop figure, one row per candidate.

    RMSE against experiment for GIPAW-PBE and every model, with the
    DFT-vs-experiment noise floor shaded. Candidates run down the page so that
    each gets enough room for one lollipop per method.

    Parameters
    ----------
    results
        Benchmark results.

    Returns
    -------
    go.Figure
        Lollipop figure.
    """
    # Models first, then DFT, with colours as in the parity plots; methods with
    # no results (a failed model) are left out, as there.
    methods = [
        m
        for m in (*MODELS, DFT)
        if not np.isnan(results["methods"][m]["rmse_vs_expt"]).all()
    ]
    colours = {m: trace_colour(i) for i, m in enumerate(methods)}
    n = len(results["names"])
    idx = np.arange(n)
    # Keep one candidate's lollipops within 40% of its row, so the gap to the next
    # candidate is clearly wider than the spacing inside a group.
    offsets = np.linspace(-0.2, 0.2, len(methods)) if len(methods) > 1 else [0.0]
    row_px = 60
    fig = go.Figure()
    lo, hi = NOISE_FLOOR[0] - NOISE_FLOOR[1], NOISE_FLOOR[0] + NOISE_FLOOR[1]
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"symbol": "square", "size": 12, "color": "#d9d9d6"},
            name=(
                f"DFT vs experiment noise floor "
                f"({NOISE_FLOOR[0]} ± {NOISE_FLOOR[1]} ppm)"
            ),
        )
    )

    def stems(y, x, colour, name):
        """
        Add one series of horizontal lollipops (stems from zero plus markers).

        Parameters
        ----------
        y
            Candidate positions.
        x
            Lollipop lengths (ppm).
        colour
            Series colour.
        name
            Legend and hover label.
        """
        xs, ys = [], []
        for yi, xi in zip(y, x, strict=True):
            xs += [0, xi, None]
            ys += [yi, yi, None]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": colour, "width": 2},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=name,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker={
                    "color": colour,
                    "size": 8,
                    "line": {"color": "white", "width": 1},
                },
                name=name,
                legendgroup=name,
                customdata=np.column_stack([idx, results["names"]]),
                hovertemplate=(
                    f"<b>{name}</b><br>candidate %{{customdata[0]}} "
                    "(%{customdata[1]})<br>%{x:.3f} ppm<extra></extra>"
                ),
            )
        )

    for method, offset in zip(methods, offsets, strict=True):
        stems(
            idx + offset,
            results["methods"][method]["rmse_vs_expt"],
            colours[method],
            f"{method} vs experiment",
        )
    # Shade alternate rows so each candidate's lollipops read as one group.
    for i in idx[1::2]:
        fig.add_hrect(
            y0=i - 0.5, y1=i + 0.5, fillcolor="#f2f2ef", line_width=0, layer="below"
        )
    # Mark the correct structure's row.
    fig.add_hrect(
        y0=CORRECT_CANDIDATE - 0.5,
        y1=CORRECT_CANDIDATE + 0.5,
        fillcolor="#fff3c4",
        opacity=0.7,
        line_width=0,
        layer="below",
    )
    # Drawn after the row shading so the band shows on every row.
    fig.add_vrect(
        x0=lo,
        x1=hi,
        fillcolor="#d9d9d6",
        opacity=0.6,
        line_width=0,
        layer="below",
    )
    ink, grid = "#2b2b29", "#e6e6e3"
    labels = [f"{i} (correct)" if i == CORRECT_CANDIDATE else str(i) for i in idx]
    fig.update_yaxes(
        tickmode="array",
        tickvals=idx.tolist(),
        ticktext=labels,
        # Exactly one row per candidate, the first at the top.
        range=[n - 0.5, -0.5],
        showgrid=False,
        zeroline=False,
        linecolor=grid,
    )
    fig.update_yaxes(title_text="Cocaine candidate structure")
    fig.update_xaxes(title_text="RMSE / ppm", rangemode="tozero", gridcolor=grid)
    margin = {"l": 120, "r": 20, "t": 140, "b": 60}
    plot_px = n * row_px
    fig.update_layout(
        template="plotly_white",
        height=plot_px + margin["t"] + margin["b"],
        title={"text": "¹H shift RMSE against experiment", "x": 0.5},
        margin=margin,
        # Legend 10 px above the plot, below the title, with room for two lines.
        legend={
            "orientation": "h",
            "y": 1 + 10 / plot_px,
            "yanchor": "bottom",
            "x": 0,
        },
        font={"color": ink},
    )
    return fig


def test_structure_id_cocaine(
    metrics: dict, parity: dict, results: dict, candidates: dict
) -> None:
    """
    Build the table, figures and structure links for the app.

    Parameters
    ----------
    metrics
        Table metrics (written by the decorator).
    parity
        Parity data (written by the decorator).
    results
        Benchmark results.
    candidates
        Site shieldings per candidate.
    """
    label_model_column(OUT_PATH / "structure_id_cocaine_metrics_table.json")
    lollipop_figure(results).write_json(OUT_PATH / "figure_lollipop.json")
    sigma = candidates["sigma"]
    plot_shielding_fit(
        {m: sigma[m][CORRECT_CANDIDATE] for m in (*MODELS, DFT)},
        Y,
        np.ones_like(Y),
        title="Correct cocaine structure: predicted ¹H shielding vs experiment",
        x_label="Predicted isotropic shielding / ppm",
        hoverdata=SITE_HOVER,
        slope=-1.0,
    ).write_json(OUT_PATH / "figure_shielding_expt_structure.json")
    write_struct_info(CANDIDATES_PATH, OUT_PATH)
    (OUT_PATH / "structure_links.json").write_text(json.dumps(structure_links()))
