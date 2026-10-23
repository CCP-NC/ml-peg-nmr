"""Shared analysis helpers for the NMR shielding benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

from ase import Atoms
from ase.io import read, write
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

# Plotly's default trace colours. ml-peg's parity plots leave trace colours unset,
# so Plotly takes them from this list in plotting order (the models, then DFT),
# starting again after the last one. The NMR figures drawn here follow the same
# rule, so each method keeps one colour across the page.
TRACE_COLOURS = pc.qualitative.Plotly


def trace_colour(position: int) -> str:
    """
    Get the colour Plotly gives the trace at a position in plotting order.

    Parameters
    ----------
    position
        Position of the method among those plotted (models first, then DFT).

    Returns
    -------
    str
        Colour, from ``TRACE_COLOURS``, repeating after the last one.
    """
    return TRACE_COLOURS[position % len(TRACE_COLOURS)]


def weighted_fit(
    sigma: np.ndarray, delta: np.ndarray, w: np.ndarray
) -> tuple[float, float, float]:
    """
    Fit delta = a * sigma + b by weighted least squares.

    Parameters
    ----------
    sigma, delta, w
        Shieldings, experimental shifts and weights.

    Returns
    -------
    tuple[float, float, float]
        Slope, intercept and weighted RMS residual.
    """
    a, b = np.polyfit(sigma, delta, 1, w=np.sqrt(w))
    rmse = float(np.sqrt(np.sum(w * (delta - (a * sigma + b)) ** 2) / np.sum(w)))
    return float(a), float(b), rmse


def weighted_mae(sigma: np.ndarray, delta: np.ndarray, w: np.ndarray) -> float:
    """
    Weighted mean absolute residual about the weighted least-squares line.

    Uses the same line as ``weighted_fit``, so RMSE and MAE describe one fit.

    Parameters
    ----------
    sigma, delta, w
        Shieldings, experimental shifts and weights.

    Returns
    -------
    float
        Weighted MAE (ppm).
    """
    a, b, _ = weighted_fit(sigma, delta, w)
    return float(np.sum(w * np.abs(delta - (a * sigma + b))) / np.sum(w))


def label_model_column(table_file: Path) -> None:
    """
    Display "Model" rather than ml-peg's default "MLIP" table header.

    These are not interatomic potentials. Only the column's display name
    changes; its id ("MLIP"), which the app uses as a lookup key, is untouched.

    Parameters
    ----------
    table_file
        Metrics-table JSON written by ``build_table``.
    """
    table = json.loads(table_file.read_text())
    for column in table["columns"]:
        if column["id"] == "MLIP":
            column["name"] = "Model"
    table_file.write_text(json.dumps(table))


def plot_shielding_fit(
    sigma: dict[str, np.ndarray],
    delta: np.ndarray,
    weights: np.ndarray,
    title: str,
    x_label: str,
    hoverdata: dict[str, list],
    slope: float | None = None,
) -> go.Figure:
    """
    Plot predicted shieldings against experimental shifts with each fitted line.

    The paper-style view of the parity plot (Kellner et al. 2025, Fig. 2): the
    x-axis is the predicted shielding, and each method's least-squares line
    delta = a * sigma + b, the conversion the table's errors are measured
    about, is drawn with its slope and intercept. The errors are the vertical
    scatter about the line, so they equal the table's RMSE and MAE.

    Parameters
    ----------
    sigma
        Method name to predicted shieldings, one per point (methods with
        missing values are skipped).
    delta
        Experimental shifts, one per point.
    weights
        Regression weights, one per point.
    title
        Figure title.
    x_label
        Label of the x-axis.
    hoverdata
        Hover label name to one value per point.
    slope
        Fixed slope, with only the intercept fitted. Default is None, which fits
        both.

    Returns
    -------
    go.Figure
        Scatter of each method with its fitted line and slope/intercept text,
        all three toggled together from the legend.
    """
    customdata = list(zip(*hoverdata.values(), strict=True))
    hovertemplate = "<b>σ: </b>%{x}<br><b>δ expt: </b>%{y}<br>" + "".join(
        f"<b>{key}: </b>%{{customdata[{i}]}}<br>" for i, key in enumerate(hoverdata)
    )
    fig = go.Figure()
    # Colours by position among the methods plotted, as in the parity plots.
    plotted = {name: s for name, s in sigma.items() if not np.isnan(s).any()}
    shown = {name: (trace_colour(i), s) for i, (name, s) in enumerate(plotted.items())}
    # Slope/intercept text sits in the empty lower-left corner (the slope is
    # negative), one line per method, as text traces in the method's legend
    # group so that hiding a method from the legend hides its text too.
    x_all = np.concatenate([s for _, s in shown.values()])
    x_text = x_all.min()
    y_step = 0.065 * (delta.max() - delta.min())
    y_text = delta.min() + y_step * np.arange(len(shown), 0, -1)
    for (name, (colour, s)), y in zip(shown.items(), y_text, strict=True):
        if slope is None:
            a, b, _ = weighted_fit(s, delta, weights)
        else:
            a = slope
            b = np.average(delta - a * s, weights=weights)
        fig.add_trace(
            go.Scatter(
                x=np.round(s, 3),
                y=delta,
                name=name,
                legendgroup=name,
                mode="markers",
                marker={"color": colour},
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        )
        ends = np.array([s.min(), s.max()])
        for trace in (
            go.Scatter(x=ends, y=a * ends + b, mode="lines", line={"color": colour}),
            go.Scatter(
                x=[x_text],
                y=[y],
                mode="text",
                text=[f"<b>{name}</b>: slope {a:.4f}, intercept {b:.2f}"],
                textposition="middle right",
                textfont={"color": colour},
            ),
        ):
            trace.update(legendgroup=name, showlegend=False, hoverinfo="skip")
            fig.add_trace(trace)
    fig.add_annotation(
        text=(
            "Linear regression, δ = slope × σ + intercept"
            if slope is None
            else f"δ = slope × σ + intercept, slope fixed at {slope:g}"
        ),
        x=x_text,
        y=delta.min() + y_step * (len(shown) + 1),
        xanchor="left",
        showarrow=False,
    )
    fig.update_layout(
        title={"text": title},
        xaxis={"title": {"text": x_label}},
        yaxis={"title": {"text": "Experimental shift / ppm"}},
    )
    return fig


def write_structures(
    structure_files: dict[str, Path], out_dir: Path, url_prefix: str
) -> dict[str, str]:
    """
    Save light copies of structures for the app's structure viewer.

    Only species, positions and cell are kept, so the files stay small.

    Parameters
    ----------
    structure_files
        Name to the input structure file (e.g. ``.magres``).
    out_dir
        App data directory to write ``<name>.xyz`` files into.
    url_prefix
        URL of ``out_dir`` as served by the app (``/assets/...``).

    Returns
    -------
    dict[str, str]
        Name to the URL of its saved structure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = {}
    for name, path in structure_files.items():
        atoms = read(path)
        bare = Atoms(
            atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=True
        )
        write(out_dir / f"{name}.xyz", bare, format="extxyz")
        urls[name] = f"{url_prefix}/{name}.xyz"
    return urls
