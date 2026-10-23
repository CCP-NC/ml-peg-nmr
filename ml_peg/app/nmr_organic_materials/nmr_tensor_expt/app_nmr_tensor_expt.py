"""Run NMR Shift Tensors benchmark app."""

from __future__ import annotations

from dash.html import B, Div, P

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.load import read_plot
from ml_peg.app.utils.nmr import (
    TEXT_STYLE,
    link_structures,
    plots_from_table_column,
    read_links,
)

BENCHMARK_NAME = "NMR Shift Tensors"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/nmr_organic_materials.html"
    "#nmr-shift-tensors"
)
DATA_PATH = APP_ROOT / "data" / "nmr_organic_materials" / "nmr_tensor_expt"
INFO_PATH = DATA_PATH / "info.json"
TENSOR_METRICS = ("¹³C tensor RMSE", "¹³C tensor MAE")
# Shape columns (weight 0) -> key of their parity plots.
CSA_KEYS = {
    "Isotropic shift δiso RMSE": "iso",
    "Reduced anisotropy ζ RMSE": "zeta",
    "Asymmetry η RMSE": "eta",
    "Span Ω RMSE": "span",
    "Skew κ RMSE": "skew",
}


class NMRTensorExptApp(BaseApp):
    """NMR 13C tensor benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        plots = {
            metric: [
                read_plot(
                    DATA_PATH / f"figure_{kind}_13C_tensor.json",
                    id=f"{BENCHMARK_NAME}-{kind}-{metric}",
                )
                for kind in ("parity", "shielding")
            ]
            for metric in TENSOR_METRICS
        } | {
            metric: [
                read_plot(
                    DATA_PATH / f"figure_parity_{key}.json",
                    id=f"{BENCHMARK_NAME}-parity-{key}",
                )
            ]
            for metric, key in CSA_KEYS.items()
        }
        plots_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plots=plots,
        )
        links = read_links(DATA_PATH / "structure_links.json")
        struct_id = f"{BENCHMARK_NAME}-struct-placeholder"
        if links:
            for metric in TENSOR_METRICS:  # three points per site
                link_structures(plots[metric], struct_id, links["component"])
            for metric in CSA_KEYS:  # one point per site
                link_structures(plots[metric], struct_id, links["site"])


def get_app() -> NMRTensorExptApp:
    """
    Get NMR 13C tensor benchmark app layout and callback registration.

    Returns
    -------
    NMRTensorExptApp
        Benchmark layout and callback registration.
    """
    return NMRTensorExptApp(
        name=BENCHMARK_NAME,
        description=(
            "Principal components of the ¹³C chemical-shift tensor for 118 carbon "
            "sites in 19 molecular crystals, predicted by machine-learned "
            "shielding models and compared with experiment."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "nmr_tensor_expt_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
            P(
                [
                    B("Click a cell for its plots:"),
                    " a tensor RMSE or MAE cell shows predicted against experimental "
                    "components (all three together), and predicted shielding against "
                    "experimental shift with each method's fitted line, slope and "
                    "intercept; δiso (not scored) from the same tensors; a shape "
                    "column (not scored) shows the parity plots of ζ with η "
                    "(Haeberlen) and Ω with κ (Herzfeld–Berger), the two conventions "
                    "in use. Click a data point to see its crystal structure, with the "
                    "site highlighted.",
                ],
                style=TEXT_STYLE,
            ),
            P(
                [
                    B("Choice of δzz:"),
                    " in Haeberlen ordering, δzz is whichever of δ11 and δ33 lies "
                    "farther from δiso. When the two are almost equally far, a few "
                    "ppm of numerical error can make the prediction pick the other "
                    "one and flip the sign of its ζ with respect to the experimental "
                    "value, although the tensors agree. As an example, for "
                    "L-serine's carboxyl carbon, every predicted component is within "
                    "5 ppm of the experimental values, yet its own ζ is 129 ppm off. "
                    "So the predicted ζ is measured along the experiment's δzz. η "
                    "keeps each tensor's own ordering, so it stays between 0 and 1.",
                ],
                style=TEXT_STYLE,
            ),
            P(
                [
                    B("Thresholds:"),
                    " Good is set near GIPAW-PBE DFT's own error against the same "
                    "experiments; Bad at twice that error. Both can be changed in the "
                    "Thresholds boxes above.",
                ],
                style=TEXT_STYLE,
            ),
        ],
        info_path=INFO_PATH,
    )
