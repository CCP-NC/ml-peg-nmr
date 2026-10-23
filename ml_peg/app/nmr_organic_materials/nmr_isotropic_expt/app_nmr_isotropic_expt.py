"""Run NMR Isotropic Shifts benchmark app."""

from __future__ import annotations

from dash.html import B, Div, P

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.load import read_plot
from ml_peg.app.utils.nmr import (
    link_structures,
    plots_from_table_column,
    read_links,
)

BENCHMARK_NAME = "NMR Isotropic Shifts"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/nmr_organic_materials.html"
    "#nmr-isotropic-shifts"
)
DATA_PATH = APP_ROOT / "data" / "nmr_organic_materials" / "nmr_isotropic_expt"
INFO_PATH = DATA_PATH / "info.json"
# Table column (e.g. "¹H RMSE") -> nucleus key of its plots.
NUCLEI = {"1H": "¹H", "13C": "¹³C", "15N": "¹⁵N"}
COLUMN_KEYS = {
    f"{label} {stat}": key for stat in ("RMSE", "MAE") for key, label in NUCLEI.items()
}


class NMRIsotropicExptApp(BaseApp):
    """NMR isotropic shift benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        # RMSE and MAE columns of a nucleus open the same two plots.
        plots = {
            metric: [
                read_plot(
                    DATA_PATH / f"figure_{kind}_{key}.json",
                    id=f"{BENCHMARK_NAME}-{kind}-{metric}",
                )
                for kind in ("parity", "shielding")
            ]
            for metric, key in COLUMN_KEYS.items()
        }
        plots_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plots=plots,
        )
        links = read_links(DATA_PATH / "structure_links.json")
        for metric, key in COLUMN_KEYS.items():
            if key in links:
                link_structures(
                    plots[metric], f"{BENCHMARK_NAME}-struct-placeholder", links[key]
                )


def get_app() -> NMRIsotropicExptApp:
    """
    Get NMR isotropic shift benchmark app layout and callback registration.

    Returns
    -------
    NMRIsotropicExptApp
        Benchmark layout and callback registration.
    """
    return NMRIsotropicExptApp(
        name=BENCHMARK_NAME,
        description=(
            "Isotropic ¹H, ¹³C and ¹⁵N chemical shifts of 49 molecular crystals, "
            "predicted by machine-learned shielding models and compared with "
            "experiment."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "nmr_isotropic_expt_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
            P(
                [
                    B("Click a cell for two plots:"),
                    " predicted shift against experimental shift, and predicted "
                    "shielding against experimental shift with each method's fitted "
                    "line, slope and intercept. Click a data point to see its crystal "
                    "structure, with the site highlighted.",
                ]
            ),
            P(
                [
                    B("Thresholds:"),
                    " Good is set near GIPAW-PBE DFT's own error against the same "
                    "experiments; Bad at twice that error. Both can be changed in the "
                    "Thresholds boxes above.",
                ]
            ),
        ],
        info_path=INFO_PATH,
    )
