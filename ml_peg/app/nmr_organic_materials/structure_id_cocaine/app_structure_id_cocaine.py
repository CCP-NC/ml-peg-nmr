"""Run the cocaine structure-identification benchmark app."""

from __future__ import annotations

from dash.dcc import Markdown
from dash.html import A, B, Div, P

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.load import read_plot
from ml_peg.app.utils.nmr import (
    TEXT_STYLE,
    link_structures,
    plots_from_table_column,
    read_links,
)

BENCHMARK_NAME = "Structure Identification - Cocaine"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/nmr_organic_materials.html"
    "#structure-identification-cocaine"
)
DATA_PATH = APP_ROOT / "data" / "nmr_organic_materials" / "structure_id_cocaine"
INFO_PATH = DATA_PATH / "info.json"
METRICS = (
    "Identification margin",
    "Rank of correct structure",
    "¹H RMSE (correct structure)",
)
CASE_STUDY_URL = (
    "https://atomistic-cookbook.org/examples/shiftml-structure-match/"
    "shiftml-structure-match.html#the-cocaine-benchmark-dataset"
)

FIGURE_CAPTION = (
    "¹H RMSE against experiment for each of the 29 candidates, from GIPAW-PBE DFT "
    "and each machine-learning model; the yellow shaded row is the correct "
    "structure (candidate 0). The vertical grey band, 0.33 ± 0.16 ppm, is "
    "GIPAW-PBE's typical error for molecular solids ([Salager et al., *J. Am. Chem. "
    "Soc.* 132, 2564 (2010)](https://doi.org/10.1021/ja909449k))."
)


class StructureIdCocaineApp(BaseApp):
    """Cocaine structure-identification app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        plots = {
            metric: [
                read_plot(
                    DATA_PATH / f"figure_{kind}_expt_structure.json",
                    id=f"{BENCHMARK_NAME}-figure-{kind}-{i}",
                )
                for kind in ("parity", "shielding")
            ]
            for i, metric in enumerate(METRICS)
        }
        plots_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plots=plots,
        )
        links = read_links(DATA_PATH / "structure_links.json")
        if links:
            for graphs in plots.values():
                link_structures(graphs, f"{BENCHMARK_NAME}-struct-placeholder", links)


def get_app() -> StructureIdCocaineApp:
    """
    Get the cocaine structure-identification app.

    Returns
    -------
    StructureIdCocaineApp
        Benchmark layout and callback registration.
    """
    return StructureIdCocaineApp(
        name=BENCHMARK_NAME,
        description=(
            "Identifying the correct crystal structure of cocaine (the prediction "
            "that matches its X-ray structure) among 29 candidates from its ¹H "
            "spectrum, compared with experiment, following the atomistic-cookbook "
            "case study."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "structure_id_cocaine_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
            P(
                [
                    B("Click a cell for two plots:"),
                    " for the correct structure, predicted against experimental ¹H "
                    "shifts, and predicted shielding against experimental shift with "
                    "each method's line (slope fixed at −1) and intercept. Click a "
                    "data point to see the crystal structure, with that "
                    "site's protons highlighted.",
                ],
                style=TEXT_STYLE,
            ),
            P(
                [
                    B("Margin:"),
                    " ¹H RMSE of the best wrong candidate minus that of the correct "
                    "structure. Shieldings are converted to shifts with the slope "
                    "fixed at −1 and an intercept fitted per candidate, as in the ",
                    A("atomistic-cookbook case study", href=CASE_STUDY_URL),
                    ", so candidates are compared by spectral pattern. A margin of "
                    "zero or below means the correct structure is not identified.",
                ],
                style=TEXT_STYLE,
            ),
            P(
                [
                    B("Thresholds:"),
                    " Good 0.3 ppm, near GIPAW-PBE DFT's own margin of 0.26 ppm; Bad "
                    "0 ppm, no separation. Both can be changed in the Thresholds "
                    "boxes above.",
                ],
                style=TEXT_STYLE,
            ),
            read_plot(
                DATA_PATH / "figure_lollipop.json", id=f"{BENCHMARK_NAME}-lollipop"
            ),
            Markdown(FIGURE_CAPTION, style=TEXT_STYLE),
        ],
        info_path=INFO_PATH,
    )
