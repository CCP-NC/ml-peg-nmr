"""Tests for the NMR analysis, calc and app helpers."""

from __future__ import annotations

import json

from ase import Atoms
from ase.build import molecule
from ase.io import write
from dash.dcc import Graph
from dash.exceptions import PreventUpdate
import numpy as np
import pytest

from ml_peg.analysis.utils.nmr import (
    csa_params,
    identification,
    label_model_column,
    principal,
    weighted_fit,
    weighted_mae,
)
from ml_peg.app.utils import nmr as app_nmr
from ml_peg.calcs.utils.nmr import bare_copy, read_structure


# Calc helpers: preparing each structure for the model
def test_bare_copy_strips_reference_data():
    """Only species, positions and cell reach the model; reference data is dropped."""
    atoms = Atoms(
        "H2O",
        positions=[(0, 0, 0), (0, 0, 1), (0, 1, 0)],
        cell=[5, 5, 5],
        pbc=False,
    )
    atoms.arrays["CS"] = np.array([0, 1, 2])
    atoms.info["NAME"] = "x"

    bare = bare_copy(atoms)

    assert sorted(bare.arrays) == ["numbers", "positions"]
    assert bare.get_chemical_symbols() == ["H", "H", "O"]
    assert bare.positions == pytest.approx(atoms.positions)
    assert bare.cell[:] == pytest.approx(atoms.cell[:])
    assert bare.info == {}
    assert bare.pbc.all()


def test_read_structure_needs_exactly_one_magres(tmp_path):
    """A structure folder must hold exactly one .magres file."""
    water = molecule("H2O", cell=[10, 10, 10], pbc=True)

    test_folder_s1 = tmp_path / "H" / "s1"
    test_folder_s1.mkdir(parents=True)
    write(test_folder_s1 / "s1.magres", water)

    test_folder_s0 = tmp_path / "H" / "s0"
    test_folder_s0.mkdir(parents=True)

    _, bare = read_structure(test_folder_s1)  # should succeed

    assert sorted(bare.arrays) == ["numbers", "positions"]

    with pytest.raises(FileNotFoundError, match="No .magres file"):
        read_structure(test_folder_s0)  # should fail

    write(test_folder_s1 / "s2.magres", water)

    with pytest.raises(ValueError, match="found 2"):
        read_structure(test_folder_s1)


# Analysis helpers
# Isotropic and tensor benchmarks: the weighted fit
# Shared data for the weighting tests below: three sites, the first a methyl group
# (3 protons sharing one shift, so weight 3).
SIGMA = np.array([20.0, 25.0, 30.0])
DELTA = -SIGMA + 30 + np.array([0.0, 1.0, -2.0])
WEIGHTS = np.array([3, 1, 1])


def test_weighted_fit_exact_line():
    """Points on an exact line give slope -1, intercept 30 and zero RMSE."""
    sigma = np.array([10.0, 20.0, 30.0])
    # Shifts exactly on delta = -sigma + 30, with no scatter.
    slope, intercept, rmse = weighted_fit(sigma, -sigma + 30, np.ones_like(sigma))
    assert slope == pytest.approx(-1)
    assert intercept == pytest.approx(30)
    assert rmse == pytest.approx(0, abs=1e-12)


def test_weight_equals_repeated_rows():
    """A weight of 3 gives the same fit and MAE as writing the row out 3 times."""
    # The methyl row written out 3 times and the others once, each with weight 1.
    sigma_rep = np.repeat(SIGMA, WEIGHTS)
    delta_rep = np.repeat(DELTA, WEIGHTS)
    weighted = weighted_fit(SIGMA, DELTA, WEIGHTS)
    repeated = weighted_fit(sigma_rep, delta_rep, np.ones(len(sigma_rep)))

    assert weighted == pytest.approx(repeated)
    # Slope, intercept and RMSE worked out by hand.
    assert weighted == pytest.approx((-1.15, 33.25, 0.7746), abs=1e-4)
    assert weighted_mae(SIGMA, DELTA, WEIGHTS) == pytest.approx(0.6)
    assert weighted_mae(sigma_rep, delta_rep, np.ones(len(sigma_rep))) == pytest.approx(
        0.6
    )


def test_weighted_mae_uses_the_rmse_line():
    """The MAE is measured about the weighted RMSE line, not a refitted one."""
    # Weighted MAE about the line weighted_fit returns, worked out by hand (0.6).
    slope, intercept, _ = weighted_fit(SIGMA, DELTA, WEIGHTS)
    residuals = np.abs(DELTA - (slope * SIGMA + intercept))
    expected = np.sum(WEIGHTS * residuals) / np.sum(WEIGHTS)

    # The same MAE about a different line, the unweighted fit, is 0.8 (5 is the
    # total weight). The second assert checks the test can tell the two apart.
    u_slope, u_intercept, _ = weighted_fit(SIGMA, DELTA, np.ones(3))
    other = np.sum(WEIGHTS * np.abs(DELTA - (u_slope * SIGMA + u_intercept))) / 5

    assert weighted_mae(SIGMA, DELTA, WEIGHTS) == pytest.approx(expected)
    assert weighted_mae(SIGMA, DELTA, WEIGHTS) != pytest.approx(other)


# Tensor benchmark: principal values and CSA parameters
def test_principal_sorted_and_symmetrised():
    """Principal values come out ascending, from the symmetric part of the tensor."""
    # Flattened 3 x 3 tensor, row by row; not symmetric (0.5 against 0.1).
    tensor = np.array([1, 0.5, 0, 0.1, 2, 0, 0, 0, 3])
    values = principal(tensor)

    # Without symmetrising, the eigenvalues would be 0.9523, 2.0477, 3.0.
    assert values == pytest.approx(np.array([[0.9169, 2.0831, 3.0]]), abs=1e-4)


def test_principal_marks_failed_tensors_as_nan():
    """A failed (NaN) tensor gives NaN values without affecting the others."""
    tensors = np.array(
        [
            # A valid tensor, diagonal given in descending order (3, 2, 1).
            [3, 0, 0, 0, 2, 0, 0, 0, 1],
            # A failed prediction: every entry missing.
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )
    values = principal(tensors)

    # The valid tensor is still computed, and sorted ascending.
    assert values[0] == pytest.approx(np.array([1, 2, 3]), abs=1e-4)
    # The failed one stays visible as NaN instead of raising an error.
    assert np.isnan(values[1]).all()


def test_csa_params_hand_checked_tensor():
    """A hand-worked tensor gives the expected iso, zeta, eta, span and skew."""
    delta = np.array([[10, 0, -4]])
    out = csa_params(delta)

    assert out["iso"] == pytest.approx([2])
    assert out["zeta"] == pytest.approx([8])
    assert out["eta"] == pytest.approx([0.5])
    assert out["span"] == pytest.approx([14])
    assert out["skew"] == pytest.approx([-0.4286], abs=1e-4)
    assert out["zz"].tolist() == [0]


def test_csa_params_with_experiments_dzz():
    """With the experiment's dzz forced on, zeta flips sign and eta can exceed 1."""
    delta = np.array([[10, 0, -4]])
    zz = np.array([2])  #  force del_zz = del_33
    out = csa_params(delta, zz)

    assert out["zeta"] == pytest.approx([-6])
    assert out["eta"] == pytest.approx([1.6667], abs=1e-4)


def test_own_ordering_eta_is_between_0_and_1():
    """Own-ordering eta lies in [0, 1] and span is non-negative for random tensors."""
    rng = np.random.default_rng(0)
    delta = rng.normal(100, 50, size=(500, 3))
    out = csa_params(delta)

    assert np.all(out["eta"] >= 0)
    assert np.all(out["eta"] <= 1 + 1e-12)
    assert np.all(out["span"] >= 0)


# Cocaine benchmark: identification
@pytest.mark.parametrize(
    ("rmse", "correct", "expected_rank", "expected_margin"),
    [
        ([0.3, 0.5, 0.4], 0, 1, 0.1),
        ([0.5, 0.3, 0.4], 0, 3, -0.2),
        ([0.5, 0.4, 0.3], 2, 1, 0.1),
    ],
)
def test_identification(rmse, correct, expected_rank, expected_margin):
    """Rank and margin of the correct cocaine candidate follow from the RMSEs."""
    rank, margin = identification(np.array(rmse), correct)

    assert rank == expected_rank
    assert margin == pytest.approx(expected_margin)


def test_identification_is_not_scored_if_any_candidate_failed():
    """A NaN RMSE for any candidate gives no rank and no margin."""
    rank, margin = identification(np.array([0.3, np.nan, 0.4]), correct=0)

    assert np.isnan(rank)
    assert np.isnan(margin)


# All three benchmarks: the metrics table
def test_label_model_column_renames_display_name_only(tmp_path):
    """The first column is shown as "Model" while its id stays "MLIP"."""
    table_file = tmp_path / "table.json"
    table = {
        "data": [{"id": "ShiftML3", "MLIP": "ShiftML3", "Score": 0.6}],
        "columns": [{"id": "MLIP", "name": "MLIP"}, {"id": "Score", "name": "Score"}],
    }
    table_file.write_text(json.dumps(table))

    label_model_column(table_file)
    result = json.loads(table_file.read_text())

    assert result["columns"] == [
        {"id": "MLIP", "name": "Model"},
        {"id": "Score", "name": "Score"},
    ]
    assert result["data"] == table["data"]  # unchanged


# App helpers
def test_read_links(tmp_path):
    """Structure links are read from file, or empty if the analysis has not run."""
    assert app_nmr.read_links(tmp_path / "missing.json") == {}
    table_file = tmp_path / "valid_file.json"
    table = {"1H": {"structs": ["a"], "highlights": [[0]]}}
    table_file.write_text(json.dumps(table))

    assert app_nmr.read_links(tmp_path / "valid_file.json") == table


def test_plots_from_table_column(monkeypatch):
    """Clicking a column shows its plots; other clicks show the hint or do nothing."""
    callbacks = []

    def capture_callback(*_args, **_kwargs):
        def register(func):
            callbacks.append(func)
            return func

        return register

    monkeypatch.setattr(app_nmr, "callback", capture_callback)
    monkeypatch.setattr(app_nmr, "register_plot_download_callbacks", lambda: None)
    monkeypatch.setattr(app_nmr, "register_plot_settings_callbacks", lambda: None)
    monkeypatch.setattr(app_nmr, "plot_with_download_controls", lambda plot: plot)
    figure = {"data": [], "layout": {}}
    plots = [Graph(id="parity", figure=figure), Graph(id="fit", figure=figure)]
    app_nmr.plots_from_table_column("table", "plot", {"¹H RMSE": plots})
    show_plots = callbacks[0]

    # No cell clicked: the hint is shown.
    result, active_cell = show_plots(None)
    assert result.children == app_nmr.TABLE_HINT
    assert active_cell is None

    # A column without plots: nothing changes.
    with pytest.raises(PreventUpdate):
        show_plots({"column_id": "Score"})

    # A column with plots: one Div per plot, in order.
    result, active_cell = show_plots({"column_id": "¹H RMSE"})
    assert [cell.children.id for cell in result.children] == ["parity", "fit"]
    assert active_cell is None
