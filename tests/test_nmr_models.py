"""Tests for the NMR model registry, model selection, loader and shielding mocks."""

from __future__ import annotations

import sys
from types import ModuleType

from ase.build import molecule
import numpy as np
import pytest
from yaml import safe_load

import ml_peg.models as settings
from ml_peg.models.get_models import (
    NMR_MODELS_FILE,
    get_nmr_models,
    load_nmr_predictor,
    validate_config,
)

NMR_REGISTRY = safe_load(NMR_MODELS_FILE.read_text(encoding="utf8"))
MOCK = {"module": "ml_peg.models.mock", "class_name": "MockCalculator"}
MOCK_ERROR = {"module": "ml_peg.models.mock", "class_name": "MockErrorCalculator"}


class _FakeEnsemble:
    """Minimal stand-in for ShiftML's committee of models."""

    def get_cs_iso_ensemble(self, atoms):
        """Return three fixed committee predictions per atom."""
        return np.tile([1.0, 2.0, 3.0], (len(atoms), 1))

    def get_cs_tensor_ensemble(self, atoms, return_symmetric=True):
        """Return three zero shielding tensors per atom."""
        return np.zeros((len(atoms), 3, 3, 3))


@pytest.fixture
def fake_shiftml(monkeypatch):
    """Provide a fake ``shiftml.ase`` module that records how ShiftML is called."""
    calls = []

    def fake_load(model_version, force_download=False, device=None):
        calls.append(
            {
                "model_version": model_version,
                "device": device,
            }
        )
        return _FakeEnsemble()

    shiftml_module = ModuleType("shiftml")
    ase_module = ModuleType("shiftml.ase")
    ase_module.ShiftML = fake_load
    shiftml_module.ase = ase_module

    monkeypatch.setitem(sys.modules, "shiftml", shiftml_module)
    monkeypatch.setitem(sys.modules, "shiftml.ase", ase_module)

    return calls


@pytest.fixture
def nmr_settings(monkeypatch):
    """Set ml-peg's mock and models-file options for one test."""

    def configure(run_mock=False, mock_only=False, models_file=None):
        monkeypatch.setattr(settings, "run_mock", run_mock)
        monkeypatch.setattr(settings, "mock_only", mock_only)
        if models_file is not None:
            monkeypatch.setattr(settings, "models_file", models_file)

    return configure


def test_nmr_config_is_valid():
    """Every NMR model entry passes ml-peg's model-config validation."""
    for name in NMR_REGISTRY:
        validate_config(name, NMR_REGISTRY[name])


@pytest.mark.parametrize(
    ("run_mock", "mock_only", "models", "expected_names"),
    [
        (True, False, None, ["mock", "ShiftML3", "ShiftML4"]),
        (False, False, None, ["ShiftML3", "ShiftML4"]),
        (True, True, None, ["mock"]),
        (False, False, "ShiftML4", ["ShiftML4"]),
        (False, False, ["ShiftML3"], ["ShiftML3"]),
        (False, False, "mace-mp-0a,ShiftML3", ["ShiftML3"]),
    ],
)
def test_get_nmr_models_selection(
    run_mock, mock_only, models, expected_names, nmr_settings
):
    """The NMR model list follows the mock options and the selected model names."""
    nmr_settings(run_mock, mock_only)

    assert list(get_nmr_models(models).keys()) == expected_names


def test_get_nmr_models_warns_when_no_nmr_model_selected(nmr_settings):
    """Selecting only non-NMR models gives an empty list and a warning."""
    nmr_settings()

    with pytest.warns(UserWarning, match="None of the selected models"):
        result = get_nmr_models("mace-mp-0a")

    assert result == {}


def test_mock_only_requires_run_mock(nmr_settings):
    """Asking for the mock model only, without running mocks, is an error."""
    nmr_settings(run_mock=False, mock_only=True)

    with pytest.raises(ValueError):
        get_nmr_models()


def test_mock_error_models_file_is_used_as_given(nmr_settings):
    """A models file of ml-peg mock entries is used as given, for failure tests."""
    nmr_settings(models_file=settings.MODELS_ROOT / "mock_error.yml")

    assert list(get_nmr_models().keys()) == ["mock-error"]


@pytest.mark.parametrize("name", ["ShiftML3", "ShiftML4"])
def test_load_nmr_predictor_passes_registry_config(name, fake_shiftml):
    """The real-model branch passes model_version and device from nmr_models.yml."""
    calls = fake_shiftml
    predictor = load_nmr_predictor(NMR_REGISTRY[name], precision="high")

    assert calls[-1] == {"model_version": name, "device": "cpu"}
    # The fake committee predicts 1, 2 and 3 for each atom, so the mean is 2.
    iso = predictor.get_cs_iso_ensemble(molecule("H2O"))
    assert iso.mean(axis=1) == pytest.approx([2, 2, 2])


def test_load_nmr_predictor_rejects_unknown_precision():
    """An unknown precision is rejected, as for the MLIP wrappers."""
    with pytest.raises(ValueError):
        load_nmr_predictor(MOCK, precision="medium")


@pytest.mark.parametrize("precision", ["low", "high"])
def test_shielding_mocks(precision):
    """The zero mock returns arrays of the right shapes; the error mock always fails."""
    atoms = molecule("H2O")
    mock = load_nmr_predictor(MOCK, precision=precision)
    mock_iso = mock.get_cs_iso_ensemble(atoms)
    mock_tensor = mock.get_cs_tensor_ensemble(atoms)

    assert mock_iso.shape == (3, 1)
    assert np.all(mock_iso == 0)
    assert mock_tensor.shape == (3, 3, 3, 1)
    assert np.all(mock_tensor == 0)

    error = load_nmr_predictor(MOCK_ERROR, precision=precision)
    with pytest.raises(ValueError):
        error.get_cs_iso_ensemble(atoms)
    with pytest.raises(ValueError):
        error.get_cs_tensor_ensemble(atoms)


@pytest.mark.parametrize("name, members", [("ShiftML3", 8), ("ShiftML4", 7)])
def test_registered_shiftml_model_predicts(name, members):
    """A real ShiftML model gives finite shieldings of the expected shapes."""
    pytest.importorskip("shiftml")

    atoms = molecule("CH4")
    atoms.center(vacuum=5.0)
    atoms.pbc = True

    predictor = load_nmr_predictor(NMR_REGISTRY[name], precision="high")
    iso = predictor.get_cs_iso_ensemble(atoms)
    tensor = predictor.get_cs_tensor_ensemble(atoms, return_symmetric=True)

    assert iso.shape == (5, members)
    assert tensor.shape == (5, 3, 3, members)
    assert np.isfinite(iso).all()
    assert np.isfinite(tensor).all()
