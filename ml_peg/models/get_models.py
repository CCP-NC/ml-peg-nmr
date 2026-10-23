"""Get MLIPs to be used for calculations or analysis."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any
from warnings import warn

import yaml

from ml_peg.models import models_file
from ml_peg.models.mock import MockErrorShieldingPredictor, MockShieldingPredictor

if TYPE_CHECKING:
    from ml_peg.models.models import Precision


@lru_cache(maxsize=1)
def _load_models_yaml(
    filepath: Path | str | None = None,
) -> dict[str, Any]:
    """
    Load and cache models.yml to prevent repeated expensive YAML parsing.

    Parameters
    ----------
    filepath
        Path to YAML file with models. Default is `models_file`.

    Returns
    -------
    dict[str, Any]
        Parsed models.yml registry.
    """
    filepath = filepath if filepath else models_file
    with open(filepath, encoding="utf8") as file:
        return yaml.safe_load(file) or {}


def load_model_configs(
    mlips: Iterable[str] | tuple[str, ...],
    filepath: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, str | None]]:
    """
    Load model configurations and level of theory metadata from models.yml.

    Parameters
    ----------
    mlips
        Iterable of model identifiers to load configurations for.
    filepath
        Path to YAML file with models. Default is `models_file`.

    Returns
    -------
    tuple[dict[str, Any], dict[str, str | None]]
        A tuple containing:
        - model_configs: Dictionary mapping model names to their configuration dicts
        - model_levels: Dictionary mapping model names to their level of
          theory (or ``None``)
    """
    filepath = filepath if filepath else models_file
    all_models = _load_models_yaml(filepath)

    model_levels: dict[str, str | None] = {}
    model_configs: dict[str, Any] = {}
    for mlip in mlips:
        cfg = deepcopy(all_models.get(mlip) or {})
        if not isinstance(cfg, dict):
            cfg = {}
        model_configs[mlip] = cfg
        model_levels[mlip] = cfg.get("level_of_theory")

    return model_configs, model_levels


def get_subset(
    all_models: dict[str, Any], models: None | str | Iterable = None
) -> dict[str, Any]:
    """
    Get a subset of models from a dictionary.

    Parameters
    ----------
    all_models
        Dictionary of models to extract a subset of.
    models
        Models to select fromm `all_models`. If `None`, all models will be selected.
        If an iterable, all models with matching keys will be selected. If a string,
        this will be treated as a comma-separated list.

    Returns
    -------
    dict[str, Any]
        Subset of `all_models` matching `models`, as described above.
    """
    if models is None:
        return all_models

    if isinstance(models, str):
        models = models.split(",")

    try:
        return {model: all_models[model] for model in models}
    except KeyError as err:
        for model in models:
            if model not in all_models:
                raise ValueError(
                    f"Model name '{model}' not recognised. Please check models.yml"
                ) from err


def load_models(
    models: None | str | Iterable = None,
    filepath: Path | str | None = None,
    run_mock: bool | None = None,
    mock_only: bool | None = None,
) -> dict[str, Any]:
    """
    Load models for use in calculations.

    Parameters
    ----------
    models
        Models to select from `filepath`. If `None`, all models will be selected.
        If an iterable, all models with matching keys will be selected. If a string,
        this will be treated as a comma-separated list.
    filepath
        Path to YAML file with models. Default is `models_file`.
    run_mock
            Whether to include mock model in the loaded models. Default is False.
    mock_only
            Whether to load only mock model, ignoring `models`. Default is False.

    Returns
    -------
    dict[str, Any]
        Loaded models from models.yml and/or loaded mock model.
    """
    from ml_peg.models.models import (
        DpaCalc,
        FairChemCalc,
        GenericASECalc,
        GraceCalc,
        MatterSimCalc,
        MockCalc,
        OrbCalc,
        SevenNetCalc,
        UPETCalc,
        VivaceCalc,
    )

    if run_mock is None:
        from ml_peg.models import run_mock
    if mock_only is None:
        from ml_peg.models import mock_only

    if mock_only and not run_mock:
        raise ValueError("Cannot set `mock_only` without `run_mock`")

    loaded_models = {"mock": MockCalc()} if run_mock else {}
    if mock_only:
        return loaded_models

    filepath = filepath if filepath else models_file
    all_models = _load_models_yaml(filepath)

    for name, cfg in get_subset(all_models, models).items():
        print(f"Loading model from {filepath}: {name}")

        match cfg["class_name"]:
            case "DP":
                loaded_models[name] = DpaCalc(
                    kwargs=cfg.get("kwargs", {}),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )
            case "FAIRChemCalculator":
                kwargs = cfg.get("kwargs", {})
                loaded_models[name] = FairChemCalc(
                    model_name=kwargs["model_name"],
                    task_name=kwargs.get("task_name", "omat"),
                    device=cfg.get("device", "cpu"),
                    overrides=kwargs.get("overrides", {}),
                    default_dtype=cfg.get("overwrite_dtype", None),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )
            case "OrbCalc":
                kwargs = cfg.get("kwargs", {})
                loaded_models[name] = OrbCalc(
                    name=kwargs["name"],
                    device=cfg.get("device", "cpu"),
                    default_dtype=cfg.get("overwrite_dtype", None),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )
            case "mace" | "mace_mp" | "mace_off" | "mace_omol" | "mace_polar":
                loaded_models[name] = GenericASECalc(
                    module=cfg["module"],
                    class_name=cfg["class_name"],
                    device=cfg.get("device", "auto"),
                    default_dtype=cfg.get("overwrite_dtype", None),
                    kwargs=cfg.get("kwargs", {}),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )
            case "MatterSimCalculator":
                loaded_models[name] = MatterSimCalc(
                    module=cfg["module"],
                    class_name=cfg["class_name"],
                    device=cfg.get("device", "cpu"),
                    default_dtype=cfg.get("overwrite_dtype", None),
                    kwargs=cfg.get("kwargs", {}),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )
            case "UPETCalculator":
                loaded_models[name] = UPETCalc(
                    module=cfg["module"],
                    class_name=cfg["class_name"],
                    device=cfg.get("device", "cpu"),
                    default_dtype=cfg.get("overwrite_dtype", None),
                    kwargs=cfg.get("kwargs", {}),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )
            case "SevenNetCalculator":
                loaded_models[name] = SevenNetCalc(
                    device=cfg.get("device", "cpu"),
                    kwargs=cfg.get("kwargs", {}),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )
            case "MLFFCalculator":
                loaded_models[name] = VivaceCalc(
                    device=cfg.get("device", "auto"),
                    kwargs=cfg.get("kwargs", {}),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )
            case "grace_fm":
                loaded_models[name] = GraceCalc(
                    module=cfg["module"],
                    class_name=cfg["class_name"],
                    device=cfg.get("device", "cpu"),
                    kwargs=cfg.get("kwargs", {}),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                    default_dtype=cfg.get("overwrite_dtype", None),
                )
            case _:
                loaded_models[name] = GenericASECalc(
                    module=cfg["module"],
                    class_name=cfg["class_name"],
                    device=cfg.get("device", "auto"),
                    default_dtype=cfg.get("overwrite_dtype", None),
                    kwargs=cfg.get("kwargs", {}),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )

    return loaded_models


def get_model_names(
    models: None | Iterable = None,
    filepath: Path | str | None = None,
) -> list[str]:
    """
    Load models names for use in analysis.

    Parameters
    ----------
    models
        Models to select from `filepath`. If `None`, all models will be selected.
        If an iterable, all models with matching keys will be selected. If a string,
        this will be treated as a comma-separated list.
    filepath
        Path to YAML file with models. Default is `models_file`.

    Returns
    -------
    list[str]
        Loaded model names from `filepath`.
    """
    filepath = filepath if filepath else models_file
    all_models = _load_models_yaml(filepath)

    model_names = []
    for name in get_subset(all_models, models):
        model_names.append(name)

    return model_names


# NMR shielding models (not interatomic potentials). They are loaded from their own
# list, so that the MLIP benchmarks never run them and they never run the MLIP
# benchmarks. They provide shielding predictions instead of an ASE calculator, and
# ml-peg's mock entries map to the shielding mocks in ``ml_peg.models.mock``.
NMR_MODELS_FILE = Path(__file__).parent / "nmr_models.yml"
MOCK_MODULE = "ml_peg.models.mock"
MOCK_CONFIG = {"module": MOCK_MODULE, "class_name": "MockCalculator"}


def get_nmr_models(
    models: None | str | Iterable = None,
) -> dict[str, dict[str, Any]]:
    """
    Get NMR model configurations, optionally restricted to a selection.

    Models come from ``nmr_models.yml``. A models file made of ml-peg mock entries
    (``--models-file ml_peg/models/mock_error.yml``) is used as given, to test
    failure handling.

    Parameters
    ----------
    models
        Model names, comma-separated or as a list (e.g. from ``--models``). Names
        that are not NMR models (e.g. MLIPs selected for other benchmarks) are
        ignored. Default is None, which selects every model in
        ``nmr_models.yml``.

    Returns
    -------
    dict[str, dict[str, Any]]
        Model name to configuration.
    """
    from ml_peg import models as settings

    if settings.mock_only and not settings.run_mock:
        raise ValueError("Cannot set `mock_only` without `run_mock`")
    chosen = {"mock": MOCK_CONFIG} if settings.run_mock else {}
    if settings.mock_only:
        return chosen

    given = _load_models_yaml(settings.models_file)
    mocks = {n: c for n, c in given.items() if c.get("module") == MOCK_MODULE}
    all_models = mocks if mocks and len(mocks) == len(given) else None
    if all_models is None:
        all_models = _load_models_yaml(NMR_MODELS_FILE)
    if models is None:
        return chosen | dict(all_models)
    selected = models.split(",") if isinstance(models, str) else list(models)
    picked = {name: cfg for name, cfg in all_models.items() if name in selected}
    if not picked:
        warn(
            f"None of the selected models ({', '.join(selected)}) is an NMR model in "
            f"{NMR_MODELS_FILE.name}; the NMR benchmarks will not run.",
            stacklevel=2,
        )
    return chosen | picked


def load_nmr_predictor(config: dict[str, Any], *, precision: Precision) -> Any:
    """
    Instantiate a shielding predictor from its configuration.

    Parameters
    ----------
    config
        Entry from ``nmr_models.yml``, or an ml-peg mock entry.
    precision
        Unused precision argument, kept for the common model API (checked, as for
        the MLIP wrappers). ShiftML has no dtype option, so it runs at its native
        precision for both "low" and "high".

    Returns
    -------
    Any
        Object providing ``get_cs_iso_ensemble(atoms)`` and
        ``get_cs_tensor_ensemble(atoms)``.
    """
    from ml_peg.models.models import check_precision

    check_precision(precision)
    if config.get("module") == MOCK_MODULE:
        mocks = {
            "MockCalculator": MockShieldingPredictor,
            "MockErrorCalculator": MockErrorShieldingPredictor,
        }
        return mocks[config["class_name"]]()
    predictor = getattr(import_module(config["module"]), config["class_name"])
    return predictor(**config.get("kwargs", {}), device=config.get("device", "cpu"))
