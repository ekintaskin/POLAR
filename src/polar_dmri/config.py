"""Reading run settings from YAML. Times are in ms and gradients in mT/m."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .constraints import ScannerConstraints
from .fitting import FitSettings
from .models import SignalModel, Substrate, get_model
from .noise import RicianNoise
from .optimize import OptimizerSettings

_SECTIONS = {"model", "protocol_space", "noise", "substrates", "loss", "optimizer", "fit", "evaluation"}


@dataclass(frozen=True)
class RunConfig:
    model: SignalModel
    substrates: list[Substrate]
    n_shells: int
    include_b0: bool
    constraints: ScannerConstraints
    noise: RicianNoise
    n_noise_realizations: int
    loss_parameters: tuple[str, ...] | None
    weights: Mapping[str, float]
    optimizer: OptimizerSettings
    seed: int | None
    fit: FitSettings
    evaluation_noise_realizations: int


def _section(raw: Mapping[str, Any], name: str, allowed: set[str], required: set[str] = frozenset()):
    section = dict(raw.get(name) or {})
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in '{name}': {sorted(unknown)}. Allowed: {sorted(allowed)}.")
    missing = set(required) - set(section)
    if missing:
        raise ValueError(f"Missing keys in '{name}': {sorted(missing)}.")
    return section


def _milli_pair(pair) -> tuple[float, float]:
    low, high = pair
    return float(low) * 1e-3, float(high) * 1e-3


def _build_constraints(raw: Mapping[str, Any]) -> tuple[ScannerConstraints, int, bool]:
    section = _section(
        raw,
        "protocol_space",
        {"n_shells", "include_b0", "delta_ms", "Delta_ms", "G_mT_per_m", "min_gap_ms",
         "te_overhead_ms", "te_max_ms", "b_max"},
        {"n_shells"},
    ) 
    kwargs: dict[str, Any] = {}
    if "delta_ms" in section:
        kwargs["delta_bounds"] = _milli_pair(section["delta_ms"])
    if "Delta_ms" in section:
        kwargs["big_delta_bounds"] = _milli_pair(section["Delta_ms"])
    if "G_mT_per_m" in section:
        kwargs["G_bounds"] = _milli_pair(section["G_mT_per_m"])
    if "min_gap_ms" in section:
        kwargs["min_gap"] = float(section["min_gap_ms"]) * 1e-3
    if "te_overhead_ms" in section:
        kwargs["te_overhead"] = float(section["te_overhead_ms"]) * 1e-3
    if "te_max_ms" in section:
        te_max = section["te_max_ms"]
        kwargs["te_max"] = None if te_max is None else float(te_max) * 1e-3
    if "b_max" in section:
        kwargs["b_max"] = None if section["b_max"] is None else float(section["b_max"])
    return ScannerConstraints(**kwargs), int(section["n_shells"]), bool(section.get("include_b0", True))


def _build_model(raw: Mapping[str, Any]) -> SignalModel:
    section = dict(raw.get("model") or {})
    if "name" not in section:
        raise ValueError("'model' must define a 'name'.")
    name = section.pop("name")
    if "T2_ms" in section:
        section["T2"] = float(section.pop("T2_ms")) * 1e-3
    if "bounds" in section:
        section["bounds"] = {
            key: (float(low), float(high)) for key, (low, high) in section["bounds"].items()
        }
    try:
        return get_model(name, **section)
    except TypeError as error:
        raise ValueError(f"Invalid settings for model {name!r}: {error}") from error


def _build_noise(raw: Mapping[str, Any], model: SignalModel) -> tuple[RicianNoise, int]:
    section = _section(
        raw, "noise", {"snr", "te_ref_ms", "T2_ms", "n_directions", "n_realizations"}
    )
    kwargs: dict[str, Any] = {}
    if "snr" in section:
        kwargs["snr"] = float(section["snr"])
    if "te_ref_ms" in section:
        kwargs["te_ref"] = float(section["te_ref_ms"]) * 1e-3
    if "T2_ms" in section:
        kwargs["t2"] = float(section["T2_ms"]) * 1e-3
    elif hasattr(model, "T2"):
        kwargs["t2"] = float(model.T2)
    if "n_directions" in section:
        kwargs["n_directions"] = int(section["n_directions"])
    return RicianNoise(**kwargs), int(section.get("n_realizations", 10))


def _build_substrates(
    raw: Mapping[str, Any], model: SignalModel, base_dir: str | os.PathLike
) -> list[Substrate]:
    """Build the grid. Axes named ``*_files`` hold paths relative to ``base_dir``."""
    section = _section(raw, "substrates", {"grid"}, {"grid"})
    axes = {}
    for name, values in section["grid"].items():
        convert = (lambda value: str(Path(base_dir) / value)) if name.endswith("_files") else float
        axes[name] = [convert(value) for value in _as_list(values)]
    return model.make_substrates(axes)


def _as_list(values) -> list:
    return list(values) if isinstance(values, (list, tuple)) else [values]


def build_run_config(raw: Mapping[str, Any], base_dir: str | os.PathLike = ".") -> RunConfig:
    """Validate a parsed configuration and build the objects it describes.

    Relative file paths in the configuration are resolved against ``base_dir``.
    """
    unknown = set(raw) - _SECTIONS
    if unknown:
        raise ValueError(f"Unknown sections: {sorted(unknown)}. Allowed: {sorted(_SECTIONS)}.")

    model = _build_model(raw)
    substrates = _build_substrates(raw, model, base_dir)
    model.check_substrates(substrates)
    constraints, n_shells, include_b0 = _build_constraints(raw)
    noise, n_noise = _build_noise(raw, model)

    loss = _section(raw, "loss", {"parameters", "weights"})
    optimizer = _section(raw, "optimizer", set(OptimizerSettings.__dataclass_fields__) | {"seed"})
    seed = optimizer.pop("seed", None)
    fit = _section(raw, "fit", set(FitSettings.__dataclass_fields__))
    evaluation = _section(raw, "evaluation", {"n_realizations"})

    parameters = loss.get("parameters")
    return RunConfig(
        model=model,
        substrates=substrates,
        n_shells=n_shells,
        include_b0=include_b0,
        constraints=constraints,
        noise=noise,
        n_noise_realizations=n_noise,
        loss_parameters=None if parameters is None else tuple(parameters),
        weights={key: float(value) for key, value in (loss.get("weights") or {}).items()},
        optimizer=OptimizerSettings(**optimizer),
        seed=None if seed is None else int(seed),
        fit=FitSettings(**fit),
        evaluation_noise_realizations=int(evaluation.get("n_realizations", 100)),
    )


def load_run_config(path: str | os.PathLike) -> RunConfig:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} does not contain a configuration mapping.")
    return build_run_config(raw, Path(path).parent)
