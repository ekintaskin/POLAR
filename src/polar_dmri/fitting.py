"""Model fitting with CMA-ES in normalized parameter coordinates."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Could not import matplotlib")
    import cma

from .models.base import SignalModel
from .protocol import PGSEProtocol

_INVALID_COST = 1e10


@dataclass(frozen=True)
class FitSettings:
    """CMA-ES settings of the inner fit."""

    sigma0: float = 0.12
    maxiter: int = 500
    popsize: int = 16
    ftarget: float = 1e-20

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FitResult:
    params: dict[str, float]
    cost: float

    def values(self, names) -> np.ndarray:
        return np.array([self.params[name] for name in names], dtype=float)


def fit_signal(
    model: SignalModel,
    signal: np.ndarray,
    protocol: PGSEProtocol,
    settings: FitSettings | None = None,
    seed: int | None = None,
) -> FitResult:
    """Fit ``model`` to ``signal`` by least squares.

    The search runs on the model's [0, 1] parameter coordinates and starts near
    the center of the box.
    """
    settings = settings or FitSettings()
    if seed is not None and seed < 1:
        raise ValueError("seed must be positive: CMA-ES treats 0 as 'seed from the clock'.")
    names = model.parameter_names
    n_params = len(names)
    measured = np.asarray(signal, dtype=float)

    def decode(z: np.ndarray) -> dict[str, float]:
        return {
            parameter.name: float(parameter.from_unit(value))
            for parameter, value in zip(model.parameters, z)
        }

    def cost(z: np.ndarray) -> float:
        residual = measured - model.signal(decode(z), protocol)
        if not np.all(np.isfinite(residual)):
            return _INVALID_COST
        return float(np.dot(residual, residual))

    rng = np.random.default_rng(seed)
    z0 = np.clip(0.5 + 0.05 * rng.normal(size=n_params), 0.0, 1.0)
    options = {
        "bounds": [np.zeros(n_params), np.ones(n_params)],
        "maxiter": settings.maxiter,
        "popsize": settings.popsize,
        "ftarget": settings.ftarget,
        "verbose": -9,
        "verb_log": 0,
    }
    if seed is not None:
        options["seed"] = int(seed)

    global_state = np.random.get_state()
    try:
        es = cma.CMAEvolutionStrategy(z0, settings.sigma0, options)
        while not es.stop():
            candidates = es.ask()
            es.tell(candidates, [cost(np.asarray(z, dtype=float)) for z in candidates])
        z_best = np.clip(np.asarray(es.result.xbest, dtype=float), 0.0, 1.0)
    finally:
        np.random.set_state(global_state)
    return FitResult(decode(z_best), cost(z_best))
