"""The optimization objective."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from joblib import Parallel, delayed

from .fitting import FitSettings, fit_signal
from .models.base import SignalModel, Substrate
from .noise import RicianNoise
from .protocol import PGSEProtocol

INVALID_LOSS = 1e12


def make_seed_table(rng: np.random.Generator, n_noise: int, n_substrates: int) -> np.ndarray:
    """Draw one noise seed per (noise realization, substrate) pair.
    Evaluating several protocols with the same table gives them matched noise for a fair comparison.
    """
    return rng.integers(
        0, np.iinfo(np.uint32).max, size=(n_noise, n_substrates), dtype=np.uint32
    )


def estimate_for_substrate(
    model: SignalModel,
    substrate: Substrate,
    protocol: PGSEProtocol,
    noise: RicianNoise,
    seeds: Sequence[int],
    fit_settings: FitSettings,
) -> np.ndarray:

    names = model.parameter_names
    estimates = np.full((len(seeds), len(names)), np.nan)
    clean = model.simulate(substrate, protocol)
    if not np.all(np.isfinite(clean)):
        return estimates
    for index, seed in enumerate(seeds):
        noisy = noise.apply(clean, np.random.default_rng(int(seed)))
        fit = fit_signal(model, noisy, protocol, fit_settings)
        estimates[index] = fit.values(names)
    return estimates


def squared_errors(
    model: SignalModel,
    truth: np.ndarray,
    estimates: np.ndarray,
    parameters: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Weighted squared error in [0, 1] coordinates, averaged over parameters.
    Only ``parameters`` contribute (all when None). Non-finite estimates give ``INVALID_LOSS``.
    """

    names = model.parameter_names
    parameters = names if parameters is None else tuple(parameters)
    weights = weights or {}
    unknown = (set(parameters) | set(weights)) - set(names)
    if unknown or not parameters:
        raise ValueError(f"Loss parameters must be from {names}; got unknown {sorted(unknown)}.")

    indices = [names.index(name) for name in parameters]
    scale = np.array([float(weights.get(name, 1.0)) for name in parameters])
    if np.any(scale < 0) or np.sum(scale) <= 0:
        raise ValueError("Loss weights must not be negative, and at least one must be positive.")
    estimates = np.asarray(estimates, dtype=float)
    invalid = ~np.all(np.isfinite(estimates), axis=-1)
    difference = (
        model.to_unit(np.where(invalid[..., None], np.asarray(truth), estimates))
        - model.to_unit(truth)
    )[..., indices]
    return np.where(invalid, INVALID_LOSS, np.sum(scale * difference**2, axis=-1) / np.sum(scale))


def estimate_all(
    model: SignalModel,
    protocols: Sequence[PGSEProtocol],
    substrates: Sequence[Substrate],
    noise: RicianNoise,
    seed_table: np.ndarray,
    fit_settings: FitSettings,
    n_jobs: int = -1,
) -> list[np.ndarray]:
    """Return an ``(n_substrates, n_noise, n_parameters)`` array per protocol."""

    tasks = [
        delayed(estimate_for_substrate)(
            model, substrate, protocol, noise, seed_table[:, j], fit_settings
        )
        for protocol in protocols
        for j, substrate in enumerate(substrates)
    ]
    flat = Parallel(n_jobs=n_jobs, backend="loky")(tasks)
    n_substrates = len(substrates)
    return [
        np.stack(flat[i * n_substrates : (i + 1) * n_substrates]) for i in range(len(protocols))
    ]
