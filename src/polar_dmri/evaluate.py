"""Evaluating and comparing protocols by parameter recovery performance."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .fitting import FitSettings
from .models.base import SignalModel, Substrate
from .noise import RicianNoise
from .objective import estimate_all, make_seed_table, squared_errors
from .protocol import PGSEProtocol


@dataclass(frozen=True)
class EvaluationResult:
    """Fit results for one protocol."""

    model: SignalModel
    truth: np.ndarray
    estimates: np.ndarray

    def loss(
        self,
        parameters: Sequence[str] | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> float:
        
        errors = squared_errors(self.model, self.truth[:, None], self.estimates, parameters, weights)
        return float(np.mean(errors))

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-parameter error statistics, pooled over substrates and noise."""
        
        truth = self.truth[:, None]
        error = self.estimates - truth
        unit_error = self.model.to_unit(self.estimates) - self.model.to_unit(truth)
        spread = self.estimates - self.estimates.mean(axis=1, keepdims=True)
        return {
            name: {
                "bias": float(np.mean(error[..., i])),
                "std": float(np.sqrt(np.mean(spread[..., i] ** 2))),
                "rmse": float(np.sqrt(np.mean(error[..., i] ** 2))),
                "normalized_rmse": float(np.sqrt(np.mean(unit_error[..., i] ** 2))),
            }
            for i, name in enumerate(self.model.parameter_names)
        }

    def save_csv(self, path: str | os.PathLike) -> None:
        """Write one row per fit, with truth and estimate for each parameter."""
        names = self.model.parameter_names
        header = ["substrate", "noise"]
        for name in names:
            header += [f"{name}_true", f"{name}_est"]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for substrate, (truth, estimates) in enumerate(zip(self.truth, self.estimates)):
                for noise, estimate in enumerate(estimates):
                    row = [substrate, noise]
                    for true_value, estimated in zip(truth, estimate):
                        row += [float(true_value), float(estimated)]
                    writer.writerow(row)


def compare_protocols(
    model: SignalModel,
    protocols: Mapping[str, PGSEProtocol],
    substrates: Sequence[Substrate],
    noise: RicianNoise | None = None,
    n_noise_realizations: int = 100,
    fit_settings: FitSettings | None = None,
    seed: int = 0,
    n_jobs: int = -1,
) -> dict[str, EvaluationResult]:
    """Evaluate several protocols with matched noise seeds."""

    if n_noise_realizations < 1:
        raise ValueError("n_noise_realizations must be at least 1.")
    noise = noise or RicianNoise()
    fit_settings = fit_settings or FitSettings()
    model.check_substrates(substrates)
    truth = np.stack([model.ground_truth(substrate) for substrate in substrates])
    seed_table = make_seed_table(
        np.random.default_rng(seed), n_noise_realizations, len(substrates)
    )
    estimates = estimate_all(
        model, list(protocols.values()), substrates, noise, seed_table, fit_settings, n_jobs
    )
    return {
        name: EvaluationResult(model, truth, estimate)
        for name, estimate in zip(protocols, estimates)
    }


def evaluate_protocol(
    model: SignalModel,
    protocol: PGSEProtocol,
    substrates: Sequence[Substrate],
    noise: RicianNoise | None = None,
    n_noise_realizations: int = 100,
    fit_settings: FitSettings | None = None,
    seed: int = 0,
    n_jobs: int = -1,
) -> EvaluationResult:

    results = compare_protocols(
        model, {"protocol": protocol}, substrates, noise, n_noise_realizations, fit_settings,
        seed, n_jobs,
    )  # fmt: skip
    return results["protocol"]


def write_summary_csv(results: Mapping[str, EvaluationResult], path: str | Path) -> None:
    """Write one row per protocol and parameter, plus each protocol's loss."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protocol", "parameter", "bias", "std", "rmse", "normalized_rmse", "loss"])
        for name, result in results.items():
            loss = result.loss()
            for parameter, stats in result.summary().items():
                writer.writerow(
                    [name, parameter, stats["bias"], stats["std"], stats["rmse"],
                     stats["normalized_rmse"], loss]
                )
