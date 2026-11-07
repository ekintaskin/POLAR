"""Protocol optimization with an outer CMA-ES loop."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import cma
import numpy as np

from .constraints import ScannerConstraints
from .fitting import FitSettings
from .models.base import SignalModel, Substrate
from .noise import RicianNoise
from .objective import INVALID_LOSS, estimate_all, make_seed_table, squared_errors
from .protocol import PGSEProtocol


@dataclass(frozen=True)
class OptimizerSettings:
    """CMA-ES settings of the outer protocol search."""

    popsize: int = 16
    maxiter: int = 1000
    sigma0: float = 0.2
    min_generations: int = 100
    convergence_window: int = 50
    tolfun: float = 2e-6
    tolfunhist: float = 2e-5
    tolflatfitness: int = 100
    tolstagnation: int = 100
    tolx: float = 1e-6

    def __post_init__(self):
        if self.convergence_window < 2:
            raise ValueError("convergence_window must be at least 2.")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OptimizationResult:
    """Outcome of a run.

    ``loss`` is the lowest value seen, measured under that generation's noise. 
    Use :func:`~polar_dmri.evaluate.evaluate_protocol` on independent substrates to compare protocols.
    """

    protocol: PGSEProtocol
    loss: float
    n_iterations: int
    n_evaluations: int
    stop_reason: dict
    seed: int
    output_dir: Path | None


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _prepare_output_dir(output_dir: str | os.PathLike) -> Path:
    path = Path(output_dir)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"{path} is not empty; choose a new output directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _has_plateaued(medians: Sequence[float], window: int) -> bool:
    """Compare the last two windows of per-generation median losses."""
    previous = np.asarray(medians[-2 * window : -window], dtype=float)
    latest = np.asarray(medians[-window:], dtype=float)
    reduction = previous.mean() - latest.mean()
    standard_error = np.sqrt((previous.var(ddof=1) + latest.var(ddof=1)) / window)
    return bool(reduction <= standard_error)


def _versions() -> dict[str, str]:
    from . import __version__

    return {"polar_dmri": __version__, "cma": cma.__version__, "numpy": np.__version__}


def optimize_protocol(
    model: SignalModel,
    substrates: Sequence[Substrate],
    n_shells: int,
    constraints: ScannerConstraints | None = None,
    noise: RicianNoise | None = None,
    n_noise_realizations: int = 10,
    loss_parameters: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
    settings: OptimizerSettings | None = None,
    fit_settings: FitSettings | None = None,
    initial_protocol: PGSEProtocol | None = None,
    include_b0: bool = True,
    seed: int | None = None,
    n_jobs: int = -1,
    output_dir: str | os.PathLike | None = None,
    verbose: bool = True,
) -> OptimizationResult:
    """Search for the PGSE protocol that minimizes parameter estimation error.

    Parameters
    ----------
    n_shells
        Number of non-b0 shells.
    seed
        Sets the noise draws and the protocol search's sampling. When None, a seed
        is generated and returned in the result. The inner fits are not seeded.
    output_dir
        If given, must be new or empty. Receives ``metadata.json``,
        ``trace.jsonl``, ``population.jsonl``, ``best_protocol.csv`` and
        ``result.json``. The trace is written every generation, so an interrupted
        run keeps its best protocol so far.
    """

    constraints = constraints or ScannerConstraints()
    noise = noise or RicianNoise()
    settings = settings or OptimizerSettings()
    fit_settings = fit_settings or FitSettings()
    if n_shells < 1:
        raise ValueError("n_shells must be at least 1.")
    if n_noise_realizations < 1:
        raise ValueError("n_noise_realizations must be at least 1.")
    model.check_substrates(substrates)
    truth = np.stack([model.ground_truth(substrate) for substrate in substrates])

    squared_errors(model, truth[0], truth[0], loss_parameters, weights)

    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0]) % (2**32 - 1) + 1
    seed = int(seed)
    if not 1 <= seed < 2**32:
        raise ValueError(
            "seed must be between 1 and 2**32 - 1: CMA-ES treats 0 as 'seed from the clock'."
        )
    noise_rng = np.random.default_rng(seed)

    n_variables = 3 * n_shells
    x0 = (
        constraints.encode(initial_protocol)
        if initial_protocol is not None
        else np.full(n_variables, 0.5)
    )
    if x0.size != n_variables:
        raise ValueError(f"initial_protocol must have {n_shells} shells.")

    run_dir = None if output_dir is None else _prepare_output_dir(output_dir)
    if run_dir is not None:
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "versions": _versions(),
            "model": model.name,
            "parameters": [asdict(parameter) for parameter in model.parameters],
            "n_shells": n_shells,
            "include_b0": include_b0,
            "constraints": constraints.as_dict(),
            "noise": noise.as_dict(),
            "n_noise_realizations": n_noise_realizations,
            "loss_parameters": list(loss_parameters or model.parameter_names),
            "weights": dict(weights or {}),
            "optimizer": settings.as_dict(),
            "fit": fit_settings.as_dict(),
            "seed": seed,
            "substrates": [
                {
                    **substrate.params,
                    **{k: v for k, v in substrate.extras.items() if isinstance(v, str)},
                }
                for substrate in substrates
            ],
        }
        (run_dir / "metadata.json").write_text(
            json.dumps(_json_safe(metadata), indent=2) + "\n", encoding="utf-8"
        )

    es = cma.CMAEvolutionStrategy(
        x0,
        settings.sigma0,
        {
            "bounds": [np.zeros(n_variables), np.ones(n_variables)],
            "popsize": settings.popsize,
            "maxiter": settings.maxiter,
            "tolfun": settings.tolfun,
            "tolfunhist": settings.tolfunhist,
            "tolflatfitness": settings.tolflatfitness,
            "tolstagnation": settings.tolstagnation,
            "tolx": settings.tolx,
            "seed": seed,
            "verbose": 1 if verbose else -9,
            "verb_disp": 1 if verbose else 0,
            "verb_log": 0,
        },
    )

    iteration = 0
    medians: list[float] = []
    plateaued = False
    while not plateaued and not es.stop():
        iteration += 1
        seed_table = make_seed_table(noise_rng, n_noise_realizations, len(substrates))
        candidates = [np.asarray(x, dtype=float) for x in es.ask()]
        protocols = [constraints.decode(x, include_b0) for x in candidates]
        feasible = [constraints.is_feasible(protocol) for protocol in protocols]
        
        penalties = [
            INVALID_LOSS * (1.0 + constraints.violation(protocol)) for protocol in protocols
        ]

        estimates = iter(
            estimate_all(
                model,
                [protocol for protocol, ok in zip(protocols, feasible) if ok],
                substrates,
                noise,
                seed_table,
                fit_settings,
                n_jobs,
            )
            if any(feasible)
            else []
        )
        losses = [
            float(
                np.mean(
                    squared_errors(model, truth[:, None], next(estimates), loss_parameters, weights)
                )
            )
            if ok
            else penalty
            for ok, penalty in zip(feasible, penalties)
        ]

        es.tell(candidates, losses)
        if verbose:
            es.disp()
        feasible_losses = [loss for loss, ok in zip(losses, feasible) if ok]
        median_loss = float(np.median(feasible_losses)) if feasible_losses else float("nan")
        medians.append(median_loss)
        window = settings.convergence_window
        plateaued = (
            iteration >= max(settings.min_generations, 2 * window)
            and iteration % window == 0
            and _has_plateaued(medians, window)
        )
        if run_dir is not None:
            with open(run_dir / "population.jsonl", "a", encoding="utf-8") as handle:
                for index, (protocol, loss) in enumerate(zip(protocols, losses)):
                    entry = {
                        "iter": iteration,
                        "candidate": index,
                        "loss": loss,
                        "sigma": float(es.sigma),
                        "protocol": protocol.as_dict(),
                    }
                    handle.write(json.dumps(entry) + "\n")
            best_so_far = constraints.decode(es.result.xbest, include_b0)
            with open(run_dir / "trace.jsonl", "a", encoding="utf-8") as handle:
                entry = {
                    "iter": iteration,
                    "evaluations": int(es.countevals),
                    "generation_best_loss": min(losses),
                    "median_loss": median_loss,
                    "best_loss": float(es.result.fbest),
                    "sigma": float(es.sigma),
                    "best_protocol": best_so_far.as_dict(),
                }
                handle.write(json.dumps(entry) + "\n")

    best_protocol = constraints.decode(es.result.xbest, include_b0)
    if not constraints.is_feasible(best_protocol):
        raise RuntimeError(
            f"No feasible protocol found in {iteration} generations: "
            "the limits may be too tight, for example b_max against the delta and Delta ranges."
        )
    result = OptimizationResult(
        protocol=best_protocol,
        loss=float(es.result.fbest),
        n_iterations=iteration,
        n_evaluations=int(es.countevals),
        stop_reason={"median_loss_plateau": iteration} if plateaued else _json_safe(dict(es.stop())),
        seed=seed,
        output_dir=run_dir,
    )
    if run_dir is not None:
        best_protocol.save_csv(run_dir / "best_protocol.csv")
        summary = {
            "loss": result.loss,
            "n_iterations": result.n_iterations,
            "n_evaluations": result.n_evaluations,
            "stop_reason": result.stop_reason,
            "seed": seed,
            "te_ms": best_protocol.te * 1e3,
        }
        (run_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return result
