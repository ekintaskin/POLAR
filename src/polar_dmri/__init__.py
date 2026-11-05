"""Optimize diffusion MRI acquisition protocols for parameter recovery."""

from .constraints import ScannerConstraints
from .evaluate import EvaluationResult, compare_protocols, evaluate_protocol, write_summary_csv
from .fitting import FitResult, FitSettings, fit_signal
from .models import (
    CEXI,
    Parameter,
    SignalModel,
    Substrate,
    available_models,
    get_model,
    make_substrate_grid,
    register_model,
)
from .noise import RicianNoise
from .optimize import OptimizationResult, OptimizerSettings, optimize_protocol
from .protocol import PGSEProtocol, pgse_b_value

__version__ = "0.1.0"

__all__ = [
    "CEXI",
    "EvaluationResult",
    "FitResult",
    "FitSettings",
    "OptimizationResult",
    "OptimizerSettings",
    "PGSEProtocol",
    "Parameter",
    "RicianNoise",
    "ScannerConstraints",
    "SignalModel",
    "Substrate",
    "available_models",
    "compare_protocols",
    "evaluate_protocol",
    "fit_signal",
    "get_model",
    "make_substrate_grid",
    "optimize_protocol",
    "pgse_b_value",
    "register_model",
    "write_summary_csv",
]
