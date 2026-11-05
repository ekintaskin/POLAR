"""Optimize diffusion MRI acquisition protocols for parameter recovery."""

from .constraints import ScannerConstraints
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
from .protocol import PGSEProtocol, pgse_b_value

__version__ = "0.1.0"

__all__ = [
    "CEXI",
    "PGSEProtocol",
    "Parameter",
    "ScannerConstraints",
    "SignalModel",
    "Substrate",
    "available_models",
    "get_model",
    "make_substrate_grid",
    "pgse_b_value",
    "register_model",
]
