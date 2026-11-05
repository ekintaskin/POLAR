"""Optimize diffusion MRI acquisition protocols for parameter recovery."""

from .constraints import ScannerConstraints
from .protocol import PGSEProtocol, pgse_b_value

__version__ = "0.1.0"

__all__ = [
    "PGSEProtocol",
    "ScannerConstraints",
    "pgse_b_value",
]
