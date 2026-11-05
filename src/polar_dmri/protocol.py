"""PGSE acquisition protocols.

Units are SI (seconds, T/m), except b-values (s/mm^2).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np

GYROMAGNETIC_RATIO = 267.5153151e6  # rad/s/T


def pgse_b_value(delta, big_delta, G) -> np.ndarray:
    """Stejskal-Tanner b-value in s/mm^2 for rectangular PGSE pulses."""
    delta, big_delta, G = np.asarray(delta), np.asarray(big_delta), np.asarray(G)
    return (GYROMAGNETIC_RATIO * G * delta) ** 2 * (big_delta - delta / 3.0) / 1e6


class Measurements(NamedTuple):

    b: np.ndarray
    delta: np.ndarray
    big_delta: np.ndarray
    G: np.ndarray
    te: float


@dataclass(frozen=True, eq=False)
class PGSEProtocol:
    """A set of PGSE shells acquired at one common echo time.

    Parameters
    ----------
    delta, big_delta, G
        Pulse duration (s), pulse separation (s) and gradient strength (T/m),
        one value per shell.
    te_overhead
        Time added to ``delta + big_delta`` to get a shell's minimum TE (s).
    include_b0
        Whether a b=0 measurement is acquired in addition to the shells.
    """

    delta: np.ndarray
    big_delta: np.ndarray
    G: np.ndarray
    te_overhead: float = 10e-3
    include_b0: bool = True

    def __post_init__(self):
        for name in ("delta", "big_delta", "G"):
            values = np.atleast_1d(np.asarray(getattr(self, name), dtype=float))
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a one-dimensional array of finite values.")
            object.__setattr__(self, name, values)
        if not self.delta.size == self.big_delta.size == self.G.size:
            raise ValueError("delta, big_delta and G must have the same length.")

    @property
    def n_shells(self) -> int:
        return self.delta.size

    @cached_property
    def b_values(self) -> np.ndarray:
        """b-value of each shell in s/mm^2."""
        return pgse_b_value(self.delta, self.big_delta, self.G)

    @cached_property
    def shell_te(self) -> np.ndarray:
        """Minimum echo time of each shell in seconds."""
        return self.delta + self.big_delta + self.te_overhead

    @cached_property
    def te(self) -> float:
        """Common echo time: the longest shell sets it for all of them."""
        return float(np.max(self.shell_te))

    @cached_property
    def measurements(self) -> Measurements:
        if not self.include_b0:
            return Measurements(self.b_values, self.delta, self.big_delta, self.G, self.te)
        pad = lambda values: np.insert(values, 0, 0.0)
        return Measurements(
            pad(self.b_values), pad(self.delta), pad(self.big_delta), pad(self.G), self.te
        )

    def as_dict(self) -> dict:
        return {
            "delta": self.delta.tolist(),
            "big_delta": self.big_delta.tolist(),
            "G": self.G.tolist(),
            "te_overhead": self.te_overhead,
            "include_b0": self.include_b0,
        }

    def save_csv(self, path: str | os.PathLike) -> None:
        """Write one row per shell. Values keep full floating-point precision."""
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["delta_ms", "Delta_ms", "G_mT_per_m", "b_s_per_mm2", "TE_ms"])
            for row in zip(self.delta, self.big_delta, self.G, self.b_values, self.shell_te):
                delta, big_delta, G, b, te = (float(value) for value in row)
                writer.writerow([delta * 1e3, big_delta * 1e3, G * 1e3, b, te * 1e3])

    @classmethod
    def load_csv(
        cls,
        path: str | os.PathLike,
        te_overhead: float | None = None,
        include_b0: bool = True,
    ) -> "PGSEProtocol":
        """Read a protocol from CSV.

        ``delta_ms``, ``Delta_ms`` and ``G_mT_per_m`` are required. The b-value
        is always recomputed. ``te_overhead`` is taken from the ``TE_ms`` column
        when not given, and defaults to 10 ms when that column is absent.
        """
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No shells found in {path}.")
        missing = {"delta_ms", "Delta_ms", "G_mT_per_m"} - set(rows[0])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}.")

        column = lambda name: np.array([float(row[name]) for row in rows])
        delta = column("delta_ms") * 1e-3
        big_delta = column("Delta_ms") * 1e-3
        if te_overhead is None:
            te_overhead = 10e-3
            if "TE_ms" in rows[0]:
                te_overhead = float(np.max(column("TE_ms") * 1e-3 - delta - big_delta))
        return cls(delta, big_delta, column("G_mT_per_m") * 1e-3, te_overhead, include_b0)
