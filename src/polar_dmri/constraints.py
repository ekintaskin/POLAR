"""Scanner limits and the optimizer's normalized protocol coordinates."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from .protocol import GYROMAGNETIC_RATIO, PGSEProtocol, pgse_b_value

# Squared-violation tolerance. Allows rounding errors of order 1e-9 s.
_VIOLATION_TOL = 1e-18


@dataclass(frozen=True)
class ScannerConstraints:

    """Limits on a PGSE protocol. Units: seconds, T/m and s/mm^2.

    Parameters
    ----------
    delta_bounds, big_delta_bounds, G_bounds
        Allowed range of each shell's delta, Delta and G.
    min_gap
        Minimum ``Delta - delta``.
    te_overhead
        Time added to ``delta + Delta`` to get a shell's minimum TE.
    te_max
        Longest allowed TE. None for no limit.
    b_max
        Largest allowed b-value. None for no limit.
    """

    delta_bounds: tuple[float, float] = (5e-3, 40e-3)
    big_delta_bounds: tuple[float, float] = (8e-3, 50e-3)
    G_bounds: tuple[float, float] = (0.010, 0.180)
    min_gap: float = 3e-3
    te_overhead: float = 10e-3
    te_max: float | None = 120e-3
    b_max: float | None = None

    def __post_init__(self):
        for name in ("delta_bounds", "big_delta_bounds", "G_bounds"):
            low, high = getattr(self, name)
            if not 0.0 < low < high:
                raise ValueError(f"{name} must satisfy 0 < low < high, got {(low, high)}.")
            
        delta_high = np.array([self.delta_bounds[1]])
        if np.any(self._big_delta_range(delta_high)[1] < self._big_delta_range(delta_high)[0]):
            raise ValueError(
                "No feasible Delta exists at the longest delta: widen big_delta_bounds or "
                "te_max, or shorten delta_bounds, min_gap or te_overhead."
            )
        if self.b_max is not None:
            delta_low = self.delta_bounds[0]
            big_delta_low = self._big_delta_range(np.array([delta_low]))[0][0]
            if pgse_b_value(delta_low, big_delta_low, self.G_bounds[0]) > self.b_max:
                raise ValueError("b_max is below the smallest b-value these bounds allow.")

    def _big_delta_range(self, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        low = np.maximum(self.big_delta_bounds[0], delta + self.min_gap)
        high = np.full(delta.shape, self.big_delta_bounds[1], dtype=float)
        if self.te_max is not None:
            high = np.minimum(high, self.te_max - self.te_overhead - delta)
        return low, high

    def _G_high(self, delta: np.ndarray, big_delta: np.ndarray) -> np.ndarray:
        """Largest allowed G for these timings; ``G_bounds[0]`` if b_max rules out all."""

        if self.b_max is None:
            return np.full(delta.shape, self.G_bounds[1], dtype=float)
        G_at_b_max = np.sqrt(
            self.b_max * 1e6 / ((GYROMAGNETIC_RATIO * delta) ** 2 * (big_delta - delta / 3.0))
        )
        return np.clip(G_at_b_max, self.G_bounds[0], self.G_bounds[1])

    def decode(self, x, include_b0: bool = True) -> PGSEProtocol:
        """Map a point of the unit cube to a protocol."""

        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size == 0 or x.size % 3 != 0:
            raise ValueError(f"Expected a vector of length 3 * n_shells, got shape {x.shape}.")
        if not np.all(np.isfinite(x)):
            raise ValueError("Optimizer vector contains non-finite values.")

        delta_unit, big_delta_unit, G_unit = np.split(np.clip(x, 0.0, 1.0), 3)
        delta = self.delta_bounds[0] + delta_unit * (self.delta_bounds[1] - self.delta_bounds[0])
        low, high = self._big_delta_range(delta)
        big_delta = low + big_delta_unit * (high - low)
        G = self.G_bounds[0] + G_unit * (self._G_high(delta, big_delta) - self.G_bounds[0])
        return PGSEProtocol(delta, big_delta, G, self.te_overhead, include_b0)

    def encode(self, protocol: PGSEProtocol) -> np.ndarray:
        """Map a protocol to the unit cube, clipping anything out of range."""

        delta_unit = (protocol.delta - self.delta_bounds[0]) / (
            self.delta_bounds[1] - self.delta_bounds[0]
        )
        low, high = self._big_delta_range(protocol.delta)
        if np.any(high <= low):
            raise ValueError("Protocol timings cannot be represented by these constraints.")
        big_delta_unit = (protocol.big_delta - low) / (high - low)
        G_span = self._G_high(protocol.delta, protocol.big_delta) - self.G_bounds[0]
        G_unit = np.divide(
            protocol.G - self.G_bounds[0], G_span, out=np.zeros_like(G_span), where=G_span > 0
        )
        return np.clip(np.concatenate((delta_unit, big_delta_unit, G_unit)), 0.0, 1.0)

    def violation(self, protocol: PGSEProtocol) -> float:
        """Sum of squared limit violations in native units; 0 when feasible."""
        
        excess = lambda values: float(np.sum(np.clip(values, 0.0, None) ** 2))
        delta, big_delta, G = protocol.delta, protocol.big_delta, protocol.G
        total = (
            excess(self.delta_bounds[0] - delta)
            + excess(delta - self.delta_bounds[1])
            + excess(self.big_delta_bounds[0] - big_delta)
            + excess(big_delta - self.big_delta_bounds[1])
            + excess(self.G_bounds[0] - G)
            + excess(G - self.G_bounds[1])
            + excess(delta + self.min_gap - big_delta)
        )
        if self.te_max is not None:
            total += excess(delta + big_delta + self.te_overhead - self.te_max)
        if self.b_max is not None:
            total += excess(protocol.b_values - self.b_max)
        return total

    def is_feasible(self, protocol: PGSEProtocol) -> bool:
        return self.violation(protocol) <= _VIOLATION_TOL

    def as_dict(self) -> dict:
        return asdict(self)
