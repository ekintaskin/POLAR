"""Measurement noise model."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class RicianNoise:
    """
    The noise level is fixed by a reference acquisition, so that longer echo
    times are penalized through attenuated signal rather than a changed noise floor:
    ``sigma = exp(-te_ref / t2) / snr``.

    Parameters
    ----------
    snr
        SNR of the b=0 signal at the reference echo time.
    te_ref
        Reference echo time (s).
    t2
        T2 used to define the reference signal (s).
    n_directions
        Gradient directions per shell. Each is corrupted independently and the
        magnitudes are averaged, which reduces the variance but not the Rician
        bias.
    """

    snr: float = 20.0
    te_ref: float = 70e-3
    t2: float = 60e-3
    n_directions: int = 20

    def __post_init__(self):
        if self.snr <= 0 or self.te_ref <= 0 or self.t2 <= 0:
            raise ValueError("snr, te_ref and t2 must be positive.")
        if self.n_directions < 1:
            raise ValueError("n_directions must be at least 1.")

    @property
    def sigma(self) -> float:
        return float(np.exp(-self.te_ref / self.t2) / self.snr)

    def apply(self, clean_signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return one noisy, direction-averaged copy of ``clean_signal``."""
        size = (self.n_directions, clean_signal.size)
        real = rng.normal(0.0, self.sigma, size=size)
        imaginary = rng.normal(0.0, self.sigma, size=size)
        return np.sqrt((clean_signal + real) ** 2 + imaginary**2).mean(axis=0)

    def as_dict(self) -> dict:
        return asdict(self)
