"""CEXI (Cellular Exchange Imaging): permeable spheres in an extracellular space.

1. Intracellular: restricted diffusion in a sphere of radius R.
2. Extracellular: Gaussian diffusion with diffusivity De.
3. Exchange: the Karger model, with k_i = 3 * kappa / R and
   k_e = k_i * f / (1 - f) by detailed balance. For a radius distribution, R is
   the volume-weighted radius.

From Gardier et al., Magn Reson Med 2023;90(4):1625-1640,
doi:10.1002/mrm.29720. This implementation derives from the CEXI tutorial code at
https://github.com/jonhrafe/CEXI-public, with volume-weighted radius
distributions, T2 attenuation and vectorized evaluation added. That code is
distributed under the following license:

MIT License

Copyright (c) 2025 CEXI Development Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import brentq

from ..protocol import PGSEProtocol
from .base import Parameter, SignalModel, Substrate, make_substrate_grid

_UM_TO_M = 1e-6
_EPS = 1e-10
_MAX_EXPONENT = 50.0
_MIN_DENOMINATOR = 1e-30

KAPPA_NO_EXCHANGE = 0.01

DEFAULT_BOUNDS = {
    "R": (1.0, 15.0),
    "Di": (0.3e-9, 2.0e-9),
    "De": (1.0e-9, 3.5e-9),
    "f": (0.3, 0.85),
    "kappa": (1e-3, 100.0),
}
_UNITS = {"R": "um", "Di": "m^2/s", "De": "m^2/s", "f": "", "kappa": "um/s"}


@lru_cache(maxsize=1)
def _sphere_roots(n_roots: int = 32) -> np.ndarray:
    """Positive zeros of j'_1(x), i.e. of (x^2 - 2) sin(x) + 2 x cos(x)."""
    g = lambda x: (x**2 - 2.0) * np.sin(x) + 2.0 * x * np.cos(x)
    x = np.linspace(1e-4, (n_roots + 2) * np.pi, 20000)
    sign_changes = np.flatnonzero(g(x[:-1]) * g(x[1:]) < 0)[:n_roots]
    return np.array([brentq(g, x[i], x[i + 1]) for i in sign_changes])


def sphere_signal(radii_um, D, b, delta, big_delta) -> np.ndarray:
    """Restricted signal in spheres; returns an array of shape (n_b, n_radii).

    ``b`` is in s/mm^2.
    """
    radii_m = np.asarray(radii_um, dtype=float).reshape(-1, 1) * _UM_TO_M
    alpha2 = (_sphere_roots() / radii_m) ** 2
    denominator = D**2 * alpha2**3 * (alpha2 * radii_m**2 - 2.0)

    delta_3d = np.asarray(delta, dtype=float)[:, None, None]
    big_delta_3d = np.asarray(big_delta, dtype=float)[:, None, None]
    d_delta = D * delta_3d * alpha2
    d_big_delta = D * big_delta_3d * alpha2
    decay = lambda exponent: np.exp(np.maximum(-exponent, -_MAX_EXPONENT))

    numerator = (
        2.0 * d_delta
        - 2.0
        + 2.0 * decay(d_delta)
        + 2.0 * decay(d_big_delta)
        - decay(d_big_delta - d_delta)
        - decay(d_big_delta + d_delta)
    )
    valid = np.abs(denominator) > _MIN_DENOMINATOR
    series = np.sum(
        np.divide(numerator, denominator, out=np.zeros_like(numerator), where=valid[None]),
        axis=-1,
    )

    b_si = np.asarray(b, dtype=float) * 1e6
    delta_1d, big_delta_1d = delta_3d[:, 0, 0], big_delta_3d[:, 0, 0]
    gamma2_G2 = b_si / (delta_1d**2 * (big_delta_1d - delta_1d / 3.0))
    return np.clip(np.exp(-2.0 * gamma2_G2[:, None] * series), _EPS, 1.0)


def radius_distribution(R) -> tuple[np.ndarray, np.ndarray, float]:
    """Return unique radii, their volume weights and the weighted mean radius."""
    radii = np.atleast_1d(np.asarray(R, dtype=float)).reshape(-1)
    if radii.size == 0 or not np.all(np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("Radii must be non-empty, finite and positive.")
    unique, counts = np.unique(radii, return_counts=True)
    weights = counts * unique**3
    weights = weights / np.sum(weights)
    return unique, weights, float(np.sum(weights * unique))


def load_radii(path: str | os.PathLike) -> np.ndarray:
    """Read sphere radii (um) from a text file with one radius per line."""
    try:
        radii = np.loadtxt(path, dtype=float, ndmin=2)
        if radii.shape[1] != 1:
            raise ValueError("expected one radius per line.")
        radius_distribution(radii)
    except ValueError as error:
        raise ValueError(f"Invalid radii file {path}: {error}") from error
    return radii.reshape(-1)


def cexi_signal(R, Di, De, f, kappa, b, delta, big_delta, te, T2=60e-3) -> np.ndarray:
    """T2-weighted CEXI signal.

    Parameters
    ----------
    R
        Sphere radius (um), or an array of radii sampling a distribution. Exchange uses the volume-weighted mean radius.
    Di, De
        Intracellular and extracellular diffusivity (m^2/s).
    f
        Intracellular volume fraction.
    kappa
        Membrane permeability (um/s).
    b, delta, big_delta
        Per-measurement b-value (s/mm^2), pulse duration and separation (s).
    te, T2
        Common echo time and the T2 of both compartments (s).
    """
    if not 0.0 < f < 1.0:
        raise ValueError(f"f must be inside (0, 1), got {f}.")
    b = np.asarray(b, dtype=float)
    delta = np.asarray(delta, dtype=float)
    big_delta = np.asarray(big_delta, dtype=float)
    radii, weights, mean_radius = radius_distribution(R)

    t2_decay = np.exp(-te / T2)
    signal = np.full(b.shape, t2_decay, dtype=float)
    weighted = b > 0
    if not np.any(weighted):
        return signal

    b_si = b[weighted] * 1e6
    delta, big_delta = delta[weighted], big_delta[weighted]
    restricted = sphere_signal(radii, Di, b[weighted], delta, big_delta) @ weights
    Di_app = -np.log(np.maximum(restricted, _EPS)) / b_si

    if kappa < KAPPA_NO_EXCHANGE:
        attenuation = f * np.exp(-Di_app * b_si) + (1.0 - f) * np.exp(-De * b_si)
    else:
        k_i = 3.0 * kappa / mean_radius
        k_e = k_i * f / (1.0 - f)
        q2 = b_si / (big_delta - delta / 3.0)
        x_e = De + k_e / q2
        x_i = Di_app + k_i / q2
        spread = np.sqrt((x_i - x_e) ** 2 + 4.0 * k_i * k_e / q2**2)
        D_slow = 0.5 * (x_i + x_e - spread)
        D_fast = 0.5 * (x_i + x_e + spread)
        f_fast = (f * Di_app + (1.0 - f) * De - D_slow) / spread
        attenuation = (1.0 - f_fast) * np.exp(-D_slow * b_si) + f_fast * np.exp(-D_fast * b_si)

    signal[weighted] = t2_decay * attenuation
    return signal


class CEXI(SignalModel):
    """CEXI with parameters R (um), Di, De (m^2/s), f and kappa (um/s).

    All five are fitted on log axes. Pass ``bounds`` to override individual
    search ranges, and ``T2`` (s) to set the assumed T2. S0 is assumed to be 1.

    Substrates have a single radius by default. To simulate polydisperse tissue,
    build them with :meth:`make_substrates` from radius distribution files. Their
    fit target for R is the volume-weighted mean radius, and the fit itself
    always assumes a single radius.
    """

    name = "cexi"

    def __init__(self, T2: float = 60e-3, bounds: Mapping[str, tuple[float, float]] | None = None):
        if T2 <= 0:
            raise ValueError("T2 must be positive.")
        unknown = set(bounds or {}) - set(DEFAULT_BOUNDS)
        if unknown:
            raise ValueError(f"Unknown CEXI parameters in bounds: {sorted(unknown)}.")
        merged = {**DEFAULT_BOUNDS, **(bounds or {})}
        if not 0.0 < merged["f"][0] < merged["f"][1] < 1.0:
            raise ValueError("Bounds for f must lie strictly inside (0, 1).")
        self.T2 = T2
        self.parameters = tuple(
            Parameter(name, float(low), float(high), _UNITS[name], "log")
            for name, (low, high) in merged.items()
        )

    def make_substrates(self, axes: Mapping[str, Sequence]) -> list[Substrate]:
        """Build a grid from single radii (``R``) or from ``radii_files``."""
        axes = dict(axes)
        files = axes.pop("radii_files", None)
        if files is None:
            return make_substrate_grid(axes)
        if "R" in axes:
            raise ValueError("Give either R or radii_files, not both.")
        files = [files] if isinstance(files, (str, os.PathLike)) else list(files)
        substrates = []
        for path in files:
            radii = load_radii(path)
            extras = {"radii": radii, "radii_file": str(path)}
            grid = make_substrate_grid({"R": [radius_distribution(radii)[2]], **axes})
            substrates += [Substrate(substrate.params, extras) for substrate in grid]
        return substrates

    def _evaluate(self, R, params: Mapping[str, float], protocol: PGSEProtocol) -> np.ndarray:
        m = protocol.measurements
        return cexi_signal(
            R, params["Di"], params["De"], params["f"], params["kappa"],
            m.b, m.delta, m.big_delta, m.te, self.T2,
        )  

    def signal(self, params: Mapping[str, float], protocol: PGSEProtocol) -> np.ndarray:
        return self._evaluate(params["R"], params, protocol)

    def simulate(self, substrate: Substrate, protocol: PGSEProtocol) -> np.ndarray:
        radii = substrate.extras.get("radii", substrate.params["R"])
        return self._evaluate(radii, substrate.params, protocol)

    def ground_truth(self, substrate: Substrate) -> np.ndarray:
        truth = super().ground_truth(substrate)
        if "radii" in substrate.extras:
            truth[self.parameter_names.index("R")] = radius_distribution(
                substrate.extras["radii"]
            )[2]
        return truth
