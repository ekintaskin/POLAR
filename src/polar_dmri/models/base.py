"""The interface a signal model implements to be optimized for."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..protocol import PGSEProtocol

_EPS = 1e-12


@dataclass(frozen=True)
class Parameter:
    """A fitted model parameter and its search range.

    ``scale`` sets how the range maps to [0, 1], which is the space both the fit
    and the optimization loss work in. Use ``"log"`` for positive parameters
    spanning orders of magnitude or whose effect is roughly multiplicative.
    """

    name: str
    low: float
    high: float
    unit: str = ""
    scale: str = "log"

    def __post_init__(self):
        if self.scale not in ("log", "linear"):
            raise ValueError(f"{self.name}: scale must be 'log' or 'linear'.")
        if not (np.isfinite(self.low) and np.isfinite(self.high) and self.low < self.high):
            raise ValueError(f"{self.name}: invalid bounds {(self.low, self.high)}.")
        if self.scale == "log" and self.low <= 0.0:
            raise ValueError(f"{self.name}: a log-scaled parameter needs a positive lower bound.")

    def to_unit(self, value):
        """Map physical values to [0, 1], clipping values outside the bounds."""
        value = np.asarray(value, dtype=float)
        if self.scale == "log":
            unit = (np.log(np.maximum(value, _EPS)) - np.log(self.low)) / (
                np.log(self.high) - np.log(self.low)
            )
        else:
            unit = (value - self.low) / (self.high - self.low)
        return np.clip(unit, 0.0, 1.0)

    def from_unit(self, unit):
        """Map [0, 1] back to physical values."""
        unit = np.asarray(unit, dtype=float)
        if self.scale == "log":
            return np.exp(np.log(self.low) + unit * (np.log(self.high) - np.log(self.low)))
        return self.low + unit * (self.high - self.low)


@dataclass(frozen=True)
class Substrate:
    """A tissue configuration the protocol is optimized for.

    ``extras`` carries whatever else a model needs to simulate the substrate but
    does not fit, such as CEXI's radius distribution.
    """

    params: Mapping[str, float]
    extras: Mapping[str, object] = field(default_factory=dict)


def make_substrate_grid(axes: Mapping[str, Sequence[float]]) -> list[Substrate]:
    """Build one substrate per combination of the given parameter values."""
    if not axes:
        raise ValueError("At least one parameter axis is required.")
    names = list(axes)
    values = [np.atleast_1d(np.asarray(axes[name], dtype=float)) for name in names]
    return [
        Substrate({name: float(value) for name, value in zip(names, combination)})
        for combination in itertools.product(*values)
    ]


class SignalModel(ABC):
    """A forward model, plus the parameters a fit should recover.

    Subclasses set ``name`` and ``parameters`` and implement :meth:`signal`.
    Instances are sent to worker processes, so they must be picklable.
    """

    name: str = "model"
    parameters: tuple[Parameter, ...] = ()

    @abstractmethod
    def signal(self, params: Mapping[str, float], protocol: PGSEProtocol) -> np.ndarray:
        """Return the noise-free signal, one value per measurement.

        The signal must be on the scale the noise model assumes, with b=0
        first if ``protocol.include_b0``. See ``protocol.measurements``.
        """

    def make_substrates(self, axes: Mapping[str, Sequence]) -> list[Substrate]:
        """Build a substrate grid. Override to accept axes other than parameters."""
        return make_substrate_grid(axes)

    def simulate(self, substrate: Substrate, protocol: PGSEProtocol) -> np.ndarray:
        """Signal of a substrate. Override if ``substrate.extras`` affect it."""
        return self.signal(substrate.params, protocol)

    def ground_truth(self, substrate: Substrate) -> np.ndarray:
        """Values a perfect fit returns, ordered as ``parameters``."""
        return np.array([substrate.params[name] for name in self.parameter_names], dtype=float)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    def to_unit(self, values) -> np.ndarray:
        """Map values of shape ``(..., n_parameters)`` to [0, 1]."""
        values = np.asarray(values, dtype=float)
        return np.stack(
            [p.to_unit(values[..., i]) for i, p in enumerate(self.parameters)], axis=-1
        )

    def from_unit(self, unit) -> np.ndarray:
        """Map values of shape ``(..., n_parameters)`` from [0, 1]."""
        unit = np.asarray(unit, dtype=float)
        return np.stack([p.from_unit(unit[..., i]) for i, p in enumerate(self.parameters)], axis=-1)

    def check_substrates(self, substrates: Sequence[Substrate]) -> None:
        """Raise if any substrate cannot be used with this model."""
        if not substrates:
            raise ValueError("At least one substrate is required.")
        expected = set(self.parameter_names)
        for index, substrate in enumerate(substrates):
            given = set(substrate.params)
            if given != expected:
                raise ValueError(
                    f"Substrate {index} must define exactly {sorted(expected)}; "
                    f"got {sorted(given)}."
                )
