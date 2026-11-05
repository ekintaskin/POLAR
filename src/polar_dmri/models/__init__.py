from __future__ import annotations

import importlib
from typing import Callable

from .base import Parameter, SignalModel, Substrate, make_substrate_grid
from .cexi import CEXI

_REGISTRY: dict[str, Callable[..., SignalModel]] = {"cexi": CEXI}


def register_model(name: str, factory: Callable[..., SignalModel]) -> None:
    """Make a model available by name, for example in configuration files."""
    _REGISTRY[name.lower()] = factory


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def get_model(name: str, **kwargs) -> SignalModel:
    """Create a model from a registered name or a ``package.module:Class`` path."""
    if ":" in name:
        module_name, _, attribute = name.partition(":")
        factory = getattr(importlib.import_module(module_name), attribute)
    else:
        try:
            factory = _REGISTRY[name.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown model {name!r}. Available: {available_models()}, "
                "or give a 'package.module:Class' path."
            ) from None
    return factory(**kwargs)


__all__ = [
    "CEXI",
    "Parameter",
    "SignalModel",
    "Substrate",
    "available_models",
    "get_model",
    "make_substrate_grid",
    "register_model",
]
