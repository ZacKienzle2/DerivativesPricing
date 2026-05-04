"""Process registry mapping name tokens to classes and noise dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Type

from .base import BaseProcess


@dataclass(frozen=True)
class ProcessSpec:
    """Carrier describing a registered process implementation.

    Attributes:
        cls: Concrete `BaseProcess` subclass.
        noise_dim: Number of independent normals required per timestep.
    """

    cls: Type[BaseProcess]
    noise_dim: int


_REGISTRY: Dict[str, ProcessSpec] = {}


def register_process(name: str, cls: Type[BaseProcess], noise_dim: int) -> None:
    """Adds a process to the registry under `name`."""
    _REGISTRY[name] = ProcessSpec(cls=cls, noise_dim=noise_dim)


def get_spec(name: str) -> ProcessSpec:
    """Returns the spec registered under `name`. Raises KeyError if absent."""
    if name not in _REGISTRY:
        raise KeyError(f"No process registered as {name!r}")
    return _REGISTRY[name]


def make_process(name: str, **params: Any) -> BaseProcess:
    """Builds a process instance by name."""
    return get_spec(name).cls(**params)


def list_processes() -> Tuple[str, ...]:
    """Returns the registered process names in insertion order."""
    return tuple(_REGISTRY.keys())


def _register_defaults() -> None:
    from .bates import BatesProcess
    from .gbm import GBMProcess
    from .heston import HestonProcess
    from .local_vol import LocalVolProcess
    from .rbergomi import RBergomiProcess
    from .sabr import SABRProcess

    register_process("GBM", GBMProcess, 1)
    register_process("Heston", HestonProcess, 2)
    register_process("Bates", BatesProcess, 3)
    register_process("SABR", SABRProcess, 2)
    register_process("LocalVol", LocalVolProcess, 1)
    register_process("RBergomi", RBergomiProcess, 2)


_register_defaults()
