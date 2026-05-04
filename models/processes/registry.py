"""Process registry mapping name tokens to classes and noise dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .base import BaseProcess


@dataclass(frozen=True)
class ProcessSpec:
    """Carrier describing a registered process implementation.

    Attributes:
        cls: Concrete `BaseProcess` subclass.
        noise_dim: Number of independent normals required per timestep.
    """

    cls: type[BaseProcess]
    noise_dim: int


_REGISTRY: dict[str, ProcessSpec] = {}


def register_process(
    name: str, cls: type[BaseProcess], noise_dim: int
) -> None:
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


def list_processes() -> tuple[str, ...]:
    """Returns the registered process names in insertion order."""
    return tuple(_REGISTRY.keys())


def autoregister(
    name: str, noise_dim: int
) -> Callable[[type[BaseProcess]], type[BaseProcess]]:
    """Class decorator equivalent to `register_process`."""

    def wrap(cls: type[BaseProcess]) -> type[BaseProcess]:
        register_process(name, cls, noise_dim)
        return cls

    return wrap
