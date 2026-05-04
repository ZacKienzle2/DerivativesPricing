"""Structural protocols decoupling consumers from concrete pricer types."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HasZMatrix(Protocol):
    """Pricer that exposes a swappable Monte Carlo driver matrix."""

    z_matrix: Any


@runtime_checkable
class HasConvergenceData(Protocol):
    """Pricer that exposes a per-sample convergence trace."""

    convergence_data: Any


@runtime_checkable
class GreekProvider(Protocol):
    """Object that can compute a Greek dictionary on demand."""

    def greeks(self) -> dict[str, float]:
        ...


@runtime_checkable
class Priceable(Protocol):
    """Object that returns `(price, std_err)` from `.price()`."""

    def price(self) -> tuple[float, float]:
        ...
