"""Explicit finite-difference Black-Scholes pricer."""

from typing import Any, Dict, Tuple

from ..options import BaseOption
from ._fd_common import solve_fd
from .base_pricer import BasePricer


class ExplicitFDPricer(BasePricer):
    """Prices vanilla and American options via the explicit FTCS scheme.

    Conditionally stable: requires `dt < ds^2 / (sigma^2 * S_max^2)`. Use
    Crank-Nicolson or implicit schemes for unconditional stability.
    """

    def __init__(
        self,
        option: BaseOption,
        num_steps: int,
        num_points: int,
        cluster_density: float = 0.0,
    ):
        super().__init__(option)
        self.n_steps = num_steps
        self.n_points = num_points
        self.cluster_density = cluster_density

    def get_params(self) -> Dict[str, Any]:
        """Returns the pricer configuration."""
        return {
            "num_steps": self.n_steps,
            "num_points": self.n_points,
            "cluster_density": self.cluster_density,
        }

    def price(self) -> Tuple[float, float]:
        """Returns `(price, 0.0)` — deterministic scheme has no MC error."""
        return (
            solve_fd(
                self.option, self.n_steps, self.n_points, "explicit",
                cluster_density=self.cluster_density,
            ),
            0.0,
        )
