"""Implicit finite-difference Black-Scholes pricer."""

from typing import Any, Dict, Tuple

from ..options import BaseOption
from ._fd_common import solve_fd
from .base_pricer import BasePricer


class ImplicitFDPricer(BasePricer):
    """Prices vanilla and American options via the fully-implicit BTCS scheme.

    Unconditionally stable but only first-order accurate in time. The LU
    factor of the tridiagonal LHS is computed once and reused across every
    timestep. American exercise can be handled via either operator-splitting
    max-projection (`american_method='project'`) or LCP-correct projected SOR
    (`american_method='psor'`).
    """

    def __init__(
        self,
        option: BaseOption,
        num_steps: int,
        num_points: int,
        american_method: str = "project",
    ):
        super().__init__(option)
        self.n_steps = num_steps
        self.n_points = num_points
        self.american_method = american_method

    def get_params(self) -> Dict[str, Any]:
        """Returns the pricer configuration."""
        return {
            "num_steps": self.n_steps,
            "num_points": self.n_points,
            "american_method": self.american_method,
        }

    def price(self) -> Tuple[float, float]:
        """Returns `(price, 0.0)` — deterministic scheme has no MC error."""
        return (
            solve_fd(
                self.option,
                self.n_steps,
                self.n_points,
                "implicit",
                american_method=self.american_method,
            ),
            0.0,
        )
