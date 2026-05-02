"""Crank-Nicolson finite-difference Black-Scholes pricer."""

from typing import Any, Dict, Tuple

from ..options import BaseOption
from ._fd_common import solve_fd
from .base_pricer import BasePricer


class CrankNicolsonPricer(BasePricer):
    """Prices vanilla and American options via the Crank-Nicolson scheme.

    Unconditionally stable and second-order accurate in both space and time.
    The LU factor of the tridiagonal LHS is reused across every timestep, and
    boundary contributions are handled symmetrically on both LHS and RHS.
    """

    def __init__(self, option: BaseOption, num_steps: int, num_points: int):
        super().__init__(option)
        self.n_steps = num_steps
        self.n_points = num_points

    def get_params(self) -> Dict[str, Any]:
        """Returns the pricer configuration."""
        return {"num_steps": self.n_steps, "num_points": self.n_points}

    def price(self) -> Tuple[float, float]:
        """Returns `(price, 0.0)` — deterministic scheme has no MC error."""
        return solve_fd(
            self.option, self.n_steps, self.n_points, "crank_nicolson"
        ), 0.0
