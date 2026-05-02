"""Crank-Nicolson finite-difference Black-Scholes pricer."""

from typing import Any, Dict, Tuple

from ..options import BaseOption
from ._fd_common import solve_fd
from .base_pricer import BasePricer


class CrankNicolsonPricer(BasePricer):
    """Prices vanilla and American options via Crank-Nicolson with Rannacher.

    Unconditionally stable and second-order accurate in both space and time.
    By default the first two timesteps run fully implicit (Rannacher
    smoothing) to damp payoff-kink oscillation near the strike. American
    exercise can be handled via operator-splitting max-projection
    (`american_method='project'`) or LCP-correct projected SOR
    (`american_method='psor'`).
    """

    def __init__(
        self,
        option: BaseOption,
        num_steps: int,
        num_points: int,
        american_method: str = "project",
        rannacher_steps: int = 2,
    ):
        super().__init__(option)
        self.n_steps = num_steps
        self.n_points = num_points
        self.american_method = american_method
        self.rannacher_steps = rannacher_steps

    def get_params(self) -> Dict[str, Any]:
        """Returns the pricer configuration."""
        return {
            "num_steps": self.n_steps,
            "num_points": self.n_points,
            "american_method": self.american_method,
            "rannacher_steps": self.rannacher_steps,
        }

    def price(self) -> Tuple[float, float]:
        """Returns `(price, 0.0)` — deterministic scheme has no MC error."""
        return (
            solve_fd(
                self.option,
                self.n_steps,
                self.n_points,
                "crank_nicolson",
                american_method=self.american_method,
                rannacher_steps=self.rannacher_steps,
            ),
            0.0,
        )
