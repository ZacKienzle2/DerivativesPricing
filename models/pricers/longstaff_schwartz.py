"""Longstaff-Schwartz American-option pricer with stabilised regression."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from numba import njit

from ..options import BaseOption
from ..simulation import generate_paths_jit
from .base_pricer import BasePricer


@njit(fastmath=True, cache=True)
def _backward_induction_jit(
    paths: npt.NDArray[np.float64],
    k: float,
    r: float,
    dt: float,
    is_call: bool,
    poly_degree: int,
) -> npt.NDArray[np.float64]:
    """Performs backward induction for the Longstaff-Schwartz algorithm.

    Asset prices are normalised by their per-step ITM maximum before fitting
    the polynomial basis. This drops the regression-matrix condition number
    by orders of magnitude and lets the linear solve stay stable for
    `poly_degree >= 3`. Continuation values are evaluated via Horner's rule.

    Args:
        paths: Simulated asset price paths, shape (num_sims, num_steps + 1).
        k: Strike price.
        r: Risk-free rate.
        dt: Time-step size.
        is_call: True for call, False for put.
        poly_degree: Polynomial regression degree.

    Returns:
        Cash-flow vector at the first exercise step, shape (num_sims,).
    """
    num_steps = paths.shape[1] - 1
    df = np.exp(-r * dt)

    if is_call:
        cash_flows = np.maximum(paths[:, -1] - k, 0.0)
    else:
        cash_flows = np.maximum(k - paths[:, -1], 0.0)

    n_basis = poly_degree + 1

    for t in range(num_steps - 1, 0, -1):
        cash_flows *= df
        s_t = paths[:, t]
        if is_call:
            exercise = np.maximum(s_t - k, 0.0)
        else:
            exercise = np.maximum(k - s_t, 0.0)

        itm_idx = np.where(exercise > 0.0)[0]
        if itm_idx.size <= poly_degree:
            continue

        x = s_t[itm_idx]
        y = cash_flows[itm_idx]
        ex = exercise[itm_idx]

        x_max = np.max(np.abs(x))
        if x_max == 0.0:
            x_max = 1.0
        x_n = x / x_max

        vander = np.empty((x_n.size, n_basis))
        for c in range(n_basis):
            vander[:, c] = x_n ** (n_basis - 1 - c)

        coeffs = np.linalg.lstsq(vander, y, rcond=-1.0)[0]

        continuation = np.full(x_n.size, coeffs[0])
        for c in range(1, n_basis):
            continuation = continuation * x_n + coeffs[c]

        for j in range(itm_idx.size):
            if ex[j] > continuation[j]:
                cash_flows[itm_idx[j]] = ex[j]

    return cash_flows


class LongstaffSchwartzPricer(BasePricer):
    """Prices American options via least-squares Monte Carlo (LSM)."""

    def __init__(
        self,
        option: BaseOption,
        num_sims: int,
        num_steps: int,
        poly_degree: int = 3,
        use_crn: bool = True,
        z_matrix: Optional[npt.NDArray[np.float64]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(option)
        self.num_sims = num_sims
        self.num_steps = num_steps
        self.poly_degree = poly_degree
        self.use_crn = use_crn
        self.z_matrix: Optional[npt.NDArray[np.float64]] = z_matrix
        self.convergence_data: Optional[npt.NDArray[np.float64]] = None
        self._rng = np.random.default_rng(seed)

    def get_params(self) -> Dict[str, Any]:
        """Returns the configuration for this pricer instance."""
        return {
            "num_sims": self.num_sims,
            "num_steps": self.num_steps,
            "poly_degree": self.poly_degree,
            "use_crn": self.use_crn,
            "z_matrix": self.z_matrix,
        }

    def _generate_z_matrix(self) -> npt.NDArray[np.float64]:
        """Draws standard normals for the simulation grid via PCG64."""
        return self._rng.standard_normal((self.num_sims, self.num_steps))

    def price(self) -> Tuple[float, float]:
        """Prices the option, returning `(price, standard_error)`."""
        if self.z_matrix is None:
            self.z_matrix = self._generate_z_matrix()

        opt = self.option
        dt = opt.T / self.num_steps
        paths = generate_paths_jit(
            opt.S, opt.T, opt.r, opt.q, opt.sigma, self.num_steps, self.z_matrix
        )
        cash_flows_t1 = _backward_induction_jit(
            paths=paths,
            k=opt.K,
            r=opt.r,
            dt=dt,
            is_call=(opt.option_type == "call"),
            poly_degree=self.poly_degree,
        )
        discounted_cfs = cash_flows_t1 * np.exp(-opt.r * dt)
        price = np.mean(discounted_cfs)
        std_err = np.std(discounted_cfs) / np.sqrt(self.num_sims)
        self.convergence_data = np.cumsum(discounted_cfs) / (
            np.arange(self.num_sims) + 1
        )
        return float(price), float(std_err)
