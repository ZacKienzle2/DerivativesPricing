"""Longstaff-Schwartz American-option pricer with stabilised regression."""

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import njit

from ..options import BaseOption
from ..simulation import generate_paths_jit
from .base_pricer import BasePricer


@njit(fastmath=True, cache=True, boundscheck=False)
def _backward_induction_jit(
    paths: npt.NDArray[np.float64],
    k: float,
    r: float,
    dt: float,
    is_call: bool,
    poly_degree: int,
) -> npt.NDArray[np.float64]:
    """Performs backward induction for the Longstaff-Schwartz algorithm.

    Spot prices at each step are mapped to the Chebyshev domain `[-1, 1]`
    via an affine rescaling fitted to the per-step ITM range. The
    Chebyshev basis is near-orthogonal under uniform spot densities, so
    the per-step Gram matrix is well-conditioned and a normal-equations
    solve via `np.linalg.solve` replaces the SVD-backed `np.linalg.lstsq`
    used previously. Basis polynomials and Gram contributions are
    accumulated in one pass per sample to avoid materialising a Vandermonde
    matrix, dropping per-step work from `O(n_itm * n_basis^2 + n_basis^3)`
    SVD to `O(n_itm * n_basis^2)` with a tight inner loop.

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
    df = math.exp(-r * dt)
    n_basis = poly_degree + 1

    if is_call:
        cash_flows = np.maximum(paths[:, -1] - k, 0.0)
    else:
        cash_flows = np.maximum(k - paths[:, -1], 0.0)

    basis = np.empty(n_basis)
    gram = np.empty((n_basis, n_basis))
    rhs = np.empty(n_basis)

    for t in range(num_steps - 1, 0, -1):
        cash_flows *= df
        s_t = paths[:, t]
        if is_call:
            exercise = np.maximum(s_t - k, 0.0)
        else:
            exercise = np.maximum(k - s_t, 0.0)

        itm_idx = np.where(exercise > 0.0)[0]
        n_itm = itm_idx.size
        if n_itm <= poly_degree:
            continue

        x_min = s_t[itm_idx[0]]
        x_max = x_min
        for j in range(1, n_itm):
            xv = s_t[itm_idx[j]]
            if xv < x_min:
                x_min = xv
            elif xv > x_max:
                x_max = xv
        span = x_max - x_min
        if span <= 0.0:
            continue
        scale = 2.0 / span
        offset = -1.0 - scale * x_min

        for a in range(n_basis):
            rhs[a] = 0.0
            for b in range(n_basis):
                gram[a, b] = 0.0

        for j in range(n_itm):
            idx = itm_idx[j]
            ui = scale * s_t[idx] + offset
            basis[0] = 1.0
            if n_basis > 1:
                basis[1] = ui
            for c in range(2, n_basis):
                basis[c] = 2.0 * ui * basis[c - 1] - basis[c - 2]
            yi = cash_flows[idx]
            for a in range(n_basis):
                rhs[a] += basis[a] * yi
                ba = basis[a]
                for b in range(a, n_basis):
                    gram[a, b] += ba * basis[b]

        for a in range(n_basis):
            for b in range(a + 1, n_basis):
                gram[b, a] = gram[a, b]

        coeffs = np.linalg.solve(gram, rhs)

        for j in range(n_itm):
            idx = itm_idx[j]
            ui = scale * s_t[idx] + offset
            t0 = 1.0
            cv = coeffs[0]
            if n_basis > 1:
                t1 = ui
                cv += coeffs[1] * t1
                for c in range(2, n_basis):
                    tk = 2.0 * ui * t1 - t0
                    cv += coeffs[c] * tk
                    t0 = t1
                    t1 = tk
            ex_j = exercise[idx]
            if ex_j > cv:
                cash_flows[idx] = ex_j

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
        z_matrix: npt.NDArray[np.float64] | None = None,
        seed: int | None = None,
    ):
        super().__init__(option)
        self.num_sims = num_sims
        self.num_steps = num_steps
        self.poly_degree = poly_degree
        self.use_crn = use_crn
        self.z_matrix: npt.NDArray[np.float64] | None = z_matrix
        self.convergence_data: npt.NDArray[np.float64] | None = None
        self._rng = np.random.default_rng(seed)

    def get_params(self) -> dict[str, Any]:
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

    def price(self) -> tuple[float, float]:
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
