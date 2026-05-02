"""Monte Carlo pricer with QMC, antithetic, control variate and PCG64 RNG."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from scipy.special import ndtri
from scipy.stats import qmc

from ..options import (
    AsianOption,
    BarrierOption,
    BaseOption,
    BasketOption,
    VanillaOption,
)
from ..simulation import generate_final_prices_jit, generate_paths_jit
from ._analytic_kernels import discrete_geom_asian_price_jit
from .base_pricer import BasePricer

_BARRIER_CODES: Dict[str, int] = {
    "up-and-in": 0,
    "up-and-out": 1,
    "down-and-in": 2,
    "down-and-out": 3,
}


def _build_bb_tree(n: int) -> Tuple[
    npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]
]:
    """Builds Brownian-Bridge bisection indices via BFS.

    Args:
        n: Number of timesteps (path increments).

    Returns:
        Tuple `(left, right, bridge)` of int64 arrays, length `n`. Slot 0 is
        the endpoint (`bridge[0] = n`); slots `1..n-1` are interior bridge
        points produced in BFS order so the bridge formula always reads from
        already-filled positions.
    """
    left = np.zeros(n, dtype=np.int64)
    right = np.zeros(n, dtype=np.int64)
    bridge = np.zeros(n, dtype=np.int64)
    bridge[0] = n
    left[0] = 0
    right[0] = n
    queue: List[Tuple[int, int]] = [(0, n)]
    head = 0
    k = 1
    while head < len(queue) and k < n:
        l, r = queue[head]
        head += 1
        b = (l + r) // 2
        if l < b < r:
            left[k] = l
            right[k] = r
            bridge[k] = b
            queue.append((l, b))
            queue.append((b, r))
            k += 1
    return left, right, bridge


@njit(fastmath=True, parallel=True, cache=True, boundscheck=False)
def _brownian_bridge_jit(
    z: npt.NDArray[np.float64],
    dt: float,
    left: npt.NDArray[np.int64],
    right: npt.NDArray[np.int64],
    bridge: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """Maps i.i.d. normals to BB-ordered increments.

    Recursive midpoint bisection of the Brownian path so the first Sobol
    dimensions set the path's terminal value, the second its midpoint, and
    so on. The resulting increment matrix has the same joint distribution
    as the input but with variance front-loaded onto early QMC dimensions.

    Args:
        z: Standard normals, shape `(num_sims, num_steps)`.
        dt: Per-step time increment.
        left, right, bridge: Output of `_build_bb_tree`.

    Returns:
        Increment matrix `dW / sqrt(dt)`, same shape as `z`, ready to feed
        into the standard cumsum-based GBM kernel.
    """
    num_sims = z.shape[0]
    n = z.shape[1]
    sqrt_t = np.sqrt(n * dt)
    inv_sqrt_dt = 1.0 / np.sqrt(dt)
    out = np.empty_like(z)

    for s in prange(num_sims):
        w = np.empty(n + 1)
        w[0] = 0.0
        w[n] = sqrt_t * z[s, 0]
        for k in range(1, n):
            l = left[k]
            r = right[k]
            b = bridge[k]
            span = r - l
            a_l = (r - b) / span
            a_r = (b - l) / span
            sigma_b = np.sqrt((b - l) * (r - b) * dt / span)
            w[b] = a_l * w[l] + a_r * w[r] + sigma_b * z[s, k]
        for j in range(n):
            out[s, j] = (w[j + 1] - w[j]) * inv_sqrt_dt
    return out


@njit(fastmath=True, parallel=True, cache=True, boundscheck=False)
def _calculate_barrier_payoffs_jit(
    paths: npt.NDArray[np.float64],
    strike: float,
    barrier: float,
    barrier_type_code: int,
    is_call: bool,
) -> npt.NDArray[np.float64]:
    """JIT barrier payoff with early-exit knock detection.

    Args:
        paths: Asset price paths, shape `(num_sims, num_steps + 1)`.
        strike: Strike price.
        barrier: Knock barrier level.
        barrier_type_code: 0=up-and-in, 1=up-and-out, 2=down-and-in, 3=down-and-out.
        is_call: True for call.

    Returns:
        Per-path undiscounted payoffs.
    """
    num_sims = paths.shape[0]
    num_steps = paths.shape[1]
    is_up = barrier_type_code <= 1
    is_in = (barrier_type_code & 1) == 0
    payoffs = np.zeros(num_sims)

    for i in prange(num_sims):
        crossed = False
        for j in range(1, num_steps):
            if is_up:
                if paths[i, j] >= barrier:
                    crossed = True
                    break
            else:
                if paths[i, j] <= barrier:
                    crossed = True
                    break
        if crossed == is_in:
            final = paths[i, num_steps - 1]
            v = final - strike if is_call else strike - final
            if v > 0.0:
                payoffs[i] = v
    return payoffs


@njit(fastmath=True, parallel=True, cache=True, boundscheck=False)
def _generate_correlated_paths_jit(
    initial_prices: npt.NDArray[np.float64],
    volatilities: npt.NDArray[np.float64],
    r: float,
    t: float,
    num_steps: int,
    chol: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """JIT correlated multi-asset GBM paths.

    Hoists `dt`, drift and `vol*sqrt(dt)` out of the time loop and applies
    the Cholesky factor on the fly per (sim, step).

    Args:
        initial_prices: Per-asset spot, shape `(num_assets,)`.
        volatilities: Per-asset volatility, shape `(num_assets,)`.
        r: Risk-free rate.
        t: Maturity.
        num_steps: Discretisation steps.
        chol: Cholesky factor of correlation matrix.
        z: I.I.D. normals, shape `(num_assets, num_sims, num_steps)`.

    Returns:
        Paths shape `(num_assets, num_sims, num_steps + 1)`.
    """
    num_assets = initial_prices.shape[0]
    num_sims = z.shape[1]
    dt = t / num_steps
    sqrt_dt = np.sqrt(dt)
    drift = (r - 0.5 * volatilities * volatilities) * dt
    vol_sqrt_dt = volatilities * sqrt_dt
    log_init = np.log(initial_prices)

    paths = np.empty((num_assets, num_sims, num_steps + 1))
    for a in range(num_assets):
        s0 = initial_prices[a]
        for s in prange(num_sims):
            paths[a, s, 0] = s0

    for s in prange(num_sims):
        log_state = log_init.copy()
        for step in range(num_steps):
            for a in range(num_assets):
                inc = 0.0
                for b in range(num_assets):
                    inc += chol[a, b] * z[b, s, step]
                log_state[a] += drift[a] + vol_sqrt_dt[a] * inc
                paths[a, s, step + 1] = np.exp(log_state[a])
    return paths


@njit(fastmath=True, parallel=True, cache=True, boundscheck=False)
def _asian_dual_average_jit(
    paths: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Computes per-path arithmetic and geometric sample-mean prices.

    Single pass over `paths[:, 1:]`. Geometric mean accumulates in log-space.

    Args:
        paths: Asset paths shape `(num_sims, num_steps + 1)`.

    Returns:
        `(arith_avg, geom_avg)` each of shape `(num_sims,)`.
    """
    num_sims = paths.shape[0]
    num_samples = paths.shape[1] - 1
    inv_n = 1.0 / num_samples
    arith = np.empty(num_sims)
    geom = np.empty(num_sims)
    for i in prange(num_sims):
        s_sum = 0.0
        l_sum = 0.0
        for j in range(1, num_samples + 1):
            s = paths[i, j]
            s_sum += s
            l_sum += np.log(s)
        arith[i] = s_sum * inv_n
        geom[i] = np.exp(l_sum * inv_n)
    return arith, geom


def _ols_beta(arith: npt.NDArray[np.float64], geom: npt.NDArray[np.float64]) -> float:
    """OLS regression slope of `arith` on `geom`."""
    g_mean = geom.mean()
    a_mean = arith.mean()
    cov = ((arith - a_mean) * (geom - g_mean)).mean()
    var = ((geom - g_mean) ** 2).mean()
    return cov / var if var > 0.0 else 0.0


class MonteCarloPricer(BasePricer):
    """Monte Carlo pricer.

    Variance-reduction tokens accepted in `variance_reduction`:
        * `"antithetic"`     — pair every draw with its negation.
        * `"sobol"`          — replace pseudo-random uniforms with scrambled Sobol.
        * `"control_variate"`— for arithmetic Asian only; uses discrete
          geometric Asian closed form as the control.

    Driver RNG is a PCG64 `Generator` seeded via the `seed` kwarg.
    """

    def __init__(
        self,
        option: BaseOption,
        num_sims: int,
        num_steps: int,
        variance_reduction: Optional[List[str]] = None,
        use_crn: bool = True,
        seed: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(option)
        vr = list(variance_reduction or [])
        if "antithetic" in vr and num_sims % 2 != 0:
            num_sims += 1
        self.num_sims = num_sims
        self.num_steps = num_steps
        self.variance_reduction = vr
        self.use_crn = use_crn
        self.z_matrix: Optional[npt.NDArray[np.float64]] = None
        self.convergence_data: Optional[npt.NDArray[np.float64]] = None
        self.discounted_payoffs: Optional[npt.NDArray[np.float64]] = None
        self._rng = np.random.default_rng(seed)

        self._payoff_handlers: Dict[type, Callable[[], npt.NDArray[np.float64]]] = {
            VanillaOption: self._payoff_vanilla,
            BasketOption: self._payoff_basket,
            BarrierOption: self._payoff_barrier,
            AsianOption: self._payoff_asian,
        }

    def get_params(self) -> Dict[str, Any]:
        """Returns the simulation configuration."""
        return {
            "num_sims": self.num_sims,
            "num_steps": self.num_steps,
            "variance_reduction": self.variance_reduction,
            "use_crn": self.use_crn,
        }

    def _generate_z_matrix(self) -> npt.NDArray[np.float64]:
        """Builds the `(num_sims, num_steps)` standard-normal driver matrix.

        Direct standard-normal draws via PCG64 unless Sobol is requested,
        in which case scrambled Sobol uniforms are mapped through `ndtri`
        (faster than `scipy.stats.norm.ppf`).
        """
        use_antithetic = "antithetic" in self.variance_reduction
        num_base_sims = self.num_sims // 2 if use_antithetic else self.num_sims

        if "sobol" in self.variance_reduction:
            sampler = qmc.Sobol(
                d=self.num_steps, scramble=True, seed=self._rng
            )
            m = int(np.ceil(np.log2(max(num_base_sims, 1))))
            samples = sampler.random_base2(m=m)[:num_base_sims]
            z = ndtri(samples)
        else:
            z = self._rng.standard_normal((num_base_sims, self.num_steps))

        if use_antithetic:
            z = np.concatenate([z, -z], axis=0)

        np.nan_to_num(z, copy=False, posinf=0.0, neginf=0.0)

        if "brownian_bridge" in self.variance_reduction and self.num_steps > 1:
            dt = self.option.T / self.num_steps
            left, right, bridge = _build_bb_tree(self.num_steps)
            z = _brownian_bridge_jit(z, dt, left, right, bridge)
        return z

    def _ensure_z(self) -> npt.NDArray[np.float64]:
        if self.z_matrix is None:
            self.z_matrix = self._generate_z_matrix()
        return self.z_matrix

    def _generate_paths(self) -> npt.NDArray[np.float64]:
        """Returns simulated paths for path-dependent payoffs."""
        if isinstance(self.option, BasketOption):
            return self._generate_correlated_paths()
        z = self._ensure_z()
        opt = self.option
        return generate_paths_jit(
            opt.S, opt.T, opt.r, opt.q, opt.sigma, self.num_steps, z
        )

    def _generate_correlated_paths(self) -> npt.NDArray[np.float64]:
        """Returns multi-asset correlated paths for a basket option."""
        opt = self.option
        if not isinstance(opt, BasketOption):
            raise TypeError("For BasketOption only.")
        z = self._rng.standard_normal(
            (opt.num_assets, self.num_sims, self.num_steps)
        )
        chol = np.linalg.cholesky(np.asarray(opt.corr_matrix, dtype=np.float64))
        return _generate_correlated_paths_jit(
            np.asarray(opt.initial_prices, dtype=np.float64),
            np.asarray(opt.volatilities, dtype=np.float64),
            opt.r,
            opt.T,
            self.num_steps,
            chol,
            z,
        )

    def _payoff_vanilla(self) -> npt.NDArray[np.float64]:
        opt = self.option
        is_call = opt.option_type == "call"
        z = self._ensure_z()
        final = generate_final_prices_jit(
            opt.S, opt.T, opt.r, opt.q, opt.sigma, self.num_steps, z
        )
        return (
            np.maximum(final - opt.K, 0.0)
            if is_call
            else np.maximum(opt.K - final, 0.0)
        )

    def _payoff_basket(self) -> npt.NDArray[np.float64]:
        opt = self.option
        is_call = opt.option_type == "call"
        paths = self._generate_paths()
        weights = np.asarray(opt.weights, dtype=np.float64)
        basket_values = weights @ paths[:, :, -1]
        return (
            np.maximum(basket_values - opt.K, 0.0)
            if is_call
            else np.maximum(opt.K - basket_values, 0.0)
        )

    def _payoff_barrier(self) -> npt.NDArray[np.float64]:
        opt = self.option
        is_call = opt.option_type == "call"
        paths = self._generate_paths()
        return _calculate_barrier_payoffs_jit(
            paths,
            opt.K,
            opt.barrier_level,
            _BARRIER_CODES[opt.barrier_type],
            is_call,
        )

    def _payoff_asian(self) -> npt.NDArray[np.float64]:
        opt = self.option
        is_call = opt.option_type == "call"
        paths = self._generate_paths()

        if opt.avg_type == "geometric":
            sample = paths[:, 1:]
            geom = np.exp(np.log(sample).mean(axis=1))
            return (
                np.maximum(geom - opt.K, 0.0)
                if is_call
                else np.maximum(opt.K - geom, 0.0)
            )

        arith, geom = _asian_dual_average_jit(paths)
        arith_payoff = (
            np.maximum(arith - opt.K, 0.0)
            if is_call
            else np.maximum(opt.K - arith, 0.0)
        )

        if "control_variate" not in self.variance_reduction:
            return arith_payoff

        geom_payoff = (
            np.maximum(geom - opt.K, 0.0)
            if is_call
            else np.maximum(opt.K - geom, 0.0)
        )
        df = np.exp(-opt.r * opt.T)
        analytic_geom = discrete_geom_asian_price_jit(
            opt.S, opt.K, opt.T, opt.r, opt.q, opt.sigma, self.num_steps, is_call
        )
        beta = _ols_beta(arith_payoff, geom_payoff)
        return arith_payoff - beta * (geom_payoff - analytic_geom / df)

    def price(self) -> Tuple[float, float]:
        """Returns `(price, standard_error)`."""
        opt = self.option
        handler = self._payoff_handlers.get(type(opt))
        if handler is None:
            for cls, fn in self._payoff_handlers.items():
                if isinstance(opt, cls):
                    handler = fn
                    break
        if handler is None:
            raise TypeError(f"MC pricer not implemented for {type(opt).__name__}")

        payoffs = handler()
        df = np.exp(-opt.r * opt.T)
        discounted = payoffs * df
        self.discounted_payoffs = discounted
        price = float(discounted.mean())
        std_err = float(discounted.std() / np.sqrt(self.num_sims))
        self.convergence_data = (
            np.cumsum(discounted) / (np.arange(self.num_sims) + 1)
        ).astype(np.float64)
        return price, std_err
