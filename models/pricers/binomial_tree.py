"""CRR binomial-tree pricer (path-independent and Hull-White Asian)."""

from typing import Any

import numpy as np
from numba import njit, prange

from models.options import AmericanOption, AsianOption, BaseOption

from .base_pricer import BasePricer


@njit(cache=True, fastmath=True, inline="always", boundscheck=False)
def _uniform_lerp(
    a: float,
    av: np.ndarray,
    va: np.ndarray,
    n_avgs: int,
) -> float:
    """O(1) linear interp on a uniformly-spaced grid with endpoint clamping.

    The averages grid `av` is built as `av[m] = av[0] + (av[-1]-av[0])*m/(n-1)`,
    so the bin index for query `a` is recoverable arithmetically without a search.

    Args:
        a: Query point.
        av: Uniform grid of length `n_avgs`; only endpoints define the bracket.
        va: Function values at the same grid points.
        n_avgs: Length of `av` and `va`.

    Returns:
        Linear interpolant clamped to `va[0]` / `va[n_avgs-1]` outside the grid.
    """
    mn = av[0]
    mx = av[n_avgs - 1]
    if mx <= mn or a <= mn:
        return va[0]
    if a >= mx:
        return va[n_avgs - 1]
    frac = (a - mn) * (n_avgs - 1) / (mx - mn)
    idx = int(frac)
    if idx >= n_avgs - 1:
        idx = n_avgs - 2
    w = frac - idx
    return va[idx] * (1.0 - w) + va[idx + 1] * w


@njit(fastmath=True, cache=True, boundscheck=False)
def _european_american_jit(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    q: float,
    n: int,
    is_call: bool,
    is_american: bool,
) -> float:
    """CRR backward induction for path-independent (European/American) options.

    Maintains a single 1D buffer of node values rolled in place. The spot grid
    at each level is built via the recurrence `st_new = st * (d/u)`, which
    avoids `pow` calls inside the time loop.

    Args:
        s: Spot price.
        k: Strike.
        t: Time to maturity.
        r: Risk-free rate.
        sigma: Volatility.
        q: Dividend yield.
        n: Number of time-steps.
        is_call: True for call.
        is_american: True for American exercise.

    Returns:
        Option price at t=0.
    """
    dt = t / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    df = np.exp(-r * dt)

    values = np.empty(n + 1)
    st_top = s * u**n
    ratio = d / u
    st = st_top
    for j in range(n + 1):
        v = st - k if is_call else k - st
        values[j] = v if v > 0.0 else 0.0
        st *= ratio

    for i in range(n - 1, -1, -1):
        st_top_i = s * u**i
        st = st_top_i
        for j in range(i + 1):
            cont = (p * values[j] + (1.0 - p) * values[j + 1]) * df
            if is_american:
                ex = st - k if is_call else k - st
                if ex > cont:
                    cont = ex
                st *= ratio
            values[j] = cont
    return values[0]


@njit(fastmath=True, parallel=True, cache=True, boundscheck=False)
def _asian_hull_white_jit(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    q: float,
    n: int,
    is_call: bool,
    n_avgs: int,
) -> float:
    """Hull-White interpolation for Asian options on a CRR tree.

    At each (i, j) node a fixed-size grid of `n_avgs` running averages spans
    the analytically reachable [min, max] interval. Values at coarser steps
    are produced via linear interpolation against the next-step grids. Memory
    is bounded by `O(n^2 * n_avgs)`; runtime by `O(n^3 * n_avgs)`. The
    backward sweep is parallel across nodes within each time level.

    Args:
        s, k, t, r, sigma, q: Standard option parameters.
        n: Number of CRR steps.
        is_call: True for call.
        n_avgs: Average-grid resolution per node.

    Returns:
        Option price at t=0.
    """
    dt = t / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    df = np.exp(-r * dt)

    st = np.empty((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            st[i, j] = s * (u**j) * (d ** (i - j))

    min_sum = np.zeros((n + 1, n + 1))
    max_sum = np.zeros((n + 1, n + 1))
    min_sum[0, 0] = s
    max_sum[0, 0] = s
    for i in range(1, n + 1):
        for j in range(i + 1):
            price = st[i, j]
            if 0 < j < i:
                min_sum[i, j] = min(min_sum[i - 1, j - 1], min_sum[i - 1, j]) + price
                max_sum[i, j] = max(max_sum[i - 1, j - 1], max_sum[i - 1, j]) + price
            elif j == 0:
                min_sum[i, j] = min_sum[i - 1, 0] + price
                max_sum[i, j] = max_sum[i - 1, 0] + price
            else:
                min_sum[i, j] = min_sum[i - 1, j - 1] + price
                max_sum[i, j] = max_sum[i - 1, j - 1] + price

    averages = np.zeros((n + 1, n + 1, n_avgs))
    denom = float(n_avgs - 1) if n_avgs > 1 else 1.0
    for i in range(n + 1):
        scale = 1.0 / (i + 1)
        for j in range(i + 1):
            mn = min_sum[i, j] * scale
            mx = max_sum[i, j] * scale
            for m in range(n_avgs):
                averages[i, j, m] = mn + (mx - mn) * (m / denom)

    values = np.zeros((n + 1, n + 1, n_avgs))
    for j in range(n + 1):
        for m in range(n_avgs):
            a = averages[n, j, m]
            v = a - k if is_call else k - a
            values[n, j, m] = v if v > 0.0 else 0.0

    for i in range(n - 1, -1, -1):
        for j in prange(i + 1):
            up_av = averages[i + 1, j + 1]
            up_va = values[i + 1, j + 1]
            dn_av = averages[i + 1, j]
            dn_va = values[i + 1, j]
            sup = st[i + 1, j + 1]
            sdn = st[i + 1, j]
            inv_next = 1.0 / (i + 2)
            steps_done = float(i + 1)
            for m in range(n_avgs):
                a = averages[i, j, m]
                a_up = (a * steps_done + sup) * inv_next
                a_dn = (a * steps_done + sdn) * inv_next
                v_up = _uniform_lerp(a_up, up_av, up_va, n_avgs)
                v_dn = _uniform_lerp(a_dn, dn_av, dn_va, n_avgs)
                values[i, j, m] = (p * v_up + (1.0 - p) * v_dn) * df

    return values[0, 0, 0]


class BinomialTreePricer(BasePricer):
    """Cox-Ross-Rubinstein binomial-tree pricer.

    Dispatches to a path-independent solver for European/American options
    and to a Hull-White interpolation solver for Asian options.
    """

    def __init__(
        self,
        option: BaseOption,
        num_steps: int,
        num_avgs: int = 50,
        richardson: bool = False,
    ):
        super().__init__(option)
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("Number of steps must be a positive integer.")
        if not isinstance(num_avgs, int) or num_avgs < 2:
            raise ValueError("num_avgs must be an integer >= 2.")
        self.num_steps = num_steps
        self.num_avgs = num_avgs
        self.richardson = richardson

    def get_params(self) -> dict[str, Any]:
        """Returns the configuration of this pricer."""
        return {
            "num_steps": self.num_steps,
            "num_avgs": self.num_avgs,
            "richardson": self.richardson,
        }

    def _price_at(self, n_steps: int) -> float:
        opt = self.option
        is_call = opt.option_type == "call"
        if isinstance(opt, AsianOption):
            return float(
                _asian_hull_white_jit(
                    opt.S, opt.K, opt.T, opt.r, opt.sigma, opt.q,
                    n_steps, is_call, self.num_avgs,
                )
            )
        return float(
            _european_american_jit(
                opt.S, opt.K, opt.T, opt.r, opt.sigma, opt.q,
                n_steps, is_call, isinstance(opt, AmericanOption),
            )
        )

    def price(self) -> tuple[float, float]:
        """Returns `(price, 0.0)`.

        With `richardson=True`, prices `(n)` and `(n+1)` step trees and
        returns `0.5 * (P_n + P_{n+1})` so the leading O(1/n) error term
        cancels — quadratic CRR convergence at 2x compute cost.
        """
        v_n = self._price_at(self.num_steps)
        if not self.richardson:
            return v_n, 0.0
        v_np1 = self._price_at(self.num_steps + 1)
        return 0.5 * (v_n + v_np1), 0.0
