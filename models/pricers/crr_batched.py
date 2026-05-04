"""Batched CRR European pricer over a strike vector.

Up/down factors and risk-neutral probability are computed once across all
strikes; the backward induction runs in parallel across strikes via
`prange`. Per-strike work is `O(n_steps^2)`, but the shared spot grid and
shared transition kernel cut redundant arithmetic.
"""

import math

import numpy as np
import numpy.typing as npt
from numba import njit, prange


@njit(fastmath=True, parallel=True, cache=True, boundscheck=False)
def crr_batched_jit(
    s: float,
    strikes: npt.NDArray[np.float64],
    t: float,
    r: float,
    q: float,
    sigma: float,
    n: int,
    is_call: bool,
    is_american: bool,
) -> npt.NDArray[np.float64]:
    """Vectorised CRR backward induction across a strike vector.

    Args:
        s: Spot.
        strikes: Strike vector, shape `(M,)`.
        t, r, q, sigma: Standard parameters.
        n: Number of CRR steps.
        is_call: True for call.
        is_american: True for American exercise.

    Returns:
        Per-strike option prices, shape `(M,)`.
    """
    m = strikes.size
    dt = t / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    df = math.exp(-r * dt)
    one_minus_p = 1.0 - p
    ratio = d / u

    out = np.empty(m)
    for s_idx in prange(m):
        k = strikes[s_idx]
        values = np.empty(n + 1)
        st_top = s * (u ** n)
        st = st_top
        for j in range(n + 1):
            v = st - k if is_call else k - st
            values[j] = v if v > 0.0 else 0.0
            st *= ratio
        for i in range(n - 1, -1, -1):
            st_top_i = s * (u ** i)
            st = st_top_i
            for j in range(i + 1):
                cont = (p * values[j] + one_minus_p * values[j + 1]) * df
                if is_american:
                    ex = st - k if is_call else k - st
                    if ex > cont:
                        cont = ex
                    st *= ratio
                values[j] = cont
        out[s_idx] = values[0]
    return out


def crr_strike_grid(
    s: float,
    strikes: npt.NDArray[np.float64],
    t: float,
    r: float,
    q: float,
    sigma: float,
    n: int,
    is_call: bool = True,
    is_american: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Returns `(strikes, prices)` from a single batched CRR sweep.

    Args:
        s: Spot.
        strikes: Strike vector. Must be 1D, contiguous, float64.
        t, r, q, sigma: Standard parameters.
        n: CRR steps.
        is_call: True for call.
        is_american: True for American.

    Returns:
        Tuple of input strikes and corresponding option prices.
    """
    strikes = np.ascontiguousarray(np.asarray(strikes, dtype=np.float64))
    prices = crr_batched_jit(s, strikes, t, r, q, sigma, n, is_call, is_american)
    return strikes, prices
