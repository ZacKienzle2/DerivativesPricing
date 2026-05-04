"""Synthetic market-quote generator.

Builds a `(strikes, maturities)` call price grid from any `BaseProcess`
that exposes `analytic_european_price`. Useful for calibration testbeds:
generate quotes from a known parameter set, perturb with bid/ask noise,
recover via the calibrator, compare.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from models.processes import BaseProcess


@dataclass(frozen=True, slots=True)
class SyntheticQuotes:
    """Carrier for a synthetic call quote slab.

    Attributes:
        strikes: Strike axis, shape `(N_K,)`.
        maturities: Maturity axis (years), shape `(N_T,)`.
        prices: Mid-call prices, shape `(N_K, N_T)`.
        bids: Bid prices, shape `(N_K, N_T)`.
        asks: Ask prices, shape `(N_K, N_T)`.
        ivs: Implied vols inverted from `prices`, shape `(N_K, N_T)`.
    """

    strikes: npt.NDArray
    maturities: npt.NDArray
    prices: npt.NDArray
    bids: npt.NDArray
    asks: npt.NDArray
    ivs: npt.NDArray


def _bs_implied_vol_brent(
    target: float,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    is_call: bool,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> float:
    """Brent-bisection inversion of the Black-Scholes price for vol.

    Robust across moneyness; returns NaN if `target` is below intrinsic.
    """
    from models.pricers._analytic_kernels import bs_price_jit

    intrinsic = (
        max(s - k * np.exp(-r * t), 0.0)
        if is_call
        else max(k * np.exp(-r * t) - s, 0.0)
    )
    if target <= intrinsic + 1e-12:
        return float("nan")
    lo, hi = 1e-6, 5.0
    f_lo = bs_price_jit(s, k, t, r, q, lo, is_call) - target
    f_hi = bs_price_jit(s, k, t, r, q, hi, is_call) - target
    if f_lo * f_hi > 0:
        return float("nan")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = bs_price_jit(s, k, t, r, q, mid, is_call) - target
        if abs(f_mid) < tol:
            return mid
        if f_mid * f_lo < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def synthetic_call_quotes(
    process: BaseProcess,
    strikes: npt.NDArray,
    maturities: npt.NDArray,
    spread_bps: float = 25.0,
    r: Optional[float] = None,
    q: float = 0.0,
    seed: Optional[int] = None,
) -> SyntheticQuotes:
    """Generates a synthetic call quote slab from a given process.

    Args:
        process: Any `BaseProcess` exposing `analytic_european_price`.
        strikes: Strike axis.
        maturities: Maturity axis.
        spread_bps: Bid/ask half-spread in basis points of mid (additive
            on top of mid; lognormal-distributed noise scales the spread
            slightly across strikes).
        r: Risk-free rate (defaults to `process.r`).
        q: Dividend yield used by the IV inverter only.
        seed: Optional PCG64 seed for the spread noise.

    Returns:
        `SyntheticQuotes` with prices, bids, asks and inverted BS IVs.
    """
    if not process.supports_analytic_european:
        raise ValueError(
            f"{type(process).__name__} has no analytic_european_price hook."
        )
    rate = float(process.r if r is None else r)
    s0 = float(process.s0)
    rng = np.random.default_rng(seed)

    ks = np.asarray(strikes, dtype=np.float64)
    ts = np.asarray(maturities, dtype=np.float64)
    n_k, n_t = ks.size, ts.size
    prices = np.empty((n_k, n_t))
    for i in range(n_k):
        for j in range(n_t):
            prices[i, j] = process.analytic_european_price(
                float(ks[i]), float(ts[j]), True
            )

    half_spread = (spread_bps / 1e4) * np.maximum(prices, 1e-4)
    jitter = 1.0 + 0.1 * rng.standard_normal(prices.shape)
    half_spread = np.abs(half_spread * jitter)
    bids = np.maximum(prices - half_spread, 1e-6)
    asks = prices + half_spread

    ivs = np.empty_like(prices)
    for i in range(n_k):
        for j in range(n_t):
            ivs[i, j] = _bs_implied_vol_brent(
                float(prices[i, j]),
                s0,
                float(ks[i]),
                float(ts[j]),
                rate,
                q,
                True,
            )

    return SyntheticQuotes(ks, ts, prices, bids, asks, ivs)


def quotes_to_long_form(
    quotes: SyntheticQuotes,
) -> Tuple[
    npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
]:
    """Flattens a `(N_K, N_T)` quote slab into long-form 1D arrays.

    Returns:
        Tuple `(strikes, maturities, prices, bids, asks, ivs)`, each
        shape `(N_K * N_T,)`, ready for direct ingestion by calibrators.
    """
    n_k, n_t = quotes.prices.shape
    k_grid, t_grid = np.meshgrid(quotes.strikes, quotes.maturities, indexing="ij")
    return (
        k_grid.reshape(-1),
        t_grid.reshape(-1),
        quotes.prices.reshape(-1),
        quotes.bids.reshape(-1),
        quotes.asks.reshape(-1),
        quotes.ivs.reshape(-1),
    )
