"""Greek surface, sensitivity strip and portfolio aggregation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._cache import cached
from ._request import PricingRequest
from .registry import GREEK_ENGINE

_GREEK_KEYS: tuple[str, ...] = ("delta", "gamma", "vega", "theta", "rho")
_SHARED_Z_PRICERS: frozenset[str] = frozenset({"Monte Carlo", "Longstaff-Schwartz"})


@cached()
def get_greek_data(
    inputs: dict[str, Any], s_range: np.ndarray
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Returns per-strike Greek series for both call and put flavours.

    Sequential outer loop. JIT-internal `prange` already saturates cores
    inside the parallel pricer kernels, so an additional Python-level
    `ThreadPoolExecutor` would be both redundant and unsafe under numba's
    default workqueue threading layer.
    """
    request = PricingRequest.from_dict(inputs)
    method = request.greek_method

    z_matrix = None
    if request.pricer_type in _SHARED_Z_PRICERS:
        num_sims = request.model_params.get("num_sims", 10000)
        num_steps = request.model_params.get("num_steps", 100)
        z_matrix = np.random.standard_normal((num_sims, num_steps))

    call_request = request.with_overrides(option_flavour="call")
    put_request = request.with_overrides(option_flavour="put")

    call_greeks: dict[str, list[float]] = {k: [] for k in _GREEK_KEYS}
    put_greeks: dict[str, list[float]] = {k: [] for k in _GREEK_KEYS}
    for s_val in s_range:
        s_float = float(s_val)
        _, pricer_c = call_request.with_overrides(s=s_float).build()
        _, pricer_p = put_request.with_overrides(s=s_float).build()
        if z_matrix is not None:
            if hasattr(pricer_c, "z_matrix"):
                pricer_c.z_matrix = z_matrix
            if hasattr(pricer_p, "z_matrix"):
                pricer_p.z_matrix = z_matrix
        greeks_c = GREEK_ENGINE.get_calculator(pricer_c, method).calculate()
        greeks_p = GREEK_ENGINE.get_calculator(pricer_p, method).calculate()
        for key in _GREEK_KEYS:
            call_greeks[key].append(greeks_c.get(key, np.nan))
            put_greeks[key].append(greeks_p.get(key, np.nan))
    return call_greeks, put_greeks


_PORTFOLIO_KEYS: tuple[str, ...] = (
    "price", "delta", "gamma", "vega", "theta", "rho",
)
_GREEK_CHANNEL: dict[str, int] = {key: idx for idx, key in enumerate(_PORTFOLIO_KEYS)}


@cached()
def compute_greek_surface(
    s_range: np.ndarray,
    t_range: np.ndarray,
    k: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
    greek: str,
) -> np.ndarray:
    """Computes a 2D `(S, T)` surface for a single Black-Scholes Greek.

    Dispatches a single vectorised parallel JIT call instead of a Python
    double loop over scalar `bs_full_greeks_jit` evaluations.
    """
    from models.pricers._analytic_kernels import bs_greek_surface_jit

    if greek not in _GREEK_CHANNEL:
        raise ValueError(f"Unknown greek: {greek!r}")
    s_arr = np.ascontiguousarray(s_range, dtype=np.float64)
    t_arr = np.ascontiguousarray(t_range, dtype=np.float64)
    out = np.empty((6, s_arr.size, t_arr.size), dtype=np.float64)
    bs_greek_surface_jit(s_arr, t_arr, k, r, q, sigma, is_call, out)
    return out[_GREEK_CHANNEL[greek]]


@cached()
def aggregate_portfolio_greeks(
    positions: list[dict[str, Any]],
) -> dict[str, float]:
    """Sums Black-Scholes Greeks across a list of vanilla option positions.

    Marshals positions into contiguous float arrays and dispatches a single
    parallel reduction kernel.
    """
    from models.pricers._analytic_kernels import bs_portfolio_aggregate_jit

    n = len(positions)
    if n == 0:
        return {key: 0.0 for key in _PORTFOLIO_KEYS}

    s_arr = np.empty(n, dtype=np.float64)
    k_arr = np.empty(n, dtype=np.float64)
    t_arr = np.empty(n, dtype=np.float64)
    r_arr = np.empty(n, dtype=np.float64)
    q_arr = np.empty(n, dtype=np.float64)
    sigma_arr = np.empty(n, dtype=np.float64)
    is_call_arr = np.empty(n, dtype=np.bool_)
    qty_arr = np.empty(n, dtype=np.float64)
    for i, pos in enumerate(positions):
        s_arr[i] = float(pos["s"])
        k_arr[i] = float(pos["k"])
        t_arr[i] = float(pos["t"])
        r_arr[i] = float(pos["r"])
        q_arr[i] = float(pos.get("q", 0.0))
        sigma_arr[i] = float(pos["sigma"])
        is_call_arr[i] = pos.get("option_type", "call") == "call"
        qty_arr[i] = float(pos.get("quantity", 1))

    totals = bs_portfolio_aggregate_jit(
        s_arr, k_arr, t_arr, r_arr, q_arr, sigma_arr, is_call_arr, qty_arr,
    )
    return {key: float(val) for key, val in zip(_PORTFOLIO_KEYS, totals)}
