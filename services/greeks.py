"""Greek surface, sensitivity strip and portfolio aggregation helpers."""

from typing import Any, Dict, List, Tuple

import numpy as np

from ._cache import cached
from .pricing import get_option_and_pricer
from .registry import GREEK_ENGINE


@cached()
def get_greek_data(
    inputs: Dict[str, Any], s_range: np.ndarray
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Returns per-strike Greek series for both call and put flavours."""
    greek_keys = ["delta", "gamma", "vega", "theta", "rho"]
    call_greeks: Dict[str, List[float]] = {k: [] for k in greek_keys}
    put_greeks: Dict[str, List[float]] = {k: [] for k in greek_keys}
    greek_method = inputs.get("model_params", {}).get("greek_method", "default")

    z_matrix = None
    if inputs["pricer_type"] in ["Monte Carlo", "Longstaff-Schwartz"]:
        model_params = inputs.get("model_params", {})
        num_sims = model_params.get("num_sims", 10000)
        num_steps = model_params.get("num_steps", 100)
        z_matrix = np.random.standard_normal((num_sims, num_steps))

    for s_val in s_range:
        local = inputs.copy()
        local["contract_params"] = inputs["contract_params"].copy()
        local["contract_params"]["s"] = s_val

        _, pricer_c = get_option_and_pricer(local, "call")
        _, pricer_p = get_option_and_pricer(local, "put")
        if z_matrix is not None:
            if hasattr(pricer_c, "z_matrix"):
                pricer_c.z_matrix = z_matrix
            if hasattr(pricer_p, "z_matrix"):
                pricer_p.z_matrix = z_matrix

        greeks_c = GREEK_ENGINE.get_calculator(pricer_c, greek_method).calculate()
        greeks_p = GREEK_ENGINE.get_calculator(pricer_p, greek_method).calculate()
        for key in greek_keys:
            call_greeks[key].append(greeks_c.get(key, np.nan))
            put_greeks[key].append(greeks_p.get(key, np.nan))
    return call_greeks, put_greeks


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
    """Computes a 2D `(S, T)` surface for a single Black-Scholes Greek."""
    from models.pricers._analytic_kernels import bs_full_greeks_jit

    idx = {"price": 0, "delta": 1, "gamma": 2, "vega": 3, "theta": 4, "rho": 5}
    if greek not in idx:
        raise ValueError(f"Unknown greek: {greek!r}")
    pos = idx[greek]
    out = np.empty((s_range.size, t_range.size))
    for i in range(s_range.size):
        for j in range(t_range.size):
            tup = bs_full_greeks_jit(
                float(s_range[i]), k, float(t_range[j]),
                r, q, sigma, is_call,
            )
            out[i, j] = tup[pos]
    return out


@cached()
def aggregate_portfolio_greeks(
    positions: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Sums Black-Scholes Greeks across a list of vanilla option positions."""
    from models.pricers._analytic_kernels import bs_full_greeks_jit

    keys = ("price", "delta", "gamma", "vega", "theta", "rho")
    agg = {k: 0.0 for k in keys}
    for pos in positions:
        is_call = pos.get("option_type", "call") == "call"
        qty = float(pos.get("quantity", 1))
        tup = bs_full_greeks_jit(
            float(pos["s"]),
            float(pos["k"]),
            float(pos["t"]),
            float(pos["r"]),
            float(pos.get("q", 0.0)),
            float(pos["sigma"]),
            is_call,
        )
        for key, val in zip(keys, tup):
            agg[key] += qty * val
    return agg
