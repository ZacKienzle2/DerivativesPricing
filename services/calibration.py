"""Calibration helpers (synthetic quote generation, Heston, SVI)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._cache import cached
from .logging import get_logger

_log = get_logger("calibration")


@cached()
def generate_synthetic_quotes(
    process_name: str,
    params: dict[str, Any],
    strikes: np.ndarray,
    maturities: np.ndarray,
    spread_bps: float = 25.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Builds synthetic call quotes from a chosen process and parameter set."""
    from models.processes import BatesProcess, HestonProcess
    from utils.synthetic_quotes import synthetic_call_quotes

    factories = {"Heston": HestonProcess, "Bates": BatesProcess}
    if process_name not in factories:
        raise ValueError(f"Synthetic quotes: process {process_name!r} not supported.")
    process = factories[process_name](**params)
    quotes = synthetic_call_quotes(
        process, strikes, maturities, spread_bps=spread_bps, seed=seed
    )
    return {
        "strikes": quotes.strikes,
        "maturities": quotes.maturities,
        "prices": quotes.prices,
        "bids": quotes.bids,
        "asks": quotes.asks,
        "ivs": quotes.ivs,
    }


@cached()
def fit_heston_to_quotes(
    quotes: dict[str, np.ndarray],
    s0: float,
    r: float = 0.0,
    q: float = 0.0,
    weights_mode: str = "vega",
    x0: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Calibrates Heston parameters to a synthetic or market quote bundle."""
    from models.calibration import HestonCalibrator
    from models.pricers.cos_pricer import cos_heston_price_jit

    n_k, n_t = quotes["prices"].shape
    k_grid, t_grid = np.meshgrid(
        quotes["strikes"], quotes["maturities"], indexing="ij"
    )
    flat_k = k_grid.reshape(-1)
    flat_t = t_grid.reshape(-1)
    flat_prices = quotes["prices"].reshape(-1)
    flat_bids = quotes["bids"].reshape(-1)
    flat_asks = quotes["asks"].reshape(-1)
    flat_ivs = quotes["ivs"].reshape(-1)

    calibrator = HestonCalibrator(s0=s0, r=r, q=q)
    init = (
        np.array(x0, dtype=float)
        if x0 is not None
        else np.array([1.5, 0.05, 0.4, -0.3, 0.05])
    )
    try:
        result = calibrator.calibrate(
            flat_k,
            flat_prices,
            maturities=flat_t,
            weights=weights_mode,
            bids=flat_bids,
            asks=flat_asks,
            market_iv=flat_ivs,
            x0=init,
        )
    except Exception:
        _log.exception("Heston calibration failed")
        raise

    fitted = result.params
    model_prices = np.empty((n_k, n_t))
    for i in range(n_k):
        for j in range(n_t):
            model_prices[i, j] = cos_heston_price_jit(
                s0,
                float(quotes["strikes"][i]),
                float(quotes["maturities"][j]),
                r,
                q,
                fitted["kappa"],
                fitted["theta"],
                fitted["eta"],
                fitted["rho"],
                fitted["v0"],
                True,
                256,
                12.0,
            )
    return {
        "params": fitted,
        "residual_norm": float(result.residual_norm),
        "n_iter": int(result.n_iter),
        "converged": bool(result.converged),
        "model_prices": model_prices,
    }


@cached()
def fit_svi_slice(
    strikes: np.ndarray, ivs: np.ndarray, f0: float, t: float
) -> dict[str, Any]:
    """Fits raw SVI to a market IV slice."""
    from models.calibration import SVICalibrator

    calibrator = SVICalibrator()
    result = calibrator.calibrate(strikes, ivs, f0=f0, t=t)
    return {
        "strikes": np.asarray(strikes, dtype=float),
        "market_iv": np.asarray(ivs, dtype=float),
        "params": result.params,
        "residual_norm": float(result.residual_norm),
        "t": float(t),
        "f0": float(f0),
        "converged": bool(result.converged),
    }
