"""Point pricing and price surface helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from models.options import BaseOption
from models.pricers import BasePricer

from ._cache import cached
from ._request import PricingRequest
from ._timing import timed
from ._validation import ValidationError
from .logging import get_logger
from .registry import GREEK_ENGINE

_log = get_logger("pricing")

_SHARED_Z_PRICERS: frozenset[str] = frozenset({"Monte Carlo", "Longstaff-Schwartz"})
_SURFACE_FAILURE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ValidationError,
    ValueError,
    ArithmeticError,
)


def get_option_and_pricer(
    inputs: PricingRequest | dict[str, Any],
    option_flavour: str | None = None,
) -> tuple[BaseOption, BasePricer]:
    """Builds the configured option contract and pricer instance.

    Accepts either a `PricingRequest` (preferred) or a UI-shaped dict
    (legacy). Dicts are validated and converted on entry.
    """
    if isinstance(inputs, PricingRequest):
        request = (
            inputs
            if option_flavour is None
            else inputs.with_overrides(option_flavour=option_flavour)
        )
    else:
        request = PricingRequest.from_dict(inputs, option_flavour=option_flavour)
    return request.build()


@cached()
@timed("services.pricing.point")
def get_point_pricing_context(inputs: dict[str, Any]) -> dict[str, Any]:
    """Returns price + Greeks for both call and put flavours of the option."""
    results: dict[str, Any] = {}
    try:
        request = PricingRequest.from_dict(inputs)
        method = request.greek_method

        _, pricer_c = request.with_overrides(option_flavour="call").build()
        price_res_c = pricer_c.price()
        greeks_c = GREEK_ENGINE.get_calculator(pricer_c, method).calculate()

        _, pricer_p = request.with_overrides(option_flavour="put").build()
        if hasattr(pricer_c, "z_matrix") and pricer_c.z_matrix is not None:
            pricer_p.z_matrix = pricer_c.z_matrix
        price_res_p = pricer_p.price()
        greeks_p = GREEK_ENGINE.get_calculator(pricer_p, method).calculate()

        results["call"] = {
            "price": price_res_c[0],
            "std_err": price_res_c[1],
            "greeks": greeks_c,
            "pricer": pricer_c,
        }
        results["put"] = {
            "price": price_res_p[0],
            "std_err": price_res_p[1],
            "greeks": greeks_p,
            "pricer": pricer_p,
        }
    except Exception as exc:
        _log.exception("point pricing failed")
        results["error"] = str(exc)
    return results


def _price_single_point(
    call_request: PricingRequest,
    put_request: PricingRequest,
    x_attr: str,
    y_attr: str,
    x_value: float,
    y_value: float,
    shared_z_matrix: npt.NDArray[np.float64] | None,
) -> tuple[float, float]:
    """Prices a single (x, y) grid point under pre-resolved request specs."""
    overrides = {x_attr: x_value, y_attr: y_value}
    try:
        _, pricer_call = call_request.with_overrides(**overrides).build()
        _, pricer_put = put_request.with_overrides(**overrides).build()
        if shared_z_matrix is not None:
            if hasattr(pricer_call, "z_matrix"):
                pricer_call.z_matrix = shared_z_matrix
            if hasattr(pricer_put, "z_matrix"):
                pricer_put.z_matrix = shared_z_matrix
        price_call, _ = pricer_call.price()
        price_put, _ = pricer_put.price()
        return float(price_call), float(price_put)
    except _SURFACE_FAILURE_EXCEPTIONS:
        _log.debug("surface point failed", exc_info=True)
        return float("nan"), float("nan")


def _vectorised_bs_surface(
    request: PricingRequest,
    x_attr: str,
    y_attr: str,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Vectorised Black-Scholes surface bypassing the per cell python loop.

    Returns `None` when any contract parameter falls outside Black-Scholes
    domain (e.g. non positive vol or maturity), letting the caller fall
    back to the generic path.
    """
    base = request.contract_params
    s = float(base["s"])
    k = float(base["k"])
    t = float(base["t"])
    r = float(base["r"])
    q = float(base.get("q", 0.0))
    sigma = float(base["sigma"])
    overrides = {x_attr: grid_x, y_attr: grid_y}
    s_arr = np.asarray(
        overrides.get("s", np.broadcast_to(s, grid_x.shape)), dtype=np.float64
    )
    k_arr = np.asarray(
        overrides.get("k", np.broadcast_to(k, grid_x.shape)), dtype=np.float64
    )
    t_arr = np.asarray(
        overrides.get("t", np.broadcast_to(t, grid_x.shape)), dtype=np.float64
    )
    sigma_arr = np.asarray(
        overrides.get("sigma", np.broadcast_to(sigma, grid_x.shape)),
        dtype=np.float64,
    )
    if np.any(t_arr <= 0) or np.any(sigma_arr <= 0):
        return None

    from scipy.special import ndtr

    sqrt_t = np.sqrt(t_arr)
    sigma_sqrt_t = sigma_arr * sqrt_t
    sig2_half = 0.5 * sigma_arr * sigma_arr
    d1 = np.log(s_arr / k_arr)
    d1 += (r - q + sig2_half) * t_arr
    d1 /= sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t

    disc_q = np.exp(-q * t_arr)
    disc_r = np.exp(-r * t_arr)
    nd1 = ndtr(d1)
    nd2 = ndtr(d2)
    sd_q = s_arr * disc_q
    kd_r = k_arr * disc_r
    call = sd_q * nd1 - kd_r * nd2
    put = kd_r - kd_r * nd2 - sd_q + sd_q * nd1
    return np.nan_to_num(call, copy=False), np.nan_to_num(put, copy=False)


@cached()
@timed("services.pricing.surface")
def get_surface_data(
    inputs: dict[str, Any],
    axis_map: dict[str, str],
    x_key: str,
    y_key: str,
    ranges: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Builds call and put price surfaces over a 2D parameter grid."""
    request = PricingRequest.from_dict(inputs)
    x_attr = axis_map[x_key]
    y_attr = axis_map[y_key]
    grid_x, grid_y = np.meshgrid(ranges[x_key], ranges[y_key])
    if request.pricer_type == "Black-Scholes":
        fast = _vectorised_bs_surface(request, x_attr, y_attr, grid_x, grid_y)
        if fast is not None:
            return fast

    shape = grid_x.shape
    shared_z_matrix: npt.NDArray[np.float64] | None = None
    if request.pricer_type in _SHARED_Z_PRICERS:
        num_sims = request.model_params.get("num_sims", 10000)
        num_steps = request.model_params.get("num_steps", 100)
        shared_z_matrix = np.random.standard_normal((num_sims, num_steps))

    call_request = request.with_overrides(option_flavour="call")
    put_request = request.with_overrides(option_flavour="put")

    surface_c = np.empty(shape)
    surface_p = np.empty(shape)
    grid_x_flat = grid_x.ravel()
    grid_y_flat = grid_y.ravel()
    cols = shape[1]
    for flat_idx in range(grid_x_flat.size):
        c, p = _price_single_point(
            call_request,
            put_request,
            x_attr,
            y_attr,
            float(grid_x_flat[flat_idx]),
            float(grid_y_flat[flat_idx]),
            shared_z_matrix,
        )
        i, j = divmod(flat_idx, cols)
        surface_c[i, j] = c
        surface_p[i, j] = p
    return np.nan_to_num(surface_c), np.nan_to_num(surface_p)
