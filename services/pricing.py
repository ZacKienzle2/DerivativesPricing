"""Point pricing and price surface helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from models.options import BaseOption
from models.pricers import BasePricer

from ._cache import cached
from .logging import get_logger
from .registry import ANALYTICAL_PRICERS, GREEK_ENGINE, OPTION_MAP, PRICER_MAP

_log = get_logger("pricing")


def get_option_and_pricer(
    inputs: dict[str, Any], option_flavour: str | None = None
) -> tuple[BaseOption, BasePricer]:
    """Builds the configured option contract and pricer instance."""
    contract_params = inputs["contract_params"].copy()
    if option_flavour:
        contract_params["option_type"] = option_flavour
    option_cls = OPTION_MAP[inputs["option_type"]]
    pricer_cls = PRICER_MAP[inputs["pricer_type"]]
    option = option_cls(**contract_params)
    model_params = inputs.get("model_params", {}).copy()
    model_params.pop("greek_method", None)
    if pricer_cls in ANALYTICAL_PRICERS:
        return option, pricer_cls(option)
    return option, pricer_cls(option, **model_params)


def get_point_pricing_context(inputs: dict[str, Any]) -> dict[str, Any]:
    """Returns price + Greeks for both call and put flavours of the option."""
    results: dict[str, Any] = {}
    try:
        greek_method = inputs.get("model_params", {}).get("greek_method", "default")

        _, pricer_c = get_option_and_pricer(inputs, "call")
        price_res_c = pricer_c.price()
        greeks_c = GREEK_ENGINE.get_calculator(pricer_c, greek_method).calculate()

        _, pricer_p = get_option_and_pricer(inputs, "put")
        if hasattr(pricer_c, "z_matrix") and pricer_c.z_matrix is not None:
            pricer_p.z_matrix = pricer_c.z_matrix
        price_res_p = pricer_p.price()
        greeks_p = GREEK_ENGINE.get_calculator(pricer_p, greek_method).calculate()

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
    base_inputs: dict[str, Any],
    axis_map: dict[str, str],
    x_axis_key: str,
    y_axis_key: str,
    x_value: float,
    y_value: float,
    shared_z_matrix: npt.NDArray[np.float64] | None,
) -> tuple[float, float]:
    """Prices a single (x, y) grid point under the configured option."""
    try:
        local_inputs = base_inputs.copy()
        local_inputs["contract_params"] = base_inputs["contract_params"].copy()
        local_inputs["contract_params"][axis_map[x_axis_key]] = x_value
        local_inputs["contract_params"][axis_map[y_axis_key]] = y_value
        _, pricer_call = get_option_and_pricer(local_inputs, "call")
        _, pricer_put = get_option_and_pricer(local_inputs, "put")
        if shared_z_matrix is not None:
            if hasattr(pricer_call, "z_matrix"):
                pricer_call.z_matrix = shared_z_matrix
            if hasattr(pricer_put, "z_matrix"):
                pricer_put.z_matrix = shared_z_matrix
        price_call, _ = pricer_call.price()
        price_put, _ = pricer_put.price()
        return float(price_call), float(price_put)
    except Exception:
        _log.debug("surface point failed", exc_info=True)
        return float("nan"), float("nan")


@cached()
def get_surface_data(
    inputs: dict[str, Any],
    axis_map: dict[str, str],
    x_key: str,
    y_key: str,
    ranges: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Builds call and put price surfaces over a 2D parameter grid."""
    grid_x, grid_y = np.meshgrid(ranges[x_key], ranges[y_key])
    shape = grid_x.shape
    shared_z_matrix = None
    if inputs["pricer_type"] in ["Monte Carlo", "Longstaff-Schwartz"]:
        model_params = inputs.get("model_params", {})
        num_sims = model_params.get("num_sims", 10000)
        num_steps = model_params.get("num_steps", 100)
        shared_z_matrix = np.random.standard_normal((num_sims, num_steps))
    surface_c = np.empty(shape)
    surface_p = np.empty(shape)
    for i, j in np.ndindex(shape):
        c, p = _price_single_point(
            inputs, axis_map, x_key, y_key,
            grid_x[i, j], grid_y[i, j], shared_z_matrix,
        )
        surface_c[i, j] = c
        surface_p[i, j] = p
    return np.nan_to_num(surface_c), np.nan_to_num(surface_p)
