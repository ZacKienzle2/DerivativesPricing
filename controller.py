import streamlit as st
from typing import Dict, Any, Tuple, Optional, List, Type
import numpy as np
import numpy.typing as npt
import dask

from models.options import BaseOption
from models.pricers import BasePricer
from utils.greek_engine import GreekEngine
from models.options import (
    VanillaOption,
    AmericanOption,
    BarrierOption,
    BasketOption,
    AsianOption,
)
from models.pricers import (
    BlackScholesPricer,
    LatticePricer,
    MonteCarloPricer,
    LongstaffSchwartzPricer,
    KemnaVorstPricer,
    LevyPricer,
    TurnbullWakemanPricer,
    HaugHaugMargrabePricer,
    ReinerRubinsteinPricer,
)
from models.payoffs import (
    long_call,
    short_call,
    long_put,
    short_put,
    long_stock,
    short_stock,
    long_bond,
    short_bond,
)

OPTION_MAP: Dict[str, Type[BaseOption]] = {
    "Vanilla European": VanillaOption,
    "American": AmericanOption,
    "Barrier": BarrierOption,
    "Basket": BasketOption,
    "Asian": AsianOption,
}
PRICER_MAP: Dict[str, Type[BasePricer]] = {
    "Black-Scholes": BlackScholesPricer,
    "Lattice": LatticePricer,
    "Monte Carlo": MonteCarloPricer,
    "Longstaff-Schwartz": LongstaffSchwartzPricer,
    "Kemna-Vorst": KemnaVorstPricer,
    "Levy": LevyPricer,
    "Turnbull-Wakeman": TurnbullWakemanPricer,
    "Haug-Haug-Margrabe": HaugHaugMargrabePricer,
    "Reiner-Rubinstein": ReinerRubinsteinPricer,
}
ANALYTICAL_PRICERS = {
    BlackScholesPricer,
    KemnaVorstPricer,
    LevyPricer,
    TurnbullWakemanPricer,
    HaugHaugMargrabePricer,
    ReinerRubinsteinPricer,
}
PAYOFF_REGISTRY = {
    "Long Call": {"func": long_call, "params": {"k": 100.0, "premium": 5.0}},
    "Short Call": {"func": short_call, "params": {"k": 100.0, "premium": 5.0}},
    "Long Put": {"func": long_put, "params": {"k": 100.0, "premium": 5.0}},
    "Short Put": {"func": short_put, "params": {"k": 100.0, "premium": 5.0}},
    "Long Stock": {"func": long_stock, "params": {"purchase_price": 100.0}},
    "Short Stock": {"func": short_stock, "params": {"sale_price": 100.0}},
    "Long Bond": {"func": long_bond, "params": {"future_value": 105.0, "price": 100.0}},
    "Short Bond": {
        "func": short_bond,
        "params": {"future_value": 105.0, "price": 100.0},
    },
}
GREEK_ENGINE = GreekEngine()


def get_option_and_pricer(
    inputs: Dict[str, Any], option_flavour: Optional[str] = None
) -> Tuple[BaseOption, BasePricer]:
    contract_params = inputs["contract_params"].copy()
    if option_flavour:
        contract_params["option_type"] = option_flavour

    OptionClass = OPTION_MAP[inputs["option_type"]]
    PricerClass = PRICER_MAP[inputs["pricer_type"]]
    option = OptionClass(**contract_params)

    model_params = inputs.get("model_params", {}).copy()
    model_params.pop("greek_method", None)

    if PricerClass in ANALYTICAL_PRICERS:
        return option, PricerClass(option)
    return option, PricerClass(option, **model_params)


def get_point_pricing_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    try:
        greek_method = inputs.get("model_params", {}).get("greek_method", "default")

        _, pricer_c = get_option_and_pricer(inputs, "call")
        price_res_c = pricer_c.price()
        greeks_c = GREEK_ENGINE.get_calculator(pricer_c, greek_method).calculate()

        _, pricer_p = get_option_and_pricer(inputs, "put")
        if hasattr(pricer_c, "z_matrix") and pricer_c.z_matrix is not None:
            # Use is not None for clarity with NumPy arrays
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
    except Exception as e:
        st.error(f"Calculation Error: {e}")
        return {}
    return results


# --- Dask helper function for parallel pricing ---
@dask.delayed
def price_single_point(
    base_inputs: Dict[str, Any],
    axis_map: Dict[str, str],
    x_axis_key: str,
    y_axis_key: str,
    x_value: float,
    y_value: float,
    shared_z_matrix: npt.NDArray[np.float64] | None,
) -> Tuple[float, float]:
    try:
        local_inputs = base_inputs.copy()
        local_inputs["contract_params"] = base_inputs["contract_params"].copy()
        local_inputs["contract_params"][axis_map[x_axis_key]] = x_value
        local_inputs["contract_params"][axis_map[y_axis_key]] = y_value

        _, pricer_call = get_option_and_pricer(local_inputs, "call")
        _, pricer_put = get_option_and_pricer(local_inputs, "put")

        if shared_z_matrix is not None:
            # Use is not None for clarity with NumPy arrays
            if hasattr(pricer_call, "z_matrix"):
                pricer_call.z_matrix = shared_z_matrix
            if hasattr(pricer_put, "z_matrix"):
                pricer_put.z_matrix = shared_z_matrix

        price_call, _ = pricer_call.price()
        price_put, _ = pricer_put.price()
        return float(price_call), float(price_put)
    except Exception as e:
        print(f"Error pricing point ({x_value}, {y_value}): {e}")
        return np.nan, np.nan


@st.cache_data
def get_surface_data(
    inputs: Dict[str, Any],
    axis_map: Dict[str, str],
    x_key: str,
    y_key: str,
    ranges: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    x_range = ranges[x_key]
    y_range = ranges[y_key]
    grid_x, grid_y = np.meshgrid(x_range, y_range)

    shared_z_matrix = None
    if inputs["pricer_type"] in ["Monte Carlo", "Longstaff-Schwartz"]:
        model_params = inputs.get("model_params", {})
        num_sims = model_params.get("num_sims", 10000)
        num_steps = model_params.get("num_steps", 100)
        shared_z_matrix = np.random.standard_normal((num_sims, num_steps))

    lazy_results = []
    for i, j in np.ndindex(grid_x.shape):
        task = price_single_point(
            inputs, axis_map, x_key, y_key, grid_x[i, j], grid_y[i, j], shared_z_matrix
        )
        lazy_results.append(task)

    results = dask.compute(*lazy_results)

    results_array = np.array(results)
    surface_c = results_array[:, 0].reshape(grid_x.shape)
    surface_p = results_array[:, 1].reshape(grid_x.shape)

    return np.nan_to_num(surface_c), np.nan_to_num(surface_p)


@st.cache_data(show_spinner=False)
def fit_svi_slice(
    strikes: np.ndarray, ivs: np.ndarray, f0: float, t: float
) -> Dict[str, Any]:
    """Fits raw SVI to a market IV slice and returns a UI-ready bundle.

    Args:
        strikes: Strike grid.
        ivs: Market implied vols.
        f0: Forward.
        t: Maturity.

    Returns:
        Dict with `strikes`, `market_iv`, `params`, `residual_norm`,
        `evaluator` (callable) and `t` for downstream rendering.
    """
    from models.calibration import SVICalibrator

    calibrator = SVICalibrator()
    result = calibrator.calibrate(strikes, ivs, f0=f0, t=t)
    params = result.params

    def evaluator(ks: np.ndarray):
        return SVICalibrator.evaluate(params, np.asarray(ks), f0, t)

    return {
        "strikes": np.asarray(strikes, dtype=float),
        "market_iv": np.asarray(ivs, dtype=float),
        "params": params,
        "residual_norm": float(result.residual_norm),
        "evaluator": evaluator,
        "t": float(t),
        "f0": float(f0),
        "converged": bool(result.converged),
    }


@st.cache_data
def get_greek_data(
    inputs: Dict[str, Any], s_range: np.ndarray
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
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
        mock_inputs = inputs.copy()
        mock_inputs["contract_params"] = inputs["contract_params"].copy()
        mock_inputs["contract_params"]["s"] = s_val

        _, pricer_c = get_option_and_pricer(mock_inputs, "call")
        _, pricer_p = get_option_and_pricer(mock_inputs, "put")

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
