"""Sidebar configuration form for the dashboard."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

_OptionBuilder = Callable[[dict[str, Any]], list[str]]
_OPTION_TYPES: tuple[str, ...] = (
    "Vanilla European",
    "American",
    "Barrier",
    "Basket",
    "Asian",
)


def _model_params(pricer_type: str) -> dict[str, Any]:
    """Returns method-specific model parameters from the sidebar form."""
    params: dict[str, Any] = {}
    if pricer_type not in ("Monte Carlo", "Longstaff-Schwartz", "Lattice"):
        return params

    with st.sidebar.expander("Model Parameters", expanded=True):
        if pricer_type == "Monte Carlo":
            st.markdown("##### Variance Reduction")
            use_av = st.toggle("Antithetic Variates", value=True)
            use_qmc = st.toggle("Quasi-Random Monte Carlo (Sobol)", value=False)
            vr = []
            if use_av:
                vr.append("antithetic")
            if use_qmc:
                vr.append("sobol")
            params["variance_reduction"] = vr
            params["greek_method"] = st.selectbox(
                "Greek Method", ["Finite Difference", "Pathwise", "Likelihood Ratio"]
            )
            params["use_crn"] = st.toggle("Common Random Numbers", value=True)
            st.markdown("##### Simulation Parameters")
            params["num_sims"] = st.slider("Simulations", 100, 50000, 10000, 100)
            params["num_steps"] = st.slider("Time Steps", 10, 1000, 100, 10)
        elif pricer_type == "Longstaff-Schwartz":
            params["greek_method"] = "Finite Difference"
            params["use_crn"] = st.toggle("Common Random Numbers", value=True)
            params["num_sims"] = st.slider("Simulations", 100, 20000, 10000, 100)
            params["num_steps"] = st.slider("Time Steps", 10, 1000, 100, 10)
            params["poly_degree"] = st.slider("Polynomial Degree", 1, 7, 3, 1)
        elif pricer_type == "Lattice":
            params["num_steps"] = st.slider("Number of Steps", 10, 5000, 500, 10)
            params["model"] = st.selectbox("Lattice Model", ["CRR", "Boyle"])
    return params


def _base_contract() -> dict[str, Any]:
    """Single-asset contract block shared across non-basket options."""
    with st.sidebar.expander("Core Parameters", expanded=True):
        s = st.number_input("Underlying Price (S)", 0.1, None, 100.0, 1.0, "%.2f")
        k = st.number_input("Strike Price (K)", 0.1, None, 100.0, 1.0, "%.2f")
        t_days = st.number_input("Days to Expiration (T)", 1, None, 90, 1)
        r = st.number_input("Risk-Free Rate (r)", 0.0, None, 0.05, 0.01, "%.2f")
        sigma = st.number_input("Volatility (sigma)", 0.01, None, 0.20, 0.01, "%.2f")
        q = st.number_input("Dividend Yield (q)", 0.0, None, 0.0, 0.005, "%.3f")
    return {"s": s, "k": k, "t": t_days / 365.0, "r": r, "sigma": sigma, "q": q}


def _vanilla_pricers(_: dict[str, Any]) -> list[str]:
    return ["Black-Scholes", "Lattice", "Monte Carlo"]


def _american_pricers(_: dict[str, Any]) -> list[str]:
    return ["Lattice", "Longstaff-Schwartz"]


def _barrier_pricers(contract: dict[str, Any]) -> list[str]:
    with st.sidebar.expander("Barrier Parameters", expanded=True):
        contract["barrier_level"] = st.number_input(
            "Barrier Level", 0.1, None, 90.0, 1.0, "%.2f"
        )
        contract["barrier_type"] = st.selectbox(
            "Barrier Type",
            ["down-and-in", "up-and-in", "down-and-out", "up-and-out"],
        )
    return ["Reiner-Rubinstein", "Monte Carlo", "Lattice"]


def _asian_pricers(contract: dict[str, Any]) -> list[str]:
    with st.sidebar.expander("Asian Parameters", expanded=True):
        contract["avg_type"] = st.selectbox(
            "Averaging Type", ["arithmetic", "geometric"]
        )
    if contract["avg_type"] == "geometric":
        return ["Kemna-Vorst", "Monte Carlo", "Lattice"]
    return ["Monte Carlo", "Lattice", "Levy", "Turnbull-Wakeman", "Haug-Haug-Margrabe"]


_BUILDERS: dict[str, _OptionBuilder] = {
    "Vanilla European": _vanilla_pricers,
    "American": _american_pricers,
    "Barrier": _barrier_pricers,
    "Asian": _asian_pricers,
}


def _basket_inputs() -> dict[str, Any]:
    """Multi-asset basket has its own contract layout and pricer set."""
    st.sidebar.markdown(
        "Basket option priced on a weighted average of underlying assets."
    )
    with st.sidebar.expander("Basket Asset Definition", expanded=True):
        asset_data = pd.DataFrame(
            [
                {"Price": 100.0, "Volatility": 0.20, "Weight": 0.5},
                {"Price": 105.0, "Volatility": 0.25, "Weight": 0.5},
            ]
        )
        edited = st.data_editor(
            asset_data, num_rows="dynamic", width="stretch", key="basket_assets"
        )
        k = st.number_input("Strike Price (K)", 0.1, None, 100.0, 1.0, "%.2f")
        t_days = st.number_input("Days to Expiration (T)", 1, None, 90, 1)
        r = st.number_input("Risk-Free Rate (r)", 0.0, None, 0.05, 0.01, "%.2f")
    with st.sidebar.expander("Correlation Matrix", expanded=False):
        corr_text = st.text_area("Matrix (CSV)", "1.0, 0.5\n0.5, 1.0")
    try:
        weights = edited["Weight"].astype(float).to_numpy()
        if not np.isclose(weights.sum(), 1.0):
            st.sidebar.error("Asset weights must sum to 1.0.")
            st.stop()
        corr = np.loadtxt(corr_text.splitlines(), delimiter=",")
        if corr.shape[0] != len(edited):
            st.sidebar.error("Matrix dimensions must match asset count.")
            st.stop()
    except Exception as exc:
        st.sidebar.error(f"Invalid correlation matrix: {exc}")
        st.stop()
    return {
        "option_type": "Basket",
        "pricer_type": "Monte Carlo",
        "contract_params": {
            "initial_prices": edited["Price"].astype(float).tolist(),
            "k": k,
            "t": t_days / 365.0,
            "r": r,
            "volatilities": edited["Volatility"].astype(float).tolist(),
            "weights": weights.tolist(),
            "corr_matrix": corr,
        },
        "model_params": _model_params("Monte Carlo"),
    }


def get_sidebar_inputs() -> dict[str, Any]:
    """Builds the configuration bundle from the sidebar form."""
    st.sidebar.header("Configuration")
    option_type = st.sidebar.selectbox("Option Type", _OPTION_TYPES)
    if option_type == "Basket":
        return _basket_inputs()

    contract_params = _base_contract()
    pricer_list = _BUILDERS.get(option_type, lambda _: [])(contract_params)
    pricer_type = st.sidebar.selectbox("Pricing Model", pricer_list)
    return {
        "option_type": option_type,
        "pricer_type": pricer_type,
        "contract_params": contract_params,
        "model_params": _model_params(pricer_type),
    }
