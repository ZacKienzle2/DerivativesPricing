from typing import Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np


def _get_model_params(pricer_type: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    # Only show the expander for models that have parameters
    models_with_params = ["Monte Carlo", "Longstaff-Schwartz", "Lattice"]
    if pricer_type not in models_with_params:
        return params

    with st.sidebar.expander("Model Parameters", expanded=True):
        if pricer_type == "Monte Carlo":
            st.markdown("##### Variance Reduction")
            use_av = st.toggle(
                "Antithetic Variates",
                value=True,
                help="Uses mirrored random numbers to reduce variance.",
            )
            use_qmc = st.toggle(
                "Quasi-Random Monte Carlo (Sobol)",
                value=False,
                help="Uses a low-discrepancy sequence for faster convergence.",
            )

            vr_methods = []
            if use_av:
                vr_methods.append("antithetic")
            if use_qmc:
                vr_methods.append("sobol")
            params["variance_reduction"] = vr_methods

            greek_methods = ["Finite Difference", "Pathwise", "Likelihood Ratio"]
            params["greek_method"] = st.selectbox("Greek Method", greek_methods)
            params["use_crn"] = st.toggle(
                "Common Random Numbers",
                value=True,
                help="Reuses random numbers for Greek calculations to reduce noise.",
            )

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


def _get_base_contract_inputs() -> Dict[str, Any]:
    with st.sidebar.expander("Core Parameters", expanded=True):
        s = st.number_input("Underlying Price (S)", 0.1, None, 100.0, 1.0, "%.2f")
        k = st.number_input("Strike Price (K)", 0.1, None, 100.0, 1.0, "%.2f")
        t_days = st.number_input("Days to Expiration (T)", 1, None, 90, 1)
        r = st.number_input("Risk-Free Rate (r)", 0.0, None, 0.05, 0.01, "%.2f")
        sigma = st.number_input("Volatility (σ)", 0.01, None, 0.20, 0.01, "%.2f")
        q = st.number_input("Dividend Yield (q)", 0.0, None, 0.0, 0.005, "%.3f")
    return {"s": s, "k": k, "t": t_days / 365.0, "r": r, "sigma": sigma, "q": q}


def get_basket_inputs() -> Dict[str, Any]:
    st.sidebar.markdown(
        "A basket option is priced on a portfolio of underlying assets."
    )
    with st.sidebar.expander("Basket Asset Definition", expanded=True):
        asset_data = pd.DataFrame(
            [
                {"Price": 100.0, "Volatility": 0.20, "Weight": 0.5},
                {"Price": 105.0, "Volatility": 0.25, "Weight": 0.5},
            ]
        )
        edited_assets = st.data_editor(
            asset_data, num_rows="dynamic", width="stretch", key="basket_assets"
        )
        k = st.number_input("Strike Price (K)", 0.1, None, 100.0, 1.0, "%.2f")
        t_days = st.number_input("Days to Expiration (T)", 1, None, 90, 1)
        r = st.number_input("Risk-Free Rate (r)", 0.0, None, 0.05, 0.01, "%.2f")

    with st.sidebar.expander("Correlation Matrix", expanded=False):
        corr_text = st.text_area(
            "Matrix (CSV format)",
            "1.0, 0.5\n0.5, 1.0",
            help="Enter the correlation matrix. Each row on a new line.",
        )
    try:
        weights = edited_assets["Weight"].astype(float).to_numpy()
        if not np.isclose(weights.sum(), 1.0):
            st.sidebar.error("Asset weights must sum to 1.0.")
            st.stop()
        corr_matrix = np.loadtxt(corr_text.splitlines(), delimiter=",")
        if corr_matrix.shape[0] != len(edited_assets):
            st.sidebar.error("Matrix dimensions must match asset count.")
            st.stop()
    except Exception as e:
        st.sidebar.error(f"Invalid Correlation Matrix: {e}")
        st.stop()

    contract_params = {
        "initial_prices": edited_assets["Price"].astype(float).tolist(),
        "k": k,
        "t": t_days / 365.0,
        "r": r,
        "volatilities": edited_assets["Volatility"].astype(float).tolist(),
        "weights": weights.tolist(),
        "corr_matrix": corr_matrix,
    }
    model_params = _get_model_params("Monte Carlo")

    return {
        "option_type": "Basket",
        "pricer_type": "Monte Carlo",
        "contract_params": contract_params,
        "model_params": model_params,
    }


def get_asian_inputs(contract_params: Dict[str, Any]) -> List[str]:
    with st.sidebar.expander("Asian Parameters", expanded=True):
        contract_params["avg_type"] = st.selectbox(
            "Averaging Type", ["arithmetic", "geometric"]
        )

    if contract_params["avg_type"] == "geometric":
        return ["Kemna-Vorst", "Monte Carlo", "Lattice"]
    return ["Monte Carlo", "Lattice", "Levy", "Turnbull-Wakeman", "Haug-Haug-Margrabe"]


def get_sidebar_inputs() -> Dict[str, Any]:
    st.sidebar.header("Configuration")
    option_type = st.sidebar.selectbox(
        "Option Type", ["Vanilla European", "American", "Barrier", "Basket", "Asian"]
    )

    if option_type == "Basket":
        return get_basket_inputs()

    contract_params = _get_base_contract_inputs()
    pricer_list: List[str] = []

    if option_type == "Barrier":
        with st.sidebar.expander("Barrier Parameters", expanded=True):
            contract_params["barrier_level"] = st.number_input(
                "Barrier Level", 0.1, None, 90.0, 1.0, "%.2f"
            )
            contract_params["barrier_type"] = st.selectbox(
                "Barrier Type",
                ["down-and-in", "up-and-in", "down-and-out", "up-and-out"],
            )
        pricer_list = ["Reiner-Rubinstein", "Monte Carlo", "Lattice"]
    elif option_type == "Asian":
        pricer_list = get_asian_inputs(contract_params)
    else:
        pricer_map = {
            "Vanilla European": ["Black-Scholes", "Lattice", "Monte Carlo"],
            "American": ["Lattice", "Longstaff-Schwartz"],
        }
        pricer_list = pricer_map.get(option_type, [])

    pricer_type = st.sidebar.selectbox("Pricing Model", pricer_list)
    model_params = _get_model_params(pricer_type)

    return {
        "option_type": option_type,
        "pricer_type": pricer_type,
        "contract_params": contract_params,
        "model_params": model_params,
    }
