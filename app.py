"""Streamlit dashboard entry point.

Wires the sidebar configuration into the controller and renders six tabs:
contract pricing, price surface, Greek sensitivity, model convergence,
strategy builder and volatility-surface calibration. All compute-heavy
operations are funnelled through `controller` so the UI stays a thin
presentation layer.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from controller import (
    ANALYTICAL_PRICERS,
    PAYOFF_REGISTRY,
    fit_svi_slice,
    get_greek_data,
    get_point_pricing_context,
    get_surface_data,
)
from ui.components import display_metrics_card, section_header, stat_strip
from ui.sidebar import get_sidebar_inputs
from utils.plotting import (
    plot_3d_surface,
    plot_greek_sensitivity,
    plot_payoff_diagram,
)


st.set_page_config(
    page_title="Derivatives Pricer",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def _load_css() -> str:
    return Path("static/style.css").read_text()


st.markdown(f"<style>{_load_css()}</style>", unsafe_allow_html=True)


for key, default in (
    ("strategy", []),
    ("call_pricer", None),
    ("put_pricer", None),
    ("svi_fit", None),
):
    if key not in st.session_state:
        st.session_state[key] = default


def render_hero() -> None:
    """Top banner."""
    st.markdown(
        '<h1 class="hero-title">Derivatives Pricer</h1>'
        '<div class="hero-subtitle">'
        'Analytic, lattice, finite-difference and Monte Carlo pricing &mdash; '
        'with stochastic-vol processes, COS Fourier and full Greek vectors.'
        '</div>',
        unsafe_allow_html=True,
    )


def render_point_pricing_tab(inputs: Dict[str, Any]) -> None:
    """Tab 0: contract pricing + Greeks summary."""
    section_header("Contract Pricing & Greeks")
    metrics = get_point_pricing_context(inputs)
    if not metrics:
        st.warning("Could not compute metrics. Please check parameters.")
        return

    st.session_state.call_pricer = metrics.get("call", {}).get("pricer")
    st.session_state.put_pricer = metrics.get("put", {}).get("pricer")

    inputs_summary = {
        "Spot": inputs["contract_params"].get("s"),
        "Strike": inputs["contract_params"].get("k"),
        "Maturity (yrs)": inputs["contract_params"].get("t"),
        "Vol": inputs["contract_params"].get("sigma"),
        "Rate": inputs["contract_params"].get("r"),
    }
    stat_strip({k: v for k, v in inputs_summary.items() if v is not None})

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        if "call" in metrics:
            display_metrics_card(
                "Call", metrics["call"], inputs["option_type"], ANALYTICAL_PRICERS
            )
    with col2:
        if "put" in metrics:
            display_metrics_card(
                "Put", metrics["put"], inputs["option_type"], ANALYTICAL_PRICERS
            )


def render_surface_tab(inputs: Dict[str, Any]) -> None:
    """Tab 1: 3D / heatmap price surface across two contract dimensions."""
    section_header("Price Surface Analysis")
    st.sidebar.header("Surface Plot Configuration")
    axis_map = {
        "Underlying Price": "s",
        "Strike Price": "k",
        "Time to Expiration": "t",
        "Volatility": "sigma",
    }
    x_key = st.sidebar.selectbox(
        "X-Axis Variable", list(axis_map.keys()), 0, key="surface_x"
    )
    y_key = st.sidebar.selectbox(
        "Y-Axis Variable", list(axis_map.keys()), 3, key="surface_y"
    )
    if x_key == y_key:
        st.error("X and Y axes must be different.")
        return

    ranges = {}
    for key in [x_key, y_key]:
        default = inputs["contract_params"][axis_map[key]]
        start, end = (default * 0.8, default * 1.2)
        min_v, max_v = st.sidebar.slider(
            f"Range for {key}", 0.01, default * 2.0, (start, end), key=f"range_{key}"
        )
        ranges[key] = np.linspace(min_v, max_v, 25)

    with st.container(border=True):
        plot_type = st.radio(
            "Plot Type",
            ["3D Surface", "2D Heatmap"],
            horizontal=True,
            key="surface_plot_type",
        )
        with st.spinner("Generating price surfaces..."):
            surface_c, surface_p = get_surface_data(
                inputs, axis_map, x_key, y_key, ranges
            )
        tab_c, tab_p = st.tabs(["Call Surface", "Put Surface"])
        for name, tab, data, cmap in (
            ("Call", tab_c, surface_c, "Viridis"),
            ("Put", tab_p, surface_p, "Plasma"),
        ):
            with tab:
                title = f"{name} Option Price Surface"
                if plot_type == "3D Surface":
                    fig = plot_3d_surface(
                        ranges[x_key], ranges[y_key], data, x_key, y_key, title
                    )
                else:
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=data.T,
                            x=ranges[x_key],
                            y=ranges[y_key],
                            colorscale=cmap,
                            colorbar={"title": "Price"},
                        )
                    )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)


def render_greeks_tab(inputs: Dict[str, Any]) -> None:
    """Tab 2: Greek sensitivity curves vs spot."""
    section_header("Greek Sensitivity Analysis")
    s_val = inputs["contract_params"]["s"]
    s_range = np.linspace(max(0.01, s_val * 0.7), s_val * 1.3, 100)

    with st.container(border=True):
        with st.spinner("Generating Greek sensitivity data..."):
            call_greeks, put_greeks = get_greek_data(inputs, s_range)
        tab_c, tab_p = st.tabs(["Call Greeks", "Put Greeks"])
        with tab_c:
            st.plotly_chart(
                plot_greek_sensitivity(
                    s_range, call_greeks, "Call Option Greeks vs. Underlying Price"
                ),
                use_container_width=True,
            )
        with tab_p:
            st.plotly_chart(
                plot_greek_sensitivity(
                    s_range, put_greeks, "Put Option Greeks vs. Underlying Price"
                ),
                use_container_width=True,
            )


def render_convergence_tab(inputs: Dict[str, Any]) -> None:
    """Tab 3: MC convergence trace."""
    section_header("Model Convergence Analysis")
    pricer_type = inputs["pricer_type"]
    if pricer_type not in ["Monte Carlo", "Longstaff-Schwartz"]:
        st.info("Convergence plots are only for numerical methods.")
        return

    with st.container(border=True):
        st.markdown(f"Visualising price convergence for the **{pricer_type}** model.")
        call_pricer = st.session_state.get("call_pricer")
        put_pricer = st.session_state.get("put_pricer")
        fig = go.Figure()
        for label, pricer in (("Call Price", call_pricer), ("Put Price", put_pricer)):
            if (
                pricer is not None
                and hasattr(pricer, "convergence_data")
                and pricer.convergence_data is not None
            ):
                fig.add_trace(
                    go.Scatter(y=pricer.convergence_data, mode="lines", name=label)
                )
        fig.update_layout(
            title=f"{pricer_type} Convergence",
            xaxis_title="Number of Simulations",
            yaxis_title="Option Price",
            template="plotly_dark",
            margin={"l": 40, "r": 20, "t": 60, "b": 40},
        )
        st.plotly_chart(fig, use_container_width=True)


def render_strategy_tab() -> None:
    """Tab 4: payoff strategy builder."""
    section_header("Strategy Payoff Builder")
    LABEL_MAP = {
        "k": "Strike",
        "premium": "Premium",
        "purchase_price": "Purchase Price",
        "sale_price": "Sale Price",
        "future_value": "Future Value",
        "price": "Price",
    }
    col1, col2 = st.columns([1, 2], gap="medium")
    with col1:
        with st.container(border=True):
            st.subheader("Build Your Strategy")
            instrument = st.selectbox("Select Instrument", list(PAYOFF_REGISTRY.keys()))
            if st.button("Add to Strategy", use_container_width=True):
                st.session_state.strategy.append(
                    {
                        "id": len(st.session_state.strategy),
                        "type": instrument,
                        "params": PAYOFF_REGISTRY[instrument]["params"].copy(),
                    }
                )
                st.rerun()
            if st.session_state.strategy and st.button(
                "Clear All", use_container_width=True
            ):
                st.session_state.strategy.clear()
                st.rerun()
        for i, pos in enumerate(st.session_state.strategy):
            with st.expander(f"Position {i + 1}: {pos['type']}", expanded=True):
                for param_name, default_val in pos["params"].items():
                    display_label = LABEL_MAP.get(
                        param_name, param_name.replace("_", " ").title()
                    )
                    pos["params"][param_name] = st.number_input(
                        label=display_label,
                        value=float(default_val),
                        key=f"pos_{pos['id']}_{param_name}",
                        format="%.2f",
                        min_value=0.0,
                        step=1.0,
                    )
                if st.button(
                    "Remove", key=f"pos_{pos['id']}_remove", use_container_width=True
                ):
                    st.session_state.strategy.pop(i)
                    st.rerun()
    with col2:
        with st.container(border=True):
            if not st.session_state.strategy:
                st.info("Add an instrument to begin building a strategy.")
                return
            try:
                prices = [
                    p
                    for pos in st.session_state.strategy
                    for k_param, p in pos["params"].items()
                    if "k" in k_param or "price" in k_param
                ]
                center = np.mean(prices) if prices else 100.0
                s_range = np.linspace(center * 0.5, center * 1.5, 200)
                total_payoff = np.zeros_like(s_range)
                components = []
                for i, pos in enumerate(st.session_state.strategy):
                    info = PAYOFF_REGISTRY[pos["type"]]
                    payoff = info["func"](s_range, **pos["params"])
                    total_payoff += payoff
                    components.append(
                        {"name": f"Position {i + 1}: {pos['type']}", "payoff": payoff}
                    )
                st.plotly_chart(
                    plot_payoff_diagram(s_range, total_payoff, components),
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Error calculating payoffs: {e}")
                st.info("Please check your position parameters.")


def render_iv_surface_tab() -> None:
    """Tab 5: SVI implied-volatility slice fitter."""
    section_header(
        "Implied Volatility Surface",
        "Fit Gatheral raw-SVI to a market vol slice. Stable across strikes and "
        "arbitrage-aware.",
    )
    col_in, col_plot = st.columns([1, 2], gap="medium")

    with col_in:
        with st.container(border=True):
            st.markdown("##### Market Slice")
            f0 = st.number_input("Forward (F)", value=100.0, min_value=0.01)
            t = st.number_input(
                "Maturity (years)", value=1.0, min_value=0.01, step=0.05
            )
            default = (
                "60, 0.30\n70, 0.26\n80, 0.23\n90, 0.21\n"
                "100, 0.20\n110, 0.21\n120, 0.23\n130, 0.27\n140, 0.32"
            )
            csv = st.text_area(
                "Strikes, IVs (one pair per line)",
                value=default,
                height=240,
            )
            fit_clicked = st.button("Fit SVI", use_container_width=True)

    if fit_clicked:
        try:
            rows = [
                [float(x.strip()) for x in line.split(",")]
                for line in csv.strip().splitlines()
                if line.strip()
            ]
            ks = np.array([r[0] for r in rows], dtype=float)
            ivs = np.array([r[1] for r in rows], dtype=float)
            st.session_state.svi_fit = fit_svi_slice(ks, ivs, f0, t)
        except Exception as exc:
            st.error(f"Failed to parse market slice: {exc}")
            st.session_state.svi_fit = None

    with col_plot:
        fit = st.session_state.get("svi_fit")
        if fit is None:
            st.info("Enter a market slice and click *Fit SVI* to start.")
            return

        market_ks = fit["strikes"]
        market_ivs = fit["market_iv"]
        eval_ks = np.linspace(market_ks.min(), market_ks.max(), 200)
        _, model_ivs = fit["evaluator"](eval_ks)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=market_ks,
                y=market_ivs,
                mode="markers",
                name="Market IV",
                marker={"size": 9, "color": "#FFB74D"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=eval_ks,
                y=model_ivs,
                mode="lines",
                name="SVI fit",
                line={"color": "#00D4AA", "width": 3},
            )
        )
        fig.update_layout(
            title=f"SVI Slice fit (T = {fit['t']:.2f})",
            xaxis_title="Strike",
            yaxis_title="Implied Volatility",
            template="plotly_dark",
            hovermode="x unified",
            margin={"l": 40, "r": 20, "t": 60, "b": 40},
        )
        st.plotly_chart(fig, use_container_width=True)

        params = fit["params"]
        stat_strip(
            {
                "a": params["a"],
                "b": params["b"],
                "rho": params["rho"],
                "m": params["m"],
                "sigma": params["sigma"],
                "Residual": fit["residual_norm"],
            },
            fmt="{:,.5f}",
        )


render_hero()
inputs = get_sidebar_inputs()

tab_names = [
    "Point Pricing",
    "Price Surface",
    "Greeks",
    "Convergence",
    "Strategy",
    "IV Surface",
]
tabs = st.tabs(tab_names)

with tabs[0]:
    render_point_pricing_tab(inputs)
with tabs[1]:
    if inputs.get("option_type") in ["Basket", "Asian"]:
        st.info("Price surface analysis is not applicable for this option type.")
    else:
        render_surface_tab(inputs)
with tabs[2]:
    if inputs.get("option_type") in ["Basket", "Asian"]:
        st.info("Greek sensitivity is not applicable for this option type.")
    else:
        render_greeks_tab(inputs)
with tabs[3]:
    render_convergence_tab(inputs)
with tabs[4]:
    render_strategy_tab()
with tabs[5]:
    render_iv_surface_tab()
