import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any

from ui.sidebar import get_sidebar_inputs
from ui.components import display_metrics_card
from utils.plotting import plot_3d_surface, plot_greek_sensitivity, plot_payoff_diagram
from controller import (
    get_point_pricing_context,
    get_surface_data,
    get_greek_data,
    PAYOFF_REGISTRY,
    ANALYTICAL_PRICERS,
)


st.set_page_config(
    page_title="Options Pricer", layout="wide", initial_sidebar_state="expanded"
)

# Load CSS
with open("static/style.css") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


for key in ["strategy", "call_pricer", "put_pricer"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "strategy" else None


def render_point_pricing_tab(inputs: Dict[str, Any]):
    st.header("Contract Pricing & Greeks")
    metrics = get_point_pricing_context(inputs)

    if not metrics:
        st.warning("Could not compute metrics. Please check parameters.")
        return

    st.session_state.call_pricer = metrics.get("call", {}).get("pricer")
    st.session_state.put_pricer = metrics.get("put", {}).get("pricer")

    col1, col2 = st.columns(2)
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


def render_surface_tab(inputs: Dict[str, Any]):
    st.header("Price Surface Analysis")
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
        plot_map = [
            ("Call", tab_c, surface_c, "Viridis"),
            ("Put", tab_p, surface_p, "Plasma"),
        ]
        for name, tab, data, cmap in plot_map:
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
                st.plotly_chart(fig, use_container_width=True)


def render_greeks_tab(inputs: Dict[str, Any]):
    st.header("Greek Sensitivity Analysis")
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


def render_convergence_tab(inputs: Dict[str, Any]):
    st.header("Model Convergence Analysis")
    pricer_type = inputs["pricer_type"]
    if pricer_type not in ["Monte Carlo", "Longstaff-Schwartz"]:
        st.info("Convergence plots are only for numerical methods.")
        return

    with st.container(border=True):
        st.markdown(f"Visualising price convergence for the {pricer_type} model.")
        call_pricer = st.session_state.get("call_pricer")
        put_pricer = st.session_state.get("put_pricer")
        fig = go.Figure()
        if (
            call_pricer
            and hasattr(call_pricer, "convergence_data")
            and call_pricer.convergence_data is not None
        ):
            fig.add_trace(
                go.Scatter(
                    y=call_pricer.convergence_data, mode="lines", name="Call Price"
                )
            )
        if (
            put_pricer
            and hasattr(put_pricer, "convergence_data")
            and put_pricer.convergence_data is not None
        ):
            fig.add_trace(
                go.Scatter(
                    y=put_pricer.convergence_data, mode="lines", name="Put Price"
                )
            )
        fig.update_layout(
            title=f"{pricer_type} Convergence",
            xaxis_title="Number of Simulations",
            yaxis_title="Option Price",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_strategy_tab():
    st.header("Strategy Payoff Builder")
    LABEL_MAP = {
        "k": "Strike",
        "premium": "Premium",
        "purchase_price": "Purchase Price",
        "sale_price": "Sale Price",
        "future_value": "Future Value",
        "price": "Price",
    }
    col1, col2 = st.columns([1, 2])
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


st.markdown(
    """<div style="text-align: center; margin-bottom: 2rem;"><h1 style="background: linear-gradient(90deg, #00D4AA 0%, #1E88E5 100%);-webkit-background-clip: text; -webkit-text-fill-color: transparent;font-size: 3rem; font-weight: 700;">Options Pricer</h1></div>""",
    unsafe_allow_html=True,
)

inputs = get_sidebar_inputs()

tab_names = [
    "Point Pricing",
    "Price Surface",
    "Greek Sensitivity",
    "Model Convergence",
    "Strategy Builder",
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
