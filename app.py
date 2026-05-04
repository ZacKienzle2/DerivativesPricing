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

from services import (
    ANALYTICAL_PRICERS,
    BATES_PRESETS,
    HESTON_PRESETS,
    PAYOFF_REGISTRY,
    PROCESS_LAB_PRESETS,
    aggregate_portfolio_greeks,
    fit_heston_to_quotes,
    fit_svi_slice,
    generate_synthetic_quotes,
    get_greek_data,
    get_point_pricing_context,
    get_surface_data,
    simulate_process_paths,
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
        '<h1 class="hero-title">Derivatives Pricer</h1>',
        unsafe_allow_html=True,
    )


def render_point_pricing_tab(inputs: Dict[str, Any]) -> None:
    """Tab 0: contract pricing + Greeks summary."""
    section_header("Contract Pricing & Greeks")
    metrics = get_point_pricing_context(inputs)
    if "error" in metrics:
        st.error(metrics["error"])
        return
    if not metrics:
        st.warning("Check parameters")
        return

    st.session_state.call_pricer = metrics.get("call", {}).get("pricer")
    st.session_state.put_pricer = metrics.get("put", {}).get("pricer")

    inputs_summary = {
        "Spot": inputs["contract_params"].get("s"),
        "Strike": inputs["contract_params"].get("k"),
        "Maturity": inputs["contract_params"].get("t"),
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
        st.error("Pick different axes")
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
        st.info("Numerical methods only")
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
                st.info("Add an instrument")
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
                st.info("Check position parameters")


def render_iv_surface_tab() -> None:
    """Tab 5: SVI implied-volatility slice fitter."""
    section_header("Implied Volatility")
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
            st.info("Enter a slice and fit")
            return

        from models.calibration import SVICalibrator

        market_ks = fit["strikes"]
        market_ivs = fit["market_iv"]
        eval_ks = np.linspace(market_ks.min(), market_ks.max(), 200)
        _, model_ivs = SVICalibrator.evaluate(
            fit["params"], eval_ks, fit["f0"], fit["t"]
        )

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


def render_process_lab_tab() -> None:
    """Tab 6: pick a process, simulate paths, plot path samples + density."""
    section_header("Process Lab")
    col_in, col_plot = st.columns([1, 2], gap="medium")
    with col_in:
        with st.container(border=True):
            st.markdown("##### Process")
            process_name = st.selectbox(
                "Dynamics",
                list(PROCESS_LAB_PRESETS.keys()),
                key="proc_lab_name",
            )
            base_params = dict(PROCESS_LAB_PRESETS[process_name])
            num_paths = st.slider("Paths", 16, 512, 128, step=16)
            num_steps = st.slider("Time steps", 32, 1024, 252, step=16)
            t = st.number_input("Maturity (years)", 0.1, 10.0, 1.0, step=0.1)
            with st.expander("Parameters", expanded=True):
                for key in base_params:
                    base_params[key] = st.number_input(
                        key, value=float(base_params[key]),
                        format="%.4f",
                        key=f"proc_lab_{key}",
                    )
            seed = st.number_input("Seed", value=42, step=1, key="proc_lab_seed")
            run = st.button("Simulate paths", use_container_width=True)

    if run:
        st.session_state["proc_lab_data"] = simulate_process_paths(
            process_name, base_params, num_paths, num_steps, t, int(seed)
        )

    data = st.session_state.get("proc_lab_data")
    with col_plot:
        if data is None:
            st.info("Configure and simulate")
            return
        with st.container(border=True):
            paths = data["paths"]
            times = data["times"]
            terminal = data["terminal"]
            display_paths = paths[: min(32, paths.shape[0])]
            fig = go.Figure()
            for i in range(display_paths.shape[0]):
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=display_paths[i],
                        mode="lines",
                        showlegend=False,
                        line={"width": 1, "color": "rgba(0, 212, 170, 0.35)"},
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=display_paths.mean(axis=0),
                    mode="lines",
                    name="Mean path",
                    line={"color": "#FFB74D", "width": 3},
                )
            )
            fig.update_layout(
                title="Sample paths",
                xaxis_title="t",
                yaxis_title="S",
                template="plotly_dark",
                margin={"l": 40, "r": 20, "t": 60, "b": 40},
            )
            st.plotly_chart(fig, use_container_width=True)

            density = go.Figure()
            density.add_trace(
                go.Histogram(
                    x=terminal,
                    nbinsx=60,
                    marker_color="rgba(30, 136, 229, 0.6)",
                    name="Terminal density",
                )
            )
            density.update_layout(
                title=f"Terminal distribution (T = {t:.2f})",
                template="plotly_dark",
                xaxis_title="S_T",
                yaxis_title="Frequency",
                margin={"l": 40, "r": 20, "t": 60, "b": 40},
                bargap=0.04,
            )
            st.plotly_chart(density, use_container_width=True)
            stat_strip(
                {
                    "Mean S_T": float(terminal.mean()),
                    "Std S_T": float(terminal.std()),
                    "Min": float(terminal.min()),
                    "Max": float(terminal.max()),
                    "Skew": float(((terminal - terminal.mean()) ** 3).mean()
                                  / max(terminal.std() ** 3, 1e-12)),
                }
            )


def _parse_grid_csv(text: str) -> np.ndarray:
    """Parses a CSV-or-newline-separated number list."""
    cleaned = text.replace("\n", ",").replace(";", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    return np.array([float(p) for p in parts], dtype=float)


def render_heston_calibration_tab() -> None:
    """Tab 7: synthetic quote generator + Heston COS calibrator.

    Lets the user pick the truth model (Heston or Bates), tune any preset
    or hand-set every parameter, and specify either an auto-generated or
    user-supplied strike / maturity vector. Quotes are perturbed with a
    bid/ask half-spread before the calibrator recovers Heston parameters.
    """
    section_header("Heston Calibration")
    col_in, col_plot = st.columns([1, 2], gap="medium")
    with col_in:
        with st.container(border=True):
            st.markdown("##### Truth Model")
            truth_model = st.radio(
                "Dynamics",
                ["Heston", "Bates"],
                horizontal=True,
                key="hcal_truth_model",
            )
            preset_map = (
                HESTON_PRESETS if truth_model == "Heston" else BATES_PRESETS
            )
            preset_name = st.selectbox(
                "Preset",
                list(preset_map.keys()) + ["Custom"],
                key="hcal_preset",
            )
            if preset_name != "Custom":
                base = dict(preset_map[preset_name])
            else:
                base = dict(next(iter(preset_map.values())))

            s0 = st.number_input("Spot", value=100.0, key="hcal_s0")
            r = st.number_input("Rate", value=0.05, step=0.01, key="hcal_r")
            q = st.number_input("Dividend yield", value=0.0, step=0.005, key="hcal_q")
            kappa = st.number_input(
                "kappa (mean reversion)",
                value=float(base["kappa"]),
                step=0.1,
                key="hcal_kappa",
            )
            theta = st.number_input(
                "theta (long-run var)",
                value=float(base["theta"]),
                step=0.005,
                format="%.4f",
                key="hcal_theta",
            )
            eta = st.number_input(
                "eta (vol of vol)",
                value=float(base["eta"]),
                step=0.05,
                key="hcal_eta",
            )
            rho = st.slider(
                "rho (corr)", -0.99, 0.99, float(base["rho"]), 0.01, key="hcal_rho"
            )
            v0 = st.number_input(
                "v0 (initial var)",
                value=float(base["v0"]),
                step=0.005,
                format="%.4f",
                key="hcal_v0",
            )
            if truth_model == "Bates":
                lam = st.number_input(
                    "lambda (jump intensity)",
                    value=float(base.get("lam", 0.3)),
                    step=0.1,
                    key="hcal_lam",
                )
                mu_j = st.number_input(
                    "mu_J (jump mean)",
                    value=float(base.get("mu_j", -0.05)),
                    step=0.01,
                    format="%.3f",
                    key="hcal_muj",
                )
                sigma_j = st.number_input(
                    "sigma_J (jump std)",
                    value=float(base.get("sigma_j", 0.15)),
                    step=0.01,
                    format="%.3f",
                    key="hcal_sigj",
                )

        with st.container(border=True):
            st.markdown("##### Strike & Maturity Grid")
            grid_mode = st.radio(
                "Grid mode",
                ["Auto", "Custom"],
                horizontal=True,
                key="hcal_grid_mode",
            )
            if grid_mode == "Auto":
                n_strikes = st.slider("Strike count", 5, 30, 9, key="hcal_nk")
                strike_pct = st.slider(
                    "Strike width (% spot)", 5, 80, 30, step=5, key="hcal_kpct"
                )
                n_mats = st.slider("Maturity count", 2, 12, 4, key="hcal_nt")
                mat_lo, mat_hi = st.slider(
                    "Maturity range (years)",
                    0.05, 5.0, (0.25, 2.0),
                    key="hcal_trange",
                )
                strikes = np.linspace(
                    s0 * (1.0 - strike_pct / 100.0),
                    s0 * (1.0 + strike_pct / 100.0),
                    int(n_strikes),
                )
                maturities = np.linspace(float(mat_lo), float(mat_hi), int(n_mats))
            else:
                k_text = st.text_area(
                    "Strikes (CSV / newline)",
                    "70, 80, 90, 95, 100, 105, 110, 120, 130",
                    key="hcal_kcsv",
                    height=80,
                )
                t_text = st.text_area(
                    "Maturities in years (CSV / newline)",
                    "0.25, 0.5, 1.0, 2.0",
                    key="hcal_tcsv",
                    height=80,
                )
                try:
                    strikes = _parse_grid_csv(k_text)
                    maturities = _parse_grid_csv(t_text)
                except ValueError as exc:
                    st.error(f"Failed to parse grid: {exc}")
                    return

            spread_bps = st.slider(
                "Bid/ask half-spread (bps)", 0, 300, 40, key="hcal_spread"
            )
            weights_mode = st.selectbox(
                "Calibrator weights",
                ["vega_spread", "vega", "spread", "uniform"],
                key="hcal_weights",
            )
            seed = st.number_input("Seed", value=7, step=1, key="hcal_seed")
            run = st.button("Generate & Calibrate", use_container_width=True)

    if run:
        truth = {
            "s0": s0, "v0": v0, "r": r, "q": q,
            "kappa": kappa, "theta": theta, "eta": eta, "rho": rho,
        }
        if truth_model == "Bates":
            truth.update({"lam": lam, "mu_j": mu_j, "sigma_j": sigma_j})
        try:
            quotes = generate_synthetic_quotes(
                truth_model, truth, strikes, maturities,
                spread_bps=float(spread_bps), seed=int(seed),
            )
        except Exception as exc:
            st.error(f"Quote generation failed: {exc}")
            return
        try:
            mode = (
                None
                if weights_mode == "uniform"
                else weights_mode
            )
            fit = fit_heston_to_quotes(
                quotes, s0, r=r, q=q,
                weights_mode=mode if mode else "vega_spread",
            )
        except Exception as exc:
            st.error(f"Calibration failed: {exc}")
            return
        st.session_state["hcal_data"] = {
            "quotes": quotes, "fit": fit, "truth": truth, "model": truth_model,
        }

    bundle = st.session_state.get("hcal_data")
    with col_plot:
        if bundle is None:
            st.info("Configure truth and calibrate")
            return
        quotes = bundle["quotes"]
        fit = bundle["fit"]
        truth = bundle["truth"]
        truth_model = bundle.get("model", "Heston")
        view = st.radio(
            "View",
            ["Prices", "Implied Vols", "Residuals"],
            horizontal=True,
            key="hcal_view",
        )
        with st.container(border=True):
            if view == "Prices":
                z = quotes["prices"]
                title = f"{truth_model} synthetic market prices"
                colorscale = "Viridis"
                colorbar_title = "Price"
                zmid = None
            elif view == "Implied Vols":
                z = quotes["ivs"]
                title = f"{truth_model} synthetic IV surface"
                colorscale = "Plasma"
                colorbar_title = "IV"
                zmid = None
            else:
                z = fit["model_prices"] - quotes["prices"]
                title = "Calibration residuals (model minus market)"
                colorscale = "RdBu"
                colorbar_title = "Δ"
                zmid = 0

            heatmap = go.Figure(
                data=go.Heatmap(
                    z=z,
                    x=quotes["maturities"],
                    y=quotes["strikes"],
                    colorscale=colorscale,
                    zmid=zmid,
                    colorbar={"title": colorbar_title},
                )
            )
            heatmap.update_layout(
                title=title,
                xaxis_title="Maturity (years)",
                yaxis_title="Strike",
                template="plotly_dark",
                margin={"l": 40, "r": 20, "t": 60, "b": 40},
            )
            st.plotly_chart(heatmap, use_container_width=True)

            iv_curves = go.Figure()
            for j, t_val in enumerate(quotes["maturities"]):
                iv_curves.add_trace(
                    go.Scatter(
                        x=quotes["strikes"],
                        y=quotes["ivs"][:, j],
                        mode="lines+markers",
                        name=f"T = {t_val:.2f}",
                    )
                )
            iv_curves.update_layout(
                title="IV smiles by maturity",
                xaxis_title="Strike",
                yaxis_title="Implied vol",
                template="plotly_dark",
                margin={"l": 40, "r": 20, "t": 60, "b": 40},
                legend={"orientation": "h", "y": -0.2},
            )
            st.plotly_chart(iv_curves, use_container_width=True)

            comparison_rows = []
            for key in ("kappa", "theta", "eta", "rho", "v0"):
                comparison_rows.append(
                    (
                        key,
                        f"{fit['params'][key]:.4f}",
                        f"{truth.get(key, 0.0):.4f}",
                        f"{(fit['params'][key] - truth.get(key, 0.0)):+.4e}",
                    )
                )
            import pandas as pd

            df = pd.DataFrame(
                comparison_rows, columns=["Param", "Fit", "Truth", "Δ"]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
            stat_strip(
                {
                    "Residual": f"{fit['residual_norm']:.2e}",
                    "Iterations": str(fit["n_iter"]),
                    "Converged": "yes" if fit["converged"] else "no",
                    "Quotes fit": str(quotes["prices"].size),
                }
            )


def render_risk_dashboard_tab() -> None:
    """Tab 8: portfolio of vanilla options, aggregate Greeks."""
    section_header("Risk Dashboard")
    import pandas as pd

    default = pd.DataFrame(
        [
            {"Type": "call", "S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05,
             "q": 0.0, "sigma": 0.2, "Quantity": 10},
            {"Type": "put", "S": 100.0, "K": 95.0, "T": 1.0, "r": 0.05,
             "q": 0.0, "sigma": 0.22, "Quantity": -5},
            {"Type": "call", "S": 100.0, "K": 110.0, "T": 0.5, "r": 0.05,
             "q": 0.0, "sigma": 0.18, "Quantity": 3},
        ]
    )
    edited = st.data_editor(
        default,
        num_rows="dynamic",
        use_container_width=True,
        key="risk_positions",
        column_config={
            "Type": st.column_config.SelectboxColumn(
                "Type", options=["call", "put"], required=True
            ),
            "S": st.column_config.NumberColumn("Spot", format="%.2f"),
            "K": st.column_config.NumberColumn("Strike", format="%.2f"),
            "T": st.column_config.NumberColumn("Maturity", format="%.2f"),
            "r": st.column_config.NumberColumn("Rate", format="%.4f"),
            "q": st.column_config.NumberColumn("Div", format="%.4f"),
            "sigma": st.column_config.NumberColumn("Vol", format="%.4f"),
            "Quantity": st.column_config.NumberColumn("Qty", format="%d"),
        },
    )

    positions = [
        {
            "option_type": str(row["Type"]),
            "s": float(row["S"]),
            "k": float(row["K"]),
            "t": float(row["T"]),
            "r": float(row["r"]),
            "q": float(row.get("q", 0.0)),
            "sigma": float(row["sigma"]),
            "quantity": float(row["Quantity"]),
        }
        for _, row in edited.iterrows()
        if pd.notna(row["S"])
    ]
    if not positions:
        st.info("Add a position")
        return
    agg = aggregate_portfolio_greeks(positions)
    stat_strip({k.title(): v for k, v in agg.items()}, fmt="{:,.4f}")

    s_centre = float(np.mean([p["s"] for p in positions]))
    s_grid = np.linspace(s_centre * 0.5, s_centre * 1.5, 200)
    pl = np.zeros_like(s_grid)
    for pos in positions:
        if pos["option_type"] == "call":
            payoff = np.maximum(s_grid - pos["k"], 0.0)
        else:
            payoff = np.maximum(pos["k"] - s_grid, 0.0)
        pl += pos["quantity"] * payoff

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s_grid,
            y=pl,
            mode="lines",
            line={"color": "#00D4AA", "width": 3},
            name="Portfolio payoff",
        )
    )
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)")
    fig.update_layout(
        title="Portfolio payoff at expiry (per-leg max-T)",
        xaxis_title="Underlying at expiry",
        yaxis_title="P/L",
        template="plotly_dark",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


render_hero()
inputs = get_sidebar_inputs()

tab_names = [
    "Pricing",
    "Surface",
    "Greeks",
    "Convergence",
    "Strategy",
    "IV",
    "Process",
    "Calibration",
    "Risk",
]
tabs = st.tabs(tab_names)

with tabs[0]:
    render_point_pricing_tab(inputs)
with tabs[1]:
    if inputs.get("option_type") in ["Basket", "Asian"]:
        st.info("Surface not available for this option")
    else:
        render_surface_tab(inputs)
with tabs[2]:
    if inputs.get("option_type") in ["Basket", "Asian"]:
        st.info("Greeks not available for this option")
    else:
        render_greeks_tab(inputs)
with tabs[3]:
    render_convergence_tab(inputs)
with tabs[4]:
    render_strategy_tab()
with tabs[5]:
    render_iv_surface_tab()
with tabs[6]:
    render_process_lab_tab()
with tabs[7]:
    render_heston_calibration_tab()
with tabs[8]:
    render_risk_dashboard_tab()
