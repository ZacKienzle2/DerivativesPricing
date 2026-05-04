"""Plotly figure helpers used by the Streamlit dashboard."""

from typing import Any

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go


def plot_mc_convergence(prices: npt.NDArray[np.float64], title: str) -> go.Figure:
    """Creates a Plotly figure showing Monte Carlo estimate convergence."""
    fig = go.Figure()
    sim_counts = np.arange(1, len(prices) + 1)
    fig.add_trace(
        go.Scatter(x=sim_counts, y=prices, mode="lines", name="Estimated Price")
    )

    final_price = prices[-1]
    fig.add_hline(
        y=final_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Final Price: {final_price:.4f}",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Number of Simulations",
        yaxis_title="Option Price",
        template="plotly_dark",
    )
    return fig


def plot_3d_surface(
    x_range: np.ndarray,
    y_range: np.ndarray,
    z_surface: np.ndarray,
    x_title: str,
    y_title: str,
    title: str,
) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Surface(
                z=z_surface.T,
                x=x_range,
                y=y_range,
                colorscale="Viridis",
                colorbar={"title": "Price"},
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_title, yaxis_title=y_title, zaxis_title="Option Price ($)"
        ),
        height=700,
        template="plotly_dark",
        margin=dict(l=65, r=50, b=65, t=90),
    )
    return fig


def plot_greek_sensitivity(
    s_range: npt.NDArray[np.float64], greek_data: dict[str, list[float]], title: str
) -> go.Figure:
    """Creates a line plot showing greek sensitivity to stock price."""
    fig = go.Figure()
    for greek, values in greek_data.items():
        if "info" not in greek:
            fig.add_trace(
                go.Scatter(x=s_range, y=values, mode="lines", name=greek.capitalize())
            )

    fig.update_layout(
        title=title,
        xaxis_title="Underlying Price ($)",
        yaxis_title="Greek Value",
        legend_title="Greeks",
        template="plotly_dark",
    )
    return fig


def plot_payoff_diagram(
    s_range: npt.NDArray[np.float64],
    total_payoff: npt.NDArray[np.float64],
    components: list[dict[str, Any]],
) -> go.Figure:
    """
    Creates an interactive payoff diagram for a financial strategy.

    Args:
        s_range: Array of underlying prices for the x-axis.
        total_payoff: Combined payoff of all positions.
        components: A list of dicts, each with a name and payoff array.

    Returns:
        A Plotly figure object.
    """
    fig = go.Figure()

    # plot individual components
    for comp in components:
        fig.add_trace(
            go.Scatter(
                x=s_range,
                y=comp["payoff"],
                mode="lines",
                name=comp["name"],
                line=dict(dash="dot", width=1.5),
            )
        )

    # plot total strategy payoff
    fig.add_trace(
        go.Scatter(
            x=s_range,
            y=total_payoff,
            mode="lines",
            name="Total Strategy",
            line=dict(color="#00C2FF", width=3),
        )
    )

    # add a horizontal line at p/l = 0
    fig.add_hline(y=0, line_dash="dash", line_color="grey")

    fig.update_layout(
        title="Strategy Payoff Diagram at Expiration",
        xaxis_title="Underlying Price ($)",
        yaxis_title="Profit / Loss ($)",
        legend_title="Positions",
        template="plotly_dark",
        height=500,
    )
    return fig
