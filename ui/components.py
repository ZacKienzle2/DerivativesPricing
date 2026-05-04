"""Reusable Streamlit UI components.

Card-based metric display, stat strips and section headers. All components
emit pure HTML so the theme `style.css` can drive presentation centrally
without per-call inline styling.
"""

from collections.abc import Iterable
from typing import Any

import pandas as pd
import streamlit as st


def _format(value: Any, fmt: str = "{:,.4f}") -> str:
    """Formats numeric values, falling back gracefully on non-numerics."""
    if value is None:
        return "n/a"
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def section_header(title: str, subtitle: str | None = None) -> None:
    """Renders a left aligned section heading with optional subtitle."""
    sub = (
        f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    )
    st.markdown(
        f'<h2 class="section-title">{title}</h2>{sub}',
        unsafe_allow_html=True,
    )


def stat_strip(stats: dict[str, Any], fmt: str = "{:,.4f}") -> None:
    """Renders a horizontal strip of `(label, value)` stat cells."""
    cells = "".join(
        f'<div class="stat-cell">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{_format(value, fmt)}</div>'
        f'</div>'
        for label, value in stats.items()
    )
    st.markdown(f'<div class="stat-strip">{cells}</div>', unsafe_allow_html=True)


def greek_pills(greeks: dict[str, Any]) -> str:
    """Returns HTML for the compact Greek pill grid embedded inside a card."""
    if not greeks or "info" in greeks:
        return ""
    pills = "".join(
        f'<div class="greek-pill">'
        f'<span class="greek-label">{name}</span>'
        f'<span class="greek-value">{_format(value, "{:,.4f}")}</span>'
        f'</div>'
        for name, value in greeks.items()
        if name not in ("info",)
    )
    return f'<div class="greek-grid">{pills}</div>'


def display_metrics_card(
    name: str,
    metrics: dict[str, Any],
    option_type: str,
    analytical_pricers: set,
) -> None:
    """Renders a price + Greeks card for a single option flavour.

    Args:
        name: `"Call"` or `"Put"`.
        metrics: Dict carrying `price`, optional `std_err`, `greeks`, `pricer`.
        option_type: Display name of the option family (e.g. `"American"`).
        analytical_pricers: Set of pricer classes treated as analytical for
            the method tag.
    """
    if not metrics:
        return

    price_html = f"${metrics['price']:,.4f}"
    se = metrics.get("std_err", 0.0)
    se_html = (
        f'<div class="std-err">&plusmn; {se:,.4f} std err</div>'
        if se and se > 0
        else ""
    )

    title = f"{option_type.replace('European', '').strip()} {name.capitalize()}"
    flavour = name.lower()
    greeks_html = greek_pills(metrics.get("greeks", {}))
    method_tag = (
        '<span class="tag">analytical</span>'
        if type(metrics.get("pricer")) in analytical_pricers
        else '<span class="tag tag-warn">numerical</span>'
    )

    html = (
        f'<div class="metric-card metric-card-{flavour}">'
        f'<div class="metric-head"><h3>{title} Price</h3>{method_tag}</div>'
        f'<h1>{price_html}</h1>{se_html}{greeks_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    extras = metrics.get("extras")
    if extras:
        with st.expander(f"{name.capitalize()} diagnostics", expanded=False):
            df = pd.DataFrame(list(extras.items()), columns=["Field", "Value"])
            st.dataframe(df, use_container_width=True, hide_index=True)


def render_kv_table(
    title: str,
    items: Iterable,
    formatter: str = "{:,.6f}",
) -> None:
    """Renders a labelled key/value DataFrame inside a bordered container."""
    with st.container(border=True):
        st.markdown(f"##### {title}")
        df = pd.DataFrame(list(items), columns=["Field", "Value"])
        df["Value"] = df["Value"].map(lambda v: _format(v, formatter))
        st.dataframe(df, use_container_width=True, hide_index=True)
