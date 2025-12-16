import streamlit as st
import pandas as pd
from typing import Dict, Any, Set


def display_metrics_card(
    name: str, metrics: Dict[str, Any], option_type: str, analytical_pricers: Set
):
    if not metrics:
        return

    price_str = f"${metrics['price']:,.4f}"
    if metrics.get("std_err", 0) > 0:
        price_str += f" (± {metrics['std_err']:,.4f})"

    color = "#00D4AA" if name.lower() == "call" else "#FF4B4B"
    title = f"{option_type.replace('European', '').strip()} {name.capitalize()} Price"

    st.markdown(
        f"""<div class="metric-card metric-card-{name.lower()}">
        <h3>{title}</h3><h1>{price_str}</h1></div>""",
        unsafe_allow_html=True,
    )

    greeks = metrics.get("greeks", {})
    if not greeks or "info" in greeks:
        return

    method = (
        "Analytical" if type(metrics["pricer"]) in analytical_pricers else "Numerical"
    )

    with st.expander(f"View {name.capitalize()} Greeks ({method})", expanded=True):
        df = pd.DataFrame([greeks]).T.rename(columns={0: "Value"})
        st.dataframe(df.style.format("{:.5f}", na_rep="N/A"), use_container_width=True)
