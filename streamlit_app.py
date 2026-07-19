# streamlit_app.py

import os
import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from data.polygon_client import fetch_data
from backtest.backtester import run_backtest, summary_stats
from models.garch_model import fit_garch, forecast_variance
from reports.pdf_exporter import generate_pdf

# ── Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── Streamlit page config
st.set_page_config(page_title="GARCH Dashboard", layout="wide")
st.title("📈 GARCH Dashboard")

# ── Cached data-fetch and backtest functions
@st.cache_data(show_spinner=False)
def get_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    logger.info(f"get_data: fetching {symbol} {start}→{end}")
    df = fetch_data(symbol, start, end)
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df.dropna(inplace=True)
    logger.info(f"get_data: {len(df)} rows after computing returns")
    return df

@st.cache_data(show_spinner=False)
def get_backtest(returns: pd.Series, window: int, alpha: float, p: int, q: int) -> pd.DataFrame:
    logger.info(f"get_backtest: window={window}, α={alpha}, p={p}, q={q}")
    return run_backtest(returns, window=window, alpha=alpha, p=p, q=q)

# ── Session state for persistent results
if "run_clicked" not in st.session_state:
    st.session_state.run_clicked = False

# ── Sidebar controls
symbol = st.sidebar.text_input("Ticker symbol", "AAPL",
    help="e.g. AAPL, MSFT, BTC-USD")
start_date = st.sidebar.date_input("Start date", date(2020,1,1),
    help="YYYY-MM-DD")
end_date = st.sidebar.date_input("End date", date.today(),
    help="YYYY-MM-DD")
window = st.sidebar.number_input("Backtest window (days)",
    value=250, min_value=1,
    help="Number of past trading days to fit each GARCH model")
alpha = st.sidebar.slider("VaR tail α level", 0.01, 0.10, 0.05,
    help="e.g. 0.05 for 95% VaR")
p_order = st.sidebar.number_input("GARCH p-order",
    value=1, min_value=1, max_value=5,
    help="Auto-regressive volatility terms (p)")
q_order = st.sidebar.number_input("GARCH q-order",
    value=1, min_value=1, max_value=5,
    help="Moving-average volatility terms (q)")

if st.sidebar.button("Run Analysis"):
    st.session_state.run_clicked = True
if st.sidebar.button("Clear Results"):
    st.session_state.run_clicked = False

# ── Main display logic
if st.session_state.run_clicked:
    symbol = symbol.upper()
    s = start_date.isoformat()
    e = end_date.isoformat()

    # 1. Fetch data (cached)
    df = get_data(symbol, s, e)
    if df.empty:
        st.error("No data returned. Check symbol and date range.")
        logger.error("No data from get_data → aborting.")
        st.stop()

    # 2. Clamp window if necessary
    max_win = len(df) - 1
    if window > max_win:
        st.warning(
            f"Window ({window}) > available data ({max_win}). "
            f"Using window={max_win}."
        )
        logger.warning(f"Clamped window {window}→{max_win}")
        window = max_win

    # 3. Run backtest (cached, with spinner)
    with st.spinner("Running backtest…"):
        bt = get_backtest(df["ret"], window, alpha, p_order, q_order)
    if bt.empty:
        st.error("Backtest failed to produce results.")
        logger.error("get_backtest returned empty DataFrame")
        st.stop()
    stats = summary_stats(bt, alpha=alpha)

    # 4. Forecasted Variance chart
    st.subheader("Forecasted Variance (σ²) Over Time")
    fig_vol = px.line(
        bt.reset_index(),
        x="date",
        y="sigma2",
        labels={"sigma2": "Forecasted Variance σ²", "date": "Date"},
        title="Forecasted Variance (σ²)"
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    logger.info("Plotted variance time series")

    # 5. Returns, VaR, ES & breach markers (consistent units)
    st.subheader("Returns with VaR/ES Thresholds")
    df2 = bt.reset_index().copy()
    df2["date"] = pd.to_datetime(df2["date"])

    fig_pv = go.Figure()
    fig_pv.add_trace(go.Scatter(
        x=df2["date"], y=df2["ret"],
        mode="lines", name="Daily Return"
    ))
    fig_pv.add_trace(go.Scatter(
        x=df2["date"], y=df2["var"],
        mode="lines", name=f"VaR ({int((1-alpha)*100)}%)"
    ))
    fig_pv.add_trace(go.Scatter(
        x=df2["date"], y=df2["es"],
        mode="lines", name=f"ES ({int((1-alpha)*100)}%)"
    ))

    breaches = df2[df2["breach"]]
    fig_pv.add_trace(go.Scatter(
        x=breaches["date"], y=breaches["ret"],
        mode="markers", marker=dict(color="red", size=8),
        name="VaR Breach"
    ))

    fig_pv.update_layout(
        title="Return-Space Backtest: Realized Returns vs VaR/ES",
        xaxis_title="Date", yaxis_title="Return"
    )
    st.plotly_chart(fig_pv, use_container_width=True)
    logger.info("Plotted returns with VaR/ES overlay")

    # 6. Hit-rate comparison
    st.subheader("Actual vs. Theoretical Breach Rate")
    hit_actual = stats["hit_rate"] * 100
    hit_theoretical = alpha * 100
    hr_df = pd.DataFrame({
        "Type": ["Actual Breach Rate", "Theoretical Rate"],
        "Percent": [hit_actual, hit_theoretical]
    })
    fig_hr = px.bar(
        hr_df, x="Type", y="Percent",
        labels={"Percent": "Breach Rate (%)"},
        title="Actual vs. Theoretical Breach Rate"
    )
    st.plotly_chart(fig_hr, use_container_width=True)
    logger.info("Plotted hit-rate comparison")

    # 7. Summary & CSV download
    st.markdown(
        f"**Total days tested:** {stats['total_days']}  \n"
        f"**Number of breaches:** {stats['breaches']} (expected: {stats['expected_breaches']:.1f})  \n"
        f"**Actual hit rate:** {hit_actual:.2f}%  \n"
        f"**95% CI for expected hit rate:** [{stats['hit_rate_ci_low']*100:.2f}%, {stats['hit_rate_ci_high']*100:.2f}%]  \n"
        f"**Kupiec UC p-value:** {stats['kupiec_pvalue']:.4f}"
    )
    csv_data = bt.to_csv(index=True)
    st.download_button(
        "Download backtest CSV",
        data=csv_data,
        file_name=f"{symbol}_backtest.csv",
        mime="text/csv"
    )
    logger.info("Offered CSV download")

    # 8. PDF export
    pdf_path = generate_pdf(symbol, start_date, end_date, stats)
    st.success(f"PDF report saved: {pdf_path}")
    logger.info(f"Generated PDF report at {pdf_path}")

else:
    st.info("Configure your parameters in the sidebar and click **Run Analysis** to begin.")
