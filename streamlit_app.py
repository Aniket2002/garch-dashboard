# streamlit_app.py

import os
import time
import logging
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from data.polygon_client import fetch_data
from backtest.backtester import run_backtest, summary_stats
from models.garch_model import fit_garch, forecast_variance
from weasyprint import HTML
from jinja2 import Environment, FileSystemLoader
import base64
import plotly.express as px

# ── Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── Streamlit page config
st.set_page_config(page_title="GARCH Dashboard", layout="wide")
st.title("📈 GARCH Dashboard")

# ── Sidebar inputs with tooltips
symbol = st.sidebar.text_input(
    "Ticker symbol", "AAPL",
    help="e.g. AAPL, MSFT, BTC-USD"
)
start = st.sidebar.date_input(
    "Start date", date(2020, 1, 1),
    help="YYYY-MM-DD"
)
end = st.sidebar.date_input(
    "End date", date.today(),
    help="YYYY-MM-DD"
)
window = st.sidebar.number_input(
    "Backtest window (days)",
    value=250, min_value=1,
    help="Number of past trading days to fit each GARCH model"
)
alpha = st.sidebar.slider(
    "VaR α level", 0.01, 0.10, 0.05,
    help="Tail probability for VaR computation"
)

if st.sidebar.button("Run Analysis"):
    logger.debug(f"Run Analysis clicked: symbol={symbol}, start={start}, end={end}, window={window}, alpha={alpha}")

    # 1. Fetch & preprocess
    df = fetch_data(symbol, start.isoformat(), end.isoformat())
    logger.debug(f"Fetched data: {len(df)} rows")
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df.dropna(inplace=True)
    logger.debug(f"Computed returns, remaining {len(df)} rows")

    # ── Auto-adjust window if too large
    max_window = len(df) - 1
    if window > max_window:
        st.warning(
            f"Backtest window ({window}) is larger than available data points "
            f"({max_window}). Using window={max_window} instead."
        )
        logger.warning(
            f"Requested window={window} > data length; auto-clamped to {max_window}"
        )
        window = max_window

    # 2. Run backtest
    bt = run_backtest(df["ret"], window=window, alpha=alpha)
    if bt.empty:
        st.warning(
            "Not enough data to run the backtest. "
            "Try a longer date range or a smaller window."
        )
        logger.warning("Streamlit: backtest returned empty DataFrame, skipping plots.")
    else:
        stats = summary_stats(bt)

        # 3. Volatility time series
        st.subheader("Forecasted Volatility (σₜ²)")
        fig_vol = px.line(
            bt.reset_index(),
            x="date",
            y="sigma2",
            labels={"sigma2": "σ²", "date": "Date"},
            title="σ² over Time"
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        logger.debug("Streamlit: plotted volatility series")

        # 4. Price & VaR overlay (long-form)
        price_with_var = df[["close"]].join(bt["var"]).dropna().reset_index()
        price_with_var["date"] = pd.to_datetime(price_with_var["date"])
        pv_long = price_with_var.melt(
            id_vars="date",
            value_vars=["close", "var"],
            var_name="Series",
            value_name="Value"
        )
        st.subheader("Price & VaR Overlay")
        fig_pv = px.line(
            pv_long,
            x="date",
            y="Value",
            color="Series",
            labels={"Value": "Price / VaR", "date": "Date", "Series": ""},
            title="Close Price vs. VaR"
        )
        st.plotly_chart(fig_pv, use_container_width=True)
        logger.debug("Streamlit: plotted price & VaR overlay (long-form)")

        # 5. Display summary stats
        st.markdown(
            f"**Total days:** {stats['total_days']}  \n"
            f"**Breaches:** {stats['breaches']}  \n"
            f"**Hit rate:** {stats['hit_rate']:.2%}"
        )
        logger.debug(f"Streamlit: displayed summary stats {stats}")

        # 6. PDF export
        env = Environment(loader=FileSystemLoader("reports/templates"))
        tmpl = env.get_template("report_template.html")
        html_out = tmpl.render(
            title=f"{symbol} GARCH Backtest",
            start_date=start,
            end_date=end,
            stats=stats,
            plot_volatility="",  # replace with base64 PNG if desired
            plot_var=""
        )
        pdf_path = f"reports/{symbol}_backtest.pdf"
        HTML(string=html_out).write_pdf(pdf_path)
        st.success(f"PDF report saved: {pdf_path}")
        logger.debug(f"Streamlit: generated PDF report at {pdf_path}")
