from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest.backtester import BacktestError, run_backtest, summary_stats
from data.polygon_client import (
    DataProviderError,
    compute_log_returns,
    fetch_data,
    load_price_csv,
)
from models.garch_model import (
    GarchSpec,
    fit_garch,
    forecast_tail_risk,
    model_diagnostics,
)
from reporting.pdf_report import build_pdf_report


def _coverage_label(p_value: float) -> str:
    if not math.isfinite(p_value):
        return "Not available"
    return "Pass at 5%" if p_value >= 0.05 else "Reject at 5%"


def _load_prices(
    source: str,
    *,
    uploaded_file: Any,
    symbol: str,
    start: date,
    end: date,
    api_key: str,
) -> pd.DataFrame:
    if source == "CSV upload":
        if uploaded_file is None:
            raise DataProviderError("Upload a CSV containing date and close columns")
        prices = load_price_csv(uploaded_file)
        date_mask = (prices.index.date >= start) & (prices.index.date <= end)
        prices = prices.loc[date_mask]
        if prices.empty:
            raise DataProviderError("the uploaded CSV has no rows in the selected date range")
        return prices
    return fetch_data(
        symbol,
        start.isoformat(),
        end.isoformat(),
        api_key=api_key or None,
    )


def _risk_figure(backtest: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=backtest.index,
            y=backtest["realized_return"],
            name="Realized return",
            mode="lines",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=backtest.index,
            y=backtest["var"],
            name="VaR threshold",
            mode="lines",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=backtest.index,
            y=backtest["expected_shortfall"],
            name="Expected shortfall",
            mode="lines",
        )
    )
    breaches = backtest.loc[backtest["breach"]]
    figure.add_trace(
        go.Scatter(
            x=breaches.index,
            y=breaches["realized_return"],
            name="VaR breach",
            mode="markers",
            marker={"size": 8, "symbol": "x"},
        )
    )
    figure.update_layout(
        title="One-day return forecasts and realized returns",
        xaxis_title="Date",
        yaxis_title="Decimal return",
        hovermode="x unified",
    )
    return figure


def _volatility_figure(backtest: pd.DataFrame) -> go.Figure:
    annualized = backtest["volatility"] * math.sqrt(252.0)
    figure = go.Figure(
        go.Scatter(
            x=backtest.index,
            y=annualized,
            name="Annualized volatility",
            mode="lines",
        )
    )
    figure.update_layout(
        title="Forecast annualized volatility",
        xaxis_title="Date",
        yaxis_title="Annualized volatility",
        hovermode="x unified",
    )
    return figure


def _price_figure(prices: pd.DataFrame, symbol: str) -> go.Figure:
    figure = go.Figure(
        go.Scatter(x=prices.index, y=prices["close"], name="Close", mode="lines")
    )
    figure.update_layout(
        title=f"{symbol} closing price",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
    )
    return figure


def main() -> None:
    st.set_page_config(page_title="GARCH Risk Dashboard", page_icon="📉", layout="wide")
    st.title("GARCH Risk Dashboard")
    st.caption(
        "Rolling one-day volatility, VaR and expected-shortfall analysis with "
        "strict out-of-sample backtesting."
    )

    with st.sidebar:
        st.header("Market data")
        source = st.radio("Source", ["Polygon", "CSV upload"], horizontal=True)
        symbol = st.text_input("Ticker", value="AAPL").strip().upper()
        default_end = date.today()
        default_start = default_end - timedelta(days=365 * 5)
        start = st.date_input("Start date", value=default_start)
        end = st.date_input("End date", value=default_end)
        api_key = ""
        uploaded_file = None
        if source == "Polygon":
            api_key = st.text_input(
                "Polygon API key",
                type="password",
                help="Optional when POLYGON_API_KEY is set in the environment.",
            )
        else:
            uploaded_file = st.file_uploader(
                "Price CSV",
                type=["csv"],
                help="Requires date and close columns; OHLCV columns are optional.",
            )

        st.header("Risk model")
        model_family = st.selectbox("Volatility model", ["GARCH(1,1)", "GJR-GARCH(1,1)"])
        distribution = st.selectbox("Innovations", ["Student-t", "Normal"])
        mean_model = st.selectbox("Conditional mean", ["Constant", "Zero"])
        window = int(
            st.number_input(
                "Rolling estimation window",
                min_value=100,
                max_value=2000,
                value=500,
                step=50,
            )
        )
        backtest_points = int(
            st.number_input(
                "Out-of-sample forecast days",
                min_value=20,
                max_value=750,
                value=250,
                step=10,
                help="Each forecast refits the model, so larger values take longer.",
            )
        )
        alpha = float(
            st.select_slider(
                "Lower-tail probability",
                options=[0.01, 0.025, 0.05, 0.10],
                value=0.05,
                format_func=lambda value: f"{value:.1%}",
            )
        )
        run_clicked = st.button("Run analysis", type="primary", use_container_width=True)

    if start > end:
        st.error("Start date must not be after end date.")
        return

    if run_clicked:
        spec = GarchSpec(
            p=1,
            o=1 if model_family.startswith("GJR") else 0,
            q=1,
            mean=mean_model,
            distribution="student_t" if distribution == "Student-t" else "normal",
        )
        try:
            with st.spinner("Loading and validating market data..."):
                prices = _load_prices(
                    source,
                    uploaded_file=uploaded_file,
                    symbol=symbol,
                    start=start,
                    end=end,
                    api_key=api_key,
                )
                returns = compute_log_returns(prices)

            minimum_required = window + 1
            if len(returns) < minimum_required:
                raise ValueError(
                    f"{len(returns)} returns are available; at least {minimum_required} "
                    "are needed for this window"
                )

            forecast_count = min(backtest_points, len(returns) - window)
            backtest_input = returns.tail(window + forecast_count)
            progress = st.progress(0.0, text="Running rolling forecasts...")

            def update_progress(completed: int, total: int) -> None:
                progress.progress(completed / total, text=f"Rolling forecast {completed}/{total}")

            backtest = run_backtest(
                backtest_input,
                window=window,
                alpha=alpha,
                spec=spec,
                progress_callback=update_progress,
            )
            progress.empty()
            if backtest.empty:
                raise ValueError("the selected settings produced no out-of-sample forecasts")

            latest_result = fit_garch(returns.tail(window), spec)
            latest_forecast = forecast_tail_risk(latest_result, alpha)
            diagnostics = model_diagnostics(latest_result)
            stats = summary_stats(backtest, alpha)
            pdf_bytes = build_pdf_report(
                title="GARCH VaR and Expected-Shortfall Report",
                symbol=symbol or "Uploaded series",
                start_date=start,
                end_date=end,
                spec=spec,
                alpha=alpha,
                window=window,
                stats=stats,
                latest_forecast=latest_forecast,
                diagnostics=diagnostics,
                backtest=backtest,
            )
            st.session_state["garch_analysis"] = {
                "prices": prices,
                "returns": returns,
                "backtest": backtest,
                "stats": stats,
                "spec": spec,
                "forecast": latest_forecast,
                "diagnostics": diagnostics,
                "parameters": latest_result.params.rename("estimate").to_frame(),
                "pdf": pdf_bytes,
                "symbol": symbol or "Uploaded series",
                "alpha": alpha,
                "window": window,
            }
        except (DataProviderError, BacktestError, ValueError, RuntimeError) as exc:
            st.session_state.pop("garch_analysis", None)
            st.error(str(exc))

    analysis = st.session_state.get("garch_analysis")
    if not analysis:
        st.info("Configure the data source and model, then run the analysis.")
        return

    forecast = analysis["forecast"]
    stats = analysis["stats"]
    diagnostics = analysis["diagnostics"]
    backtest = analysis["backtest"]
    prices = analysis["prices"]

    metric_columns = st.columns(5)
    metric_columns[0].metric("1-day volatility", f"{forecast.volatility:.2%}")
    metric_columns[1].metric(
        "Annualized volatility", f"{forecast.volatility * math.sqrt(252.0):.2%}"
    )
    metric_columns[2].metric("VaR return threshold", f"{forecast.var:.2%}")
    metric_columns[3].metric(
        "Expected shortfall", f"{forecast.expected_shortfall:.2%}"
    )
    metric_columns[4].metric(
        "Breaches",
        f"{stats['breaches']} / {stats['total_days']}",
        delta=f"Expected {stats['expected_breaches']:.1f}",
        delta_color="off",
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(
            _price_figure(prices, analysis["symbol"]), use_container_width=True
        )
    with chart_right:
        st.plotly_chart(_volatility_figure(backtest), use_container_width=True)
    st.plotly_chart(_risk_figure(backtest), use_container_width=True)

    st.subheader("Coverage diagnostics")
    coverage = pd.DataFrame(
        [
            {
                "Test": "Kupiec unconditional coverage",
                "LR statistic": stats["kupiec_lr"],
                "p-value": stats["kupiec_p_value"],
                "5% decision": _coverage_label(stats["kupiec_p_value"]),
            },
            {
                "Test": "Christoffersen independence",
                "LR statistic": stats["independence_lr"],
                "p-value": stats["independence_p_value"],
                "5% decision": _coverage_label(stats["independence_p_value"]),
            },
            {
                "Test": "Conditional coverage",
                "LR statistic": stats["conditional_coverage_lr"],
                "p-value": stats["conditional_coverage_p_value"],
                "5% decision": _coverage_label(stats["conditional_coverage_p_value"]),
            },
        ]
    )
    st.dataframe(coverage, hide_index=True, use_container_width=True)

    details_left, details_right = st.columns(2)
    with details_left:
        st.subheader("Latest model diagnostics")
        st.json(diagnostics)
    with details_right:
        st.subheader("Latest parameter estimates")
        st.dataframe(analysis["parameters"], use_container_width=True)

    export_left, export_right = st.columns(2)
    with export_left:
        st.download_button(
            "Download backtest CSV",
            data=backtest.to_csv().encode("utf-8"),
            file_name=f"{analysis['symbol']}_garch_backtest.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_right:
        st.download_button(
            "Download PDF report",
            data=analysis["pdf"],
            file_name=f"{analysis['symbol']}_garch_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.warning(
        "VaR and expected shortfall are model-based estimates. They are not guarantees, "
        "market calibration evidence or trading recommendations."
    )


if __name__ == "__main__":
    main()
