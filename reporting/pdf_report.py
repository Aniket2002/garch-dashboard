from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

import pandas as pd
from fpdf import FPDF

from models.garch_model import GarchSpec, RiskForecast


def _latin1(value: Any) -> str:
    return str(value).encode("latin-1", errors="replace").decode("latin-1")


def _format_number(value: Any, *, percentage: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _latin1(value)
    if not math.isfinite(number):
        return "n/a"
    if percentage:
        return f"{number:.2%}"
    return f"{number:,.4f}"


def _add_key_value_table(
    pdf: FPDF,
    rows: list[tuple[str, str]],
    *,
    label_width: float = 95.0,
) -> None:
    value_width = pdf.epw - label_width
    pdf.set_font("Helvetica", size=9)
    for label, value in rows:
        pdf.cell(label_width, 7, _latin1(label), border=1)
        pdf.cell(value_width, 7, _latin1(value), border=1, new_x="LMARGIN", new_y="NEXT")


def build_pdf_report(
    *,
    title: str,
    symbol: str,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    spec: GarchSpec,
    alpha: float,
    window: int,
    stats: Mapping[str, Any],
    latest_forecast: RiskForecast,
    diagnostics: Mapping[str, Any],
    backtest: pd.DataFrame,
) -> bytes:
    """Build a compact, dependency-light PDF report entirely in memory."""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 17)
    pdf.cell(0, 10, _latin1(title), new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(
        0,
        7,
        _latin1(f"{symbol} | {start_date} to {end_date}"),
        new_x="LMARGIN",
        new_y="NEXT",
        align="C",
    )
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Model specification", new_x="LMARGIN", new_y="NEXT")
    _add_key_value_table(
        pdf,
        [
            ("Model", f"GARCH({spec.p},{spec.o},{spec.q})"),
            ("Conditional mean", spec.mean),
            ("Innovation distribution", spec.distribution),
            ("Rolling window", str(window)),
            ("Tail probability", _format_number(alpha, percentage=True)),
        ],
    )
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Latest one-day forecast", new_x="LMARGIN", new_y="NEXT")
    _add_key_value_table(
        pdf,
        [
            ("Conditional mean", _format_number(latest_forecast.mean, percentage=True)),
            ("Daily volatility", _format_number(latest_forecast.volatility, percentage=True)),
            (
                "Annualized volatility",
                _format_number(latest_forecast.volatility * math.sqrt(252.0), percentage=True),
            ),
            ("VaR return threshold", _format_number(latest_forecast.var, percentage=True)),
            (
                "Expected-shortfall threshold",
                _format_number(latest_forecast.expected_shortfall, percentage=True),
            ),
        ],
    )
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Backtest summary", new_x="LMARGIN", new_y="NEXT")
    _add_key_value_table(
        pdf,
        [
            ("Forecast days", str(stats.get("total_days", "n/a"))),
            ("VaR breaches", str(stats.get("breaches", "n/a"))),
            ("Expected breaches", _format_number(stats.get("expected_breaches"))),
            ("Observed breach rate", _format_number(stats.get("breach_rate"), percentage=True)),
            ("Kupiec p-value", _format_number(stats.get("kupiec_p_value"))),
            (
                "Christoffersen independence p-value",
                _format_number(stats.get("independence_p_value")),
            ),
            (
                "Conditional-coverage p-value",
                _format_number(stats.get("conditional_coverage_p_value")),
            ),
        ],
    )
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Estimation diagnostics", new_x="LMARGIN", new_y="NEXT")
    _add_key_value_table(
        pdf,
        [
            ("Converged", str(diagnostics.get("converged", "n/a"))),
            ("Observations", str(diagnostics.get("observations", "n/a"))),
            ("AIC", _format_number(diagnostics.get("aic"))),
            ("BIC", _format_number(diagnostics.get("bic"))),
            ("Persistence", _format_number(diagnostics.get("persistence"))),
            (
                "Stationary by persistence",
                str(diagnostics.get("stationary_by_persistence", "n/a")),
            ),
        ],
    )

    if not backtest.empty:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Most recent backtest observations", new_x="LMARGIN", new_y="NEXT")
        recent = backtest.tail(20).rename_axis("date").reset_index()
        columns = ["date", "realized_return", "var", "expected_shortfall", "breach"]
        widths = [34.0, 36.0, 36.0, 48.0, 26.0]
        headers = ["Date", "Return", "VaR", "ES", "Breach"]
        pdf.set_font("Helvetica", "B", 8)
        for header, width in zip(headers, widths, strict=True):
            pdf.cell(width, 7, header, border=1, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", size=8)
        for _, row in recent.iterrows():
            values = [
                str(pd.Timestamp(row[columns[0]]).date()),
                _format_number(row[columns[1]], percentage=True),
                _format_number(row[columns[2]], percentage=True),
                _format_number(row[columns[3]], percentage=True),
                str(bool(row[columns[4]])),
            ]
            for value, width in zip(values, widths, strict=True):
                pdf.cell(width, 6, _latin1(value), border=1, align="C")
            pdf.ln()

    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(
        0,
        5,
        _latin1(
            "Educational scenario analysis only. VaR and expected shortfall are model-based "
            "estimates, not guarantees and not trading recommendations."
        ),
    )
    return bytes(pdf.output())
