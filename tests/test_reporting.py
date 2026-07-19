from __future__ import annotations

import pandas as pd

from models.garch_model import GarchSpec, RiskForecast
from reporting.pdf_report import build_pdf_report


def test_pdf_report_is_generated_in_memory():
    index = pd.date_range("2024-01-01", periods=3)
    backtest = pd.DataFrame(
        {
            "realized_return": [0.01, -0.03, 0.0],
            "var": [-0.02, -0.02, -0.02],
            "expected_shortfall": [-0.03, -0.03, -0.03],
            "breach": [False, True, False],
        },
        index=index,
    )
    pdf = build_pdf_report(
        title="Test report",
        symbol="TEST",
        start_date="2024-01-01",
        end_date="2024-01-03",
        spec=GarchSpec(),
        alpha=0.05,
        window=250,
        stats={
            "total_days": 3,
            "breaches": 1,
            "expected_breaches": 0.15,
            "breach_rate": 1 / 3,
            "kupiec_p_value": 0.4,
            "independence_p_value": 0.5,
            "conditional_coverage_p_value": 0.6,
        },
        latest_forecast=RiskForecast(
            mean=0.0,
            variance=0.0001,
            volatility=0.01,
            var=-0.02,
            expected_shortfall=-0.03,
            alpha=0.05,
            innovation_quantile=-2.0,
            innovation_expected_shortfall=-3.0,
        ),
        diagnostics={
            "converged": True,
            "observations": 250,
            "aic": 1.0,
            "bic": 2.0,
            "persistence": 0.95,
            "stationary_by_persistence": True,
        },
        backtest=backtest,
    )
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 1_000
