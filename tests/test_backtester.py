from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from backtest.backtester import (
    christoffersen_independence,
    kupiec_unconditional_coverage,
    run_backtest,
    summary_stats,
)
from models.garch_model import GarchSpec, RiskForecast


def _fixed_forecaster(
    training_returns: pd.Series,
    alpha: float,
    spec: GarchSpec,
) -> RiskForecast:
    assert len(training_returns) == 50
    assert alpha == pytest.approx(0.05)
    assert spec == GarchSpec()
    return RiskForecast(
        mean=0.0,
        variance=0.0001,
        volatility=0.01,
        var=-0.02,
        expected_shortfall=-0.03,
        alpha=alpha,
        innovation_quantile=-2.0,
        innovation_expected_shortfall=-3.0,
    )


def test_walk_forward_backtest_has_no_lookahead_and_flags_breaches():
    index = pd.date_range("2024-01-01", periods=60, freq="D")
    values = np.zeros(60)
    values[55] = -0.025
    values[57] = -0.04
    returns = pd.Series(values, index=index)
    seen_training_ends: list[pd.Timestamp] = []

    def recorder(sample: pd.Series, alpha: float, spec: GarchSpec) -> RiskForecast:
        seen_training_ends.append(sample.index[-1])
        return _fixed_forecaster(sample, alpha, spec)

    result = run_backtest(
        returns,
        window=50,
        alpha=0.05,
        forecaster=recorder,
    )

    assert len(result) == 10
    assert result.index[0] == index[50]
    assert seen_training_ends[0] == index[49]
    assert seen_training_ends == list(index[49:59])
    assert int(result["breach"].sum()) == 2
    assert int(result["es_breach"].sum()) == 1
    assert result.loc[index[55], "breach"]
    assert result.loc[index[57], "es_breach"]


def test_breach_thresholds_use_strict_lower_tail_inequality():
    values = np.zeros(52)
    values[50] = -0.02
    values[51] = -0.03
    result = run_backtest(pd.Series(values), window=50, forecaster=_fixed_forecaster)

    assert not result.iloc[0]["breach"]
    assert result.iloc[1]["breach"]
    assert not result.iloc[1]["es_breach"]


def test_backtest_returns_typed_empty_frame_when_history_is_short():
    returns = pd.Series(np.zeros(50), index=pd.date_range("2024-01-01", periods=50))
    result = run_backtest(returns, window=50, forecaster=_fixed_forecaster)
    assert result.empty
    assert list(result.columns) == [
        "realized_return",
        "conditional_mean",
        "variance",
        "volatility",
        "var",
        "expected_shortfall",
        "breach",
        "es_breach",
    ]


def test_kupiec_is_zero_when_observed_rate_equals_alpha():
    breaches = np.array([True] * 5 + [False] * 95)
    result = kupiec_unconditional_coverage(breaches, 0.05)
    assert result["exceptions"] == 5
    assert result["exception_rate"] == pytest.approx(0.05)
    assert result["lr_stat"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_kupiec_matches_binomial_likelihood_ratio():
    breaches = np.array([True] * 2 + [False] * 8)
    result = kupiec_unconditional_coverage(breaches, 0.05)
    null_log_likelihood = 2 * math.log(0.05) + 8 * math.log(0.95)
    fitted_log_likelihood = 2 * math.log(0.2) + 8 * math.log(0.8)
    expected_lr = -2.0 * (null_log_likelihood - fitted_log_likelihood)

    assert result["lr_stat"] == pytest.approx(expected_lr)
    assert result["p_value"] == pytest.approx(chi2.sf(expected_lr, df=1))


def test_christoffersen_counts_and_summary_are_finite():
    pattern = [False, False, True, False, True, True, False, False]
    independence = christoffersen_independence(pattern)
    assert independence["n00"] == 2
    assert independence["n01"] == 2
    assert independence["n10"] == 2
    assert independence["n11"] == 1
    independent_log_likelihood = 3 * math.log(3 / 7) + 4 * math.log(4 / 7)
    markov_log_likelihood = (
        2 * math.log(0.5)
        + 2 * math.log(0.5)
        + math.log(1 / 3)
        + 2 * math.log(2 / 3)
    )
    expected_independence_lr = -2.0 * (
        independent_log_likelihood - markov_log_likelihood
    )
    assert independence["lr_stat"] == pytest.approx(expected_independence_lr)
    assert math.isfinite(float(independence["p_value"]))

    index = pd.date_range("2024-01-01", periods=len(pattern))
    backtest = pd.DataFrame(
        {
            "realized_return": [-0.01, -0.01, -0.03, -0.01, -0.04, -0.05, 0.0, 0.0],
            "volatility": [0.01] * len(pattern),
            "var": [-0.02] * len(pattern),
            "expected_shortfall": [-0.035] * len(pattern),
            "breach": pattern,
        },
        index=index,
    )
    stats = summary_stats(backtest, 0.05)
    assert stats["breaches"] == 3
    assert stats["total_days"] == len(pattern)
    assert stats["conditional_coverage_lr"] == pytest.approx(
        stats["kupiec_lr"] + stats["independence_lr"]
    )
    assert stats["conditional_coverage_p_value"] == pytest.approx(
        chi2.sf(stats["conditional_coverage_lr"], df=2)
    )
    assert math.isfinite(float(stats["conditional_coverage_p_value"]))
