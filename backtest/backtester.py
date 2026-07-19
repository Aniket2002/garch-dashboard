from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2

from models.garch_model import GarchSpec, RiskForecast, fit_garch, forecast_tail_risk


class BacktestError(RuntimeError):
    """Raised when a walk-forward model fit or forecast fails."""


RiskForecaster = Callable[[pd.Series, float, GarchSpec], RiskForecast]
ProgressCallback = Callable[[int, int], None]


def _default_forecaster(
    training_returns: pd.Series,
    alpha: float,
    spec: GarchSpec,
) -> RiskForecast:
    result = fit_garch(training_returns, spec)
    return forecast_tail_risk(result, alpha)


def _empty_backtest() -> pd.DataFrame:
    frame = pd.DataFrame(
        columns=[
            "realized_return",
            "conditional_mean",
            "variance",
            "volatility",
            "var",
            "expected_shortfall",
            "breach",
            "es_breach",
        ]
    )
    frame.index = pd.DatetimeIndex([], name="date")
    return frame


def run_backtest(
    returns: pd.Series,
    *,
    window: int = 250,
    alpha: float = 0.05,
    spec: GarchSpec | None = None,
    forecaster: RiskForecaster | None = None,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Run a strict one-step-ahead rolling GARCH VaR/ES backtest.

    For forecast date ``t``, the estimation sample ends at ``t-1``. The
    realized return at ``t`` is never included in its own model fit.
    """

    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if window < 50:
        raise ValueError("window must be at least 50 observations")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between zero and 0.5")

    clean = (
        pd.to_numeric(returns, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )
    if isinstance(clean.index, pd.DatetimeIndex):
        clean = clean[~clean.index.duplicated(keep="last")].sort_index()

    if len(clean) <= window:
        return _empty_backtest()

    spec = spec or GarchSpec()
    spec.validate()
    forecast_function = forecaster or _default_forecaster
    total_forecasts = len(clean) - window
    records: list[dict[str, Any]] = []

    for completed, position in enumerate(range(window, len(clean)), start=1):
        training_sample = clean.iloc[position - window : position]
        forecast_date = clean.index[position]
        try:
            forecast = forecast_function(training_sample, alpha, spec)
        except Exception as exc:  # pragma: no cover - exact optimizer failure varies
            raise BacktestError(
                f"risk forecast failed for {forecast_date}: {exc}"
            ) from exc

        realized_return = float(clean.iloc[position])
        records.append(
            {
                "date": forecast_date,
                "realized_return": realized_return,
                "conditional_mean": forecast.mean,
                "variance": forecast.variance,
                "volatility": forecast.volatility,
                "var": forecast.var,
                "expected_shortfall": forecast.expected_shortfall,
                "breach": bool(realized_return < forecast.var),
                "es_breach": bool(realized_return < forecast.expected_shortfall),
            }
        )
        if progress_callback is not None:
            progress_callback(completed, total_forecasts)

    result = pd.DataFrame.from_records(records).set_index("date")
    if isinstance(clean.index, pd.DatetimeIndex):
        result.index = pd.DatetimeIndex(result.index, name="date")
    return result


def _xlogy(count: int, probability: float) -> float:
    if count == 0:
        return 0.0
    if probability <= 0.0:
        return -math.inf
    return count * math.log(probability)


def kupiec_unconditional_coverage(
    breaches: pd.Series | np.ndarray | list[bool],
    alpha: float,
) -> dict[str, float | int]:
    """Kupiec likelihood-ratio test of the unconditional breach frequency."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    values = np.asarray(breaches, dtype=bool)
    observations = int(values.size)
    exceptions = int(values.sum())
    if observations == 0:
        return {
            "observations": 0,
            "exceptions": 0,
            "exception_rate": math.nan,
            "lr_stat": math.nan,
            "p_value": math.nan,
        }

    empirical_probability = exceptions / observations
    null_log_likelihood = _xlogy(exceptions, alpha) + _xlogy(
        observations - exceptions, 1.0 - alpha
    )
    alternative_log_likelihood = _xlogy(exceptions, empirical_probability) + _xlogy(
        observations - exceptions, 1.0 - empirical_probability
    )
    lr_stat = max(0.0, -2.0 * (null_log_likelihood - alternative_log_likelihood))
    return {
        "observations": observations,
        "exceptions": exceptions,
        "exception_rate": empirical_probability,
        "lr_stat": lr_stat,
        "p_value": float(chi2.sf(lr_stat, df=1)),
    }


def christoffersen_independence(
    breaches: pd.Series | np.ndarray | list[bool],
) -> dict[str, float | int]:
    """Christoffersen likelihood-ratio test for clustered VaR exceptions."""

    values = np.asarray(breaches, dtype=int)
    if values.size < 2:
        return {
            "n00": 0,
            "n01": 0,
            "n10": 0,
            "n11": 0,
            "lr_stat": math.nan,
            "p_value": math.nan,
        }

    previous = values[:-1]
    current = values[1:]
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))

    total_transitions = n00 + n01 + n10 + n11
    unconditional_probability = (n01 + n11) / total_transitions
    p01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) else 0.0

    independent_log_likelihood = _xlogy(n01 + n11, unconditional_probability) + _xlogy(
        n00 + n10, 1.0 - unconditional_probability
    )
    markov_log_likelihood = (
        _xlogy(n01, p01)
        + _xlogy(n00, 1.0 - p01)
        + _xlogy(n11, p11)
        + _xlogy(n10, 1.0 - p11)
    )
    lr_stat = max(0.0, -2.0 * (independent_log_likelihood - markov_log_likelihood))
    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "lr_stat": lr_stat,
        "p_value": float(chi2.sf(lr_stat, df=1)),
    }


def summary_stats(backtest: pd.DataFrame, alpha: float) -> dict[str, float | int]:
    """Summarize forecast accuracy and formal VaR coverage tests."""

    required = {
        "realized_return",
        "volatility",
        "var",
        "expected_shortfall",
        "breach",
    }
    missing = required - set(backtest.columns)
    if missing:
        raise ValueError(f"backtest is missing columns: {sorted(missing)}")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between zero and 0.5")

    total = len(backtest)
    if total == 0:
        return {
            "total_days": 0,
            "breaches": 0,
            "expected_breaches": 0.0,
            "breach_rate": math.nan,
            "expected_rate": alpha,
            "coverage_ratio": math.nan,
            "average_daily_volatility": math.nan,
            "average_annualized_volatility": math.nan,
            "mean_var_shortfall": math.nan,
            "mean_es_shortfall": math.nan,
            "kupiec_lr": math.nan,
            "kupiec_p_value": math.nan,
            "independence_lr": math.nan,
            "independence_p_value": math.nan,
            "conditional_coverage_lr": math.nan,
            "conditional_coverage_p_value": math.nan,
        }

    breach_mask = backtest["breach"].astype(bool)
    breaches = int(breach_mask.sum())
    breach_rate = breaches / total
    kupiec = kupiec_unconditional_coverage(breach_mask, alpha)
    independence = christoffersen_independence(breach_mask)
    conditional_lr = float(kupiec["lr_stat"]) + float(independence["lr_stat"])

    var_shortfalls = (
        backtest.loc[breach_mask, "var"]
        - backtest.loc[breach_mask, "realized_return"]
    )
    es_mask = backtest["realized_return"] < backtest["expected_shortfall"]
    es_shortfalls = (
        backtest.loc[es_mask, "expected_shortfall"]
        - backtest.loc[es_mask, "realized_return"]
    )

    return {
        "total_days": total,
        "breaches": breaches,
        "expected_breaches": alpha * total,
        "breach_rate": breach_rate,
        "expected_rate": alpha,
        "coverage_ratio": breach_rate / alpha,
        "average_daily_volatility": float(backtest["volatility"].mean()),
        "average_annualized_volatility": float(
            backtest["volatility"].mean() * math.sqrt(252.0)
        ),
        "mean_var_shortfall": float(var_shortfalls.mean()) if breaches else 0.0,
        "mean_es_shortfall": float(es_shortfalls.mean()) if bool(es_mask.any()) else 0.0,
        "kupiec_lr": float(kupiec["lr_stat"]),
        "kupiec_p_value": float(kupiec["p_value"]),
        "independence_lr": float(independence["lr_stat"]),
        "independence_p_value": float(independence["p_value"]),
        "conditional_coverage_lr": conditional_lr,
        "conditional_coverage_p_value": float(chi2.sf(conditional_lr, df=2)),
    }
