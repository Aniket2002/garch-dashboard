from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from arch.univariate.distribution import Normal
from scipy.stats import norm

from models.garch_model import (
    GarchSpec,
    clean_returns,
    fit_garch,
    forecast_tail_risk,
    model_diagnostics,
)


class FakeResult:
    def __init__(self) -> None:
        self.model = SimpleNamespace(distribution=Normal())
        self.params = pd.Series(dtype=float)

    def forecast(self, horizon: int, reindex: bool):
        assert horizon == 1
        assert reindex is False
        return SimpleNamespace(
            mean=pd.DataFrame([[1.0]]),
            variance=pd.DataFrame([[4.0]]),
        )


def test_normal_tail_forecast_matches_closed_form():
    alpha = 0.05
    forecast = forecast_tail_risk(FakeResult(), alpha)
    expected_quantile = norm.ppf(alpha)
    expected_tail_mean = -norm.pdf(expected_quantile) / alpha

    assert forecast.mean == pytest.approx(0.01)
    assert forecast.volatility == pytest.approx(0.02)
    assert forecast.variance == pytest.approx(0.0004)
    assert forecast.var == pytest.approx(0.01 + 0.02 * expected_quantile)
    assert forecast.expected_shortfall == pytest.approx(
        0.01 + 0.02 * expected_tail_mean
    )
    assert forecast.expected_shortfall < forecast.var


def test_clean_returns_rejects_constant_or_too_short_series():
    with pytest.raises(ValueError, match="at least"):
        clean_returns(pd.Series([0.01, -0.01]), minimum_observations=3)
    with pytest.raises(ValueError, match="non-zero variance"):
        clean_returns(pd.Series([0.01] * 50))


def test_fit_garch_smoke_and_diagnostics_are_finite():
    rng = np.random.default_rng(8)
    returns = pd.Series(rng.normal(0.0, 0.012, 400))
    result = fit_garch(
        returns,
        GarchSpec(distribution="normal"),
        minimum_observations=100,
    )
    forecast = forecast_tail_risk(result, 0.05)
    diagnostics = model_diagnostics(result)

    assert forecast.variance >= 0.0
    assert forecast.expected_shortfall <= forecast.var
    assert diagnostics["converged"] is True
    assert diagnostics["observations"] == 400
    assert np.isfinite(diagnostics["aic"])
    assert np.isfinite(diagnostics["bic"])
