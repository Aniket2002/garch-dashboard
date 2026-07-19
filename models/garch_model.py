from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model


@dataclass(frozen=True)
class GarchSpec:
    """Specification for a univariate conditional-volatility model."""

    p: int = 1
    o: int = 0
    q: int = 1
    mean: str = "Constant"
    distribution: str = "student_t"

    def validate(self) -> None:
        if self.p <= 0 or self.q <= 0 or self.o < 0:
            raise ValueError("p and q must be positive and o cannot be negative")
        if self.mean not in {"Constant", "Zero"}:
            raise ValueError("mean must be either 'Constant' or 'Zero'")
        if self.distribution not in {"normal", "student_t"}:
            raise ValueError("distribution must be 'normal' or 'student_t'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskForecast:
    """One-step conditional mean, volatility, VaR and ES in decimal-return units."""

    mean: float
    variance: float
    volatility: float
    var: float
    expected_shortfall: float
    alpha: float
    innovation_quantile: float
    innovation_expected_shortfall: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def clean_returns(returns: pd.Series, *, minimum_observations: int = 50) -> pd.Series:
    """Return a finite, sorted decimal-return series suitable for estimation."""

    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    cleaned = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned = cleaned[~cleaned.index.duplicated(keep="last")].sort_index()

    if len(cleaned) < minimum_observations:
        raise ValueError(
            f"at least {minimum_observations} finite return observations are required"
        )
    if float(cleaned.std(ddof=0)) <= 0:
        raise ValueError("returns must have non-zero variance")
    return cleaned.astype(float)


def fit_garch(
    returns: pd.Series,
    spec: GarchSpec | None = None,
    *,
    minimum_observations: int = 50,
):
    """Fit a GARCH-family model to decimal returns.

    The ``arch`` package is fed percentage returns for numerical stability. All
    public forecasts are converted back to decimal-return units.
    """

    spec = spec or GarchSpec()
    spec.validate()
    series = clean_returns(returns, minimum_observations=minimum_observations) * 100.0
    distribution = "normal" if spec.distribution == "normal" else "t"

    model = arch_model(
        series,
        mean=spec.mean,
        vol="GARCH",
        p=spec.p,
        o=spec.o,
        q=spec.q,
        dist=distribution,
        rescale=False,
    )
    result = model.fit(disp="off", show_warning=False)
    if int(getattr(result, "convergence_flag", 0)) != 0:
        raise RuntimeError(
            f"GARCH optimizer did not converge; flag={result.convergence_flag}"
        )
    return result


def _distribution_parameters(result: Any) -> list[float]:
    names = result.model.distribution.parameter_names()
    return [float(result.params[name]) for name in names]


def innovation_tail_statistics(result: Any, alpha: float) -> tuple[float, float]:
    """Return standardized innovation quantile and lower-tail conditional mean."""

    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between zero and 0.5")

    distribution = result.model.distribution
    parameters = _distribution_parameters(result)
    quantile = float(distribution.ppf(alpha, parameters))
    partial_first_moment = float(
        distribution.partial_moment(1, quantile, parameters)
    )
    expected_tail_innovation = partial_first_moment / alpha

    if not np.isfinite(quantile) or not np.isfinite(expected_tail_innovation):
        raise RuntimeError("distribution tail statistics are not finite")
    return quantile, expected_tail_innovation


def forecast_tail_risk(result: Any, alpha: float = 0.05) -> RiskForecast:
    """Produce a one-day conditional VaR and expected shortfall forecast.

    ``var`` and ``expected_shortfall`` are return thresholds. A breach occurs
    when the realized return is below the corresponding threshold.
    """

    quantile, tail_mean = innovation_tail_statistics(result, alpha)
    forecast = result.forecast(horizon=1, reindex=False)
    mean_percent = float(forecast.mean.iloc[-1, 0])
    variance_percent_squared = float(forecast.variance.iloc[-1, 0])

    variance = max(variance_percent_squared / 100.0**2, 0.0)
    volatility = float(np.sqrt(variance))
    mean = mean_percent / 100.0
    var = mean + volatility * quantile
    expected_shortfall = mean + volatility * tail_mean

    if expected_shortfall > var + 1e-12:
        raise AssertionError("expected shortfall must not exceed the VaR threshold")

    return RiskForecast(
        mean=mean,
        variance=variance,
        volatility=volatility,
        var=var,
        expected_shortfall=expected_shortfall,
        alpha=alpha,
        innovation_quantile=quantile,
        innovation_expected_shortfall=tail_mean,
    )


def model_diagnostics(result: Any) -> dict[str, float | int | bool]:
    """Extract compact estimation diagnostics from a fitted model."""

    params = result.params
    alpha_sum = sum(float(value) for name, value in params.items() if name.startswith("alpha["))
    beta_sum = sum(float(value) for name, value in params.items() if name.startswith("beta["))
    gamma_sum = sum(float(value) for name, value in params.items() if name.startswith("gamma["))
    persistence = alpha_sum + beta_sum + 0.5 * gamma_sum

    omega = float(params.get("omega", np.nan))
    if np.isfinite(omega) and persistence < 1.0:
        unconditional_variance_decimal = (omega / (1.0 - persistence)) / 100.0**2
        unconditional_annualized_volatility = float(
            np.sqrt(max(unconditional_variance_decimal, 0.0) * 252.0)
        )
    else:
        unconditional_annualized_volatility = float("nan")

    return {
        "converged": int(getattr(result, "convergence_flag", 0)) == 0,
        "convergence_flag": int(getattr(result, "convergence_flag", 0)),
        "log_likelihood": float(result.loglikelihood),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "persistence": float(persistence),
        "stationary_by_persistence": bool(persistence < 1.0),
        "unconditional_annualized_volatility": unconditional_annualized_volatility,
        "observations": int(result.nobs),
    }
