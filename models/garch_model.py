import logging
import numpy as np
from arch import arch_model

logger = logging.getLogger(__name__)

def fit_garch(returns, p: int = 1, q: int = 1):
    """
    Fit a GARCH(p, q) on the series of returns (in decimal form).
    Returns the fitted model result.
    """
    # arch expects percentages
    series = returns.dropna() * 100
    am = arch_model(series, vol='GARCH', p=p, q=q, dist='normal')
    res = am.fit(disp='off')
    return res

def forecast_variance(res, horizon: int = 1) -> float:
    """
    Given a fitted result, forecast future variance for `horizon` days.
    Returns σ² in decimal form.
    """
    var_pct = res.forecast(horizon=horizon).variance.iloc[-1].values[0]
    return var_pct / 100**2
