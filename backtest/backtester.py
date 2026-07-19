import logging
import pandas as pd
import numpy as np
from scipy.stats import norm, chi2
from models.garch_model import fit_garch, forecast_variance

logger = logging.getLogger(__name__)

def run_backtest(
    returns: pd.Series,
    window: int = 250,
    alpha: float = 0.05,
    p: int = 1,
    q: int = 1,
) -> pd.DataFrame:
    """
    Walk-forward backtest of parametric GARCH VaR/ES.
    Returns columns: [mu, sigma, sigma2, var, es, breach, es_breach, ret].
    VaR/ES are return-space thresholds (typically negative).
    """
    logger.info(f"run_backtest: window={window}, alpha={alpha}, p={p}, q={q}, points={len(returns)}")
    records = []

    for t in range(window, len(returns)):
        train = returns.iloc[t - window : t]
        res = fit_garch(train, p=p, q=q)
        sigma2 = forecast_variance(res)
        sigma = np.sqrt(max(sigma2, 0.0))
        mu = float(train.mean())

        z_alpha = norm.ppf(alpha)
        var = mu + sigma * z_alpha
        es = mu - sigma * norm.pdf(z_alpha) / alpha

        realized_ret = float(returns.iloc[t])
        breach = realized_ret < var
        es_breach = realized_ret < es

        records.append({
            "date": returns.index[t],
            "mu": mu,
            "sigma": sigma,
            "sigma2": sigma2,
            "var": var,
            "es": es,
            "breach": breach,
            "es_breach": es_breach,
            "ret": realized_ret,
        })

        if t % 50 == 0:
            logger.debug(
                f"  step {t}: date={returns.index[t]}, sigma2={sigma2:.6f}, "
                f"VaR={var:.6f}, ES={es:.6f}, breach={breach}"
            )

    df = pd.DataFrame(records)
    if df.empty or "date" not in df.columns:
        logger.warning("run_backtest: no records generated (window > data length). Returning empty DataFrame.")
        return pd.DataFrame(columns=["sigma2", "var", "breach"], index=pd.DatetimeIndex([], name="date"))

    df = df.set_index("date")
    logger.info(f"run_backtest: completed with {len(df)} rows")
    return df

def summary_stats(bt_df: pd.DataFrame, alpha: float = 0.05) -> dict:
    total = len(bt_df)
    breaches = int(bt_df["breach"].sum()) if total > 0 else 0
    hit_rate = breaches / total if total > 0 else np.nan
    expected_breaches = alpha * total

    if total > 0 and 0 < hit_rate < 1:
        lr_uc = -2.0 * (
            (total - breaches) * np.log((1 - alpha) / (1 - hit_rate))
            + breaches * np.log(alpha / hit_rate)
        )
        kupiec_pvalue = 1.0 - chi2.cdf(lr_uc, df=1)
    else:
        kupiec_pvalue = np.nan

    ci_half = 1.96 * np.sqrt((alpha * (1 - alpha)) / total) if total > 0 else np.nan
    ci_low = max(0.0, alpha - ci_half) if total > 0 else np.nan
    ci_high = min(1.0, alpha + ci_half) if total > 0 else np.nan

    stats = {
        "total_days": int(total),
        "breaches": int(breaches),
        "hit_rate": float(hit_rate) if total > 0 else float("nan"),
        "expected_breaches": float(expected_breaches),
        "hit_rate_ci_low": float(ci_low) if total > 0 else float("nan"),
        "hit_rate_ci_high": float(ci_high) if total > 0 else float("nan"),
        "kupiec_pvalue": float(kupiec_pvalue) if not np.isnan(kupiec_pvalue) else float("nan"),
        "es_breaches": int(bt_df["es_breach"].sum()) if total > 0 else 0,
    }
    logger.info(f"summary_stats: {stats}")
    return stats
