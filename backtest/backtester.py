import logging
import pandas as pd
import numpy as np
from models.garch_model import fit_garch, forecast_variance

logger = logging.getLogger(__name__)

def run_backtest(returns: pd.Series, window: int = 250, alpha: float = 0.05) -> pd.DataFrame:
    """
    Walk‐forward backtest of GARCH(1,1) VaR breaches.
    Returns DataFrame indexed by date with columns [sigma2, var, breach].
    """
    logger.info(f"run_backtest: window={window}, alpha={alpha}, points={len(returns)}")
    records = []

    for t in range(window, len(returns)):
        train = returns.iloc[t - window : t]
        res = fit_garch(train)
        σ2 = forecast_variance(res)
        σ = np.sqrt(σ2)
        # negative Var since returns < var is a breach
        var = -σ * abs(np.percentile(train, alpha*100))
        breach = returns.iloc[t] < var

        records.append({
            "date": returns.index[t],
            "sigma2": σ2,
            "var": var,
            "breach": breach
        })

        if t % 50 == 0:
            logger.debug(f"  at t={t}, date={returns.index[t]}, var={var:.6f}, breach={breach}")

    df = pd.DataFrame(records)

    if df.empty or "date" not in df.columns:
        logger.warning(
            "run_backtest: no records generated (maybe window > data length). "
            "Returning empty DataFrame."
        )
        # empty but with correct columns and DateTimeIndex
        return pd.DataFrame(
            columns=["sigma2", "var", "breach"],
            index=pd.DatetimeIndex([], name="date")
        )

    df = df.set_index("date")
    logger.info(f"run_backtest: completed, {len(df)} rows")
    return df

def summary_stats(bt_df: pd.DataFrame) -> dict:
    total = len(bt_df)
    breaches = int(bt_df["breach"].sum()) if total > 0 else 0
    hit_rate = breaches / total if total else np.nan
    stats = {"total_days": total, "breaches": breaches, "hit_rate": hit_rate}
    logger.info(f"summary_stats: {stats}")
    return stats
