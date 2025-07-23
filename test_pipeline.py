import logging
import numpy as np
import pandas as pd

from data.polygon_client import fetch_data
from models.garch_model import fit_garch, forecast_variance

# ── Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("test_pipeline")

def main():
    try:
        logger.info("Starting end-to-end pipeline test")

        symbol = "AAPL"
        start, end = "2024-01-01", "2024-12-31"
        logger.info(f"Fetching data for {symbol} from {start} to {end}")
        df = fetch_data(symbol, start, end)
        logger.info(f"Data fetched: {len(df)} rows")

        rets = pd.Series(np.log(df["close"] / df["close"].shift(1))).dropna()
        logger.info(f"Computed returns: {len(rets)} points")

        logger.info("Fitting GARCH(1,1) model")
        res = fit_garch(rets)
        logger.debug(f"GARCH fit summary:\n{res.summary()}")

        var = forecast_variance(res)
        logger.info(f"1-day forecast variance: {var:.6f}")

        logger.info("Pipeline test completed successfully")
    except Exception:
        logger.exception("Error during pipeline test")
        exit(1)

if __name__ == "__main__":
    main()
