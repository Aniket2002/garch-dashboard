import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ── Setup logging & env
load_dotenv()
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── API Key check
POLY_KEY = os.getenv("POLYGON_API_KEY")
if not POLY_KEY:
    logger.error("POLYGON_API_KEY is not set!")
BASE_URL = "https://api.polygon.io"


def fetch_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLC & volume for `symbol` from Polygon.
    First attempts v2 aggregates; on 403/429 falls back to v1 open-close per day.
    """
    logger.info(f"fetch_data: symbol={symbol}, start={start}, end={end}")
    df = _fetch_v2(symbol, start, end)
    if df is not None:
        return df
    return _fetch_v1_open_close(symbol, start, end)


def _fetch_v2(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Try the v2 aggregates endpoint; return DataFrame or None if unauthorized/rate-limited."""
    url = f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "apiKey": POLY_KEY}
    logger.debug(f"[v2] URL={url}  PARAMS={params}")
    resp = requests.get(url, params=params)

    if resp.status_code in (403, 429):
        body = resp.json()
        logger.warning(f"[v2] HTTP {resp.status_code}: {body.get('message')}. Falling back.")
        return None

    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        logger.error(f"[v2] HTTP {resp.status_code}: {resp.text}")
        raise

    payload = resp.json()
    data = payload.get("results", [])
    logger.debug(f"[v2] Raw payload keys: {list(payload.keys())}, rows returned: {len(data)}")
    if not data:
        logger.warning(f"[v2] No data for {symbol} between {start} and {end}")
        return pd.DataFrame()

    df = pd.DataFrame(data).rename(columns={
        "t": "timestamp", "o": "open", "h": "high",
        "l": "low", "c": "close", "v": "volume"
    })
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
    df.set_index("date", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]
    logger.info(f"[v2] Fetched {len(df)} rows for {symbol}")
    return df


def _fetch_v1_open_close(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fall back to v1 open-close endpoint, one request per day.
    Throttles to 5 requests per minute.
    """
    logger.info(f"[v1] Falling back: symbol={symbol}, start={start}, end={end}")
    s_date = datetime.strptime(start, "%Y-%m-%d").date()
    e_date = datetime.strptime(end, "%Y-%m-%d").date()
    records, req_count = [], 0
    curr = s_date

    while curr <= e_date:
        date_str = curr.isoformat()
        url = f"{BASE_URL}/v1/open-close/{symbol}/{date_str}"
        params = {"adjusted": "true", "apiKey": POLY_KEY}
        logger.debug(f"[v1] Requesting {symbol} on {date_str}")
        resp = requests.get(url, params=params)

        if resp.status_code == 200:
            j = resp.json()
            if j.get("status") == "OK":
                records.append({
                    "date": curr,
                    "open": j["open"],
                    "high": j["high"],
                    "low": j["low"],
                    "close": j["close"],
                    "volume": j["volume"],
                })
                logger.debug(f"[v1] Appended data for {date_str}")
            else:
                logger.warning(f"[v1] Status not OK for {date_str}: {j}")
        else:
            logger.warning(f"[v1] HTTP {resp.status_code} on {date_str}: {resp.text}")

        req_count += 1
        if req_count % 5 == 0 and curr < e_date:
            logger.info(f"[v1] Throttling: {req_count} calls made, sleeping 60s")
            time.sleep(60)
        curr += timedelta(days=1)

    if not records:
        logger.error(f"[v1] Fallback returned no data for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("date")
    logger.info(f"[v1] Fetched {len(df)} rows via fallback for {symbol}")
    return df
