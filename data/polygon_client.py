from __future__ import annotations

import os
import re
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, BinaryIO, TextIO
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

BASE_URL = "https://api.polygon.io"
_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9:._/-]{1,32}$")
_STANDARD_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataProviderError(RuntimeError):
    """Raised when market data cannot be retrieved or normalized."""


def build_retrying_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    return session


def _validate_request(symbol: str, start: str, end: str) -> tuple[str, str, str]:
    normalized_symbol = symbol.strip().upper()
    if not _SYMBOL_PATTERN.fullmatch(normalized_symbol):
        raise ValueError("symbol contains unsupported characters")

    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if pd.isna(start_timestamp) or pd.isna(end_timestamp):
        raise ValueError("start and end must be valid dates")
    if start_timestamp > end_timestamp:
        raise ValueError("start date must not be after end date")
    return (
        normalized_symbol,
        start_timestamp.date().isoformat(),
        end_timestamp.date().isoformat(),
    )


def _response_message(response: requests.Response, *, secret: str) -> str:
    try:
        payload = response.json()
    except ValueError:
        message = response.text[:300]
    else:
        message = str(payload.get("message") or payload.get("error") or payload)[:300]
    return message.replace(secret, "[REDACTED]")


def fetch_data(
    symbol: str,
    start: str,
    end: str,
    *,
    api_key: str | None = None,
    session: requests.Session | None = None,
    timeout: float = 20.0,
    max_pages: int = 20,
) -> pd.DataFrame:
    """Fetch adjusted daily Polygon aggregates with pagination and retries.

    ``POLYGON_API_KEY`` is used only when ``api_key`` is omitted. An explicitly
    blank key is invalid. Keys are never written to logs or returned in errors.
    """

    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")

    normalized_symbol, start_date, end_date = _validate_request(symbol, start, end)
    key_candidate = os.getenv("POLYGON_API_KEY") if api_key is None else api_key
    resolved_key = key_candidate.strip() if key_candidate else ""
    if not resolved_key:
        raise DataProviderError(
            "Polygon API key is missing. Set POLYGON_API_KEY or provide a key in the dashboard."
        )

    client = session or build_retrying_session()
    url = (
        f"{BASE_URL}/v2/aggs/ticker/{normalized_symbol}/range/1/day/"
        f"{start_date}/{end_date}"
    )
    params: dict[str, Any] | None = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50_000,
        "apiKey": resolved_key,
    }
    rows: list[dict[str, Any]] = []

    for _ in range(max_pages):
        try:
            response = client.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            raise DataProviderError("Polygon request failed due to a network error") from exc
        if response.status_code in {401, 403}:
            raise DataProviderError(
                f"Polygon rejected the request ({response.status_code}): "
                f"{_response_message(response, secret=resolved_key)}"
            )
        if response.status_code == 429:
            raise DataProviderError("Polygon rate limit was exceeded after retry attempts")
        if not response.ok:
            raise DataProviderError(
                f"Polygon request failed ({response.status_code}): "
                f"{_response_message(response, secret=resolved_key)}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise DataProviderError("Polygon returned invalid JSON") from exc

        rows.extend(payload.get("results") or [])
        next_url = payload.get("next_url")
        if not next_url:
            break
        url = urljoin(BASE_URL, str(next_url))
        params = {"apiKey": resolved_key}
    else:
        raise DataProviderError(f"Polygon pagination exceeded max_pages={max_pages}")

    if not rows:
        raise DataProviderError(
            f"Polygon returned no daily bars for {normalized_symbol} in the selected period"
        )

    raw = pd.DataFrame.from_records(rows).rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )
    if "timestamp" not in raw.columns:
        raise DataProviderError("Polygon response did not contain timestamps")
    raw["date"] = pd.to_datetime(raw["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    return normalize_price_frame(raw, date_column="date")


def normalize_price_frame(
    frame: pd.DataFrame,
    *,
    date_column: str | None = None,
) -> pd.DataFrame:
    """Normalize an OHLCV-like frame to a sorted, unique DatetimeIndex."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataProviderError("price data is empty")

    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    aliases = {
        "adj close": "close",
        "adj_close": "close",
        "adjusted_close": "close",
    }
    if "date" not in normalized.columns:
        if "datetime" in normalized.columns:
            aliases["datetime"] = "date"
        elif "timestamp" in normalized.columns:
            aliases["timestamp"] = "date"
    normalized = normalized.rename(columns=aliases)

    resolved_date_column = date_column.lower() if date_column else None
    if resolved_date_column and resolved_date_column in normalized.columns:
        index = pd.to_datetime(normalized.pop(resolved_date_column), errors="coerce", utc=True)
    elif "date" in normalized.columns:
        index = pd.to_datetime(normalized.pop("date"), errors="coerce", utc=True)
    elif isinstance(normalized.index, pd.DatetimeIndex):
        index = pd.to_datetime(normalized.index, errors="coerce", utc=True)
    else:
        raise DataProviderError("price data requires a date column or DatetimeIndex")

    if "close" not in normalized.columns:
        raise DataProviderError("price data requires a close column")

    normalized.index = pd.DatetimeIndex(index).tz_convert(None)
    normalized.index.name = "date"
    for column in _STANDARD_COLUMNS:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.replace([np.inf, -np.inf], np.nan)
    normalized = normalized.loc[~normalized.index.isna()]
    normalized = normalized.loc[normalized["close"] > 0]
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
    available_columns = [column for column in _STANDARD_COLUMNS if column in normalized.columns]
    normalized = normalized[available_columns].dropna(subset=["close"])

    if normalized.empty:
        raise DataProviderError("no valid positive close prices remain after normalization")
    return normalized


def load_price_csv(
    source: str | Path | bytes | BinaryIO | TextIO,
) -> pd.DataFrame:
    """Load a user-supplied CSV containing at least date and close columns."""

    if isinstance(source, bytes):
        csv_source: Any = BytesIO(source)
    elif isinstance(source, str) and "\n" in source:
        csv_source = StringIO(source)
    else:
        csv_source = source

    try:
        frame = pd.read_csv(csv_source)
    except Exception as exc:
        raise DataProviderError(f"could not read CSV: {exc}") from exc
    return normalize_price_frame(frame)


def compute_log_returns(prices: pd.DataFrame | pd.Series) -> pd.Series:
    """Compute close-to-close log returns in decimal units."""

    close = prices["close"] if isinstance(prices, pd.DataFrame) else prices
    numeric_close = pd.to_numeric(close, errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric_close = numeric_close.dropna()
    if len(numeric_close) < 2:
        raise ValueError("at least two valid close prices are required")
    if bool((numeric_close <= 0).any()):
        raise ValueError("close prices must be strictly positive")

    returns = np.log(numeric_close / numeric_close.shift(1)).dropna()
    returns.name = "return"
    return returns.astype(float)
