from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from data.polygon_client import (
    DataProviderError,
    compute_log_returns,
    fetch_data,
    load_price_csv,
    normalize_price_frame,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any] | None, float]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None,
        timeout: float,
    ) -> FakeResponse:
        self.calls.append((url, params, timeout))
        return self.responses.pop(0)


def test_normalize_csv_and_compute_log_returns():
    csv = "date,Adj Close,Volume\n2024-01-02,100,10\n2024-01-01,99,9\n2024-01-02,101,11\n"
    frame = load_price_csv(csv)
    returns = compute_log_returns(frame)

    assert list(frame.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert frame.loc[pd.Timestamp("2024-01-02"), "close"] == pytest.approx(101.0)
    assert returns.iloc[0] == pytest.approx(np.log(101.0 / 99.0))


def test_normalizer_rejects_missing_close():
    with pytest.raises(DataProviderError, match="close"):
        normalize_price_frame(pd.DataFrame({"date": ["2024-01-01"], "open": [1.0]}))


def test_polygon_fetch_paginates_and_normalizes():
    first = FakeResponse(
        {
            "results": [
                {"t": 1704067200000, "o": 99, "h": 102, "l": 98, "c": 100, "v": 10}
            ],
            "next_url": "https://api.polygon.io/next-page",
        }
    )
    second = FakeResponse(
        {
            "results": [
                {"t": 1704153600000, "o": 100, "h": 103, "l": 99, "c": 102, "v": 11}
            ]
        }
    )
    session = FakeSession([first, second])

    frame = fetch_data(
        "aapl",
        "2024-01-01",
        "2024-01-02",
        api_key="secret",
        session=session,  # type: ignore[arg-type]
    )

    assert len(frame) == 2
    assert frame.index.name == "date"
    assert frame.iloc[-1]["close"] == pytest.approx(102.0)
    assert len(session.calls) == 2
    assert session.calls[1][1] == {"apiKey": "secret"}


def test_polygon_requires_key():
    with pytest.raises(DataProviderError, match="API key"):
        fetch_data("AAPL", "2024-01-01", "2024-01-02", api_key="")


def _single_page_session() -> FakeSession:
    return FakeSession(
        [
            FakeResponse(
                {
                    "results": [
                        {
                            "t": 1704067200000,
                            "o": 99,
                            "h": 102,
                            "l": 98,
                            "c": 100,
                            "v": 10,
                        }
                    ]
                }
            )
        ]
    )


def test_polygon_uses_environment_key_when_argument_is_omitted(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "environment-key")
    session = _single_page_session()

    fetch_data("AAPL", "2024-01-01", "2024-01-01", session=session)  # type: ignore[arg-type]

    assert session.calls[0][1]["apiKey"] == "environment-key"


def test_polygon_explicit_key_wins_over_environment(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "environment-key")
    session = _single_page_session()

    fetch_data(  # type: ignore[arg-type]
        "AAPL",
        "2024-01-01",
        "2024-01-01",
        api_key=" explicit-key ",
        session=session,
    )

    assert session.calls[0][1]["apiKey"] == "explicit-key"


@pytest.mark.parametrize("explicit_key", ["", "   \t"])
def test_polygon_explicit_blank_key_does_not_fall_back_to_environment(
    monkeypatch, explicit_key
):
    monkeypatch.setenv("POLYGON_API_KEY", "environment-key")

    with pytest.raises(DataProviderError, match="API key"):
        fetch_data("AAPL", "2024-01-01", "2024-01-01", api_key=explicit_key)


def test_polygon_errors_do_not_expose_api_key():
    api_key = "top-secret-api-key"
    session = FakeSession(
        [FakeResponse({"message": f"Invalid apiKey={api_key}"}, status_code=401)]
    )

    with pytest.raises(DataProviderError) as exc_info:
        fetch_data(  # type: ignore[arg-type]
            "AAPL",
            "2024-01-01",
            "2024-01-01",
            api_key=api_key,
            session=session,
        )

    assert api_key not in str(exc_info.value)
