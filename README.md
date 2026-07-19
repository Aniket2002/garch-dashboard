# GARCH Risk Dashboard

A reproducible Streamlit application for one-day conditional-volatility, Value-at-Risk (VaR) and expected-shortfall (ES) analysis using rolling GARCH-family models.

The project is designed as an educational risk-modelling implementation. It is not a trading system, a guarantee of future losses or evidence of market calibration.

## What the dashboard does

- loads adjusted daily prices from Polygon or a user-supplied CSV;
- computes close-to-close log returns in decimal units;
- fits GARCH(1,1) or GJR-GARCH(1,1) with Normal or standardized Student-t innovations;
- produces one-day conditional mean, variance, VaR and ES forecasts;
- performs a strict rolling out-of-sample backtest with no look-ahead;
- reports Kupiec unconditional-coverage, Christoffersen independence and conditional-coverage tests;
- shows price, forecast-volatility and return-risk charts;
- exports the backtest as CSV and a compact in-memory PDF report.

## Important modelling corrections

The implementation keeps all quantities dimensionally consistent:

```text
VaR return threshold = conditional mean + conditional volatility × innovation quantile
ES return threshold  = conditional mean + conditional volatility × conditional tail mean
```

A breach occurs when:

```text
realized return < VaR return threshold
```

Price levels are plotted separately from return thresholds. The dashboard does not overlay a price series with a return VaR series.

For Student-t models, the quantile and partial first moment are taken from the fitted standardized innovation distribution exposed by `arch`. This avoids applying an unstandardized Student-t formula to unit-variance residuals.

## Installation

CI is configured for Python 3.10, 3.11 and 3.12.

```bash
python -m venv .venv
```

Activate the environment, then install:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Data sources

### Polygon

Copy `.env.example` to `.env` and add your key:

```text
POLYGON_API_KEY=your_key_here
```

The client uses the daily aggregate endpoint, pagination, timeouts and retry handling. It does not log the API key and does not make one request per calendar day.

### CSV upload

The dashboard accepts a CSV with at least:

```text
date,close
2024-01-02,185.64
```

The loader also accepts common names such as `Adj Close`, and optional OHLCV columns.

## Run the dashboard

```bash
streamlit run streamlit_app.py
```

Rolling backtests refit the model for every forecast day. A larger estimation window or more forecast days therefore increases runtime.

## Tests and linting

```bash
python -m ruff check .
```

```bash
python -m pytest -q \
  --cov=backtest \
  --cov=data \
  --cov=models \
  --cov=reporting \
  --cov-report=term-missing \
  --cov-fail-under=80
```

The tests cover:

- closed-form Normal VaR and ES conversion;
- GARCH fitting and one-step forecasting;
- strict rolling-window timing;
- breach classification;
- Kupiec and Christoffersen statistics;
- Polygon pagination and CSV normalization;
- in-memory PDF generation.

## Statistical conventions

Returns are decimal log returns. The `arch` estimator receives percentage returns for numerical conditioning, while forecasts are converted back to decimal units.

VaR is represented as a lower return threshold. For example, a VaR threshold of `-0.025` means a one-day return below `-2.5%` is classified as a breach.

ES is the conditional mean return below the model-implied VaR quantile. It should be less than or equal to the VaR threshold in the lower tail.

The coverage tests assess different properties:

- **Kupiec:** whether the unconditional breach frequency matches the chosen tail probability;
- **Christoffersen independence:** whether breaches appear clustered;
- **Conditional coverage:** the joint unconditional-coverage and independence test.

A high p-value does not prove the model is correct. It only means the particular null hypothesis is not rejected at the selected significance level.

## Repository structure

```text
garch-dashboard/
├── backtest/
│   └── backtester.py
├── data/
│   └── polygon_client.py
├── models/
│   └── garch_model.py
├── reporting/
│   └── pdf_report.py
├── tests/
├── .github/workflows/ci.yml
├── .env.example
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── streamlit_app.py
└── README.md
```

## Limitations

- Daily data and one-day forecasts only.
- No multivariate volatility, realized-volatility inputs or intraday dynamics.
- No transaction-cost or portfolio-P&L model.
- Rolling estimation is computationally expensive because each forecast is refitted.
- Coverage tests have limited power in small samples and at very low tail probabilities.
- Polygon availability and entitlements depend on the user's own account.
- Results are conditional on the selected history, model family, distribution and window.

## Suggested CV wording

> Developed a Python GARCH risk dashboard with rolling out-of-sample volatility, VaR and expected-shortfall forecasts, Student-t tail modelling, Kupiec and Christoffersen backtests, robust market-data ingestion and downloadable risk reports.
