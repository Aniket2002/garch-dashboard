# garch-dashboard

Streamlit dashboard for conditional volatility and tail-risk backtesting with GARCH models.

Current implementation focus:
- Daily return modeling with GARCH volatility forecasts.
- Parametric VaR and Expected Shortfall (ES) in return space.
- Walk-forward breach diagnostics with hit-rate and Kupiec UC test summary.
- Plotly visualizations and PDF export.

Notes:
- This app is historical-data driven and should not be described as real-time risk monitoring.
- VaR/ES plots are shown on return units (not overlaid with price units).
