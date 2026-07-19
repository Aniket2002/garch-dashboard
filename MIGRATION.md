# Migration from the current repository

Copy every file in this bundle into the repository root and replace files with matching paths.

Delete the obsolete HTML reporting template because PDF generation now happens in memory through `fpdf2`:

```bash
git rm reports/templates/report_template.html
```

The old dependencies `weasyprint`, `jinja2` and `logger` are no longer used. The new `requirements.txt` replaces the existing file completely.

Run:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q --cov=backtest --cov=data --cov=models --cov=reporting --cov-report=term-missing --cov-fail-under=80
streamlit run streamlit_app.py
```

Suggested commit:

```bash
git add .
git commit -m "refactor: correct GARCH VaR and ES backtesting"
git push
```
