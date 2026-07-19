import os
import logging
from weasyprint import HTML
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

def generate_pdf(
    symbol: str,
    start_date,
    end_date,
    stats: dict,
    template_path: str = "reports/templates/report_template.html",
    output_dir: str = "reports"
) -> str:
    """
    Render the Jinja template and write a PDF backtest report.
    Returns the path to the generated PDF.
    """
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_path)))
    tmpl = env.get_template(os.path.basename(template_path))
    html_out = tmpl.render(
        title=f"{symbol} GARCH Backtest",
        start_date=start_date,
        end_date=end_date,
        stats=stats,
        plot_volatility="",  # placeholder for embedded image
        plot_var=""
    )

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{symbol}_backtest.pdf")
    HTML(string=html_out).write_pdf(pdf_path)
    logger.info(f"PDF report saved to {pdf_path}")
    return pdf_path
