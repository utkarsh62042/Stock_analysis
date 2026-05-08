from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import REPORTS_DIR
from utils.cache import load_results
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_report(predictions: np.ndarray | None = None) -> None:
    """
    Load cached backtest results and render an evaluation report (CSV + PNG).

    Bug 5 crash site: load_results() opens CACHE_RESULTS_PATH, which was
    never actually written because cache.save_results() silently swallowed
    the FileNotFoundError (the cache directory was never created). This
    raises:
        FileNotFoundError: [Errno 2] No such file or directory: '...backtest_results.pkl'
    """
    logger.info("Loading backtest results from cache for report generation.")
    results = load_results()

    df = pd.DataFrame(results)
    logger.info(f"Loaded {len(df)} backtest fold(s).")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = REPORTS_DIR / "backtest_summary.csv"
    df.to_csv(summary_path, index=False)
    logger.info(f"Backtest summary saved to {summary_path}.")

    metrics = ["mae", "rmse", "direction_accuracy"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))

    for ax, metric in zip(axes, metrics):
        ax.bar(df["fold"], df[metric], color="#4C72B0")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric)
        ax.set_xticks(df["fold"])

    plt.suptitle("Walk-Forward Backtest Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()

    chart_path = REPORTS_DIR / "backtest_metrics.png"
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    logger.info(f"Metrics chart saved to {chart_path}.")

    divider = "=" * 64
    print(f"\n{divider}")
    print("  BACKTEST SUMMARY")
    print(divider)
    print(df.to_string(index=False))
    print(f"{divider}\n")

    if predictions is not None:
        pred_path = REPORTS_DIR / "latest_predictions.csv"
        pd.Series(predictions, name="predicted_close").to_csv(pred_path, index=False)
        logger.info(f"Latest predictions saved to {pred_path}.")
