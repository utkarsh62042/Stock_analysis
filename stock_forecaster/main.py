"""
Stock Trend Forecasting Pipeline
=================================
Entry point. Runs all pipeline stages sequentially. Errors bubble up
naturally from each stage; only the outermost try/except catches them,
logs the full traceback, and exits with code 1.

Fix bugs one at a time: each run surfaces exactly one crash.

Stage order and associated bugs
---------------------------------
  Stage 1 — Data Ingestion        (no deliberate bug)
  Stage 2 — Feature Engineering  Bug 1: KeyError      — feature_pipeline.py → loader.py
  Stage 3 — Model Training        Bug 2: ValueError    — trainer.py ← lag_features.py
  Stage 4 — Backtesting           Bug 3: IndexError    — metrics.py ← backtester.py
  Stage 5 — Prediction            Bug 4: AttributeError — predictor.py ← model_registry.py
  Stage 6 — Report Generation     Bug 5: FileNotFoundError — report.py ← cache.py
"""

import sys
import traceback

from data.loader import load_data
from data.validator import validate
from evaluation.backtester import run_backtest
from evaluation.report import generate_report
from features.feature_pipeline import build_feature_matrix
from models.predictor import generate_predictions
from models.trainer import train
from utils.cache import save_results
from utils.logger import get_logger

logger = get_logger("main")

_DIVIDER = "=" * 70


def run_pipeline() -> None:
    logger.info(_DIVIDER)
    logger.info("STOCK TREND FORECASTING PIPELINE — START")
    logger.info(_DIVIDER)

    # ------------------------------------------------------------------
    # Stage 1: Data Ingestion
    # ------------------------------------------------------------------
    logger.info("[STAGE 1/6] Data Ingestion — START")
    df = load_data()
    df = validate(df)
    logger.info("[STAGE 1/6] Data Ingestion — COMPLETE")

    # ------------------------------------------------------------------
    # Stage 2: Feature Engineering
    # Bug 1 surfaces here: KeyError('Close') from feature_pipeline.py
    # ------------------------------------------------------------------
    logger.info("[STAGE 2/6] Feature Engineering — START")
    lag_df, rolling_df, target = build_feature_matrix(df)
    logger.info("[STAGE 2/6] Feature Engineering — COMPLETE")

    # ------------------------------------------------------------------
    # Stage 3: Model Training
    # Bug 2 surfaces here: ValueError (shape mismatch) from trainer.py
    # ------------------------------------------------------------------
    logger.info("[STAGE 3/6] Model Training — START")
    model, X, y = train(lag_df, rolling_df, target)
    logger.info("[STAGE 3/6] Model Training — COMPLETE")

    # ------------------------------------------------------------------
    # Stage 4: Backtesting
    # Bug 3 surfaces here: IndexError (empty fold) from metrics.py
    # ------------------------------------------------------------------
    logger.info("[STAGE 4/6] Backtesting — START")
    backtest_results = run_backtest(model, X, y)
    save_results(backtest_results)
    logger.info("[STAGE 4/6] Backtesting — COMPLETE")

    # ------------------------------------------------------------------
    # Stage 5: Prediction
    # Bug 4 surfaces here: AttributeError (None.predict) from predictor.py
    # ------------------------------------------------------------------
    logger.info("[STAGE 5/6] Prediction — START")
    X_latest = X.tail(20).reset_index(drop=True)
    predictions = generate_predictions(X_latest)
    logger.info("[STAGE 5/6] Prediction — COMPLETE")

    # ------------------------------------------------------------------
    # Stage 6: Report Generation
    # Bug 5 surfaces here: FileNotFoundError from report.py → cache.py
    # ------------------------------------------------------------------
    logger.info("[STAGE 6/6] Report Generation — START")
    generate_report(predictions)
    logger.info("[STAGE 6/6] Report Generation — COMPLETE")

    logger.info(_DIVIDER)
    logger.info("PIPELINE COMPLETE")
    logger.info(_DIVIDER)


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception:
        logger.error(
            "PIPELINE FAILED — full traceback below:\n%s",
            traceback.format_exc(),
        )
        sys.exit(1)
