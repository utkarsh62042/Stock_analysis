import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config.settings import BACKTEST_CONFIG
from evaluation.metrics import compute_fold_metrics
from utils.logger import get_logger

logger = get_logger(__name__)


def run_backtest(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.Series,
) -> list[dict]:
    """
    Walk-forward backtester with an expanding training window.

    For each fold, the model is re-fitted on [0 : train_end] and evaluated
    on [train_end : train_end + step_size].

    Bug 3 root cause: the step_size and n_splits come from BACKTEST_CONFIG
    and are not validated against the actual dataset length. For the last
    fold, train_end exceeds n, making the test slice empty. The empty arrays
    are forwarded to compute_fold_metrics(), which crashes with an IndexError
    when it tries to access y_true[0].
    """
    n = len(X)
    initial_train = BACKTEST_CONFIG["initial_train_size"]
    step          = BACKTEST_CONFIG["step_size"]
    n_splits      = BACKTEST_CONFIG["n_splits"]

    logger.info(
        f"Backtesting — dataset: {n} rows | "
        f"initial_train: {initial_train} | step: {step} | folds: {n_splits}"
    )

    results: list[dict] = []

    for fold in range(n_splits):
        train_end  = initial_train + fold * step
        test_start = train_end
        test_end   = test_start + step

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test  = X.iloc[test_start:test_end]
        y_test  = y.iloc[test_start:test_end]

        logger.info(
            f"Fold {fold:02d} — train rows: {len(X_train)} | "
            f"test rows: {len(X_test)} | "
            f"test range: [{test_start}:{test_end}]"
        )

        if len(X_train) == 0:
            logger.warning(f"Fold {fold:02d}: empty training set — skipping.")
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_metrics = compute_fold_metrics(y_test.values, y_pred, fold)
        results.append(fold_metrics)

    logger.info(f"Backtesting complete — {len(results)} folds evaluated.")
    return results
