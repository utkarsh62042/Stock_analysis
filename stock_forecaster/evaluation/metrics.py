import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


def compute_fold_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, fold: int
) -> dict:
    """
    Compute MAE, RMSE, and directional accuracy for a single backtest fold.

    Bug 3 crash site: when the last fold is empty (y_true has 0 elements
    because the backtester's hardcoded step_size overshot the dataset length),
    accessing y_true[0] raises:
        IndexError: index 0 is out of bounds for axis 0 with size 0
    """
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    baseline_price = y_true[0]

    if len(y_true) > 1:
        directional_hits = int(
            np.sum(
                np.sign(y_pred[1:] - y_true[:-1])
                == np.sign(y_true[1:] - y_true[:-1])
            )
        )
        direction_accuracy = directional_hits / (len(y_true) - 1)
    else:
        direction_accuracy = float("nan")

    logger.info(
        f"Fold {fold:02d} — MAE: {mae:.4f} | RMSE: {rmse:.4f} | "
        f"Dir-Acc: {direction_accuracy:.4f} | Baseline: {baseline_price:.4f}"
    )
    return {
        "fold": fold,
        "n_samples": len(y_true),
        "mae": mae,
        "rmse": rmse,
        "direction_accuracy": direction_accuracy,
        "baseline_price": float(baseline_price),
    }
