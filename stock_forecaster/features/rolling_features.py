import pandas as pd

from config.feature_config import ROLLING_WINDOWS, TARGET_COLUMN
from utils.logger import get_logger

logger = get_logger(__name__)


def compute_rolling_features(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> pd.DataFrame:
    """
    Compute rolling-window statistics (mean, std, min, max) for the target.

    NaN rows from the first (window-1) periods are intentionally retained so
    that the DataFrame length equals len(df). This diverges from lag_features,
    which drops NaN rows, causing a shape mismatch in trainer.py (Bug 2).
    """
    roll_df = pd.DataFrame(index=df.index)
    series = df[target_col]

    for window in ROLLING_WINDOWS:
        roll_df[f"{target_col}_roll_mean_{window}"] = series.rolling(window).mean()
        roll_df[f"{target_col}_roll_std_{window}"]  = series.rolling(window).std()
        roll_df[f"{target_col}_roll_min_{window}"]  = series.rolling(window).min()
        roll_df[f"{target_col}_roll_max_{window}"]  = series.rolling(window).max()

    logger.info(
        f"Rolling features computed — {len(roll_df)} rows (NaN rows retained) | "
        f"columns: {list(roll_df.columns)}"
    )
    return roll_df
