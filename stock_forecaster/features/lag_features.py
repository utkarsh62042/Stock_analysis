import pandas as pd

from config.feature_config import LAG_PERIODS, TARGET_COLUMN
from utils.logger import get_logger

logger = get_logger(__name__)


def compute_lag_features(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> pd.DataFrame:
    """
    Shift the target column by each period defined in LAG_PERIODS.

    Bug 2 root cause: NaN rows introduced by shifting are dropped before
    returning, making this DataFrame shorter than the rolling-feature DataFrame
    (which keeps the full index). When trainer.py tries to stack them as
    numpy arrays, the row-count mismatch raises a ValueError.
    """
    lag_df = pd.DataFrame(index=df.index)

    for lag in LAG_PERIODS:
        lag_df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    before = len(lag_df)
    lag_df = lag_df.dropna().reset_index(drop=True)

    logger.info(
        f"Lag features computed — retained {len(lag_df)} rows "
        f"({before - len(lag_df)} NaN rows dropped) | "
        f"columns: {list(lag_df.columns)}"
    )
    return lag_df
