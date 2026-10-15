import pandas as pd

from config.feature_config import TARGET_COLUMN
from features.lag_features import compute_lag_features
from features.rolling_features import compute_rolling_features
from utils.logger import get_logger

logger = get_logger(__name__)


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Orchestrate lag and rolling feature computation and produce the supervised
    target series (next-day close price).

    Returns:
        lag_df     — lag feature DataFrame (NaN rows already dropped)
        rolling_df — rolling feature DataFrame (full length, NaN rows retained)
        target     — pd.Series of next-day close prices aligned to df.index

    Bug 1 crash site: 'Close' column does not exist because loader.py renamed
    it to 'close_price'. The KeyError propagates to main.py and terminates
    the pipeline at Stage 2 (Feature Engineering).
    """
    logger.info("Starting feature matrix construction.")

    target = df["Close"].shift(-1).rename("target")

    lag_df = compute_lag_features(df, target_col=TARGET_COLUMN)
    rolling_df = compute_rolling_features(df, target_col=TARGET_COLUMN)

    logger.info(
        f"Feature construction complete — "
        f"lag: {lag_df.shape}, rolling: {rolling_df.shape}"
    )
    return lag_df, rolling_df, target
