import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config.settings import MODEL_PARAMS
from models.model_registry import register_model
from utils.logger import get_logger

logger = get_logger(__name__)


def train(
    lag_df: pd.DataFrame,
    rolling_df: pd.DataFrame,
    target: pd.Series,
) -> tuple["RandomForestRegressor", pd.DataFrame, pd.Series]:
    """
    Assemble the feature matrix from lag and rolling components, train a
    Random Forest regressor, and persist it to the model registry.

    Returns:
        model — fitted RandomForestRegressor
        X     — assembled feature DataFrame used for training / backtesting
        y     — aligned target Series

    Bug 2 crash site: lag_df has NaN rows dropped (≈490 rows) while
    rolling_df retains the full dataset length (500 rows). np.concatenate
    requires identical first-dimension sizes and raises:
        ValueError: all the input array dimensions for the concatenation axis
        must match exactly, but along dimension 0, the array at index 0 has
        size 490 and the array at index 1 has size 500
    """
    logger.info(
        f"Assembling feature matrix — "
        f"lag: {lag_df.shape}, rolling: {rolling_df.shape}"
    )

    X_array = np.concatenate([lag_df.values, rolling_df.values], axis=1)
    feature_cols = list(lag_df.columns) + list(rolling_df.columns)
    X = pd.DataFrame(X_array, columns=feature_cols)

    y = target.iloc[: len(X)].reset_index(drop=True)

    mask = ~X.isnull().any(axis=1)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    logger.info(f"Training on X: {X.shape}, y: {y.shape}")

    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X, y)

    register_model(model, name="forecaster")
    logger.info("Model training complete — registered to model store.")
    return model, X, y
