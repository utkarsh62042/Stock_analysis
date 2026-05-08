import numpy as np
import pandas as pd

from models.model_registry import get_model
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_predictions(X: pd.DataFrame) -> np.ndarray:
    """
    Load the registered forecasting model and produce predictions for X.

    Bug 4 crash site: get_model() returns None because the registry stores
    the model under 'model' but retrieves it under 'estimator'. Calling
    .predict() on None raises:
        AttributeError: 'NoneType' object has no attribute 'predict'
    """
    logger.info(f"Loading model from registry for inference on {len(X)} samples.")

    model = get_model()
    predictions = model.predict(X)

    logger.info(f"Predictions generated — shape: {predictions.shape}")
    return predictions
