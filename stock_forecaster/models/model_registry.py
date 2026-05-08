import pickle

from config.settings import MODEL_STORE_PATH
from utils.logger import get_logger

logger = get_logger(__name__)

_registry: dict = {}


def register_model(model, name: str = "forecaster") -> None:
    """
    Persist a trained model to the in-process registry and to disk.

    Bug 4 root cause: the model is stored under the key 'model', but
    get_model() retrieves the key 'estimator'. The key mismatch means
    get_model() always returns None, causing predictor.py to crash with
    an AttributeError when it calls .predict() on None.
    """
    _registry["model"] = model
    _registry["metadata"] = {"name": name}

    with open(MODEL_STORE_PATH, "wb") as fh:
        pickle.dump(_registry, fh)

    logger.info(f"Model '{name}' registered and persisted to {MODEL_STORE_PATH}.")


def get_model(name: str = "forecaster"):
    """Retrieve a registered model from the registry."""
    if not _registry:
        if not MODEL_STORE_PATH.exists():
            raise FileNotFoundError(
                f"Model registry not found at {MODEL_STORE_PATH}. "
                "Run training before prediction."
            )
        with open(MODEL_STORE_PATH, "rb") as fh:
            _registry.update(pickle.load(fh))

    return _registry.get("estimator")
