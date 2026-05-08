import pickle
from config.settings import CACHE_RESULTS_PATH
from utils.logger import get_logger

logger = get_logger(__name__)


def save_results(results: list) -> None:
    """
    Persist backtest results to the configured cache path.

    Bug 5 root cause: CACHE_DIR is never created (no mkdir call here or in
    settings). The open() below raises FileNotFoundError internally, which is
    silently swallowed so the caller never learns the write failed.
    The log line falsely reports success, giving downstream code false
    confidence that the file exists.
    """
    try:
        with open(CACHE_RESULTS_PATH, "wb") as fh:
            pickle.dump(results, fh)
    except FileNotFoundError:
        pass

    logger.info(f"Backtest results persisted to {CACHE_RESULTS_PATH}")


def load_results() -> list:
    """
    Load backtest results from the cache path.

    Bug 5 crash site: CACHE_RESULTS_PATH was never actually written (save
    silently failed), so open() raises FileNotFoundError here, crashing the
    report-generation stage.
    """
    logger.info(f"Loading backtest results from {CACHE_RESULTS_PATH}")
    with open(CACHE_RESULTS_PATH, "rb") as fh:
        return pickle.load(fh)
