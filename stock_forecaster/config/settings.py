from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"

# Bug 5 setup: this directory is never created before use.
CACHE_DIR = BASE_DIR / "cache" / "results"
CACHE_RESULTS_PATH = CACHE_DIR / "backtest_results.pkl"

SAMPLE_DATA_PATH = DATA_DIR / "sample_data.csv"

MODEL_STORE_PATH = BASE_DIR / "model_store.pkl"

MODEL_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 6,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
}

# Bug 3 setup: step_size=50 with n_splits=6 causes the last fold to exceed
# the dataset length (~490 rows after lag-NaN dropping), producing an empty
# test slice that is passed verbatim to metrics.py.
BACKTEST_CONFIG: dict = {
    "initial_train_size": 252,
    "step_size": 50,
    "n_splits": 6,
}
