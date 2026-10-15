import numpy as np
import pandas as pd

from config.settings import DATA_DIR, SAMPLE_DATA_PATH, FETCH_REAL_DATA, API_TIMEOUT_SECONDS
from data.fetcher import fetch_stock_data
from utils.logger import get_logger

logger = get_logger(__name__)


def _generate_synthetic_ohlcv(n_days: int = 500) -> pd.DataFrame:
    """Simulate n_days of OHLCV data using a geometric random walk."""
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=n_days, freq="B")

    log_returns = np.random.normal(loc=0.0003, scale=0.015, size=n_days)
    close = 150.0 * np.exp(np.cumsum(log_returns))

    noise = lambda scale: np.abs(np.random.normal(0, scale, n_days))
    high = close * (1 + noise(0.008))
    low = close * (1 - noise(0.008))
    open_ = close.copy()
    open_[1:] = close[:-1] * (1 + np.random.normal(0, 0.004, n_days - 1))
    volume = np.random.randint(500_000, 8_000_000, n_days).astype(float)

    return pd.DataFrame({
        "Date": dates,
        "Open": np.round(open_, 4),
        "High": np.round(high, 4),
        "Low": np.round(low, 4),
        "Close": np.round(close, 4),
        "Volume": volume,
    })


def load_data() -> pd.DataFrame:
    """
    Load OHLCV data from API or disk, generating synthetic data as fallback.

    Bug 1 root cause: 'Close' is renamed to 'close_price' here for internal
    consistency, but feature_pipeline.py still references the original column
    name 'Close', raising a KeyError when features are built.

    Bug 6 integration: Attempts to fetch real data via API if FETCH_REAL_DATA
    is True. If the API fails (which it will with the demo key), the error is
    silently caught and sample data is used as a fallback. This demonstrates
    silent error handling where the pipeline doesn't clearly communicate that
    the API attempt failed.
    """
    df = None

    # Bug 6: Try to fetch real data if enabled
    if FETCH_REAL_DATA:
        logger.info("Attempting to fetch real stock data from Alpha Vantage API...")
        df = fetch_stock_data(symbol="IBM", days=500)
        if df is not None:
            logger.info(f"Successfully loaded real data: {len(df)} rows")
        else:
            logger.warning(
                "API data fetch failed or timed out — falling back to sample data. "
                "(This is expected with the demo API key.)"
            )

    # Fallback to sample data if API fetch failed or was disabled
    if df is None:
        if not SAMPLE_DATA_PATH.exists():
            logger.info("sample_data.csv not found — generating synthetic OHLCV data.")
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            df_raw = _generate_synthetic_ohlcv()
            df_raw.to_csv(SAMPLE_DATA_PATH, index=False)
            logger.info(
                f"Synthetic data written to {SAMPLE_DATA_PATH} ({len(df_raw)} rows)."
            )

        df = pd.read_csv(SAMPLE_DATA_PATH, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    df = df.rename(columns={"Close": "close_price"})

    logger.info(f"Data loaded: {len(df)} rows | columns: {list(df.columns)}")
    return df
