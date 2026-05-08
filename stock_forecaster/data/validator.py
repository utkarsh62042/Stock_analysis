import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS: set[str] = {"Date", "Open", "High", "Low", "close_price", "Volume"}
MIN_ROWS: int = 100


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the loaded DataFrame for required columns, minimum length,
    and basic OHLCV integrity.

    Raises:
        ValueError: on schema or data-quality violations.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Dataset too small: {len(df)} rows. Minimum required: {MIN_ROWS}."
        )

    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values detected:\n{null_counts[null_counts > 0]}")

    violations = (df["High"] < df["Low"]).sum()
    if violations:
        raise ValueError(
            f"OHLCV integrity violation: High < Low on {violations} row(s)."
        )

    logger.info(
        f"Validation passed — {len(df)} rows | "
        f"date range: {df['Date'].min().date()} → {df['Date'].max().date()}"
    )
    return df
