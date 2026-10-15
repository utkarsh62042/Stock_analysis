import requests
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

# Alpha Vantage free API endpoint
ALPHA_VANTAGE_API_URL = "https://www.alphavantage.co/query"
ALPHA_VANTAGE_API_KEY = "demo"  # Demo key is heavily rate-limited


def fetch_stock_data(
    symbol: str = "IBM",
    days: int = 500,
) -> pd.DataFrame | None:
    """
    Attempt to fetch real historical stock data from Alpha Vantage API.

    Returns:
        DataFrame with OHLCV data if successful, None if API fails.

    Bug 6 root cause: The demo API key is rate-limited and will frequently
    return 'Note' or 'Error Message' in the response. The code checks for
    these fields but doesn't raise an exception—it simply returns None,
    which is silently handled upstream. The logging doesn't distinguish
    between "API returned error" and "network error", making debugging
    difficult. Additionally, no timeout is set on the request, which can
    cause the process to hang indefinitely if the API is unresponsive.
    """
    try:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "outputsize": "full",
        }

        # Bug 6a: No timeout specified - could hang indefinitely
        logger.info(f"Fetching stock data for {symbol} from Alpha Vantage API...")
        response = requests.get(ALPHA_VANTAGE_API_URL, params=params)
        response.raise_for_status()

        data = response.json()

        # Bug 6b: Rate limit or error responses are silently swallowed
        if "Error Message" in data:
            logger.warning(f"API error: {data['Error Message']}")
            return None
        if "Note" in data:
            logger.warning(f"API rate limited: {data['Note']}")
            return None

        # Bug 6c: Mismatch in expected structure—if API returns unexpected
        # format, the parsing below may silently fail or produce incorrect data
        time_series = data.get("Time Series (Daily)")
        if not time_series:
            logger.warning("API response missing Time Series (Daily) field")
            return None

        records = []
        for date_str, ohlcv in sorted(time_series.items())[:days]:
            records.append({
                "Date": pd.to_datetime(date_str),
                "Open": float(ohlcv["1. open"]),
                "High": float(ohlcv["2. high"]),
                "Low": float(ohlcv["3. low"]),
                "Close": float(ohlcv["4. close"]),
                "Volume": float(ohlcv["5. volume"]),
            })

        if not records:
            logger.warning("API returned no records")
            return None

        df = pd.DataFrame(records)
        logger.info(f"Successfully fetched {len(df)} rows from API")
        return df

    except requests.exceptions.Timeout:
        logger.error(f"API request timed out for {symbol}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error while fetching {symbol}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for {symbol}: {e}")
        return None
    except (KeyError, ValueError) as e:
        # Bug 6d: Silent failure if API response format is unexpected
        logger.error(f"Failed to parse API response: {e}")
        return None
