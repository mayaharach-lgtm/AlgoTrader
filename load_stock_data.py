import yfinance as yf
import pandas as pd

def load_data(ticker: str, period: str = "6mo"):
    """
    Fetch OHLCV data for a single ticker from Yahoo Finance.
    Returns a pandas DataFrame or None if no data.
    """
    try:
        # auto_adjust=True avoids dividend/split adjustments
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data is None or data.empty:
            print(f"⚠️ No data found for {ticker}")
            return None
        return data
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return None


def get_close_prices(data: pd.DataFrame, ticker: str | None = None) -> list[float]:
    """
    Extract a 1-D list of closing prices from a DataFrame that may be:
    - simple single-index columns with 'Close' or 'Adj Close'
    - MultiIndex columns (e.g., ('Close', 'AAPL')), even if a single ticker
    Always returns: list[float]
    """
    if data is None or data.empty:
        return []

    cols = data.columns

    # Case A: simple columns
    if "Close" in cols:
        close = data["Close"]
    elif "Adj Close" in cols:
        close = data["Adj Close"]
    # Case B: MultiIndex columns (e.g., ('Close', 'AAPL'))
    elif isinstance(cols, pd.MultiIndex):
        if ticker is not None:
            try:
                close = data.loc[:, ("Close", ticker)]
            except Exception:
                close = data.loc[:, ("Close", slice(None))].iloc[:, 0]
        else:
            close = data.loc[:, ("Close", slice(None))].iloc[:, 0]
    else:
        raise ValueError("No 'Close' or 'Adj Close' column present in the data.")

    # If still DataFrame (rare), reduce to first column
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    return close.astype(float).tolist()


# Default top 10 tickers if user doesn't provide their own
TOP_10_TICKERS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "GOOGL", # Alphabet (Google)
    "META",  # Meta (Facebook)
    "TSLA",  # Tesla
    "NVDA",  # Nvidia
    "JPM",   # JP Morgan
    "V",     # Visa
    "NFLX"   # Netflix
]
