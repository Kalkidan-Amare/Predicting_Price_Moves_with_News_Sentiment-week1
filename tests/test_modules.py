import pandas as pd

from src import eda
from src.technical_indicators import compute_indicators
from src.sentiment_correlation import compute_daily_returns


def test_headline_length_stats():
    df = pd.DataFrame({"headline_length": [10, 15, 20]})
    stats = eda.headline_length_stats(df)
    assert stats["mean"] == 15


def test_compute_indicators_columns():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    price_df = pd.DataFrame({
        "date": dates,
        "open": range(60),
        "high": range(1, 61),
        "low": range(60),
        "close": range(60),
        "volume": [1_000_000] * 60,
    })
    result = compute_indicators(price_df)
    assert {"sma_20", "rsi_14", "macd"}.issubset(result.columns)


def test_daily_returns_shape():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5, freq="D"),
        "close": [100, 101, 102, 103, 104],
    })
    returns = compute_daily_returns(df)
    assert len(returns) == 4
