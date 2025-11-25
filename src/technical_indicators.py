from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

try:
    import talib
except ImportError:  # pragma: no cover
    talib = None


def _moving_average(series: pd.Series, window: int = 20) -> pd.Series:
    if talib:
        return pd.Series(talib.SMA(series.values, window), index=series.index, name=f"sma_{window}")
    return series.rolling(window=window, min_periods=window//2).mean().rename(f"sma_{window}")


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    if talib:
        return pd.Series(talib.RSI(series.values, window), index=series.index, name=f"rsi_{window}")
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename(f"rsi_{window}")


def _macd(series: pd.Series) -> pd.DataFrame:
    if talib:
        macd, signal, hist = talib.MACD(series.values)
        return pd.DataFrame({
            "macd": macd,
            "macd_signal": signal,
            "macd_hist": hist,
        }, index=series.index)
    exp12 = series.ewm(span=12, adjust=False).mean()
    exp26 = series.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return pd.DataFrame({"macd": macd, "macd_signal": signal, "macd_hist": hist})


def compute_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "close"}
    if not required.issubset(price_df.columns):
        raise ValueError(f"Price DataFrame must include {required}")

    df = price_df.copy().set_index("date")
    df["sma_20"] = _moving_average(df["close"], 20)
    df["sma_50"] = _moving_average(df["close"], 50)
    df["rsi_14"] = _rsi(df["close"], 14)
    macd_df = _macd(df["close"])
    df = pd.concat([df, macd_df], axis=1)
    return df.reset_index()


def summarize_indicators(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "latest_close": float(df["close"].iloc[-1]),
        "latest_rsi": float(df["rsi_14"].iloc[-1]),
        "latest_macd_hist": float(df["macd_hist"].iloc[-1]),
    }
