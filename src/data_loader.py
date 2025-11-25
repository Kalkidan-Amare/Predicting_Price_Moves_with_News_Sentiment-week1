from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from .config import get_settings


settings = get_settings()


def load_news_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load financial news data and parse timestamps."""
    csv_path = Path(path) if path else settings.resolve_news_path()
    if not csv_path.exists():
        raise FileNotFoundError(f"News dataset not found at {csv_path}. Place headlines CSV under data/news/.")

    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError("Expected 'date' column in news dataset.")

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["headline_length"] = df["headline"].fillna("").str.len()
    df["publisher"] = df["publisher"].fillna("unknown").str.strip()
    df["stock"] = df["stock"].fillna(settings.default_ticker).str.upper()
    return df.dropna(subset=["date"])


def _download_prices(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No price data returned for {ticker} between {start:%Y-%m-%d} and {end:%Y-%m-%d}.")
    data.index.name = "date"
    data = data.reset_index()
    return data.rename(columns=str.lower)


def load_price_data(
    ticker: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    symbol = (ticker or settings.default_ticker).upper()
    if csv_path:
        path = Path(csv_path)
    else:
        path = settings.resolve_prices_path(symbol)

    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        end = end or datetime.utcnow()
        start = start or end - timedelta(days=settings.lookback_days)
        df = _download_prices(symbol, start, end)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols.difference({c.lower() for c in df.columns})
    if missing:
        raise ValueError(f"Price data missing columns: {missing}")

    lower = {c.lower(): c for c in df.columns}
    df = df.rename(columns={lower.get(col, col): col for col in ["date", "open", "high", "low", "close", "volume"]})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date")
