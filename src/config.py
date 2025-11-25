from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    """Central place for configurable paths and defaults."""

    data_dir: Path = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
    news_path: Path = Path(os.getenv("NEWS_CSV", "news/headlines.csv"))
    prices_dir: Path = Path(os.getenv("PRICES_DIR", "prices"))
    reports_dir: Path = Path(os.getenv("REPORTS_DIR", "reports"))
    default_ticker: str = os.getenv("DEFAULT_TICKER", "AAPL")
    lookback_days: int = int(os.getenv("YFINANCE_LOOKBACK", "365"))

    def resolve_news_path(self) -> Path:
        return (self.data_dir / self.news_path).resolve()

    def resolve_prices_path(self, ticker: str | None = None) -> Path:
        symbol = (ticker or self.default_ticker).upper()
        return (self.data_dir / self.prices_dir / f"{symbol}.csv").resolve()

    def ensure_dirs(self) -> None:
        for directory in [self.data_dir, self.data_dir / self.prices_dir, Path(self.reports_dir)]:
            Path(directory).mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
