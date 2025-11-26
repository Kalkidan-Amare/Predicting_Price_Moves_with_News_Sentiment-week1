from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
try:
    from textblob import TextBlob
except ModuleNotFoundError:  # pragma: no cover
    TextBlob = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ModuleNotFoundError:  # pragma: no cover
    SentimentIntensityAnalyzer = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    if SentimentIntensityAnalyzer is None:
        raise ImportError(
            "vaderSentiment is not installed. Install it via `pip install vaderSentiment` to run sentiment scoring."
        )
    return SentimentIntensityAnalyzer()


analyzer = _get_analyzer() if SentimentIntensityAnalyzer else None


@dataclass
class SentimentResult:
    method: str
    pearson: float
    pearson_p: float
    spearman: float
    spearman_p: float


def score_headline(headline: str) -> Tuple[float, float]:
    if analyzer is None or TextBlob is None:
        raise ImportError(
            "Sentiment libraries missing. Install `textblob` and `vaderSentiment` to score headlines."
        )
    text = headline or ""
    vader = analyzer.polarity_scores(text)["compound"]
    blob = TextBlob(text).sentiment.polarity
    return vader, blob


def compute_daily_sentiment(df_news: pd.DataFrame) -> pd.DataFrame:
    sentiments = df_news["headline"].apply(score_headline)
    df_news["sent_vader"], df_news["sent_blob"] = zip(*sentiments)
    df_news["date_only"] = df_news["date"].dt.tz_convert("UTC").dt.floor("D")
    grouped = df_news.groupby(["date_only", "stock"])[["sent_vader", "sent_blob"]].mean().reset_index()
    grouped = grouped.rename(columns={"date_only": "date"})
    return grouped


def compute_daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    df["return"] = df["close"].pct_change()
    return df[["date", "return"]].dropna()


def correlate_sentiment_returns(merged: pd.DataFrame, method: Literal["vader", "blob"] = "vader") -> SentimentResult:
    if merged.empty:
        raise ValueError("Merged dataset empty. Check date alignment.")
    col = f"sent_{method}"
    merged = merged.dropna(subset=[col, "return"])
    pearson, pearson_p = pearsonr(merged[col], merged["return"])
    spearman, spearman_p = spearmanr(merged[col], merged["return"])
    return SentimentResult(method, float(pearson), float(pearson_p), float(spearman), float(spearman_p))
