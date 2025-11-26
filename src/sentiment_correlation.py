from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Tuple

import pandas as pd
from scipy.stats import pearsonr, spearmanr
try:  # pragma: no cover - optional dependency
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ModuleNotFoundError:  # pragma: no cover
    SentimentIntensityAnalyzer = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from textblob import TextBlob
except ModuleNotFoundError:  # pragma: no cover
    TextBlob = None  # type: ignore[assignment]


_ANALYZER = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
_SENT_WARNED = False


@dataclass
class SentimentResult:
    method: str
    pearson: float
    pearson_p: float
    spearman: float
    spearman_p: float


def score_headline(headline: str) -> Tuple[float, float]:
    text = headline or ""
    global _SENT_WARNED
    if _ANALYZER is None or TextBlob is None:
        if not _SENT_WARNED:
            warnings.warn(
                "Sentiment libraries not installed; defaulting scores to 0. "
                "Install `vaderSentiment` and `textblob` for full functionality.",
                RuntimeWarning,
            )
            _SENT_WARNED = True
        return 0.0, 0.0
    vader = _ANALYZER.polarity_scores(text)["compound"]
    blob = TextBlob(text).sentiment.polarity
    return vader, blob


def compute_daily_sentiment(df_news: pd.DataFrame) -> pd.DataFrame:
    if df_news.empty:
        return pd.DataFrame(columns=["date", "stock", "sent_vader", "sent_blob"])
    sentiments = df_news["headline"].apply(score_headline)
    df_news[["sent_vader", "sent_blob"]] = list(sentiments)
    df_news["date_only"] = df_news["date"].dt.tz_convert("UTC").dt.floor("D")
    grouped = (
        df_news.groupby(["date_only", "stock"])[["sent_vader", "sent_blob"]]
        .mean()
        .reset_index()
    )
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
