from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud


sns.set_theme(style="whitegrid")


def headline_length_stats(df: pd.DataFrame) -> pd.Series:
    return df["headline_length"].describe(percentiles=[0.1, 0.5, 0.9])


def articles_per_publisher(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    return df.groupby("publisher").size().sort_values(ascending=False).head(top_n)


def publication_trend(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("date").resample("D").size().rename("article_count").to_frame()


def publication_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["weekday"] = temp["date"].dt.day_name()
    temp["hour"] = temp["date"].dt.hour
    pivot = temp.pivot_table(index="weekday", columns="hour", values="headline", aggfunc="count").fillna(0)
    ordered = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex(ordered)
    plt.figure(figsize=(14, 5))
    sns.heatmap(pivot, cmap="mako", linewidths=.5)
    plt.title("Publication Density by Weekday/Hour (UTC)")
    plt.tight_layout()
    return pivot


def keyword_wordcloud(df: pd.DataFrame, max_words: int = 100) -> WordCloud:
    text = " ".join(df["headline"].dropna().tolist())
    wc = WordCloud(width=1200, height=600, background_color="white", max_words=max_words).generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Top Keywords in Headlines")
    return wc


def top_keywords(df: pd.DataFrame, n: int = 20) -> List[Tuple[str, int]]:
    tokens = (
        df["headline"].fillna("")
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", " ", regex=True)
        .str.split()
    )
    counter: Counter = Counter()
    for words in tokens:
        counter.update(w for w in words if len(w) > 3)
    return counter.most_common(n)


def save_plot(fig, path: str) -> None:
    fig.savefig(path, bbox_inches="tight", dpi=150)
