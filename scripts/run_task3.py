import argparse
from pathlib import Path

import pandas as pd

from src.data_loader import load_news_data, load_price_data
from src.sentiment_correlation import (
    SentimentResult,
    compute_daily_returns,
    compute_daily_sentiment,
    correlate_sentiment_returns,
)


def run(ticker: str, output_dir: Path) -> None:
    news = load_news_data()
    news = news[news["stock"] == ticker.upper()]
    sentiment = compute_daily_sentiment(news)

    prices = load_price_data(ticker)
    returns = compute_daily_returns(prices)

    merged = pd.merge(sentiment, returns, on="date", how="inner")
    results = [
        correlate_sentiment_returns(merged, method="vader"),
        correlate_sentiment_returns(merged, method="blob"),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / f"{ticker}_sentiment_returns.csv", index=False)
    report_path = output_dir / f"{ticker}_correlation.txt"
    lines = [
        f"Method: {res.method}\nPearson: {res.pearson:.4f} (p={res.pearson_p:.4f})\n"
        f"Spearman: {res.spearman:.4f} (p={res.spearman_p:.4f})\n"
        for res in results
    ]
    report_path.write_text("\n".join(lines))
    print("Correlation analysis saved to", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Run Task 3 sentiment-return correlation analysis")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--output", type=Path, default=Path("reports/task3"))
    args = parser.parse_args()
    run(args.ticker.upper(), args.output)


if __name__ == "__main__":
    main()
