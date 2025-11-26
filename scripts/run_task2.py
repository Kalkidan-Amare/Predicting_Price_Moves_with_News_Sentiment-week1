import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt

from src.data_loader import load_price_data
from src.technical_indicators import compute_indicators, summarize_indicators


def run(ticker: str, start: str | None, end: str | None, output_dir: Path) -> None:
    start_dt = datetime.fromisoformat(start) if start else None
    end_dt = datetime.fromisoformat(end) if end else None
    prices = load_price_data(ticker, start=start_dt, end=end_dt)
    indicators = compute_indicators(prices)
    output_dir.mkdir(parents=True, exist_ok=True)

    indicators.to_csv(output_dir / f"{ticker}_indicators.csv", index=False)
    summary = summarize_indicators(indicators)
    (output_dir / f"{ticker}_summary.txt").write_text(str(summary))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(indicators["date"], indicators["close"], label="Close", color="black")
    ax.plot(indicators["date"], indicators["sma_20"], label="SMA 20")
    ax.plot(indicators["date"], indicators["sma_50"], label="SMA 50")
    ax.set_title(f"{ticker} Price with Moving Averages")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(output_dir / f"{ticker}_price_ma.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Indicators and plots saved under {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run Task 2 technical indicator pipeline")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--start", help="Start date YYYY-MM-DD", default=None)
    parser.add_argument("--end", help="End date YYYY-MM-DD", default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/task2"))
    args = parser.parse_args()
    run(args.ticker.upper(), args.start, args.end, args.output)


if __name__ == "__main__":
    main()
