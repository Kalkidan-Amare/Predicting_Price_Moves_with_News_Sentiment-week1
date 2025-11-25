import argparse
from pathlib import Path

import pandas as pd

from src import eda
from src.data_loader import load_news_data


def run(output_dir: Path) -> None:
    df = load_news_data()
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = eda.headline_length_stats(df)
    stats.to_csv(output_dir / "headline_length_stats.csv")

    publishers = eda.articles_per_publisher(df, top_n=20)
    publishers.to_csv(output_dir / "top_publishers.csv")

    trend = eda.publication_trend(df)
    trend.to_csv(output_dir / "publication_trend.csv")

    pivot = eda.publication_heatmap(df)
    pivot.to_csv(output_dir / "publication_heatmap.csv")

    eda.keyword_wordcloud(df)
    print("EDA artifacts saved to", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Run Task 1 EDA pipeline")
    parser.add_argument("--output", type=Path, default=Path("reports/task1"), help="Directory to store outputs")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
