# Predicting Price Moves with News Sentiment

Week 1 focuses on wiring an end‑to‑end workflow that ingests financial headlines, profiles the news stream, computes technical indicators, and measures how sentiment nudges daily stock returns. The repository is structured so every task runs from a CLI script, making it easy to regenerate the exact charts and CSVs used in the interim and final submissions.

## Folder Layout
```
Predicting_Price_Moves_with_News_Sentiment-week1/
├── .github/workflows/unittests.yml   # GitHub Actions pipeline
├── .vscode/settings.json             # Recommended editor settings
├── data/                             # Raw + processed data (gitignored)
├── notebooks/                        # Exploratory notebooks (optional)
├── reports/                          # Generated CSVs/PNGs per task
├── scripts/                          # CLI entrypoints
├── src/                              # Reusable modules
├── tests/                            # Pytest suite
├── requirements.txt
└── README.md
```

## Task Checklist
| Task | Focus | Deliverables |
| --- | --- | --- |
| **Task 1 – EDA & Stats** | Headline lengths, publisher counts, publication cadence, keyword clouds | `scripts/run_task1.py` → CSVs + PNGs under `reports/task1/` |
| **Task 2 – Technical indicators** | SMA(20/50), RSI(14), MACD, price overlays | `scripts/run_task2.py` → indicator CSVs, MA charts under `reports/task2/` |
| **Task 3 – Sentiment vs. returns** | VADER/TextBlob scoring, daily return alignment, correlation stats | `scripts/run_task3.py` → merged dataset + correlation report under `reports/task3/` |

## Getting Started
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon stopwords punkt
```

Optional `.env` keys (defaults shown):
```
DATA_DIR=data
YFINANCE_LOOKBACK=365
PRICES_DIR=prices
DEFAULT_TICKER=AAPL
```

## CLI Usage
```bash
# Task 1 – headline profiling
python scripts/run_task1.py --output reports/task1

# Task 2 – indicators & plots for a ticker
python scripts/run_task2.py --ticker AAPL --output reports/task2

# Task 3 – sentiment vs returns correlation
python scripts/run_task3.py --ticker AAPL --output reports/task3
```
Each script accepts `--help` for additional flags (date windows, custom data paths, etc.). All outputs land inside `reports/` so they can be dropped directly into the interim/final report.

## Testing & CI
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```
The GitHub Actions workflow mirrors the local steps: install deps, run pytest, and lint with flake8/black (both lint/test steps are `continue-on-error` so feedback appears even if one stage fails).

## Branch Strategy
- `task-1` – Environment scaffolding + EDA utilities
- `task-2` – Technical indicators and plotting
- `task-3` – Sentiment scoring, correlation analysis, report assets

All work merges back into `main` via Pull Requests so reviewer history stays intact.

## Reporting Expectations
- **Interim:** Task 1 write‑up + Task 2 WIP highlights (max 3 pages).
- **Final:** Medium-style article covering all three tasks, ≤10 pages / 10 plots, plus supporting CSV/PNG artifacts inside `reports/`.
