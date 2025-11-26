# Predicting Price Moves with News Sentiment

## Overview
This project analyzes the Financial News and Stock Price Integration Dataset (FNSPID) to quantify how news sentiment influences stock price movements. The work spans three tasks: environment setup and EDA, technical indicator computation, and sentiment-versus-return correlation analysis. All deliverables target Nova Financial Solutions' goal of boosting forecasting accuracy through data-driven insights.

## Repository Structure
`
Predicting_Price_Moves_with_News_Sentiment-week1/
 .github/workflows/unittests.yml   # CI pipeline
 .vscode/settings.json             # VS Code configuration
 .gitignore
 README.md
 requirements.txt
 notebooks/                        # Notebook experiments
 scripts/                          # CLI entrypoints for tasks
 src/                              # Core analysis modules
 tests/                            # Unit tests
 data/                             # Raw/intermediate data (gitignored)
`

## Tasks Overview
1. **Task 1  EDA & Stats**
   - Descriptive statistics for text lengths and publisher counts
   - Publication frequency analysis
   - Keyword/topic extraction using NLP
2. **Task 2  Technical Indicators**
   - Load OHLCV data from CSV or yfinance
   - Compute MA, RSI, MACD via TA-Lib / pandas fallbacks
   - Visualize price action with overlays
3. **Task 3  Sentiment vs Returns**
   - Sentiment scoring (VADER/TextBlob)
   - Daily return calculation
   - Correlation analysis & reporting

## Getting Started
`ash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon stopwords punkt
`

### Environment Variables
Create a .env file if you plan to store API keys or custom data paths:
`
DATA_DIR=data
YFINANCE_LOOKBACK=365
`

## Scripts
| Script | Description |
|--------|-------------|
| scripts/run_task1.py | Executes Task 1 EDA pipeline and saves summary artifacts |
| scripts/run_task2.py | Fetches price data, computes indicators, exports plots |
| scripts/run_task3.py | Runs sentiment scoring, aligns with returns, and reports correlations |

Each script accepts CLI arguments; run with python scripts/run_taskX.py --help for usage details.

## Testing & CI
`ash
pytest -v --cov=src
`
GitHub Actions executes linting (lake8), formatting (lack --check), and tests on every push/PR targeting main and feature branches.

## Branch Strategy
- 	ask-1  Environment, EDA scaffolding, documentation
- 	ask-2  Technical indicators and quantitative analysis
- 	ask-3  Sentiment correlation analysis and reporting

Merge via Pull Requests to ensure review history for interim/final submissions.

## Reporting
- **Interim report:** Task 1 progress, Task 2 WIP (3 pages)
- **Final report:** All tasks roughly aoround(10 pages, 10 plots), Medium-style narrative
