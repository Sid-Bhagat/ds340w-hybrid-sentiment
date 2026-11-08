"""Configuration for the Li (2025) replication.

Li, S. (2025). "Momentum, volume and investor sentiment study for U.S.
technology sector stocks - A hidden Markov model based principal component
analysis." PLOS ONE 20(9): e0331658. doi:10.1371/journal.pone.0331658

Section 4.1: ten years of weekly Yahoo Finance data, 2015-01-01 to 2024-12-31.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"  # yfinance treats `end` as exclusive

# The seven large tech stocks in the paper.
TECH_TICKERS = ["AMZN", "AAPL", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

# Market / macro series used as PCA input (Section 4.1): stock indices,
# long and short rates, commodities, and the dollar index.
MACRO_TICKERS = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "RATE_LONG_10Y": "^TNX",
    "RATE_SHORT_13W": "^IRX",
    "GOLD": "GC=F",
    "CRUDE_OIL": "CL=F",
    "USD_INDEX": "DX-Y.NYB",
}
RATE_SERIES = {"RATE_LONG_10Y", "RATE_SHORT_13W"}  # differenced, not pct-changed

# Market control in the regressions (S&P 500 weekly return).
BENCHMARK_TICKER = "^GSPC"

# PCA: keep enough components for >= 75% variance, at most 5.
PCA_VARIANCE_TARGET = 0.75
PCA_MAX_COMPONENTS = 5

# HMM: 4 hidden states (the paper's AIC/BIC-based choice), fitted on the
# full sample as in the paper.
HMM_N_STATES = 4
HMM_N_ITER = 1000
HMM_CANDIDATE_STATES = (2, 3, 4, 5, 6)
SEED = 42

# Momentum: one-lag autocovariance of weekly return over the past 4 weeks.
MOMENTUM_LOOKBACK_WEEKS = 4

# Eq. 17 is skipped for a state with fewer weeks than this.
MIN_OBS_PER_STATE = 6
