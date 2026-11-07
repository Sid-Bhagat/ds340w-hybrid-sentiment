"""Shared paths, dates, folds and the 2023 seal."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
LOGS = ROOT / "logs"

START = "2010-01-01"
SEAL_DATE = "2023-01-01"          # first sealed date
FINAL_END = "2024-01-01"          # exclusive end when FINAL=1
TEST_YEARS = [2018, 2019, 2020, 2021, 2022]
FINAL_TEST_YEAR = 2023

# Li (2025) comparison group (reporting subset only; models use the full universe)
LI_STOCKS = ["AMZN", "AAPL", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

SEED = 340


def is_final() -> bool:
    return os.environ.get("FINAL") == "1"


def end_date() -> str:
    """Exclusive end date for any data load."""
    return FINAL_END if is_final() else SEAL_DATE


def enforce_seal(df, date_col="date"):
    """Drop (and refuse) 2023+ rows unless FINAL=1."""
    import pandas as pd
    if is_final():
        return df
    d = pd.to_datetime(df[date_col]) if date_col in df.columns else pd.to_datetime(df.index)
    mask = d < pd.Timestamp(SEAL_DATE)
    return df.loc[mask.values] if hasattr(mask, "values") else df.loc[mask]


def test_years():
    return TEST_YEARS + ([FINAL_TEST_YEAR] if is_final() else [])
