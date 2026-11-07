"""Expanding walk-forward folds. Train = rows whose target date (next_date) is before the test year,
so no training label is computed from a test-year price. Test = rows dated in the test year."""
import pandas as pd

from config import PROC, enforce_seal


def load_panel():
    p = enforce_seal(pd.read_parquet(PROC / "panel.parquet"))
    return p


def split(panel, year):
    start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")
    tr = panel.loc[(panel["date"] < start) & (panel["next_date"] < start)]
    te = panel.loc[(panel["date"] >= start) & (panel["date"] < end)]
    return tr, te


def regimes_for(year):
    r = pd.read_parquet(PROC / "regimes.parquet")
    r = r.loc[r["fold"] == year, ["date", "p_high"]]
    r["date"] = r["date"].astype("datetime64[ns]")
    return r
