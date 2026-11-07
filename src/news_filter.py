"""Phase 1a: filter FNSPID headlines to S&P 500 tickers, 2010+, assign trading day.

Timing rule: headline after 4:00pm ET, on a non-trading day, or without a time -> NEXT trading day.
Rows whose trading day is >= 2023-01-01 (or beyond the pre-2023 calendar) go to a SEALED file.
"""
import os

import numpy as np
import pandas as pd

from config import PROC, RAW, START, end_date
from tickers import RULES, canonical, expand_share_classes

SRCS = [RAW / "Stock_news" / "All_external.csv", RAW / "Stock_news" / "nasdaq_exteral_data.csv"]
OUT = PROC / "news_filtered.parquet"
SEALED = PROC / "news_sealed_2023.parquet"
CHUNK = 100_000          # the big file has full article text, keep chunks small (15 GB RAM)


def trading_days():
    """Trading calendar = SPY dates in prices.parquet (pre-2023 only unless FINAL=1 built it)."""
    p = pd.read_parquet(PROC / "prices.parquet", columns=["date", "yf_ticker"])
    return pd.DatetimeIndex(sorted(p.loc[p["yf_ticker"] == "SPY", "date"].unique()))


def assign_trade_date(ts_utc_str, cal):
    """Return (ts_et, trade_date). trade_date is NaT if beyond the calendar."""
    # FNSPID stores date-only items as 00:00:00 UTC (64% of rows): treat those as "no time"
    has_time = (ts_utc_str.str.len() > 10) & ~ts_utc_str.str.endswith(" 00:00:00")
    ts_utc_str = ts_utc_str.where(has_time, ts_utc_str.str.slice(0, 10))
    ts = pd.to_datetime(ts_utc_str, utc=True, errors="coerce", format="mixed").dt.tz_convert("America/New_York")
    ts_naive = ts.dt.tz_localize(None)
    # date-only strings carry no time zone info: use the calendar date as written, not shifted to ET
    ts_naive = ts_naive.where(has_time, pd.to_datetime(ts_utc_str.where(~has_time), errors="coerce",
                                                       format="%Y-%m-%d"))
    d = ts_naive.dt.normalize()
    after_close = (ts_naive.dt.hour > 16) | ((ts_naive.dt.hour == 16) & (
        (ts_naive.dt.minute > 0) | (ts_naive.dt.second > 0)))
    # side="left": same day if it is a trading day, else next trading day
    idx = np.searchsorted(cal.values, d.values.astype("datetime64[ns]"), side="left")
    on_day = (idx < len(cal)) & (cal.values[np.minimum(idx, len(cal) - 1)] == d.values.astype("datetime64[ns]"))
    push = after_close | ~has_time | ~pd.Series(on_day, index=d.index)
    # after-close/no-time on a trading day -> next trading day (idx+1); non-trading day: idx is already next
    idx = idx + (push & pd.Series(on_day, index=d.index)).to_numpy().astype(int)
    trade = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    ok = idx < len(cal)
    trade.loc[ok] = cal.values[idx[ok]]
    return ts_naive, trade


def keep_set():
    """Canonical tickers to keep -> output (yfinance) ticker. Current universe, plus removed
    historical members if Phase 1c built data/processed/hist_members.csv."""
    uni = pd.read_csv(PROC / "universe.csv")
    to_yf = dict(zip(uni["ticker"], uni["yf_ticker"]))
    hist = PROC / "hist_members.csv"
    if hist.exists():
        h = pd.read_csv(hist)
        to_yf.update({t: y for t, y in zip(h["ticker"], h["yf_ticker"]) if t not in to_yf})
    return to_yf


def main():
    to_yf = keep_set()
    raw_ok = set(to_yf) | {r[0] for r in RULES}       # cheap prefilter before aliasing
    print("keep tickers", len(to_yf), flush=True)
    cal = trading_days()
    print("calendar", cal.min().date(), cal.max().date(), len(cal), flush=True)
    keep, sealed = [], []
    n_read = n_tick = 0
    for src in SRCS:
        if not src.exists():
            print("MISSING", src, flush=True)
            continue
        reader = pd.read_csv(src, usecols=["Date", "Article_title", "Stock_symbol"], chunksize=CHUNK,
                             dtype=str)
        for i, ch in enumerate(reader):
            n_read += len(ch)
            ch = ch.dropna(subset=["Date", "Article_title", "Stock_symbol"])
            ch = ch.loc[ch["Stock_symbol"].isin(raw_ok) & (ch["Date"] >= START)]   # ISO string compare
            ch = ch.assign(Stock_symbol=canonical(ch["Stock_symbol"], ch["Date"].str.slice(0, 10)))
            ch = ch.loc[ch["Stock_symbol"].isin(to_yf.keys())]
            n_tick += len(ch)
            if len(ch):
                ts_et, trade = assign_trade_date(ch["Date"].str.replace(" UTC", "", regex=False), cal)
                out = pd.DataFrame({"ticker": ch["Stock_symbol"].map(to_yf), "ts_et": ts_et,
                                    "trade_date": trade, "headline": ch["Article_title"].str.strip()})
                out = out.dropna(subset=["ts_et"])
                is_sealed = out["trade_date"].isna() | (out["trade_date"] >= pd.Timestamp(end_date()))
                keep.append(out.loc[~is_sealed])
                sealed.append(out.loc[is_sealed])
            if i % 20 == 0:
                print(f"{src.name} chunk {i} read={n_read} kept_ticker_rows={n_tick}", flush=True)
    for name, parts, path in [("main", keep, OUT), ("sealed", sealed, SEALED)]:
        df = expand_share_classes(pd.concat(parts, ignore_index=True))
        # same story can appear in both files / twice: keep the earliest timestamp per ticker+day+headline
        df = (df.sort_values("ts_et").drop_duplicates(subset=["ticker", "trade_date", "headline"])
                .reset_index(drop=True))
        df.to_parquet(str(path) + ".tmp", index=False)
        os.replace(str(path) + ".tmp", path)
        print(name, df.shape, df["ts_et"].min(), df["ts_et"].max(), df["ticker"].nunique(), "tickers",
              flush=True)
    print("rows read", n_read, "S&P500 2010+ rows", n_tick, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
