"""Phase 1c: historical S&P 500 membership -> daily mask + prices for removed members.

Source: github.com/fja05680/sp500 "S&P 500 Historical Components & Changes (Updated).csv"
(dated full-constituent snapshots). Symbols are aliased with tickers.RULES (renames only;
recycled-symbol drop rules are NOT applied here because the source already labels by company).
Only snapshots before 2023-01-01 are used.

Outputs (atomic):
  data/processed/membership.parquet  (date, ticker[yfinance]) one row per member per trading day
  data/processed/hist_members.csv    removed-since members recovered (ticker, yf_ticker, ...)
  data/processed/hist_members_audit.csv  every removed member: recovered or why not
  data/processed/prices_removed.parquet  yfinance prices for recovered removed members
"""
import os
import time

import pandas as pd
import yfinance as yf

from config import PROC, RAW, SEAL_DATE, START
from news_filter import trading_days
from tickers import RULES

SRC = RAW / "sp500_hist" / "hist_components_updated.csv"
MIN_COVER = 0.90      # yfinance must cover >= 90% of the member's in-window trading days


def atomic(df, path, csv=False):
    tmp = str(path) + ".tmp"
    df.to_csv(tmp, index=False) if csv else df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def alias_membership(sym, date):
    for raw, before, target in RULES:
        if target is not None and sym == raw and date < pd.Timestamp(before):
            return target
    return sym


def snapshots():
    h = pd.read_csv(SRC)
    h["date"] = pd.to_datetime(h["date"])
    h = h.loc[h["date"] < pd.Timestamp(SEAL_DATE)].sort_values("date")
    first = h.loc[h["date"] <= pd.Timestamp(START)].tail(1)          # snapshot in force on day 1
    h = pd.concat([first, h.loc[h["date"] > pd.Timestamp(START)]])
    rows = [(d, alias_membership(s, d)) for d, t in zip(h["date"], h["tickers"]) for s in t.split(",")]
    return pd.DataFrame(rows, columns=["snap", "ticker"]).drop_duplicates()


def daily_mask(snap, cal):
    dates = pd.DataFrame({"date": pd.DatetimeIndex(cal).astype("datetime64[ns]")})
    snap = snap.assign(snap=snap["snap"].astype("datetime64[ns]"))
    sd = pd.DataFrame({"snap": sorted(snap["snap"].unique())})
    m = pd.merge_asof(dates, sd, left_on="date", right_on="snap")       # latest snapshot <= date
    return m.merge(snap, on="snap")[["date", "ticker"]]


def main():
    cal = trading_days()
    snap = snapshots()
    mask = daily_mask(snap, cal)
    uni = pd.read_csv(PROC / "universe.csv")
    to_yf = dict(zip(uni["ticker"], uni["yf_ticker"]))
    members = sorted(mask["ticker"].unique())
    removed = [t for t in members if t not in to_yf]
    print("snapshots", snap["snap"].nunique(), "ever-members", len(members), "current", len(members) - len(removed),
          "removed", len(removed), flush=True)

    days_in = mask.groupby("ticker")["date"].agg(["min", "max", "count"])
    yf_sym = {t: t.replace(".", "-") for t in removed}
    data = yf.download(list(yf_sym.values()), start=START, end=SEAL_DATE, auto_adjust=True,
                       progress=False, threads=True, group_by="column")
    close, vol = data["Close"].copy(), data["Volume"].copy()
    # the big threaded batch gets rate-limited: retry empty symbols in small unthreaded batches
    empty = [y for y in yf_sym.values() if y not in close.columns or close[y].notna().sum() == 0]
    print("empty after batch:", len(empty), "-> retrying", flush=True)
    for i in range(0, len(empty), 10):
        time.sleep(2)
        d = yf.download(empty[i:i + 10], start=START, end=SEAL_DATE, auto_adjust=True,
                        progress=False, threads=False, group_by="column")
        if d is None or not len(d):
            continue
        for y in empty[i:i + 10]:
            if y in d["Close"].columns and d["Close"][y].notna().any():
                close = close.reindex(close.index.union(d.index))
                vol = vol.reindex(vol.index.union(d.index))
                close[y] = d["Close"][y].reindex(close.index)
                vol[y] = d["Volume"][y].reindex(vol.index)
    audit, keep_px = [], []
    for t in removed:
        y = yf_sym[t]
        mdays = mask.loc[mask["ticker"] == t, "date"]
        c = close[y].dropna() if y in close.columns else pd.Series(dtype=float)
        cover = c.index.isin(mdays).sum() / len(mdays)
        ok = cover >= MIN_COVER
        audit.append({"ticker": t, "yf_ticker": y, "member_from": days_in.loc[t, "min"].date(),
                      "member_to": days_in.loc[t, "max"].date(), "member_days": int(days_in.loc[t, "count"]),
                      "yf_rows": len(c), "yf_first": c.index.min().date() if len(c) else None,
                      "coverage": round(cover, 3), "recovered": ok})
        if ok:
            keep_px.append(pd.DataFrame({"date": c.index, "yf_ticker": y, "close": c.values,
                                         "volume": vol[y].reindex(c.index).values}))
    audit = pd.DataFrame(audit)
    rec = audit.loc[audit["recovered"]]
    px = pd.concat(keep_px, ignore_index=True)
    assert px["date"].max() < pd.Timestamp(SEAL_DATE)
    # mask in yfinance symbols; drop removed members we could not price
    mask["ticker"] = mask["ticker"].map(lambda t: to_yf.get(t, t.replace(".", "-")))
    priced = set(uni["yf_ticker"]) | set(rec["yf_ticker"])
    mask_out = mask.loc[mask["ticker"].isin(priced)].reset_index(drop=True)
    atomic(audit, PROC / "hist_members_audit.csv", csv=True)
    atomic(rec[["ticker", "yf_ticker", "member_from", "member_to"]], PROC / "hist_members.csv", csv=True)
    atomic(px, PROC / "prices_removed.parquet")
    atomic(mask_out, PROC / "membership.parquet")
    per_day = mask.groupby("date").size()
    print("removed recovered", len(rec), "missing", len(audit) - len(rec), flush=True)
    print("index members/day: min", per_day.min(), "max", per_day.max(), flush=True)
    print("priced members/day:", mask_out.groupby("date").size().describe().round(1).to_dict(), flush=True)
    print("prices_removed", px.shape, px["date"].min(), px["date"].max(), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
