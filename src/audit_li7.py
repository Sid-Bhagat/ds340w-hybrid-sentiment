"""Li-7 coverage audit: headlines per year at each pipeline stage (2010-2022 only; 2023 never counted).

(a) raw All_external.csv  (b) raw nasdaq_exteral_data.csv  (by raw Date year, raw Stock_symbol)
(c) rows passing the filter step before cross-file dedupe/aliasing (by trade_date year, raw symbol)
(d) news_filtered.parquet after dedupe + alias + share-class rules (by trade_date year, ticker)
(e) headlines landing on panel stock-days (member days with prices), sum of n_news
Output: results/li7_coverage_audit.csv; raw rows cached in data/processed/li7_raw_rows.parquet
"""
import os

import pandas as pd

from config import PROC, RESULTS, SEAL_DATE
from news_filter import CHUNK, SRCS, assign_trade_date, trading_days

SYMS = ["AAPL", "AMZN", "MSFT", "NVDA", "GOOGL", "GOOG", "TSLA", "META", "FB"]
LI = ["AAPL", "AMZN", "MSFT", "NVDA", "GOOGL", "TSLA", "META", "GOOG"]
YEARS = list(range(2010, 2023))


def raw_rows():
    cache = PROC / "li7_raw_rows.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    parts = []
    for src in SRCS:
        for ch in pd.read_csv(src, usecols=["Date", "Article_title", "Stock_symbol"], chunksize=CHUNK, dtype=str):
            ch = ch.loc[ch["Stock_symbol"].isin(SYMS)]
            if len(ch):
                parts.append(ch.assign(file=src.name))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(str(cache) + ".tmp", index=False)
    os.replace(str(cache) + ".tmp", cache)
    return df


def table(s, stage):
    t = s.groupby(["sym", "year"]).size().unstack(fill_value=0).reindex(columns=YEARS, fill_value=0)
    t.insert(0, "stage", stage)
    return t.reset_index()


def main():
    raw = raw_rows()
    raw["year"] = pd.to_numeric(raw["Date"].str.slice(0, 4), errors="coerce")
    print("raw rows", len(raw), "missing title", raw["Article_title"].isna().sum(),
          "unparseable year", raw["year"].isna().sum(), flush=True)
    print("Date formats:", raw["Date"].str.replace(r"\d", "9", regex=True).value_counts().head(5).to_dict(), flush=True)
    pre = raw.loc[raw["year"].between(2010, 2022)].rename(columns={"Stock_symbol": "sym"})
    out = [table(pre.loc[pre["file"] == f], f"a/b raw {f}") for f in pre["file"].unique()]
    # (c) the filter's own steps, per raw symbol, before cross-file dedupe and aliasing
    cal = trading_days()
    c = raw.dropna(subset=["Date", "Article_title", "Stock_symbol"])
    c = c.loc[c["Date"] >= "2010-01-01"]
    ts, trade = assign_trade_date(c["Date"].str.replace(" UTC", "", regex=False), cal)
    c = c.assign(ts=ts, trade=trade).dropna(subset=["ts"])
    c = c.loc[c["trade"].notna() & (c["trade"] < pd.Timestamp(SEAL_DATE))]
    c = c.assign(sym=c["Stock_symbol"], year=c["trade"].dt.year)
    out.append(table(c, "c filtered, pre-dedupe (both files)"))
    c2 = c.assign(h=c["Article_title"].str.strip()).drop_duplicates(["sym", "trade", "h"])
    out.append(table(c2, "c2 after cross-file dedupe (raw symbol)"))
    n = pd.read_parquet(PROC / "news_filtered.parquet", columns=["ticker", "trade_date"])
    n = n.loc[n["ticker"].isin(LI)]
    out.append(table(n.assign(sym=n["ticker"], year=n["trade_date"].dt.year), "d news_filtered (aliased)"))
    p = pd.read_parquet(PROC / "panel.parquet", columns=["ticker", "date", "n_news"])
    p = p.loc[p["ticker"].isin(LI)]
    e = p.assign(sym=p["ticker"], year=p["date"].dt.year).groupby(["sym", "year"])["n_news"].sum()
    e = e.unstack(fill_value=0).reindex(columns=YEARS, fill_value=0).astype(int)
    e.insert(0, "stage", "e panel (member days)")
    out.append(e.reset_index())
    res = pd.concat(out, ignore_index=True)
    res["total"] = res[YEARS].sum(axis=1)
    res.to_csv(str(RESULTS / "li7_coverage_audit.csv") + ".tmp", index=False)
    os.replace(str(RESULTS / "li7_coverage_audit.csv") + ".tmp", RESULTS / "li7_coverage_audit.csv")
    with pd.option_context("display.width", 250):
        print(res.sort_values(["sym", "stage"]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
