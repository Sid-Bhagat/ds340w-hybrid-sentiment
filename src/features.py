"""Phase 2: daily stock panel. Every feature on row (ticker, date t) uses info known at t's close.

Target y = 1 if the stock's next-day return > SPY's next-day return (t -> t+1).
Rows are restricted to that day's S&P 500 members (membership.parquet, Phase 1c).
Market features (for the PCA-HMM regimes, fit per fold in regimes.py) -> market.parquet.
"""
import os

import numpy as np
import pandas as pd

from config import PROC, end_date, enforce_seal

PRICE_FEATS = ["ret1", "ret5", "ret20", "vol20", "volz"]
EVENTS = ["ev_earnings_beat", "ev_earnings_miss", "ev_guidance", "ev_buyback", "ev_dividend",
          "ev_merger_acq", "ev_fda_approval", "ev_fda_rejection", "ev_investigation_lawsuit",
          "ev_downgrade", "ev_upgrade"]
SENT_FEATS = ["sent", "n_news", "has_news", "sent_surprise"] + EVENTS
MARKET_FEATS = ["mkt_ret1", "mkt_ret5", "mkt_ret20", "mkt_vol20", "xs_disp"]


def atomic_parquet(df, path):
    df.to_parquet(str(path) + ".tmp", index=False)
    os.replace(str(path) + ".tmp", path)


def load_prices():
    p = pd.concat([pd.read_parquet(PROC / "prices.parquet"), pd.read_parquet(PROC / "prices_removed.parquet")])
    p = p.rename(columns={"yf_ticker": "ticker"})
    p["date"] = p["date"].astype("datetime64[ns]")
    p["ticker"] = p["ticker"].astype(str)
    p = enforce_seal(p).drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])
    return p.reset_index(drop=True)


def price_features(p):
    g = p.groupby("ticker", sort=False)
    p["ret1"] = g["close"].pct_change(fill_method=None)
    p["ret5"] = g["close"].pct_change(5, fill_method=None)
    p["ret20"] = g["close"].pct_change(20, fill_method=None)
    p["vol20"] = g["ret1"].transform(lambda s: s.rolling(20, min_periods=15).std())
    lv = np.log1p(p["volume"])
    m = lv.groupby(p["ticker"]).transform(lambda s: s.rolling(20, min_periods=15).mean())
    sd = lv.groupby(p["ticker"]).transform(lambda s: s.rolling(20, min_periods=15).std())
    p["volz"] = (lv - m) / sd.replace(0, np.nan)
    # next-day return and date (t+1) - used ONLY for the target and fold purging
    p["ret_next"] = g["ret1"].shift(-1)
    p["next_date"] = g["date"].shift(-1)
    return p


def daily_sentiment():
    n = pd.read_parquet(PROC / "news_scored.parquet", columns=["ticker", "trade_date", "score"] + EVENTS)
    n = n.rename(columns={"trade_date": "date"})
    n["date"] = n["date"].astype("datetime64[ns]")
    n = enforce_seal(n)
    agg = {"score": ["mean", "size"]} | {e: "max" for e in EVENTS}
    d = n.groupby(["ticker", "date"]).agg(agg)
    d.columns = ["sent", "n_news"] + EVENTS
    return d.reset_index()


def add_surprise(panel):
    """sent_surprise = today's mean score - mean of the stock's news-day scores over the previous
    60 trading days (t-60..t-1, past only). 0 on no-news days or with no past news."""
    s = panel["sent"].where(panel["has_news"] == 1)
    g = s.groupby(panel["ticker"])
    past_sum = g.transform(lambda x: x.fillna(0).rolling(60, min_periods=1).sum().shift(1))
    past_cnt = g.transform(lambda x: x.notna().astype(float).rolling(60, min_periods=1).sum().shift(1))
    trail = past_sum / past_cnt.replace(0, np.nan)
    panel["sent_surprise"] = (panel["sent"] - trail).where(panel["has_news"] == 1).fillna(0.0)
    return panel


def main():
    p = price_features(load_prices())
    spy = p.loc[p["ticker"] == "SPY", ["date", "ret1", "ret5", "ret20", "vol20", "ret_next"]]
    spy = spy.rename(columns={"ret1": "mkt_ret1", "ret5": "mkt_ret5", "ret20": "mkt_ret20",
                              "vol20": "mkt_vol20", "ret_next": "mkt_ret_next"})
    stocks = p.loc[p["ticker"] != "SPY"].merge(spy[["date", "mkt_ret_next"]], on="date", how="inner")

    # sentiment: computed on every trading day of each stock (so the 60-day window is by trading day)
    s = daily_sentiment()
    stocks = stocks.merge(s, on=["ticker", "date"], how="left")
    stocks["has_news"] = stocks["n_news"].notna().astype(int)
    stocks[["sent", "n_news"] + EVENTS] = stocks[["sent", "n_news"] + EVENTS].fillna(0.0)
    stocks = add_surprise(stocks)
    n_news_rows = int(stocks["n_news"].sum())

    # point-in-time membership: only that day's index members
    mem = pd.read_parquet(PROC / "membership.parquet")
    mem["date"] = mem["date"].astype("datetime64[ns]")
    panel = stocks.merge(mem, on=["date", "ticker"], how="inner")

    # market features: SPY + cross-sectional dispersion of member returns (both known at t's close)
    disp = panel.groupby("date")["ret1"].std().rename("xs_disp").reset_index()
    market = spy.merge(disp, on="date", how="inner").sort_values("date")
    market = market.dropna(subset=MARKET_FEATS).reset_index(drop=True)

    panel["y"] = (panel["ret_next"] > panel["mkt_ret_next"]).astype(int)
    panel["excess_next"] = panel["ret_next"] - panel["mkt_ret_next"]     # portfolio P&L only
    panel = panel.dropna(subset=PRICE_FEATS + ["ret_next", "mkt_ret_next", "next_date"])
    panel = panel.loc[panel["date"].isin(market["date"])]
    assert panel["date"].max() < pd.Timestamp(end_date()) and panel["next_date"].max() < pd.Timestamp(end_date())
    panel = panel[["date", "next_date", "ticker"] + PRICE_FEATS + SENT_FEATS +
                  ["y", "ret_next", "mkt_ret_next", "excess_next"]].sort_values(["date", "ticker"])
    panel = panel.reset_index(drop=True)
    atomic_parquet(panel, PROC / "panel.parquet")
    atomic_parquet(market[["date"] + MARKET_FEATS], PROC / "market.parquet")
    print("panel", panel.shape, panel["date"].min().date(), panel["date"].max().date(),
          panel["ticker"].nunique(), "tickers; y mean %.3f" % panel["y"].mean(), flush=True)
    print("news headlines on stock-days (pre-mask)", n_news_rows, "in panel", int(panel["n_news"].sum()), flush=True)
    print("nulls", int(panel.isna().sum().sum()), "market", market.shape, flush=True)


if __name__ == "__main__":
    main()
