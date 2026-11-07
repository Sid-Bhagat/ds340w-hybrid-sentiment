"""Phase 0: current S&P 500 universe + SPY daily prices from yfinance."""
import io

import pandas as pd
import requests
import yfinance as yf

from config import PROC, START, end_date, LI_STOCKS

WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_universe():
    html = requests.get(WIKI, headers={"User-Agent": "ds340w-research-script"}, timeout=30).text
    tables = pd.read_html(io.StringIO(html))
    cur = tables[0][["Symbol", "Security", "GICS Sector"]].rename(
        columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
    cur["yf_ticker"] = cur["ticker"].str.replace(".", "-", regex=False)   # BRK.B -> BRK-B
    cur["li_group"] = cur["ticker"].isin(LI_STOCKS)
    cur.to_csv(PROC / "universe.csv", index=False)
    # change log table (for optional historical membership)
    tables[1].to_csv(PROC / "sp500_changes_raw.csv", index=False)
    return cur


def download(tickers):
    data = yf.download(tickers, start=START, end=end_date(), auto_adjust=True,
                       progress=False, threads=True, group_by="column")
    close = data["Close"].stack().rename("close")
    vol = data["Volume"].stack().rename("volume")
    df = pd.concat([close, vol], axis=1).reset_index()
    df.columns = ["date", "yf_ticker", "close", "volume"]
    return df.dropna(subset=["close"])


if __name__ == "__main__":
    uni = get_universe()
    print("universe size", len(uni))
    df = download(uni["yf_ticker"].tolist() + ["SPY"])
    assert df["date"].max() < pd.Timestamp(end_date())
    df.to_parquet(PROC / "prices.parquet", index=False)
    print(df.shape, df["date"].min(), df["date"].max(), df["yf_ticker"].nunique(), "tickers")
