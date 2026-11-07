"""Count raw FNSPID headlines per Stock_symbol x year (2010+), across both files.

Used to find renamed tickers (FB->META etc.) and which removed S&P 500 members have news.
Output: data/processed/fnspid_symbol_year_counts.csv (symbol, year, n).
"""
import os

import pandas as pd

from config import PROC, START
from news_filter import SRCS, CHUNK

OUT = PROC / "fnspid_symbol_year_counts.csv"


def main():
    tot = []
    for src in SRCS:
        for i, ch in enumerate(pd.read_csv(src, usecols=["Date", "Stock_symbol"], chunksize=CHUNK, dtype=str)):
            ch = ch.dropna()
            ch = ch.loc[ch["Date"] >= START]
            tot.append(ch.groupby([ch["Stock_symbol"], ch["Date"].str.slice(0, 4)]).size())
            if i % 50 == 0:
                print(src.name, "chunk", i, flush=True)
    s = pd.concat(tot).groupby(level=[0, 1]).sum()
    s.index.names = ["symbol", "year"]
    df = s.rename("n").reset_index()
    df.to_csv(str(OUT) + ".tmp", index=False)
    os.replace(str(OUT) + ".tmp", OUT)
    print("symbols", df["symbol"].nunique(), "rows", len(df), "total", df["n"].sum(), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
