"""SEC 8-K metadata (no filing text): ticker -> CIK -> submissions JSON -> daily 8-K features.

Universe = universe.csv (current) + hist_members.csv (recovered removed members), yfinance tickers.
CIK map: SEC company_tickers.json (current tickers; our canonical tickers already follow renames).
Raw JSON cached in data/raw/edgar/ (submissions CIK##########.json + older "files" pages).
Keeps forms 8-K and 8-K/A filed 2010+. acceptanceDateTime is UTC (verified; see DECISIONS) -> converted to ET.
Timing: after 16:00:00 ET or on a non-trading day -> next trading day.
Rows whose trading day is >= 2023-01-01 (or beyond the calendar) -> SEALED file, never loaded.

  python edgar.py fetch   # download (<= 8 req/s, retry with backoff, cached)
  python edgar.py build   # parse -> edgar_8k.parquet, edgar_8k_sealed_2023.parquet, edgar_daily.parquet
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

from config import PROC, RAW, START, end_date

EDGAR = RAW / "edgar"


def user_agent() -> dict:
    """SEC asks every client to identify itself. Set SEC_USER_AGENT="Your Name your@email" before `fetch`.
    Read lazily so importing this module (for FEATS_8K, PREDECESSORS, ...) does not need the variable."""
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise SystemExit('Set SEC_USER_AGENT="Your Name your@email" to run edgar.py fetch')
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}

MIN_GAP = 1 / 8                       # at most 8 requests per second
FORMS = {"8-K", "8-K/A"}
ITEMS = {"i202": ["2.02"], "i101": ["1.01"], "i201": ["2.01"], "i502": ["5.02"], "i701_801": ["7.01", "8.01"]}
NOT_OTHER = {"2.02", "1.01", "2.01", "5.02", "7.01", "8.01", "9.01"}   # 9.01 = exhibits, attached to most 8-Ks
FEATS_8K = ["has_8k", "log_n_8k"] + list(ITEMS) + ["any_other_item"]
_last = [0.0]

# Predecessor CIKs for tickers whose current CIK is a newer holding entity (found by the coverage check:
# member years with zero 8-Ks; names from SEC cik-lookup-data.txt, each verified to file 8-Ks in the gap
# years). Predecessor filings are used only BEFORE the successor CIK's first 8-K (some keep filing as
# subsidiaries afterwards, which would double count).
PREDECESSORS = {"XOM": [34088], "BLK": [1364742], "APA": [6769], "DIS": [1001039], "GOOGL": [1288776],
                "GOOG": [1288776], "SKYD": [813828], "NAVI": [1593538], "BKR": [808362], "CI": [701221],
                "DD": [30554], "DINO": [48039], "AVGO": [1441634, 1649338], "DXC": [23082], "ETN": [31277],
                "FTI": [1135152], "ICE": [1174746], "MDT": [64670], "PRGO": [820096], "VTRS": [69499, 1623613],
                "XRX": [108772]}
# current SEC ticker belongs to a different (recycled-symbol) company or is missing
OVERRIDE = {"KG": 1047699, "GENZ": 732485}


def get_json(url, path):
    """Cached GET with rate limit and exponential backoff. Returns parsed JSON or None (404)."""
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(6):
        wait = MIN_GAP - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = requests.get(url, headers=user_agent(), timeout=30)
        except requests.RequestException as e:
            print("net error", url, e, flush=True)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            data = r.json()
            tmp = str(path) + ".tmp"
            with open(tmp, "w") as f:
                f.write(r.text)
            os.replace(tmp, path)
            return data
        if r.status_code == 404:
            return None
        print("HTTP", r.status_code, url, "retry", attempt, flush=True)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after retries: {url}")


def universe():
    u = pd.read_csv(PROC / "universe.csv")[["yf_ticker"]].assign(current=True)
    h = pd.read_csv(PROC / "hist_members.csv")[["yf_ticker"]].assign(current=False)
    return pd.concat([u, h]).drop_duplicates("yf_ticker").reset_index(drop=True)


def cik_map():
    ct = json.loads((EDGAR / "company_tickers.json").read_text())
    m = {v["ticker"].upper(): int(v["cik_str"]) for v in ct.values()}
    u = universe()
    u["cik"] = u["yf_ticker"].map(lambda t: OVERRIDE.get(t, m.get(t.upper(), m.get(t.upper().replace("-", ".")))))
    return u


def fetch():
    EDGAR.mkdir(parents=True, exist_ok=True)
    u = cik_map()
    miss = u.loc[u["cik"].isna(), "yf_ticker"].tolist()
    print("tickers", len(u), "mapped", u["cik"].notna().sum(), "unmapped", len(miss), miss, flush=True)
    ciks = set(u["cik"].dropna().astype(int)) | {c for v in PREDECESSORS.values() for c in v}
    for i, cik in enumerate(sorted(ciks)):
        name = f"CIK{cik:010d}.json"
        sub = get_json(f"https://data.sec.gov/submissions/{name}", EDGAR / name)
        if sub is None:
            print("no submissions", cik, flush=True)
            continue
        for f in sub.get("filings", {}).get("files", []):
            if f.get("filingTo", "9999") >= START:      # older pages only if they reach 2010+
                get_json(f"https://data.sec.gov/submissions/{f['name']}", EDGAR / f["name"])
        if i % 50 == 0:
            print("cik", i, flush=True)
    print("FETCH DONE", flush=True)


def filings_for(cik):
    """All 8-K / 8-K/A rows (2010+) for one CIK from the cached recent block + older pages."""
    sub = json.loads((EDGAR / f"CIK{cik:010d}.json").read_text())
    blocks = [sub["filings"]["recent"]]
    for f in sub["filings"].get("files", []):
        p = EDGAR / f["name"]
        if p.exists():
            blocks.append(json.loads(p.read_text()))
    rows = []
    for b in blocks:
        df = pd.DataFrame({k: b[k] for k in ["accessionNumber", "form", "filingDate", "acceptanceDateTime", "items"]})
        rows.append(df.loc[df["form"].isin(FORMS) & (df["filingDate"] >= START)])
    return pd.concat(rows, ignore_index=True).drop_duplicates("accessionNumber")


def to_trade_date(ts_et, cal):
    """ts_et: naive Eastern timestamps. After 16:00:00 ET or non-trading day -> next trading day.
    Returns trade dates (NaT if beyond the calendar)."""
    d = ts_et.dt.normalize().to_numpy().astype("datetime64[ns]")
    c = cal.to_numpy().astype("datetime64[ns]")
    idx = np.searchsorted(c, d, side="left")                  # same day if trading day, else next
    on_day = (idx < len(c)) & (c[np.minimum(idx, len(c) - 1)] == d)
    after = ((ts_et.dt.hour > 16) | ((ts_et.dt.hour == 16) & ((ts_et.dt.minute > 0) | (ts_et.dt.second > 0)))).to_numpy()
    idx = idx + (on_day & after).astype(int)
    out = np.full(len(d), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = idx < len(c)
    out[ok] = c[idx[ok]]
    return pd.Series(out, index=ts_et.index)


def parse_acceptance(s):
    """'2022-10-27T20:30:22.000Z' is UTC (verified: Apple's 4:30pm ET earnings 8-Ks show 20:30Z in EDT and
    21:30Z in EST) -> convert to naive America/New_York time (DST-aware)."""
    ts = pd.to_datetime(s.str.slice(0, 19), format="%Y-%m-%dT%H:%M:%S", errors="coerce").dt.tz_localize("UTC")
    return ts.dt.tz_convert("America/New_York").dt.tz_localize(None)


def build():
    from news_filter import trading_days
    cal = trading_days()
    u = cik_map().dropna(subset=["cik"])
    parts = []
    for t, cik in zip(u["yf_ticker"], u["cik"].astype(int)):
        cur = filings_for(cik)
        cut = cur["filingDate"].min() if len(cur) else "9999-12-31"
        own = [cur.assign(ticker=t, cik=cik)]
        for pc in PREDECESSORS.get(t, []):
            pf = filings_for(pc)
            own.append(pf.loc[pf["filingDate"] < cut].assign(ticker=t, cik=pc))
        parts.append(pd.concat(own, ignore_index=True).drop_duplicates("accessionNumber"))
    f = pd.concat(parts, ignore_index=True)
    f["ts_et"] = parse_acceptance(f["acceptanceDateTime"])
    f = f.dropna(subset=["ts_et"])
    f["trade_date"] = to_trade_date(f["ts_et"], cal)
    sealed = f["trade_date"].isna() | (f["trade_date"] >= pd.Timestamp(end_date()))
    keep = f.loc[~sealed].reset_index(drop=True)
    for name, df in [("edgar_8k.parquet", keep), ("edgar_8k_sealed_2023.parquet", f.loc[sealed])]:
        df.to_parquet(str(PROC / name) + ".tmp", index=False)
        os.replace(str(PROC / name) + ".tmp", PROC / name)
    # daily features
    items = keep["items"].fillna("").str.split(",").map(lambda xs: {x.strip() for x in xs if x.strip()})
    for col, codes in ITEMS.items():
        keep[col] = items.map(lambda s, c=codes: int(any(x in s for x in c)))
    keep["any_other_item"] = items.map(lambda s: int(len(s - NOT_OTHER) > 0))
    g = keep.groupby(["ticker", "trade_date"])
    daily = g[list(ITEMS) + ["any_other_item"]].max()
    daily["n_8k"] = g.size()
    daily = daily.reset_index().rename(columns={"trade_date": "date"})
    daily.to_parquet(str(PROC / "edgar_daily.parquet") + ".tmp", index=False)
    os.replace(str(PROC / "edgar_daily.parquet") + ".tmp", PROC / "edgar_daily.parquet")
    print("8-K filings kept", len(keep), "sealed", int(sealed.sum()), "tickers", keep["ticker"].nunique(),
          "daily rows", len(daily), "max trade", keep["trade_date"].max().date(), flush=True)
    print("acceptance hour ET (top):", keep["ts_et"].dt.hour.value_counts().head(6).to_dict(), flush=True)
    print("BUILD DONE", flush=True)


if __name__ == "__main__":
    {"fetch": fetch, "build": build}[sys.argv[1]]()
