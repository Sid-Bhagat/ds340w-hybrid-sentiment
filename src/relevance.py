"""Robustness variant (added before 2023, motivated by the annotation audit): relevance-filtered news.

A headline is kept for its ticker only if it mentions the ticker symbol or a company name:
- names: universe.csv `name` + SEC entity names of the ticker's CIK(s) (current, formerNames, predecessor
  CIKs from edgar.PREDECESSORS), legal suffixes removed, matched case-insensitively on word boundaries;
  plus the small hand-written ALIASES dict (logged in DECISIONS.md).
- symbol: whole-word, case-sensitive. Symbols of 1-2 letters (A, T, F, ON, ...) only count as "$T",
  "(T)" or ":T" (e.g. NYSE:T), since as bare words they are ordinary English.
Sentiment features are rebuilt from relevant headlines exactly as in features.py (self-check: the same
code on ALL headlines reproduces panel.parquet's sentiment columns exactly).
Outputs: data/processed/news_relevance.parquet (flag per news row), data/processed/panel_rel.parquet,
results/relevance_survival.csv
"""
import json
import os
import re

import numpy as np
import pandas as pd

from config import PROC, RESULTS, enforce_seal
from edgar import EDGAR, OVERRIDE, PREDECESSORS, cik_map
from features import EVENTS, SENT_FEATS, add_surprise, load_prices

SUFFIX = re.compile(r"\b(inc|incorporated|corp|corporation|co|company|companies|holdings?|hldgs|ltd|limited|plc|llc|"
                    r"l\.?p|n\.?v|s\.?a|ag|se|group|the|cl a|class a|/[a-z]{2}/?|de|new)\b\.?", re.I)
# well-known short names / brands not derivable from legal names (kept small; logged)
ALIASES = {"GOOGL": ["Alphabet", "Google"], "GOOG": ["Alphabet", "Google"], "META": ["Facebook", "Meta"],
           "AMZN": ["Amazon"], "MSFT": ["Microsoft"], "AAPL": ["Apple"], "NVDA": ["Nvidia"], "TSLA": ["Tesla"],
           "BRK-B": ["Berkshire"], "JPM": ["JPMorgan", "JP Morgan", "Chase"], "BAC": ["BofA", "Bank of America"],
           "WFC": ["Wells Fargo"], "GS": ["Goldman"], "MS": ["Morgan Stanley"], "KO": ["Coca-Cola", "Coke"],
           "PEP": ["Pepsi", "PepsiCo"], "MCD": ["McDonald's", "McDonalds"], "WMT": ["Walmart", "Wal-Mart"],
           "HD": ["Home Depot"], "DIS": ["Disney"], "JNJ": ["J&J", "Johnson & Johnson"], "PG": ["P&G", "Procter"],
           "XOM": ["Exxon"], "CVX": ["Chevron"], "IBM": ["IBM"], "GE": ["General Electric"], "T": ["AT&T"],
           "VZ": ["Verizon"], "UNH": ["UnitedHealth"], "BKNG": ["Priceline", "Booking"], "SKYD": ["CBS", "Viacom", "Paramount"],
           "HPQ": ["HP", "Hewlett"], "HPE": ["HPE", "Hewlett Packard Enterprise"], "LLY": ["Lilly"], "MRK": ["Merck"],
           "BMY": ["Bristol-Myers", "Bristol Myers"], "ABBV": ["AbbVie"], "CMCSA": ["Comcast"], "NFLX": ["Netflix"],
           "MMM": ["3M"]}
MIN_SYMBOL_BARE = 3
# trailing descriptor words: "Ford Motor" -> also "Ford", "Agilent Technologies" -> also "Agilent"
DESCRIPTORS = re.compile(r"\s+(motors?|technologies|technology|industries|systems|international|pharmaceuticals|"
                         r"financial|entertainment|resources|communications|brands|enterprises|platforms|laboratories|"
                         r"labs|energy|services|software|networks|solutions|therapeutics|bancorp|bancorporation|"
                         r"properties|realty|trust|worldwide|global)$", re.I)


# generic first words never used alone as a short name (in addition to first words shared by 2+ companies)
GENERIC_FIRST = {"american", "general", "first", "united", "national", "international", "southern", "western",
                 "eastern", "northern", "public", "pacific", "texas", "global", "digital", "capital", "royal",
                 "best", "home", "old", "dollar", "under", "bath", "bed", "big", "family", "foot", "air", "eli",
                 "quest", "align", "waters", "intuitive", "progressive", "charter", "crown", "match", "live", "take",
                 "electronic", "automatic", "applied", "analog", "advance", "advanced", "genuine", "consolidated",
                 "discover", "universal", "union", "alliance", "republic", "state", "equity", "host", "realty",
                 "principal", "regions", "cardinal", "news", "sealed", "packaging", "steel", "mid", "fifth",
                 "laboratory", "henry", "jack", "palo", "check", "ross", "simon", "kimberly", "church", "health",
                 "healthcare", "bank", "citizens", "huntington", "regency", "federal", "extra", "public", "prologis",
                 "norwegian", "carnival", "las", "san", "new", "north", "south", "east", "west", "great", "life",
                 "energy", "power", "mobile", "first", "trust", "service", "data", "smart", "solar", "edison",
                 "tractor", "darden", "zions", "arthur", "martin", "lincoln", "hartford", "travelers", "everest",
                 "constellation", "pinnacle", "dominion", "duke", "sherwin", "ball", "target", "gap", "key",
                 "ace", "wells", "morgan", "goldman", "state", "monster", "mettler", "thermo", "becton", "bristol",
                 "marathon", "boston", "johnson", "charles", "photonics", "delta", "discovery", "fidelity"}


def clean_name(n):
    n = str(n).split("/")[0]                       # drop SEC tags like "APPLE COMPUTER / FA", "KEYCORP/"
    n = re.sub(r"[,\.]", " ", n)
    n = re.sub(r"\(.*?\)", " ", n)
    n = SUFFIX.sub(" ", n)
    n = re.sub(r"\s+", " ", n).strip(" -&'")
    return n if len(n) >= 3 else None


def sec_names(cik):
    p = EDGAR / f"CIK{int(cik):010d}.json"
    if not p.exists():
        return []
    s = json.loads(p.read_text())
    # former names only if still in use in 2009 or later (drops e.g. KeyCorp's 1990s "Society Corp")
    return [s.get("name", "")] + [x.get("name", "") for x in s.get("formerNames", []) if str(x.get("to", "")) >= "2009"]


def name_table():
    u = pd.read_csv(PROC / "universe.csv")
    names = {t: [n] for t, n in zip(u["yf_ticker"], u["name"])}
    for t, cik in zip(*cik_map()[["yf_ticker", "cik"]].dropna().to_numpy().T):
        ciks = [cik] + PREDECESSORS.get(t, [])
        for c in ciks:
            names.setdefault(t, []).extend(sec_names(c))
    out = {}
    first_count = {}
    for t, ns in names.items():                    # how many companies share each first word
        for w in {(re.findall(r"[A-Za-z][A-Za-z'&\-]{3,}", c) or [""])[0].lower()
                  for c in (clean_name(n) for n in ns) if c}:
            first_count[w] = first_count.get(w, 0) + 1
    for t, ns in names.items():
        base = {c for c in (clean_name(n) for n in ns) if c}
        short = {DESCRIPTORS.sub("", c) for c in base}
        first = {w for w in (re.findall(r"[A-Za-z][A-Za-z'&\-]{3,}", c)[:1] for c in base) for w in w}
        short |= first
        # any single-word short variant must be distinctive (not generic, not shared by 2+ companies)
        short = {w for w in short if " " in w or (w.lower() not in GENERIC_FIRST and first_count.get(w.lower(), 0) < 2)}
        cl = sorted({c for c in base | short if len(c) >= 4 or c.upper() in {"IBM", "HP", "HPE", "BOFA"}} | base,
                    key=len, reverse=True)
        out[t] = cl
    for t, al in ALIASES.items():                  # aliases are used verbatim (no cleaning/length rule)
        out[t] = sorted(set(out.get(t, [])) | set(al), key=len, reverse=True)
    return out


def ticker_regex(t, names):
    sym = re.escape(t.replace("-", "."))
    parts = [rf"\${sym}\b", rf"\({sym}\)", rf":\s?{sym}\b"]
    if len(t.replace("-", "")) >= MIN_SYMBOL_BARE:
        parts.append(rf"(?<![A-Za-z]){sym}(?![A-Za-z])")
    name_parts = [r"\b" + re.escape(n).replace(r"\ ", r"[\s\-]+") + r"(?:'s|s)?\b" for n in names]
    sym_re = re.compile("|".join(parts))                                  # case-sensitive symbol
    name_re = re.compile("|".join(name_parts), re.I) if name_parts else None
    return sym_re, name_re


def flag_relevant(news, names):
    rel = np.zeros(len(news), dtype=bool)
    for t, idx in news.groupby("ticker").indices.items():
        sym_re, name_re = ticker_regex(t, names.get(t, []))
        h = news["headline"].to_numpy()[idx]
        hit = np.array([bool(sym_re.search(x)) or (name_re is not None and bool(name_re.search(x))) for x in h])
        rel[idx] = hit
    return rel


def sentiment_block(news):
    """Exactly features.py's sentiment steps, for a given news frame -> (ticker, date) sentiment columns."""
    n = news.rename(columns={"trade_date": "date"})
    n["date"] = n["date"].astype("datetime64[ns]")
    n = enforce_seal(n)
    agg = {"score": ["mean", "size"]} | {e: "max" for e in EVENTS}
    d = n.groupby(["ticker", "date"]).agg(agg)
    d.columns = ["sent", "n_news"] + EVENTS
    d = d.reset_index()
    p = load_prices()
    grid = p.loc[p["ticker"] != "SPY", ["ticker", "date"]]
    spy_dates = p.loc[p["ticker"] == "SPY", ["date"]]
    grid = grid.merge(spy_dates, on="date", how="inner")         # same row set/order as features.main
    s = grid.merge(d, on=["ticker", "date"], how="left")
    s["has_news"] = s["n_news"].notna().astype(int)
    s[["sent", "n_news"] + EVENTS] = s[["sent", "n_news"] + EVENTS].fillna(0.0)
    return add_surprise(s)[["ticker", "date"] + SENT_FEATS]


def main():
    news = pd.read_parquet(PROC / "news_scored.parquet", columns=["ticker", "trade_date", "headline", "score"] + EVENTS)
    panel = pd.read_parquet(PROC / "panel.parquet")
    # self-check: the rebuilt block on ALL headlines must equal the panel's sentiment columns
    full = panel[["ticker", "date"]].merge(sentiment_block(news), on=["ticker", "date"], how="left")
    diff = max(float(np.nanmax(np.abs(full[c].to_numpy(float) - panel[c].to_numpy(float)))) for c in SENT_FEATS)
    assert diff == 0.0, f"rebuild does not reproduce panel sentiment (max diff {diff})"
    print("self-check: rebuilt sentiment on all headlines == panel (max diff 0)", flush=True)

    names = name_table()
    news["relevant"] = flag_relevant(news, names)
    news[["ticker", "trade_date", "headline", "relevant"]].to_parquet(str(PROC / "news_relevance.parquet") + ".tmp", index=False)
    os.replace(str(PROC / "news_relevance.parquet") + ".tmp", PROC / "news_relevance.parquet")
    li = ["AAPL", "AMZN", "MSFT", "NVDA", "GOOGL", "TSLA", "META"]
    on_panel = news.merge(panel[["ticker", "date"]].rename(columns={"date": "trade_date"}), on=["ticker", "trade_date"])
    y = lambda d: d["trade_date"].dt.year
    surv = pd.DataFrame({"headlines_all": on_panel.groupby(y(on_panel)).size(),
                         "headlines_relevant": on_panel.groupby(y(on_panel))["relevant"].sum().astype(int),
                         "share_relevant": on_panel.groupby(y(on_panel))["relevant"].mean(),
                         "li7_all": on_panel.loc[on_panel["ticker"].isin(li)].groupby(y(on_panel)).size(),
                         "li7_share_relevant": on_panel.loc[on_panel["ticker"].isin(li)].groupby(y(on_panel))["relevant"].mean()})
    surv = surv.rename_axis("year").reset_index()
    surv.to_csv(str(RESULTS / "relevance_survival.csv") + ".tmp", index=False)
    os.replace(str(RESULTS / "relevance_survival.csv") + ".tmp", RESULTS / "relevance_survival.csv")
    print(surv.round(3).to_string(index=False), flush=True)
    print("overall share relevant (panel days): %.3f" % on_panel["relevant"].mean(), flush=True)

    rel_block = sentiment_block(news.loc[news["relevant"]].drop(columns="relevant"))
    pr = panel.drop(columns=SENT_FEATS).merge(rel_block, on=["ticker", "date"], how="left")
    assert len(pr) == len(panel) and pr[SENT_FEATS].notna().all().all()
    pr = pr[panel.columns]
    assert (pr["date"].to_numpy() == panel["date"].to_numpy()).all() and (pr["ticker"].to_numpy() == panel["ticker"].to_numpy()).all()
    pr.to_parquet(str(PROC / "panel_rel.parquet") + ".tmp", index=False)
    os.replace(str(PROC / "panel_rel.parquet") + ".tmp", PROC / "panel_rel.parquet")
    print("panel_rel", pr.shape, "has_news share: all %.3f relevant %.3f" % (panel["has_news"].mean(), pr["has_news"].mean()),
          flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
