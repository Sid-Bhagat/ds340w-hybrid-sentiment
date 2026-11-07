"""Diagnostic variant M2_irr / M3_irr / PLACEBO_irr: sentiment built ONLY from headlines the relevance
filter rejected (complement of the _rel set). Added before 2023 was opened, diagnostic for the relevance
result. Same sentiment rebuild (relevance.sentiment_block), folds, regimes, preprocessing, logit, seeds.
Outputs: data/processed/panel_irr.parquet, data/processed/preds_irr.parquet, results/coefs_irr.csv
"""
import os

import pandas as pd

from config import PROC, RESULTS, SEED, enforce_seal, test_years
from features import EVENTS, SENT_FEATS
from folds import regimes_for, split
from models import SENT, SETS, fit_logit, prep, shuffle_within_date
from relevance import sentiment_block


def build_panel_irr():
    news = pd.read_parquet(PROC / "news_scored.parquet", columns=["ticker", "trade_date", "headline", "score"] + EVENTS)
    flags = pd.read_parquet(PROC / "news_relevance.parquet")
    assert len(flags) == len(news) and (flags["ticker"].to_numpy() == news["ticker"].to_numpy()).all() \
        and (flags["headline"].to_numpy() == news["headline"].to_numpy()).all(), "relevance flags misaligned"
    irr = news.loc[~flags["relevant"].to_numpy()]
    panel = pd.read_parquet(PROC / "panel.parquet")
    pi = panel.drop(columns=SENT_FEATS).merge(sentiment_block(irr), on=["ticker", "date"], how="left")
    assert len(pi) == len(panel) and pi[SENT_FEATS].notna().all().all()
    pi = pi[panel.columns]
    assert (pi["date"].to_numpy() == panel["date"].to_numpy()).all() and (pi["ticker"].to_numpy() == panel["ticker"].to_numpy()).all()
    pi.to_parquet(str(PROC / "panel_irr.parquet") + ".tmp", index=False)
    os.replace(str(PROC / "panel_irr.parquet") + ".tmp", PROC / "panel_irr.parquet")
    print("panel_irr", pi.shape, "rejected headlines", len(irr), "has_news share %.3f" % pi["has_news"].mean(), flush=True)
    return pi


def main():
    panel = enforce_seal(build_panel_irr())
    preds, coef_rows = [], []
    for year in test_years():
        tr, te = split(panel, year)
        reg = regimes_for(year)
        tr, te = tr.merge(reg, on="date", how="left"), te.merge(reg, on="date", how="left")
        tr, te = prep(tr, te)
        out = te[["date", "ticker", "has_news"]].rename(columns={"has_news": "has_news_irr"}).copy()
        out["p_M2_irr"] = fit_logit(tr, te, SETS["M2"], "M2_irr", year, coef_rows)
        out["p_M3_irr"] = fit_logit(tr, te, SETS["M3"], "M3_irr", year, coef_rows)
        trp = shuffle_within_date(tr, SENT, SEED + year)
        tep = shuffle_within_date(te, SENT, SEED + year + 1)
        out["p_PLACEBO_irr"] = fit_logit(trp, tep, SETS["M2"], "PLACEBO_irr", year, coef_rows)
        print(year, "irr models done; test has_news_irr %.3f" % out["has_news_irr"].mean(), flush=True)
        preds.append(out)
    preds = pd.concat(preds, ignore_index=True)
    preds.to_parquet(str(PROC / "preds_irr.parquet") + ".tmp", index=False)
    os.replace(str(PROC / "preds_irr.parquet") + ".tmp", PROC / "preds_irr.parquet")
    c = pd.DataFrame(coef_rows)
    c.to_csv(str(RESULTS / "coefs_irr.csv") + ".tmp", index=False)
    os.replace(str(RESULTS / "coefs_irr.csv") + ".tmp", RESULTS / "coefs_irr.csv")
    print("preds_irr", preds.shape, "DONE", flush=True)


if __name__ == "__main__":
    main()
