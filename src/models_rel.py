"""Robustness variant M2_rel / M3_rel / PLACEBO_rel: M2, M3, PLACEBO refit with sentiment features built
from RELEVANT headlines only (panel_rel.parquet, src/relevance.py). Added before 2023 was opened,
motivated by the annotation audit. Same folds, regimes, preprocessing, logit and seeds as models.py.
The main specification (preds.parquet) is not touched.
Outputs: data/processed/preds_rel.parquet, results/coefs_rel.csv
"""
import os

import pandas as pd

from config import PROC, RESULTS, SEED, enforce_seal, test_years
from folds import regimes_for, split
from models import SENT, SETS, fit_logit, prep, shuffle_within_date


def main():
    panel = enforce_seal(pd.read_parquet(PROC / "panel_rel.parquet"))
    preds, coef_rows = [], []
    for year in test_years():
        tr, te = split(panel, year)
        reg = regimes_for(year)
        tr, te = tr.merge(reg, on="date", how="left"), te.merge(reg, on="date", how="left")
        tr, te = prep(tr, te)
        out = te[["date", "ticker", "has_news"]].rename(columns={"has_news": "has_news_rel"}).copy()
        out["p_M2_rel"] = fit_logit(tr, te, SETS["M2"], "M2_rel", year, coef_rows)
        out["p_M3_rel"] = fit_logit(tr, te, SETS["M3"], "M3_rel", year, coef_rows)
        trp = shuffle_within_date(tr, SENT, SEED + year)
        tep = shuffle_within_date(te, SENT, SEED + year + 1)
        out["p_PLACEBO_rel"] = fit_logit(trp, tep, SETS["M2"], "PLACEBO_rel", year, coef_rows)
        print(year, "rel models done; test has_news_rel %.3f" % out["has_news_rel"].mean(), flush=True)
        preds.append(out)
    preds = pd.concat(preds, ignore_index=True)
    preds.to_parquet(str(PROC / "preds_rel.parquet") + ".tmp", index=False)
    os.replace(str(PROC / "preds_rel.parquet") + ".tmp", PROC / "preds_rel.parquet")
    c = pd.DataFrame(coef_rows)
    c.to_csv(str(RESULTS / "coefs_rel.csv") + ".tmp", index=False)
    os.replace(str(RESULTS / "coefs_rel.csv") + ".tmp", RESULTS / "coefs_rel.csv")
    print("preds_rel", preds.shape, "DONE", flush=True)


if __name__ == "__main__":
    main()
