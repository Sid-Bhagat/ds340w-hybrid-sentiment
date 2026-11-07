"""M4 = M3 + SEC 8-K metadata features (added before 2023 was opened, after seeing 2018-2022 results).

Same folds, regimes, preprocessing (models.prep; log_n_8k z-scored on the training fold but NOT clipped,
see comment below; 8-K dummies stay 0/1), same logit (date-clustered SEs) and LightGBM params/seed.
Writes data/processed/preds_m4.parquet (p_M4, p_LGBM_M4, has_8k) and results/coefs_m4.csv.
The M0-M3/PLACEBO/LGBM/LSTM predictions are not refit or changed.
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import PROC, RESULTS, enforce_seal, test_years
from edgar import FEATS_8K
from folds import load_panel, regimes_for, split
from models import LGB_PARAMS, SETS, fit_logit, prep

M4 = SETS["M3"] + FEATS_8K
DUMMIES = ["i202", "i101", "i201", "i502", "i701_801", "any_other_item"]


def add_8k(panel):
    e = enforce_seal(pd.read_parquet(PROC / "edgar_daily.parquet"))
    e["date"] = e["date"].astype("datetime64[ns]")
    p = panel.merge(e, on=["ticker", "date"], how="left")
    p[DUMMIES + ["n_8k"]] = p[DUMMIES + ["n_8k"]].fillna(0)
    p["has_8k"] = (p["n_8k"] > 0).astype(int)
    p["log_n_8k"] = np.log1p(p["n_8k"])
    return p


def main():
    panel = add_8k(load_panel())
    preds, coef_rows = [], []
    for year in test_years():
        tr, te = split(panel, year)
        reg = regimes_for(year)
        tr, te = tr.merge(reg, on="date", how="left"), te.merge(reg, on="date", how="left")
        tr, te = prep(tr, te)
        # z-score with train stats but NO 1/99 clip: only ~6% of days have an 8-K, so the 99th pct is
        # log(2) and clipping would make log_n_8k = log(2) * has_8k (perfectly collinear). n_8k is small/bounded.
        mu, sd = tr["log_n_8k"].mean(), tr["log_n_8k"].std()
        for d in (tr, te):
            d["log_n_8k"] = (d["log_n_8k"] - mu) / (sd if sd > 0 else 1.0)
        out = te[["date", "ticker", "has_8k"]].copy()
        out["p_M4"] = fit_logit(tr, te, M4, "M4", year, coef_rows)
        m = lgb.LGBMClassifier(**LGB_PARAMS).fit(tr[M4], tr["y"])
        out["p_LGBM_M4"] = m.predict_proba(te[M4])[:, 1]
        print(year, "M4 + LGBM_M4 done; train has_8k %.3f test has_8k %.3f" % (tr["has_8k"].mean(), te["has_8k"].mean()),
              flush=True)
        preds.append(out)
    preds = pd.concat(preds, ignore_index=True)
    preds.to_parquet(str(PROC / "preds_m4.parquet") + ".tmp", index=False)
    os.replace(str(PROC / "preds_m4.parquet") + ".tmp", PROC / "preds_m4.parquet")
    c = pd.DataFrame(coef_rows)
    c.to_csv(str(RESULTS / "coefs_m4.csv") + ".tmp", index=False)
    os.replace(str(RESULTS / "coefs_m4.csv") + ".tmp", RESULTS / "coefs_m4.csv")
    print("preds_m4", preds.shape, "DONE", flush=True)


if __name__ == "__main__":
    main()
