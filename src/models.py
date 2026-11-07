"""Phase 3: walk-forward models on the full universe (Li-7 is a reporting subset in evaluate.py).

M0 price/volume; M1 + regime (p_high); M2 + sentiment & events; M3 + sentiment x regime;
LGBM on M3 inputs; PLACEBO = M2 with sentiment shuffled across stocks within each date.
Logits: statsmodels, SEs clustered by date. Preprocessing fit on the training fold only.
Outputs: data/processed/preds.parquet, results/coefs.csv
"""
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import PROC, RESULTS, SEED, test_years
from features import EVENTS, PRICE_FEATS
from folds import load_panel, regimes_for, split

SENT = ["sent", "log_n_news", "has_news", "sent_surprise"] + EVENTS
INTER = ["sent_x_high", "surprise_x_high"]
SETS = {"M0": PRICE_FEATS, "M1": PRICE_FEATS + ["p_high"],
        "M2": PRICE_FEATS + ["p_high"] + SENT, "M3": PRICE_FEATS + ["p_high"] + SENT + INTER}
CONT = PRICE_FEATS + ["sent", "log_n_news", "sent_surprise"]      # clipped + z-scored on train
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=1000,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  n_jobs=-1, random_state=SEED, verbose=-1)


def prep(tr, te):
    """Clip at train 1/99 pct, z-score with train mean/sd, add interactions (p_high stays raw 0-1)."""
    tr, te = tr.copy(), te.copy()
    for d in (tr, te):
        d["log_n_news"] = np.log1p(d["n_news"])
    for c in CONT:
        lo, hi = tr[c].quantile([0.01, 0.99])
        mu, sd = tr[c].clip(lo, hi).mean(), tr[c].clip(lo, hi).std()
        for d in (tr, te):
            d[c] = (d[c].clip(lo, hi) - mu) / (sd if sd > 0 else 1.0)
    for d in (tr, te):
        d["sent_x_high"] = d["sent"] * d["p_high"]
        d["surprise_x_high"] = d["sent_surprise"] * d["p_high"]
    return tr, te


def shuffle_within_date(d, cols, seed):
    """Placebo: permute the sentiment block across stocks within each date (rows move together)."""
    d = d.copy()
    rng = np.random.default_rng(seed)
    order = d.groupby("date", sort=False).ngroup().to_numpy()
    perm = np.lexsort((rng.random(len(d)), order))         # random order within each date block
    pos = np.lexsort((np.arange(len(d)), order))           # original order within each date block
    vals = d[cols].to_numpy()
    out = np.empty_like(vals)
    out[pos] = vals[perm]
    d.loc[:, cols] = out
    return d


def fit_logit(tr, te, cols, name, year, coef_rows):
    cols = [c for c in cols if tr[c].std() > 0]            # e.g. an event never seen in training
    X = sm.add_constant(tr[cols].to_numpy(dtype=float))
    groups = tr["date"].factorize()[0]
    res = sm.Logit(tr["y"].to_numpy(), X).fit(disp=0, maxiter=200, cov_type="cluster",
                                             cov_kwds={"groups": groups})
    names = ["const"] + cols
    cov = pd.DataFrame(res.cov_params(), index=names, columns=names)
    for i, n in enumerate(names):
        coef_rows.append({"fold": year, "model": name, "term": n, "coef": res.params[i], "se": res.bse[i],
                          "z": res.tvalues[i], "p": res.pvalues[i],
                          "cov_with_sent": cov.loc[n, "sent"] if "sent" in names else np.nan})
    return res.predict(sm.add_constant(te[cols].to_numpy(dtype=float), has_constant="add"))


def main():
    panel = load_panel()
    preds, coef_rows = [], []
    t0 = time.time()
    for year in test_years():
        tr, te = split(panel, year)
        reg = regimes_for(year)
        tr, te = tr.merge(reg, on="date", how="left"), te.merge(reg, on="date", how="left")
        assert tr["p_high"].notna().all() and te["p_high"].notna().all()
        tr, te = prep(tr, te)
        out = te[["date", "ticker", "y", "ret_next", "excess_next", "has_news", "p_high"]].copy()
        out["fold"] = year
        for name, cols in SETS.items():
            out["p_" + name] = fit_logit(tr, te, cols, name, year, coef_rows)
            print(year, name, "done %.1f min" % ((time.time() - t0) / 60), flush=True)
        trp = shuffle_within_date(tr, SENT, SEED + year)
        tep = shuffle_within_date(te, SENT, SEED + year + 1)
        out["p_PLACEBO"] = fit_logit(trp, tep, SETS["M2"], "PLACEBO", year, coef_rows)
        m = lgb.LGBMClassifier(**LGB_PARAMS).fit(tr[SETS["M3"]], tr["y"])
        out["p_LGBM"] = m.predict_proba(te[SETS["M3"]])[:, 1]
        print(year, "placebo+lgbm done %.1f min; n_train %d n_test %d" % ((time.time() - t0) / 60, len(tr), len(te)),
              flush=True)
        preds.append(out)
    preds = pd.concat(preds, ignore_index=True)
    path = PROC / "preds.parquet"
    preds.to_parquet(str(path) + ".tmp", index=False)
    os.replace(str(path) + ".tmp", path)
    coefs = pd.DataFrame(coef_rows)
    coefs.to_csv(str(RESULTS / "coefs.csv") + ".tmp", index=False)
    os.replace(str(RESULTS / "coefs.csv") + ".tmp", RESULTS / "coefs.csv")
    print("preds", preds.shape, "coefs", coefs.shape, "DONE", flush=True)


if __name__ == "__main__":
    main()
