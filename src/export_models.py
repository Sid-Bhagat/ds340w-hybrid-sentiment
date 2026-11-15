"""Quick-start export: refit the final pre-2023 fold (train through 2021, test 2022) and save the models.

Uses the frozen pre-2023 snapshot (data/processed/pre2023/), the state the committed results came from,
and the pipeline's own code: folds.split, models.prep / fit_logit / LGB_PARAMS, and lstm.Net / run_epoch /
predict with lstm.py's fold-2022 steps. No pipeline file is changed. Needs CUDA (the LSTM was trained on GPU).

Outputs (atomic):
  models/logit_coefs.csv      M0-M3 coefficients (term order = model input order)
  models/lgbm_M3.txt          LightGBM booster (M3 inputs)
  models/lstm_M3.pt           LSTM state_dict (M3 inputs, 20-day sequences)
  models/meta.json            feature lists, fold, seeds, train-fold preprocessing stats
  models/refit_check.csv      refit vs committed 2022 fold metrics (results/metrics_full.csv)
  models/sample_expected.csv  metrics of these models on sample/ (quickstart.py compares against it)
  sample/sample_2022.csv.gz   2022 rows of Li-7 + 20 random tickers, plus each ticker's 19 rows before 2022;
                              preprocessed model inputs, target y, ticker, date only (no headline text)
"""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from config import LI_STOCKS, PROC, RESULTS, ROOT, SEED, enforce_seal
from evaluate import logloss_rows
from folds import split
from lstm import FEATS, MAX_EPOCHS, PATIENCE, SEQ, Net, predict, run_epoch
from models import CONT, LGB_PARAMS, SETS, fit_logit, prep
from sklearn.metrics import roc_auc_score

YEAR = 2022
SNAP = PROC / "pre2023"
MODELS, SAMPLE = ROOT / "models", ROOT / "sample"
N_RANDOM = 20
LOGITS = ["M0", "M1", "M2", "M3"]


def atomic(path, write):
    tmp = str(path) + ".tmp"
    write(tmp)
    os.replace(tmp, path)


def metric_row(model, y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {"model": model, "n": len(y), "accuracy": ((p > 0.5) == y).mean(),
            "auc": roc_auc_score(y, p), "logloss": logloss_rows(y, p).mean()}


def load():
    panel = enforce_seal(pd.read_parquet(SNAP / "panel.parquet"))
    reg = pd.read_parquet(SNAP / "regimes.parquet")
    reg = reg.loc[reg["fold"] == YEAR, ["date", "p_high"]]
    reg["date"] = reg["date"].astype("datetime64[ns]")
    return panel, reg


def train_stats(tr):
    """The clip/z-score constants models.prep derives from the training fold (saved for documentation)."""
    tr = tr.copy()
    tr["log_n_news"] = np.log1p(tr["n_news"])
    out = {}
    for c in CONT:
        lo, hi = tr[c].quantile([0.01, 0.99])
        x = tr[c].clip(lo, hi)
        out[c] = {"clip_lo": float(lo), "clip_hi": float(hi), "mean": float(x.mean()), "sd": float(x.std())}
    return out


def fit_lstm(panel, reg, dev):
    """lstm.main's steps for the 2022 fold only (same seeds, split, early stopping)."""
    torch.manual_seed(SEED + YEAR)
    np.random.seed(SEED + YEAR)
    tr, _ = split(panel, YEAR)
    hist = panel.loc[panel["date"] < pd.Timestamp(f"{YEAR + 1}-01-01")]
    tr, hist = tr.merge(reg, on="date"), hist.merge(reg, on="date")
    _, hist = prep(tr, hist)
    hist = hist.sort_values(["ticker", "date"]).reset_index(drop=True)
    pos = hist.groupby("ticker").cumcount().to_numpy()
    ok = pos >= SEQ - 1
    start = pd.Timestamp(f"{YEAR}-01-01")
    is_tr = ok & (hist["date"] < start).to_numpy() & (hist["next_date"] < start).to_numpy()
    is_te = ok & (hist["date"] >= start).to_numpy()
    tr_dates = np.sort(hist.loc[is_tr, "date"].unique())
    cut = tr_dates[int(0.8 * len(tr_dates))]
    fit = is_tr & (hist["next_date"] < cut).to_numpy()
    val = is_tr & (hist["date"] >= cut).to_numpy()
    X = torch.tensor(hist[FEATS].to_numpy(np.float32), device=dev)
    y = torch.tensor(hist["y"].to_numpy(np.float32), device=dev)
    e_fit, e_val, e_te = (torch.tensor(np.flatnonzero(m), device=dev) for m in (fit, val, is_te))
    net = Net(len(FEATS)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = np.inf, None, 0
    for ep in range(MAX_EPOCHS):
        net.train()
        tl = run_epoch(net, X, y, e_fit, opt)
        net.eval()
        vl = run_epoch(net, X, y, e_val)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        print(f"lstm ep{ep} train {tl:.5f} val {vl:.5f}", flush=True)
        if bad >= PATIENCE:
            break
    net.load_state_dict(best_state)
    p = predict(net, X, e_te).cpu().numpy()
    te_rows = hist.loc[is_te, ["date", "ticker", "y"]].copy()
    te_rows["p_LSTM"] = p
    return net.cpu(), te_rows, hist, pd.Timestamp(cut)


def main():
    assert torch.cuda.is_available(), "CUDA needed to refit the LSTM as frozen"
    MODELS.mkdir(exist_ok=True)
    SAMPLE.mkdir(exist_ok=True)
    panel, reg = load()
    tr, te = split(panel, YEAR)
    stats = train_stats(tr)
    tr, te = tr.merge(reg, on="date", how="left"), te.merge(reg, on="date", how="left")
    assert tr["p_high"].notna().all() and te["p_high"].notna().all()
    tr, te = prep(tr, te)

    # Logits M0-M3 (models.fit_logit; coefficients come from its coef rows)
    coef_rows, refit = [], te[["date", "ticker", "y"]].copy()
    for name in LOGITS:
        refit["p_" + name] = fit_logit(tr, te, SETS[name], name, YEAR, coef_rows)
        print(name, "fit", flush=True)
    coefs = pd.DataFrame(coef_rows)[["model", "term", "coef"]]
    atomic(MODELS / "logit_coefs.csv", lambda t: coefs.to_csv(t, index=False))

    # LightGBM on M3 inputs (models.LGB_PARAMS)
    m = lgb.LGBMClassifier(**LGB_PARAMS).fit(tr[SETS["M3"]], tr["y"])
    refit["p_LGBM"] = m.predict_proba(te[SETS["M3"]])[:, 1]
    atomic(MODELS / "lgbm_M3.txt", lambda t: m.booster_.save_model(t))
    print("LGBM fit", flush=True)

    # LSTM
    net, lstm_te, hist, cut = fit_lstm(panel, reg, "cuda")
    atomic(MODELS / "lstm_M3.pt", lambda t: torch.save(net.state_dict(), t))
    refit = refit.merge(lstm_te[["date", "ticker", "p_LSTM"]], on=["date", "ticker"], how="left")

    # Refit vs committed 2022 fold metrics
    committed = pd.read_csv(RESULTS / "metrics_full.csv")
    committed = committed.loc[committed["fold"].astype(str) == str(YEAR)].set_index("model")
    rows = []
    for name in LOGITS + ["LGBM", "LSTM"]:
        q = refit.dropna(subset=["p_" + name])
        r = metric_row(name, q["y"], q["p_" + name])
        c = committed.loc[name]
        r.update({"n_committed": int(c["n"]),
                  "max_abs_diff": max(abs(r[k] - c[k]) for k in ("accuracy", "auc", "logloss"))})
        rows.append(r)
    check = pd.DataFrame(rows)
    atomic(MODELS / "refit_check.csv", lambda t: check.to_csv(t, index=False))
    print(check.to_string(index=False), flush=True)

    # Predictions vs the frozen prediction files
    old = pd.read_parquet(SNAP / "preds.parquet").query("fold == @YEAR")
    old = old.merge(pd.read_parquet(SNAP / "preds_lstm.parquet")[["date", "ticker", "p_LSTM"]],
                    on=["date", "ticker"], how="left")
    cmp = refit.merge(old, on=["date", "ticker"], suffixes=("", "_old"))
    assert len(cmp) == len(refit) == len(old)
    for name in LOGITS + ["LGBM", "LSTM"]:
        print(f"max |refit - frozen| p_{name}: {np.nanmax(np.abs(cmp['p_' + name] - cmp['p_' + name + '_old'])):.2e}")

    # Sample: Li-7 + 20 random other 2022 tickers; each ticker's 2022 rows + its 19 rows before 2022
    rng = np.random.default_rng(SEED)
    others = sorted(set(te["ticker"]) - set(LI_STOCKS))
    tickers = sorted(LI_STOCKS) + sorted(rng.choice(others, N_RANDOM, replace=False).tolist())
    start = pd.Timestamp(f"{YEAR}-01-01")
    h = hist.loc[hist["ticker"].isin(tickers)]
    pre = h.loc[h["date"] < start].groupby("ticker").tail(SEQ - 1)
    cur = h.loc[h["date"] >= start]
    sample = pd.concat([pre, cur]).sort_values(["ticker", "date"])[["date", "ticker", "y"] + SETS["M3"]]
    sample = sample.reset_index(drop=True)
    sample[SETS["M3"]] = sample[SETS["M3"]].astype("float64")   # 'sent' is float32; exact upcast, as the models saw it
    sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
    path = SAMPLE / "sample_2022.csv.gz"
    atomic(path, lambda t: sample.to_csv(t, index=False, compression="gzip"))
    back = pd.read_csv(path)
    bad = [c for c in SETS["M3"] + ["y"] if not np.array_equal(back[c].to_numpy(), sample[c].to_numpy())]
    assert back.shape == sample.shape and not bad, f"csv round trip differs: {bad}"
    print("sample", sample.shape, "tickers", len(tickers), "rows 2022", len(cur),
          "MB %.2f" % (os.path.getsize(path) / 1e6), flush=True)

    # Expected metrics of these models on the sample (from the export-time predictions)
    s = refit.merge(cur[["date", "ticker"]], on=["date", "ticker"])
    exp = pd.DataFrame([metric_row(n, s.dropna(subset=["p_" + n])["y"], s.dropna(subset=["p_" + n])["p_" + n])
                        for n in LOGITS + ["LGBM", "LSTM"]])
    atomic(MODELS / "sample_expected.csv", lambda t: exp.to_csv(t, index=False))
    print(exp.to_string(index=False), flush=True)

    meta = {"fold": YEAR, "train": "panel rows with date and next_date < 2022-01-01 (2010-02-02 .. 2021)",
            "test": "rows dated 2022", "source": "data/processed/pre2023 (frozen pre-2023 snapshot)",
            "n_train": int(len(tr)), "n_test": int(len(te)), "features": {k: SETS[k] for k in SETS},
            "logit_terms": {n: coefs.loc[coefs["model"] == n, "term"].tolist() for n in LOGITS},
            "lgbm_params": LGB_PARAMS, "lstm": {"seq": SEQ, "hidden": 32, "inputs": FEATS, "seed": SEED + YEAR,
                                                "early_stop_cut": str(cut.date())},
            "preprocessing": "models.prep: clip at train 1/99 pct, z-score with train mean/sd; p_high raw",
            "train_stats": stats, "sample_tickers": tickers, "random_seed": SEED}
    def write_json(t):
        with open(t, "w") as f:
            json.dump(meta, f, indent=1, default=str)
    atomic(MODELS / "meta.json", write_json)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
