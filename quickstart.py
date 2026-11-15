"""Quick start: score the saved 2022-fold models on the small sample, on CPU, with no data download.

Models (models/) were fit on 2010-2021 and tested on 2022 by src/export_models.py with the frozen settings.
The sample (sample/) holds 2022 rows for the Li-7 stocks + 20 random S&P 500 tickers, with each ticker's
19 rows before 2022 so the LSTM's 20-day windows are complete. Inputs are already preprocessed with the
training fold's statistics (models.prep), so no pipeline code or raw data is needed.

  python quickstart.py
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
MODELS, SAMPLE = ROOT / "models", ROOT / "sample"
LOGITS = ["M0", "M1", "M2", "M3"]


class Net(nn.Module):
    """Same architecture as src/lstm.py."""
    def __init__(self, n_in, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1]).squeeze(-1)


def auc(y, p):
    """ROC AUC via average ranks (ties handled like sklearn)."""
    r = pd.Series(p).rank().to_numpy()
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def scores(name, y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    q = np.clip(p, 1e-7, 1 - 1e-7)                     # same clip as src/evaluate.py
    return {"model": name, "n": len(y), "accuracy": ((p > 0.5) == y).mean(), "auc": auc(y, p),
            "logloss": -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()}


def main():
    t0 = time.time()
    meta = json.loads((MODELS / "meta.json").read_text())
    feats = meta["features"]["M3"]
    seq = meta["lstm"]["seq"]
    s = pd.read_csv(SAMPLE / "sample_2022.csv.gz", parse_dates=["date"])
    s = s.sort_values(["ticker", "date"])
    s = s.reset_index(drop=True)
    test = (s["date"] >= "2022-01-01").to_numpy()
    print(f"sample: {len(s):,} rows, {s['ticker'].nunique()} tickers, {test.sum():,} test rows (2022)")

    preds = pd.DataFrame(index=s.index)
    coefs = pd.read_csv(MODELS / "logit_coefs.csv")
    for name in LOGITS:
        c = coefs.loc[coefs["model"] == name].set_index("term")["coef"]
        cols = [t for t in c.index if t != "const"]
        preds["p_" + name] = 1 / (1 + np.exp(-(c["const"] + s[cols].to_numpy(float) @ c[cols].to_numpy())))

    booster = lgb.Booster(model_file=str(MODELS / "lgbm_M3.txt"))
    preds["p_LGBM"] = booster.predict(s[feats].to_numpy(float))

    net = Net(len(feats), meta["lstm"]["hidden"])
    net.load_state_dict(torch.load(MODELS / "lstm_M3.pt", map_location="cpu", weights_only=True))
    net.eval()
    pos = s.groupby("ticker").cumcount().to_numpy()
    ends = np.flatnonzero(test & (pos >= seq - 1))      # a full 20-row history of the same ticker
    X = torch.tensor(s[feats].to_numpy(np.float32))
    idx = torch.tensor(ends)[:, None] - torch.arange(seq - 1, -1, -1)[None, :]
    with torch.no_grad():
        p = torch.sigmoid(net(X[idx])).numpy()
    preds["p_LSTM"] = np.nan
    preds.loc[ends, "p_LSTM"] = p

    rows = []
    for name in LOGITS + ["LGBM", "LSTM"]:
        m = test & preds["p_" + name].notna().to_numpy()
        rows.append(scores(name, s.loc[m, "y"], preds.loc[m, "p_" + name]))
    out = pd.DataFrame(rows).set_index("model")

    full = ROOT / "results" / "metrics_full.csv"
    if full.exists():
        f = pd.read_csv(full)
        f = f.loc[f["fold"].astype(str) == "2022"].set_index("model")
        for k in ("accuracy", "auc", "logloss"):
            out["full_2022_" + k] = f[k]
    pd.set_option("display.width", 140)
    print("\nSample metrics (2022 test rows), with the committed full-panel 2022 fold for comparison:")
    print(out.round(4).to_string())

    exp = pd.read_csv(MODELS / "sample_expected.csv").set_index("model")
    diff = (out[["accuracy", "auc", "logloss"]] - exp[["accuracy", "auc", "logloss"]]).abs().to_numpy().max()
    print(f"\nCheck vs export-time predictions on this sample: max abs metric diff {diff:.1e} "
          f"({'OK' if diff < 1e-4 else 'MISMATCH'})")
    print("\nNote: the sample is 27 tickers, so its metrics will not exactly match the full-panel results "
          "(118,090 test rows, results/metrics_full.csv). AUCs near 0.5 are expected; the full-panel 2022 fold is "
          "also near 0.5.")
    print(f"done in {time.time() - t0:.1f} s on CPU")


if __name__ == "__main__":
    main()
