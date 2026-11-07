"""Phase 3b: LSTM on M3's inputs, 20-day sequences per stock, same walk-forward folds.

Per fold: preprocessing fit on the training fold (models.prep). Early stopping on the LAST 20%
of the training fold's dates (purged so fit-part labels end before validation starts); the test
year is never used for training or stopping. Windows only contain feature rows dated <= t.
Output: data/processed/preds_lstm.parquet (date, ticker, fold, p_LSTM)
"""
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from config import PROC, SEED, test_years
from folds import load_panel, regimes_for, split
from models import SETS, prep

SEQ = 20
FEATS = SETS["M3"]
BATCH = 4096
MAX_EPOCHS = 15
PATIENCE = 3


class Net(nn.Module):
    def __init__(self, n_in, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1]).squeeze(-1)


def windows(X, ends):
    """X: (N, F) GPU tensor sorted by ticker, date. ends: row index of day t -> (B, SEQ, F)."""
    idx = ends[:, None] - torch.arange(SEQ - 1, -1, -1, device=X.device)[None, :]
    return X[idx]


def run_epoch(net, X, y, ends, opt=None):
    lossf = nn.BCEWithLogitsLoss(reduction="sum")
    total, n = 0.0, 0
    order = ends[torch.randperm(len(ends), device=ends.device)] if opt else ends
    for i in range(0, len(order), BATCH):
        e = order[i:i + BATCH]
        with torch.set_grad_enabled(opt is not None):
            loss = lossf(net(windows(X, e)), y[e])
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total += loss.item()
        n += len(e)
    return total / n


@torch.no_grad()
def predict(net, X, ends):
    net.eval()
    return torch.cat([torch.sigmoid(net(windows(X, ends[i:i + BATCH]))) for i in range(0, len(ends), BATCH)])


def main():
    assert torch.cuda.is_available(), "CUDA not available - stop and tell Sid"
    dev = "cuda"
    panel = load_panel()
    out = []
    t0 = time.time()
    for year in test_years():
        torch.manual_seed(SEED + year)
        np.random.seed(SEED + year)
        tr, te = split(panel, year)
        hist = panel.loc[panel["date"] < pd.Timestamp(f"{year + 1}-01-01")]
        reg = regimes_for(year)
        tr, hist = tr.merge(reg, on="date"), hist.merge(reg, on="date")
        _, hist = prep(tr, hist)                       # train-fold stats applied to all rows
        hist = hist.sort_values(["ticker", "date"]).reset_index(drop=True)
        pos = hist.groupby("ticker").cumcount().to_numpy()
        ok = pos >= SEQ - 1                            # full 20-row history of the same ticker
        start = pd.Timestamp(f"{year}-01-01")
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
            print(f"{year} ep{ep} train {tl:.5f} val {vl:.5f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            if bad >= PATIENCE:
                break
        net.load_state_dict(best_state)
        p = predict(net, X, e_te).cpu().numpy()
        r = hist.loc[is_te, ["date", "ticker"]].copy()
        r["fold"], r["p_LSTM"] = year, p
        out.append(r)
        print(f"{year}: fit {fit.sum()} val {val.sum()} (from {pd.Timestamp(cut).date()}) test {is_te.sum()} "
              f"of {len(te)} test rows; best val {best:.5f}", flush=True)
    out = pd.concat(out, ignore_index=True)
    path = PROC / "preds_lstm.parquet"
    out.to_parquet(str(path) + ".tmp", index=False)
    os.replace(str(path) + ".tmp", path)
    print("preds_lstm", out.shape, "DONE", flush=True)


if __name__ == "__main__":
    main()
