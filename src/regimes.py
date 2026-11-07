"""PCA + 2-state Gaussian HMM market regimes, refit inside each training fold.

Probabilities are FORWARD-FILTERED with our own forward pass: P(state_t | x_1..x_t), using HMM
parameters estimated on the training fold only. hmmlearn's predict_proba is NOT used (it smooths
with future observations). State 1 = high-volatility regime (ordered by mean SPY 20-day vol).
"""
import os

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

from config import PROC, SEED, test_years
from features import MARKET_FEATS

N_STATES = 2
N_PCS = 2
N_INIT = 5


def forward_filter(X, startprob, transmat, means, covars):
    """Filtered state probabilities, row t uses X[0..t] only. Log-space for stability."""
    T, K = len(X), len(startprob)
    logb = np.column_stack([multivariate_normal(means[k], covars[k], allow_singular=True).logpdf(X)
                            for k in range(K)]).reshape(T, K)
    logA = np.log(transmat + 1e-300)
    out = np.empty((T, K))
    la = np.log(startprob + 1e-300) + logb[0]
    out[0] = np.exp(la - logsumexp(la))
    for t in range(1, T):
        la = logsumexp(np.log(out[t - 1] + 1e-300)[:, None] + logA, axis=0) + logb[t]
        out[t] = np.exp(la - logsumexp(la))
    return out


def fit_regimes(market: pd.DataFrame, train_end: pd.Timestamp):
    """Fit scaler+PCA+HMM on dates < train_end; return (filtered p_high for ALL rows, fit info)."""
    market = market.sort_values("date").reset_index(drop=True)
    tr = market["date"] < train_end
    Xtr = market.loc[tr, MARKET_FEATS].to_numpy()
    mu, sd = Xtr.mean(0), Xtr.std(0)
    pca = PCA(n_components=N_PCS, random_state=SEED).fit((Xtr - mu) / sd)
    Z = pca.transform((market[MARKET_FEATS].to_numpy() - mu) / sd)
    best, best_ll = None, -np.inf
    for i in range(N_INIT):
        m = GaussianHMM(N_STATES, covariance_type="full", n_iter=200, random_state=SEED + i).fit(Z[tr.to_numpy()])
        ll = m.score(Z[tr.to_numpy()])
        if ll > best_ll:
            best, best_ll = m, ll
    probs = forward_filter(Z, best.startprob_, best.transmat_, best.means_, best.covars_)
    # label: high-vol state = higher train-period filtered-weighted mean of SPY 20d vol
    vol = market.loc[tr, "mkt_vol20"].to_numpy()
    w = probs[tr.to_numpy()]
    hi = int(np.argmax((w * vol[:, None]).sum(0) / w.sum(0)))
    info = {"train_end": train_end, "train_start": market.loc[tr, "date"].min(),
            "train_last": market.loc[tr, "date"].max(), "loglik": best_ll,
            "pca_var": pca.explained_variance_ratio_.sum(), "p_stay": np.diag(best.transmat_)[[1 - hi, hi]]}
    return probs[:, hi], info


def main():
    market = pd.read_parquet(PROC / "market.parquet")
    market["date"] = market["date"].astype("datetime64[ns]")
    rows = []
    for y in test_years():
        start, end = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")
        sub = market.loc[market["date"] < end]
        p, info = fit_regimes(sub, start)
        rows.append(pd.DataFrame({"fold": y, "date": sub["date"].to_numpy(), "p_high": p}))
        test_p = p[(sub["date"] >= start).to_numpy()]
        print(y, "train", info["train_start"].date(), "->", info["train_last"].date(),
              "pca var %.2f" % info["pca_var"], "p_stay low/high", np.round(info["p_stay"], 3),
              "mean p_high train %.2f test %.2f" % (p[(sub['date'] < start).to_numpy()].mean(), test_p.mean()),
              flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(str(PROC / "regimes.parquet") + ".tmp", index=False)
    os.replace(str(PROC / "regimes.parquet") + ".tmp", PROC / "regimes.parquet")
    print("regimes", out.shape, "DONE", flush=True)


if __name__ == "__main__":
    main()
