"""Phase 3 evaluation: every table twice, full universe and Li-7 subset (same models).

Metrics per fold + pooled; paired daily log-loss tests (Newey-West); H2/H3 from coefs.csv;
long-short portfolios; news coverage per test year. Writes results/*.csv and results/tables/*.tex.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from config import LI_STOCKS, PROC, RESULTS, enforce_seal

NW_LAGS = 5
COST = 0.0010                       # 10 bps per side, charged on turnover
PAIRS = [("M1", "M2", "H1: + sentiment"), ("M2", "M3", "H2 (OOS): + interactions"),
         ("M1", "PLACEBO", "placebo vs M1"), ("PLACEBO", "M2", "real vs shuffled sentiment"),
         ("M0", "M1", "+ regime"), ("M3", "LGBM", "LGBM vs M3"), ("M3", "LSTM", "LSTM vs M3"),
         ("M3", "M4", "M4 vs M3: + 8-K"), ("LGBM", "LGBM_M4", "LGBM: + 8-K inputs"),
         ("M1", "M2_rel", "H1_rel: + relevant sentiment"), ("M2_rel", "M3_rel", "H2_rel (OOS): + interactions"),
         ("M1", "PLACEBO_rel", "placebo_rel vs M1"), ("PLACEBO_rel", "M2_rel", "real vs shuffled relevant sentiment")]
REL_TESTS = ["H1_rel: + relevant sentiment", "H2_rel (OOS): + interactions", "placebo_rel vs M1",
             "real vs shuffled relevant sentiment"]
TEX = RESULTS / "tables"


def save(df, name, index=False):
    TEX.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(RESULTS / f"{name}.csv") + ".tmp", index=index)
    os.replace(str(RESULTS / f"{name}.csv") + ".tmp", RESULTS / f"{name}.csv")
    with open(str(TEX / f"{name}.tex") + ".tmp", "w") as f:
        f.write(df.to_latex(index=index, float_format="%.4f", escape=True))
    os.replace(str(TEX / f"{name}.tex") + ".tmp", TEX / f"{name}.tex")


def nw_mean(x, lags=NW_LAGS):
    x = np.asarray(x, dtype=float)
    r = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return r.params[0], r.tvalues[0], r.pvalues[0]


def logloss_rows(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def load_preds():
    p = pd.read_parquet(PROC / "preds.parquet")
    lstm = PROC / "preds_lstm.parquet"
    if lstm.exists():
        p = p.merge(pd.read_parquet(lstm)[["date", "ticker", "p_LSTM"]], on=["date", "ticker"], how="left")
    rel = PROC / "preds_rel.parquet"
    if rel.exists():
        p = p.merge(pd.read_parquet(rel)[["date", "ticker", "has_news_rel", "p_M2_rel", "p_M3_rel", "p_PLACEBO_rel"]],
                    on=["date", "ticker"], how="left")
    m4 = PROC / "preds_m4.parquet"
    if m4.exists():
        p = p.merge(pd.read_parquet(m4)[["date", "ticker", "has_8k", "p_M4", "p_LGBM_M4"]], on=["date", "ticker"],
                    how="left")
    return enforce_seal(p)


def models_in(p):
    return [c[2:] for c in p.columns if c.startswith("p_") and c != "p_high"]


def metrics(p, models):
    rows = []
    for fold, d in list(p.groupby("fold")) + [("pooled", p)]:
        for m in models:
            q = d.dropna(subset=["p_" + m])
            if q["y"].nunique() < 2:
                continue
            rows.append({"fold": fold, "model": m, "n": len(q),
                         "accuracy": ((q["p_" + m] > 0.5) == q["y"]).mean(),
                         "auc": roc_auc_score(q["y"], q["p_" + m]),
                         "logloss": logloss_rows(q["y"], q["p_" + m]).mean()})
    return pd.DataFrame(rows)


def paired_tests(p, models):
    rows = []
    for a, b, label in PAIRS:
        if a not in models or b not in models:
            continue
        q = p.dropna(subset=["p_" + a, "p_" + b])
        diff = logloss_rows(q["y"], q["p_" + a]) - logloss_rows(q["y"], q["p_" + b])   # >0: b better
        daily = diff.groupby(q["date"]).mean()
        m, t, pv = nw_mean(daily)
        rows.append({"test": label, "base": a, "model": b, "days": len(daily),
                     "mean_daily_logloss_gain": m, "nw_t": t, "p_value": pv})
    return pd.DataFrame(rows)


def portfolio(p, m, frac=0.10, hold=1):
    """Long top decile / short bottom decile of predicted prob (k = max(1, round(n*frac))).
    hold=1: rebalance daily. hold=5: set weights every 5th test day and hold them (no drift) in between;
    a held stock with no return that day (e.g. left the index) contributes 0."""
    q = p.dropna(subset=["p_" + m])[["date", "ticker", "ret_next", "p_" + m]].copy()
    q["n"] = q.groupby("date")["ticker"].transform("size")
    q["k"] = np.maximum(1, np.round(q["n"] * frac)).astype(int)
    q["rk_hi"] = q.groupby("date")["p_" + m].rank(ascending=False, method="first")
    q["rk_lo"] = q.groupby("date")["p_" + m].rank(ascending=True, method="first")
    q = q.loc[q["n"] >= 2]
    q["w"] = np.where(q["rk_hi"] <= q["k"], 1.0 / q["k"], 0.0) - np.where(q["rk_lo"] <= q["k"], 1.0 / q["k"], 0.0)
    W = q.pivot_table(index="date", columns="ticker", values="w", fill_value=0.0).sort_index()
    if hold > 1:
        reb = np.arange(len(W)) % hold == 0
        W = W.where(pd.Series(reb, index=W.index), axis=0).ffill()
    R = q.pivot_table(index="date", columns="ticker", values="ret_next").reindex_like(W).fillna(0.0)
    gross = (W * R).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1)
    turnover.iloc[0] = W.iloc[0].abs().sum()
    net = gross - COST * turnover.reindex(gross.index)
    return gross, net, turnover


def portfolio_table(p, models, hold=1):
    rows, series = [], {}
    for m in models:
        gross, net, to = portfolio(p, m, hold=hold)
        series[m] = net
        for kind, s in [("gross", gross), ("net_10bps", net)]:
            mean, t, pv = nw_mean(s)
            rows.append({"model": m, "returns": kind, "days": len(s), "mean_daily_bps": 1e4 * mean,
                         "nw_t": t, "p_value": pv, "ann_sharpe": mean / s.std() * np.sqrt(252),
                         "mean_daily_turnover": to.mean()})
    return pd.DataFrame(rows), pd.DataFrame(series)


def coef_tables(group_suffix=""):
    c = pd.read_csv(RESULTS / "coefs.csv")
    h2 = c.loc[(c["model"] == "M3") & c["term"].isin(["sent_x_high", "surprise_x_high"]),
               ["fold", "term", "coef", "se", "z", "p"]]
    h3 = c.loc[c["model"].isin(["M1", "M2"]) & (c["term"] == "p_high")].pivot(index="fold", columns="model",
                                                                               values=["coef", "se", "p"])
    h3.columns = [f"{a}_{b}" for a, b in h3.columns]
    h3 = h3.reset_index()
    h3["coef_change_M2_minus_M1"] = h3["coef_M2"] - h3["coef_M1"]
    # sentiment effect by regime from M3: low = b_sent, high = b_sent + b_sent_x_high
    rows = []
    for fold, d in c.loc[c["model"] == "M3"].groupby("fold"):
        d = d.set_index("term")
        b, i = d.loc["sent", "coef"], d.loc["sent_x_high", "coef"]
        vb, vi, cv = d.loc["sent", "se"] ** 2, d.loc["sent_x_high", "se"] ** 2, d.loc["sent_x_high", "cov_with_sent"]
        rows += [{"fold": fold, "regime": "low-vol", "coef": b, "se": np.sqrt(vb)},
                 {"fold": fold, "regime": "high-vol", "coef": b + i, "se": np.sqrt(vb + vi + 2 * cv)}]
    by_reg = pd.DataFrame(rows)
    sig = c.loc[c["model"].isin(["M2", "M3"]) & ~c["term"].isin(["const"])][["fold", "model", "term", "coef", "se", "p"]]
    return h2, h3, by_reg, sig


def coverage(li_only):
    """Per test year: headlines, stock-days, share of stock-days with news, universe size."""
    panel = enforce_seal(pd.read_parquet(PROC / "panel.parquet", columns=["date", "ticker", "n_news", "has_news"]))
    if li_only:
        panel = panel.loc[panel["ticker"].isin(LI_STOCKS)]
    panel["year"] = panel["date"].dt.year
    g = panel.groupby("year")
    t = pd.DataFrame({"headlines": g["n_news"].sum().astype(int), "stock_days": g.size(),
                      "share_with_news": g["has_news"].mean(),
                      "tickers_per_day": g.size() / g["date"].nunique(), "unique_tickers": g["ticker"].nunique()})
    return t.reset_index()


def coverage_8k(li_only):
    """Per year: 8-K filings on panel stock-days, share of stock-days with an 8-K, tickers with any 8-K."""
    panel = enforce_seal(pd.read_parquet(PROC / "panel.parquet", columns=["date", "ticker"]))
    e = enforce_seal(pd.read_parquet(PROC / "edgar_daily.parquet"))
    e["date"] = e["date"].astype("datetime64[ns]")
    p = panel.merge(e[["ticker", "date", "n_8k", "i202"]], on=["ticker", "date"], how="left").fillna({"n_8k": 0, "i202": 0})
    if li_only:
        p = p.loc[p["ticker"].isin(LI_STOCKS)]
    g = p.groupby(p["date"].dt.year)
    t = pd.DataFrame({"filings_8k": g["n_8k"].sum().astype(int), "stock_days": g.size(),
                      "share_with_8k": g["n_8k"].apply(lambda s: (s > 0).mean()),
                      "share_with_item_2_02": g["i202"].mean(),
                      "tickers_with_8k": p.loc[p["n_8k"] > 0].groupby(p["date"].dt.year)["ticker"].nunique()})
    return t.rename_axis("year").reset_index()


def main():
    p = load_preds()
    models = models_in(p)
    for group, sub in [("full", p), ("li7", p.loc[p["ticker"].isin(LI_STOCKS)])]:
        met = metrics(sub, models)
        save(met, f"metrics_{group}")
        pt = paired_tests(sub, models)
        save(pt, f"paired_tests_{group}")
        port, daily = portfolio_table(sub, models)
        save(port, f"portfolio_{group}")
        daily.to_csv(RESULTS / f"portfolio_daily_net_{group}.csv")
        save(coverage(group == "li7"), f"coverage_{group}")
        # pre-specified robustness: (a) news stock-days only, (b) 5-day rebalanced portfolio
        news = paired_tests(sub.loc[sub["has_news"] == 1], models)
        news = news.loc[news["base"].isin(["M1", "PLACEBO"]) & news["model"].isin(["M2", "PLACEBO"])]
        save(news, f"robust_newsonly_paired_{group}")
        save(metrics(sub.loc[sub["has_news"] == 1], models), f"robust_newsonly_metrics_{group}")
        save(portfolio_table(sub, models, hold=5)[0], f"robust_portfolio_hold5_{group}")
        if "M2_rel" in models:   # relevance-filter robustness variant
            relt = paired_tests(sub, models)
            save(relt.loc[relt["test"].isin(REL_TESTS)], f"rel_paired_{group}")
            nd = paired_tests(sub.loc[sub["has_news_rel"] == 1], models)
            save(nd.loc[nd["test"].isin(REL_TESTS)], f"rel_newsonly_paired_{group}")
            save(metrics(sub.loc[sub["has_news_rel"] == 1], models), f"rel_newsonly_metrics_{group}")
        if "M4" in models:   # M4 extension: 8-K days only, and 8-K coverage
            k8 = paired_tests(sub.loc[sub["has_8k"] == 1], models)
            save(k8.loc[k8["model"].isin(["M4", "LGBM_M4"])], f"m4_8konly_paired_{group}")
            save(metrics(sub.loc[sub["has_8k"] == 1], models), f"m4_8konly_metrics_{group}")
            save(coverage_8k(group == "li7"), f"coverage_8k_{group}")
        pooled = met.loc[met["fold"] == "pooled"].set_index("model")[["accuracy", "auc", "logloss"]]
        print(f"== {group}: pooled\n{pooled.round(4).to_string()}\n{pt.round(4).to_string()}\n"
              f"{port.loc[port['returns'] == 'net_10bps', ['model', 'mean_daily_bps', 'nw_t', 'ann_sharpe']].round(3).to_string()}",
              flush=True)
    h2, h3, by_reg, sig = coef_tables()
    save(h2, "h2_interactions")
    save(h3, "h3_regime_coef_M1_vs_M2")
    save(by_reg, "sentiment_coef_by_regime")
    save(sig, "coefs_M2_M3")
    if (RESULTS / "coefs_m4.csv").exists():
        c4 = pd.read_csv(RESULTS / "coefs_m4.csv")
        save(c4.loc[c4["term"].isin(["has_8k", "log_n_8k", "i202", "i101", "i201", "i502", "i701_801",
                                     "any_other_item", "sent", "sent_x_high", "p_high"]),
                    ["fold", "term", "coef", "se", "z", "p"]], "m4_8k_coefs")
    print(h2.round(4).to_string(), "\n", h3.round(4).to_string(), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
