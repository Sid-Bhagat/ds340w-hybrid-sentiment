"""Leakage tests (Phase 2). Run: cd /home/sid/ds340w && .venv/bin/python -m pytest -q tests"""
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402
from features import PRICE_FEATS, SENT_FEATS  # noqa: E402
from folds import split  # noqa: E402
from regimes import fit_regimes, forward_filter  # noqa: E402

PROC = config.PROC


@pytest.fixture(scope="module")
def panel():
    assert os.environ.get("FINAL") != "1", "tests must run with the seal on"
    return pd.read_parquet(PROC / "panel.parquet")


@pytest.fixture(scope="module")
def market():
    m = pd.read_parquet(PROC / "market.parquet")
    m["date"] = m["date"].astype("datetime64[ns]")
    return m


# 1. shifting the target by one day must change results
def test_shifted_target_changes_results(panel):
    d = panel.loc[(panel["date"] >= "2016-01-01") & (panel["date"] < "2019-01-01")].copy()
    d = d.sort_values(["ticker", "date"])
    d["y_shift"] = d.groupby("ticker")["y"].shift(1)
    d = d.dropna(subset=["y_shift"])
    tr, te = d["date"] < "2018-01-01", d["date"] >= "2018-01-01"
    X = d[PRICE_FEATS].to_numpy()
    p1 = LogisticRegression(max_iter=500).fit(X[tr], d.loc[tr, "y"]).predict_proba(X[te])[:, 1]
    p2 = LogisticRegression(max_iter=500).fit(X[tr], d.loc[tr, "y_shift"].astype(int)).predict_proba(X[te])[:, 1]
    assert np.abs(p1 - p2).max() > 1e-3
    assert (d["y"] != d["y_shift"]).mean() > 0.3


# 2. no feature may correlate with the target more than its own lag does (beyond noise)
def _corr_gaps(d, feats):
    d = d.sort_values(["ticker", "date"])
    out = {}
    for f in feats:
        lag = d.groupby("ticker")[f].shift(1)
        ok = lag.notna()
        c0 = abs(np.corrcoef(d.loc[ok, f], d.loc[ok, "y"])[0, 1])
        c1 = abs(np.corrcoef(lag[ok], d.loc[ok, "y"])[0, 1])
        out[f] = (c0, c1)
    return out


def test_no_feature_beats_its_lag(panel):
    d = panel.sample(400_000, random_state=config.SEED) if len(panel) > 400_000 else panel
    d = panel.loc[panel["ticker"].isin(d["ticker"].unique()[:200])]
    feats = [f for f in PRICE_FEATS + SENT_FEATS if d[f].std() > 0]
    for f, (c0, c1) in _corr_gaps(d, feats).items():
        assert c0 < 0.05, f"{f}: |corr with target| = {c0:.3f}, suspiciously high"
        assert c0 - c1 < 0.02, f"{f}: corr {c0:.3f} vs lag {c1:.3f}"


def test_corr_check_catches_a_planted_leak(panel):
    d = panel.loc[panel["ticker"].isin(panel["ticker"].unique()[:100])].copy()
    d["leak"] = d["excess_next"]                      # uses tomorrow's return
    c0, c1 = _corr_gaps(d, ["leak"])["leak"]
    assert c0 > 0.5 and c0 - c1 > 0.02


# 3. fold boundaries are strict
@pytest.mark.parametrize("year", config.TEST_YEARS)
def test_fold_boundaries(panel, year):
    tr, te = split(panel, year)
    start = pd.Timestamp(f"{year}-01-01")
    assert tr["date"].max() < start and tr["next_date"].max() < start   # no train label from test year
    assert te["date"].min() >= start and te["date"].max() < pd.Timestamp(f"{year + 1}-01-01")
    assert len(set(tr.index) & set(te.index)) == 0


def test_regimes_use_train_only(market):
    start = pd.Timestamp("2020-01-01")
    sub = market.loc[market["date"] < "2021-01-01"]
    p_full, info = fit_regimes(sub, start)
    assert info["train_last"] < start
    # changing test-period observations must not change train-period probabilities or the fit
    sub2 = sub.copy()
    sub2.loc[sub2["date"] >= start, "mkt_ret1"] *= -3
    p2, info2 = fit_regimes(sub2, start)
    tr = (sub["date"] < start).to_numpy()
    assert np.allclose(p_full[tr], p2[tr]) and np.isclose(info["loglik"], info2["loglik"])


def test_forward_filter_is_causal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    args = (np.array([.5, .5]), np.array([[.95, .05], [.1, .9]]), np.array([[0, 0], [1, 1.]]),
            np.array([np.eye(2), 2 * np.eye(2)]))
    full = forward_filter(X, *args)
    part = forward_filter(X[:150], *args)
    assert np.allclose(full[:150], part)             # future data never changes past probabilities
    assert np.allclose(full.sum(1), 1)


def test_not_hmmlearn_predict_proba():
    import regimes
    assert "predict_proba" not in inspect.getsource(regimes).split('"""', 2)[2]


# 4. 2023 is unreachable without FINAL=1
def test_2023_sealed(panel):
    assert not config.is_final()
    assert config.end_date() == "2023-01-01" and 2023 not in config.test_years()
    assert panel["date"].max() < pd.Timestamp("2023-01-01")
    assert panel["next_date"].max() < pd.Timestamp("2023-01-01")
    for f in ["prices.parquet", "prices_removed.parquet", "market.parquet", "news_scored.parquet"]:
        col = "trade_date" if f.startswith("news") else "date"
        assert pd.read_parquet(PROC / f, columns=[col])[col].max() < pd.Timestamp("2023-01-01"), f
    fake = pd.DataFrame({"date": pd.to_datetime(["2022-12-30", "2023-01-03"]), "x": [1, 2]})
    assert len(config.enforce_seal(fake)) == 1


def test_sealed_news_not_read_by_pipeline():
    src = Path(config.ROOT) / "src"
    for f in ["features.py", "regimes.py", "folds.py", "models.py", "evaluate.py", "lstm.py"]:
        if (src / f).exists():
            assert "news_sealed" not in (src / f).read_text(), f


# 5. SEC 8-K timing (M4 extension)
def test_8k_acceptance_is_utc_converted_to_et():
    from edgar import parse_acceptance
    s = pd.Series(["2022-10-27T20:30:22.000Z", "2022-01-27T21:30:36.000Z"])   # Apple earnings 8-Ks
    t = parse_acceptance(s)
    assert list(t.dt.strftime("%Y-%m-%d %H:%M:%S")) == ["2022-10-27 16:30:22", "2022-01-27 16:30:36"]


def test_8k_timing_rule():
    from edgar import to_trade_date
    from news_filter import trading_days
    cal = trading_days()
    ts = pd.Series(pd.to_datetime(["2022-03-14 15:59:59", "2022-03-14 16:00:00", "2022-03-14 16:00:01",
                                   "2022-03-18 17:30:00", "2022-03-19 10:00:00", "2022-07-04 09:00:00",
                                   "2022-12-30 16:30:00"]))
    got = to_trade_date(ts, cal)
    want = pd.to_datetime(["2022-03-14", "2022-03-14", "2022-03-15", "2022-03-21", "2022-03-21",
                           "2022-07-05", None])                     # after-close on last day -> beyond calendar
    assert got.iloc[:6].tolist() == list(want[:6]) and pd.isna(got.iloc[6])


def test_8k_files_respect_timing_and_seal():
    k = pd.read_parquet(PROC / "edgar_8k.parquet", columns=["ts_et", "trade_date"])
    assert k["trade_date"].max() < pd.Timestamp("2023-01-01")
    day = k["ts_et"].dt.normalize()
    assert (k["trade_date"] >= day).all()                                     # never earlier than filing
    after = (k["ts_et"] - day) > pd.Timedelta(hours=16)
    assert (k.loc[after, "trade_date"] > day[after]).all()                    # after close -> later day
    s = pd.read_parquet(PROC / "edgar_8k_sealed_2023.parquet", columns=["trade_date"])
    assert (s["trade_date"].isna() | (s["trade_date"] >= pd.Timestamp("2023-01-01"))).all()
    d = pd.read_parquet(PROC / "edgar_daily.parquet", columns=["date"])
    assert d["date"].max() < pd.Timestamp("2023-01-01")


def test_8k_features_do_not_beat_their_lag(panel):
    from models_m4 import add_8k
    d = add_8k(panel.loc[panel["ticker"].isin(panel["ticker"].unique()[:200])])
    for f, (c0, c1) in _corr_gaps(d, ["has_8k", "log_n_8k", "i202", "i502", "any_other_item"]).items():
        assert c0 < 0.05 and c0 - c1 < 0.02, f


def test_sealed_8k_not_read_by_models():
    src = Path(config.ROOT) / "src"
    for f in ["features.py", "models.py", "models_m4.py", "evaluate.py", "lstm.py", "figures.py"]:
        assert "sealed" not in (src / f).read_text(), f
