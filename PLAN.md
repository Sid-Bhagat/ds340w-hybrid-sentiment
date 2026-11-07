# PLAN — DS 340W hybrid model (sentiment x regime)

Started ~05:30 Sat. Phases run in order; commit after each. 2023 is sealed (FINAL=1 only, set by Sid).

| Phase | What | Scripts | Output |
|---|---|---|---|
| 0 | Universe + prices (current S&P 500 + SPY, yfinance, 2010-2022 now; 2023 only with FINAL=1) | `src/prices.py` | `data/processed/prices.parquet` |
| 1 | FNSPID headlines (chunked read, S&P 500 tickers, 2010-2023), FinBERT score = P(pos)-P(neg), event regex tags, timing rule -> trading day | `src/news_filter.py`, `src/finbert_score.py`, `src/events.py` | `data/processed/news_scored.parquet` |
| 1b | Stratified 200-headline hand check sample | `src/hand_check.py` | `results/hand_check.csv` |
| 1c | (optional) historical S&P 500 membership | `src/membership.py` | membership mask or survivorship note |
| 2 | Daily panel: price/volume features, daily sentiment, surprise (trailing 60d, past only), events; PCA+Gaussian HMM regimes refit per fold with own forward filter; pytest leakage tests | `src/features.py`, `src/regimes.py`, `tests/test_leakage.py` | `data/processed/panel.parquet` |
| 3 | Logit M0-M3, LightGBM, placebo; per-fold + pooled acc/AUC/logloss; H1 (NW t-test on daily paired logloss diff), H2 (interaction terms), H3 (regime coefs M1 vs M2); decile long-short portfolio w/ and w/o 10bps/side | `src/models.py`, `src/evaluate.py` | `results/*.csv` |
| 3b | LSTM (only if Phase 3 done by 15:30; drop if not done by 18:30) | `src/lstm.py` | added to tables |
| 4 | Tables (CSV+LaTeX), figures (300dpi), SUMMARY.md, MORNING_REPORT.md | `src/report.py` | `results/` |

Folds: expanding walk-forward; test years 2018-2022; train = all data from 2010 through year before test.
Deviations/judgement calls are logged in DECISIONS.md.
