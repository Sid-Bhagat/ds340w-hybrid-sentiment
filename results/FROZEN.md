# FROZEN: what will be run on 2023 (written before 2023 was opened)

Frozen 2026-11-07 ~13:23, before any 2023 row was loaded, scored, or evaluated. FINAL=1 has never been
set. The 2023 run happens **once**, only after Sid says "open 2023". Nothing below may be changed after
looking at 2023 data. Any deviation must be logged in DECISIONS.md and flagged in SUMMARY.md.

## Fold
- **Test year 2023** (all 2023 trading days that have a next-day target; the last day of 2023 has none).
- **Train** = panel rows dated 2010-02-02 .. 2022-12-30 whose target date (next_date) is before 2023-01-01
  (`folds.split`). Expanding window, same purge rule as the pre-2023 folds.
- Universe = point-in-time S&P 500 members (`membership.parquet` rebuilt with 2023 snapshots), same alias
  rules (`src/tickers.py`, unchanged), same yfinance recovery rule (>= 90% coverage of member days).
- Li-7 subset = AMZN, AAPL, GOOGL, META, MSFT, NVDA, TSLA (same fitted models, evaluated on their rows).

## Models (exactly these; same features, preprocessing, hyperparameters, seeds; no retuning)
| model | definition |
|---|---|
| M0 | logit, price/volume: ret1, ret5, ret20, vol20, volz |
| M1 | M0 + p_high (2-state PCA-HMM, forward-filtered, refit on the 2010-2022 training dates) |
| M2 | M1 + sent, log1p(n_news), has_news, sent_surprise, 11 event dummies |
| M3 | M2 + sent x p_high, sent_surprise x p_high |
| PLACEBO | M2 with the sentiment+event block shuffled across stocks within each date (train seed 340+2023, test seed 340+2024) |
| LGBM | LightGBM on M3 inputs: 300 trees, lr 0.05, 31 leaves, min_child_samples 1000, subsample 0.8 (freq 1), colsample 0.8, lambda 1, seed 340 |
| LSTM | 1-layer LSTM (hidden 32) on M3 inputs, 20-row windows, AdamW lr 1e-3 wd 1e-4, batch 4096, max 15 epochs, patience 3, validation = last 20% of training dates (purged), seed 340+2023 |

Preprocessing (unchanged): clip at the train 1st/99th percentile, z-score with train mean/sd, p_high raw,
interactions = z(sent) x p_high. Logit SEs are clustered by date. Regimes use SPY 1/5/20-day returns,
SPY 20-day vol, and cross-sectional dispersion -> scaler + PCA(2) + GaussianHMM(2, full cov, best of 5
seeds 340..344), all fit on training dates only, then filtered with our own forward pass.

## Tests and outputs (each for Full and Li-7)
**Primary (confirmatory):**
1. **H1:** M2 vs M1, paired daily log-loss difference, Newey-West t (5 lags), two-sided, 2023 test days.
2. **H2:** (i) significance of sent_x_high and surprise_x_high in M3 (2010-2022 training fit, date-clustered SEs);
   (ii) out of sample, M3 vs M2 paired log-loss (as in 1).
3. **H3:** p_high coefficient in M1 vs M2 (2010-2022 training fit).

**Secondary:**
4. Accuracy, AUC and log loss for every model: 2023 alone and pooled 2018-2023.
5. Paired tests: M2 vs PLACEBO, PLACEBO vs M1, M1 vs M0, LGBM vs M3, LSTM vs M3.
6. Sentiment coefficient by regime (low = b_sent, high = b_sent + b_sent_x_high, with SEs).
7. Daily decile long-short (k = max(1, round(0.1 n))): mean daily spread, NW t, annualised Sharpe,
   turnover, gross and net of 10 bps per side.

**Pre-specified robustness (same as pre-2023):**
8. News-only: tests 1 and 5 (M2 vs M1, PLACEBO vs M1, M2 vs PLACEBO) restricted to test stock-days with
   has_news == 1, plus news-day metrics.
9. 5-day rebalanced decile long-short (weights set every 5th test day, held without drift): gross and
   net (10 bps/side), NW t, Sharpe, turnover.

Significance level 0.05, two-sided. No other models, features, tests or subsets will be reported as
2023 results.

## Required mechanical changes when opening 2023 (no modelling changes)
The pre-2023 code hard-codes the seal, so the FINAL run needs these edits (and only these), committed
separately and logged:
- `news_filter.py:89`, `membership.py:44,72,80,107`, `features.py:104`: replace `SEAL_DATE` with
  `config.end_date()` (= 2024-01-01 when FINAL=1).
- `prices.py`: download 2010-2023 for the **frozen** ticker list (`universe.csv` as committed), plus SPY.
  Do **not** re-scrape Wikipedia (today's constituent list would change the universe).
- Trading calendar = SPY dates through 2023 (`news_filter.trading_days`). Rerun the filter so 2023
  headlines get trade dates, then `finbert_score.py` (it scores only new headlines; cached scores are
  reused).
- Pipeline order: prices -> membership -> news_filter -> finbert_score -> features -> regimes ->
  pytest (with FINAL=1 the "2023 sealed" tests are expected to fail; all other tests must pass) ->
  models -> lstm -> evaluate -> figures.

## Added before 2023 was opened, after seeing 2018-2022 results: SEC 8-K extension (M4)
Requested by Sid on 2026-11-07 after the 2018-2022 results above had been seen. **Exploratory**: it is not
part of the primary hypotheses H1-H3, and its 2018-2022 numbers are not independent of that look.
- **M4** = M3 + 8-K features: has_8k, log(1+n_8k) (z-scored on the training fold, not clipped), and
  dummies for items 2.02, 1.01, 2.01, 5.02, 7.01/8.01 and any_other_item (excluding 9.01). Logit with
  date-clustered SEs, same folds, regimes, preprocessing and seeds (`src/models_m4.py`).
- **LGBM_M4** = LightGBM on M4 inputs, same LGB_PARAMS and seed (secondary check).
- 8-K data: `src/edgar.py`. CIK map = company_tickers.json + OVERRIDE (KG, GENZ) + PREDECESSORS
  (20 tickers + GOOG), predecessor filings only before the successor's first 8-K. acceptanceDateTime is
  UTC -> ET; after 16:00:00 ET or non-trading day -> next trading day. Raw JSON cache in data/raw/edgar/
  (fetched 2026-11-07). **For 2023, do not refetch**: the cache already holds the filings, so rerun
  `edgar.py build` only.
- **Tests on 2023 (Full and Li-7):** (i) M4 vs M3 paired daily log loss, NW t (5 lags), two-sided, all test
  days; (ii) the same on test stock-days with has_8k == 1; (iii) LGBM_M4 vs LGBM, all days and has_8k == 1;
  (iv) daily and 5-day-rebalanced decile long-short for M4 and LGBM_M4, gross and net 10 bps/side, NW t,
  Sharpe, turnover; (v) M4 8-K coefficients from the 2010-2022 training fit; (vi) accuracy/AUC/log loss and
  8-K coverage for 2023.
- Mechanical 2023 edit: `edgar.py` build uses `SEAL_DATE` -> replace with `config.end_date()`, like the
  other files listed above. Pipeline order: after `finbert_score.py`, run `edgar.py build`; after
  `models.py`, run `models_m4.py` (before `lstm.py`/`evaluate.py`).

## Added before 2023 was opened, motivated by the annotation audit: relevance-filtered news (M2_rel/M3_rel)
Requested by Sid on 2026-11-07 after the blind LLM annotation (34/94 disagreements = headline about a
different company) and after the 2018-2022 results had been seen. **Robustness variant only.** The main
specification (M0-M3, PLACEBO, LGBM, LSTM, M4) is unchanged and remains primary.
- Relevance rule, name table, alias dict and symbol rule exactly as in `src/relevance.py` (logged in
  DECISIONS.md). Sentiment features rebuilt with features.py's steps from relevant headlines only.
- Models: M2_rel, M3_rel and PLACEBO_rel (`src/models_rel.py`), same folds, regimes, preprocessing,
  logit and seeds as M2/M3/PLACEBO.
- **Tests on 2023 (Full and Li-7):** H1_rel = M2_rel vs M1; H2_rel (OOS) = M3_rel vs M2_rel;
  PLACEBO_rel vs M1; M2_rel vs PLACEBO_rel. Paired daily log loss, NW t (5 lags), two-sided, on all test
  days and on test stock-days with has_news_rel == 1. Also accuracy/AUC/log loss and the 2023 relevance
  survival share.
- 2023 pipeline: after `finbert_score.py` and `features.py`, run `relevance.py` (its self-check must pass)
  and then `models_rel.py` before `evaluate.py`. relevance.py has no date constants to change.

## Comparability note
The FINAL run re-downloads adjusted prices and adds 2022-12-30 as a test row for 2022 (it gains a
target), so the rerun 2018-2022 numbers may differ slightly from the frozen ones in SUMMARY.md. The
frozen pre-2023 numbers stay as reported. 2023 is reported as a new section and the differences are noted.

## Code at freeze (sha256, first 16 hex chars; updated with the M4 extension and relevance variant, 2026-11-07 ~14:15)
| file | sha256 |
|---|---|
| `src/config.py` | `cc13fe7de36bcc78` |
| `src/tickers.py` | `55d90dda280fe03b` |
| `src/news_filter.py` | `7337a87300f48c77` |
| `src/events.py` | `c1aa9767b39b6c37` |
| `src/finbert_score.py` | `989d3d2b68e4291a` |
| `src/membership.py` | `b76b1d6a87e46df5` |
| `src/prices.py` | `dbec4aa3aa46fed9` |
| `src/features.py` | `08bc6cd50792f61c` |
| `src/regimes.py` | `2c7cff6185332cd6` |
| `src/folds.py` | `7928e8248f163351` |
| `src/models.py` | `226eadc99b9b72a2` |
| `src/lstm.py` | `9722a1af7d105539` |
| `src/edgar.py` | `c13179639f50ec2c` |
| `src/models_m4.py` | `9a9363b366c57448` |
| `src/relevance.py` | `84f16b2dc7dbbabd` |
| `src/models_rel.py` | `fcee2818c006d63b` |
| `src/evaluate.py` | `fbe217656a101572` |
| `src/figures.py` | `f3137415e4282af1` |
| `tests/test_leakage.py` | `fccff241049e0e27` |
Git: the commit that adds this file ("Freeze ...").
