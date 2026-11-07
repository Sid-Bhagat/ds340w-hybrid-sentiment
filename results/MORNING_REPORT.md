# MORNING REPORT (session 3, Sat 2026-11-07, 12:50-13:30)

## What finished (all pre-2023; FINAL=1 never set; 2023 never loaded)
- **Step 1, Li comparison group + aliases:** LI_STOCKS = AMZN AAPL GOOGL META MSFT NVDA TSLA. I mapped FB->META
  and put Alphabet news on both GOOG and GOOGL. I also mapped about 35 other renames, and dropped recycled
  tickers that were attaching other companies' news. Example: "WTW" headlines from 2010-19 were Weight
  Watchers, not Willis Towers Watson. The changes are in `src/tickers.py`, and the reasons are in DECISIONS.md.
- **Step 2 / Phase 1c, historical membership:** done at 13:06, well before the 14:30 cutoff. I built a
  daily point-in-time mask. 751 companies were members in 2010-22. Of the 310 removed since, 107 were
  recovered and 203 are missing (delisted, so yfinance has no data). Survivorship bias is reduced, not
  removed.
- **Phase 2:** panel of 1,329,657 stock-days, 545 tickers. PCA-HMM regimes are refit per fold and use my own
  forward filter. **13/13 leakage tests pass** (`.venv/bin/python -m pytest -q tests`).
- **Phase 3:** M0-M3 logits, LightGBM and placebo; metrics, NW tests, H2/H3, portfolios. Every result is
  reported for both the Full universe and Li-7.
- **Phase 3b LSTM:** done and included in all tables and the portfolio test.
- **Phase 4:** CSV and LaTeX tables (`results/tables/`), 4 figures at 300 dpi (`results/figures/`), and
  `results/SUMMARY.md`.

## Key numbers (out-of-sample 2018-22)
- Pooled AUC: Full M0 0.5045, M1 0.5091, M2 0.5096, M3 0.5093, LGBM 0.5042, LSTM 0.5107.
  Li-7: M2 0.5255, M3 0.5260.
- **H1 (sentiment adds to regime model): not significant.** Full NW t = 1.24 (p 0.21); Li-7 t = 1.08
  (p 0.28). Real vs shuffled-placebo sentiment: t = 1.77 (p 0.077).
- **Regime helps:** M1 vs M0 t = 2.26 (p 0.024). The regime coefficient is significant in every fold and
  unchanged when sentiment is added (H3).
- **H2:** sentiment matters only in the high-vol regime (coefficient about +0.035 to +0.045 per SD,
  significant in the 2019/2020 training fits; low-vol about 0). The interaction itself is marginal
  (p 0.06-0.38) and does not improve out-of-sample loss.
- **Portfolios:** gross long-short +5 to +8 bps/day on Full (LSTM t = 2.18), but daily turnover is about 2,
  so **every strategy is negative after 10 bps costs** (about -14 to -29 bps/day).

## Update (13:23): pre-2023 robustness, audit, freeze
- **Li-7 audit:** no pipeline bug. The raw FNSPID files have 0 headlines for AAPL, AMZN, MSFT, GOOG/L,
  TSLA and META/FB before 2018. The per-stage counts table is in SUMMARY.md.
- **News-days-only H1 (Full):** M2 vs M1 t = 2.04 (p = 0.041); real vs shuffled t = 2.35 (p = 0.019).
  Li-7: not significant.
- **5-day rebalance:** turnover falls from about 2.1 to about 0.6. Full gross is about 10 bps/day
  (t about 2.3); net is +3 to +5 bps/day (t about 1, not significant).
- **LSTM:** best validation epoch 1-3 in every fold, so it is likely undertrained (not retuned).
- `results/FROZEN.md` lists exactly what will run on 2023.

## Update (13:59): SEC 8-K extension (M4), before 2023
- 103,830 8-Ks (2010-22) for 595 tickers, from EDGAR metadata. acceptanceDateTime is **UTC**, not ET
  (verified on Apple earnings 8-Ks), so it was converted. 21 holding-company CIK changes were fixed with
  verified predecessor CIKs.
- **M4 adds nothing:** no significant 8-K coefficient; M4 vs M3 t = -1.74 (all days), -1.95 (8-K days).
  LGBM_M4 is no better than LGBM. Old results are unchanged. FROZEN.md labels M4 as added after seeing
  the 2018-22 results.

## Update (14:15): relevance-filtered news robustness (M2_rel/M3_rel), before 2023
- Only 43.1% of headlines name their tagged company (symbol or name). On those alone, sentiment adds
  nothing: H1_rel p = 0.80 (all days), 0.90 (news days), against the main spec's 0.21 / 0.041. The small
  main-spec news-day gain comes mostly from headlines about other companies or the market. Main results
  unchanged (diff 0). Added to FROZEN.md as a robustness variant.

## Update (14:45): rejected-headlines diagnostic (last addition before 2023)
- On news days the sentiment gain sits in headlines that do NOT name the company: rejected-only gain 9.7e-5
  [2.3e-6, 1.9e-4], p = 0.045, against relevant-only -4.0e-6 [-6.8e-5, 6.0e-5] with a similar SE. That
  points to spillover (sector/peer/market news), not low power. Main results unchanged (39 CSVs, diff 0).

## Update (15:15): 2023 HOLDOUT (confirmatory, run once with FINAL=1)
- Gate passed under your option 1 (LSTM tolerance 1e-5). Infra fixes committed, then everything in
  FROZEN.md was run once on 2023 (249 test days, 119,271 stock-days; Li-7 1,743).
- **No sentiment hypothesis is confirmed.** H1 is reversed: sentiment makes 2023 predictions worse (Full
  gain -8.2e-5, t -2.53; Li-7 t -2.27). The shuffled placebo also hurts on Full (t -2.28), so much of the
  loss is generic overfitting from adding inputs. On Li-7, real sentiment is worse than shuffled (p 0.020).
  H2 not confirmed (OOS p 0.65; in-sample interaction p 0.44). H3 confirmed in-sample (regime coefficient
  0.059, p 0.0005, unchanged with sentiment), but the regime hurts out of sample in 2023 (t -2.29).
- News-days H1 and placebo: not confirmed on Full (sign reversed, p 0.07 / 0.10); reversed on Li-7.
  The 2018-22 "spillover" pattern reversed: rejected-headline sentiment significantly hurts in 2023 (p 0.041).
- M4 (8-K): not confirmed. Portfolios: every daily strategy loses after costs; the 5-day gross spread
  that was positive in 2018-22 is negative in 2023 (about -7 to -13 bps/day, mostly not significant).
- 2023 coverage: 154,733 headlines (40.3% of stock-days with news); Li-7 80.4%; 8-Ks 5.1% of stock-days.
- Details: SUMMARY.md "2023 holdout (confirmatory)"; files in results/final_2023/.

## Caveats for the paper (details in SUMMARY.md)
- 2021 news is thin: 61,896 headlines, 24.5% of stock-days have news (vs about 53% in 2018-19).
- Li-7 news is sparse in FNSPID: AMZN and META only in 2020, MSFT only in 2022, AAPL only in 2020 and 2022.
- 64% of headlines are date-only and moved to the next trading day. That is conservative, but it likely
  biases sentiment effects toward zero.
- 203 removed S&P members could not be priced, so some survivorship bias remains.

## Needs you
1. **Hand check:** `results/hand_check.csv` is being handled in a separate session, so I did not label or
   commit it. Its files (`src/hand_check.py`, `results/hand_check_*.csv/md`) are left uncommitted for that
   session.
2. Check the alias table in `src/tickers.py`. Unverified calls I made: CBS/VIAC/PARA -> SKYD (yfinance
   SKYD has history back to 2010). Calls I declined: VMRK (I couldn't tell whether it succeeds AVB or EQR).
3. When ready, say "open 2023" to rerun everything once with FINAL=1.
4. The CLAUDE.md edit (revised clock times) is your uncommitted change. I left it alone.
