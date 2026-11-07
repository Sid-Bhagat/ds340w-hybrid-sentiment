# PROGRESS (single source of truth; keep < 60 lines)

Updated: 2026-11-07 12:52. Project root: /home/sid/ds340w (venv: .venv, `source .venv/bin/activate`).

## Current phase / next step
Session 3 (resumed 12:50 Sat). Sid's new plan: Step1 Li-7 + ticker aliases -> Step2 Phase 1c
historical membership (HARD STOP 14:30, else revert to current constituents) -> Phases 2,3,3b,4.
3b: start only if Phase 3 done by 19:00, drop if not done 22:00. All pre-2023 + MORNING_REPORT by 23:00.
Step 1 DONE 12:58. Step 2 (1c) DONE 13:06: membership.parquet mask, 107/310 removed recovered.
Phase 2 DONE 13:08 (panel 1,329,657 rows, 545 tickers; 13/13 leakage tests pass).
Phase 3 DONE 13:12 (models.py 2 min, evaluate.py -> results/*.csv + tables/*.tex).
Phase 3b LSTM DONE 13:13 (evaluate rerun incl. LSTM). Figures DONE 13:20.
Now: Phase 4 results/SUMMARY.md + results/MORNING_REPORT.md.

## Phase checklist
- [x] Phase 0: universe + prices (data/processed/prices.parquet), committed dd03553
- [x] Phase 1: news filter -> FinBERT -> events -> timing rule
  - [x] download All_external.csv
  - [x] verify size + sha256 vs HF metadata (06:01)
  - [x] src/news_filter.py on All_external (06:03)  - [x] extended to both files (06:20)
  - [ ] src/events.py (empty stub), src/finbert_score.py
- [x] 1b hand check (results/hand_check.csv, blind; key in hand_check_key.csv) | [x] 1c membership
- [x] 2 panel + leakage tests | [x] 3 models | [x] 3b LSTM | [ ] 4 outputs (tables+figs done; SUMMARY/MORNING_REPORT next)

## Background jobs
| job | cmd | PID | log | started | expected output |
|---|---|---|---|---|---|
(all earlier jobs finished: downloads, filter passes 1-4, finbert 1-4, symbol scan, membership)
| models (Phase 3) | `cd src; python models.py` (nohup) | see log | logs/models.log | 2026-11-07 13:10 | data/processed/preds.parquet + results/coefs.csv; log ends DONE |
| LSTM (3b) | `cd src; python lstm.py` (nohup) | see log | logs/lstm.log | 2026-11-07 13:12 | data/processed/preds_lstm.parquet; log ends DONE; then rerun evaluate.py. Drop if not done 22:00 |
Resume rule: check `ps -p 2091` and log tail before rerunning. HF resumes partial downloads (.incomplete in data/raw/.cache).

## Key file map
- CLAUDE.md (rules/spec), PLAN.md, DECISIONS.md, PROGRESS.md
- src/config.py, prices.py, download_fnspid.py, news_filter.py, events.py, finbert_score.py, hand_check.py
- data/processed: news_filtered/news_scored/news_sealed_2023.parquet, finbert_ckpt/ (all gitignored)
- data/processed: prices.parquet (current), prices_removed.parquet, membership.parquet, hist_members(_audit).csv, universe.csv; sp500_changes_raw.csv is junk. src/tickers.py = alias rules. data/ gitignored

## Verified checks log
- 2026-11-07 05:53 torch.cuda.is_available() = True
- 2026-11-07 05:53 HF metadata: Stock_news/All_external.csv = 5731397037 bytes (repo Zihan1004/FNSPID)

- 2026-11-07 06:01 download: 5731397037 bytes == HF size; sha256 == HF lfs sha256 (VERIFIED in log)
- CSV cols: Date (UTC, e.g. '2020-06-05 06:30:54 UTC'), Article_title, Stock_symbol, Url, Publisher, ... Article etc. Convert UTC->America/New_York for timing rule.

- 2026-11-07 06:03 news_filter on All_external: 13,057,514 rows read -> 635,577 kept (383 tickers, 2010-01-03..2020-06-11), sealed=0, no NaNs. File max date is 2020-06-11.

- 2026-11-07 06:09 finbert pass1: 430,361 unique headlines scored in 2.3 min (batch 256, no OOM); news_scored 635,577 rows, 0 nulls, mean score 0.081

- 2026-11-07 06:15 nasdaq_exteral_data.csv: 23232979597 bytes == HF size; sha256 == HF (VERIFIED in logs/download_nasdaq.log)

- 2026-11-07 06:20 news_filter pass2: 28,606,813 rows read -> 1,156,590 kept (451 tickers); by year 2010 24k .. 2019 157k, 2020 118k, 2021 62k, 2022 109k; max trade_date 2022-12-30; none >=2023; 0 nulls

- 2026-11-07 06:23 finbert pass2: 346,696 new hl scored in 1.8 min; news_scored.parquet 1,156,590 rows, 0 nulls, 16 cols (ticker ts_et trade_date headline score 11 ev_*)
- 2026-11-07 12:55 filter pass3 (aliases): 1,202,648 rows, 462 tickers, 0 nulls, max trade 2022-12-30; finbert3 scored 19,561 new; news_scored 1,202,648 rows
- 2026-11-07 13:05 membership: 751 cos, 441 current + 107 recovered removed priced; priced/day 344 (2010)..472 (2022). filter pass4 1,387,883 rows 553 tickers; finbert4 +106,718; news_scored 1,387,883 rows
- 2026-11-07 13:07 panel.parquet 1,329,657 x 27, 2010-02-02..2022-12-29, 0 nulls, y mean .499; market.parquet 3252; regimes.parquet 13742 (5 folds); pytest 13 passed
- 2026-11-07 13:11 preds.parquet 574,824 test rows (2018-22), coefs.csv 405 rows; pooled AUC full M2 .5096 / li7 M2 .5255
- 2026-11-07 06:35 hand_check: 200 rows (67 neg/67 neu/66 pos); agreement fn tested on fake labels

## Open problems
- Note: `python -I` breaks `import config` for our own scripts; use -I only for code reading untrusted data.
- Sid TODO: fill my_label in results/hand_check.csv (positive/neutral/negative), then `python src/hand_check.py --agree`.
- 2021 news is thin (62k vs 109k in 2022); check per-stock coverage in Phase 2. Survivorship reduced by 1c, not removed (203 delisted names missing).
