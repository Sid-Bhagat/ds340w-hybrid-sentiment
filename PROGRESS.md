# PROGRESS (single source of truth; keep < 60 lines)

Updated: 2026-11-07 15:15. Project root: /home/sid/ds340w (venv: .venv, `source .venv/bin/activate`).

## Current phase / next step
Session 3 (resumed 12:50 Sat). Sid's new plan: Step1 Li-7 + ticker aliases -> Step2 Phase 1c
historical membership (HARD STOP 14:30, else revert to current constituents) -> Phases 2,3,3b,4.
3b: start only if Phase 3 done by 19:00, drop if not done 22:00. All pre-2023 + MORNING_REPORT by 23:00.
Step 1 DONE 12:58. Step 2 (1c) DONE 13:06: membership.parquet mask, 107/310 removed recovered.
Phase 2 DONE 13:08 (panel 1,329,657 rows, 545 tickers; 13/13 leakage tests pass).
Phase 3 DONE 13:12 (models.py 2 min, evaluate.py -> results/*.csv + tables/*.tex).
Phase 3b LSTM DONE 13:13 (evaluate rerun incl. LSTM). Figures DONE 13:20.
Sessions 3-3e (12:50-14:45): steps 1-2, phases 2-4, audit, robustness, M4, _rel, _irr (see DECISIONS.md).
Session 4 DONE (15:15): infra fixes + gate passed (option 1); 2023 HOLDOUT RUN ONCE with FINAL=1 per FROZEN.md.
Outputs: results/final_2023/ (2023-only tests), results/final_2023/pooled_2018_2023/; figures now 2018-23.
Verdict: no sentiment hypothesis confirmed; H1 reversed (Full t -2.53, Li-7 t -2.27); H3 confirmed in-sample.
data/processed now holds FINAL (2010-2023) data; pre-2023 copy in data/processed/pre2023/. STOPPED.

## Phase checklist
- [x] Phase 0: universe + prices (data/processed/prices.parquet), committed dd03553
- [x] Phase 1: news filter -> FinBERT -> events -> timing rule
  - [x] download All_external.csv
  - [x] verify size + sha256 vs HF metadata (06:01)
  - [x] src/news_filter.py on All_external (06:03)  - [x] extended to both files (06:20)
  - [ ] src/events.py (empty stub), src/finbert_score.py
- [x] 1b hand check (results/hand_check.csv, blind; key in hand_check_key.csv) | [x] 1c membership
- [x] 2 panel + leakage tests | [x] 3 models | [x] 3b LSTM | [x] 4 outputs

## Background jobs
| job | cmd | PID | log | started | expected output |
|---|---|---|---|---|---|
(all jobs finished: downloads, filter 1-4, finbert 1-4, symbol scan, membership, models, LSTM, Li-7 audit,
 EDGAR fetch/build, models_m4, relevance, models_rel. Logs in logs/; none running.)
Resume rule: check `ps -p 2091` and log tail before rerunning. HF resumes partial downloads (.incomplete in data/raw/.cache).

## Key file map
- CLAUDE.md (rules/spec), PLAN.md, DECISIONS.md, PROGRESS.md
- src/edgar.py (8-K), models_m4.py (M4), relevance.py + models_rel.py + models_irr.py (_rel/_irr); src/config.py, prices.py, download_fnspid.py, news_filter.py, events.py, finbert_score.py, hand_check.py
- data/processed: news_filtered/news_scored/news_sealed_2023.parquet, finbert_ckpt/ (all gitignored)
- data/processed: prices.parquet (current), prices_removed.parquet, membership.parquet, hist_members(_audit).csv, universe.csv; sp500_changes_raw.csv is junk. src/tickers.py = alias rules. data/ gitignored

## Verified checks log
- 2026-11-07 05:53-06:35 Phase 1 checks (cuda ok; both FNSPID files size+sha256 == HF; filter/finbert/hand_check counts) - see git history of this file (commit 8232f59)

- 2026-11-07 12:55 filter pass3 (aliases): 1,202,648 rows, 462 tickers, 0 nulls, max trade 2022-12-30; finbert3 scored 19,561 new; news_scored 1,202,648 rows
- 2026-11-07 13:05 membership: 751 cos, 441 current + 107 recovered removed priced; priced/day 344 (2010)..472 (2022). filter pass4 1,387,883 rows 553 tickers; finbert4 +106,718; news_scored 1,387,883 rows
- 2026-11-07 13:07 panel.parquet 1,329,657 x 27, 2010-02-02..2022-12-29, 0 nulls, y mean .499; market.parquet 3252; regimes.parquet 13742 (5 folds); pytest 13 passed
- 2026-11-07 13:11 preds.parquet 574,824 test rows (2018-22), coefs.csv 405 rows; pooled AUC full M2 .5096 / li7 M2 .5255
- 13:20 li7_coverage_audit.csv (9 syms x 6 stages); robust_* CSVs; hold=1 portfolio == previous (4e-15)
- 13:55 edgar: 1,386 JSON parse ok; 103,830 8-Ks/595 tickers pre-2023, 30,677 sealed; 0 member-yrs w/o 8-K; old CSVs diff 0
- 14:08 relevance: self-check diff 0; 43.1% relevant; panel_rel 1,329,657 rows; preds_rel 574,824; 32 old CSVs diff 0

## Open problems
- Note: `python -I` breaks `import config` for our own scripts; use -I only for code reading untrusted data.
- Sid TODO: fill my_label in results/hand_check.csv (positive/neutral/negative), then `python src/hand_check.py --agree`.
- 2021 news is thin (62k vs 109k in 2022); check per-stock coverage in Phase 2. Survivorship reduced by 1c, not removed (203 delisted names missing).
- 2026-11-15 14:48 publishing prep (course submission): README.md, requirements.txt (102 pins), EDGAR UA from SEC_USER_AGENT, docs/ ignored, hand-check artifacts committed (55bf23b, 88c802c). Not pushed. Email remains in history of cdec6c8 (not rewritten).
