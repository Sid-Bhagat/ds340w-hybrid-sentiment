# PROGRESS (single source of truth; keep < 60 lines)

Updated: 2026-11-07 12:52. Project root: /home/sid/ds340w (venv: .venv, `source .venv/bin/activate`).

## Current phase / next step
Session 3 (resumed 12:50 Sat). Sid's new plan: Step1 Li-7 + ticker aliases -> Step2 Phase 1c
historical membership (HARD STOP 14:30, else revert to current constituents) -> Phases 2,3,3b,4.
3b: start only if Phase 3 done by 19:00, drop if not done 22:00. All pre-2023 + MORNING_REPORT by 23:00.
Step 1 DONE 13:20 (aliases, 1,202,648 news rows, 462 tickers). Now: Step 2 `cd src; python membership.py`.

## Phase checklist
- [x] Phase 0: universe + prices (data/processed/prices.parquet), committed dd03553
- [x] Phase 1: news filter -> FinBERT -> events -> timing rule
  - [x] download All_external.csv
  - [x] verify size + sha256 vs HF metadata (06:01)
  - [x] src/news_filter.py on All_external (06:03)  - [x] extended to both files (06:20)
  - [ ] src/events.py (empty stub), src/finbert_score.py
- [x] 1b hand check (results/hand_check.csv, blind; key in hand_check_key.csv) | [ ] 1c membership (optional)
- [ ] 2 panel + leakage tests | [ ] 3 models | [ ] 3b LSTM (deadline Sat 15:30) | [ ] 4 outputs

## Background jobs
| job | cmd | PID | log | started | expected output |
|---|---|---|---|---|---|
| FNSPID download | `python src/download_fnspid.py` (nohup) | 2091 (finished) | logs/download_fnspid.log | 2026-11-07 05:54 | data/raw/Stock_news/All_external.csv, 5731397037 bytes, sha256 5d4c018036bd82ca821da71b7a9c0c7db3289642e0fc6f897ea69f4a0c5135c3, log ends "VERIFIED size+sha256 OK" |
| news filter | `python src/news_filter.py` (nohup) | 2501 | logs/news_filter.log | 2026-11-07 06:02:28 | data/processed/news_filtered.parquet + news_sealed_2023.parquet; log ends DONE |
| big FNSPID download | `python src/download_fnspid.py Stock_news/nasdaq_exteral_data.csv` (nohup) | 2829 | logs/download_nasdaq.log | 2026-11-07 06:04:47 | data/raw/Stock_news/nasdaq_exteral_data.csv = 23232979597 bytes; log ends VERIFIED |
| finbert (pass 1, All_external headlines only) | `python src/finbert_score.py` (nohup) | 3625 | logs/finbert.log | 2026-11-07 06:06:52 | data/processed/finbert_ckpt/part_*.parquet (cache by headline; resumable) then news_scored.parquet. Rerun later after big-file filter: only NEW headlines get scored |
| news filter pass 2 (both files) | `python src/news_filter.py` (nohup) | 6408 | logs/news_filter2.log | 2026-11-07 06:19:13 | overwrites news_filtered.parquet + news_sealed_2023.parquet; log ends DONE. Idempotent, safe to rerun if it died |
| finbert pass 2 | `python src/finbert_score.py` (nohup) | 6568 | logs/finbert2.log | 2026-11-07 06:21:02 | finbert_ckpt/part_00005+ ; news_scored.parquet (1,156,590 rows). Resumable: just rerun |
| symbol-year scan | `cd src; python symbol_counts.py` (nohup) | 10067 | logs/symbol_counts.log | 2026-11-07 12:51 | data/processed/fnspid_symbol_year_counts.csv; log ends DONE |
| news filter pass 3 (aliases) | `cd src; python news_filter.py` (nohup) | see log | logs/news_filter3.log | 2026-11-07 13:10 | news_filtered.parquet (v1 backed up as *_v1_backup.parquet); log ends DONE. Then `python finbert_score.py` |
Resume rule: check `ps -p 2091` and log tail before rerunning. HF resumes partial downloads (.incomplete in data/raw/.cache).

## Key file map
- CLAUDE.md (rules/spec), PLAN.md, DECISIONS.md, PROGRESS.md
- src/config.py, prices.py, download_fnspid.py, news_filter.py, events.py, finbert_score.py, hand_check.py
- data/processed: news_filtered/news_scored/news_sealed_2023.parquet, finbert_ckpt/ (all gitignored)
- data/processed: prices.parquet, universe.csv, sp500_changes_raw.csv (parquet is gitignored)

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
- 2026-11-07 13:18 filter pass3 (aliases): 1,202,648 rows, 462 tickers, 0 nulls, max trade 2022-12-30; finbert3 scored 19,561 new; news_scored 1,202,648 rows
- 2026-11-07 06:35 hand_check: 200 rows (67 neg/67 neu/66 pos); agreement fn tested on fake labels

## Open problems
- Note: `python -I` breaks `import config` for our own scripts; use -I only for code reading untrusted data.
- Sid TODO: fill my_label in results/hand_check.csv (positive/neutral/negative), then `python src/hand_check.py --agree`.
- 2021 news is thin (62k vs 109k in 2022); check per-stock coverage in Phase 2. Survivorship: universe = current S&P 500 (Step 2 fixing).
- sp500_changes_raw.csv is WRONG (Wikipedia navbox, not the changes table). Need another source for 1c.
