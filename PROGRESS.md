# PROGRESS (single source of truth; keep < 60 lines)

Updated: 2026-11-07 06:02. Project root: /home/sid/ds340w (venv: .venv, `source .venv/bin/activate`).

## Current phase / next step
Phase 1 in progress. Download DONE+VERIFIED. Next: write src/news_filter.py (chunked read, S&P500 tickers, 2010-2022; 2023 -> sealed file).

## Phase checklist
- [x] Phase 0: universe + prices (data/processed/prices.parquet), committed dd03553
- [ ] Phase 1: news filter -> FinBERT -> events -> timing rule
  - [x] download All_external.csv
  - [x] verify size + sha256 vs HF metadata (06:01)
  - [ ] src/news_filter.py, src/events.py (empty stub), src/finbert_score.py
- [ ] 1b hand check  | [ ] 1c membership (optional)
- [ ] 2 panel + leakage tests | [ ] 3 models | [ ] 3b LSTM (deadline Sat 15:30) | [ ] 4 outputs

## Background jobs
| job | cmd | PID | log | started | expected output |
|---|---|---|---|---|---|
| FNSPID download | `python src/download_fnspid.py` (nohup) | 2091 (finished) | logs/download_fnspid.log | 2026-11-07 05:54 | data/raw/Stock_news/All_external.csv, 5731397037 bytes, sha256 5d4c018036bd82ca821da71b7a9c0c7db3289642e0fc6f897ea69f4a0c5135c3, log ends "VERIFIED size+sha256 OK" |
Resume rule: check `ps -p 2091` and log tail before rerunning. HF resumes partial downloads (.incomplete in data/raw/.cache).

## Key file map
- CLAUDE.md (rules/spec), PLAN.md, DECISIONS.md, PROGRESS.md
- src/config.py (paths, seal, LI_STOCKS), src/prices.py, src/download_fnspid.py, src/events.py (empty)
- data/processed: prices.parquet, universe.csv, sp500_changes_raw.csv (parquet is gitignored)

## Verified checks log
- 2026-11-07 05:53 torch.cuda.is_available() = True
- 2026-11-07 05:53 HF metadata: Stock_news/All_external.csv = 5731397037 bytes (repo Zihan1004/FNSPID)

- 2026-11-07 06:01 download: 5731397037 bytes == HF size; sha256 == HF lfs sha256 (VERIFIED in log)
- CSV cols: Date (UTC, e.g. '2020-06-05 06:30:54 UTC'), Article_title, Stock_symbol, Url, Publisher, ... Article etc. Convert UTC->America/New_York for timing rule.

## Open problems
- LI_STOCKS is still the PLACEHOLDER (AAPL MSFT AMZN GOOGL JPM JNJ XOM). Sid's message had an
  unfilled "[list]"; real 7 tickers needed. Edit src/config.py, rerun `python src/evaluate.py` later.
- Note: `python -I` breaks `import config` for our own scripts; use -I only for code reading untrusted data.
