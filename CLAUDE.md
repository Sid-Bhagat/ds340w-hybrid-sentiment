# DS 340W hybrid model: build plan

## Environment
- WSL Ubuntu 24.04. Venv at .venv: activate it first and install packages with `uv pip install`.
- Run long jobs with nohup and log to logs/.
- Never combine `cd` with a file write in one shell command (it triggers a manual approval). Use absolute paths instead (e.g. `curl -o /home/sid/ds340w/data/raw/edgar/file.json ...`) or do the write in a Python script.
- Hardware: 15 GB RAM in WSL, so read FNSPID in chunks and never load the full file. RTX 4070 Laptop with 8 GB VRAM, of which about 1 GB is used by Windows. i9-14900HX. FinBERT (ProsusAI/finbert) is already cached.
- Library versions: pandas 3.x (copy-on-write, so never use chained assignment; use .loc), numpy 2.x, transformers 5.x, torch 2.6 cu124. Write code for these versions. If my older Li replication code breaks under pandas 3, fix it and don't downgrade.
- First check torch.cuda.is_available(). If it's False, stop and tell me.

## Context
I have 36 hours until my DS 340W paper is due. Build and run my hybrid model pipeline fast, with no leakage. Read the repo first (my Li PCA-HMM replication may be in it), then write PLAN.md with the phases below and run them in order. Commit after each phase. Stop and ask me only if a phase is blocked. Keep the code simple and readable. Don't add any features beyond this list.

## Design
- Universe: current S&P 500 constituents; Li's 7 stocks flagged as a comparison group. Prices from yfinance, 2010-2023, plus SPY.
- Target: next-day beat-the-market = 1 if the stock's next-day return > SPY's next-day return.
- Folds: expanding walk-forward, test years 2018, 2019, 2020, 2021, 2022.
- 2023 is SEALED: write code that refuses to load 2023 rows unless FINAL=1 is set.

## Phase 1: News + FinBERT (start this first; it runs in the background)
- FNSPID news from Hugging Face. Read it in chunks, keep only S&P 500 tickers, 2010-2023, headline + timestamp.
- FinBERT: max_len 64, fp16, batch 256 (drop to 128 on OOM), 6-8 DataLoader workers for tokenization. Score = P(pos) - P(neg). Checkpoint to parquet every 100k rows so a crash doesn't lose progress.
- Event tags: regex keyword list for earnings beat/miss, guidance, buyback, dividend, merger/acquisition, FDA approval/rejection, investigation/lawsuit, downgrade/upgrade.
- Timing rule: a headline after 4:00pm ET, on a non-trading day, or without a time goes to the NEXT trading day.

## Phase 1b: Hand check (while FinBERT runs)
- Export a stratified random sample of 200 scored headlines to results/hand_check.csv with an empty "my_label" column for me to fill in. Later, report agreement and Cohen's kappa.

## Phase 1c: Historical membership (optional, only if Phase 1 finishes early)
- Use a public historical S&P 500 membership list so each date only includes that day's members. Otherwise, note survivorship bias in SUMMARY.md.

## Phase 2: Daily table (features use info available at day t's close only)
- Price/volume features: 1/5/20-day returns, 20-day volatility, volume z-score.
- Daily sentiment: mean score, headline count, has_news flag (0 sentiment if no news), event dummies.
- Sentiment surprise: today's score minus the stock's trailing 60-day mean (past only).
- Regimes: PCA on market features + Gaussian HMM, refit inside each training fold. Compute FORWARD-FILTERED probabilities with my own forward pass. Do NOT use hmmlearn predict_proba (it smooths with future data).
- pytest leakage tests: shifting the target by one day must change results; no feature may correlate with the target more than its lag does; fold boundaries are strict; 2023 is unreachable without FINAL=1.

## Phase 3: Models (same folds for all of them)
- Logistic regressions: M0 price/volume; M1 + regime; M2 + sentiment and events; M3 + sentiment x regime interactions.
- LightGBM on M3's inputs with light fixed tuning, n_jobs=-1 on CPU.
- Placebo: M2 with sentiment shuffled across stocks within each date.
- Report accuracy, AUC, and log loss for each fold and pooled.
- H1: M2 vs M1 using daily paired log-loss differences, Newey-West t-test.
- H2: significance of the interaction terms in M3.
- H3: regime coefficients in M1 vs M2.
- Portfolio: each day go long the top decile and short the bottom decile of predicted probability. Report mean daily spread, Newey-West t-stat, Sharpe, and the same with 10 bps per-side costs.

## Phase 3b: LSTM (only if Phase 3 is done by hour 10)
- Train in PyTorch on CUDA with M3's inputs, 20-day sequences per stock, the same walk-forward folds, early stopping on the last 20% of each training fold (never the test year), and a fixed seed.
- Add it to all result tables and the portfolio test.
- If it isn't done by hour 13, drop it and list it as future work.

## Phase 4: Outputs for the paper
- Tables saved as CSV and LaTeX in results/.
- Figures (PNG, 300 dpi): cumulative long-short returns, accuracy by fold, regime timeline, sentiment coefficient by regime.
- results/SUMMARY.md: every number the paper needs, plus any deviations from this plan.
- Run everything in Phase 3/3b before 2023 is opened. Only after I say "open 2023": rerun all models once with FINAL=1 and append to SUMMARY.md.

## Overnight autonomy rules (override "stop and ask" above)
- I am asleep. Never wait for my input. If something is ambiguous or blocked, choose the simplest option that respects the leakage rules, log it in DECISIONS.md (what, why, alternatives), and keep going.
- If a step fails 3 times, log it in DECISIONS.md, skip it, and continue with the next phase.
- Run long jobs (FinBERT, walk-forward) with nohup, logging to logs/. Check the log every few minutes until the job finishes. Don't end your turn while a job is running.
- Never set FINAL=1 and never touch 2023 data. Only I open 2023, in a later session.
- Work through Phases 1 to 4 (pre-2023 results only), committing after each phase.
- Clock times (start was about 5:30am Saturday): run Phase 3b (LSTM) only if Phase 3 is done by 3:30pm Saturday; drop it if it isn't finished by 6:30pm.
- When everything is done, write results/MORNING_REPORT.md: what finished, key numbers, anything skipped, and what needs me (including filling in hand_check.csv).

## Usage budget rules
- Never print or read whole data files, parquet files, or logs. Use head, tail -n 50, wc -l, or short pandas summaries.
- While a long job runs, check its log every 10-15 minutes with tail, not more often.
- Keep tool output short (pipe noisy commands through tail). Don't re-read files you've already read unless they changed.

## Progress file and crash safety (context gets cleared often; disk crashed once)
- PROGRESS.md is the single source of truth for where the project stands. At the start of every session, read CLAUDE.md, PROGRESS.md, and DECISIONS.md first. Update PROGRESS.md after every meaningful step.
- PROGRESS.md: current phase and next step; phase checklist; running background jobs (command, PID, log path, start time, expected output); key file map; open problems. Keep it under 60 lines.
- Before starting a background job, record it in PROGRESS.md. When resuming, check PIDs (ps -p) and output files before rerunning anything. Never restart a job that is still running or already done.
- Write every output atomically: write to a .tmp file, then os.replace() to the final name, so a crash never leaves a half-written file.
- After any download or large write, verify it: file size against the source, and the row count or that it opens. Log the check.
- Commit code and small results after every step that works, not just at phase ends.

## Revised clock times (override all earlier times)
- Phase 2 resumed about 12:45pm Saturday after a 6-hour idle gap.
- Phase 3b (LSTM): start only if Phase 3 is done by 7:00pm Saturday; drop it if it isn't finished by 10:00pm.
- All pre-2023 results and MORNING_REPORT.md done by 11:00pm Saturday.
