# DECISIONS log

- **No Li replication code in repo.** The repo contained only CLAUDE.md and empty data/src/tests dirs. Built everything from scratch. The identity of "Li's 7 stocks" is not in the repo, so `LI_STOCKS` in `src/config.py` is a placeholder (AAPL, MSFT, AMZN, GOOGL, JPM, JNJ, XOM). It is used only to report a comparison-group subset of results; **Sid: edit the list and rerun `python src/evaluate.py`** (cheap). Alternative: skip the group entirely (rejected — subset reporting is cheap and the flag is easy to fix).
- **FNSPID file:** used `Stock_news/All_external.csv` (5.7 GB, headlines) instead of `nasdaq_exteral_data.csv` (23 GB, full articles). We only need headline + timestamp.
- **2023 sealing:** prices are downloaded only through 2022-12-31 unless FINAL=1. FNSPID 2023 headlines are written to a separate sealed file, never scored or loaded without FINAL=1. The last trading day of 2022 therefore has no target and is dropped.
