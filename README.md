# DS 340W: News sentiment × market regime for next-day beat-the-market

## Question

Does daily news sentiment predict whether an S&P 500 stock beats SPY the next day, and does that
relationship depend on the market regime? Regimes come from a PCA + 2-state Gaussian HMM, filtered
forward in time. Sentiment is FinBERT applied to FNSPID headlines. Seven large tech stocks (AMZN, AAPL,
GOOGL, META, MSFT, NVDA, TSLA) are a comparison group that follows Li (2025).

- **Target:** y = 1 if the stock's next-day return > SPY's next-day return (day t → t+1).
- **Features:** only information available at day t's close.
- **Models:** logistic regressions M0 (price/volume), M1 (+ regime), M2 (+ sentiment and events),
  M3 (+ sentiment × regime), LightGBM on M3's inputs, an LSTM, and a placebo (sentiment shuffled across
  stocks within each date). Extensions and robustness variants are also included (SEC 8-K features,
  relevance-filtered news, rejected-news sentiment).
- **Evaluation:** accuracy, AUC and log loss per fold and pooled. Paired daily log-loss tests with
  Newey-West standard errors (H1–H3). A daily long/short decile portfolio, with and without 10 bps per side.

## Data

| Source | What we use | Where it goes |
|---|---|---|
| FNSPID (Hugging Face, `Zihan1004/FNSPID`) | `Stock_news/All_external.csv` and `Stock_news/nasdaq_exteral_data.csv`: headlines with timestamps | `data/raw/Stock_news/` |
| yfinance | Adjusted daily prices for the universe and SPY, 2010–2023 | `data/processed/prices.parquet` |
| Wikipedia, "List of S&P 500 companies" | Current constituents, saved once as `universe.csv` | `data/processed/universe.csv` |
| github.com/fja05680/sp500 | Historical S&P 500 membership, so each date uses that day's members | `data/raw/sp500_hist/` |
| SEC EDGAR (`data.sec.gov` submissions JSON) | 8-K filing metadata only (item numbers, timestamps), for the M4 extension | `data/raw/edgar/` |
| FinBERT (`ProsusAI/finbert`) | Headline sentiment, score = P(pos) − P(neg) | `data/processed/finbert_ckpt/` |

Raw data, caches and processed parquet files are not in the repository (`data/` is gitignored).

## Method in brief

1. **News:** keep headlines that match S&P 500 tickers (aliases in `src/tickers.py`), from 2010 on. Assign
   each headline to a trading day with the timing rule: after 4:00 pm ET, on a non-trading day, or
   without a time → next trading day.
2. **Sentiment and events:** FinBERT score per unique headline; regex event tags (earnings beat/miss,
   guidance, buyback, dividend, M&A, FDA, investigation/lawsuit, downgrade/upgrade); daily mean score,
   headline count, `has_news`; sentiment surprise = today's score − trailing 60-day mean (past days only).
3. **Price/volume:** 1, 5 and 20-day returns, 20-day volatility, volume z-score.
4. **Regimes:** PCA on market features, then a 2-state Gaussian HMM refit inside each training fold.
   Probabilities come from our own forward pass, not hmmlearn's smoothed `predict_proba`.
5. **Membership:** a stock-day is included only if the stock was an S&P 500 member that day.
6. **Folds:** expanding walk-forward (see Data split below). Every model uses the same folds.

Leakage tests are in `tests/test_leakage.py`: the target shift changes results, no feature correlates with
the target more than its lag does, fold boundaries are strict, and the 2023 seal and timing rules are checked.

## Data split

Each test year is predicted by a model trained only on earlier years. The 2023 validation year is sealed
and was opened once, after `results/FROZEN.md` was written. That frozen file fixed the models, tests and
robustness checks in advance.

| Part (course rubric name) | Years | Rule | Stock-day rows | Share of all rows |
|---|---|---|---:|---:|
| Training | 2010–2017 | Expanding window. The 2018 model is trained on 2010–2017. Later folds train on all earlier years. | 754,833 | 52.08% |
| Test | 2018–2022 | Walk-forward folds. Each year is predicted by a model trained only on earlier years. | 575,301 | 39.69% |
| Validation (unseen) | 2023 | Sealed (`FINAL=1` gate) until `FROZEN.md` was written. Opened once; the single holdout run is in `results/final_2023/`. | 119,271 | 8.23% |
| **Total** | 2010–2023 | `data/processed/panel.parquet`, 558 tickers | **1,449,405** | **100%** |

Counts are from `data/processed/panel.parquet` (2010-02-02 to 2023-12-28). Rows per year: 2010 79,869;
2011 87,886; 2012 89,871; 2013 93,275; 2014 95,555; 2015 98,191; 2016 103,172; 2017 107,014; 2018 109,731;
2019 113,282; 2020 116,323; 2021 117,398; 2022 118,567; 2023 119,271.

The pre-2023 run reported 1,329,657 panel rows in `results/SUMMARY.md`. The current file has 1,330,134
rows before 2023, because prices were re-downloaded for the 2023 run (see `DECISIONS.md`, Session 4b).

## AI use

Pipeline code written with Claude Code (Anthropic). FinBERT validation labels produced by Claude Opus 5.5
and Claude Haiku 5.5 as annotators. Annotator A (Opus 5.5) labeled blind to the FinBERT output. Annotator B
(Haiku 5.5) was not blind to A's labels or the FinBERT output. See `results/hand_check_agreement.md`.

## Setup

Requires Python 3.11 and an NVIDIA GPU with CUDA for FinBERT and the LSTM (developed on an RTX 4070 Laptop,
8 GB). The raw FNSPID files are large (`All_external.csv` 5.4 GB, `nasdaq_exteral_data.csv` 22 GB). They are
read in chunks, so 15 GB of RAM is enough.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

`requirements.txt` pins all 102 packages (a full freeze of the venv that produced the results). It includes the
PyTorch CUDA 12.4 index (`torch==2.6.0+cu124`).

The EDGAR fetch needs a contact in the User-Agent, as the SEC asks. Set it in your environment:

```bash
export SEC_USER_AGENT="Your Name your@email.example"
```

## Running the pipeline

Run from the repository root. FinBERT scoring is checkpointed and resumable, so an interrupted run does not
start over.

```bash
# 0. Universe, prices, membership
python -c "import sys; sys.path.insert(0, 'src'); import prices; prices.get_universe()"   # writes universe.csv
python src/prices.py                       # yfinance prices + SPY (2010 to 2022 unless FINAL=1)
python src/membership.py                   # historical membership mask + prices of removed members

# 1. News
python src/download_fnspid.py                                    # All_external.csv (size + sha256 check)
python src/download_fnspid.py Stock_news/nasdaq_exteral_data.csv
python src/news_filter.py                  # S&P 500 tickers, 2010+, trading-day timing rule
python src/finbert_score.py                # FinBERT on unique headlines (checkpointed, resumable)
python src/finbert_score.py --build        # join scores + event tags onto the filtered news

# 2. SEC 8-K metadata (needs SEC_USER_AGENT)
python src/edgar.py fetch
python src/edgar.py build

# 3. Daily panel and regimes
python src/features.py                     # panel.parquet (leakage-tested features)
python src/regimes.py                      # PCA-HMM regimes, refit per fold, forward-filtered

# 4. Models and evaluation
python src/models.py                       # M0-M3, placebo, logit + LightGBM -> results/*.csv
python src/models_m4.py                    # M4 (adds 8-K features)
python src/lstm.py                         # LSTM on CUDA (fixed seed, early stopping on last 20% of each train fold)
python src/relevance.py                    # relevance-filtered news -> panel_rel.parquet
python src/models_rel.py                   # M2_rel / M3_rel / PLACEBO_rel
python src/models_irr.py                   # rejected-headline sentiment -> panel_irr.parquet
python src/evaluate.py                     # all tables (Full and Li-7), H1-H3, portfolios
python src/figures.py                      # PNG figures, 300 dpi

# 5. Tests and hand check
pytest tests/
python src/hand_check.py --agree --input results/hand_check_llm_A.csv --col llm_label
```

Do not re-run `python src/hand_check.py` without `--agree`. That draws the 200-headline sample again and
overwrites the blind sample that annotators labelled.

**Known test status:** with the 2023 data now on disk, three tests that assert the pre-2023 boundary fail by
design: `test_2023_sealed`, one case of `test_8k_timing_rule`, and the max-date check in
`test_8k_files_respect_timing_and_seal`. The other tests pass (see `DECISIONS.md`, Session 4b).

**Sealed 2023 run:** `FINAL=1` opens 2023. It has already been run once, following `results/FROZEN.md`, and
its outputs are in `results/final_2023/`. You do not need `FINAL=1` to reproduce the 2018–2022 results.

**Li (2025) replication:** a separate from-scratch implementation lives in `li_replication/`. It has its own
requirements and venv:

```bash
cd li_replication
uv venv --python 3.11 .venv && source .venv/bin/activate && uv pip install -r requirements.txt
bash run_all.sh
```

## Results

- **Summary of every reported number:** [`results/SUMMARY.md`](results/SUMMARY.md) (includes the 2023 holdout
  section and all deviations from the plan).
- **Figures (300 dpi):** [`results/figures/`](results/figures/): cumulative long-short returns, accuracy by fold,
  regime timeline, sentiment coefficient by regime.
- **Tables:** CSV in `results/`, LaTeX in `results/tables/`.
- **Li replication:** `li_replication/COMPARISON.md`.
- **Hand check of FinBERT labels:** `results/hand_check_agreement.md`.

## Known limitations

- Survivorship bias is reduced by historical membership, not removed. 203 delisted names have no price data.
- FNSPID coverage is thin in 2021 (about 62k headlines, against 109k in 2022).
- The owner's own hand labels (`my_label`) in `results/hand_check.csv` are not yet filled in. The reference
  labels are two LLM annotators, not a human.
- The 2023 holdout was run once. Rerunning it with `FINAL=1` would not be a new test.
