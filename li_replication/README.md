# Li (2025) PCA-HMM replication

Replication of:

> Li, S. (2025). Momentum, volume and investor sentiment study for U.S. technology sector stocks—A hidden Markov model based principal component analysis. *PLOS ONE* 20(9): e0331658. doi:10.1371/journal.pone.0331658

The paper has no public code, so this is a from-scratch implementation of its Sections 3–4. It replicates the analysis of the seven large tech stocks:

1. **PCA** on standardized weekly macro/market series, keeping enough components for at least 75% of the variance (at most 5).
2. A **4-state Gaussian HMM** for each stock, fitted on the full 2015–2024 sample (as in the paper). Its inputs are the PCs plus the stock's weekly return, volume and price range (High − Low) / Close. States are labelled by mean weekly return (0 = lowest).
3. **Eq. 16**: OLS of weekly return on the S&P 500 return, momentum, volume and state dummies.
4. **Eq. 17**: the same OLS without dummies, within each state and on all weeks.

Momentum is the lag-1 autocovariance of weekly returns over the past 4 weeks.

Not included: the sub-sector stocks (Table 5), the risk-aversion parameter γ, and Fig 4.
`COMPARISON.md` compares every finding with the paper and documents the reproducibility checks.

## Data

Yahoo Finance via `yfinance`, **weekly** bars (`interval="1wk"`, auto-adjusted), from **2015-01-01 to 2024-12-31**. `end` is exclusive, so the last bar is the week of 2024-12-26. Each series has 522 weeks, and the regressions use 518 of them.

- Stocks: AMZN, AAPL, GOOGL, META, MSFT, NVDA, TSLA
- PCA inputs: S&P 500 (^GSPC), Nasdaq (^IXIC), 10-year yield (^TNX), 13-week T-bill (^IRX), gold (GC=F), crude oil (CL=F), US Dollar Index (DX-Y.NYB). The two rates are differenced; the others enter as percentage returns.
- Regression market control: S&P 500 weekly return

The first run downloads every series into `data/` (one CSV per ticker) and later runs reuse that cache. The committed results come from a download on 2026-11-08. Yahoo revises its history: our September 2026 and November 2026 downloads differ for gold futures (see `COMPARISON.md`).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
cd li_replication
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python run.py                                # cache in data/, outputs in results/
.venv/bin/python run.py --data-dir path/to/cache       # use another CSV cache
.venv/bin/python run.py --results-dir path/to/output   # write elsewhere
```

A run takes about 10 seconds once the data is cached. The seed is fixed (`--seed`, default 42) and BLAS runs single-threaded, so repeated runs give byte-identical output.

## Expected outputs (`results/`)

| File | Contents |
|---|---|
| `pca_explained_variance.csv` | Explained-variance ratio for PC1–PC5, cumulative share, and which PCs were retained (4, 84.3%) |
| `hmm_states.csv` | Per stock and state: weeks, share of weeks, mean/sd return, mean momentum, volume, price range, S&P 500 return, and expected duration |
| `hmm_fit.csv` | Number of states (4), weeks, log-likelihood, and EM convergence per stock |
| `hmm_state_selection.csv` | Log-likelihood, AIC and BIC for 2–6 states per stock |
| `hmm_states_weekly.csv` | Decoded state for each week and stock |
| `eq16.csv` | Eq. 16: coefficient, SE, p-value, N and R² per stock and term |
| `eq16_state_dummies_ftest.csv` | Joint F-test that all Eq. 16 state dummies are zero |
| `eq17.csv` | Eq. 17 within each state: coefficient, SE, p-value, N and R² |
| `eq17_all_weeks.csv` | Eq. 17 on all weeks (the second row per stock in Li's Table 4) |
| `descriptive_stats.csv` | Li Table 2 analogue: return, momentum and volume statistics |
| `figures/price_trend_<T>.png` | Weekly close coloured by HMM state (Fig 2 analogue) |
| `figures/state_boxplots_<T>.png` | Return, momentum and volume by state (Fig 3 analogue) |
| `run_info.json` | Data folder, seed, date range, row counts, and package versions |
| `reproducibility/` | The checks behind `COMPARISON.md` (see below) |

## Reproducibility checks

`check_reproducibility.sh` reruns the pipeline on the September 2026 cache, compares the result with the September outputs, checks that two runs are byte-identical, and lists coefficient changes between the September and fresh data. It needs `old/` (the September code, cache and outputs), which is kept locally and not in git.

## Layout

```
run.py                    entry point
lirep/config.py           dates, tickers, PCA/HMM settings, seed
lirep/data.py             yfinance download and CSV cache
lirep/features.py         returns, momentum, price range, macro returns
lirep/pca_step.py         PCA
lirep/hmm_step.py         Gaussian HMM, state ordering, AIC/BIC
lirep/regression.py       Eq. 16 / Eq. 17 OLS
lirep/plots.py            figures
compare.py                reproducibility comparisons
check_reproducibility.sh  runs all reproducibility checks
```

## AI use

Code written and cleaned up with assistance from Claude (Anthropic).
