# SUMMARY: sentiment x regime hybrid model (pre-2023 results only)

2023 is sealed and was never loaded (no FINAL=1). All numbers are out-of-sample over the test years
2018-2022 (expanding walk-forward). Each main result is shown twice: **Full** = point-in-time S&P 500
universe, **Li-7** = AMZN, AAPL, GOOGL, META, MSFT, NVDA, TSLA (the same fitted models, evaluated on those
7 tickers' rows). Source CSVs are in `results/`, LaTeX in `results/tables/`, figures in `results/figures/`.

## Data
- News: FNSPID headlines (both files), 1,387,883 headlines on 553 tickers, scored with FinBERT
  (score = P(pos) - P(neg), mean 0.070). 1,159,098 of them fall on member stock-days in the panel.
- Prices: yfinance, 2010-2022, current constituents + 107 recovered ex-members + SPY.
- Panel: 1,329,657 stock-days, 545 tickers, 2010-02-02 to 2022-12-29. Target y = next-day return >
  SPY next-day return, mean 0.499.
- Test rows: 574,824 Full, 8,058 Li-7.

### News coverage and universe size by year (`coverage_full.csv`, `coverage_li7.csv`)
| year | Full headlines | Full share of stock-days with news | Full tickers/day (unique) | Li-7 headlines | Li-7 share with news | Li-7 tickers/day |
|---|---|---|---|---|---|---|
| 2010 | 22,544 | 0.171 | 344 (348) | 0 | 0.000 | 5 |
| 2011 | 42,072 | 0.261 | 349 (357) | 224 | 0.076 | 5 |
| 2012 | 55,721 | 0.318 | 359 (367) | 179 | 0.066 | 5 |
| 2013 | 55,828 | 0.316 | 370 (382) | 144 | 0.066 | 5 |
| 2014 | 64,955 | 0.330 | 379 (386) | 118 | 0.030 | 6 |
| 2015 | 92,285 | 0.418 | 390 (401) | 180 | 0.057 | 6 |
| 2016 | 110,897 | 0.447 | 409 (424) | 332 | 0.077 | 6 |
| 2017 | 116,932 | 0.456 | 426 (440) | 617 | 0.124 | 6 |
| **2018** | 148,026 | 0.529 | 437 (451) | 825 | 0.179 | 6 |
| **2019** | 158,876 | 0.526 | 450 (466) | 2,825 | 0.285 | 6 |
| **2020** | 120,488 | 0.375 | 460 (471) | 3,871 | 0.336 | 6 |
| **2021** | **61,896** | **0.245** | 466 (479) | 2,168 | 0.193 | 7 |
| **2022** | 108,578 | 0.346 | 472 (484) | 12,712 | 0.561 | 7 |

Test years are in bold. **2021 is thin**: it has half the headlines of 2019 and only 24.5% of
stock-days have news. Li-7 coverage is very uneven, and the raw files lack the missing years (see the Li-7 coverage audit
below). In raw FNSPID, AMZN has news only in 2020, META only in 2020 (filed as FB), MSFT only in 2022,
AAPL only in 2020 and 2022, and TSLA in 2019, 2020 and 2022.
NVDA and Alphabet are the only Li-7 names with steady coverage. The Li-7 index count is 5 in 2010-13,
6 from 2014 and 7 from late 2020, because META joined the S&P 500 in Dec 2013 and TSLA in Dec 2020.

## Main results

### Predictive accuracy, pooled 2018-2022 (`metrics_full.csv`, `metrics_li7.csv`)
| model | Full acc | Full AUC | Full logloss | Li-7 acc | Li-7 AUC | Li-7 logloss |
|---|---|---|---|---|---|---|
| M0 price/volume | 0.5020 | 0.5045 | 0.6932 | 0.5177 | 0.5249 | 0.6925 |
| M1 + regime | 0.5056 | 0.5091 | 0.6931 | 0.5154 | 0.5218 | 0.6925 |
| M2 + sentiment/events | 0.5056 | 0.5096 | 0.6930 | 0.5177 | 0.5255 | 0.6924 |
| M3 + sent x regime | 0.5051 | 0.5093 | 0.6930 | 0.5202 | 0.5260 | 0.6924 |
| Placebo (M2, shuffled sent.) | 0.5055 | 0.5091 | 0.6931 | 0.5179 | 0.5223 | 0.6925 |
| LightGBM (M3 inputs) | 0.5028 | 0.5042 | 0.6944 | 0.5163 | 0.5157 | 0.6936 |
| LSTM (M3 inputs, 20d) | 0.5063 | 0.5107 | 0.6930 | 0.5093 | 0.5167 | 0.6927 |

Per-fold Full AUC for M2: 2018 0.5072, 2019 0.5056, 2020 0.5134, 2021 0.5115, 2022 0.4989. All models are
near chance in 2022. Per-fold values for every model are in the CSVs, and `figures/accuracy_by_fold.png`
plots them.

### Paired daily log-loss tests, Newey-West (5 lags), 1,258 test days (`paired_tests_*.csv`)
Gain = mean daily log loss of base minus model, so a positive gain means the model is better.
| test | Full gain | Full t | Full p | Li-7 gain | Li-7 t | Li-7 p |
|---|---|---|---|---|---|---|
| **H1**: M2 vs M1 (+ sentiment) | 1.65e-5 | 1.24 | 0.214 | 1.07e-4 | 1.08 | 0.278 |
| H2 out-of-sample: M3 vs M2 | -9.3e-6 | -0.97 | 0.334 | -7.8e-6 | -0.38 | 0.701 |
| M2 vs placebo (real vs shuffled sentiment) | 2.77e-5 | 1.77 | 0.077 | 9.0e-5 | 0.83 | 0.408 |
| placebo vs M1 | -1.11e-5 | -1.41 | 0.157 | 1.66e-5 | 0.36 | 0.721 |
| M1 vs M0 (+ regime) | 1.22e-4 | **2.26** | **0.024** | -2.1e-5 | -0.18 | 0.854 |
| LightGBM vs M3 | -1.31e-3 | -4.47 | <0.001 | -1.19e-3 | -1.42 | 0.157 |
| LSTM vs M3 | 5.2e-5 | 0.57 | 0.571 | -3.39e-4 | -1.07 | 0.284 |

**H1 is not supported out of sample on all stock-days** (Full p = 0.21; Li-7 p = 0.28). **On news
stock-days only (a pre-specified robustness check, below) it is significant for the Full universe**
(t = 2.04, p = 0.041; real vs shuffled t = 2.35, p = 0.019). Real sentiment beats the
shuffled-sentiment placebo only weakly (p = 0.077, Full). The regime adds a small but significant gain on
the Full universe. LightGBM overfits (significantly worse than M3).

### H2: sentiment x regime interactions in M3, training-fold fits, date-clustered SEs (`h2_interactions.csv`)
| fold (training sample) | sent x high: coef (se), p | surprise x high: coef (se), p |
|---|---|---|
| 2018 (2010-17) | 0.0426 (0.0228), 0.062 | -0.0387 (0.0221), 0.081 |
| 2019 (2010-18) | 0.0400 (0.0218), 0.067 | -0.0362 (0.0210), 0.086 |
| 2020 (2010-19) | 0.0325 (0.0216), 0.133 | -0.0319 (0.0210), 0.128 |
| 2021 (2010-20) | 0.0327 (0.0276), 0.237 | -0.0239 (0.0272), 0.378 |
| 2022 (2010-21) | 0.0230 (0.0264), 0.383 | -0.0145 (0.0260), 0.575 |

The sentiment effect by regime (`sentiment_coef_by_regime.csv`, `figures/sentiment_coef_by_regime.png`)
is the logit coefficient per SD of daily sentiment. In the **low-vol** regime it is about 0 in every fold
(-0.007 to 0.009, all CIs cover 0). In the **high-vol** regime it is +0.032 to +0.045. That high-vol effect
is significant at 5% in the 2019 and 2020 training fits (0.0446, se 0.0178; 0.0375, se 0.0176) and
borderline in 2018. The *difference* (the interaction) is only marginal (p ~0.06-0.09 in early folds,
weaker as samples grow) and does not improve out-of-sample log loss. **H2: weak, in-sample-only support.**

### H3: regime coefficient (p_high) in M1 vs M2 (`h3_regime_coef_M1_vs_M2.csv`)
| fold | M1 coef (se) | M2 coef (se) | change |
|---|---|---|---|
| 2018 | 0.0377 (0.0145) | 0.0373 (0.0145) | -0.0003 |
| 2019 | 0.0383 (0.0146) | 0.0374 (0.0146) | -0.0009 |
| 2020 | 0.0324 (0.0138) | 0.0317 (0.0139) | -0.0007 |
| 2021 | 0.0381 (0.0163) | 0.0375 (0.0163) | -0.0006 |
| 2022 | 0.0487 (0.0168) | 0.0482 (0.0168) | -0.0005 |

The regime coefficient is positive and significant (p < 0.03) in every fold and **essentially unchanged**
when sentiment is added (less than 2% shrinkage). Sentiment and regime carry separate information, and
sentiment does not explain away the regime effect.

Other M2 coefficients, 2022 fold (2010-21 training, `coefs_M2_M3.csv`): sent 0.0167 (p 0.13);
log news count -0.0127 (p 0.011); earnings-beat tag 0.0601 (p 0.001); guidance 0.0431 (p 0.074). The
other event tags are not significant.

### Long-short portfolio: top vs bottom decile of predicted probability, daily (`portfolio_*.csv`)
Mean daily spread in bps, NW t, annualised Sharpe. Net = 10 bps per side on daily turnover.
| model | Full gross bps (t, SR) | Full net bps (t, SR) | Li-7 gross bps (t, SR) | Li-7 net bps (t, SR) |
|---|---|---|---|---|
| M0 | 7.20 (1.63, 0.74) | -13.30 (-3.06, -1.37) | -2.71 (-0.39, -0.17) | -21.50 (-3.22, -1.35) |
| M1 | 6.26 (1.43, 0.66) | -14.33 (-3.32, -1.51) | -2.58 (-0.38, -0.17) | -20.76 (-3.17, -1.33) |
| M2 | 4.89 (1.18, 0.55) | -17.59 (-4.30, -1.97) | 5.35 (0.77, 0.34) | -14.16 (-2.11, -0.90) |
| M3 | 5.31 (1.36, 0.61) | -17.46 (-4.54, -2.01) | 4.91 (0.71, 0.31) | -14.62 (-2.16, -0.93) |
| Placebo | 4.98 (1.22, 0.55) | -17.30 (-4.29, -1.92) | -2.23 (-0.30, -0.14) | -22.82 (-3.09, -1.41) |
| LightGBM | 3.37 (0.91, 0.43) | -28.70 (-7.78, -3.65) | 10.74 (1.52, 0.69) | -19.48 (-2.77, -1.24) |
| LSTM | 7.87 (**2.18**, 1.02) | -14.20 (-3.95, -1.83) | 4.11 (0.63, 0.26) | -15.78 (-2.47, -1.01) |

Mean daily turnover is about 2.0-2.3 (3.0-3.2 for LightGBM), so each strategy pays roughly 20-30 bps a day
in costs. Gross spreads are positive on the Full universe but significant only for the LSTM (t = 2.18).
**No daily-rebalanced strategy survives 10 bps costs.** With 5-day rebalancing (robustness check
below), net returns are +3 to +5 bps/day but not significant. The Li-7 portfolio is 1 long / 1 short (a decile of about 7
stocks), so it is very noisy. Figure: `figures/cumulative_long_short.png`.

### Regimes
2-state PCA-HMM, forward-filtered, refit each fold. Mean out-of-sample P(high-vol) by test year: 2018 0.45,
2019 0.30, 2020 0.67, 2021 0.24, 2022 0.87. Both states are persistent (P(stay) 0.95-0.98).
Figure: `figures/regime_timeline.png`.

## Li-7 coverage audit (`results/li7_coverage_audit.csv`, `src/audit_li7.py`)
Headlines per year at each stage, 2010-2022 (2023 is not counted). Stages (a) and (b) use the raw `Date`
year and the raw `Stock_symbol`. Stages (c) to (e) use the assigned trading-day year. GOOG and GOOGL are
identical from stage (d) on because company news goes to both share classes, and FB is relabelled to META
at stage (d).

| ticker | stage | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | (a) raw All_external | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 473 | 0 | 0 | 473 |
| AAPL | (b) raw nasdaq_exteral | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 473 | 0 | 3,094 | 3,567 |
| AAPL | (c) filtered, pre-dedupe | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 946 | 0 | 3,067 | 4,013 |
| AAPL | (c2) after cross-file dedupe | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 472 | 0 | 2,988 | 3,460 |
| AAPL | (d) after alias rules | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 472 | 0 | 2,988 | 3,460 |
| AAPL | (e) on panel stock-days | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 472 | 0 | 2,968 | 3,440 |
| AMZN | (a) raw All_external | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 278 | 0 | 0 | 278 |
| AMZN | (b) raw nasdaq_exteral | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 278 | 0 | 0 | 278 |
| AMZN | (c) filtered, pre-dedupe | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 556 | 0 | 0 | 556 |
| AMZN | (c2) after cross-file dedupe | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 277 | 0 | 0 | 277 |
| AMZN | (d) after alias rules | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 277 | 0 | 0 | 277 |
| AMZN | (e) on panel stock-days | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 277 | 0 | 0 | 277 |
| MSFT | (b) raw nasdaq_exteral | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2,679 | 2,679 |
| MSFT | (c) filtered, pre-dedupe | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2,655 | 2,655 |
| MSFT | (c2) after cross-file dedupe | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2,577 | 2,577 |
| MSFT | (d) after alias rules | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2,577 | 2,577 |
| MSFT | (e) on panel stock-days | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2,565 | 2,565 |
| NVDA | (a) raw All_external | 0 | 226 | 183 | 144 | 118 | 180 | 337 | 615 | 526 | 508 | 309 | 0 | 0 | 3,146 |
| NVDA | (b) raw nasdaq_exteral | 0 | 226 | 183 | 144 | 118 | 180 | 337 | 615 | 526 | 508 | 309 | 711 | 2,368 | 6,225 |
| NVDA | (c) filtered, pre-dedupe | 0 | 452 | 366 | 288 | 236 | 360 | 666 | 1,236 | 1,050 | 1,014 | 624 | 707 | 2,360 | 9,359 |
| NVDA | (c2) after cross-file dedupe | 0 | 224 | 179 | 144 | 118 | 180 | 332 | 617 | 525 | 506 | 312 | 696 | 2,321 | 6,154 |
| NVDA | (d) after alias rules | 0 | 224 | 179 | 144 | 118 | 180 | 332 | 617 | 525 | 506 | 312 | 696 | 2,321 | 6,154 |
| NVDA | (e) on panel stock-days | 0 | 224 | 179 | 144 | 118 | 180 | 332 | 617 | 525 | 506 | 312 | 696 | 2,318 | 6,151 |
| GOOGL | (a) raw All_external | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 298 | 767 | 689 | 0 | 0 | 1,754 |
| GOOGL | (b) raw nasdaq_exteral | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 298 | 767 | 689 | 0 | 0 | 1,754 |
| GOOGL | (c) filtered, pre-dedupe | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 596 | 1,528 | 1,384 | 0 | 0 | 3,508 |
| GOOGL | (c2) after cross-file dedupe | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 298 | 764 | 689 | 0 | 0 | 1,751 |
| GOOGL | (d) after alias rules | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 300 | 2,319 | 2,378 | 1,472 | 1,867 | 8,336 |
| GOOGL | (e) on panel stock-days | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 300 | 2,319 | 2,378 | 1,472 | 1,860 | 8,329 |
| GOOG | (a) raw All_external | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 47 | 713 | 444 | 0 | 0 | 1,204 |
| GOOG | (b) raw nasdaq_exteral | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 47 | 2,141 | 2,081 | 1,472 | 1,878 | 7,619 |
| GOOG | (c) filtered, pre-dedupe | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 94 | 2,845 | 2,528 | 1,472 | 1,869 | 8,808 |
| GOOG | (c2) after cross-file dedupe | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 47 | 2,086 | 2,073 | 1,472 | 1,867 | 7,545 |
| GOOG | (d) after alias rules | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 300 | 2,319 | 2,378 | 1,472 | 1,867 | 8,336 |
| GOOG | (e) on panel stock-days | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 300 | 2,319 | 2,378 | 1,472 | 1,860 | 8,329 |
| TSLA | (a) raw All_external | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 733 | 1,142 | 0 | 0 | 1,875 |
| TSLA | (b) raw nasdaq_exteral | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 733 | 1,142 | 0 | 3,169 | 5,044 |
| TSLA | (c) filtered, pre-dedupe | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,460 | 2,290 | 0 | 3,146 | 6,896 |
| TSLA | (c2) after cross-file dedupe | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 730 | 1,144 | 0 | 3,018 | 4,892 |
| TSLA | (d) after alias rules | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 730 | 1,144 | 0 | 3,018 | 4,892 |
| TSLA | (e) on panel stock-days | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3,001 | 3,001 |
| META | (d) after alias rules | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 432 | 0 | 0 | 432 |
| META | (e) on panel stock-days | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 432 | 0 | 0 | 432 |
| FB | (a) raw All_external | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 433 | 0 | 0 | 433 |
| FB | (b) raw nasdaq_exteral | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 433 | 0 | 0 | 433 |
| FB | (c) filtered, pre-dedupe | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 866 | 0 | 0 | 866 |
| FB | (c2) after cross-file dedupe | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 432 | 0 | 0 | 432 |

**Finding: no pipeline bug. The raw FNSPID files lack these headlines.** Before 2018, both raw files have
0 headlines for AAPL, AMZN, MSFT, GOOG/GOOGL, TSLA and META/FB; only NVDA has news. AAPL, AMZN and FB
appear only in 2020, and AAPL and TSLA again in 2022. MSFT appears only in 2022, and only in the nasdaq
file. Every symbol-format variant was checked (case, whitespace, other names), and all raw Date strings
parse (`YYYY-MM-DD HH:MM:SS UTC`). The losses between stages are expected:
- (c)->(c2): `nasdaq_exteral_data.csv` contains the `All_external.csv` rows verbatim through 2020-06, so
  the cross-file dedupe halves 2010-2020 counts (e.g. AAPL 2020: 473 + 473 -> 472).
- (b)->(c), about 1% in 2022: headlines whose trading day is 2022-12-30 or later go to the sealed file.
- (d)->(e): TSLA has 0 on panel days in 2019-20 because it joined the S&P 500 on 2020-12-21 (the
  point-in-time membership rule) and the 2020 headlines end in June. A few late-2022 rows are lost
  because their trading day has no next-day target.

## Pre-specified robustness checks (before 2023; same fitted models, no refit)
### (a) News-only test stock-days (has_news == 1)
Sentiment can only help on days with news, so these tests use only test stock-days with at least one
headline. Gain = daily mean log loss of base minus model, Newey-West (5 lags).

**Full universe: paired tests**

| test | days | mean_daily_logloss_gain | nw_t | p_value |
|---|---|---|---|---|
| H1: + sentiment | 1258 | 6.4e-05 | 2.043 | 0.041 |
| placebo vs M1 | 1258 | -1.28e-05 | -1.271 | 0.204 |
| real vs shuffled sentiment | 1258 | 7.68e-05 | 2.351 | 0.019 |

**Full universe: pooled metrics on news days**

| model | n | accuracy | auc | logloss |
|---|---|---|---|---|
| M0 | 230,886 | 0.5010 | 0.5039 | 0.6932 |
| M1 | 230,886 | 0.5035 | 0.5061 | 0.6931 |
| M2 | 230,886 | 0.5036 | 0.5076 | 0.6931 |
| M3 | 230,886 | 0.5036 | 0.5077 | 0.6931 |
| PLACEBO | 230,886 | 0.5033 | 0.5059 | 0.6932 |
| LGBM | 230,886 | 0.5028 | 0.5041 | 0.6945 |
| LSTM | 230,379 | 0.5067 | 0.5106 | 0.6930 |

**Li-7: paired tests**

| test | days | mean_daily_logloss_gain | nw_t | p_value |
|---|---|---|---|---|
| H1: + sentiment | 1208 | 0.000123 | 0.390 | 0.697 |
| placebo vs M1 | 1208 | 2.7e-05 | 0.218 | 0.828 |
| real vs shuffled sentiment | 1208 | 9.64e-05 | 0.289 | 0.773 |

**Li-7: pooled metrics on news days**

| model | n | accuracy | auc | logloss |
|---|---|---|---|---|
| M0 | 2,534 | 0.5122 | 0.5214 | 0.6924 |
| M1 | 2,534 | 0.5158 | 0.5214 | 0.6924 |
| M2 | 2,534 | 0.5241 | 0.5334 | 0.6920 |
| M3 | 2,534 | 0.5249 | 0.5323 | 0.6920 |
| PLACEBO | 2,534 | 0.5170 | 0.5223 | 0.6924 |
| LGBM | 2,534 | 0.5241 | 0.5160 | 0.6943 |
| LSTM | 2,534 | 0.5146 | 0.5239 | 0.6923 |

**Result:** on the Full universe, restricting to news days makes sentiment significant out of sample.
M2 beats M1 with t = 2.04 (p = 0.041), and real sentiment beats shuffled sentiment with t = 2.35
(p = 0.019); the shuffled placebo does not beat M1 (p = 0.20). On all days the same tests gave p = 0.21
and 0.077: the gain is concentrated on news days and is diluted by the roughly 60% of test stock-days
without news. Li-7 remains insignificant (p = 0.70; only 2,534 news stock-days). This is one of two
robustness checks, so read p = 0.04 with that in mind.

### (b) Lower-turnover portfolio: decile long-short rebalanced every 5 trading days
Weights are set on every 5th test day and held without drift. A held stock with no return that day
(e.g. it left the index) contributes 0. Costs are 10 bps per side on turnover.

**Full universe**

| model | gross bps (NW t, SR) | net 10 bps/side (NW t, SR) | daily turnover |
|---|---|---|---|
| M0 | 10.70 (2.37, 1.19) | 4.63 (1.03, 0.51) | 0.607 |
| M1 | 10.32 (2.32, 1.14) | 4.29 (0.97, 0.48) | 0.603 |
| M2 | 9.87 (2.32, 1.14) | 3.76 (0.88, 0.43) | 0.612 |
| M3 | 9.81 (2.42, 1.14) | 3.71 (0.92, 0.43) | 0.610 |
| PLACEBO | 9.10 (2.16, 1.04) | 2.97 (0.71, 0.34) | 0.613 |
| LGBM | 2.27 (0.68, 0.31) | -4.68 (-1.41, -0.64) | 0.695 |
| LSTM | 6.62 (1.82, 0.89) | 0.45 (0.12, 0.06) | 0.617 |

**Li-7**

| model | gross bps (NW t, SR) | net 10 bps/side (NW t, SR) | daily turnover |
|---|---|---|---|
| M0 | 2.62 (0.39, 0.16) | -3.20 (-0.48, -0.20) | 0.582 |
| M1 | 1.17 (0.17, 0.07) | -4.35 (-0.65, -0.28) | 0.552 |
| M2 | 10.90 (1.57, 0.69) | 5.43 (0.79, 0.35) | 0.547 |
| M3 | 10.84 (1.53, 0.69) | 5.35 (0.76, 0.34) | 0.548 |
| PLACEBO | 8.26 (1.10, 0.50) | 2.55 (0.34, 0.16) | 0.571 |
| LGBM | 11.14 (1.62, 0.74) | 4.64 (0.67, 0.31) | 0.650 |
| LSTM | -0.17 (-0.02, -0.01) | -5.90 (-0.86, -0.38) | 0.574 |

**Result:** holding for 5 days cuts turnover from about 2.1 to about 0.6 per day. Full-universe gross
spreads rise to about 9-11 bps/day and become significant (t = 2.2-2.4 for M0-M3 and the placebo; LGBM
and LSTM are not significant). Net of costs they are +3 to +5 bps/day but **not significant** (t
0.7-1.0, Sharpe 0.3-0.5). Sentiment models do not beat M0/M1 or the placebo here. On Li-7, M2/M3
are +5.4 bps/day net (t 0.79), which is not significant.

### LSTM training check
Each fold trained for 4-6 epochs with patience 3. The best validation epoch (1-indexed) was 2018: 2,
2019: 1, 2020: 1, 2021: 2, 2022: 3, with best validation loss 0.69308 / 0.69316 / 0.69309 / 0.69282 /
0.69279, barely below log 2 = 0.69315. Validation loss is best after 1-3 epochs and then rises while
training loss keeps falling, so the LSTM is **likely undertrained or signal-starved** and its results are
a lower bound for this architecture. It was not retuned (DECISIONS.md).

## Extension: SEC 8-K event features (M4). Added before 2023 was opened, after seeing the 2018-2022 results
Metadata only (form, acceptance time, item numbers), no filing text. `src/edgar.py`, `src/models_m4.py`.
- **Data:** tickers -> CIK via SEC `company_tickers.json` (609/610 mapped). Two overrides: KG -> King
  Pharmaceuticals (the KG ticker now belongs to Kestrel Group), and GENZ -> Genzyme (no current ticker).
  A coverage check (member years with zero 8-Ks) flagged 22 tickers. Two were the overrides above; for
  the other 20 the current CIK is a newer holding entity (GOOG shares Alphabet's CIK and gets the same fix). For each, a predecessor CIK was looked up by name in SEC `cik-lookup-data.txt` and verified to
  file 8-Ks in the gap years: XOM, BLK, APA, DIS, GOOG/GOOGL (Google Inc), SKYD (Paramount Global), NAVI,
  BKR, CI, DD, DINO, AVGO, DXC, ETN, FTI, ICE, MDT, PRGO, VTRS, XRX. Predecessor filings are used only
  before the successor's first 8-K. After the fix, **no member ticker-year (200+ days) has zero 8-Ks**; the
  only panel ticker with no 8-Ks is CCEP (a foreign filer that files 6-Ks). Forms 8-K and 8-K/A filed 2010+:
  103,830 on pre-2023 trading days for 595 tickers; 30,677 rows with trading day >= 2023 (or after the
  last 2022 trading day) are in the sealed file and never loaded.
- **Timing:** `acceptanceDateTime` turned out to be **UTC, not Eastern**. Apple's 4:30pm ET earnings 8-Ks
  show 20:30Z (EDT) and 21:30Z (EST). It is converted to America/New_York, then the news rule is applied:
  after 16:00:00 ET or on a non-trading day -> next trading day. Accepted times cluster at 16-17h and
  6-9h ET, as expected.
- **Features** per stock-day: has_8k, log(1+n_8k), and dummies for item 2.02, 1.01, 2.01, 5.02, 7.01/8.01,
  and any_other_item (any other item; 9.01 exhibits excluded because they come with most 8-Ks).
  log_n_8k is z-scored on the training fold but **not clipped**: the 99th percentile is log 2, so the
  1/99 clip used for other inputs would make it exactly log 2 x has_8k.

### 8-K coverage per year (panel stock-days; `coverage_8k_full.csv`, `coverage_8k_li7.csv`)
| year | Full 8-Ks | Full share stock-days w/ 8-K | Full share w/ item 2.02 | Full tickers w/ 8-K | Li-7 8-Ks | Li-7 share w/ 8-K | Li-7 tickers w/ 8-K |
|---|---|---|---|---|---|---|---|
| 2010 | 4791 | 0.0570 | 0.0170 | 347 | 39 | 0.0330 | 5 |
| 2011 | 5499 | 0.0600 | 0.0170 | 354 | 53 | 0.0400 | 5 |
| 2012 | 5619 | 0.0590 | 0.0180 | 365 | 54 | 0.0420 | 5 |
| 2013 | 5526 | 0.0560 | 0.0170 | 378 | 60 | 0.0450 | 6 |
| 2014 | 5728 | 0.0570 | 0.0170 | 386 | 67 | 0.0430 | 6 |
| 2015 | 6228 | 0.0600 | 0.0170 | 400 | 63 | 0.0390 | 6 |
| 2016 | 6337 | 0.0580 | 0.0170 | 422 | 63 | 0.0410 | 6 |
| 2017 | 6374 | 0.0570 | 0.0170 | 440 | 64 | 0.0420 | 6 |
| 2018 | 6346 | 0.0550 | 0.0170 | 450 | 44 | 0.0280 | 6 |
| 2019 | 6364 | 0.0530 | 0.0170 | 462 | 55 | 0.0360 | 6 |
| 2020 | 7140 | 0.0590 | 0.0170 | 470 | 70 | 0.0460 | 6 |
| 2021 | 6512 | 0.0530 | 0.0170 | 477 | 72 | 0.0400 | 7 |
| 2022 | 6258 | 0.0510 | 0.0170 | 481 | 74 | 0.0410 | 7 |

### M4 results (M4 = M3 + 8-K features; logit, same folds/preprocessing/seeds; LGBM_M4 = LightGBM on M4 inputs)
Pooled 2018-2022:

| group | model | accuracy | auc | logloss |
|---|---|---|---|---|
| Full | M3 | 0.5051 | 0.5093 | 0.6930 |
| Full | LGBM | 0.5028 | 0.5042 | 0.6944 |
| Full | M4 | 0.5048 | 0.5091 | 0.6931 |
| Full | LGBM_M4 | 0.5020 | 0.5037 | 0.6943 |
| Li-7 | M3 | 0.5202 | 0.5260 | 0.6924 |
| Li-7 | LGBM | 0.5163 | 0.5157 | 0.6936 |
| Li-7 | M4 | 0.5185 | 0.5260 | 0.6924 |
| Li-7 | LGBM_M4 | 0.5105 | 0.5126 | 0.6936 |

Paired daily log loss vs M3 (and LGBM_M4 vs LGBM), Newey-West 5 lags; gain > 0 means the 8-K model is
better:

| group | subset | test | days | gain | nw_t | p |
|---|---|---|---|---|---|---|
| Full | all days | M4 vs M3: + 8-K | 1,258 | -7.06e-06 | -1.7400 | 0.0810 |
| Full | all days | LGBM: + 8-K inputs | 1,258 | 3.32e-05 | 0.5600 | 0.5780 |
| Full | has_8k == 1 | M4 vs M3: + 8-K | 1,257 | -0.000171 | -1.9500 | 0.0510 |
| Full | has_8k == 1 | LGBM: + 8-K inputs | 1,257 | -8.16e-05 | -0.4400 | 0.6630 |
| Li-7 | all days | M4 vs M3: + 8-K | 1,258 | 1.32e-05 | 0.5000 | 0.6180 |
| Li-7 | all days | LGBM: + 8-K inputs | 1,258 | 9.51e-06 | 0.0400 | 0.9700 |
| Li-7 | has_8k == 1 | M4 vs M3: + 8-K | 248 | -0.000382 | -0.6100 | 0.5400 |
| Li-7 | has_8k == 1 | LGBM: + 8-K inputs | 248 | -0.00141 | -0.9500 | 0.3410 |

Portfolios (decile long-short, 10 bps/side; daily and 5-day rebalanced):

| group | rebalance | model | gross bps (t) | net bps (t) | net Sharpe | turnover |
|---|---|---|---|---|---|---|
| Full | daily | M3 | 5.31 (1.36) | -17.46 (-4.54) | -2.01 | 2.28 |
| Full | daily | LGBM | 3.37 (0.91) | -28.70 (-7.78) | -3.65 | 3.21 |
| Full | daily | M4 | 4.92 (1.25) | -18.33 (-4.72) | -2.10 | 2.32 |
| Full | daily | LGBM_M4 | 2.98 (0.81) | -29.23 (-8.05) | -3.81 | 3.22 |
| Full | 5-day | M3 | 9.81 (2.42) | 3.71 (0.92) | 0.43 | 0.61 |
| Full | 5-day | LGBM | 2.27 (0.68) | -4.68 (-1.41) | -0.64 | 0.69 |
| Full | 5-day | M4 | 9.12 (2.26) | 2.99 (0.74) | 0.35 | 0.61 |
| Full | 5-day | LGBM_M4 | 3.06 (0.91) | -3.88 (-1.15) | -0.52 | 0.69 |
| Li-7 | daily | M3 | 4.91 (0.71) | -14.62 (-2.16) | -0.93 | 1.95 |
| Li-7 | daily | LGBM | 10.74 (1.52) | -19.48 (-2.77) | -1.24 | 3.02 |
| Li-7 | daily | M4 | 8.55 (1.21) | -11.29 (-1.64) | -0.72 | 1.98 |
| Li-7 | daily | LGBM_M4 | 0.72 (0.10) | -30.18 (-4.12) | -1.91 | 3.09 |
| Li-7 | 5-day | M3 | 10.84 (1.53) | 5.35 (0.76) | 0.34 | 0.55 |
| Li-7 | 5-day | LGBM | 11.14 (1.62) | 4.64 (0.67) | 0.31 | 0.65 |
| Li-7 | 5-day | M4 | 11.64 (1.68) | 6.14 (0.89) | 0.40 | 0.55 |
| Li-7 | 5-day | LGBM_M4 | -0.47 (-0.07) | -7.10 (-1.02) | -0.48 | 0.66 |

8-K coefficients in M4 by fold (training fit, date-clustered SEs; `m4_8k_coefs.csv`):

| term | 2018 | 2019 | 2020 | 2021 | 2022 |
|---|---|---|---|---|---|
| any_other_item | -0.0104 (p 0.71) | 0.0197 (p 0.46) | 0.0202 (p 0.43) | 0.0274 (p 0.25) | 0.0259 (p 0.26) |
| has_8k | 0.0448 (p 0.53) | 0.0253 (p 0.71) | 0.0177 (p 0.78) | 0.0323 (p 0.60) | 0.0465 (p 0.43) |
| i101 | 0.0220 (p 0.53) | 0.0104 (p 0.75) | 0.0279 (p 0.37) | 0.0093 (p 0.75) | 0.0059 (p 0.83) |
| i201 | -0.0299 (p 0.72) | -0.0023 (p 0.98) | -0.0071 (p 0.92) | -0.0080 (p 0.91) | -0.0057 (p 0.93) |
| i202 | -0.0065 (p 0.83) | 0.0026 (p 0.93) | -0.0019 (p 0.94) | -0.0060 (p 0.81) | -0.0052 (p 0.83) |
| i502 | -0.0258 (p 0.39) | -0.0223 (p 0.43) | -0.0253 (p 0.34) | -0.0218 (p 0.39) | -0.0267 (p 0.27) |
| i701_801 | -0.0169 (p 0.49) | -0.0040 (p 0.86) | 0.0011 (p 0.96) | 0.0065 (p 0.75) | 0.0027 (p 0.89) |
| log_n_8k | -0.0129 (p 0.47) | -0.0104 (p 0.54) | -0.0076 (p 0.63) | -0.0115 (p 0.44) | -0.0129 (p 0.37) |

**Result: the 8-K features add nothing.** No 8-K coefficient is significant in any fold. Out of sample,
M4 is slightly *worse* than M3 on all days (Full t = -1.74, p = 0.081) and on 8-K days (t = -1.95,
p = 0.051), so the extra inputs add estimation noise. LightGBM with 8-K inputs is no better than plain
LightGBM. Portfolios do not improve on the Full universe. On Li-7 they are noisy (1 long / 1 short) and
not significant. M0-M3, PLACEBO, LGBM and LSTM results are unchanged: they were not refit, and every
pre-existing CSV value is identical.

## Robustness: relevance-filtered news (M2_rel / M3_rel). Added before 2023 was opened, motivated by the annotation audit
The blind LLM annotation (`results/hand_check_agreement.md`) found that 34 of the 94 FinBERT
disagreements were headlines about a *different* company than the tagged ticker. This variant keeps a
headline for its ticker only if it names the company or its symbol (`src/relevance.py`):
- **Names** come from universe.csv plus SEC entity names for the ticker's CIK(s): current, former names
  still in use in 2009+, and predecessor CIKs. Legal suffixes are removed (Inc/Corp/Co/Holdings/Ltd/plc/...)
  and names match case-insensitively on word boundaries. Two shorter variants are added: the name without
  a trailing descriptor ("Ford Motor" -> "Ford"), and a distinctive first word ("W W Grainger" ->
  "Grainger"). A single-word variant is dropped if it is generic or shared by 2+ companies ("American",
  "General", "Boston", "Marathon", ...). A small alias dict covers well-known short names (Alphabet/Google,
  Facebook, J&J, P&G, 3M, ...; full list in the code and DECISIONS.md).
- **Symbols** match whole-word and case-sensitively. Symbols of 1-2 letters only count as "$T", "(T)" or
  ":T", because bare "A", "T" and "ON" are ordinary words.
- Sentiment features are rebuilt from the relevant headlines with exactly the features.py steps.
  **Self-check:** the same rebuild applied to all headlines reproduces panel.parquet's sentiment columns
  with max difference 0. Then M2, M3 and the placebo are refit on the relevance-filtered features (same
  folds, regimes, preprocessing, seeds).
- **Main specification unchanged:** every value in the 32 pre-existing results CSVs is identical
  (max abs change 0), checked the same way as for M4.

### Share of headlines that survive (headlines on panel stock-days; `relevance_survival.csv`)
| year | headlines (panel days) | relevant | share relevant | Li-7 headlines | Li-7 share relevant |
|---|---|---|---|---|---|
| 2010 | 22,544 | 10,675 | 0.474 | 0 | - |
| 2011 | 42,072 | 20,425 | 0.485 | 224 | 0.554 |
| 2012 | 55,721 | 28,808 | 0.517 | 179 | 0.587 |
| 2013 | 55,828 | 23,701 | 0.425 | 144 | 0.507 |
| 2014 | 64,955 | 27,514 | 0.424 | 118 | 0.610 |
| 2015 | 92,285 | 39,293 | 0.426 | 180 | 0.561 |
| 2016 | 110,897 | 46,321 | 0.418 | 332 | 0.545 |
| 2017 | 116,932 | 47,671 | 0.408 | 617 | 0.536 |
| 2018 | 148,026 | 62,923 | 0.425 | 825 | 0.560 |
| 2019 | 158,876 | 70,972 | 0.447 | 2,825 | 0.383 |
| 2020 | 120,488 | 48,843 | 0.405 | 3,871 | 0.369 |
| 2021 | 61,896 | 27,167 | 0.439 | 2,168 | 0.236 |
| 2022 | 108,578 | 45,108 | 0.415 | 12,712 | 0.286 |

Overall **43.1%** of headlines (499,421 of 1,159,098) name their company. Stock-days with news
fall from 37.1% to 21.2%. A spot check of rejected headlines shows mostly market wraps ("US STOCKS-Wall
St..."), list articles ("10 Stocks That Could Rise 10x") and other companies ("Humana's (HUM) Medicare..."
tagged HCA). Known misses: brand/person-only mentions (e.g. "Elon Musk" for TSLA), "Delta" alone for DAL
(treated as generic), and "IPG" for IPGP.

### Results (pooled 2018-2022)
| group | model | accuracy | AUC | log loss |
|---|---|---|---|---|
| Full | M1 | 0.5056 | 0.5091 | 0.6931 |
| Full | M2 | 0.5056 | 0.5096 | 0.6930 |
| Full | M3 | 0.5051 | 0.5093 | 0.6930 |
| Full | PLACEBO | 0.5055 | 0.5091 | 0.6931 |
| Full | M2_rel | 0.5054 | 0.5092 | 0.6931 |
| Full | M3_rel | 0.5051 | 0.5089 | 0.6931 |
| Full | PLACEBO_rel | 0.5055 | 0.5090 | 0.6931 |
| Li-7 | M1 | 0.5154 | 0.5218 | 0.6925 |
| Li-7 | M2 | 0.5177 | 0.5255 | 0.6924 |
| Li-7 | M3 | 0.5202 | 0.5260 | 0.6924 |
| Li-7 | PLACEBO | 0.5179 | 0.5223 | 0.6925 |
| Li-7 | M2_rel | 0.5171 | 0.5265 | 0.6924 |
| Li-7 | M3_rel | 0.5181 | 0.5268 | 0.6924 |
| Li-7 | PLACEBO_rel | 0.5168 | 0.5239 | 0.6924 |

Paired daily log loss, Newey-West 5 lags (gain > 0: second model better):

| group | days used | test | days | gain | NW t | p |
|---|---|---|---|---|---|---|
| Full | all days | H1_rel: + relevant sentiment | 1258 | -1.99e-06 | -0.26 | 0.797 |
| Full | all days | H2_rel (OOS): + interactions | 1258 | -1.08e-05 | -1.65 | 0.099 |
| Full | all days | placebo_rel vs M1 | 1258 | -9.36e-06 | -1.46 | 0.143 |
| Full | all days | real vs shuffled relevant sentiment | 1258 | 7.38e-06 | 0.77 | 0.441 |
| Full | relevant-news days | H1_rel: + relevant sentiment | 1258 | -3.95e-06 | -0.12 | 0.904 |
| Full | relevant-news days | H2_rel (OOS): + interactions | 1258 | -1.08e-05 | -0.48 | 0.630 |
| Full | relevant-news days | placebo_rel vs M1 | 1258 | -2.49e-05 | -2.25 | 0.024 |
| Full | relevant-news days | real vs shuffled relevant sentiment | 1258 | 2.09e-05 | 0.62 | 0.535 |
| Li-7 | all days | H1_rel: + relevant sentiment | 1258 | 0.000114 | 1.70 | 0.088 |
| Li-7 | all days | H2_rel (OOS): + interactions | 1258 | 6.22e-06 | 0.26 | 0.798 |
| Li-7 | all days | placebo_rel vs M1 | 1258 | 8.56e-05 | 2.00 | 0.045 |
| Li-7 | all days | real vs shuffled relevant sentiment | 1258 | 2.89e-05 | 0.37 | 0.710 |
| Li-7 | relevant-news days | H1_rel: + relevant sentiment | 1035 | 0.000166 | 0.66 | 0.512 |
| Li-7 | relevant-news days | H2_rel (OOS): + interactions | 1035 | -0.000112 | -1.05 | 0.294 |
| Li-7 | relevant-news days | placebo_rel vs M1 | 1035 | 0.000111 | 1.23 | 0.219 |
| Li-7 | relevant-news days | real vs shuffled relevant sentiment | 1035 | 5.49e-05 | 0.20 | 0.840 |

M3_rel interaction sent_x_high by training fold: 2018: 0.0285 (p 0.044); 2019: 0.0252 (p 0.070); 2020: 0.0208 (p 0.124); 2021: 0.0216 (p 0.177); 2022: 0.0159 (p 0.298).

**Result: relevance filtering does not strengthen the sentiment effect; on the Full universe it removes
it.** H1_rel: t = -0.26 (p = 0.80) on all days, and t = -0.12 (p = 0.90) on relevant-news days, compared
with the main specification's t = 1.24 and t = 2.04 (p = 0.041). Real vs shuffled relevant sentiment is
not significant (p = 0.44 / 0.54). So the small news-day gain in the main specification comes mostly from
headlines that do *not* name the tagged company, likely sector, peer and market news that still carries
information for the stock. Note, though, that the relevance filter also cuts the news sample by more than
half. On Li-7, H1_rel is borderline (t = 1.70, p = 0.088), but the relevant-sentiment placebo also beats M1
(t = 2.00, p = 0.045), so that gain is not specific to real sentiment. The interaction is significant only
in the 2018 training fit.

## Diagnostic: rejected-headlines-only sentiment (M2_irr / M3_irr). Added before 2023 was opened, diagnostic for the relevance result
Sentiment built only from the headlines the relevance filter **rejected** (the complement of the _rel
set: 56.9% of headlines), with the same rebuild (`relevance.sentiment_block`), folds, regimes,
preprocessing, logit and seeds (`src/models_irr.py`). Gain = mean daily log loss of M1 minus the M2
variant (> 0: sentiment helps), Newey-West SE (5 lags), 95% CI = gain ± 1.96 SE. News-day share = share of
2018-2022 test stock-days with at least one headline of that kind. News days = that variant's own news
days. (`h1_variants_*.csv`)

**Full universe**

| news used | days | H1 gain (M2 variant vs M1) | NW SE | 95% CI | t | p | news-day share | n test days / stock-days |
|---|---|---|---|---|---|---|---|---|
| all headlines | all days | 1.65e-5 | 1.33e-5 | [-9.57e-6, 4.27e-5] | 1.24 | 0.214 | 0.402 | 1,258 / 574,824 |
| all headlines | news days | 6.40e-5 | 3.13e-5 | [2.59e-6, 1.25e-4] | 2.04 | 0.041 | 0.402 | 1,258 / 230,886 |
| relevant only | all days | -1.99e-6 | 7.73e-6 | [-1.71e-5, 1.32e-5] | -0.26 | 0.797 | 0.242 | 1,258 / 574,824 |
| relevant only | news days | -3.95e-6 | 3.26e-5 | [-6.79e-5, 6.00e-5] | -0.12 | 0.904 | 0.242 | 1,258 / 138,858 |
| rejected only | all days | 2.15e-5 | 1.44e-5 | [-6.61e-6, 4.97e-5] | 1.50 | 0.134 | 0.293 | 1,258 / 574,824 |
| rejected only | news days | 9.70e-5 | 4.83e-5 | [2.30e-6, 1.92e-4] | 2.01 | 0.045 | 0.293 | 1,258 / 168,319 |

**Li-7**

| news used | days | H1 gain (M2 variant vs M1) | NW SE | 95% CI | t | p | news-day share | n test days / stock-days |
|---|---|---|---|---|---|---|---|---|
| all headlines | all days | 1.07e-4 | 9.83e-5 | [-8.60e-5, 2.99e-4] | 1.08 | 0.278 | 0.314 | 1,258 / 8,058 |
| all headlines | news days | 1.23e-4 | 3.17e-4 | [-4.98e-4, 7.44e-4] | 0.39 | 0.697 | 0.314 | 1,208 / 2,534 |
| relevant only | all days | 1.14e-4 | 6.71e-5 | [-1.71e-5, 2.46e-4] | 1.70 | 0.088 | 0.253 | 1,258 / 8,058 |
| relevant only | news days | 1.66e-4 | 2.53e-4 | [-3.30e-4, 6.63e-4] | 0.66 | 0.512 | 0.253 | 1,035 / 2,035 |
| rejected only | all days | 9.61e-5 | 1.21e-4 | [-1.42e-4, 3.34e-4] | 0.79 | 0.428 | 0.299 | 1,258 / 8,058 |
| rejected only | news days | 3.74e-5 | 4.07e-4 | [-7.60e-4, 8.35e-4] | 0.09 | 0.927 | 0.299 | 1,163 / 2,406 |

Rejected-only placebo tests (`irr_paired_*.csv`, `irr_newsonly_paired_*.csv`). Full universe:

| days | test | gain | t | p |
|---|---|---|---|---|
| all days | H1_irr: + rejected-headline sentiment | 2.15e-5 | 1.50 | 0.134 |
| all days | placebo_irr vs M1 | -1.32e-5 | -1.91 | 0.057 |
| all days | real vs shuffled rejected sentiment | 3.47e-5 | 2.19 | 0.029 |
| rejected-news days | H1_irr: + rejected-headline sentiment | 9.70e-5 | 2.01 | 0.045 |
| rejected-news days | placebo_irr vs M1 | -5.62e-6 | -0.48 | 0.630 |
| rejected-news days | real vs shuffled rejected sentiment | 1.03e-4 | 2.04 | 0.042 |

**Interpretation (Full universe).** The evidence favours **spillover over low power**. On news days the
whole gain sits in the rejected headlines: the effect is 9.7e-5 (95% CI 2.3e-6 to 1.9e-4, p = 0.045), and
real rejected sentiment beats its shuffled placebo (p = 0.042). The relevant-only effect is essentially
zero (-4.0e-6), not a similar-sized effect with a wider interval: its CI (-6.8e-5 to 6.0e-5) is about as
wide as the all-headline CI (SE 3.3e-5 vs 3.1e-5) and sits just below the all-headline point estimate of
6.4e-5. Caveats: the relevant and rejected CIs overlap, and their difference was not formally tested. Li-7
is too small to say anything (all CIs are wide and include 0).

## Deviations from the plan, and caveats
1. **Universe = point-in-time S&P 500 membership** (Phase 1c, fja05680/sp500 snapshots), not current
   constituents. 751 companies were members in 2010-22. Of the 310 no longer in the index, 107 were
   recovered from yfinance and **203 are missing** (delisted names that yfinance no longer serves). Priced
   members per day range from 344 (2010) to 472 (2022), against about 500 in the index. **Survivorship
   bias is reduced, not eliminated**: missing names are mostly acquired or failed firms.
2. **Ticker aliasing**: renames (FB->META, ANTM->ELV, PCLN->BKNG, ...) are mapped, and recycled
   symbols that pointed to other companies are dropped (WTW = Weight Watchers before 2022, DOC, COR, GEN,
   ES, CZR, BNY). Company news is attached to both share classes (GOOG/GOOGL, FOX/FOXA, NWS/NWSA). The
   list is not exhaustive. See DECISIONS.md.
3. **Date-only headlines (64% of FNSPID) are moved to the next trading day.** This is conservative (no
   look-ahead), but a headline published before the close is then matched one day late, after most of the
   price reaction. That likely **biases sentiment effects toward zero** and is a limitation for the paper.
4. **FNSPID coverage gaps**: 2021 is thin, and Li-7 news is missing for most years for AMZN, META, MSFT
   and AAPL. Li-7 sentiment results rest on few stock-years.
5. Regimes use 2 states; feature = filtered P(high-vol). Market features are SPY 1/5/20-day returns, SPY
   20-day vol, and cross-sectional return dispersion.
6. H2 coefficient inference uses each fold's training fit. The out-of-sample H2 test is M3 vs M2 log loss.
7. LSTM: 1,506 test rows (0.26%) lack 20 prior rows and get no prediction. Paired comparisons use common
   rows.
8. M4 (8-K features) was added after the 2018-2022 results had been seen (see FROZEN.md), so it is
   exploratory for 2018-2022.
9. The relevance-filter variant (M2_rel/M3_rel) was added after the 2018-2022 results and the annotation
   audit had been seen; it is a robustness check, not a replacement for the main specification.
10. The rejected-only diagnostic (M2_irr/M3_irr) was added after the relevance result had been seen.
11. Hand check: `results/hand_check.csv` is handled in a separate session, and agreement / kappa are not
   reported here.
12. 2023 results: see "2023 holdout (confirmatory)" at the end (run once with FINAL=1, per FROZEN.md).

## 2023 holdout (confirmatory)
Run once on 2026-11-07 (~15:00-15:10) with FINAL=1, exactly as listed in `results/FROZEN.md`. Every
model trains on 2010-2022 (rows whose target date is before 2023-01-01) and is tested on the **249 trading
days of 2023**. Features, hyperparameters and seeds are unchanged. Outputs: `results/final_2023/` (2023
only) and `results/final_2023/pooled_2018_2023/` (all folds rerun). The frozen 2018-2022 CSVs in
`results/` are left as they were. Technical steps are logged in DECISIONS.md: BLDR's CIK was fetched
because it was missing from the EDGAR cache, and a wrapper (`src/eval_2023.py`) runs the unchanged
`evaluate.main` on the 2023 fold.

**Verdict key.** Confirmed = same sign as hypothesised and p < 0.05 in 2023. Reversed = significant in the
opposite direction (p < 0.05). Not confirmed = otherwise. Gain = mean daily log loss of base minus model
(> 0: the added component helps). Note that 2023 has 249 test days against 1,258 in 2018-2022, so power is
much lower.

### 2023 coverage
- **Full:** 154,733 headlines on panel stock-days; 40.3% of stock-days have news; 479 tickers/day
  (491 unique). 8-K: 6,375 filings, 5.1% of stock-days (item 2.02 1.7%), 487 tickers.
- **Li-7:** 29,455 headlines; **80.4%** of stock-days have news (vs 18-56% in 2018-22: FNSPID's 2023
  mega-cap coverage is far denser). 8-K: 70 filings, 4.0% of stock-days.
- Relevance filter: 45.3% of 2023 headlines (Full) and 31.9% (Li-7) name their company
  (`final_2023/pooled_2018_2023/relevance_survival.csv`).

### 2023 accuracy (`final_2023/metrics_*.csv`)
**Full**

| model | accuracy | AUC | log loss |
|---|---|---|---|
| M0 | 0.4989 | 0.4994 | 0.6932 |
| M1 | 0.4941 | 0.4915 | 0.6935 |
| M2 | 0.4932 | 0.4907 | 0.6936 |
| M3 | 0.4936 | 0.4907 | 0.6936 |
| PLACEBO | 0.4940 | 0.4913 | 0.6935 |
| LGBM | 0.5018 | 0.5005 | 0.6941 |
| LSTM | 0.4951 | 0.4942 | 0.6933 |
| M4 | 0.4938 | 0.4909 | 0.6936 |
| LGBM_M4 | 0.5012 | 0.5006 | 0.6941 |

**Li-7**

| model | accuracy | AUC | log loss |
|---|---|---|---|
| M0 | 0.5043 | 0.5334 | 0.6934 |
| M1 | 0.5055 | 0.5519 | 0.6930 |
| M2 | 0.4871 | 0.5389 | 0.6941 |
| M3 | 0.4882 | 0.5395 | 0.6940 |
| PLACEBO | 0.5077 | 0.5520 | 0.6930 |
| LGBM | 0.5003 | 0.5345 | 0.6950 |
| LSTM | 0.5198 | 0.5550 | 0.6930 |
| M4 | 0.4859 | 0.5393 | 0.6939 |
| LGBM_M4 | 0.5060 | 0.5319 | 0.6952 |

### Primary hypotheses
**Full universe**

| test | 2018-22 frozen: gain (t, p) | 2023 gain | 2023 t | 2023 p | days | verdict |
|---|---|---|---|---|---|---|
| H1: M2 vs M1 (+ sentiment), all days | 1.65e-5 (t 1.24, p 0.214) | -8.20e-5 | -2.53 | 0.012 | 249 | **Reversed** |
| H2 (OOS): M3 vs M2 (+ interactions) | -9.32e-6 (t -0.97, p 0.334) | 2.62e-6 | 0.45 | 0.651 | 249 | **Not confirmed** |

- **H2 in-sample** (M3 fit on 2010-2022): sent_x_high 0.0176 (se 0.0225, p 0.435); surprise_x_high -0.0115 (p 0.605). Sentiment effect low-vol 0.0122 (se 0.0116), high-vol 0.0298 (se 0.0198). **Verdict: not confirmed.**
- **H3** (2010-2022 training fit): p_high coefficient M1 0.0586 (p 0.0005), M2 0.0577 (p 0.0006), change -0.0009. The regime coefficient is significant and unchanged when sentiment is added. **Verdict: confirmed (in-sample).** Out of sample, though, adding the regime *hurt* in 2023:
| test | 2018-22 frozen: gain (t, p) | 2023 gain | 2023 t | 2023 p | days | verdict |
|---|---|---|---|---|---|---|
| M1 vs M0 (+ regime), all days | 1.22e-4 (t 2.26, p 0.024) | -3.19e-4 | -2.29 | 0.022 | 249 | **Reversed** |

**Li-7** (same fitted models; H2/H3 coefficients are the Full-universe fit)

| test | 2018-22 frozen: gain (t, p) | 2023 gain | 2023 t | 2023 p | days | verdict |
|---|---|---|---|---|---|---|
| H1: M2 vs M1, all days | 1.07e-4 (t 1.08, p 0.278) | -1.03e-3 | -2.27 | 0.023 | 249 | **Reversed** |
| H2 (OOS): M3 vs M2 | -7.83e-6 (t -0.38, p 0.701) | 3.20e-5 | 1.54 | 0.123 | 249 | **Not confirmed** |
| M1 vs M0 (+ regime) | -2.12e-5 (t -0.18, p 0.854) | 4.09e-4 | 0.94 | 0.350 | 249 | **Not confirmed** |

### News-days H1 and placebo
**Full**

| test | 2018-22 frozen: gain (t, p) | 2023 gain | 2023 t | 2023 p | days | verdict |
|---|---|---|---|---|---|---|
| H1 on news days (M2 vs M1) | 6.40e-5 (t 2.04, p 0.041) | -2.72e-4 | -1.80 | 0.072 | 249 | **Not confirmed (sign reversed)** |
| Real vs shuffled sentiment, news days | 7.68e-5 (t 2.35, p 0.019) | -2.49e-4 | -1.64 | 0.102 | 249 | **Not confirmed (sign reversed)** |
| Placebo vs M1, news days | -1.28e-5 (t -1.27, p 0.204) | -2.22e-5 | -1.70 | 0.089 | 249 | **Not confirmed** |
| Real vs shuffled sentiment, all days | 2.77e-5 (t 1.77, p 0.077) | -6.30e-5 | -1.89 | 0.059 | 249 | **Not confirmed (sign reversed)** |
| Placebo vs M1, all days | -1.11e-5 (t -1.41, p 0.157) | -1.90e-5 | -2.28 | 0.023 | 249 | **Placebo also hurts (p < 0.05)** |

**Li-7**

| test | 2018-22 frozen: gain (t, p) | 2023 gain | 2023 t | 2023 p | days | verdict |
|---|---|---|---|---|---|---|
| H1 on news days (M2 vs M1) | 1.23e-4 (t 0.39, p 0.697) | -1.34e-3 | -2.35 | 0.019 | 242 | **Reversed** |
| Real vs shuffled sentiment, news days | 9.64e-5 (t 0.29, p 0.773) | -1.40e-3 | -2.42 | 0.016 | 242 | **Reversed** |
| Placebo vs M1, news days | 2.70e-5 (t 0.22, p 0.828) | 6.66e-5 | 1.05 | 0.293 | 242 | **Not confirmed** |

### All / relevant-only / rejected-only (H1 effect sizes, 2023 vs 2018-22)
**Full**

| news used | days | 2018-22 gain [95% CI] | 2023 gain | 2023 95% CI | t | p | news-day share | n days / stock-days |
|---|---|---|---|---|---|---|---|---|
| all headlines | all days | 1.65e-5 [-9.57e-6, 4.27e-5] | -8.20e-5 | [-1.46e-4, -1.84e-5] | -2.53 | 0.012 | 0.403 | 249 / 119,271 |
| all headlines | news days | 6.40e-5 [2.59e-6, 1.25e-4] | -2.72e-4 | [-5.67e-4, 2.42e-5] | -1.80 | 0.072 | 0.403 | 249 / 48,010 |
| relevant only | all days | -1.99e-6 [-1.71e-5, 1.32e-5] | -1.76e-5 | [-5.17e-5, 1.65e-5] | -1.01 | 0.312 | 0.289 | 249 / 119,271 |
| relevant only | news days | -3.95e-6 [-6.79e-5, 6.00e-5] | -9.10e-5 | [-2.84e-4, 1.02e-4] | -0.92 | 0.356 | 0.289 | 249 / 34,470 |
| rejected only | all days | 2.15e-5 [-6.61e-6, 4.97e-5] | -8.41e-5 | [-1.44e-4, -2.44e-5] | -2.76 | 0.006 | 0.268 | 249 / 119,271 |
| rejected only | news days | 9.70e-5 [2.30e-6, 1.92e-4] | -3.46e-4 | [-6.78e-4, -1.42e-5] | -2.04 | 0.041 | 0.268 | 249 / 31,950 |

**Li-7**

| news used | days | 2018-22 gain [95% CI] | 2023 gain | 2023 95% CI | t | p | news-day share | n days / stock-days |
|---|---|---|---|---|---|---|---|---|
| all headlines | all days | 1.07e-4 [-8.60e-5, 2.99e-4] | -1.03e-3 | [-1.92e-3, -1.41e-4] | -2.27 | 0.023 | 0.804 | 249 / 1,743 |
| all headlines | news days | 1.23e-4 [-4.98e-4, 7.44e-4] | -1.34e-3 | [-2.45e-3, -2.20e-4] | -2.35 | 0.019 | 0.804 | 242 / 1,402 |
| relevant only | all days | 1.14e-4 [-1.71e-5, 2.46e-4] | -1.20e-3 | [-1.99e-3, -4.05e-4] | -2.96 | 0.003 | 0.785 | 249 / 1,743 |
| relevant only | news days | 1.66e-4 [-3.30e-4, 6.63e-4] | -1.58e-3 | [-2.57e-3, -5.89e-4] | -3.12 | 0.002 | 0.785 | 242 / 1,369 |
| rejected only | all days | 9.61e-5 [-1.42e-4, 3.34e-4] | -8.35e-4 | [-1.81e-3, 1.39e-4] | -1.68 | 0.093 | 0.804 | 249 / 1,743 |
| rejected only | news days | 3.74e-5 [-7.60e-4, 8.35e-4] | -1.10e-3 | [-2.32e-3, 1.32e-4] | -1.75 | 0.080 | 0.804 | 242 / 1,402 |

Rejected-only placebo tests, 2023 Full: on news days, real vs shuffled rejected sentiment -3.55e-4 (p 0.035). Relevant-only: real vs shuffled -4.94e-5 (p 0.614).

**Verdicts.**
- **Spillover pattern: reversed (Full).** In 2018-22 the news-day gain sat in rejected headlines (+9.7e-5, p 0.045). In 2023 rejected-only sentiment significantly *hurts*, on all days (p 0.006) and on news days (p 0.041), and it is worse than its shuffled placebo (p 0.035). Relevant-only stays not significant.
- **Li-7: reversed.** Relevant-only sentiment significantly hurts in 2023 (news days p 0.002; vs shuffled p 0.001). Li-7 news coverage is far denser in 2023 than in any earlier year.

### M4 (8-K features) and portfolios
| test | 2018-22 frozen: gain (t, p) | 2023 gain | 2023 t | 2023 p | days | verdict |
|---|---|---|---|---|---|---|
| Full: M4 vs M3, all days | -7.06e-6 (t -1.74, p 0.081) | 2.71e-6 | 0.34 | 0.733 | 249 | **Not confirmed** |
| Full: M4 vs M3, 8-K days | -1.71e-4 (t -1.95, p 0.051) | 1.36e-4 | 0.84 | 0.402 | 249 | **Not confirmed** |
| Full: LGBM_M4 vs LGBM | 3.32e-5 (t 0.56, p 0.578) | -3.15e-5 | -0.38 | 0.704 | 249 | **Not confirmed** |
| Li-7: M4 vs M3, all days | 1.32e-5 (t 0.50, p 0.618) | 7.74e-5 | 1.78 | 0.075 | 249 | **Not confirmed** |
| Li-7: M4 vs M3, 8-K days | -3.82e-4 (t -0.61, p 0.540) | -4.85e-5 | -0.04 | 0.964 | 62 | **Not confirmed** |

8-K coefficients in the 2010-2022 fit: none significant (all p > 0.27). **Verdict: M4 not confirmed**,
consistent with the 2018-22 null.

**Portfolios 2023, Full** (decile long-short; 10 bps/side)

| model | daily gross bps (t) | daily net bps (t) | 5-day gross bps (t) | 5-day net bps (t) | 5-day turnover |
|---|---|---|---|---|---|
| M0 | -3.95 (-0.57) | -21.37 (-3.13) | -11.22 (-1.64) | -17.06 (-2.51) | 0.58 |
| M1 | -4.46 (-0.60) | -21.58 (-2.95) | -12.89 (-1.74) | -18.50 (-2.51) | 0.56 |
| M2 | -8.65 (-1.30) | -29.13 (-4.49) | -12.54 (-1.86) | -18.10 (-2.70) | 0.56 |
| M3 | -8.44 (-1.30) | -28.86 (-4.57) | -11.76 (-1.80) | -17.33 (-2.66) | 0.56 |
| PLACEBO | -4.54 (-0.62) | -23.14 (-3.23) | -12.01 (-1.67) | -17.67 (-2.47) | 0.57 |
| LGBM | -3.30 (-0.84) | -35.26 (-9.12) | -7.69 (-1.91) | -14.49 (-3.59) | 0.68 |
| LSTM | 0.60 (0.11) | -17.33 (-3.10) | -10.08 (-2.04) | -15.82 (-3.21) | 0.57 |
| M4 | -7.76 (-1.21) | -28.59 (-4.57) | -12.51 (-1.88) | -18.12 (-2.73) | 0.56 |
| LGBM_M4 | -3.59 (-0.81) | -35.48 (-8.02) | -6.72 (-1.56) | -13.46 (-3.11) | 0.67 |

**Portfolios 2023, Li-7** (1 long / 1 short)

| model | daily gross bps (t) | daily net bps (t) | 5-day gross bps (t) | 5-day net bps (t) | 5-day turnover |
|---|---|---|---|---|---|
| M0 | 5.05 (0.30) | -13.02 (-0.80) | -14.95 (-0.78) | -20.57 (-1.09) | 0.56 |
| M1 | 0.11 (0.01) | -16.76 (-1.10) | -13.90 (-0.80) | -19.12 (-1.11) | 0.52 |
| M2 | 11.31 (0.82) | -12.38 (-0.92) | -25.58 (-1.62) | -30.96 (-1.97) | 0.54 |
| M3 | 7.76 (0.57) | -16.01 (-1.21) | -30.44 (-1.99) | -35.82 (-2.35) | 0.54 |
| PLACEBO | 0.61 (0.04) | -16.58 (-1.07) | -2.97 (-0.17) | -8.12 (-0.46) | 0.51 |
| LGBM | 24.59 (1.53) | -4.40 (-0.28) | 7.67 (0.61) | 1.16 (0.09) | 0.65 |
| LSTM | 7.45 (0.43) | -12.07 (-0.72) | -15.98 (-0.99) | -21.36 (-1.34) | 0.54 |
| M4 | 7.80 (0.56) | -16.13 (-1.20) | -28.65 (-1.82) | -34.11 (-2.18) | 0.55 |
| LGBM_M4 | 16.48 (0.93) | -13.32 (-0.76) | 21.87 (1.77) | 15.37 (1.25) | 0.65 |

**Verdict: not confirmed.** Every daily strategy loses after costs on the Full universe (-17 to -35
bps/day, t -2.95 to -9.1). Daily gross spreads are near zero or negative and not significant (-8.7 to +0.6 bps). The 5-day gross spreads,
positive and significant in 2018-22 (about +10 bps, t about 2.3), are negative in 2023 (-6.7 to -12.9
bps, t -1.56 to -2.04; significant only for the LSTM at t -2.04 and borderline for LightGBM at t -1.91).
On Li-7 everything is not significant except M3 5-day (gross -30.4 bps, t -1.99, borderline).

### Bottom line
In the confirmatory 2023 holdout, **no sentiment hypothesis is confirmed**. Adding sentiment to the
regime model (H1) makes out-of-sample predictions significantly *worse* on the Full universe (t -2.53)
and on Li-7 (t -2.27). On the Full universe a shuffled-sentiment placebo with the same columns also hurts
(t -2.28), so much of the loss is generic overfitting from adding ~15 inputs when the year differs from
training. Real sentiment is no better than shuffled (all days p 0.059; news days p 0.10). On Li-7, real
sentiment is significantly worse than shuffled (p 0.020). The interaction effect (H2) is not confirmed.
The regime coefficient is stable and significant in the training fit (H3 confirmed in-sample), but the
regime also hurts out of sample in 2023 (t -2.29). Pooled over 2018-2023 (rerun), H1 is t 0.01 and real
vs shuffled is t 0.89.

### Rerun drift
The FINAL rerun re-downloaded adjusted prices and added a target for 2022-12-30. Its 2018-2022 fold metrics
differ from the frozen ones by at most 0.0043 AUC, 0.00045 log loss and 0.0031 accuracy. The frozen
2018-2022 numbers above remain the reference.
