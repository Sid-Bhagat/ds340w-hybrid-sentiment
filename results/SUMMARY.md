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
stock-days have news. Li-7 coverage is very uneven. In raw FNSPID, AMZN has news only in 2020, META only
in 2020 (filed as FB), MSFT only in 2022, AAPL only in 2020 and 2022, and TSLA in 2019, 2020 and 2022.
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

**H1 is not supported out of sample** (Full p = 0.21; Li-7 p = 0.28). Real sentiment beats the
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
**No strategy survives 10 bps costs.** The Li-7 portfolio is 1 long / 1 short (a decile of about 7
stocks), so it is very noisy. Figure: `figures/cumulative_long_short.png`.

### Regimes
2-state PCA-HMM, forward-filtered, refit each fold. Mean out-of-sample P(high-vol) by test year: 2018 0.45,
2019 0.30, 2020 0.67, 2021 0.24, 2022 0.87. Both states are persistent (P(stay) 0.95-0.98).
Figure: `figures/regime_timeline.png`.

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
8. Hand check: `results/hand_check.csv` is handled in a separate session, and agreement / kappa are not
   reported here.
9. 2023 results are not run. After "open 2023", rerun everything once with FINAL=1 and append a section
   here.
