# Li (2025) vs this replication

Paper: Li, S. (2025), PLOS ONE 20(9): e0331658, doi:10.1371/journal.pone.0331658.
Page numbers are the paper's own (x / 30).

Our numbers come from the fresh yfinance run in `results/`, downloaded 2026-11-08, 2015-01-01 to 2024-12-31, weekly, N = 518 per stock.
Significance is judged at 5%. Li marks 10/5/1% as \*/\*\*/\*\*\*, and the detail tables below use the same marks.
Verdicts are **match**, **same sign, not significant** (Li significant, ours not), or **mismatch**.

## Findings

| # | Li finding | Li's reported result (source) | Our replication | Verdict |
|---|---|---|---|---|
| 1 | PCA variance explained | 5 principal components explain more than 75% of the macro inputs (§4.2, p. 10; contribution (3), p. 3). PCA uses 12 series (Table 1, p. 10). | The first 5 PCs explain 92.6% and the first 4 explain 84.3% (`pca_explained_variance.csv`). Our 75% retention rule keeps **4** PCs for the HMM. Our PCA uses 7 series. | **match** (more than 75% with 5 PCs). We feed 4 PCs to the HMM where Li feeds 5. |
| 2 | Number of regimes | 4 states. AIC/BIC "not significantly varying"; 4 is where "most models depict the lowest AIC and BIC" (§4.2, pp. 10–11). | 4 states, as fixed in the config. AIC is lowest at 5–6 states for every stock. BIC is lowest at 4 only for NVDA (5 for AMZN, GOOGL, META; 6 for AAPL, MSFT, TSLA) (`hmm_state_selection.csv`). | **mismatch**. AIC/BIC do not point to 4. |
| 3 | Market beta (S&P 500 return) | Significant at 1% for all 7 stocks in Eq. 16 and in Eq. 17 on all weeks. Eq. 17 betas 1.06–1.90 (Table 4, pp. 23–24). | Significant at 1% for all 7 stocks in both. Eq. 17 all-weeks betas are within 0.02 of Li's for every stock (detail A). | **match** |
| 4 | Momentum | "Nearly all large tech companies show no momentum effect at both stock level and individual state level" (§4.3.1, p. 14). Table 4: no stock-level momentum coefficient is significant. | Not significant for 6 of 7 stocks in Eq. 16 and in Eq. 17 all weeks. **MSFT is significant at 1%** (5.11 in Eq. 16, 5.95 in Eq. 17 all; Li has 1.91 and 1.77, both n.s.). Per state, 3 of 27 Eq. 17 regressions are significant at 5% (Li: 1 of 27). | **match** for 6 stocks; **mismatch** for MSFT |
| 5 | Volume (Eq. 17, all weeks) | Significant at 1% for GOOGL, META (negative), MSFT, TSLA; at 5% for NVDA; n.s. for AMZN, AAPL (§4.3.2, p. 20; Table 4). | GOOGL\*\*\*, META\*\* (negative), MSFT\*\*\*, TSLA\*\*\*, NVDA\*\*; AMZN\* (10% only), AAPL n.s. Every sign agrees (detail A). | **match** |
| 6 | Volume effect absorbed by regimes | GOOGL, META and NVDA lose their volume significance once the state dummies are added (Eq. 16); MSFT and TSLA keep it (§4.3.3, p. 21; Table 4). | GOOGL and NVDA lose it; MSFT\*\* and TSLA\*\* keep it. **META keeps it** (−1.39e-10, p < 0.001). AMZN: Li\*\*\*, ours\* (detail B). | **match** for GOOGL, NVDA, MSFT, TSLA; **mismatch** for META, **same sign, not significant** for AMZN |
| 7 | Regime dummies (investor-sentiment effect) | Most states significant, "universally under 1%", for all stocks **except AAPL and TSLA** (§4.3.3, p. 20; Table 4). | Joint F-test of the Eq. 16 dummies: AAPL n.s. (p = 0.66); AMZN, GOOGL, MSFT, NVDA p < 1e-6; META p = 0.017; **TSLA p = 8e-18** (detail B). **This depends on the data vintage.** On the September cache, AAPL is significant (p = 1e-46) and TSLA is not (p = 0.18). | **match** for 6 stocks, but which stock mismatches flips between vintages: TSLA (fresh), AAPL (September) |
| 8a | Per-regime effects: beta | Beta significant at 1% in almost every state with more than ~10 weeks; the tiny states (1–9 weeks) are n.s. or undefined (Table 4). | Significant at 5% in 24 of 27 state regressions. The exceptions are AMZN state 0 (36 weeks), NVDA state 3 (157 weeks) and TSLA state 0 (71 weeks) (detail C). | **match** (with 3 exceptions, all in larger states than Li's n.s. ones) |
| 8b | Per-regime effects: volume | Volume matters in several states for every stock except AAPL (§4.3.2, p. 20). Table 4 shows 5%-significant, positive volume coefficients in AMZN (2 states), META, NVDA and TSLA (2 states); none in AAPL, GOOGL or MSFT. | 5%-significant only in GOOGL, META and TSLA (one state each), and **all negative**; none in AAPL, AMZN, MSFT or NVDA (detail C). | **mismatch** (AAPL with no effect matches) |
| 8c | Per-regime effects: state sizes | Very uneven, with tiny states: AAPL 6 and 11 weeks, GOOGL 5, META 7, TSLA 1, 9 and 17 (Table 4). | Also uneven: AAPL 4 weeks (Eq. 17 skipped), MSFT 10, META 20, TSLA 22, NVDA 28 (`hmm_states.csv`). | **match** (qualitatively) |

Not replicated (outside the September code): Table 5 (sub-sector stocks), the risk-aversion parameter γ in Tables 4–5, Table 3 (correlations), and Fig 4 (PC box plots).

### Detail A: Eq. 17 on all weeks, no state dummies (Li Table 4, second row per stock)

| Stock | β Li | β ours | Momentum Li | Momentum ours | Volume Li | Volume ours | adj. R² Li / ours |
|---|---|---|---|---|---|---|---|
| AMZN | 1.2327\*\*\* | 1.2490\*\*\* | 0.4013 | −1.3536 | 9.259e-12 | 1.680e-11\* | 0.397 / 0.393 |
| AAPL | 1.0856\*\*\* | 1.0878\*\*\* | 0.8070 | −2.5572 | 9.541e-13 | 6.860e-13 | 0.462 / 0.464 |
| GOOGL | 1.1565\*\*\* | 1.1541\*\*\* | 1.0433 | 2.0154 | 4.775e-11\*\*\* | 4.920e-11\*\*\* | 0.479 / 0.480 |
| META | 1.2881\*\*\* | 1.2807\*\*\* | 0.6951 | 0.7937 | −8.105e-11\*\*\* | −7.784e-11\*\* | 0.333 / 0.332 |
| MSFT | 1.0609\*\*\* | 1.0614\*\*\* | 1.7722 | 5.9469\*\*\* | 6.484e-11\*\*\* | 7.624e-11\*\*\* | 0.553 / 0.561 |
| NVDA | 1.8787\*\*\* | 1.8817\*\*\* | 0.7547 | −1.2777 | 4.799e-12\*\* | 4.286e-12\*\* | 0.409 / 0.411 |
| TSLA | 1.9008\*\*\* | 1.9007\*\*\* | −0.0417 | −0.1358 | 5.391e-11\*\*\* | 5.300e-11\*\*\* | 0.307 / 0.307 |

### Detail B: Eq. 16 (with state dummies)

Li reports a constant plus all four state dummies. We omit state 0 (the lowest-return state), so we compare the dummies as a block, using a joint F-test (`eq16_state_dummies_ftest.csv`).

| Stock | β Li / ours | Momentum Li / ours | Volume Li / ours | Li: dummies sig. at 5% | Ours: joint F p-value | adj. R² Li / ours |
|---|---|---|---|---|---|---|
| AMZN | 1.1607\*\*\* / 0.9572\*\*\* | −0.2381 / −1.4975 | 4.393e-11\*\*\* / 1.996e-11\* | 3 of 4 | 7.6e-12 | 0.431 / 0.452 |
| AAPL | 1.0817\*\*\* / 1.0808\*\*\* | 0.5415 / −2.2129 | −5.307e-13 / 1.176e-12 | 0 of 4 | 0.66 | 0.461 / 0.462 |
| GOOGL | 1.1700\*\*\* / 1.0805\*\*\* | 0.8977 / 2.2151 | 1.373e-12 / 2.310e-11 | 3 of 4 | 1.6e-7 | 0.485 / 0.511 |
| META | 1.3197\*\*\* / 1.2892\*\*\* | 0.9048 / 0.8839 | 1.931e-11 / −1.392e-10\*\*\* | 4 of 4 | 0.017 | 0.382 / 0.342 |
| MSFT | 1.0629\*\*\* / 0.9126\*\*\* | 1.9075 / 5.1137\*\*\* | 5.847e-11\*\*\* / 5.859e-11\*\* | 1 of 4 | 2.7e-9 | 0.553 / 0.594 |
| NVDA | 1.9372\*\*\* / 1.2958\*\*\* | 1.4888 / −0.5992 | 3.037e-12 / −1.593e-12 | 3 of 4 | 3.5e-20 | 0.417 / 0.507 |
| TSLA | 1.8884\*\*\* / 1.1947\*\*\* | −0.1905 / −0.9655\* | 5.068e-11\*\*\* / 3.000e-11\*\* | 0 of 4 | 8.2e-18 | 0.306 / 0.407 |

### Detail C: Eq. 17 within states (5%-significant terms; weeks per state in brackets)

State labels are arbitrary, so states can't be matched one-to-one between Li and us. Ours are ordered by mean return (0 = lowest).

| Stock | Li (Table 4, states 1–4) | Ours (states 0–3) |
|---|---|---|
| AMZN | [15, 219, 72, 212]; β in 2, 3, 4; volume + in 2, 4 | [36, 177, 211, 94]; β in 1, 2, 3; momentum + in 1 |
| AAPL | [247, 6, 11, 254]; β in 1, 3, 4 | [4 (skipped), 35, 213, 266]; β in 1, 2, 3 |
| GOOGL | [65, 246, 202, 5]; β in 1, 2, 3 | [35, 157, 274, 52]; β in all; volume − in 3 |
| META | [147, 7, 273, 91]; β in 1, 3, 4; momentum + in 3; volume + in 4 | [40, 222, 236, 20]; β in all; momentum + in 1; volume − in 0 |
| MSFT | [147, 109, 14, 248]; β in all | [61, 288, 10, 159]; β in all |
| NVDA | [130, 22, 174, 192]; β in all; volume + in 3 | [80, 28, 253, 157]; β in 0, 1, 2 |
| TSLA | [1 (undefined), 9, 17, 491]; β in 3, 4; volume + in 2, 4 | [71, 22, 308, 117]; β in 1, 2, 3; momentum − in 2; volume − in 0 |

### Data check against Li's Table 2 (p. 21)

`descriptive_stats.csv` reproduces Li's Table 2 closely: N = 518 for every series.
Mean weekly returns are within 2% of Li's (S&P 500: 0.00233 vs 0.00229; NVDA: 0.01310 vs 0.01301). Return standard deviations, and the momentum and volume statistics, are within about 2%.
The one visible gap is AMZN: our maximum weekly return is 0.200 vs Li's 0.153, and our mean is 0.00612 vs Li's 0.00574 (7%).
So our stock-level inputs are essentially Li's.

## Reproducibility

Scripts: `check_reproducibility.sh` and `compare.py`. Outputs are in `results/reproducibility/`.

### Step 3: the September cache (`old/data`) vs the September outputs (`old/outputs`)

- **Ported code, September labelling:** 266 of 266 values match (Eq. 16 and Eq. 17 coefficients and p-values, and the AIC/BIC tables), with a maximum relative difference of 1.1e-11. The HMM state labels are identical.
  13 of 14 figures are pixel-identical. In the GOOGL price trend, 8 pixels at the corners of the axes frame differ, while the plotted data is the same.
  The September run **was seeded**: the old config already passed `random_state=42` to the HMM. So no difference comes from a missing seed, and nothing had to be forced to match.
- **Final code** (states ordered by mean weekly return; BLAS pinned to one thread):
  - Every ticker's states map one-to-one onto the September states.
  - Eq. 17 matches to 2e-13 and AIC/BIC to 1e-14.
  - Eq. 16 matches to 1.3e-4 relative. That largest gap is the volume coefficient: the omitted base state changed, and raw volume (~1e8) makes the design matrix badly conditioned.
  - No coefficient changes sign or 5% significance.
  - The figures differ only because the state colours follow the new order.
- **Two runs of the final code** give byte-identical output (24 of 24 files; `run_info.json` excluded).
  With multithreaded BLAS, the HMM log-likelihoods and transition durations differed in the last bits between runs (at most 5.5e-13 relative). Pinning BLAS to one thread removed that.

### Step 4: fresh yfinance download (2026-11-08) vs September

- **Data:** the same 14 series, 522 weekly rows each, on the same dates.
  Yahoo has **revised gold futures (GC=F)**: 228 weeks from 2020-05-28 onward differ by up to 1.9%.
  The stock series differ from September by at most 1.4e-6 in weekly returns. The other macro series are identical.
- **PCA:** 84.32% → 84.29% for 4 PCs.
- **HMM states:** the share of weeks that stay in the matched state is 99.6% for AMZN, 99.8% for MSFT and 98.8% for NVDA, but only 74% for AAPL, 72% for GOOGL, 71% for META and 54% for TSLA.
- **Cause, verified:** we reran the fresh data with only the September GC=F file swapped in. That reproduces the September states for 100% of weeks, with 0 sign or significance changes. The gold revision is the only cause.
- **Changes:** 28 of 169 comparable coefficients changed sign or 5% significance (1 in Eq. 16, 27 in Eq. 17, 0 in Eq. 17 on all weeks).
  Where the Eq. 16 base state maps to a different September state (AAPL, GOOGL, META, TSLA), the Eq. 16 constant and dummies are not comparable and are left out of the table.
  The joint F-test of the dummies, which doesn't depend on the base state, **does change**. AAPL goes from p = 1e-46 (September) to p = 0.66 (fresh), and TSLA from p = 0.18 to p = 8e-18. The other five stocks stay significant at 5% (`eq16_state_dummies_ftest.csv` in `results/` and in `results/reproducibility/september_cache_run/`).
  "Sept." states below are this code's mean-return labels on the September cache; each maps one-to-one onto the September labels.

| Stock | Table | State (fresh → Sept.) | Term | Sept. coef (p) | Fresh coef (p) | Change |
|---|---|---|---|---|---|---|
| AAPL | Eq. 16 | all | volume | -1.53e-12 (0.673) | 1.18e-12 (0.803) | sign |
| AAPL | Eq. 17 | 2 → 0 | const | -0.0317 (1.17e-09) | -0.00299 (0.601) | 5% sig. |
| GOOGL | Eq. 17 | 0 → 1 | const | 0.0385 (0.0781) | -0.0463 (0.0774) | sign |
| GOOGL | Eq. 17 | 0 → 1 | sp500_return | 0.215 (0.362) | 1.08 (1.75e-06) | 5% sig. |
| GOOGL | Eq. 17 | 0 → 1 | volume | -1.5e-10 (0.118) | 1.61e-10 (0.0598) | sign |
| GOOGL | Eq. 17 | 1 → 0 | const | -0.0254 (0.00012) | -0.00587 (0.451) | 5% sig. |
| GOOGL | Eq. 17 | 1 → 0 | volume | -5.98e-11 (0.0258) | 7.11e-12 (0.873) | sign, 5% sig. |
| GOOGL | Eq. 17 | 3 → 3 | const | 0.014 (0.354) | 0.0808 (0.00431) | 5% sig. |
| GOOGL | Eq. 17 | 3 → 3 | volume | 1.12e-10 (0.044) | -2.47e-10 (0.0179) | sign |
| META | Eq. 17 | 0 → 1 | const | 0.0815 (0.0283) | 0.0639 (0.0944) | 5% sig. |
| META | Eq. 17 | 0 → 1 | momentum | 0.771 (0.841) | -0.0384 (0.991) | sign |
| META | Eq. 17 | 1 → 2 | const | -0.00274 (0.494) | 0.0023 (0.605) | sign |
| META | Eq. 17 | 1 → 2 | volume | 9.81e-11 (0.0387) | 1.67e-11 (0.773) | 5% sig. |
| META | Eq. 17 | 2 → 3 | const | 0.0353 (0.00114) | -0.00158 (0.85) | sign, 5% sig. |
| META | Eq. 17 | 2 → 3 | sp500_return | 0.317 (0.108) | 1.17 (1.16e-20) | 5% sig. |
| META | Eq. 17 | 3 → 0 | const | -0.0239 (0.014) | -0.00142 (0.978) | 5% sig. |
| META | Eq. 17 | 3 → 0 | sp500_return | 0.251 (0.0544) | 1.01 (0.0388) | 5% sig. |
| META | Eq. 17 | 3 → 0 | volume | -9.57e-11 (0.183) | 1.44e-10 (0.683) | sign |
| NVDA | Eq. 17 | 0 → 0 | volume | -3.44e-14 (0.995) | 1.14e-13 (0.984) | sign |
| NVDA | Eq. 17 | 3 → 3 | volume | -1.16e-11 (0.0263) | -1.02e-11 (0.0541) | 5% sig. |
| TSLA | Eq. 17 | 0 → 2 | const | 0.00459 (0.624) | -0.0391 (0.0616) | sign |
| TSLA | Eq. 17 | 0 → 2 | sp500_return | 0.851 (6.67e-05) | 0.25 (0.341) | 5% sig. |
| TSLA | Eq. 17 | 0 → 2 | momentum | 2.24 (0.338) | -1.03 (0.652) | sign |
| TSLA | Eq. 17 | 0 → 2 | volume | 1.19e-11 (0.695) | -8.81e-11 (0.012) | sign, 5% sig. |
| TSLA | Eq. 17 | 2 → 1 | const | -0.000505 (0.969) | 0.0073 (0.351) | sign |
| TSLA | Eq. 17 | 2 → 1 | momentum | -0.673 (0.656) | -3.11 (0.039) | 5% sig. |
| TSLA | Eq. 17 | 3 → 3 | const | -0.0383 (0.152) | 0.0146 (0.567) | sign |
| TSLA | Eq. 17 | 3 → 3 | volume | 7.21e-11 (0.0157) | 4.42e-11 (0.0879) | 5% sig. |

All Li comparisons above use the fresh run. On the September cache, rows 3–6 get the same verdicts: every beta, momentum and volume coefficient has the same sign and 5% significance in Eq. 16 and in Eq. 17 on all weeks.
Row 7 changes as described above. Row 8 is the most fragile part: even between our own two data vintages, the per-state results change for four of the seven stocks.

## Why results differ from Li

1. **The results that don't depend on the HMM match; the HMM-based ones don't.** Eq. 17 on all weeks uses no HMM output, and it reproduces Li's betas to within 0.02, Li's volume signs and significance, and Li's adjusted R² to within 0.01. The stock data also matches Li's Table 2. So the gaps come from the regimes, not from the data or the OLS.
2. **Different PCA inputs.** Li's Table 1 lists 12 series: S&P 500, Nasdaq, 13-week, 5-, 10- and 30-year Treasuries, crude oil, gold, natural gas, and EUR/USD, JPY/USD and AUD/USD.
   The replication uses 7: S&P 500, Nasdaq, 13-week and 10-year rates, gold, crude oil, and the US Dollar Index. It also feeds 4 PCs to the HMM (from the 75% rule) where Li uses 5.
   Different PCs give different regimes.
3. **The HMM fit is fragile.** EM runs from a single seeded start and finds a local optimum. A 2% revision to one of seven PCA inputs moved 26–46% of the weeks to a different state for four stocks. It also swapped which stock's regime dummies are significant (AAPL vs TSLA, row 7).
   Li gives neither the seed nor the hmmlearn version. Li also doesn't specify the covariance type, or whether the HMM inputs are standardized (we standardize and use diagonal covariances).
   So the exact regimes can't be recovered from the paper. State labels are arbitrary, which means per-state results can only be compared as patterns.
4. **Specification details the paper leaves open:**
   - Eq. 16 as printed has a constant and all four state indicators, which are perfectly collinear. How Li identified it is unclear; we omit one state.
   - The momentum timing isn't stated. Ours uses the 4-week window ending in week t, the same week as the dependent return. A one-week lag would leave the Table 2 summary statistics almost unchanged but move the coefficients. That could explain why our momentum coefficients differ from Li's (MSFT especially) even though beta and volume match. This is a hypothesis we haven't tested.
   - The AIC/BIC parameter count isn't given.
5. **The regimes are partly built from the dependent variable.** The HMM sees each stock's own weekly return, so state dummies will "explain" returns to some degree by construction. Ours are ordered by mean return, which makes that visible. This applies to Li's design as well. It means the regime-dummy finding (row 7) is weaker evidence of a sentiment effect than it looks.
6. **Data vintage.** Yahoo revises adjusted prices and continuous futures contracts, and Li's download date is unknown. Our own September and November downloads already differ for gold.
