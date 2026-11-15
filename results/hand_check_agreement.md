# FinBERT validation: blind LLM annotation (annotator A)

## Method
- **Annotator:** Claude Opus 5.5 (model ID `claude-opus-5-5`), running in Claude Code. Date: 2026-11-07.
- **Sample:** the 200 headlines in `results/hand_check_frozen.csv`, a frozen copy of `results/hand_check.csv`. The sample was stratified by FinBERT band (67 neg / 67 neu / 66 pos), so these numbers are not population rates.
- **Blinding:** I labeled every headline by reading it, working from the frozen file, which holds only id, ticker, date, and headline. I did not open `hand_check_key.csv`, the scored or filtered parquet files, or any FinBERT output until all 200 labels were saved. No model, script, or keyword rule assigned the labels.
- **Label definitions:** sentiment toward the *named ticker*, from an investor's point of view.
  - positive: likely good news for the stock (beats, upgrades, approvals, buybacks, strong guidance, wins).
  - negative: likely bad news (misses, downgrades, lawsuits, investigations, rejections, weak guidance, losses).
  - neutral: factual or routine, mixed, or not really about the stock.
- **FinBERT label:** score = P(pos) − P(neg). Negative if < −1/3, positive if > 1/3, neutral otherwise (from `src/hand_check.py`).
- **Correction disclosed:** after unblinding, I found a transcription offset in batch 2. Id 31 had been skipped, so ids 32–50 carried the labels meant for ids 33–51. The blind labels were moved back to their correct ids. Id 31 (FSLR, "The Solar Energy Industry's New Reality") had no blind label and was labeled neutral after unblinding. I never looked at its FinBERT score. Excluding it changes nothing material.

## Agreement (n = 200)
- **Raw agreement:** 0.530
- **Cohen's kappa:** 0.294 (fair)

Command: `python src/hand_check.py --agree --input results/hand_check_llm_A.csv --col llm_label`

## Confusion matrix (rows = LLM, columns = FinBERT)

| LLM \ FinBERT | negative | neutral | positive |
|---|---|---|---|
| **negative** | 27 | 7 | 0 |
| **neutral**  | 33 | 50 | 37 |
| **positive** | 7 | 10 | 29 |

## FinBERT per-class precision / recall (LLM labels as reference)

| class | precision | recall | LLM support |
|---|---|---|---|
| negative | 0.403 | 0.794 | 34 |
| neutral  | 0.746 | 0.417 | 120 |
| positive | 0.439 | 0.630 | 46 |

**Pattern:** most disagreements (70 of 94) are headlines I labeled neutral that FinBERT gave a polarity. In 34 of those, the headline is about a *different* company than the ticker it is tagged to, for example a peer's earnings beat or loss. FinBERT scores the headline's tone, not its relevance to the ticker. Only 7 disagreements have opposite polarity, all LLM-positive vs FinBERT-negative, and they come from mixed results or negative words used in a good-for-the-ticker way. FinBERT never called positive a headline I called negative.

## Five most informative disagreements

| id | ticker | headline | LLM | FinBERT (score) | why |
|---|---|---|---|---|---|
| 93 | WMT | Walmart's Private-Label Electronics Threat to Roku, Amazon, and Others Is Getting Very Real | positive | negative (−0.845) | "Threat" is good for Walmart, the named stock. FinBERT reads the word, not who is being threatened. |
| 104 | BAX | Baxter (BAX) Beats Q1 Earnings Estimates, Sales Down Y/Y | positive | negative (−0.939) | Mixed headline. FinBERT weights "down" heavily, while the EPS beat is the usual market driver. |
| 9 | BA | Aerospace supplier Teledyne will continue to cut jobs | neutral | negative (−0.963) | The ticker link is wrong: the headline is about Teledyne, not Boeing. |
| 15 | COP | Hess (HES) Surpasses Earnings and Revenue Estimates in Q2 | neutral | positive (+0.928) | Same issue in the positive direction: a peer's beat gets attributed to COP. |
| 197 | DECK | 'Ugly Shoes' Still Hot On Social Media | positive | negative (−0.622) | Domain slang. "Ugly shoes" is Deckers' product category (UGG/Crocs-style), and "still hot" is demand news. |

**Implication for the paper:** FinBERT's polarity is roughly right when the headline is about the named firm. Opposite-sign errors are rare (7/200). The main noise source is FNSPID's ticker-to-headline mapping (off-target headlines) plus FinBERT's low neutral recall. This argues for treating the daily sentiment score as noisy, and possibly for down-weighting headlines that don't mention the firm. This is one LLM annotator, so it is not ground truth. The owner's own `my_label` column is still unfilled.

---

# Annotator B and reliability of the reference labels

## Method (annotator B)
- **Annotator:** Claude Haiku 5.5 (model ID `claude-haiku-5-5`), run in a separate Claude Code side session on 2026-11-07.
- **Input:** `results/hand_check_frozen.csv` only (id, ticker, date, headline). Labeled by reading each headline. No model, script, or keyword rule was used.
- **Definitions:** same as annotator A (sentiment toward the named ticker, investor's point of view; positive / negative / neutral, where neutral covers routine, mixed, and off-target headlines).
- **Output:** `results/hand_check_llm_B.csv` (200 rows, 0 missing labels; B labels: 115 neutral, 43 negative, 42 positive).
- **Independence caveat:** B was labeled in a conversation where A's labels and the FinBERT key had already been seen. B was not blind to A or FinBERT. The high A–B agreement may therefore be inflated (shared context and a shared underlying model family), so treat κ(A,B) as an upper bound on the reliability of the reference labels.
- **Consensus set:** the 169 headlines where A and B give the same label. The 31 disagreements are excluded from the consensus comparison.

## Agreement and kappa (n = 200 unless noted)

| Comparison | Agreement | Cohen's κ |
|---|---|---|
| A vs B (reference reliability) | 0.845 | 0.728 |
| FinBERT vs B | 0.605 | 0.407 |
| FinBERT vs A (from above) | 0.530 | 0.294 |
| FinBERT vs consensus (A = B), n = 169 | 0.598 | 0.379 |

## A vs B confusion matrix (rows = A, columns = B)

| A \ B | negative | neutral | positive |
|---|---|---|---|
| **negative** | 32 | 2 | 0 |
| **neutral**  | 11 | 102 | 7 |
| **positive** | 0 | 11 | 35 |

## FinBERT vs consensus confusion matrix (n = 169; rows = consensus label, columns = FinBERT)

| consensus \ FinBERT | negative | neutral | positive |
|---|---|---|---|
| **negative** | 25 | 7 | 0 |
| **neutral**  | 23 | 50 | 29 |
| **positive** | 2 | 7 | 26 |

## Interpretation
- **Reference reliability:** A and B agree at κ = 0.73, which is substantial. This is the ceiling for the comparison. The inflation caveat above applies.
- **FinBERT against the more reliable reference:** κ rises from 0.29 (A) to 0.41 (B) and 0.38 on the consensus set. The biggest gap is in neutral recall: 50 of 102 consensus-neutral headlines get a polar FinBERT label. FinBERT's polarity is closer to the labels than the single-annotator κ suggested, but it is still only moderate agreement.
- **Pattern:** the A–B and FinBERT–consensus disagreements both concentrate on neutral vs polar. Opposite-sign errors are rare: 0 consensus-negative headlines get a FinBERT-positive label, and 2 consensus-positive headlines get a FinBERT-negative label (consensus set, n = 169).
- **Still needed:** the owner's own `my_label` column is unfilled, so the reference is still two LLM annotators. Filling it would give a human reference.
