"""Phase 1b: stratified random sample of 200 scored headlines for Sid to label by hand.

Strata = FinBERT score band (negative < -1/3 <= neutral <= 1/3 < positive), ~67 each.
The FinBERT score/label are kept OUT of hand_check.csv (in hand_check_key.csv) so labelling is blind.
Fill `my_label` with positive / neutral / negative. Agreement + Cohen's kappa:
run `python src/hand_check.py --agree` after filling it in.
"""
import os
import sys

import pandas as pd

from config import RESULTS, SEED, PROC

OUT = RESULTS / "hand_check.csv"            # what Sid labels (blind: no model score shown)
KEY = RESULTS / "hand_check_key.csv"        # FinBERT score/label per id, used only for agreement


def band(s):
    return pd.cut(s, [-1.0001, -1 / 3, 1 / 3, 1.0001], labels=["negative", "neutral", "positive"])


def make_sample(n=200):
    d = pd.read_parquet(PROC / "news_scored.parquet", columns=["ticker", "trade_date", "headline", "score"])
    d = d.drop_duplicates("headline")
    d["finbert_label"] = band(d["score"])
    per = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    parts = [d.loc[d["finbert_label"] == lab].sample(k, random_state=SEED)
             for lab, k in zip(["negative", "neutral", "positive"], per)]
    s = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)   # shuffle
    s.insert(0, "id", range(1, len(s) + 1))
    s["my_label"] = ""
    s["score"] = s["score"].round(3)
    s["trade_date"] = s["trade_date"].dt.date
    RESULTS.mkdir(exist_ok=True)
    for path, cols in [(OUT, ["id", "ticker", "trade_date", "headline", "my_label"]),
                       (KEY, ["id", "score", "finbert_label"])]:
        s[cols].to_csv(str(path) + ".tmp", index=False)
        os.replace(str(path) + ".tmp", path)
    print("wrote", OUT, s.shape, s["finbert_label"].value_counts().to_dict())


def agreement(path=OUT, col="my_label"):
    from sklearn.metrics import cohen_kappa_score
    s = pd.read_csv(path).merge(pd.read_csv(KEY), on="id").dropna(subset=[col])
    s = s.loc[s[col].astype(str).str.strip() != ""]
    mine = s[col].str.strip().str.lower()
    print("n labelled", len(s), "agreement %.3f" % (mine == s["finbert_label"]).mean(),
          "kappa %.3f" % cohen_kappa_score(mine, s["finbert_label"]))


if __name__ == "__main__":
    def arg(name, default):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
    if "--agree" in sys.argv:
        agreement(arg("--input", OUT), arg("--col", "my_label"))
    else:
        make_sample()
