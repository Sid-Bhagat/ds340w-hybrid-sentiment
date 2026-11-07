"""2023 holdout evaluation runner (technical wrapper; no change to evaluate.py's tests).

FROZEN.md specifies the tests on 2023 test days. evaluate.py pools all folds, so this wrapper runs the
unchanged evaluate.main() with predictions restricted to fold 2023 and writes to results/final_2023/.
Requires FINAL=1. The pooled 2018-2023 run is the plain `FINAL=1 python evaluate.py`.
"""
import shutil

import evaluate
from config import RESULTS, is_final

OUT = RESULTS / "final_2023"


def main():
    assert is_final(), "eval_2023 needs FINAL=1"
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ["coefs.csv", "coefs_m4.csv"]:          # coef tables are read from RESULTS by evaluate
        shutil.copy(RESULTS / f, OUT / f)
    orig = evaluate.load_preds
    evaluate.load_preds = lambda: (lambda p: p.loc[p["fold"] == 2023].reset_index(drop=True))(orig())
    evaluate.RESULTS, evaluate.TEX = OUT, OUT / "tables"
    evaluate.main()


if __name__ == "__main__":
    main()
