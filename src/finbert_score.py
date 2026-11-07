"""Phase 1b: FinBERT score = P(pos) - P(neg) for each UNIQUE headline, checkpointed + resumable.

Scores are cached in data/processed/finbert_ckpt/part_*.parquet (one part per 100k headlines).
Re-running only scores headlines that have no cached score, so it can be run again after the
news file grows. `build()` then joins scores + event tags onto the filtered news.
Only pre-2023 news_filtered.parquet is read here; the sealed file is never touched.
"""
import glob
import os
import sys
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import PROC
from events import tag_events

MODEL = "ProsusAI/finbert"
MAX_LEN = 64
BATCH = 256
PART_ROWS = 100_000
WORKERS = 6
CKPT = PROC / "finbert_ckpt"


def done_headlines():
    parts = sorted(glob.glob(str(CKPT / "part_*.parquet")))
    if not parts:
        return set(), 0
    return set(pd.concat([pd.read_parquet(p, columns=["headline"]) for p in parts])["headline"]), len(parts)


class Collate:
    """Tokenizes inside DataLoader workers (keeps the GPU fed)."""
    def __init__(self):
        self.tok = None

    def __call__(self, texts):
        if self.tok is None:
            self.tok = AutoTokenizer.from_pretrained(MODEL)
        return self.tok(list(texts), truncation=True, max_length=MAX_LEN, padding=True,
                        return_tensors="pt")


@torch.no_grad()
def score_part(model, texts, batch, pos_i, neg_i):
    dl = DataLoader(texts, batch_size=batch, shuffle=False, num_workers=WORKERS, collate_fn=Collate())
    out = []
    for enc in dl:
        enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
        p = torch.softmax(model(**enc).logits.float(), dim=-1)
        out.append(torch.stack([p[:, pos_i], p[:, neg_i]], 1).cpu())
    return torch.cat(out).numpy()


def run_scoring():
    assert torch.cuda.is_available(), "CUDA not available - stop and tell Sid"
    CKPT.mkdir(parents=True, exist_ok=True)
    news = pd.read_parquet(PROC / "news_filtered.parquet", columns=["headline"])
    uniq = pd.Series(news["headline"].unique())
    done, n_parts = done_headlines()
    todo = uniq[~uniq.isin(done)].reset_index(drop=True)
    print(f"unique headlines {len(uniq)}, already scored {len(done)}, to score {len(todo)}", flush=True)
    if not len(todo):
        return
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).half().cuda().eval()
    lab = {v.lower(): k for k, v in model.config.id2label.items()}
    pos_i, neg_i = lab["positive"], lab["negative"]
    batch, t0 = BATCH, time.time()
    for start in range(0, len(todo), PART_ROWS):
        texts = todo.iloc[start:start + PART_ROWS].tolist()
        while True:
            try:
                probs = score_part(model, texts, batch, pos_i, neg_i)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch <= 32:
                    raise
                batch //= 2
                print("OOM -> batch", batch, flush=True)
        df = pd.DataFrame({"headline": texts, "p_pos": probs[:, 0], "p_neg": probs[:, 1]})
        df["score"] = df["p_pos"] - df["p_neg"]
        path = CKPT / f"part_{n_parts:05d}.parquet"
        df.to_parquet(str(path) + ".tmp", index=False)
        os.replace(str(path) + ".tmp", path)
        n_parts += 1
        el = time.time() - t0
        print(f"part {n_parts} done: {start + len(texts)}/{len(todo)}  {el/60:.1f} min  batch={batch}",
              flush=True)


def build():
    """Join cached scores + event tags onto news_filtered -> news_scored.parquet."""
    news = pd.read_parquet(PROC / "news_filtered.parquet")
    sc = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(CKPT / "part_*.parquet")))])
    sc = sc.drop_duplicates("headline")
    out = news.merge(sc[["headline", "score"]], on="headline", how="left")
    assert out["score"].notna().all(), "unscored headlines remain - run scoring first"
    out = pd.concat([out, tag_events(out["headline"])], axis=1)
    path = PROC / "news_scored.parquet"
    out.to_parquet(str(path) + ".tmp", index=False)
    os.replace(str(path) + ".tmp", path)
    print("news_scored", out.shape, "score mean %.3f" % out["score"].mean(), flush=True)


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    else:
        run_scoring()
        build()
