"""Download FNSPID Stock_news/All_external.csv and verify size + sha256 against HF metadata."""
import hashlib
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

import config

REPO, FILE = "Zihan1004/FNSPID", "Stock_news/All_external.csv"


def main():
    info = HfApi().get_paths_info(REPO, [FILE], repo_type="dataset", expand=True)[0]
    want_size, want_sha = info.size, info.lfs.sha256
    print(f"expected size={want_size} sha256={want_sha}", flush=True)
    path = hf_hub_download(REPO, FILE, repo_type="dataset", local_dir=config.RAW)
    got = os.path.getsize(path)
    print(f"downloaded {path} size={got}", flush=True)
    if got != want_size:
        sys.exit(f"SIZE MISMATCH: {got} != {want_size}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    if h.hexdigest() != want_sha:
        sys.exit(f"SHA256 MISMATCH: {h.hexdigest()} != {want_sha}")
    print("VERIFIED size+sha256 OK", flush=True)


if __name__ == "__main__":
    main()
