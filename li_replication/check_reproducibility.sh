#!/usr/bin/env bash
# Reproducibility checks (README "Reproducibility"). Needs old/ (the
# September cache and outputs, not in git) and a filled data/ cache.
#   1. run.py on old/data, compared with old/outputs (September results)
#   2. a second run on old/data, byte-compared with the first
#   3. the fresh run in results/, compared with the September-cache run
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
TMP=$(mktemp -d)
REF=results/reproducibility/september_cache_run
OUT=results/reproducibility
mkdir -p "$REF"

$PY run.py --data-dir old/data --results-dir "$TMP/run1" > "$TMP/run1.log"
$PY run.py --data-dir old/data --results-dir "$TMP/run2" > "$TMP/run2.log"
cp "$TMP"/run1/*.csv "$REF"/

{
  echo "## 1. Final code on old/data vs old/outputs (September)"
  $PY compare.py september "$TMP/run1" --save "$OUT/september_vs_final.csv" | grep -E 'mapping|values match|figures' || true
  echo
  echo "## 2. Two runs of the final code on old/data"
  $PY compare.py identical "$TMP/run1" "$TMP/run2"
} > "$OUT/checks.txt.tmp"
mv "$OUT/checks.txt.tmp" "$OUT/checks.txt"

$PY compare.py changes "$REF" results --save "$OUT/fresh_vs_september" > "$OUT/fresh_vs_september.txt.tmp"
mv "$OUT/fresh_vs_september.txt.tmp" "$OUT/fresh_vs_september.txt"
rm -rf "$TMP"
cat "$OUT/checks.txt"
grep 'changed sign' "$OUT/fresh_vs_september.txt"
