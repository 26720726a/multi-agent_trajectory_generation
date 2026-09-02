#!/usr/bin/env bash
# S7 D — 본 수집.  전체 격자 1,620 인스턴스를 16 shard 로 병렬 수집한다.
# shard 는 인스턴스(uid) 단위로 갈리므로 이어 붙여도 uid 기준 split 이 깨지지 않는다.
set -euo pipefail
PY=${PY:-/home/kjs/Desktop/kuaicv/cvpr2027/Mahoi-WM/mahoi-wm/.venv/bin/python}
N=${N:-16}
OUT=results/labels/s7
mkdir -p "$OUT" logs
for s in $(seq 0 $((N-1))); do
  $PY scripts/collect_avoid_labels.py --out "$OUT/main_shard$s.csv" \
      --limit 1620 --stratify --shard "$s" --n-shards "$N" \
      > "logs/s7_main_shard$s.log" 2>&1 &
done
wait
head -1 "$OUT/main_shard0.csv" > "$OUT/main_avoid.csv"
for s in $(seq 0 $((N-1))); do tail -n +2 "$OUT/main_shard$s.csv" >> "$OUT/main_avoid.csv"; done
wc -l "$OUT/main_avoid.csv"
