#!/usr/bin/env python3
"""S5-3E 라벨 파일 확정: 중복 제거 + uid 기준 split 열 부착."""
from __future__ import annotations
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analyze_labels import uid_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    R = list(csv.DictReader(open(a.csv)))
    seen, keep = set(), []
    for r in R:
        if r['dedup_key'] in seen:
            continue
        seen.add(r['dedup_key']); keep.append(r)
    sp = uid_split([r['uid'] for r in keep])
    fields = list(R[0]) + ['split']
    with open(a.out, 'w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in keep:
            r['split'] = sp[r['uid']]
            w.writerow(r)
    print(f'{len(R)} -> {len(keep)} rows  ({a.out})')


if __name__ == '__main__':
    main()
