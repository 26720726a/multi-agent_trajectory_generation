#!/usr/bin/env python3
"""S6 진단 — 왜 val 손실이 1~3 epoch 만에 바닥을 치는가.

두 가지를 잰다.

1. **유효 표본 수.**  한 클러스터 상태의 16 행은 토큰 특성이 거의 같다 —
   위치·속도·서브골·phase·장애물이 전부 동일하고 route 원핫·rank·
   cautious/split 만 다르다.  즉 행 수(94,949)가 아니라 클러스터 상태 수
   (5,937)가 독립 표본에 가깝다.
2. **라벨 분산 분해.**  top-1 regret 은 **그룹 안**의 순서만 본다.  그러므로
   그룹 내 분산이 전체에서 차지하는 비중이 이 과제의 상한을 정한다.
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learn.common import load, split_of


def decomp(d, name):
    gid, y = d['gid'], d['loss'].astype(np.float64)
    o = np.argsort(gid, kind='mergesort')
    gid, y = gid[o], y[o]
    bnd = np.flatnonzero(np.diff(gid)) + 1
    gs = np.split(y, bnd)
    gm = np.array([g.mean() for g in gs])
    n = np.array([len(g) for g in gs])
    within = float(sum(((g - g.mean()) ** 2).sum() for g in gs))
    between = float((n * (gm - y.mean()) ** 2).sum())
    tot = within + between
    # 토큰 특성(장면)만 같고 후보만 다른 행이 실제로 몇 개인지
    return dict(split=name, rows=len(y), groups=len(gs),
                rows_per_group=float(n.mean()),
                var_total=tot / len(y), within_frac=within / tot,
                between_frac=between / tot,
                mean=float(y.mean()), std=float(y.std()),
                zero_frac=float((y == 0).mean()),
                group_range_median=float(np.median([g.max() - g.min() for g in gs])),
                degenerate_frac=float(np.mean([g.max() == g.min() for g in gs])))


def main():
    d = load()
    print(f"{'split':6s} {'행':>8s} {'그룹':>7s} {'행/그룹':>7s} "
          f"{'라벨 std':>8s} {'그룹내 분산':>10s} {'그룹간 분산':>10s} "
          f"{'퇴화그룹':>8s} {'그룹 폭 중앙값':>12s}")
    for s in ('train', 'val', 'test'):
        r = decomp(split_of(d, s), s)
        print(f"{r['split']:6s} {r['rows']:>8,} {r['groups']:>7,} "
              f"{r['rows_per_group']:>7.1f} {r['std']:>8.3f} "
              f"{r['within_frac']:>10.3f} {r['between_frac']:>10.3f} "
              f"{r['degenerate_frac']:>8.3f} {r['group_range_median']:>12.3f}")
    import json
    json.dump([decomp(split_of(d, s), s) for s in ('train', 'val', 'test')],
              open('results/s6/diag_variance.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
