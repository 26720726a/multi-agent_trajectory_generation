#!/usr/bin/env python3
"""S6-4 — 캐시 텐서 자체를 기준으로 **특성-오라클 regret** 과 **정보 손실**을 잰다.

    python3 scripts/diag_feature_oracle_npz.py \
        --caches learn/cache/s6.npz learn/cache/s6_4.npz \
        --names onehot geom --out results/s6_4

S6-3 은 mode 문자열에서 동치류를 만들었다.  여기서는 **모델이 실제로 먹는
입력 텐서 전체**(agent + mask + obst + glob)를 바이트로 해싱해 동치류를
만든다 — 인코딩이 무엇이든 같은 방법으로 잴 수 있고, S6-3 의 값을 재현하는지로
방법 자체를 검증할 수 있다.

같은 동치류의 후보는 모델이 원리적으로 구별할 수 없다.
    낙관 = 동치류 안 최소를 집는다 / 중립 = 평균 / 비관 = 최대
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learn.common import load, split_of


def sig(d, i):
    h = hashlib.blake2b(digest_size=16)
    for k in ('agent', 'agent_mask', 'obst', 'obst_mask', 'glob'):
        h.update(np.ascontiguousarray(d[k][i]).tobytes())
    return h.digest()


def analyse(d, split):
    sp = split_of(d, split)
    gid, L = sp['gid'], sp['loss'].astype(np.float64)
    order = np.argsort(gid, kind='mergesort')
    bnd = np.flatnonzero(np.diff(gid[order])) + 1
    opt, neu, pes, ncls, ncand = [], [], [], [], []
    per_group = {}
    for g in np.split(order, bnd):
        cls = defaultdict(list)
        for i in g:
            cls[sig(sp, i)].append(i)
        ncls.append(len(cls)); ncand.append(len(g))
        per_group[int(gid[g[0]])] = len(cls)
        if L[g].max() == L[g].min():
            continue                     # 퇴화 그룹 — regret 이 정의상 0
        lo = L[g].min()
        opt.append(min(L[v].min() for v in cls.values()) - lo)
        neu.append(min(L[v].mean() for v in cls.values()) - lo)
        pes.append(min(L[v].max() for v in cls.values()) - lo)
    return dict(split=split, n_groups=len(ncls), n_nondegen=len(neu),
                oracle_optimistic=float(np.mean(opt)),
                oracle_neutral=float(np.mean(neu)),
                oracle_pessimistic=float(np.mean(pes)),
                frac_oracle_neutral_zero=float(np.mean(np.array(neu) == 0)),
                classes_per_group_mean=float(np.mean(ncls)),
                candidates_per_group_mean=float(np.mean(ncand)),
                frac_groups_fully_separated=float(np.mean(
                    np.array(ncls) == np.array(ncand)))), per_group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--caches', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--out', default='results/s6_4')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    res, cls_by_name = {}, {}
    for name, path in zip(a.names, a.caches):
        d = load(path)
        res[name], cls_by_name[name] = {}, {}
        for s in ('train', 'val', 'test'):
            res[name][s], cls_by_name[name][s] = analyse(d, s)
            r = res[name][s]
            print(f"{name:7s} {s:5s} 동치류/그룹 {r['classes_per_group_mean']:.3f} "
                  f"(후보 {r['candidates_per_group_mean']:.3f}, 완전분리 그룹 "
                  f"{r['frac_groups_fully_separated']:.4f})  오라클 regret "
                  f"낙관 {r['oracle_optimistic']:.5f} 중립 {r['oracle_neutral']:.5f} "
                  f"비관 {r['oracle_pessimistic']:.5f}  0인 그룹 "
                  f"{r['frac_oracle_neutral_zero']:.5f}")
    json.dump(res, open(f'{a.out}/feature_oracle_geo.json', 'w'), indent=1)

    if len(a.names) == 2:
        old, new = a.names
        info = {}
        for s in ('train', 'val', 'test'):
            o, n = cls_by_name[old][s], cls_by_name[new][s]
            lost = {g: (o[g], n[g]) for g in o if n[g] < o[g]}
            gain = {g: (o[g], n[g]) for g in o if n[g] > o[g]}
            info[s] = dict(n_groups=len(o),
                           n_groups_fewer_classes=len(lost),
                           n_groups_more_classes=len(gain),
                           mean_classes_old=float(np.mean(list(o.values()))),
                           mean_classes_new=float(np.mean(list(n.values()))),
                           examples_fewer=[dict(gid=g, old=v[0], new=v[1])
                                           for g, v in list(lost.items())[:20]])
            print(f'{s:5s} 동치류가 줄어든 그룹 {len(lost)} / 늘어난 그룹 {len(gain)}'
                  f'  (평균 {np.mean(list(o.values())):.3f} -> '
                  f'{np.mean(list(n.values())):.3f})')
        json.dump(info, open(f'{a.out}/route_encoding_info_check.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
