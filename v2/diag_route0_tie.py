#!/usr/bin/env python3
"""S7 A-2 후속 — "route 0 이 최소값을 과점한다"는 것이 실제 이점인가 동점 구조인가.

S6-4 §8 은 이렇게 쟀다:

    임의 후보가 "전원 route 0" 일 확률          18.0 %
    그룹의 최소값 집합에 그런 후보가 있을 확률   69.4 %   -> 3.8 배 "과대표집"

그런데 최소값 집합은 후보 하나가 아니다.  손실이 정확히 0 인 후보가 많아
(S5-3E: 전체 행의 47.5 %) **동점 집합의 크기가 크다.**  크기 T 의 집합이
기저비율 p 로 뽑히기만 해도 "하나라도 있을" 확률은 1-(1-p)^T 로 올라간다.

그래서 관측값을 **동점 크기를 고정한 귀무모형**과 대조한다:

    H0: 최소값 집합은 그룹의 후보 중 크기 T 인 균일 무작위 부분집합
        P(전원 route 0 후보가 하나라도 포함) = 1 - C(N-A, T)/C(N, T)

관측이 H0 과 같으면 route 0 이점은 **없다** — "과대표집"은 동점의 산물이다.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from collections import defaultdict
from math import comb
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learn.data import parse_mode


def run(rows, key, tag):
    g = defaultdict(list)
    for r in rows:
        g[(r['uid'], r['replan_idx'], r['cluster_id'])].append(r)
    obs, exp, base, tie, n = [], [], [], [], 0
    obs_nz, exp_nz, n_nz = [], [], 0
    mean_r0, mean_not = [], []
    for rs in g.values():
        y = np.array([float(r[key]) for r in rs])
        a = np.array([all(parse_mode(r['mode'], int(r['n_agents']))[0][j] == 0
                          for j in map(int, r['members'].split(',')))
                      for r in rs])
        if len(y) < 2 or y.max() == y.min():
            continue
        N, A = len(y), int(a.sum())
        mins = y == y.min()
        T = int(mins.sum())
        n += 1
        base.append(A / N); tie.append(T)
        obs.append(bool((mins & a).any()))
        exp.append(1.0 - (comb(N - A, T) / comb(N, T) if N - A >= T else 0.0))
        if a.any():
            mean_r0.append(float(y[a].mean()))
        if (~a).any():
            mean_not.append(float(y[~a].mean()))
        if y.min() > 1e-9:                     # 최소값이 0 이 아닌 그룹만
            n_nz += 1
            obs_nz.append(bool((mins & a).any()))
            exp_nz.append(1.0 - (comb(N - A, T) / comb(N, T) if N - A >= T else 0.0))
    o, e, b = float(np.mean(obs)), float(np.mean(exp)), float(np.mean(base))
    out = dict(label=tag, n_nondegenerate_groups=n,
               p_candidate_all_route0=b,
               p_min_has_all_route0_OBSERVED=o,
               p_min_has_all_route0_H0_TIE=e,
               oversampling_vs_base=o / b if b else float('nan'),
               oversampling_vs_H0=o / e if e else float('nan'),
               excess_over_H0=o - e,
               tie_size_mean=float(np.mean(tie)),
               tie_size_median=float(np.median(tie)),
               group_size_mean=float(np.mean([len(v) for v in g.values()])),
               mean_loss_all_route0=float(np.mean(mean_r0)),
               mean_loss_not_all_route0=float(np.mean(mean_not)),
               nonzero_min_groups=n_nz,
               nonzero_min_OBSERVED=float(np.mean(obs_nz)) if obs_nz else float('nan'),
               nonzero_min_H0_TIE=float(np.mean(exp_nz)) if exp_nz else float('nan'))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--keys', default='loss_max,old_loss_max')
    a = p.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    res = {}
    names = {'loss_max': 'S7 회피 라벨', 'old_loss_max': 'S5-3E 라벨'}
    for k in a.keys.split(','):
        if k in rows[0]:
            res[k] = run(rows, k, names.get(k, k))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        json.dump(res, open(a.out, 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
