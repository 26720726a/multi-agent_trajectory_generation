"""S6-3 D1 보강 — 충돌이 top-1 regret 을 얼마나 막고 있는가 (상한).

모델이 볼 수 있는 특성만으로 완벽하게 고른다면 regret 이 얼마까지 내려가는가.
같은 특성 튜플을 가진 후보들은 모델이 구별할 수 없으므로 하나의 동치류다.
  * 낙관: 동치류 안에서 운 좋게 최소를 집는다      -> min
  * 중립: 동치류 안에서 무작위로 집는다            -> mean   (기본)
  * 비관: 동치류 안에서 최악을 집는다              -> max
"""
import csv, json, sys, itertools
from collections import defaultdict
import numpy as np
sys.path.insert(0, '.')
from learn.data import parse_mode, N_ROUTE

rows = list(csv.DictReader(open('results/labels/main_progress_dedup.csv',
                                newline='', encoding='utf8')))
groups = defaultdict(list)
for r in rows:
    groups[(r['uid'], int(r['replan_idx']), int(r['cluster_id']))].append(r)

out = {}
for want in ('train', 'val', 'test'):
    opt, neu, pes, canon = [], [], [], []
    n_nd = 0
    for gk, rs in groups.items():
        if rs[0]['split'] != want:
            continue
        n = int(rs[0]['n_agents'])
        mem = [int(v) for v in rs[0]['members'].split(',')]
        L = np.array([float(r['loss_max']) for r in rs])
        if L.max() == L.min():
            continue                      # 퇴화 그룹 — regret 이 정의상 0
        n_nd += 1
        cls = defaultdict(list)
        for i, r in enumerate(rs):
            ro, rk, c, s = parse_mode(r['mode'], n)
            cls[(tuple((min(ro[a], N_ROUTE - 1), rk[a]) for a in mem), c, s)].append(i)
        lo = L.min()
        opt.append(min(L[v].min() for v in cls.values()) - lo)
        neu.append(min(L[v].mean() for v in cls.values()) - lo)
        pes.append(min(L[v].max() for v in cls.values()) - lo)
        ci = [i for i, r in enumerate(rs) if int(r['mode_idx']) == 0]
        if ci:
            canon.append(L[ci[0]] - lo)
    out[want] = dict(n_nondegen=n_nd,
                     oracle_optimistic=float(np.mean(opt)),
                     oracle_neutral=float(np.mean(neu)),
                     oracle_pessimistic=float(np.mean(pes)),
                     canonical=float(np.mean(canon)),
                     frac_oracle_neutral_zero=float(np.mean(np.array(neu) == 0)))
    print(f"{want:5s} n={n_nd:5d}  특성-오라클 regret  낙관 {out[want]['oracle_optimistic']:.4f}"
          f"  중립 {out[want]['oracle_neutral']:.4f}  비관 {out[want]['oracle_pessimistic']:.4f}"
          f"  | canonical {out[want]['canonical']:.4f}")
json.dump(out, open('results/s6_3/feature_oracle.json', 'w'), indent=1)
