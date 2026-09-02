#!/usr/bin/env python3
"""S7 A-1a — 기존 라벨 `loss_max` 를 경로 성분과 회피 성분으로 가른다 (롤아웃 없음).

    python3 scripts/diag_label_decomp.py

기존 라벨(S5-3E)은 **모든 후보를 route 0 자로** 잰다:

    loss_i = prog_alone_i(route 0 주행, route 0 자) - prog_together_i(route r 주행, route 0 자)

한 그룹(= 같은 클러스터 상태) 안에서 `prog_alone` 은 후보와 무관한 상수다
(수집 스크립트가 상태당 한 번만 굴린다).  따라서 **후보 사이의 loss 변동은
전부 `prog_together` 에서 나온다.**  그 변동이 route 선택으로 얼마나 설명되는지
route 기하(`results/labels/route_geom.csv`)로 선형 투영해 잰다.

여기서 나오는 `loss_route` 는 **대리 지표**다 (총 경로길이 초과분에 회귀계수를
곱한 값).  정확한 가법 분해는 A-1b(`scripts/diag_label_decomp_exact.py`)가
실제 단독 롤아웃으로 한다.
"""
from __future__ import annotations
import csv, json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learn.data import parse_mode

CSV = 'results/labels/main_progress_dedup.csv'
GEOM = 'results/labels/route_geom.csv'
OUT = 'results/s7'


def load_geom(path):
    """(uid, agent) -> {route_idx: 총 호길이}"""
    g = defaultdict(dict)
    with open(path, newline='', encoding='utf8') as f:
        for r in csv.DictReader(f):
            g[(r['uid'], int(r['agent']))][int(r['route_idx'])] = float(r['length'])
    return g


def ols(x, y):
    """기울기·절편 없는 단순 회귀 R^2 (절편 포함)."""
    X = np.stack([x, np.ones_like(x)], 1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ beta
    ss_res = float(((y - fit) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(beta[0]), float(beta[1]), (1 - ss_res / ss_tot) if ss_tot > 0 else float('nan'), fit


def main():
    os.makedirs(OUT, exist_ok=True)
    geom = load_geom(GEOM)

    # ---- 행별로 (구성원 loss, 그 구성원의 route 초과길이) 를 모은다 -----------
    mem_loss, mem_exc, mem_split = [], [], []
    grp = defaultdict(list)             # (uid, replan, cluster) -> [(loss_max, exc_max, exc_of_argmax, all_r0)]
    n_rows = 0
    with open(CSV, newline='', encoding='utf8') as f:
        for r in csv.DictReader(f):
            n_rows += 1
            n = int(r['n_agents'])
            routes, _, _, _ = parse_mode(r['mode'], n)
            mem = [int(v) for v in r['members'].split(',')]
            losses = [float(v) for v in r['losses'].split(',')]
            exc = []
            for j in mem:
                lib = geom[(r['uid'], j)]
                exc.append(lib[routes[j]] - lib[0])
            mem_loss.extend(losses)
            mem_exc.extend(exc)
            mem_split.extend([r['split']] * len(mem))
            a = int(np.argmax(losses))
            grp[(r['uid'], r['replan_idx'], r['cluster_id'])].append(
                (float(r['loss_max']), max(exc), exc[a], all(e == 0.0 for e in exc)))

    ml = np.array(mem_loss); me = np.array(mem_exc)
    sp = np.array(mem_split)

    res = {'n_rows': n_rows, 'n_member_obs': len(ml), 'n_groups': len(grp)}

    # ---- 검산: 전원 route 0 후보에서 route 초과길이가 정확히 0 인가 ----------
    zero = me == 0.0
    res['sanity_all_route0_excess_zero'] = dict(
        n_all_route0_members=int(zero.sum()),
        frac=float(zero.mean()),
        max_abs_excess_on_those=float(np.abs(me[zero]).max()) if zero.any() else 0.0)

    # ---- 전체 풀링 회귀 ------------------------------------------------------
    b, c, r2, fit = ols(me, ml)
    lr, la = fit - c, ml - fit          # 경로 성분(중심 제거 전) / 잔차 = 회피 성분
    res['pooled'] = dict(
        beta=b, intercept=c, r2_route=r2,
        var_total=float(ml.var()), var_route=float(lr.var()), var_avoid=float(la.var()),
        share_route=float(lr.var() / ml.var()), share_avoid=float(la.var() / ml.var()),
        dominance_route_frac=float((np.abs(lr) > np.abs(la)).mean()),
        excess_mean=float(me.mean()), excess_p90=float(np.percentile(me, 90)),
        excess_max=float(me.max()))

    # ---- 그룹 내(후보 간) 변동만 — 모델이 실제로 골라야 하는 축 --------------
    #      그룹 평균을 빼고 같은 회귀를 돌린다.
    gi = defaultdict(list)
    with open(CSV, newline='', encoding='utf8') as f:
        for k, r in enumerate(csv.DictReader(f)):
            gi[(r['uid'], r['replan_idx'], r['cluster_id'])].append(k)
    # 그룹별 loss_max / exc_max 를 배열로
    gl, ge = [], []
    for key, rows in grp.items():
        L = np.array([t[0] for t in rows]); E = np.array([t[1] for t in rows])
        if len(L) < 2 or L.max() == L.min():
            continue
        gl.append(L - L.mean()); ge.append(E - E.mean())
    gl = np.concatenate(gl); ge = np.concatenate(ge)
    bg, cg, r2g, fitg = ols(ge, gl)
    lrg, lag = fitg - cg, gl - fitg
    res['within_group'] = dict(
        n_nondegenerate_groups=int(sum(1 for rows in grp.values()
                                       if len(rows) >= 2
                                       and max(t[0] for t in rows) != min(t[0] for t in rows))),
        n_obs=int(len(gl)), beta=bg, r2_route=r2g,
        var_total=float(gl.var()), var_route=float(lrg.var()), var_avoid=float(lag.var()),
        share_route=float(lrg.var() / gl.var()), share_avoid=float(lag.var() / gl.var()),
        dominance_route_frac=float((np.abs(lrg) > np.abs(lag)).mean()))

    # ---- route 초과길이 층별 loss (S6-4 §8 의 재확인, 구성원 단위) ----------
    bins = [(-1e-9, 1e-9), (1e-9, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 1e9)]
    strat = []
    for lo, hi in bins:
        s = (me > lo) & (me <= hi) if lo > 0 else (me >= lo) & (me <= hi)
        strat.append(dict(lo=lo if lo > 0 else 0.0, hi=hi if hi < 1e8 else None,
                          n=int(s.sum()),
                          loss_mean=float(ml[s].mean()) if s.any() else None,
                          loss_median=float(np.median(ml[s])) if s.any() else None))
    res['loss_by_route_excess'] = strat

    json.dump(res, open(f'{OUT}/label_decomp_proxy.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
