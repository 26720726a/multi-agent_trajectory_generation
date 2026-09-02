#!/usr/bin/env python3
"""S6 — 라벨 CSV 를 학습 텐서로 바꾼다 (계획서 §5.2).

    python3 learn/data.py --out learn/cache/s6.npz

한 행 = (클러스터 상태, 후보 mode).  토큰 구성:

에이전트 슬롯 8 × 15
    0:2   위치  (클러스터 중심 기준 로컬)
    2:4   속도
    4:6   서브골 (같은 로컬 프레임)          <- augment_subgoals.py 가 만든다
    6:10  phase 원핫 4 [TO_WP, DWELL, TO_GOAL, DONE]
    10:14 route 원핫 4 (후보 mode 가 이 에이전트에 준 경로)
    14    yield rank / (n-1)                 (후보 mode 의 양보 순서)

**`--route-geom` (S6-4)**: route 원핫 4칸을 기하 6칸으로 바꿔 슬롯이 17차원이
된다 (`build_route_geom()` 참조).  기본 동작은 바뀌지 않는다.
장애물 슬롯 8 × 4
    중심(2, 로컬) + half-extents(2).  외접원 근사를 쓰지 않는다.
CLS 4
    k, 국소 밀도, cautious, split

**phase 는 4칸이다.** DONE 슬롯이 없으면 완료된 에이전트가 0 벡터가 되어
패딩과 구별되지 않는다.

`mode` 문자열 (`r0102|B>A>C>D|s`) 을 풀어 per-agent route/rank 와 전역
cautious/split 로 나눈다.  route/rank 는 에이전트마다 다르므로 토큰에,
cautious/split 은 후보 전체에 걸리므로 CLS 에 넣는다 (§5.1 의 "전역 특성은
CLS" 원칙).
"""
from __future__ import annotations
import argparse, csv, json, os, sys
import numpy as np

D_AGENT, D_OBST, D_GLOB = 15, 4, 4
SLOTS = 8
N_ROUTE = 4


def parse_mode(label: str, n: int):
    """`r0102|B>A>C>D|s` -> (route_idx[n], yield_rank[n], cautious, split)."""
    parts = label.split('|')
    routes = [int(c) for c in parts[0][1:]]
    order = [ord(c) - 65 for c in parts[1].split('>')]
    rank = [0] * n
    for pos, ag in enumerate(order):
        rank[ag] = pos
    flags = parts[2] if len(parts) > 2 else ''
    return routes, rank, ('c' in flags), ('s' in flags)


#: S6-4 route 기하 인코딩의 칸 수와 뜻.
GEO_COLS = 6
GEO_NAMES = ('len_ratio', 'excess_total', 'excess_leg1', 'excess_leg2',
             'sin_dheading', 'cos_dheading')


def load_route_geom(path: str):
    """(uid, agent) -> route_idx 별 (length, len1, len2, hx, hy) 리스트."""
    out = {}
    with open(path, newline='', encoding='utf8') as f:
        for r in csv.DictReader(f):
            out.setdefault((r['uid'], int(r['agent'])), {})[int(r['route_idx'])] = (
                float(r['length']), float(r['len1']), float(r['len2']),
                float(r['hx']), float(r['hy']))
    return {k: [v[j] for j in sorted(v)] for k, v in out.items()}


def route_geom_feat(lib, j):
    """route j 의 기하를 **route 0(최단경로) 기준 상대값**으로 6칸에 담는다.

    route 0 은 canonical 후보가 쓰는 경로다 (worldmodel.sample_modes).  라벨이
    route 0 을 공통 자로 쓰므로(S5-3E) 기준을 route 0 으로 두는 것이 자연스럽다.
    route 0 자신은 (1, 0, 0, 0, 0, 1) 로 고정된다.

    `excess_leg1`·`excess_leg2` 를 넣는 이유: 라이브러리가 중복을 거르는 키가
    `(round(len1, 2), round(len2, 2))` 다 (`worldmodel.py:139`).  즉 이 두 칸은
    **라이브러리 자신의 식별 키**라, 원핫이 구별하던 것을 하나도 잃지 않는다.
    총길이/비율만으로는 leg 배분이 다른 두 경로가 겹칠 수 있다.

    `rollout` 은 색인을 `min(j, len(lib)-1)` 로 자르므로 여기서도 같게 자른다
    (`worldmodel.py:205`).
    """
    j = min(j, len(lib) - 1)
    L, l1, l2, hx, hy = lib[j]
    L0, l10, l20, hx0, hy0 = lib[0]
    # 방향 편차는 각도 대신 sin/cos 로 — 불연속(±π 접힘)을 피한다.
    cos = hx * hx0 + hy * hy0
    sin = hx0 * hy - hy0 * hx
    return (L / max(L0, 1e-6), L - L0, l1 - l10, l2 - l20, sin, cos)


def load_subgoals(d: str):
    """(uid, replan_idx, agent) -> (pos_x, pos_y, sg_x, sg_y)."""
    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith('.csv'):
            continue
        with open(os.path.join(d, fn), newline='', encoding='utf8') as f:
            for r in csv.DictReader(f):
                out[(r['uid'], int(r['replan_idx']), int(r['agent']))] = (
                    float(r['pos_x']), float(r['pos_y']),
                    float(r['sg_x']), float(r['sg_y']))
    return out


def build(csv_path: str, sg_dir: str, geom_path: str = None):
    sg = load_subgoals(sg_dir)
    geo = load_route_geom(geom_path) if geom_path else None
    d_agent = D_AGENT + (GEO_COLS - N_ROUTE if geo else 0)
    rows = []
    with open(csv_path, newline='', encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    N = len(rows)
    agent = np.zeros((N, SLOTS, d_agent), np.float32)
    amask = np.zeros((N, SLOTS), bool)
    obst = np.zeros((N, SLOTS, D_OBST), np.float32)
    omask = np.zeros((N, SLOTS), bool)
    glob = np.zeros((N, D_GLOB), np.float32)
    loss = np.zeros(N, np.float32)
    stalled = np.zeros(N, np.int64)
    kk = np.zeros(N, np.int64)
    nn = np.zeros(N, np.int64)
    midx = np.zeros(N, np.int64)
    uid = np.empty(N, object)
    split = np.empty(N, object)
    dep = np.empty(N, object)
    phase_key = np.empty(N, object)
    gkey = np.empty(N, object)
    pos_err = 0.0
    miss = 0

    for r_i, r in enumerate(rows):
        n = int(r['n_agents'])
        ri = int(r['replan_idx'])
        mem = [int(v) for v in r['members'].split(',')]
        atok = json.loads(r['agent_tokens'])
        otok = json.loads(r['obstacle_tokens'])
        routes, rank, cautious, splt = parse_mode(r['mode'], n)

        # 클러스터 중심은 라벨 수집이 쓴 것과 같게 구성원 절대위치의 평균이다.
        try:
            P = np.array([sg[(r['uid'], ri, i)][:2] for i in mem], float)
            S = np.array([sg[(r['uid'], ri, i)][2:] for i in mem], float)
        except KeyError:
            miss += 1
            continue
        center = P.mean(0)
        # 새너티: 재도출한 위치가 CSV 의 로컬 위치와 맞는지 (5자리 반올림 여유)
        pos_err = max(pos_err, float(np.abs(
            (P - center) - np.array(atok, float)[:, :2]).max()))

        for j, i in enumerate(mem):
            v = agent[r_i, j]
            v[0:2] = P[j] - center
            v[2:4] = atok[j][2:4]
            v[4:6] = S[j] - center
            v[6:10] = atok[j][4:8]
            if geo is None:
                v[10 + min(routes[i], N_ROUTE - 1)] = 1.0
                v[14] = rank[i] / max(1, n - 1)
            else:
                v[10:16] = route_geom_feat(geo[(r['uid'], i)], routes[i])
                v[16] = rank[i] / max(1, n - 1)
        amask[r_i, :len(mem)] = True
        for j, o in enumerate(otok[:SLOTS]):
            obst[r_i, j] = o
        omask[r_i, :min(len(otok), SLOTS)] = True
        glob[r_i] = (int(r['k']), float(r['density']), float(cautious), float(splt))

        loss[r_i] = float(r['loss_max'])
        stalled[r_i] = int(r['stalled'])
        kk[r_i] = int(r['k'])
        nn[r_i] = n
        midx[r_i] = int(r['mode_idx'])
        uid[r_i] = r['uid']
        split[r_i] = r['split']
        dep[r_i] = r['uid'].split('_')[1]
        phase_key[r_i] = ''.join(sorted(r['phase_sig']))
        gkey[r_i] = f"{r['uid']}|{ri}|{r['cluster_id']}"

    if miss:
        raise SystemExit(f'서브골 없음 {miss}행 — augment_subgoals.py 를 먼저 돌려라')
    print(f'행 {N:,}  위치 재도출 최대오차 {pos_err:.2e} m')
    assert pos_err < 1e-4, '재도출한 궤적이 라벨 수집과 다르다'

    # 3-클래스 (A3): 0 = 상호작용 없음, 1 = 정상, 2 = 정체
    # 우선순위는 "상호작용 없음"이 먼저다 — loss 가 비트 단위 0 이면 합동과
    # 단독 궤적이 같다는 뜻이므로, 그 정체는 회피가 만든 것이 아니다.
    cls3 = np.where(loss == 0.0, 0, np.where(stalled == 1, 2, 1)).astype(np.int64)

    # 정규화 대상(연속) 차원.  원핫·rank·플래그는 제외한다.
    cont = ([0, 1, 2, 3, 4, 5] + list(range(10, 10 + GEO_COLS))
            if geo else [0, 1, 2, 3, 4, 5])

    _, gid = np.unique(gkey.astype(str), return_inverse=True)
    return dict(agent_cont=np.array(cont, np.int64),
                agent=agent, agent_mask=amask, obst=obst, obst_mask=omask,
                glob=glob, loss=loss, stalled=stalled, cls3=cls3, k=kk,
                n_agents=nn, mode_idx=midx, uid=uid.astype(str),
                split=split.astype(str), dep=dep.astype(str),
                phase_key=phase_key.astype(str), gid=gid,
                pos_err=np.float64(pos_err))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='results/labels/main_progress_dedup.csv')
    p.add_argument('--subgoals', default='results/labels/subgoal')
    p.add_argument('--out', default='learn/cache/s6.npz')
    p.add_argument('--route-geom', default=None,
                   help='주면 route 원핫 4칸을 기하 6칸으로 바꾼다 (S6-4)')
    a = p.parse_args()
    d = build(a.csv, a.subgoals, a.route_geom)
    print(f'에이전트 슬롯 {d["agent"].shape[2]}차원  '
          f'연속 차원 {d["agent_cont"].tolist()}')
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    np.savez_compressed(a.out, **d)
    for s in ('train', 'val', 'test'):
        m = d['split'] == s
        print(f'  {s:5s} 행 {m.sum():>7,}  uid {len(set(d["uid"][m])):>4,}  '
              f'클러스터상태 {len(set(d["gid"][m])):>6,}  '
              f'정체 {d["stalled"][m].mean():.4f}  '
              f'loss>0 {(d["loss"][m] > 0).mean():.4f}')
    print(f'저장 {a.out}')


if __name__ == '__main__':
    main()
