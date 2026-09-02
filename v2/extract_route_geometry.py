#!/usr/bin/env python3
"""S6-4 D0 — route 라이브러리의 **기하**를 뽑는다 (rollout 없음).

    python3 scripts/extract_route_geometry.py --out results/labels/route_geom.csv \
        --uids results/labels/main_progress_dedup.csv --shard 0 --n-shards 8

`PlanMode.routes[i]` 는 "에이전트 i 의 몇 번째 경로냐"라는 **장면-국소 색인**일
뿐이다 (S6-3 §7).  그 색인이 가리키는 실제 경로의 기하를 여기서 뽑는다.

route 라이브러리는 `WorldModel.__init__` 에서 `_routes_for` 로 만들어지고
**상태에 의존하지 않는다** — 에이전트의 start/waypoint/goal 과 월드만 쓰므로
uid 마다 한 번만 계산하면 replan 전체에 쓸 수 있다 (확인:
`planning/worldmodel.py:124`).  라벨 수집과 같은 인자를 쓴다
(`WorldModel(prob, seed=0)`, k_routes=PLANNER.k_routes, margin=0.12).

뽑는 값 (에이전트 i, route j 마다)
    length   폴리라인 총 호길이 (leg1 + leg2)
    hx, hy   시작점에서의 진행 방향 단위벡터 (leg1 의 첫 구간)
    sx, sy   시작점 좌표 (위치 특성과 중복인지 확인용)
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.generate import axis_from_config, buildable, grid
from planning.worldmodel import WorldModel

FIELDS = ['uid', 'agent', 'route_idx', 'n_routes', 'length', 'len1', 'len2',
          'hx', 'hy', 'sx', 'sy']


def heading(g) -> np.ndarray:
    """시작점에서의 진행 방향 단위벡터.  Guide.pts 는 ds=0.05 로 조밀하다."""
    d = g.pts[min(4, len(g.pts) - 1)] - g.pts[0]
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--uids', required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--n-shards', type=int, default=1)
    a = p.parse_args()

    want = {r['uid'] for r in csv.DictReader(
        open(a.uids, newline='', encoding='utf8'))}
    cfg = json.load(open('bench/configs/difficulty.json'))
    picked = [x for x in grid(axis_from_config(cfg['axis'])) if x.uid in want]
    picked.sort(key=lambda x: x.uid)
    picked = [x for i, x in enumerate(picked) if i % a.n_shards == a.shard]

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    t0 = time.perf_counter()
    with open(a.out, 'w', newline='', encoding='utf8') as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        for ii, x in enumerate(picked, 1):
            wm = WorldModel(buildable(x), seed=0)
            for i, lib in enumerate(wm.library):
                for j, rt in enumerate(lib):
                    h = heading(rt.leg1)
                    wr.writerow({'uid': x.uid, 'agent': i, 'route_idx': j,
                                 'n_routes': len(lib),
                                 'length': f'{rt.length:.6f}',
                                 'len1': f'{rt.leg1.length:.6f}',
                                 'len2': f'{rt.leg2.length:.6f}',
                                 'hx': f'{h[0]:.6f}', 'hy': f'{h[1]:.6f}',
                                 'sx': f'{rt.leg1.pts[0][0]:.6f}',
                                 'sy': f'{rt.leg1.pts[0][1]:.6f}'})
            if ii % 25 == 0:
                print(f'{ii}/{len(picked)} {x.uid} '
                      f'{time.perf_counter()-t0:.0f}s', flush=True)
    print(f'done {len(picked)} uid {time.perf_counter()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
