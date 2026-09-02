#!/usr/bin/env python3
"""S6 보조 — 라벨 CSV 에 없는 **로컬 서브골**을 재도출한다.

`collect_progress_labels.py` 가 저장한 `agent_tokens` 는 8차원
[pos-center(2), vel(2), phase 원핫(4)] 뿐이라 계획서 §5.2 가 요구하는
서브골(2)이 빠져 있다.  서브골은 (waypoint | goal) 절대좌표라 클러스터
중심을 알아야 로컬로 바꿀 수 있고, 중심은 절대 위치가 있어야 나온다 —
CSV 에는 중심을 뺀 값만 있으므로 복원이 불가능하다.

라벨을 다시 만들지 않는다.  **rollout 없이** `run_wm_planner(seed=0)` 만
다시 돌려 `res.states` 를 얻는다 (라벨 수집이 쓴 것과 같은 궤적이다).
비용은 라벨 수집의 rollout 부분(96%)이 빠진 나머지다.

    python3 scripts/augment_subgoals.py --out results/labels/subgoal/shard0.csv \
        --uids results/labels/main_progress_dedup.csv --shard 0 --n-shards 16

서브골 정의는 `planning/control.py:585` 의 `target_pt` 와 같다:
TO_WP 면 waypoint, 그 밖(DWELL/TO_GOAL/DONE)이면 goal.
"""
from __future__ import annotations
import argparse, csv, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.generate import axis_from_config, buildable, grid
from planning.control import PHASE_TO_WP
from planning.execute import WMConfig, run_wm_planner
import json

FIELDS = ['uid', 'replan_idx', 'agent', 'pos_x', 'pos_y', 'sg_x', 'sg_y', 'phase']


def subgoal(agent, phase) -> np.ndarray:
    """control.py 의 `target_pt` 와 같은 규칙."""
    return np.asarray(agent.waypoint if phase == PHASE_TO_WP else agent.goal, float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--uids', required=True, help='이 CSV 에 나오는 uid 만 처리')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--n-shards', type=int, default=1)
    a = p.parse_args()

    want = set()
    with open(a.uids, newline='', encoding='utf8') as f:
        for r in csv.DictReader(f):
            want.add(r['uid'])

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
            prob = buildable(x)
            res = run_wm_planner(prob, WMConfig(seed=0))
            for ri, st in enumerate(res.states):
                for i in range(prob.n):
                    sg = subgoal(prob.agents[i], st.phase[i])
                    wr.writerow({'uid': x.uid, 'replan_idx': ri, 'agent': i,
                                 'pos_x': f'{st.pos[i][0]:.6f}',
                                 'pos_y': f'{st.pos[i][1]:.6f}',
                                 'sg_x': f'{sg[0]:.6f}', 'sg_y': f'{sg[1]:.6f}',
                                 'phase': int(st.phase[i])})
            if ii % 10 == 0:
                print(f'{ii}/{len(picked)} {x.uid} '
                      f'{time.perf_counter()-t0:.0f}s', flush=True)
    print(f'done {len(picked)} uid {time.perf_counter()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
