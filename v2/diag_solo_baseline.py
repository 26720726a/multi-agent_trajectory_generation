#!/usr/bin/env python3
"""S5-3E A상 진단: 단독 기준선이 자유로운가(A-1), 후보가 결과를 바꾸는가(A-2).

라벨을 바꾸기 전에 두 가지만 본다.

A-1  현행 단독 롤아웃(다른 구성원을 PHASE_DONE 으로 얼림)은 자유롭지 않다 —
     얼어붙은 몸이 그 자리에 그대로 남아 VO(alpha=1.0)와 안전 투영에 잡힌다.
     `solo_state(..., mode="free")` 는 그들을 멀리 치우고 의존성 대기를 풀어
     진짜 단독 상태를 만든다.  둘의 궤적을 최단 경로(route 0)와 겹쳐 본다.

A-2  같은 클러스터 상태에서 후보 16개의 10초 전진 거리 분산.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.generate import axis_from_config, buildable, grid
from planning.control import (PHASE_DONE, PHASE_DWELL, PHASE_TO_GOAL,
                              PHASE_TO_WP)
from planning.execute import WMConfig, interaction_clusters, run_wm_planner
from planning.worldmodel import WorldModel
from config import PHYSICS, PLANNER

#: 단독 롤아웃에서 다른 에이전트를 치워 놓을 거리(m).  VO 의 원거리 컷오프
#: (R + (v_i+v_j)*tau)와 안전 투영의 seg-seg 거리 모두 여유롭게 넘어선다.
FAR = 1.0e4


def remaining_on(route, pos, phase) -> float:
    """route 0 위에서 본 남은 호길이 (m).  phase 전환이 자연히 이어진다."""
    if phase == PHASE_DONE:
        return 0.0
    if phase == PHASE_TO_WP:
        return route.leg1.remaining(route.leg1.project_full(pos)) + route.leg2.length
    if phase == PHASE_DWELL:
        return route.leg2.length
    return route.leg2.remaining(route.leg2.project_full(pos))


def solo_state(st, i, n, mode="free"):
    """에이전트 i 만 살아 있는 상태.

    mode="frozen" : 현행 방식 — 나머지를 PHASE_DONE 으로만 바꾼다 (제자리 유지).
    mode="free"   : 나머지를 FAR 로 치우고 의존성 대기를 푼다.
    """
    ss = st.copy()
    ss.phase[:] = PHASE_DONE
    ss.phase[i] = st.phase[i]
    ss.vel[:] = 0.0
    ss.vel[i] = st.vel[i]
    if mode == "free":
        for j in range(n):
            if j == i:
                continue
            ss.pos[j] = np.array([FAR + 10.0 * j, FAR])
        # 의존성 대기(pred_done)를 푼다: 남의 HOI 를 기다리는 것은 회피 비용이
        # 아니라 과업이고, 얼어붙은 선행자는 영원히 끝나지 않는다.
        ss.wp_out = np.where(np.arange(n) == i, ss.wp_out, 0)
    return ss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results/s5/a_diag.json')
    p.add_argument('--limit', type=int, default=6)
    p.add_argument('--states', type=int, default=20)
    a = p.parse_args()

    cfg = json.load(open('bench/configs/difficulty.json'))
    all_inst = list(grid(axis_from_config(cfg['axis'])))
    by = {n: [x for x in all_inst if x.n_agents == n] for n in (2, 3, 4)}
    q, r = divmod(a.limit, 3)
    inst = [x for n, take in ((2, q + (r > 0)), (3, q + (r > 1)), (4, q))
            for x in by[n][:take]]

    a1_rows, a2_rows, traj_dump = [], [], None
    for x in inst:
        prob = buildable(x)
        res = run_wm_planner(prob, WMConfig(seed=0))
        wm = WorldModel(prob, seed=0)
        modes = wm.sample_modes(max_modes=PLANNER.max_modes)
        Hs = PLANNER.horizon_steps(prob.dt)
        r0 = [wm.library[i][0] for i in range(prob.n)]        # 최단 경로
        picked = 0
        for ri, st in enumerate(res.states):
            cs = interaction_clusters(st.pos, st.phase, PHYSICS.interact_cluster_radius)
            for ci, c in enumerate(cs):
                if len(c) < 2:
                    continue
                if picked >= a.states:
                    continue
                picked += 1
                # ---- A-1: 얼린 단독 vs 자유 단독 ------------------------- #
                for i in c:
                    rem0 = remaining_on(r0[i], st.pos[i], st.phase[i])
                    row = {'uid': x.uid, 'replan_idx': ri, 'cluster': ci,
                           'agent': i, 'k': len(c), 'phase': int(st.phase[i])}
                    for tag in ('frozen', 'free'):
                        ss = solo_state(st, i, prob.n, tag)
                        ro = wm.rollout(ss, modes[0], horizon=Hs)
                        pe = ro.end_state
                        row[f'prog_{tag}'] = rem0 - remaining_on(
                            r0[i], pe.pos[i], pe.phase[i])
                        # 최단 경로에서 벗어난 정도
                        tr = ro.traj.pos[:, i]
                        # 두 leg 를 합쳐서 본다 — 10초 안에 waypoint 를 넘어가면
                        # leg1 만 재던 거리는 경로 이탈이 아니라 phase 전환이다.
                        gd = np.vstack([r0[i].leg1.pts, r0[i].leg2.pts])
                        dev = [float(np.min(np.linalg.norm(gd - q0[None, :], axis=1)))
                               for q0 in tr]
                        row[f'dev_{tag}'] = max(dev)
                        row[f'steps_{tag}'] = len(tr) - 1
                        row[f'stalled_{tag}'] = int(ro.stalled)
                        row[f'endphase_{tag}'] = int(pe.phase[i])
                        if traj_dump is None and tag == 'free' and len(c) >= 2:
                            traj_dump = {'uid': x.uid, 'agent': i, 'replan_idx': ri}
                        if traj_dump is not None and traj_dump.get('uid') == x.uid \
                                and traj_dump.get('agent') == i \
                                and traj_dump.get('replan_idx') == ri:
                            traj_dump[f'traj_{tag}'] = tr.tolist()
                            traj_dump['guide'] = gd[::10].tolist()
                    a1_rows.append(row)
                # ---- A-2: 후보 간 10초 전진 거리 분산 -------------------- #
                base = {i: remaining_on(r0[i], st.pos[i], st.phase[i]) for i in c}
                progs = []
                for mi, m in enumerate(modes):
                    ro = wm.rollout(st, m, horizon=Hs)
                    pe = ro.end_state
                    progs.append([base[i] - remaining_on(r0[i], pe.pos[i], pe.phase[i])
                                  for i in c])
                P = np.asarray(progs)                       # (n_modes, k)
                a2_rows.append({'uid': x.uid, 'replan_idx': ri, 'cluster': ci,
                                'k': len(c), 'n_modes': len(modes),
                                'per_agent_std': P.std(0).tolist(),
                                'sum_std': float(P.sum(1).std()),
                                'sum_ptp': float(np.ptp(P.sum(1))),
                                'n_distinct': int(len(set(
                                    tuple(np.round(v, 4)) for v in P)))})
        print(x.uid, len(a1_rows), len(a2_rows), flush=True)

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump({'a1': a1_rows, 'a2': a2_rows, 'traj': traj_dump},
              open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
