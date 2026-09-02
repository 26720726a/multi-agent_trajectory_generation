#!/usr/bin/env python3
"""S5-3E 진행 거리 라벨 수집: 한 행 = (클러스터 상태, 후보 mode).

앞선 세 시범은 모두 "언제 끝났나"를 쟀고 셋 다 실패했다.  여기서는 고정
지평선 H 동안의 **진행량**을 재고, 같은 상태에서 혼자 굴렸을 때의 진행량과의
차이를 회피 손실로 쓴다.

    progress_i = (t=0 남은 호길이) - (t=H 남은 호길이)      [route 0 위에서]
    loss_i     = progress_alone_i - progress_together_i
    라벨       = max_i loss_i   (합도 함께 저장)

단독 기준선(A-1)
----------------
다른 에이전트를 PHASE_DONE 으로 바꾸기만 하면 **치워지지 않는다** — 얼어붙은
몸이 제자리에 남아 VO(alpha=1.0)와 안전 투영에 그대로 잡힌다.  그래서 여기서는
멀리 치우고(FAR), 남의 HOI 를 기다리는 의존성 대기도 푼다.  경로는 최단
경로(route 0 = canonical mode)를 쓴다.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.generate import axis_from_config, buildable, grid
from planning.control import (PHASE_DONE, PHASE_DWELL, PHASE_TO_GOAL,
                              PHASE_TO_WP)
from planning.execute import WMConfig, interaction_clusters, run_wm_planner
from planning.worldmodel import WorldModel
from config import PHYSICS, PLANNER

#: 단독 롤아웃에서 다른 에이전트를 치워 놓을 거리(m).  VO 의 원거리 컷오프와
#: 안전 투영의 seg-seg 거리를 모두 여유롭게 넘어선다.
FAR = 1.0e4

#: 분류 헤드 대안 (b): 함께 굴렸을 때 이만큼도 못 나아가면 막힌 것으로 본다.
BLOCKED_PROGRESS_M = 0.5

FIELDS = ['uid', 'planner_seed', 'replan_idx', 'cluster_id', 'mode_idx',
          'members', 'agent_tokens', 'obstacle_tokens', 'mode', 'k', 'n_agents',
          'phase_sig', 'density', 'loss_max', 'loss_sum', 'loss_min',
          'losses', 'prog_alone_sum', 'prog_together_sum', 'prog_together_min',
          'stalled', 'blocked', 'dwell_alone', 'dwell_together', 'dedup_key']


def remaining_on(route, pos, phase) -> float:
    """route 위에서 본 남은 호길이(m).

    phase 전환은 저절로 이어진다 — TO_WP 면 leg1 의 남은 길이에 leg2 전체를
    더하므로, 10초 안에 waypoint 를 지나 TO_GOAL 로 넘어가도 같은 척도의
    단조 감소가 된다 (B-2 결정 3).
    """
    if phase == PHASE_DONE:
        return 0.0
    if phase == PHASE_TO_WP:
        return route.leg1.remaining(route.leg1.project_full(pos)) + route.leg2.length
    if phase == PHASE_DWELL:
        return route.leg2.length
    return route.leg2.remaining(route.leg2.project_full(pos))


def dwell_steps(prob, ro, i) -> int:
    """이 rollout 창 안에서 에이전트 i 가 waypoint 에 멈춰 있던 스텝 수.

    DWELL 중에는 controller 가 위치를 waypoint 로 정확히 스냅해 두므로
    (control.py 의 도착 처리) 궤적만 보고 셀 수 있다.  이 정지는 회피가 아니라
    과업이고 단독 기준선에도 똑같이 들어가야 상쇄된다 (B-2 결정 4).
    """
    wp = np.asarray(prob.agents[i].waypoint, float)
    d = np.linalg.norm(ro.traj.pos[:, i] - wp[None, :], axis=1)
    return int(np.count_nonzero(d < 1e-9))


def solo_state(st, i, n):
    """에이전트 i 만 살아 있는 상태 — 나머지는 치우고 의존성 대기도 푼다."""
    ss = st.copy()
    ss.phase[:] = PHASE_DONE
    ss.phase[i] = st.phase[i]
    ss.vel[:] = 0.0
    ss.vel[i] = st.vel[i]
    for j in range(n):
        if j == i:
            continue
        ss.pos[j] = np.array([FAR + 10.0 * j, FAR])
    ss.wp_out = np.where(np.arange(n) == i, ss.wp_out, 0)
    return ss


def pick_instances(limit, stratify=False):
    """시범(stratify=False)은 S5-3/3'/3" 와 **같은 20개**를 그대로 쓴다.

    본 수집은 축(n_agents x dep_mode x size x couple_prob)을 고르게 훑는다 —
    앞의 20개는 전부 chain 에 낮은 seed 라 본 수집의 대표성이 없다.
    """
    cfg = json.load(open('bench/configs/difficulty.json'))
    all_inst = list(grid(axis_from_config(cfg['axis'])))
    if not stratify:
        by = {n: [x for x in all_inst if x.n_agents == n] for n in (2, 3, 4)}
        q, r = divmod(limit, 3)
        return [x for n, take in ((2, q + (r > 0)), (3, q + (r > 1)), (4, q))
                for x in by[n][:take]]
    cells = {}
    for x in all_inst:
        cells.setdefault((x.n_agents, x.dep_mode, x.size, x.couple_prob), []).append(x)
    keys = sorted(cells)
    out, r = [], 0
    while len(out) < min(limit, len(all_inst)):
        added = False
        for k in keys:
            if r < len(cells[k]) and len(out) < limit:
                out.append(cells[k][r]); added = True
        if not added:
            break
        r += 1
    return out


def collect_instance(x, wr, counters):
    prob = buildable(x)
    res = run_wm_planner(prob, WMConfig(seed=0))
    wm = WorldModel(prob, seed=0)
    modes = wm.sample_modes(max_modes=PLANNER.max_modes)
    Hs = PLANNER.horizon_steps(prob.dt)
    r0 = [wm.library[i][0] for i in range(prob.n)]           # 최단 경로
    obst = [[(o.x0 + o.x1) / 2, (o.y0 + o.y1) / 2,
             (o.x1 - o.x0) / 2, (o.y1 - o.y0) / 2]
            for o in prob.world.obstacles[:8]]
    for ri, st in enumerate(res.states):
        cs = [c for c in interaction_clusters(
            st.pos, st.phase, PHYSICS.interact_cluster_radius) if len(c) >= 2]
        if not cs:
            continue
        members = sorted({i for c in cs for i in c})
        # 단독 기준선은 후보와 무관하므로 상태당 한 번만 굴린다 (구성원 수 회).
        t0 = time.perf_counter()
        alone, dwell_alone = {}, {}
        for i in members:
            ro = wm.rollout(solo_state(st, i, prob.n), modes[0], horizon=Hs)
            pe = ro.end_state
            alone[i] = (remaining_on(r0[i], st.pos[i], st.phase[i])
                        - remaining_on(r0[i], pe.pos[i], pe.phase[i]))
            dwell_alone[i] = dwell_steps(prob, ro, i)
        counters['solo_s'] += time.perf_counter() - t0
        counters['solo_n'] += len(members)
        # 함께 굴리는 rollout 은 클러스터가 아니라 (상태, 후보)에 딸린다.
        t0 = time.perf_counter()
        joint = []
        for m in modes:
            ro = wm.rollout(st, m, horizon=Hs)
            joint.append(ro)
        counters['joint_s'] += time.perf_counter() - t0
        counters['joint_n'] += len(modes)
        for ci, c in enumerate(cs):
            base = {i: remaining_on(r0[i], st.pos[i], st.phase[i]) for i in c}
            center = st.pos[c].mean(0)
            agent = [[*(st.pos[i] - center), *st.vel[i],
                      *np.eye(4)[int(st.phase[i])]] for i in c]
            otok = [[o[0] - center[0], o[1] - center[1], o[2], o[3]] for o in obst]
            phase_sig = ''.join(str(int(st.phase[i])) for i in c)
            for mi, (m, ro) in enumerate(zip(modes, joint)):
                pe = ro.end_state
                tog = {i: base[i] - remaining_on(r0[i], pe.pos[i], pe.phase[i])
                       for i in c}
                loss = [alone[i] - tog[i] for i in c]
                dedup = hashlib.sha1(('|'.join(
                    f'{st.pos[i][0]:.2f},{st.pos[i][1]:.2f},{int(st.phase[i])}'
                    for i in c) + '#' + m.label(prob)).encode()).hexdigest()[:16]
                wr.writerow({
                    'uid': x.uid, 'planner_seed': 0, 'replan_idx': ri,
                    'cluster_id': ci, 'mode_idx': mi,
                    'members': ','.join(map(str, c)),
                    'agent_tokens': json.dumps([[round(v, 5) for v in a]
                                                for a in agent]),
                    'obstacle_tokens': json.dumps([[round(v, 5) for v in o]
                                                   for o in otok]),
                    'mode': m.label(prob), 'k': len(c), 'n_agents': prob.n,
                    'phase_sig': phase_sig,
                    'density': len(c) / (np.pi * PHYSICS.interact_cluster_radius ** 2),
                    'loss_max': max(loss), 'loss_sum': sum(loss),
                    'loss_min': min(loss),
                    'losses': ','.join(f'{v:.5f}' for v in loss),
                    'prog_alone_sum': sum(alone[i] for i in c),
                    'prog_together_sum': sum(tog[i] for i in c),
                    'prog_together_min': min(tog.values()),
                    'stalled': int(ro.stalled),
                    'blocked': int(min(tog.values()) < BLOCKED_PROGRESS_M),
                    'dwell_alone': sum(dwell_alone[i] for i in c),
                    'dwell_together': sum(dwell_steps(prob, ro, i) for i in c),
                    'dedup_key': dedup})
                counters['rows'] += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, default=20)
    p.add_argument('--stratify', action='store_true')
    #: 인스턴스를 shard 로 갈라 여러 프로세스로 돌린다.  uid 단위로 갈리므로
    #: 나중에 이어 붙여도 uid 기준 split 이 깨지지 않는다.
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--n-shards', type=int, default=1)
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    counters = {'rows': 0, 'solo_s': 0.0, 'solo_n': 0, 'joint_s': 0.0, 'joint_n': 0}
    t_all = time.perf_counter()
    with open(a.out, 'w', newline='', encoding='utf8') as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        picked = pick_instances(a.limit, a.stratify)
        picked = [x for i, x in enumerate(picked) if i % a.n_shards == a.shard]
        for ii, x in enumerate(picked, 1):
            collect_instance(x, wr, counters)
            print(ii, x.uid, counters['rows'],
                  f"{time.perf_counter()-t_all:.1f}s", flush=True)
    counters['wall_s'] = time.perf_counter() - t_all
    json.dump(counters, open(a.out.replace('.csv', '_meta.json'), 'w'), indent=1)
    print(json.dumps(counters, indent=1))


if __name__ == '__main__':
    main()
