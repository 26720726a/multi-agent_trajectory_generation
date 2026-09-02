#!/usr/bin/env python3
"""S7 라벨: 회피 성분만 남긴 손실.  한 행 = (클러스터 상태, 후보 mode).

S5-3E 는 **모든 후보를 route 0 자로** 쟀다 (B-2 결정 2).  그래서 route 0 을
쓰는 후보가 정의상 유리해지고, S6-5 가 확인했듯 모델이 배울 것의 대부분이
"route 0 을 골라라"가 된다 — canonical 이 학습 없이 하는 일이다.

여기서는 **자를 후보마다 바꾼다.**

    base_i     = (후보의 route r_i 위에서) t=0 남은 호길이
    together_i = base_i - (합동 롤아웃 t=H 의 route r_i 위 남은 호길이)
    alone_i    = base_i - (단독 롤아웃 t=H 의 route r_i 위 남은 호길이)
    loss_avoid_i = alone_i - together_i      <- 새 주 라벨 (base_i 가 상쇄된다)

`base_i` 가 분자·분모에서 정확히 상쇄되므로 **경로 선택 비용이 사라지고
회피 비용만 남는다.**  canonical 도 route 0 위에서 남을 피해야 하므로
`loss_avoid` 가 0 이 아니다 — 모든 후보가 같은 출발선에 선다 (B-1).

경로 선택 비용은 버리지 않고 **별도 열**로 저장한다 (B-2).  롤아웃이 필요
없다 — t=0 의 남은 호길이 차이다:

    route_cost_i = (route r_i 위 t=0 남은 호길이) - (route 0 위 t=0 남은 호길이)

통합 시 `loss_avoid + w * route_cost` 로 합치면 된다.  모델은 롤아웃이 필요한
부분만 근사한다 (계획서 §1.3 증류 프레이밍).

같은 롤아웃에서 **S5-3E 라벨과 그 정확한 가법 분해**도 함께 낸다 (A-1b).
route 0 자 위에서

    old_loss_i = A0_i - T0_i = (A0_i - Ar0_i) + (Ar0_i - T0_i)
               = loss_route_i + loss_avoid0_i

    A0_i  단독, route 0 주행, route 0 자   (= S5-3E 의 prog_alone)
    Ar0_i 단독, route r 주행, route 0 자
    T0_i  합동, route r 주행, route 0 자   (= S5-3E 의 prog_together)

`A0_i` 는 (i, route 0) 캐시 항목이고 canonical 후보가 늘 그것을 요구하므로
**추가 롤아웃 비용이 0 이다.**

단독 기준선 캐싱 (B-3 결정 1·2)
-------------------------------
키는 `(구성원, route 색인)` 이다.  단독 mode 를 직접 만들어 쓰기 때문에
(항등 양보순서, cautious=False, split_side=False) 양보 순서와 플래그는
키에 들어가지 않는다.  `--check-mode-flags` 가 그 선택의 대가를 잰다.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.generate import axis_from_config, buildable, grid
from planning.execute import WMConfig, interaction_clusters, run_wm_planner
from planning.worldmodel import PlanMode, WorldModel
from config import PHYSICS, PLANNER
from scripts.collect_progress_labels import (BLOCKED_PROGRESS_M, dwell_steps,
                                             remaining_on, solo_state)

FIELDS = ['uid', 'planner_seed', 'replan_idx', 'cluster_id', 'mode_idx',
          'members', 'agent_tokens', 'obstacle_tokens', 'mode', 'k', 'n_agents',
          'phase_sig', 'density',
          # --- 새 주 라벨: 회피 성분 (후보 자기 route 자) ---
          'loss_max', 'loss_sum', 'loss_min', 'losses',
          # --- 경로 선택 비용 (롤아웃 없음, B-2) ---
          'route_cost_max', 'route_cost_sum', 'route_costs',
          # --- S5-3E 라벨과 그 정확한 가법 분해 (route 0 자, A-1b) ---
          'old_loss_max', 'old_loss_sum', 'old_losses',
          'loss_route_max', 'loss_route_sum', 'loss_routes',
          'loss_avoid0_max', 'loss_avoid0_sum', 'loss_avoid0s',
          # --- 진행량 원자료 (자기 route 자) ---
          'prog_alone_sum', 'prog_together_sum', 'prog_together_min',
          'stalled', 'blocked', 'dwell_alone', 'dwell_together', 'dedup_key']


def solo_mode(n: int, i: int, r: int) -> PlanMode:
    """에이전트 i 를 route r 에 태우는 단독용 mode.

    양보 순서는 항등, cautious/split_side 는 끈다 — 혼자일 때 route 이외의
    mode 성분이 기준선에 섞이지 않게 한다 (B-3 결정 2).
    """
    return PlanMode(tuple(r if j == i else 0 for j in range(n)),
                    tuple(range(n)), False, False)


def collect_instance(x, wr, counters):
    prob = buildable(x)
    res = run_wm_planner(prob, WMConfig(seed=0))
    wm = WorldModel(prob, seed=0)
    modes = wm.sample_modes(max_modes=PLANNER.max_modes)
    Hs = PLANNER.horizon_steps(prob.dt)
    lib = wm.library
    rt = lambda i, r: lib[i][min(r, len(lib[i]) - 1)]
    obst = [[(o.x0 + o.x1) / 2, (o.y0 + o.y1) / 2,
             (o.x1 - o.x0) / 2, (o.y1 - o.y0) / 2]
            for o in prob.world.obstacles[:8]]

    for ri, st in enumerate(res.states):
        cs = [c for c in interaction_clusters(
            st.pos, st.phase, PHYSICS.interact_cluster_radius) if len(c) >= 2]
        if not cs:
            continue
        members = sorted({i for c in cs for i in c})
        # 이 상태에서 실제로 쓰이는 (구성원, route) 쌍만 굴린다.
        need = sorted({(i, min(m.routes[i], len(lib[i]) - 1))
                       for i in members for m in modes})

        t0 = time.perf_counter()
        alone = {}
        for i, r in need:
            ro = wm.rollout(solo_state(st, i, prob.n), solo_mode(prob.n, i, r),
                            horizon=Hs)
            pe = ro.end_state
            alone[(i, r)] = dict(
                # 자기 route 자 (새 라벨용)
                own=remaining_on(rt(i, r), st.pos[i], st.phase[i])
                    - remaining_on(rt(i, r), pe.pos[i], pe.phase[i]),
                # route 0 자 (S5-3E 라벨·분해용)
                r0=remaining_on(rt(i, 0), st.pos[i], st.phase[i])
                   - remaining_on(rt(i, 0), pe.pos[i], pe.phase[i]),
                dwell=dwell_steps(prob, ro, i))
        counters['solo_s'] += time.perf_counter() - t0
        counters['solo_n'] += len(need)
        counters['solo_keys'] += len(need)
        counters['solo_members'] += len(members)

        t0 = time.perf_counter()
        joint = [wm.rollout(st, m, horizon=Hs) for m in modes]
        counters['joint_s'] += time.perf_counter() - t0
        counters['joint_n'] += len(modes)

        for ci, c in enumerate(cs):
            center = st.pos[c].mean(0)
            agent = [[*(st.pos[i] - center), *st.vel[i],
                      *np.eye(4)[int(st.phase[i])]] for i in c]
            otok = [[o[0] - center[0], o[1] - center[1], o[2], o[3]] for o in obst]
            phase_sig = ''.join(str(int(st.phase[i])) for i in c)
            base0 = {i: remaining_on(rt(i, 0), st.pos[i], st.phase[i]) for i in c}
            for mi, (m, ro) in enumerate(zip(modes, joint)):
                pe = ro.end_state
                avoid, rcost, old, lroute, lav0 = [], [], [], [], []
                tog_own, alone_own = [], []
                for i in c:
                    r = min(m.routes[i], len(lib[i]) - 1)
                    baser = remaining_on(rt(i, r), st.pos[i], st.phase[i])
                    t_own = baser - remaining_on(rt(i, r), pe.pos[i], pe.phase[i])
                    t_r0 = base0[i] - remaining_on(rt(i, 0), pe.pos[i], pe.phase[i])
                    a_own, a_r0 = alone[(i, r)]['own'], alone[(i, r)]['r0']
                    a0_r0 = alone[(i, 0)]['r0']
                    avoid.append(a_own - t_own)          # 새 주 라벨
                    rcost.append(baser - base0[i])       # 경로 선택 비용
                    old.append(a0_r0 - t_r0)             # S5-3E 라벨
                    lroute.append(a0_r0 - a_r0)          # 경로 성분 (정확)
                    lav0.append(a_r0 - t_r0)             # 회피 성분 (route 0 자)
                    tog_own.append(t_own); alone_own.append(a_own)
                dedup = hashlib.sha1(('|'.join(
                    f'{st.pos[i][0]:.2f},{st.pos[i][1]:.2f},{int(st.phase[i])}'
                    for i in c) + '#' + m.label(prob)).encode()).hexdigest()[:16]
                j = lambda v: ','.join(f'{q:.5f}' for q in v)
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
                    'loss_max': max(avoid), 'loss_sum': sum(avoid),
                    'loss_min': min(avoid), 'losses': j(avoid),
                    'route_cost_max': max(rcost), 'route_cost_sum': sum(rcost),
                    'route_costs': j(rcost),
                    'old_loss_max': max(old), 'old_loss_sum': sum(old),
                    'old_losses': j(old),
                    'loss_route_max': max(lroute), 'loss_route_sum': sum(lroute),
                    'loss_routes': j(lroute),
                    'loss_avoid0_max': max(lav0), 'loss_avoid0_sum': sum(lav0),
                    'loss_avoid0s': j(lav0),
                    'prog_alone_sum': sum(alone_own),
                    'prog_together_sum': sum(tog_own),
                    'prog_together_min': min(tog_own),
                    'stalled': int(ro.stalled),
                    'blocked': int(min(tog_own) < BLOCKED_PROGRESS_M),
                    'dwell_alone': sum(alone[(i, min(m.routes[i], len(lib[i]) - 1))]['dwell']
                                       for i in c),
                    'dwell_together': sum(dwell_steps(prob, ro, i) for i in c),
                    'dedup_key': dedup})
                counters['rows'] += 1


def check_mode_flags(x, n_states, out):
    """B-3 결정 2 의 대가를 잰다.

    단독 기준선을 (i, route) 로만 캐싱한다는 것은 **후보의 cautious/split_side 를
    기준선에 반영하지 않는다**는 뜻이다 (양보 순서 자체는 혼자일 때 무의미하지만,
    `speed_scale` 과 `side_bias` 는 양보 순서에서 파생되므로 혼자여도 작동한다 —
    `planning/control.py:589,692`).  그 차이를 실측한다.
    """
    prob = buildable(x)
    res = run_wm_planner(prob, WMConfig(seed=0))
    wm = WorldModel(prob, seed=0)
    modes = wm.sample_modes(max_modes=PLANNER.max_modes)
    Hs = PLANNER.horizon_steps(prob.dt)
    lib = wm.library
    rt = lambda i, r: lib[i][min(r, len(lib[i]) - 1)]
    done = 0
    for ri, st in enumerate(res.states):
        cs = [c for c in interaction_clusters(
            st.pos, st.phase, PHYSICS.interact_cluster_radius) if len(c) >= 2]
        if not cs:
            continue
        members = sorted({i for c in cs for i in c})
        for i in members:
            for m in modes:
                r = min(m.routes[i], len(lib[i]) - 1)
                ss, sd = float(m.speed_scale(prob.n)[i]), int(m.side(prob.n)[i])
                a = wm.rollout(solo_state(st, i, prob.n),
                               solo_mode(prob.n, i, r), horizon=Hs)
                b = wm.rollout(solo_state(st, i, prob.n), m, horizon=Hs)
                pa, pb = a.end_state, b.end_state
                ga = (remaining_on(rt(i, r), st.pos[i], st.phase[i])
                      - remaining_on(rt(i, r), pa.pos[i], pa.phase[i]))
                gb = (remaining_on(rt(i, r), st.pos[i], st.phase[i])
                      - remaining_on(rt(i, r), pb.pos[i], pb.phase[i]))
                out.append(dict(uid=x.uid, replan=ri, agent=i, route=r,
                                sscale=ss, side=sd, cautious=bool(m.cautious),
                                split=bool(m.split_side),
                                prog_canon_flags=ga, prog_cand_flags=gb,
                                diff=gb - ga))
        done += 1
        if done >= n_states:
            return


def pick_instances(limit, stratify=False):
    """S5-3E 의 시범 20 개는 전부 chain 에 낮은 seed 였다 (그 보고서 D 절).

    S7 의 시범은 지시서 C-1 대로 (n_agents x dep_mode x size) 층화로 뽑는다.
    """
    cfg = json.load(open('bench/configs/difficulty.json'))
    all_inst = list(grid(axis_from_config(cfg['axis'])))
    if not stratify:
        cells = {}
        for x in all_inst:
            cells.setdefault((x.n_agents, x.dep_mode, x.size), []).append(x)
    else:
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, default=40)
    p.add_argument('--stratify', action='store_true',
                   help='본 수집: couple_prob 까지 넣은 전체 격자')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--n-shards', type=int, default=1)
    p.add_argument('--check-mode-flags', type=int, default=0,
                   help='B-3 결정 2 검증: 앞 N 개 인스턴스에서 상태 1개씩')
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    counters = {'rows': 0, 'solo_s': 0.0, 'solo_n': 0, 'solo_keys': 0,
                'solo_members': 0, 'joint_s': 0.0, 'joint_n': 0}
    t_all = time.perf_counter()
    picked = pick_instances(a.limit, a.stratify)
    picked = [x for i, x in enumerate(picked) if i % a.n_shards == a.shard]

    if a.check_mode_flags:
        rows = []
        for x in picked[:a.check_mode_flags]:
            check_mode_flags(x, 1, rows)
        d = np.array([r['diff'] for r in rows])
        rep = dict(n=len(rows),
                   n_exact_zero=int((np.abs(d) < 1e-12).sum()),
                   frac_exact_zero=float((np.abs(d) < 1e-12).mean()),
                   abs_median=float(np.median(np.abs(d))),
                   abs_p90=float(np.percentile(np.abs(d), 90)),
                   abs_max=float(np.abs(d).max()), mean=float(d.mean()))
        for key, sel in (('cautious', [r['cautious'] for r in rows]),
                         ('split', [r['split'] for r in rows])):
            s = np.array(sel, bool)
            rep[f'by_{key}'] = {
                'true_n': int(s.sum()),
                'true_abs_median': float(np.median(np.abs(d[s]))) if s.any() else None,
                'false_abs_max': float(np.abs(d[~s]).max()) if (~s).any() else None}
        json.dump(dict(summary=rep, rows=rows[:500]),
                  open(a.out.replace('.csv', '_modeflags.json'), 'w'), indent=1)
        print(json.dumps(rep, indent=1))
        return

    with open(a.out, 'w', newline='', encoding='utf8') as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        for ii, x in enumerate(picked, 1):
            collect_instance(x, wr, counters)
            print(ii, x.uid, counters['rows'],
                  f"{time.perf_counter()-t_all:.1f}s", flush=True)
    counters['wall_s'] = time.perf_counter() - t_all
    counters['solo_per_member'] = (counters['solo_keys'] /
                                   max(counters['solo_members'], 1))
    counters['solo_overhead_frac'] = counters['solo_s'] / max(counters['joint_s'], 1e-9)
    json.dump(counters, open(a.out.replace('.csv', '_meta.json'), 'w'), indent=1)
    print(json.dumps(counters, indent=1))


if __name__ == '__main__':
    main()
