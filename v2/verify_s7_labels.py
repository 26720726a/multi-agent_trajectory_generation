#!/usr/bin/env python3
"""S7 새너티 1·2: 무작위 행 재계산 대조, 순열 등변성.

`scripts/verify_labels.py` 의 S7 판 — 자가 후보마다 다르고(자기 route),
경로 선택 비용이 별도 열이라 재계산 식이 다르다.  가법 분해
`old_loss = loss_route + loss_avoid0` 도 이 자리에서 다시 확인한다.
"""
from __future__ import annotations
import argparse, csv, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.generate import axis_from_config, buildable, grid
from planning.execute import WMConfig, interaction_clusters, run_wm_planner
from planning.worldmodel import WorldModel
from config import PHYSICS, PLANNER
from scripts.collect_progress_labels import remaining_on, solo_state
from scripts.collect_avoid_labels import solo_mode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    R = list(csv.DictReader(open(a.csv)))
    rng = random.Random(0)
    sample = rng.sample(R, min(a.n, len(R)))
    cfg = json.load(open('bench/configs/difficulty.json'))
    inst = {x.uid: x for x in grid(axis_from_config(cfg['axis']))}

    checks, cache = [], {}
    for r in sample:
        x = inst[r['uid']]
        if x.uid not in cache:
            prob = buildable(x)
            res = run_wm_planner(prob, WMConfig(seed=0))
            wm = WorldModel(prob, seed=0)
            cache[x.uid] = (prob, res, wm,
                            wm.sample_modes(max_modes=PLANNER.max_modes))
        prob, res, wm, modes = cache[x.uid]
        Hs = PLANNER.horizon_steps(prob.dt)
        lib = wm.library
        rt = lambda i, q: lib[i][min(q, len(lib[i]) - 1)]
        st = res.states[int(r['replan_idx'])]
        cs = [c for c in interaction_clusters(
            st.pos, st.phase, PHYSICS.interact_cluster_radius) if len(c) >= 2]
        c = cs[int(r['cluster_id'])]
        m = modes[int(r['mode_idx'])]
        rj = wm.rollout(st, m, horizon=Hs)
        pe = rj.end_state
        avoid, rcost, old, lroute, lav0 = {}, {}, {}, {}, {}
        for i in c:
            q = min(m.routes[i], len(lib[i]) - 1)
            b0 = remaining_on(rt(i, 0), st.pos[i], st.phase[i])
            br = remaining_on(rt(i, q), st.pos[i], st.phase[i])
            s0 = wm.rollout(solo_state(st, i, prob.n), solo_mode(prob.n, i, 0),
                            horizon=Hs).end_state
            sr = wm.rollout(solo_state(st, i, prob.n), solo_mode(prob.n, i, q),
                            horizon=Hs).end_state
            a0_r0 = b0 - remaining_on(rt(i, 0), s0.pos[i], s0.phase[i])
            ar_r0 = b0 - remaining_on(rt(i, 0), sr.pos[i], sr.phase[i])
            ar_own = br - remaining_on(rt(i, q), sr.pos[i], sr.phase[i])
            t_r0 = b0 - remaining_on(rt(i, 0), pe.pos[i], pe.phase[i])
            t_own = br - remaining_on(rt(i, q), pe.pos[i], pe.phase[i])
            avoid[i] = ar_own - t_own
            rcost[i] = br - b0
            old[i] = a0_r0 - t_r0
            lroute[i] = a0_r0 - ar_r0
            lav0[i] = ar_r0 - t_r0
        L = [avoid[i] for i in c]
        # 순열 등변성
        perm = list(range(len(c))); rng.shuffle(perm)
        cp = [c[j] for j in perm]
        center = st.pos[c].mean(0)
        tok = [[*(st.pos[i] - center), *st.vel[i], *np.eye(4)[int(st.phase[i])]]
               for i in c]
        tok_p = [[*(st.pos[i] - st.pos[cp].mean(0)), *st.vel[i],
                  *np.eye(4)[int(st.phase[i])]] for i in cp]
        equivar = all(np.allclose(tok[perm[j]], tok_p[j], atol=1e-9)
                      for j in range(len(c)))
        Lp = [avoid[i] for i in cp]
        checks.append({
            'uid': r['uid'], 'replan_idx': r['replan_idx'],
            'cluster_id': r['cluster_id'], 'mode_idx': r['mode_idx'],
            'members_match': r['members'] == ','.join(map(str, c)),
            'mode_match': r['mode'] == m.label(prob),
            'loss_max_abs_err': abs(float(r['loss_max']) - max(L)),
            'loss_sum_abs_err': abs(float(r['loss_sum']) - sum(L)),
            'route_cost_max_abs_err': abs(float(r['route_cost_max'])
                                          - max(rcost.values())),
            'old_loss_max_abs_err': abs(float(r['old_loss_max'])
                                        - max(old.values())),
            'loss_route_max_abs_err': abs(float(r['loss_route_max'])
                                          - max(lroute.values())),
            'additivity_abs_err': max(abs(old[i] - lroute[i] - lav0[i]) for i in c),
            'stalled_match': int(r['stalled']) == int(rj.stalled),
            'perm_equivariant_tokens': bool(equivar),
            'perm_label_max_invariant': abs(max(L) - max(Lp)) < 1e-12,
            'perm_label_sum_invariant': abs(sum(L) - sum(Lp)) < 1e-12})
        print(checks[-1]['uid'], 'err=%.2e' % checks[-1]['loss_max_abs_err'],
              'equivar=%s' % equivar, flush=True)

    e = lambda k: max(c[k] for c in checks)
    rep = {
        'n': len(checks),
        'sanity_1_all_match': all(
            c['members_match'] and c['mode_match'] and c['stalled_match']
            and c['loss_max_abs_err'] < 1e-5 and c['loss_sum_abs_err'] < 1e-5
            and c['old_loss_max_abs_err'] < 1e-5 for c in checks),
        'max_loss_max_abs_err': e('loss_max_abs_err'),
        'max_loss_sum_abs_err': e('loss_sum_abs_err'),
        'max_old_loss_max_abs_err': e('old_loss_max_abs_err'),
        'max_route_cost_abs_err': e('route_cost_max_abs_err'),
        'max_loss_route_abs_err': e('loss_route_max_abs_err'),
        'max_additivity_abs_err': e('additivity_abs_err'),
        'sanity_2_equivariant': all(c['perm_equivariant_tokens'] for c in checks),
        'sanity_2_label_invariant': all(
            c['perm_label_max_invariant'] and c['perm_label_sum_invariant']
            for c in checks),
        'rows': checks}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        json.dump(rep, open(a.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
