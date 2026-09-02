#!/usr/bin/env python3
"""S5-3E 새너티 1·2: 무작위 행 재계산 대조, 순열 등변성.

1) 표본 행을 골라 원본 인스턴스를 다시 세우고 라벨을 처음부터 다시 계산해
   저장된 값과 비교한다.
2) 클러스터 구성원 순서를 섞으면 특성(agent_tokens)도 같이 섞이고 라벨
   (max/sum)은 불변인가.
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
        r0 = [wm.library[i][0] for i in range(prob.n)]
        st = res.states[int(r['replan_idx'])]
        cs = [c for c in interaction_clusters(
            st.pos, st.phase, PHYSICS.interact_cluster_radius) if len(c) >= 2]
        c = cs[int(r['cluster_id'])]
        m = modes[int(r['mode_idx'])]
        base = {i: remaining_on(r0[i], st.pos[i], st.phase[i]) for i in c}
        alone, tog = {}, {}
        for i in c:
            ro = wm.rollout(solo_state(st, i, prob.n), modes[0], horizon=Hs)
            alone[i] = base[i] - remaining_on(r0[i], ro.end_state.pos[i],
                                              ro.end_state.phase[i])
        rj = wm.rollout(st, m, horizon=Hs)
        for i in c:
            tog[i] = base[i] - remaining_on(r0[i], rj.end_state.pos[i],
                                            rj.end_state.phase[i])
        loss = [alone[i] - tog[i] for i in c]
        # 순열 등변성: 구성원 순서를 섞어 특성과 라벨을 다시 만든다
        perm = list(range(len(c))); rng.shuffle(perm)
        cp = [c[j] for j in perm]
        center = st.pos[c].mean(0)
        tok = [[*(st.pos[i] - center), *st.vel[i], *np.eye(4)[int(st.phase[i])]]
               for i in c]
        tok_p = [[*(st.pos[i] - st.pos[cp].mean(0)), *st.vel[i],
                  *np.eye(4)[int(st.phase[i])]] for i in cp]
        equivar = all(np.allclose(tok[perm[j]], tok_p[j], atol=1e-9)
                      for j in range(len(c)))
        loss_p = [alone[i] - tog[i] for i in cp]
        checks.append({
            'uid': r['uid'], 'replan_idx': r['replan_idx'],
            'cluster_id': r['cluster_id'], 'mode_idx': r['mode_idx'],
            'members_csv': r['members'], 'members_recomputed': ','.join(map(str, c)),
            'members_match': r['members'] == ','.join(map(str, c)),
            'mode_csv': r['mode'], 'mode_recomputed': m.label(prob),
            'mode_match': r['mode'] == m.label(prob),
            'loss_max_csv': float(r['loss_max']), 'loss_max_recomputed': max(loss),
            'loss_max_abs_err': abs(float(r['loss_max']) - max(loss)),
            'loss_sum_abs_err': abs(float(r['loss_sum']) - sum(loss)),
            'stalled_csv': int(r['stalled']), 'stalled_recomputed': int(rj.stalled),
            'perm_equivariant_tokens': bool(equivar),
            'perm_label_max_invariant': abs(max(loss) - max(loss_p)) < 1e-12,
            'perm_label_sum_invariant': abs(sum(loss) - sum(loss_p)) < 1e-12})
        print(checks[-1]['uid'], 'err=%.2e' % checks[-1]['loss_max_abs_err'],
              'equivar=%s' % equivar, flush=True)

    rep = {
        'n': len(checks),
        'sanity_1_all_match': all(
            c['members_match'] and c['mode_match'] and c['loss_max_abs_err'] < 1e-9
            and c['loss_sum_abs_err'] < 1e-9 and c['stalled_csv'] == c['stalled_recomputed']
            for c in checks),
        'max_loss_max_abs_err': max(c['loss_max_abs_err'] for c in checks),
        'sanity_2_equivariant': all(c['perm_equivariant_tokens'] for c in checks),
        'sanity_2_label_invariant': all(
            c['perm_label_max_invariant'] and c['perm_label_sum_invariant']
            for c in checks),
        'rows': checks}
    if a.out:
        json.dump(rep, open(a.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
