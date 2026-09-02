#!/usr/bin/env python3
"""S7 라벨 분석: A-1b 정확 분해 + C 게이트 5종 + 층 분포.

    python3 scripts/analyze_s7_labels.py --csv results/labels/s7/pilot40.csv \
        --out results/s7/pilot40_report.json

A-1b (정확 가법 분해, route 0 자 위에서)
    old_loss_i = loss_route_i + loss_avoid0_i
        loss_route_i  = A0_i  - Ar0_i   경로 선택으로 잃은 진행량 (혼자일 때)
        loss_avoid0_i = Ar0_i - T0_i    같은 경로에서 남을 피하느라 잃은 진행량

C 게이트 (지시서 C-2)
    1 canonical 의 loss_avoid 가 0 이 아닌 비율 >= 50 %
    2 교사 최적이 canonical 인 비율 < 40 %
    3 loss_avoid 양수 비율 >= 60 %
    4 분포 비퇴화
    5 퇴화 그룹 비율 보고
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learn.data import parse_mode
from scripts.analyze_labels import q, uid_split

G1_CANON_NONZERO = 0.50
G2_TEACHER_CANON = 0.40
G3_POS_FRAC = 0.60


def vshare(y, a, b):
    """가법 분해 y = a + b 의 분산 비중.  공분산은 절반씩 나눠 배분한다."""
    y, a, b = map(np.asarray, (y, a, b))
    vy = float(y.var())
    if vy <= 0:
        return dict(var_total=0.0, share_a=float('nan'), share_b=float('nan'))
    va, vb = float(a.var()), float(b.var())
    cov = float(np.cov(a, b, bias=True)[0, 1])
    return dict(var_total=vy, var_a=va, var_b=vb, cov=cov,
                share_a=(va + cov) / vy, share_b=(vb + cov) / vy,
                share_a_novcov=va / vy, share_b_novcov=vb / vy,
                dominance_a_frac=float((np.abs(a) > np.abs(b)).mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--split-out', default=None)
    a = p.parse_args()
    R = list(csv.DictReader(open(a.csv)))
    rep = {'path': a.csv, 'rows': len(R)}
    F = lambda k: np.array([float(r[k]) for r in R])
    per = lambda k: np.concatenate([np.array(r[k].split(','), float) for r in R])

    new, old = F('loss_max'), F('old_loss_max')
    q(new, 'loss_avoid_max(NEW)', rep)
    q(old, 'old_loss_max(S5-3E)', rep)
    q(F('loss_sum'), 'loss_avoid_sum', rep)
    q(F('route_cost_max'), 'route_cost_max', rep)
    q(F('loss_route_max'), 'loss_route_max', rep)
    q(F('loss_avoid0_max'), 'loss_avoid0_max', rep)
    q(per('losses'), 'loss_avoid_per_agent', rep)
    q(F('prog_alone_sum'), 'prog_alone_sum', rep)
    q(F('prog_together_sum'), 'prog_together_sum', rep)
    rep['corr_new_old'] = float(np.corrcoef(new, old)[0, 1])
    rep['stalled_frac'] = float(F('stalled').mean())
    rep['blocked_frac'] = float(F('blocked').mean())
    rep['n_distinct_loss'] = int(len(set(np.round(new, 4))))
    rep['n_distinct_old'] = int(len(set(np.round(old, 4))))

    # ---- A-1b 정확 가법 분해 -------------------------------------------- #
    ol, lr, la = per('old_losses'), per('loss_routes'), per('loss_avoid0s')
    rep['A1b_decomp_member'] = vshare(ol, lr, la)
    rep['A1b_decomp_member']['max_additivity_err'] = float(np.abs(ol - lr - la).max())
    q(lr, 'A1b_loss_route_member', rep); q(la, 'A1b_loss_avoid0_member', rep)

    # 그룹 내(후보 간) 변동만 — 모델이 실제로 골라야 하는 축
    gi = defaultdict(list)
    for i, r in enumerate(R):
        gi[(r['uid'], r['replan_idx'], r['cluster_id'])].append(i)
    cy, ca, cb = [], [], []
    for idx in gi.values():
        if len(idx) < 2:
            continue
        y, u, v = old[idx], F('loss_route_max')[idx], F('loss_avoid0_max')[idx]
        if y.max() == y.min():
            continue
        cy.append(y - y.mean()); ca.append(u - u.mean()); cb.append(v - v.mean())
    if cy:
        rep['A1b_decomp_within_group'] = vshare(np.concatenate(cy),
                                                np.concatenate(ca),
                                                np.concatenate(cb))
        rep['A1b_decomp_within_group']['n_groups'] = len(cy)

    # 검산: 전원 route 0 후보에서 loss_route 와 route_cost 가 0 인가
    allr0 = np.array([all(parse_mode(r['mode'], int(r['n_agents']))[0][j] == 0
                          for j in map(int, r['members'].split(',')))
                      for r in R])
    rep['A1b_sanity_all_route0'] = dict(
        n=int(allr0.sum()), frac=float(allr0.mean()),
        max_abs_loss_route=float(np.abs(F('loss_route_max')[allr0]).max()) if allr0.any() else 0.0,
        max_abs_route_cost=float(np.abs(F('route_cost_max')[allr0]).max()) if allr0.any() else 0.0,
        max_abs_new_minus_old=float(np.abs(new[allr0] - old[allr0]).max()) if allr0.any() else 0.0)

    # ---- 그룹 단위: canonical·교사 최적 --------------------------------- #
    canon_new, canon_old, teach_new, teach_old = [], [], [], []
    degen_new, degen_old, tie_new = [], [], []
    tmode_new, tmode_old = Counter(), Counter()
    for key, idx in gi.items():
        mi = np.array([int(R[i]['mode_idx']) for i in idx])
        ci = np.flatnonzero(mi == 0)
        yn, yo = new[idx], old[idx]
        degen_new.append(yn.max() == yn.min()); degen_old.append(yo.max() == yo.min())
        if len(ci) == 0:
            continue
        c = int(ci[0])
        canon_new.append(yn[c]); canon_old.append(yo[c])
        if yn.max() != yn.min():
            teach_new.append(bool(yn[c] == yn.min()))
            tmode_new[int(mi[int(np.argmin(yn))])] += 1
            tie_new.append(int((yn == yn.min()).sum()))
        if yo.max() != yo.min():
            teach_old.append(bool(yo[c] == yo.min()))
            tmode_old[int(mi[int(np.argmin(yo))])] += 1
    cn, co = np.array(canon_new), np.array(canon_old)
    rep['groups'] = dict(
        n=len(gi),
        degenerate_frac_new=float(np.mean(degen_new)),
        degenerate_frac_old=float(np.mean(degen_old)),
        n_nondegenerate_new=int(len(teach_new)),
        n_nondegenerate_old=int(len(teach_old)),
        teacher_is_canonical_new=float(np.mean(teach_new)) if teach_new else float('nan'),
        teacher_is_canonical_old=float(np.mean(teach_old)) if teach_old else float('nan'),
        teacher_mode_idx_top10_new=tmode_new.most_common(10),
        teacher_mode_idx_top10_old=tmode_old.most_common(10),
        tie_size_mean_new=float(np.mean(tie_new)) if tie_new else float('nan'))
    q(cn, 'canonical_loss_avoid(NEW)', rep)
    q(co, 'canonical_old_loss(S5-3E)', rep)
    rep['canonical_nonzero_frac_new'] = float((np.abs(cn) > 1e-9).mean())
    rep['canonical_nonzero_frac_old'] = float((np.abs(co) > 1e-9).mean())

    # ---- route 0 편향의 직접 측정 (S6-4 §8 의 거울상) -------------------- #
    #  "임의 후보가 전원 route 0" 확률 대비 "최소값 집합에 전원 route 0 후보가
    #  있을" 확률.  옛 라벨에서 3.8 배 과대표집이었다.
    bias = {}
    for tag, y in (('new', new), ('old', old)):
        base_rate, min_rate, n_g = [], [], 0
        for idx in gi.values():
            yy = y[idx]
            if len(idx) < 2 or yy.max() == yy.min():
                continue
            n_g += 1
            r0 = allr0[idx]
            base_rate.append(float(r0.mean()))
            min_rate.append(bool((r0 & (yy == yy.min())).any()))
        b, mn = float(np.mean(base_rate)), float(np.mean(min_rate))
        bias[tag] = dict(n_groups=n_g, p_candidate_all_route0=b,
                         p_min_has_all_route0=mn,
                         oversampling_ratio=mn / b if b > 0 else float('nan'))
    rep['route0_bias'] = bias

    # 새 라벨이 거꾸로 긴 우회를 favor 하는가 — route_cost 와의 관계
    rc = F('route_cost_max')
    rep['route_cost_vs_loss'] = dict(
        corr_new=float(np.corrcoef(rc, new)[0, 1]),
        corr_old=float(np.corrcoef(rc, old)[0, 1]),
        teacher_route_cost_mean_new=float(np.mean(
            [F('route_cost_max')[idx][int(np.argmin(new[idx]))]
             for idx in gi.values() if len(idx) > 1 and new[idx].max() != new[idx].min()])),
        teacher_route_cost_mean_old=float(np.mean(
            [F('route_cost_max')[idx][int(np.argmin(old[idx]))]
             for idx in gi.values() if len(idx) > 1 and old[idx].max() != old[idx].min()])),
        all_candidates_route_cost_mean=float(rc.mean()))

    # 후보 간 분산
    sds = np.array([np.std(new[idx]) for idx in gi.values()])
    q(sds, 'per_cluster_loss_std', rep)
    rep['cluster_std_zero_frac'] = float((sds < 1e-9).mean())

    # k / phase 층화
    for f in ('k', 'phase_sig'):
        rep[f'by_{f}'] = {}
        for v in sorted({r[f] for r in R}):
            s = np.array([r[f] == v for r in R])
            rep[f'by_{f}'][v] = dict(n=int(s.sum()), p50=float(np.median(new[s])),
                                     pos=float((new[s] > 1e-9).mean()),
                                     old_pos=float((old[s] > 1e-9).mean()))

    # 중복
    seen, keep = set(), []
    for i, r in enumerate(R):
        if r['dedup_key'] in seen:
            continue
        seen.add(r['dedup_key']); keep.append(i)
    rep['rows_after_dedup'] = len(keep)
    rep['dup_frac'] = 1.0 - len(keep) / len(R)

    # ---- C 게이트 ------------------------------------------------------- #
    g = {}
    v = rep['canonical_nonzero_frac_new']
    g['C1_canonical_loss_avoid_nonzero>=0.50'] = (v, v >= G1_CANON_NONZERO)
    v = rep['groups']['teacher_is_canonical_new']
    g['C2_teacher_is_canonical<0.40'] = (v, v < G2_TEACHER_CANON)
    v = float((new > 1e-9).mean())
    g['C3_loss_pos_frac>=0.60'] = (v, v >= G3_POS_FRAC)
    v = rep['n_distinct_loss']
    g['C4_nondegenerate'] = (v, v > 20)
    g['C5_degenerate_group_frac(report only)'] = (rep['groups']['degenerate_frac_new'], True)
    rep['gates'] = {k: {'value': x, 'pass': bool(ok)} for k, (x, ok) in g.items()}
    rep['gate_all_pass'] = all(ok for _, ok in g.values())

    # split (본 수집용)
    sp = uid_split([r['uid'] for r in R])
    rows_split = [sp[r['uid']] for r in R]
    rep['split_rows'] = {s: rows_split.count(s) for s in ('train', 'val', 'test')}
    rep['split_uids'] = {s: len({u for u, x in sp.items() if x == s})
                         for s in ('train', 'val', 'test')}
    sets = {s: {u for u, x in sp.items() if x == s} for s in ('train', 'val', 'test')}
    rep['sanity_4_uid_overlap'] = {'train&val': len(sets['train'] & sets['val']),
                                   'train&test': len(sets['train'] & sets['test']),
                                   'val&test': len(sets['val'] & sets['test'])}
    # D 추가 요구: split 간 라벨 분포
    rep['split_label_dist'] = {}
    for s in ('train', 'val', 'test'):
        m = np.array([x == s for x in rows_split])
        if m.any():
            rep['split_label_dist'][s] = dict(
                n=int(m.sum()), pos=float((new[m] > 1e-9).mean()),
                p50=float(np.median(new[m])), mean=float(new[m].mean()),
                stalled=float(F('stalled')[m].mean()))
    tr = new[[i for i, s in enumerate(rows_split) if s == 'train']]
    if tr.size > 1 and tr.std() > 0:
        z = (new - tr.mean()) / tr.std()
        rep['sanity_3_zrange'] = {'min': float(z.min()), 'max': float(z.max()),
                                  'abs>5': int((np.abs(z) > 5).sum())}
    tok = np.concatenate([np.array(json.loads(r['agent_tokens']), float).ravel()
                          for r in R[:2000]])
    rep['sanity_3_token_absmax'] = float(np.abs(tok).max())

    if a.split_out:
        with open(a.split_out, 'w', newline='', encoding='utf8') as f:
            w = csv.writer(f); w.writerow(['uid', 'split'])
            for u, s in sorted(sp.items()):
                w.writerow([u, s])
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        json.dump(rep, open(a.out, 'w'), indent=1)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
