#!/usr/bin/env python3
"""S6-5 — 기존 체크포인트의 예측을 **재학습 없이** 층화해 다시 본다.

    python3 scripts/diag_stratify_hard_easy.py --split val

층 (비퇴화 그룹만)
    easy  canonical(`mode_idx=0`)이 그 그룹 loss_max 최소값 집합(동점 포함)에 속한다
    hard  그렇지 않다 — canonical 이 최적이 아니다

S6-4 §8 은 "전원 route 0" 후보와 canonical 을 같은 것으로 놓았다.  둘은
사실 다른 조건이므로(전원 route 0 이면서 양보순서/플래그가 다른 후보가 있다)
세 기준을 따로 세어 §2 에서 대조한다.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learn.common import apply_norm, ckpt_path, load, split_of, to_dev
from learn.data import parse_mode
from learn.evaluate import predict
from learn.model import ModeScorer

OUT = 'results/s6_5'
CSV = 'results/labels/main_progress_dedup.csv'


def scores_and_probs(tag, cache, split, dev):
    """순위 점수와 3-클래스 확률 — evaluate.py 와 같은 식."""
    d = load(cache)
    ev, tr = split_of(d, split), split_of(d, 'train')
    ck = torch.load(ckpt_path(tag), map_location=dev, weights_only=False)
    st = {k: np.array(v, np.float32) for k, v in ck['norm'].items()}
    e = apply_norm(ev, st)
    t = to_dev({k: e[k] for k in ('agent', 'agent_mask', 'obst', 'obst_mask',
                                  'glob')}, dev)
    m = ModeScorer(e['agent'].shape[2], e['obst'].shape[2], e['glob'].shape[1],
                   n_cls=ck['n_cls']).to(dev)
    m.load_state_dict(ck['model'])
    lg, rg = predict(m, t)
    pred = (rg * ck['sd'] + ck['mu']).cpu().numpy()
    p = torch.softmax(lg, -1).cpu().numpy() if ck['n_cls'] == 3 else None
    if p is None:
        return pred, None
    l2 = float(tr['loss'][tr['cls3'] == 2].mean())
    return p[:, 1] * pred + p[:, 2] * l2, p


def boot(dv, rng, n=10000):
    if len(dv) == 0:
        return dict(n=0, diff=float('nan'), lo=float('nan'), hi=float('nan'))
    bs = dv[rng.integers(0, len(dv), size=(n, len(dv)))].mean(1)
    return dict(n=len(dv), diff=float(dv.mean()),
                lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)),
                sig=bool(np.percentile(bs, 97.5) < 0 or np.percentile(bs, 2.5) > 0))


def auc(y, s):
    """y(bool) 를 s 로 얼마나 가르는가 — Mann-Whitney U."""
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float('nan')
    r = np.empty(len(s), float)
    o = np.argsort(s, kind='mergesort')
    sv = np.asarray(s)[o]
    i = 0
    while i < len(sv):                                  # 동점은 평균 순위
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = int(y.sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val')
    ap.add_argument('--seed', type=int, default=20260901)
    ap.add_argument('--d-quantile', type=float, default=0.75,
                    help='규칙 D: canonical 의 P(정체) 가 이 분위수 위면 모델로')
    ap.add_argument('--d-threshold', type=float, default=None,
                    help='주면 그 절대 문턱을 쓴다 (val 에서 정한 값을 test 에)')
    ap.add_argument('--e-threshold', type=float, default=None,
                    help='규칙 E 의 여유폭 delta (m).  주지 않으면 이 split 에서 최적값을 훑는다')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    geo, p_geo = scores_and_probs('geo', 'learn/cache/s6_4.npz', a.split, dev)
    rnk, _ = scores_and_probs('ranking', 'learn/cache/s6.npz', a.split, dev)

    d = load('learn/cache/s6_4.npz')
    ev = split_of(d, a.split)
    gid, L, midx = ev['gid'], ev['loss'], ev['mode_idx']

    # --- CSV 와 행이 1:1 인지 확인하고 "구성원 전원 route 0" 플래그를 만든다 ---
    rows = list(csv.DictReader(open(CSV, newline='', encoding='utf8')))
    keep = [i for i, r in enumerate(rows) if r['split'] == a.split]
    rows = [rows[i] for i in keep]
    assert len(rows) == len(L), '행 수가 다르다'
    assert np.allclose([float(r['loss_max']) for r in rows], L, atol=1e-5), \
        'CSV 와 캐시의 행 순서가 다르다'
    all_r0 = np.zeros(len(rows), bool)
    for i, r in enumerate(rows):
        ro, _, _, _ = parse_mode(r['mode'], int(r['n_agents']))
        all_r0[i] = all(ro[j] == 0 for j in map(int, r['members'].split(',')))

    order = np.argsort(gid, kind='mergesort')
    bnd = np.flatnonzero(np.diff(gid[order])) + 1
    groups = [g for g in np.split(order, bnd)]

    rec = []
    for g in groups:
        Lg = L[g]
        lo = float(Lg.min())
        if Lg.max() == lo:
            continue                                   # 퇴화 그룹 제외
        ci = np.flatnonzero(midx[g] == 0)
        if len(ci) == 0:
            continue                                   # canonical 이 없는 그룹
        c = int(ci[0])
        mins = Lg == lo
        sg = geo[g]
        rec.append(dict(
            idx=g, lo=lo,
            canon_margin=float(sg[c] - sg.min()),   # 모델이 본 canonical 의 열세폭
            canon_regret=float(Lg[c] - lo),
            geo_regret=float(Lg[int(np.argmin(geo[g]))] - lo),
            rnk_regret=float(Lg[int(np.argmin(rnk[g]))] - lo),
            rand_regret=float((Lg - lo).mean()),
            easy_canon=bool(mins[c]),
            easy_r0=bool((mins & all_r0[g]).any()),
            has_r0=bool(all_r0[g].any()),
            has_non_r0=bool((~all_r0[g]).any()),
            canon_pstall=float(p_geo[g][c, 2]) if p_geo is not None else 0.0,
            k=int(len(g)), spread=float(Lg.max() - lo),
            geo_pick=int(g[int(np.argmin(geo[g]))]), canon_row=int(g[c]),
        ))
    n = len(rec)
    easy = np.array([r['easy_canon'] for r in rec])
    arr = lambda k: np.array([r[k] for r in rec])

    res = dict(split=a.split, n_nondegen_groups=n,
               criteria=dict(
                   easy_canonical_is_min=float(easy.mean()),
                   easy_some_all_route0_is_min=float(arr('easy_r0').mean()),
                   group_has_all_route0_candidate=float(arr('has_r0').mean()),
                   group_has_non_all_route0_candidate=float(arr('has_non_r0').mean())),
               layers={})
    for name, sel in (('easy', easy), ('hard', ~easy)):
        e = dict(n_groups=int(sel.sum()), frac=float(sel.mean()))
        for who, k in (('geo', 'geo_regret'), ('ranking', 'rnk_regret'),
                       ('canonical', 'canon_regret'), ('random', 'rand_regret')):
            e[who] = float(arr(k)[sel].mean()) if sel.any() else float('nan')
        e['canonical_regret_max'] = float(arr('canon_regret')[sel].max()) if sel.any() else 0.0
        e['canonical_regret_nonzero_groups'] = int((arr('canon_regret')[sel] > 0).sum()) if sel.any() else 0
        res['layers'][name] = e
    # easy 를 §8 기준(전원 route 0)으로 잡았을 때 canonical regret 이 0 인가
    r0 = arr('easy_r0')
    res['gate1_check_r0_layer'] = dict(
        n_groups=int(r0.sum()),
        canonical_regret_mean=float(arr('canon_regret')[r0].mean()),
        canonical_regret_max=float(arr('canon_regret')[r0].max()),
        nonzero_groups=int((arr('canon_regret')[r0] > 0).sum()))

    hard = ~easy
    res['bootstrap_hard'] = {
        'geo-canonical': boot(arr('geo_regret')[hard] - arr('canon_regret')[hard], rng),
        'geo-random': boot(arr('geo_regret')[hard] - arr('rand_regret')[hard], rng),
        'ranking-canonical': boot(arr('rnk_regret')[hard] - arr('canon_regret')[hard], rng),
        'ranking-random': boot(arr('rnk_regret')[hard] - arr('rand_regret')[hard], rng),
        'geo-ranking': boot(arr('geo_regret')[hard] - arr('rnk_regret')[hard], rng)}
    # ---- 검산: 층을 canonical 의 라벨로 정의했으므로 canonical 에게 불리하다.
    #      층을 "모델이 틀린 그룹"으로 뒤집으면 같은 크기의 이점이 canonical 쪽으로 간다.
    hard_m = arr('geo_regret') > 0
    res['mirror_check'] = dict(
        note=('층 정의를 모델 기준으로 뒤집은 대조 — 게이트 2 의 우세가 '
              '층화가 만든 것인지 보기 위한 것'),
        hard_model_frac=float(hard_m.mean()),
        geo=float(arr('geo_regret')[hard_m].mean()),
        canonical=float(arr('canon_regret')[hard_m].mean()),
        random=float(arr('rand_regret')[hard_m].mean()),
        canonical_minus_geo=boot(arr('canon_regret')[hard_m] - arr('geo_regret')[hard_m], rng))
    json.dump(res, open(f'{OUT}/stratified_regret_{a.split}.json', 'w'), indent=1)

    # ------------------------------------------------------------------ 규칙
    ps = arr('canon_pstall')
    thr = a.d_threshold if a.d_threshold is not None else float(
        np.quantile(ps, a.d_quantile))
    use_model_C = arr('has_non_r0')          # 규칙 C 의 조건 (지시서 문자 그대로)
    use_model_D = ps > thr
    rules = {}
    for nm, use in (('A_canonical', np.zeros(n, bool)),
                    ('B_model', np.ones(n, bool)),
                    ('C_hybrid_route0', use_model_C),
                    ('D_hybrid_pstall', use_model_D)):
        reg = np.where(use, arr('geo_regret'), arr('canon_regret'))
        rules[nm] = dict(mean=float(reg.mean()),
                         model_used_frac=float(use.mean()),
                         vs_canonical=boot(reg - arr('canon_regret'), rng))
    # --- 규칙 E (참고) — 모델이 "canonical 이 최선보다 delta 이상 나쁘다"고 볼 때만 교체.
    #     라벨을 쓰지 않으므로 실제로 굴릴 수 있는 규칙이다.
    cm = arr('canon_margin')
    cand = [0.0] + [float(np.quantile(cm, q)) for q in
                    (.1, .2, .3, .4, .5, .6, .7, .8, .9)]
    e_curve = {}
    for dlt in cand:
        u = cm > dlt
        e_curve[f'{dlt:.4f}'] = dict(
            mean=float(np.where(u, arr('geo_regret'), arr('canon_regret')).mean()),
            model_used_frac=float(u.mean()))
    e_thr = (a.e_threshold if a.e_threshold is not None
             else float(min(cand, key=lambda x: e_curve[f'{x:.4f}']['mean'])))
    rules['E_hybrid_margin'] = None          # 아래에서 채운다
    for nm, use in (('E_hybrid_margin', cm > e_thr),
                    ('F_oracle_layer', ~easy)):
        reg = np.where(use, arr('geo_regret'), arr('canon_regret'))
        rules[nm] = dict(mean=float(reg.mean()),
                         model_used_frac=float(use.mean()),
                         vs_canonical=boot(reg - arr('canon_regret'), rng))
    rules['E_hybrid_margin']['threshold'] = e_thr
    rules['F_oracle_layer']['note'] = ('라벨을 봐야 계산되는 상한 — 실행 가능한 '
                                       '규칙이 아니다')

    sweep = {}
    for q in (0.10, 0.25, 0.50, 0.75):
        u = ps > float(np.quantile(ps, 1 - q))
        reg = np.where(u, arr('geo_regret'), arr('canon_regret'))
        sweep[f'top_{int(q*100)}pct'] = dict(mean=float(reg.mean()),
                                             threshold=float(np.quantile(ps, 1 - q)))
    json.dump(dict(split=a.split, n_groups=n, d_threshold=thr, e_threshold=e_thr,
                   rules=rules, d_sweep=sweep, e_curve=e_curve),
              open(f'{OUT}/hybrid_rules_{a.split}.json', 'w'), indent=1)

    # ---- 층(hard)을 라벨 없이 알아볼 수 있는가 (규칙 F 가 실행 가능한지의 전제)
    pf = f'{OUT}/layer_predictability.json'
    lp = json.load(open(pf)) if os.path.exists(pf) else {}
    lp[a.split] = dict(
        n=n, hard_frac=float(hard.mean()),
        auc_model_margin=auc(hard, cm),
        auc_canon_pstall=auc(hard, ps),
        auc_k=auc(hard, arr('k')),
        auc_group_spread_ORACLE=auc(hard, arr('spread')))
    json.dump({k: lp[k] for k in ('val', 'test') if k in lp}, open(pf, 'w'), indent=1)

    # 규칙 C 가 canonical 과 다른 선택을 한 사례
    ex = []
    for i, r in enumerate(rec):
        if not use_model_C[i] or r['geo_pick'] == r['canon_row']:
            continue
        ex.append(dict(split=a.split, uid=rows[r['canon_row']]['uid'],
                       replan_idx=int(rows[r['canon_row']]['replan_idx']),
                       cluster_id=int(rows[r['canon_row']]['cluster_id']),
                       model_mode=rows[r['geo_pick']]['mode'],
                       canonical_mode=rows[r['canon_row']]['mode'],
                       model_loss=float(L[r['geo_pick']]),
                       canonical_loss=float(L[r['canon_row']]),
                       group_min=r['lo'],
                       delta=float(L[r['geo_pick']] - L[r['canon_row']])))
    ex.sort(key=lambda x: -abs(x['delta']))
    return res, rules, sweep, thr, ex


if __name__ == '__main__':
    res, rules, sweep, thr, ex = main()
    print(json.dumps(dict(criteria=res['criteria'], layers=res['layers'],
                          gate1=res['gate1_check_r0_layer'],
                          boot=res['bootstrap_hard'],
                          rules={k: (round(v['mean'], 4), round(v['model_used_frac'], 3),
                                     round(v['vs_canonical']['lo'], 4),
                                     round(v['vs_canonical']['hi'], 4))
                                 for k, v in rules.items()},
                          d_threshold=thr, sweep=sweep,
                          e_threshold=rules['E_hybrid_margin']['threshold']),
                     indent=1, ensure_ascii=False))
    ef = f'{OUT}/hybrid_examples.json'
    allex = json.load(open(ef)) if os.path.exists(ef) else {}
    allex[res['split']] = ex[:20]
    json.dump({k: allex[k] for k in ('val', 'test') if k in allex},
              open(ef, 'w'), indent=1, ensure_ascii=False)
    print(f'규칙 C 가 canonical 과 다르게 고른 그룹 {len(ex)}건')
