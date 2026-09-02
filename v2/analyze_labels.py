#!/usr/bin/env python3
"""S5-3E 라벨 분석: C 게이트 판정, 중복 제거, uid split, 새너티 6종."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GATE_POS_FRAC = 0.60
GATE_CLS_LO, GATE_CLS_HI = 0.05, 0.95


def load(path):
    return list(csv.DictReader(open(path)))


def q(v, name, out):
    v = np.asarray(v, float)
    out[name] = {'n': int(v.size), 'min': float(v.min()), 'p10': float(np.percentile(v, 10)),
                 'p50': float(np.median(v)), 'p90': float(np.percentile(v, 90)),
                 'max': float(v.max()), 'mean': float(v.mean()),
                 'pos': float((v > 1e-9).mean()), 'neg': float((v < -1e-9).mean()),
                 'zero': float((np.abs(v) <= 1e-9).mean())}
    return out[name]


def uid_split(uids, frac=(0.70, 0.15, 0.15)):
    """**uid 단위** 분할.  행 단위 무작위 분할은 금지되어 있다 (D).

    uid 해시로 순서를 고정한 뒤 행 수를 채워 가며 배정한다.  순수 해시 분할은
    uid 마다 행 수가 3배까지 차이 나서 목표 비율이 크게 어긋난다 (시범에서
    70/15/15 를 노렸는데 34/45/20 이 나왔다).  순서가 해시로 정해지므로
    재실행해도 같은 결과이고, 한 uid 의 행은 반드시 한 split 에만 들어간다.
    """
    cnt = {}
    for u in uids:
        cnt[u] = cnt.get(u, 0) + 1
    order = sorted(cnt, key=lambda u: hashlib.sha1(u.encode()).hexdigest())
    total = sum(cnt.values())
    target = {'train': frac[0] * total, 'val': frac[1] * total, 'test': frac[2] * total}
    have = {'train': 0, 'val': 0, 'test': 0}
    out = {}
    for u in order:
        s = min(target, key=lambda k: (have[k] + cnt[u]) / max(target[k], 1e-9))
        out[u] = s
        have[s] += cnt[u]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--split-out', default=None)
    a = p.parse_args()
    R = load(a.csv)
    rep = {'path': a.csv, 'rows': len(R)}

    lmax = np.array([float(r['loss_max']) for r in R])
    lsum = np.array([float(r['loss_sum']) for r in R])
    lmin = np.array([float(r['loss_min']) for r in R])
    per = np.concatenate([np.array(r['losses'].split(','), float) for r in R])
    stalled = np.array([int(r['stalled']) for r in R])
    blocked = np.array([int(r['blocked']) for r in R])
    q(lmax, 'loss_max', rep); q(lsum, 'loss_sum', rep)
    q(lmin, 'loss_min', rep); q(per, 'loss_per_agent', rep)
    q(np.array([float(r['prog_together_sum']) for r in R]), 'prog_together_sum', rep)
    q(np.array([float(r['prog_alone_sum']) for r in R]), 'prog_alone_sum', rep)
    rep['corr_sum_max'] = float(np.corrcoef(lsum, lmax)[0, 1])
    rep['stalled_frac'] = float(stalled.mean())
    rep['blocked_frac'] = float(blocked.mean())
    rep['n_distinct_loss_max'] = int(len(set(np.round(lmax, 4))))
    rep['dwell_alone_sum'] = int(sum(int(r['dwell_alone']) for r in R))
    rep['dwell_together_sum'] = int(sum(int(r['dwell_together']) for r in R))

    # 후보 간 loss 분산 (클러스터별)
    byc = {}
    for r in R:
        byc.setdefault((r['uid'], r['replan_idx'], r['cluster_id']), []).append(float(r['loss_max']))
    sds = np.array([np.std(v) for v in byc.values()])
    rep['cluster_states'] = len(byc)
    q(sds, 'per_cluster_loss_std', rep)
    rep['cluster_std_zero_frac'] = float((sds < 1e-9).mean())

    # k / phase 층화
    rep['by_k'] = {}
    for k in sorted({r['k'] for r in R}):
        v = lmax[[i for i, r in enumerate(R) if r['k'] == k]]
        rep['by_k'][k] = {'n': int(v.size), 'p50': float(np.median(v)),
                          'pos': float((v > 1e-9).mean())}

    # 중복 제거 (같은 클러스터 구성·후보)
    seen, keep = set(), []
    for i, r in enumerate(R):
        if r['dedup_key'] in seen:
            continue
        seen.add(r['dedup_key']); keep.append(i)
    rep['rows_after_dedup'] = len(keep)
    rep['dup_frac'] = 1.0 - len(keep) / len(R)

    # ---- 게이트 ---------------------------------------------------------- #
    g = {}
    g['1_loss_pos_frac>=0.60'] = (float((lmax > 1e-9).mean()), float((lmax > 1e-9).mean()) >= GATE_POS_FRAC)
    g['2_nondegenerate'] = (rep['n_distinct_loss_max'], rep['n_distinct_loss_max'] > 20)
    g['3_cls_stalled_5_95'] = (rep['stalled_frac'], GATE_CLS_LO <= rep['stalled_frac'] <= GATE_CLS_HI)
    g['3b_cls_blocked_5_95'] = (rep['blocked_frac'], GATE_CLS_LO <= rep['blocked_frac'] <= GATE_CLS_HI)
    g['4_rows>=1900'] = (len(R), len(R) >= 1900)
    g['5_candidate_variance'] = (float(np.median(sds)), float(np.median(sds)) > 1e-9)
    rep['gates'] = {k: {'value': v, 'pass': bool(ok)} for k, (v, ok) in g.items()}
    rep['gate_all_pass'] = all(ok for _, ok in g.values())

    # ---- 새너티 (일부는 여기서, 나머지는 별도 스크립트) ------------------- #
    sp = uid_split([r['uid'] for r in R])
    rows_split = [sp[r['uid']] for r in R]
    rep['split_rows'] = {s: rows_split.count(s) for s in ('train', 'val', 'test')}
    rep['split_uids'] = {s: len({u for u, v in sp.items() if v == s}) for s in ('train', 'val', 'test')}
    sets = {s: {u for u, v in sp.items() if v == s} for s in ('train', 'val', 'test')}
    rep['sanity_4_uid_overlap'] = {
        'train&val': len(sets['train'] & sets['val']),
        'train&test': len(sets['train'] & sets['test']),
        'val&test': len(sets['val'] & sets['test'])}
    # 정규화 후 범위 (z-score, train 통계)
    tr = lmax[[i for i, s in enumerate(rows_split) if s == 'train']]
    if tr.size > 1 and tr.std() > 0:
        z = (lmax - tr.mean()) / tr.std()
        rep['sanity_3_zrange'] = {'min': float(z.min()), 'max': float(z.max()),
                                  'abs>5': int((np.abs(z) > 5).sum())}
    # 토큰 값 범위
    tok = np.concatenate([np.array(json.loads(r['agent_tokens']), float).ravel() for r in R[:2000]])
    rep['sanity_3_token_absmax'] = float(np.abs(tok).max())

    if a.split_out:
        with open(a.split_out, 'w', newline='', encoding='utf8') as f:
            w = csv.writer(f); w.writerow(['uid', 'split'])
            for u, s in sorted(sp.items()):
                w.writerow([u, s])
    if a.out:
        json.dump(rep, open(a.out, 'w'), indent=1)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
