#!/usr/bin/env python3
"""S6 — A1/A2/A3 와 기준선의 top-1 regret 을 **짝지어** 비교한다.

게이트가 요구하는 것은 "무작위보다 **유의하게** 나을 것"이므로 평균만으로는
부족하다.  같은 클러스터 상태에서 두 선택자의 regret 차이를 짝지어 모으고,
클러스터 상태 단위 부트스트랩(10,000회)으로 95% 구간을 낸다.

    python3 learn/compare.py --split val
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learn.common import apply_norm, ckpt_path, load, split_of, to_dev
from learn.model import ModeScorer
from learn.evaluate import predict

OUT = 'results/s6'


def group_slices(gid):
    o = np.argsort(gid, kind='mergesort')
    b = np.flatnonzero(np.diff(gid[o])) + 1
    return o, np.split(np.arange(len(gid)), b)


def scores_for(tag, ev, tr_raw, dev):
    ck = torch.load(ckpt_path(tag), map_location=dev, weights_only=False)
    st = {k: np.array(v, np.float32) for k, v in ck['norm'].items()}
    e = apply_norm(ev, st)
    t = to_dev({k: e[k] for k in ('agent', 'agent_mask', 'obst', 'obst_mask', 'glob')}, dev)
    m = ModeScorer(e['agent'].shape[2], e['obst'].shape[2], e['glob'].shape[1],
                   n_cls=ck['n_cls']).to(dev)
    m.load_state_dict(ck['model'])
    lg, rg = predict(m, t)
    pred = (rg * ck['sd'] + ck['mu']).cpu().numpy()
    if ck['n_cls'] == 3:
        p = torch.softmax(lg, -1).cpu().numpy()
        l2 = float(tr_raw['loss'][tr_raw['cls3'] == 2].mean())
        return p[:, 1] * pred + p[:, 2] * l2
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val')
    ap.add_argument('--tags', nargs='+', default=['A1', 'A2', 'A3'])
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20260901)
    ap.add_argument('--out-json', default=None)
    ap.add_argument('--cache', default='learn/cache/s6.npz')
    ap.add_argument('--tag-cache', nargs='*', default=[],
                    help='tag=캐시경로 — 특성 인코딩이 다른 모델을 함께 비교할 때'
                         ' (S6-4: 원핫 캐시와 기하 캐시)')
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = load(a.cache)
    ev, tr_raw = split_of(d, a.split), split_of(d, 'train')
    gid, loss, midx = ev['gid'], ev['loss'], ev['mode_idx']
    o, groups = group_slices(gid)
    loss_s, midx_s = loss[o], midx[o]

    # 태그마다 다른 캐시를 쓸 수 있다.  두 캐시는 같은 CSV 를 같은 순서로 읽어
    # 만들어지므로 행이 1:1 로 맞는다 — 라벨이 같은지로 확인한다.
    override = dict(kv.split('=', 1) for kv in a.tag_cache)
    evs = {a.cache: (ev, tr_raw)}
    for path in set(override.values()):
        if path in evs:
            continue
        d2 = load(path)
        e2, t2 = split_of(d2, a.split), split_of(d2, 'train')
        assert np.array_equal(e2['loss'], ev['loss']) and \
               np.array_equal(e2['gid'], ev['gid']), f'{path} 의 행 순서가 다르다'
        evs[path] = (e2, t2)
    sel = {}
    for t in a.tags:
        e, tr = evs[override.get(t, a.cache)]
        sel[t] = scores_for(t, e, tr, dev)[o]
    per = {}                                    # 선택자 -> 그룹별 regret
    degen = np.array([bool(loss_s[g].max() == loss_s[g].min()) for g in groups])
    for t in a.tags:
        per[t] = np.array([float(loss_s[g][np.argmin(sel[t][g])] - loss_s[g].min())
                           for g in groups])
    per['random'] = np.array([float((loss_s[g] - loss_s[g].min()).mean())
                              for g in groups])
    per['canonical'] = np.array([
        float(loss_s[g][int(np.flatnonzero(midx_s[g] == 0)[0])] - loss_s[g].min())
        if (midx_s[g] == 0).any() else np.nan for g in groups])
    per['teacher'] = np.zeros(len(groups))

    rng = np.random.default_rng(a.seed)
    out = {'split': a.split, 'n_groups': len(groups),
           'degenerate_frac': float(degen.mean()), 'pairs': {}}
    for scope, keep in (('all', np.ones(len(groups), bool)), ('nondegen', ~degen)):
        pairs = {}
        names = list(a.tags) + ['random', 'canonical']
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                m = keep & np.isfinite(per[x]) & np.isfinite(per[y])
                dv = per[x][m] - per[y][m]
                idx = rng.integers(0, len(dv), size=(a.boot, len(dv)))
                bs = dv[idx].mean(1)
                pairs[f'{x}-{y}'] = dict(
                    n=int(m.sum()), diff=float(dv.mean()),
                    lo=float(np.percentile(bs, 2.5)),
                    hi=float(np.percentile(bs, 97.5)),
                    p_better=float((bs < 0).mean()))
        out['pairs'][scope] = pairs
        out.setdefault('means', {})[scope] = {
            k: float(np.nanmean(v[keep])) for k, v in per.items()}

    fn = a.out_json or f'{OUT}/compare_{a.split}.json'
    os.makedirs(os.path.dirname(fn) or '.', exist_ok=True)
    json.dump(out, open(fn, 'w'), indent=1)
    for scope in ('all', 'nondegen'):
        print(f'--- {scope} (n={int((~degen).sum()) if scope=="nondegen" else len(groups)}) ---')
        print('  평균 regret: ' + '  '.join(
            f'{k} {v:.3f}' for k, v in out['means'][scope].items()))
        for k, v in out['pairs'][scope].items():
            star = '유의' if v['hi'] < 0 or v['lo'] > 0 else '  -  '
            print(f"  {k:22s} 차 {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] {star}")


if __name__ == '__main__':
    main()
