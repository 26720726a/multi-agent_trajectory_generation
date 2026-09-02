#!/usr/bin/env python3
"""S8 C — 표본 조건 S1/S2/S3 의 top-1 regret 을 같은 행 위에서 비교한다.

    python3 scripts/eval_s8.py --split val

모든 모델의 점수를 **전체 캐시(`learn/cache/s6_4.npz`)의 val/test 전 행**에서
계산한 뒤, 그룹 부분집합만 바꿔 비교한다.  그래야 세 모델이 같은 자 위에 선다.

l2 (순위 점수의 정체 항)는 **모델마다 학습에 쓴 train 에서** 가져온다 —
S1 은 전체 train, S2·S3 은 필터된 train.  이걸 평가 캐시에서 다시 계산하면
모델이 학습 때 쓴 값과 달라져 점수가 틀어진다.

그룹 부분집합
    all       전체
    active    그룹 loss 최소값 > 0        <- S8 의 필터 (지시서의 "비퇴화")
    nondegen  그룹 max != min             <- S6-5·evaluate.py 의 정의
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learn.common import apply_norm, ckpt_path, load, split_of, to_dev
from learn.evaluate import auc, predict, within_group_rank
from learn.model import ModeScorer

OUT = 'results/s8'
FULL = 'learn/cache/s6_4.npz'
#: 태그 -> (체크포인트 태그, l2 를 가져올 캐시).  S3' 는 S2 의 체크포인트를
#: 전체 표본에서 평가하는 것 — "학습만 필터"의 실질이다 (S3 는 조기 종료가
#: ep0 을 골라 사실상 미학습이다).
MODELS = {'S1': ('S1', FULL),
          'S2': ('S2', 'learn/cache/s8_S2.npz'),
          'S3': ('S3', 'learn/cache/s8_S3.npz'),
          "S3p": ('S2', 'learn/cache/s8_S2.npz')}


def score_of(tag, l2_cache, ev, dev):
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
    if ck['n_cls'] != 3:
        return pred, None
    p = torch.softmax(lg, -1).cpu().numpy()
    tr = split_of(load(l2_cache), 'train')
    l2 = float(tr['loss'][tr['cls3'] == 2].mean())
    return p[:, 1] * pred + p[:, 2] * l2, p


def boot(dv, rng, n=10000):
    if len(dv) == 0:
        return dict(n=0, diff=float('nan'), lo=float('nan'), hi=float('nan'),
                    sig=False)
    bs = dv[rng.integers(0, len(dv), size=(n, len(dv)))].mean(1)
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return dict(n=len(dv), diff=float(dv.mean()), lo=lo, hi=hi,
                sig=bool(hi < 0 or lo > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val')
    ap.add_argument('--seed', type=int, default=20260901)
    ap.add_argument('--boot', type=int, default=10000)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    d = load(FULL)
    ev = split_of(d, a.split)
    gid, L, midx = ev['gid'], ev['loss'], ev['mode_idx']
    SP = {t: score_of(tg, c, ev, dev) for t, (tg, c) in MODELS.items()}
    S = {t: v[0] for t, v in SP.items()}
    P3 = {t: v[1] for t, v in SP.items()}

    o = np.argsort(gid, kind='mergesort')
    groups = np.split(o, np.flatnonzero(np.diff(gid[o])) + 1)

    rec = []
    for g in groups:
        Lg = L[g]; lo = float(Lg.min())
        ci = np.flatnonzero(midx[g] == 0)
        r = dict(n=len(g), active=bool(lo > 0), nondegen=bool(Lg.max() != lo),
                 k=int(ev['k'][g[0]]), phase=str(ev['phase_key'][g[0]]),
                 dep=str(ev['dep'][g[0]]),
                 rows=g,
                 canonical=float(Lg[int(ci[0])] - lo) if len(ci) else float('nan'),
                 random=float((Lg - lo).mean()), teacher=0.0)
        for t in MODELS:
            j = int(np.argmin(S[t][g]))
            r[t] = float(Lg[j] - lo)
            r[f'{t}_picks_canonical'] = bool(len(ci) and j == int(ci[0]))
        rec.append(r)
    arr = lambda key: np.array([x[key] for x in rec], float)
    sel = {'all': np.ones(len(rec), bool),
           'active': np.array([x['active'] for x in rec]),
           'nondegen': np.array([x['nondegen'] for x in rec])}
    fin = np.isfinite(arr('canonical'))
    names = list(MODELS) + ['canonical', 'random', 'teacher']

    res = dict(split=a.split, n_groups=len(rec),
               frac={k: float(v.mean()) for k, v in sel.items()},
               n_by_scope={k: int(v.sum()) for k, v in sel.items()},
               groups_without_canonical=int((~fin).sum()))

    # ---- 범위별 평균 regret 과 짝지은 부트스트랩 --------------------------
    res['means'], res['pairs'] = {}, {}
    for scope, m0 in sel.items():
        m = m0 & fin
        res['means'][scope] = {k: float(arr(k)[m].mean()) for k in names}
        res['means'][scope]['n'] = int(m.sum())
        p = {}
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                if 'teacher' in (x, y) and x != 'teacher':
                    continue
                p[f'{x}-{y}'] = boot(arr(x)[m] - arr(y)[m], rng, a.boot)
        res['pairs'][scope] = p

    # ---- C-5 분류 AUC 와 회귀 RMSE — **같은 행 위에서** 범위별로 --------
    #  evaluate.py 는 조건마다 다른 평가 표본을 쓰므로 조건 간 AUC 를 직접
    #  비교할 수 없다.  여기서는 행을 고정하고 범위만 바꾼다.
    res['classification'] = {}
    for scope, m0 in sel.items():
        rows = np.concatenate([rec[i]['rows'] for i in np.flatnonzero(m0)])
        c = dict(n_rows=int(len(rows)),
                 stalled_rate=float((ev['stalled'][rows] == 1).mean()),
                 cls0_rate=float((ev['cls3'][rows] == 0).mean()))
        for t in MODELS:
            p = P3[t]
            if p is None:
                continue
            c[t] = dict(
                auc_stalled=auc(ev['stalled'][rows] == 1, p[rows, 2]),
                auc_nointeract=auc(ev['cls3'][rows] == 0, p[rows, 0]),
                acc3=float((p[rows].argmax(1) == ev['cls3'][rows]).mean()),
                picks_canonical=float(np.mean(
                    [rec[i][f'{t}_picks_canonical'] for i in np.flatnonzero(m0)])))
        res['classification'][scope] = c

    # ---- 그룹 내 순위 상관 — 범위별, 같은 행 위에서 -----------------------
    res['within_group'] = {}
    for scope, m0 in sel.items():
        rows = np.concatenate([rec[i]['rows'] for i in np.flatnonzero(m0)])
        e = {}
        for t in MODELS:
            e[t] = within_group_rank(gid[rows], L[rows], S[t][rows])
        e['canonical'] = within_group_rank(
            gid[rows], L[rows], np.where(midx[rows] == 0, 0.0, 1.0))
        res['within_group'][scope] = e

    # ---- C-3 거울 검산 ----------------------------------------------------
    # S8 의 필터(active)는 **그룹 전체의 라벨**로 정의되고 어느 선택자의 성적도
    # 쓰지 않는다.  대조를 위해, S6-5 가 쓴 **선택자의 라벨로 정의한 층**을
    # active 안에서 다시 만들어 대칭이 나타나는지 본다.
    res['mirror'] = {}
    act = sel['active'] & fin
    for t in ('S1', 'S2', 'S3p'):
        hc = act & (arr('canonical') > 0)          # canonical 이 틀린 그룹
        hm = act & (arr(t) > 0)                    # 모델이 틀린 그룹
        res['mirror'][t] = dict(
            active_n=int(act.sum()),
            canonical_wrong_frac=float((arr('canonical')[act] > 0).mean()),
            model_wrong_frac=float((arr(t)[act] > 0).mean()),
            unconditioned=boot(arr(t)[act] - arr('canonical')[act], rng, a.boot),
            layer_canonical_wrong=boot(arr(t)[hc] - arr('canonical')[hc], rng, a.boot),
            layer_model_wrong=boot(arr('canonical')[hm] - arr(t)[hm], rng, a.boot))

    # ---- 층별 (k / phase / dep), active 범위 -----------------------------
    res['strata'] = {}
    for key in ('k', 'phase', 'dep'):
        vals = [x[key] for x in rec]
        for v in sorted(set(vals)):
            mv = np.array([x == v for x in vals]) & act
            if mv.sum() < 20:
                continue
            e = dict(n_groups=int(mv.sum()))
            for who in names:
                e[who] = float(arr(who)[mv].mean())
            for t in ('S1', 'S2', 'S3p'):
                e[f'{t}-canonical'] = boot(arr(t)[mv] - arr('canonical')[mv],
                                           rng, a.boot)
            res['strata'][f'{key}={v}'] = e

    json.dump(res, open(f'{OUT}/regret_{a.split}.json', 'w'), indent=1)
    print(f"=== {a.split}  그룹 {len(rec)}  "
          f"active {int(sel['active'].sum())}  nondegen {int(sel['nondegen'].sum())} ===")
    for scope in ('all', 'active', 'nondegen'):
        print(f"--- {scope} (n={res['means'][scope]['n']})")
        print('   평균 regret: ' + '  '.join(
            f'{k} {v:.3f}' for k, v in res['means'][scope].items() if k != 'n'))
        for k, v in res['pairs'][scope].items():
            if not k.endswith('-canonical'):
                continue
            print(f"   {k:18s} {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] "
                  f"{'유의' if v['sig'] else ' - '}")


if __name__ == '__main__':
    main()
