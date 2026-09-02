#!/usr/bin/env python3
"""S8 A — 표본 조건 S2/S3 캐시를 만들고 A-1 통계를 낸다.

    python3 scripts/make_s8_caches.py

**라벨 정의는 건드리지 않는다.**  S6-4 의 기하 캐시(`learn/cache/s6_4.npz`)를
그대로 읽어 `split` 열만 바꾼다 — 제외할 행은 `drop` 으로 표시하면
`learn/common.split_of` 가 자연히 걸러낸다.  텐서·라벨·gid·uid 는 한 바이트도
바뀌지 않는다.

세 조건 (지시서 A)
    S1  현행       train 전체        eval 전체
    S2  둘 다 필터  train 활성 그룹    eval 활성 그룹
    S3  학습만 필터 train 활성 그룹    eval 전체

**활성 그룹** = 그룹의 `loss_max` **최소값 > 0** 인 그룹.  S7 이 확정한 것:
최소값이 0 인 그룹에서는 canonical 이 동점으로 자동 정답이 되고 어떤 모델도
그보다 잘할 수 없다.

주의 — 두 가지 "퇴화"를 구분한다:
    활성(active)   min > 0        <- S8 의 필터
    비퇴화(nondegen) max != min   <- S6-5·evaluate.py 의 정의
둘은 다르다 (후보 16개가 전부 같은 양수 값이면 활성이면서 퇴화다).
"""
from __future__ import annotations
import json, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC = 'learn/cache/s6_4.npz'
OUT_DIR = 'learn/cache'
REPORT = 'results/s8/sample_conditions.json'


def group_flags(gid, loss):
    """행마다 (그룹 최소값 > 0, 그룹 max != min)."""
    o = np.argsort(gid, kind='mergesort')
    b = np.flatnonzero(np.diff(gid[o])) + 1
    active = np.zeros(len(gid), bool)
    nondeg = np.zeros(len(gid), bool)
    for g in np.split(o, b):
        L = loss[g]
        active[g] = L.min() > 0
        nondeg[g] = L.max() != L.min()
    return active, nondeg


def stats(split, loss, stalled, k, uid, gid, active, nondeg, name):
    """한 표본 조건의 split 별 통계."""
    out = {}
    for s in ('train', 'val', 'test'):
        m = split == s
        if not m.any():
            out[s] = dict(rows=0)
            continue
        gg = np.unique(gid[m])
        out[s] = dict(
            rows=int(m.sum()), groups=int(len(gg)), uids=int(len(np.unique(uid[m]))),
            loss_pos_frac=float((loss[m] > 0).mean()),
            loss_mean=float(loss[m].mean()), loss_p50=float(np.median(loss[m])),
            stalled_frac=float((stalled[m] == 1).mean()),
            active_group_frac=float(active[m].mean()),
            nondegen_group_frac=float(nondeg[m].mean()),
            k_dist={str(int(v)): int(c) for v, c in
                    sorted(Counter(k[m].tolist()).items())})
    sets = {s: set(np.unique(uid[split == s]).tolist()) for s in ('train', 'val', 'test')}
    out['uid_overlap'] = {'train&val': len(sets['train'] & sets['val']),
                          'train&test': len(sets['train'] & sets['test']),
                          'val&test': len(sets['val'] & sets['test'])}
    out['dropped_rows'] = int((split == 'drop').sum())
    out['name'] = name
    return out


def main():
    z = np.load(SRC, allow_pickle=False)
    d = {kk: z[kk] for kk in z.files}
    gid, loss, split0 = d['gid'], d['loss'], d['split']
    active, nondeg = group_flags(gid, loss)

    os.makedirs(os.path.dirname(REPORT) or '.', exist_ok=True)
    rep = {'source': SRC, 'rows': len(gid),
           'groups': int(len(np.unique(gid))),
           'active_rows_frac': float(active.mean()),
           'active_groups': int(len(np.unique(gid[active]))),
           'nondegen_groups': int(len(np.unique(gid[nondeg]))),
           'conditions': {}}

    conds = {
        'S1': split0.copy(),
        'S2': np.where(active, split0, 'drop'),
        'S3': np.where(active | (split0 != 'train'), split0, 'drop'),
    }
    for name, sp in conds.items():
        rep['conditions'][name] = stats(sp, loss, d['stalled'], d['k'], d['uid'],
                                        gid, active, nondeg, name)
        if name == 'S1':
            continue
        out = f'{OUT_DIR}/s8_{name}.npz'
        dd = dict(d); dd['split'] = sp.astype(split0.dtype)
        np.savez(out, **dd)
        rep['conditions'][name]['cache'] = out

    # S2 와 S3 의 train 이 정말 같은가 (같으면 학습 결과가 비트 단위로 같아야 한다)
    rep['S2_S3_train_identical'] = bool(
        np.array_equal(conds['S2'] == 'train', conds['S3'] == 'train'))
    json.dump(rep, open(REPORT, 'w'), indent=1)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
