#!/usr/bin/env python3
"""S6 — 평가.  분류 AUC / 회귀 RMSE / **top-1 regret** / 층별 / 순열 불변성 / 속도.

    python3 learn/evaluate.py --tag A2 --split val

top-1 regret 이 핵심이다 (계획서 §6 P4-3).  MSE 가 아니라 "모델이 1등으로 고른
후보와 실제 최선의 차이"가 downstream 과 직결된다.  비교군은 셋이다:

    모델    — 예측 점수 최소 후보
    교사    — 실제 loss_max 최소 후보 (**정의상 regret 0**)
    무작위  — 후보를 균등 확률로.  표집하지 않고 후보별 regret 분포를
              그대로 쓴다 (평균은 기댓값과 정확히 같다)

후보 16개가 전부 같은 라벨인 클러스터는 regret 이 정의상 0 이므로, 그 행을
**포함한 값과 제외한 값을 둘 다** 낸다.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learn.common import (apply_norm, ckpt_path, load, split_of, stall_score,
                          to_dev)
from learn.model import ModeScorer

OUT = 'results/s6'


def auc(y, s):
    """rank 기반 ROC-AUC (동점은 평균 rank)."""
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    r = np.empty(len(s), float)
    sr = s[order]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def wquantile(v, w, qs):
    o = np.argsort(v, kind='mergesort')
    v, w = np.asarray(v)[o], np.asarray(w)[o]
    c = np.cumsum(w) / w.sum()
    return [float(v[np.searchsorted(c, q, side='left').clip(0, len(v) - 1)])
            for q in qs]


def regret_table(gid, loss, stalled, score, midx=None, keep=None):
    """그룹(클러스터 상태)별 top-1 regret — 모델 / 교사 / 무작위 / canonical.

    canonical 은 `mode_idx == 0`(최단 경로·항등 양보 순서·감속 없음) 을 늘
    고르는 선택자다.  플래너가 아무것도 고르지 못할 때 떨어지는 자리이므로
    "모델을 쓰지 않을 때"의 실제 기준선이다 (무작위는 그 자리가 아니다).
    """
    if keep is None:
        keep = np.ones(len(gid), bool)
    if midx is None:
        midx = np.full(len(gid), -1)
    gid, loss, stalled, score, midx = (gid[keep], loss[keep], stalled[keep],
                                       score[keep], midx[keep])
    order = np.argsort(gid, kind='mergesort')
    gid, loss, stalled, score, midx = (gid[order], loss[order], stalled[order],
                                       score[order], midx[order])
    bnd = np.flatnonzero(np.diff(gid)) + 1
    groups = np.split(np.arange(len(gid)), bnd)

    out = {}
    m_reg, o_reg, r_mean, c_reg = [], [], [], []
    m_pick, o_pick, r_pick, c_pick = [], [], [], []   # 정체 후보 선택률
    m_abs, o_abs, r_abs, c_abs = [], [], [], []       # 고른 후보의 실제 loss
    pooled_v, pooled_w = [], []
    degen, has_canon = [], []
    for g in groups:
        L, S, C, M = loss[g], score[g], stalled[g], midx[g]
        lo = L.min()
        degen.append(bool(L.max() == lo))
        a = int(np.argmin(S))
        m_reg.append(float(L[a] - lo)); m_pick.append(float(C[a])); m_abs.append(float(L[a]))
        b = int(np.argmin(L))
        o_reg.append(0.0); o_pick.append(float(C[b])); o_abs.append(float(L[b]))
        r_mean.append(float((L - lo).mean())); r_pick.append(float(C.mean()))
        r_abs.append(float(L.mean()))
        c = np.flatnonzero(M == 0)
        has_canon.append(len(c) > 0)
        j = int(c[0]) if len(c) else a
        c_reg.append(float(L[j] - lo)); c_pick.append(float(C[j])); c_abs.append(float(L[j]))
        pooled_v.append(L - lo); pooled_w.append(np.full(len(g), 1.0 / len(g)))
    degen = np.array(degen); has_canon = np.array(has_canon)
    pooled_v = np.concatenate(pooled_v); pooled_w = np.concatenate(pooled_w)
    gsz = np.array([len(g) for g in groups])
    grp_of_pool = np.repeat(np.arange(len(groups)), gsz)

    for name, sel in (('all', np.ones(len(groups), bool)), ('nondegen', ~degen)):
        pm = sel[grp_of_pool]
        e = dict(n_groups=int(sel.sum()))
        for who, reg, pk, ab in (('model', np.array(m_reg), m_pick, m_abs),
                                 ('teacher', np.array(o_reg), o_pick, o_abs),
                                 ('random', np.array(r_mean), r_pick, r_abs),
                                 ('canonical', np.array(c_reg), c_pick, c_abs)):
            s2 = sel & has_canon if who == 'canonical' else sel
            if not s2.any():
                continue
            q = (wquantile(pooled_v[pm], pooled_w[pm], [.5, .9])
                 if who == 'random' else
                 [float(np.percentile(reg[s2], 50)), float(np.percentile(reg[s2], 90))])
            e[who] = dict(mean=float(reg[s2].mean()), p50=q[0], p90=q[1],
                          stall_rate=float(np.array(pk)[s2].mean()),
                          picked_loss=float(np.array(ab)[s2].mean()),
                          n=int(s2.sum()))
        out[name] = e
    out['degenerate_frac'] = float(degen.mean())
    out['n_groups_total'] = len(groups)
    return out


def within_group_rank(gid, loss, score):
    """그룹 안에서 **예측 순서가 실제 순서와 얼마나 맞는가** — S6 에 없던 지표.

    top-1 regret 은 그룹 안의 1등만 보므로, 순위 손실이 그룹 내 신호를 실제로
    옮겼는지는 regret 만으로는 드러나지 않는다.  동점이 많으므로 (비퇴화
    그룹의 최소 동점 후보가 평균 6~7개) 동점을 보정하는 **Kendall tau-b** 를
    쓰고 Spearman 도 함께 낸다.  퇴화 그룹은 실제 순서가 없으므로 제외한다.

    부호: score 는 "예측 loss"(작을수록 좋다), loss 는 실제값이므로
    **양수가 맞는 방향**이다.
    """
    o = np.argsort(gid, kind='mergesort')
    gid, loss, score = gid[o], loss[o], score[o]
    bnd = np.flatnonzero(np.diff(gid)) + 1
    taus, rhos, hits = [], [], []
    for g in np.split(np.arange(len(gid)), bnd):
        y, x = loss[g], score[g]
        if y.max() == y.min():
            continue
        dy = np.sign(y[:, None] - y[None, :])
        dx = np.sign(x[:, None] - x[None, :])
        iu = np.triu_indices(len(g), 1)
        cy, cx = dy[iu], dx[iu]
        n0 = len(cy)
        n1 = int((cx == 0).sum()); n2 = int((cy == 0).sum())
        den = np.sqrt((n0 - n1) * (n0 - n2))
        if den > 0:
            taus.append(float((cx * cy).sum() / den))
        ry, rx = _avgrank(y), _avgrank(x)
        if rx.std() > 0:
            rhos.append(float(np.corrcoef(rx, ry)[0, 1]))
        hits.append(float(y[int(np.argmin(x))] == y.min()))
    return dict(kendall_tau_b=float(np.mean(taus)) if taus else float('nan'),
                spearman=float(np.mean(rhos)) if rhos else float('nan'),
                top1_hit=float(np.mean(hits)) if hits else float('nan'),
                n_groups=len(hits), n_tau=len(taus))


def _avgrank(v):
    o = np.argsort(v, kind='mergesort')
    r = np.empty(len(v), float)
    sv = v[o]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def predict(model, t, bs=4096):
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, t['agent'].shape[0], bs):
            sl = slice(i, i + bs)
            outs.append(model(t['agent'][sl], t['agent_mask'][sl], t['obst'][sl],
                              t['obst_mask'][sl], t['glob'][sl]))
    return (torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs]))


def perm_check(model, t, seed=0, trials=5, n=256):
    """에이전트/장애물 슬롯 순서를 섞어도 예측이 같아야 한다 (≤1e-5)."""
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    sub = {k: v[:n] for k, v in t.items()
           if k in ('agent', 'agent_mask', 'obst', 'obst_mask', 'glob')}
    model.eval()
    with torch.no_grad():
        c0, r0 = model(sub['agent'], sub['agent_mask'], sub['obst'],
                       sub['obst_mask'], sub['glob'])
        dc = dr = 0.0
        for _ in range(trials):
            pa = torch.stack([torch.randperm(8, generator=g)
                              for _ in range(n)]).to(sub['agent'].device)
            po = torch.stack([torch.randperm(8, generator=g)
                              for _ in range(n)]).to(sub['agent'].device)
            a = torch.gather(sub['agent'], 1, pa[..., None].expand(-1, -1, sub['agent'].shape[2]))
            am = torch.gather(sub['agent_mask'], 1, pa)
            o = torch.gather(sub['obst'], 1, po[..., None].expand(-1, -1, sub['obst'].shape[2]))
            om = torch.gather(sub['obst_mask'], 1, po)
            c1, r1 = model(a, am, o, om, sub['glob'])
            dc = max(dc, float((c1 - c0).abs().max()))
            dr = max(dr, float((r1 - r0).abs().max()))
    return dict(cls=dc, reg=dr)


def speed(model, t, dev, batch=512, iters=50):
    sub = {k: t[k][:batch] for k in ('agent', 'agent_mask', 'obst', 'obst_mask', 'glob')}
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(**sub)
        if dev == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(**sub)
        if dev == 'cuda':
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='A2')
    ap.add_argument('--cache', default='learn/cache/s6.npz')
    ap.add_argument('--split', default='val')
    ap.add_argument('--cpu-speed', action='store_true')
    ap.add_argument('--out-json', default=None, help='결과 json 경로 (기본 results/s6/eval_<tag>_<split>.json)')
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(ckpt_path(a.tag), map_location=dev, weights_only=False)
    st = {k: np.array(v, np.float32) for k, v in ck['norm'].items()}
    n_cls, mu, sd, variant = ck['n_cls'], ck['mu'], ck['sd'], ck['variant']

    d = load(a.cache)
    ev = apply_norm(split_of(d, a.split), st)
    tr_raw = split_of(d, 'train')
    t = to_dev({k: ev[k] for k in ('agent', 'agent_mask', 'obst', 'obst_mask', 'glob')}, dev)

    model = ModeScorer(ev['agent'].shape[2], ev['obst'].shape[2],
                       ev['glob'].shape[1], n_cls=n_cls).to(dev)
    model.load_state_dict(ck['model'])

    lg, rg = predict(model, t)
    pred = (rg * sd + mu).cpu().numpy()
    pstall = stall_score(lg, n_cls).cpu().numpy()
    loss, stalled, gid = ev['loss'], ev['stalled'], ev['gid']

    # --- 순위 점수 -------------------------------------------------------
    # A1/A2 는 회귀 출력 그 자체.  A3 는 클래스 확률로 기댓값을 만든다:
    #   E[loss] = P(정상)*ŷ + P(정체)*(train 정체 행의 평균 loss)
    # (A3 의 회귀 헤드는 "정상" 행에서만 학습되므로 그대로 쓰면 편향된다)
    score_reg = pred.copy()
    if n_cls == 3:
        p = torch.softmax(lg, -1).cpu().numpy()
        l2 = float(tr_raw['loss'][tr_raw['cls3'] == 2].mean())
        score = p[:, 1] * pred + p[:, 2] * l2
    else:
        score = pred.copy()

    res = dict(tag=a.tag, variant=variant, split=a.split, n=len(loss),
               n_groups=int(len(np.unique(gid))))

    # --- 분류 ------------------------------------------------------------
    res['auc_stalled'] = auc(stalled == 1, pstall)
    res['stalled_rate'] = float((stalled == 1).mean())
    if n_cls == 3:
        p3 = torch.softmax(lg, -1).cpu().numpy()
        res['auc_nointeract'] = auc(ev['cls3'] == 0, p3[:, 0])
        res['acc3'] = float((p3.argmax(1) == ev['cls3']).mean())

    # --- 회귀 ------------------------------------------------------------
    pos = loss > 0
    rmse = lambda m: float(np.sqrt(((pred[m] - loss[m]) ** 2).mean())) if m.any() else float('nan')
    const = float(tr_raw['loss'][tr_raw['loss'] > 0].mean())
    res['rmse'] = dict(
        all=rmse(np.ones(len(loss), bool)), positive=rmse(pos),
        zero=rmse(loss == 0),
        const_baseline_all=float(np.sqrt(((const - loss) ** 2).mean())),
        const_baseline_positive=float(np.sqrt(((const - loss[pos]) ** 2).mean())),
        label_std_all=float(loss.std()), label_std_positive=float(loss[pos].std()))

    # --- top-1 regret ----------------------------------------------------
    res['regret'] = regret_table(gid, loss, stalled, score, ev['mode_idx'])
    res['regret_regonly'] = regret_table(gid, loss, stalled, score_reg, ev['mode_idx'])

    # --- 그룹 내 순위 상관 (S6-2) ----------------------------------------
    res['within_group'] = within_group_rank(gid, loss, score)
    res['within_group_canonical'] = within_group_rank(
        gid, loss, np.where(ev['mode_idx'] == 0, 0.0, 1.0))

    # --- 층별 ------------------------------------------------------------
    strata = {}
    for key, vals in (('k', ev['k']), ('phase', ev['phase_key']), ('dep', ev['dep'])):
        for v in sorted(set(vals.tolist())):
            m = vals == v
            if m.sum() < 200:
                continue
            e = dict(n=int(m.sum()), auc=auc(stalled[m] == 1, pstall[m]),
                     stalled_rate=float((stalled[m] == 1).mean()),
                     rmse_pos=rmse(m & pos), pos_rate=float(pos[m].mean()))
            # regret 은 그룹이 층 안에서 온전할 때만 뜻이 있다 — k/phase/dep 는
            # 모두 클러스터 상태 단위 속성이라 그룹이 쪼개지지 않는다.
            e['regret'] = regret_table(gid, loss, stalled, score, ev['mode_idx'], keep=m)
            strata[f'{key}={v}'] = e
    res['strata'] = strata

    # --- 순열 불변성 / 속도 ----------------------------------------------
    res['perm'] = perm_check(model, t)
    res['speed_ms_512'] = dict(gpu=speed(model, t, dev))
    if a.cpu_speed:
        mc = ModeScorer(ev['agent'].shape[2], ev['obst'].shape[2],
                        ev['glob'].shape[1], n_cls=n_cls)
        mc.load_state_dict({k: v.cpu() for k, v in ck['model'].items()})
        tc = to_dev({k: ev[k] for k in ('agent', 'agent_mask', 'obst', 'obst_mask', 'glob')}, 'cpu')
        res['speed_ms_512']['cpu'] = speed(mc, tc, 'cpu')

    fn = a.out_json or f'{OUT}/eval_{a.tag}_{a.split}.json'
    os.makedirs(os.path.dirname(fn) or '.', exist_ok=True)
    json.dump(res, open(fn, 'w'), indent=1)
    r = res['regret']['all']; rn = res['regret']['nondegen']
    print(f"[{a.tag}/{a.split}] AUC {res['auc_stalled']:.4f}  "
          f"RMSE+ {res['rmse']['positive']:.3f}m  "
          f"regret 전체 {r['model']['mean']:.3f} (무작위 {r['random']['mean']:.3f})  "
          f"비퇴화 {rn['model']['mean']:.3f} (무작위 {rn['random']['mean']:.3f})  "
          f"tau_b {res['within_group']['kendall_tau_b']:.4f}  "
          f"perm {max(res['perm'].values()):.2e}  {fn}")


if __name__ == '__main__':
    main()
