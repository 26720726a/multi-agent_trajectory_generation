#!/usr/bin/env python3
"""S6 — 2-헤드 모델 학습 (A1 / A2 / A3), S6-2 에서 **그룹 내 순위 손실**을 더했다.

    python3 learn/train.py --variant A2
    python3 learn/train.py --variant A2 --frac 0.25 --tag A2_f025
    python3 learn/train.py --variant A3 --lam-rank 1.0 --tag A3R   # S6-2

원본 `Mahoi-WM/mahoi-wm/scripts/train_wm.py` 를 가져와 **입력 특성과 라벨
정의만** 바꿨다.  모델 구조(learn/model.py)와 학습 루프는 거의 그대로다.

A 세 방식 (loss_max = 0 인 47.5% 를 어떻게 다룰 것인가)
    A1  그대로       — 전 행을 회귀 타깃으로
    A2  회귀 제외    — loss_max > 0 인 행만 회귀 손실에 (§5.3 Mask 구조와 동일)
    A3  3-클래스     — [상호작용 없음 / 정상 / 정체], 회귀는 "정상"만

**회귀 타깃은 표준화한다.** 미터 원 단위로 두면 MSE 가 BCE 보다 커져
λ 가 의도대로 동작하지 않는다.  train 통계를 val/test 에 그대로 쓴다.

S6-2 (`--lam-rank > 0`): 배치를 **클러스터 상태 그룹 단위**로 만들고
그룹 내 top-1 소프트맥스 교차엔트로피를 더한다 (learn/common.py).
`--lam-rank 0` 은 손실은 S6 과 같고 배치 구성만 그룹 단위인 대조군이다.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learn.common import (apply_norm, cls_loss, group_index, listnet_loss,
                          load, norm_stats, rank_score, rank_targets,
                          split_of, targets, to_dev)
from learn.model import ModeScorer

OUT = 'results/s6'


def prep(d, variant, dev):
    y_cls, n_cls, rmask = targets(d, variant)
    t = to_dev(dict(agent=d['agent'], agent_mask=d['agent_mask'],
                    obst=d['obst'], obst_mask=d['obst_mask'], glob=d['glob'],
                    y_reg=d['loss'], reg_mask=rmask.astype(bool)), dev)
    t['y_cls'] = torch.from_numpy(y_cls).to(dev)
    return t, n_cls


def prep_groups(d, dev):
    """그룹 색인 + 소프트 타깃 + 그룹 가중 (순위 손실용)."""
    idx, msk = group_index(d['gid'])
    tgt, w = rank_targets(d['loss'], idx, msk)
    return dict(idx=torch.from_numpy(idx).to(dev),
                mask=torch.from_numpy(msk).to(dev),
                target=torch.from_numpy(tgt).to(dev),
                w=torch.from_numpy(w).to(dev))


def evaluate(model, t, n_cls, mu, sd, lam, lam_rank=0.0, gr=None, l2=0.0):
    model.eval()
    with torch.no_grad():
        lg, rg = model(t['agent'], t['agent_mask'], t['obst'], t['obst_mask'],
                       t['glob'])
        pred = rg * sd + mu
        bce = float(cls_loss(lg, t['y_cls'], n_cls))
        m = t['reg_mask']
        mse = float(nn.functional.mse_loss(pred[m], t['y_reg'][m])) if m.any() else 0.0
        rk = 0.0
        if gr is not None:
            s = rank_score(lg, rg, n_cls, mu, sd, l2)
            rk = float(listnet_loss(s[gr['idx']], gr['target'], gr['mask'], gr['w']))
    return bce, mse, rk, bce + lam * mse / sd ** 2 + lam_rank * rk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=('A1', 'A2', 'A3'), default='A2')
    ap.add_argument('--cache', default='learn/cache/s6.npz')
    ap.add_argument('--frac', type=float, default=1.0, help='train uid 비율')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--lam-rank', type=float, default=0.0,
                    help='>0 이면 그룹 단위 배치 + 순위 손실 (S6-2)')
    ap.add_argument('--group-batch', action='store_true',
                    help='lam-rank=0 이어도 그룹 단위로 배치를 만든다 (대조군)')
    ap.add_argument('--seed', type=int, default=20260901)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--out', default=OUT, help='산출물 디렉터리')
    a = ap.parse_args()
    tag = a.tag or a.variant
    grouped = a.lam_rank > 0 or a.group_batch

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.deterministic = True
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    d = load(a.cache)
    tr, va = split_of(d, 'train'), split_of(d, 'val')
    if a.frac < 1.0:
        uids = np.unique(tr['uid'])
        rng = np.random.default_rng(a.seed)
        pick = set(rng.permutation(uids)[:int(round(a.frac * len(uids)))].tolist())
        m = np.array([u in pick for u in tr['uid']])
        tr = {k: v[m] for k, v in tr.items()}
        print(f'  train {a.frac:.0%}: uid {len(pick):,} / 행 {int(m.sum()):,}')

    # 순위 점수의 정체 항 L̄₂ — 평가(evaluate.py)가 쓰는 값과 같은 정의다.
    l2 = float(tr['loss'][tr['cls3'] == 2].mean())

    st = norm_stats(tr)
    tr, va = apply_norm(tr, st), apply_norm(va, st)
    t_tr, n_cls = prep(tr, a.variant, dev)
    t_va, _ = prep(va, a.variant, dev)
    g_tr = prep_groups(tr, dev) if grouped else None
    g_va = prep_groups(va, dev) if grouped else None

    rm = t_tr['reg_mask']
    mu = float(t_tr['y_reg'][rm].mean()); sd = float(t_tr['y_reg'][rm].std())
    print(f'  {a.variant}: 회귀 표본 {int(rm.sum()):,}/{len(rm):,} '
          f'({float(rm.float().mean()):.1%})  타깃 표준화 mu {mu:.3f}m sd {sd:.3f}m')
    if grouped:
        G, S = g_tr['idx'].shape
        print(f'  그룹 배치: 그룹 {G:,} × 슬롯 {S}  '
              f'순위 손실에 쓰는 비퇴화 그룹 {int(g_tr["w"].sum()):,} '
              f'({float(g_tr["w"].mean()):.1%})  L̄₂ {l2:.3f}m  '
              f'λ_rank {a.lam_rank}')

    model = ModeScorer(tr['agent'].shape[2], tr['obst'].shape[2],
                       tr['glob'].shape[1], n_cls=n_cls).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    print(f'  파라미터 {npar:,}  device {dev}')
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    os.makedirs(a.out, exist_ok=True)
    ckpt = f'{a.out}/model_{tag}.pt'
    N = t_tr['agent'].shape[0]
    hist, best = [], (1e18, -1)
    g = torch.Generator(); g.manual_seed(a.seed)
    t0 = time.perf_counter()
    for ep in range(a.epochs):
        model.train()
        tc = trg = trk = nb = 0.0
        if grouped:
            S = g_tr['idx'].shape[1]
            gpb = max(1, a.batch // S)          # 행 예산은 그대로 (32 그룹 × 16)
            perm = torch.randperm(g_tr['idx'].shape[0], generator=g).to(dev)
            starts = range(0, len(perm), gpb)
        else:
            perm = torch.randperm(N, generator=g).to(dev)
            starts = range(0, N, a.batch)
        for i in starts:
            if grouped:
                gb = perm[i:i + gpb]
                gm = g_tr['mask'][gb]
                idx = g_tr['idx'][gb].reshape(-1)
                flat = gm.reshape(-1)
            else:
                idx = perm[i:i + a.batch]
                gm = flat = None
            lg, rg = model(t_tr['agent'][idx], t_tr['agent_mask'][idx],
                           t_tr['obst'][idx], t_tr['obst_mask'][idx],
                           t_tr['glob'][idx])
            if grouped:
                lc = cls_loss(lg[flat], t_tr['y_cls'][idx][flat], n_cls)
                m = t_tr['reg_mask'][idx] & flat
            else:
                lc = cls_loss(lg, t_tr['y_cls'][idx], n_cls)
                m = t_tr['reg_mask'][idx]
            tgt = (t_tr['y_reg'][idx][m] - mu) / sd
            lr_ = nn.functional.mse_loss(rg[m], tgt) if m.any() else rg.sum() * 0
            loss = lc + a.lam * lr_
            lk = rg.sum() * 0
            if a.lam_rank > 0:
                s = rank_score(lg, rg, n_cls, mu, sd, l2).view(gm.shape)
                lk = listnet_loss(s, g_tr['target'][gb], gm, g_tr['w'][gb])
                loss = loss + a.lam_rank * lk
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tc += float(lc.detach()); trg += float(lr_.detach())
            trk += float(lk.detach()); nb += 1
        sched.step()
        vb, vm, vk, vtot = evaluate(model, t_va, n_cls, mu, sd, a.lam,
                                    a.lam_rank, g_va, l2)
        hist.append(dict(epoch=ep, train_cls=tc / nb, train_reg=trg / nb,
                         train_rank=trk / nb, val_cls=vb, val_reg_s=vm / sd ** 2,
                         val_rmse=vm ** 0.5, val_rank=vk, val_total=vtot))
        if vtot < best[0]:
            best = (vtot, ep)
            torch.save(dict(model=model.state_dict(), mu=mu, sd=sd, n_cls=n_cls,
                            norm={k: v.tolist() for k, v in st.items()},
                            variant=a.variant, args=vars(a)), ckpt)
        if ep - best[1] >= a.patience:
            print(f'  early stop @ {ep} (best {best[1]})'); break
        if ep % 10 == 0:
            print(f'  ep {ep:>3}  train {tc/nb:.4f}/{trg/nb:.4f}/{trk/nb:.4f}  '
                  f'val {vb:.4f}/{vm**0.5:.3f}m/{vk:.4f}  total {vtot:.4f}',
                  flush=True)
    dt = time.perf_counter() - t0
    print(f'  학습 {dt/60:.1f}분  best epoch {best[1]}  val_total {best[0]:.4f}')
    json.dump(dict(history=hist, best_epoch=best[1], seconds=dt, n_params=npar,
                   variant=a.variant, args=vars(a), n_train=int(N),
                   reg_frac=float(rm.float().mean()), mu=mu, sd=sd, l2=l2),
              open(f'{a.out}/hist_{tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
