#!/usr/bin/env python3
"""P3-2 — 후보(mode) 평가기 학습.  완주 분류 + makespan 회귀 두 헤드.

    python scripts/train_wm.py                      # 전체 train
    python scripts/train_wm.py --frac 0.25          # 데이터 규모 곡선용

플래너에는 붙이지 않는다 (P3-4).  여기서는 학습과 val 평가까지만 하고
**test 는 건드리지 않는다.**

구조에서 지켜야 하는 두 가지
  * **positional encoding 없음.** 에이전트 토큰은 self-attention 으로만 섞이고
    순서를 알려주는 항이 없다 — 순열을 바꿔도 같은 예측이 나와야 한다
    (P3-1 은 특성 수준, 여기 STEP 6-5 는 모델 수준 검증).
  * **패딩 마스크.** n=2 인데 슬롯이 8개이므로 빈 토큰이 attention 에
    참여하면 안 된다.  key_padding_mask 로 막고, 풀링도 마스크 평균으로 한다.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = f"{ROOT}/bench/runs/dataset"

# P3-1 §6-3: 에이전트 19차원 중 vel(2)·rho(1)는 t=0 에서 상수라 정보가 0이다.
# phase(3)도 지금은 상수지만 t>0 데이터를 넣으면 살아나므로 남긴다.
DROP = ("vel_x", "vel_y", "rho")


def load(name, fields):
    z = np.load(f"{DS}/{name}.npz", allow_pickle=True)
    keep = [i for i, f in enumerate(fields) if f not in DROP]
    return dict(agent=z["agent"][:, :, keep], agent_mask=z["agent_mask"],
                obst=z["obst"], obst_mask=z["obst_mask"], glob=z["glob"],
                y_cls=z["y_cls"], y_reg=z["y_reg"], y_actual=z["y_actual"],
                reg_mask=z["reg_mask"], uid=z["uid"]), keep


def to_dev(d, dev):
    out = {}
    for k, v in d.items():
        if k == "uid":
            out[k] = v
        elif v.dtype == bool:
            out[k] = torch.from_numpy(v).to(dev)
        else:
            out[k] = torch.from_numpy(v.astype(np.float32)).to(dev)
    return out


class ModeScorer(nn.Module):
    """에이전트/장애물 토큰을 한 시퀀스로 붙여 encoder 에 넣고, 두 헤드로 뺀다.

    전역 특성은 **CLS 토큰**으로 넣는다.  세 방식을 놓고 고른 이유:
      - 각 토큰에 broadcast: 전역값이 토큰마다 반복되어 attention 이 그것을
        다시 배우게 만든다.  차원만 늘고 얻는 게 없다.
      - 헤드 직전 concat: 전역 특성이 토큰 간 상호작용에 관여하지 못한다.
        그런데 `total_rank` 는 "이 후보가 이 집합에서 몇 등이냐"라서 토큰들과
        같이 읽혀야 의미가 산다.
      - **CLS**: 전역값이 attention 에 참여하면서도 한 번만 들어간다.  풀링
        지점이기도 해서 헤드가 읽기 자연스럽다.
    """

    def __init__(self, d_agent, d_obst, d_glob, d_model=128, nhead=4,
                 layers=3, ff=256, dropout=0.1):
        super().__init__()
        self.a_in = nn.Linear(d_agent, d_model)
        self.o_in = nn.Linear(d_obst, d_model)
        self.g_in = nn.Linear(d_glob, d_model)
        # 토큰 종류를 알려주는 학습 임베딩.  순서가 아니라 **종류**이므로
        # 순열 불변성을 깨지 않는다 (에이전트끼리는 모두 같은 값을 받는다).
        self.kind = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.kind, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head_cls = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                      nn.Linear(d_model, 1))
        self.head_reg = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                      nn.Linear(d_model, 1))

    def forward(self, agent, agent_mask, obst, obst_mask, glob):
        B = agent.shape[0]
        a = self.a_in(agent) + self.kind[0]
        o = self.o_in(obst) + self.kind[1]
        g = self.g_in(glob).unsqueeze(1) + self.kind[2]          # CLS
        x = torch.cat([g, a, o], 1)
        cls_m = torch.ones(B, 1, dtype=torch.bool, device=agent.device)
        keep = torch.cat([cls_m, agent_mask, obst_mask], 1)
        h = self.enc(x, src_key_padding_mask=~keep)
        h = self.norm(h[:, 0])                                    # CLS 풀링
        return self.head_cls(h).squeeze(-1), self.head_reg(h).squeeze(-1)


def evaluate(model, d, ry_mu, ry_sd):
    model.eval()
    with torch.no_grad():
        lg, rg = model(d["agent"], d["agent_mask"], d["obst"], d["obst_mask"], d["glob"])
        pred = rg * ry_sd + ry_mu
        bce = nn.functional.binary_cross_entropy_with_logits(lg, d["y_cls"])
        m = d["reg_mask"]
        mse = nn.functional.mse_loss(pred[m], d["y_reg"][m]) if m.any() else torch.tensor(0.)
    return lg.detach(), pred.detach(), float(bce), float(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=1.0, help="train uid 비율 (규모 곡선)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lam", type=float, default=1.0, help="회귀 손실 가중치")
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fields = json.load(open(f"{DS}/norm.json"))["agent_fields"]

    tr_np, keep = load("train", fields)
    va_np, _ = load("val", fields)
    print(f"에이전트 차원 {len(fields)} -> {len(keep)}  (제거: {list(DROP)})")

    if args.frac < 1.0:
        uids = np.unique(tr_np["uid"])
        rng = np.random.default_rng(args.seed)
        pick = set(rng.permutation(uids)[:int(round(args.frac * len(uids)))].tolist())
        m = np.array([u in pick for u in tr_np["uid"]])
        tr_np = {k: v[m] for k, v in tr_np.items()}
        print(f"  train {args.frac:.0%}: uid {len(pick):,} / 행 {m.sum():,}")

    tr = to_dev(tr_np, dev); va = to_dev(va_np, dev)
    # 회귀 타깃 표준화 — 분류 손실과 스케일을 맞춘다 (원 단위는 6~34초)
    rm = tr["reg_mask"]
    ry_mu = float(tr["y_reg"][rm].mean()); ry_sd = float(tr["y_reg"][rm].std())
    print(f"  회귀 타깃 표준화: mu {ry_mu:.2f}s  sd {ry_sd:.2f}s")

    model = ModeScorer(tr["agent"].shape[2], tr["obst"].shape[2],
                       tr["glob"].shape[1]).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    print(f"  파라미터 {npar:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    N = tr["agent"].shape[0]
    hist = []; best = (1e9, -1); ckpt = f"{DS}/model_{args.tag}.pt"
    g = torch.Generator(device="cpu"); g.manual_seed(args.seed)
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(N, generator=g).to(dev)
        tl_c = tl_r = nb = 0.0
        for i in range(0, N, args.batch):
            idx = perm[i:i + args.batch]
            lg, rg = model(tr["agent"][idx], tr["agent_mask"][idx],
                           tr["obst"][idx], tr["obst_mask"][idx], tr["glob"][idx])
            lc = nn.functional.binary_cross_entropy_with_logits(lg, tr["y_cls"][idx])
            m = tr["reg_mask"][idx]
            tgt = (tr["y_reg"][idx][m] - ry_mu) / ry_sd
            lr_ = nn.functional.mse_loss(rg[m], tgt) if m.any() else rg.sum() * 0
            loss = lc + args.lam * lr_
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl_c += float(lc); tl_r += float(lr_); nb += 1
        sched.step()
        _, _, vb, vm = evaluate(model, va, ry_mu, ry_sd)
        vtot = vb + args.lam * (vm / ry_sd ** 2)
        hist.append(dict(epoch=ep, train_cls=tl_c / nb, train_reg=tl_r / nb,
                         val_cls=vb, val_reg_s=vm / ry_sd ** 2, val_rmse=vm ** 0.5,
                         val_total=vtot))
        if vtot < best[0]:
            best = (vtot, ep)
            torch.save(dict(model=model.state_dict(), ry_mu=ry_mu, ry_sd=ry_sd,
                            keep=keep, args=vars(args)), ckpt)
        if ep - best[1] >= args.patience:
            print(f"  early stop @ {ep} (best {best[1]})"); break
        if ep % 20 == 0:
            print(f"  ep {ep:>3}  train {tl_c/nb:.4f}/{tl_r/nb:.4f}  "
                  f"val {vb:.4f}/{vm**0.5:.3f}s  total {vtot:.4f}", flush=True)
    dt = time.perf_counter() - t0
    print(f"  학습 {dt/60:.1f}분, best epoch {best[1]}, val_total {best[0]:.4f}")
    json.dump(dict(history=hist, best_epoch=best[1], seconds=dt, n_params=npar,
                   args=vars(args), n_train=int(N)),
              open(f"{DS}/hist_{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
