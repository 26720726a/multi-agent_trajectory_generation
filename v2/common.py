#!/usr/bin/env python3
"""S6 — 학습/평가가 함께 쓰는 적재·정규화·손실 정의."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

#: 정규화할 연속 차원의 기본값 — pos(2) + vel(2) + subgoal(2).
#: 원핫(phase 4, route 4)과 rank, 플래그는 건드리지 않는다.
#: S6-4 의 기하 캐시는 route 6칸도 연속이므로 캐시가 `agent_cont` 를 들고 온다.
AGENT_CONT = slice(0, 6)


def agent_cont(d):
    """이 캐시에서 표준화할 에이전트 차원 (없으면 S6 기본값)."""
    v = d.get('agent_cont')
    return AGENT_CONT if v is None else np.asarray(v)
GLOB_CONT = slice(0, 2)       # k, density

VARIANTS = ('A1', 'A2', 'A3')


def load(path='learn/cache/s6.npz'):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def split_of(d, name):
    m = d['split'] == name
    return {k: (v[m] if getattr(v, 'ndim', 0) and len(v) == len(d['split']) else v)
            for k, v in d.items() if k != 'pos_err'}


def norm_stats(tr):
    """train 의 **유효 슬롯**만으로 평균/표준편차를 낸다 (패딩 제외)."""
    ac = agent_cont(tr)
    a = tr['agent'][tr['agent_mask']][:, ac]
    o = tr['obst'][tr['obst_mask']]
    g = tr['glob'][:, GLOB_CONT]
    f = lambda x: (x.mean(0), np.maximum(x.std(0), 1e-6))
    (am, asd), (om, osd), (gm, gsd) = f(a), f(o), f(g)
    return dict(a_mu=am, a_sd=asd, o_mu=om, o_sd=osd, g_mu=gm, g_sd=gsd)


def apply_norm(d, s):
    d = dict(d)
    ac = agent_cont(d)
    a = d['agent'].copy()
    a[..., ac] = (a[..., ac] - s['a_mu']) / s['a_sd']
    a[~d['agent_mask']] = 0.0
    o = d['obst'].copy()
    o[...] = (o - s['o_mu']) / s['o_sd']
    o[~d['obst_mask']] = 0.0
    g = d['glob'].copy()
    g[:, GLOB_CONT] = (g[:, GLOB_CONT] - s['g_mu']) / s['g_sd']
    d['agent'], d['obst'], d['glob'] = a, o, g
    return d


def to_dev(d, dev):
    out = {}
    for k, v in d.items():
        if v.dtype == bool:
            out[k] = torch.from_numpy(v).to(dev)
        elif v.dtype.kind in 'fi':
            out[k] = torch.from_numpy(
                v.astype(np.int64 if v.dtype.kind == 'i' else np.float32)).to(dev)
        else:
            out[k] = v
    return out


def targets(d, variant):
    """(분류 타깃, 분류 클래스 수, 회귀 마스크) — A 세 방식의 유일한 차이."""
    if variant == 'A1':
        return d['stalled'].astype(np.float32), 1, np.ones(len(d['loss']), bool)
    if variant == 'A2':
        return d['stalled'].astype(np.float32), 1, d['loss'] > 0
    if variant == 'A3':
        return d['cls3'].astype(np.int64), 3, d['cls3'] == 1
    raise ValueError(variant)


def cls_loss(logits, y, n_cls):
    if n_cls == 1:
        return nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(-1), y)
    return nn.functional.cross_entropy(logits, y)


def stall_score(logits, n_cls):
    """정체 확률 — A1/A2 는 sigmoid, A3 는 클래스 2 의 softmax."""
    if n_cls == 1:
        return torch.sigmoid(logits.squeeze(-1))
    return torch.softmax(logits, -1)[:, 2]


# --- S6-2: 그룹(클러스터 상태) 단위 순위 손실 --------------------------------
#
# S6 §11 의 진단: `loss_max` 분산의 58~66% 가 **그룹 간**(장면 난이도)에 있고
# top-1 regret 은 **그룹 안**의 순서만 본다.  MSE 는 그 대부분을 그룹 간
# 차이를 줄이는 데 쓴다.  아래 손실은 그룹마다 소프트맥스를 걸므로 점수에
# 그룹별 상수를 더해도 값이 변하지 않는다 — 즉 그룹 간 성분이 손실에서
# 정확히 사라진다.

def group_index(gid, max_size=None):
    """gid -> (idx[G,S] int64, mask[G,S] bool).

    같은 클러스터 상태의 행은 캐시 안에서 인접해 있지 않으므로(확인) 여기서
    모은다.  그룹 크기는 대부분 16 이지만 dedup 으로 9~15 인 그룹이 조금 있다.
    """
    o = np.argsort(gid, kind='mergesort')
    b = np.flatnonzero(np.diff(gid[o])) + 1
    gs = np.split(o, b)
    S = max_size or max(len(g) for g in gs)
    idx = np.zeros((len(gs), S), np.int64)
    msk = np.zeros((len(gs), S), bool)
    for i, g in enumerate(gs):
        idx[i, :len(g)] = g
        msk[i, :len(g)] = True
    return idx, msk


def rank_targets(loss, idx, mask):
    """소프트 타깃 t[G,S] 와 그룹 가중 w[G].

    타깃은 **그룹 최소 라벨을 갖는 후보 전부에 균등**하게 건다.  top-1 regret
    은 최소 후보 중 아무거나 고르면 0 이므로, 동점 최소를 하나로 좁히는 것은
    지표가 요구하지 않는 일이다 (비퇴화 그룹의 최소 동점 후보가 평균 6~7개다).

    퇴화 그룹(16개 라벨이 전부 같음)은 w=0 이다 — 어느 후보를 골라도 regret
    이 0 이라 순위 정보가 없다.
    """
    L = np.where(mask, loss[idx], np.inf)
    mn = L.min(1, keepdims=True)
    t = (L == mn).astype(np.float32)
    t /= t.sum(1, keepdims=True)
    hi = np.where(mask, loss[idx], -np.inf).max(1, keepdims=True)
    w = (hi > mn)[:, 0].astype(np.float32)
    return t, w


def rank_score(logits, reg, n_cls, mu, sd, l2):
    """**평가가 쓰는 순위 점수와 같은 식**을 표준화 단위로 (evaluate.py §순위 점수).

        E[loss] = P(상호작용 없음)·0 + P(정상)·ŷ + P(정체)·L̄₂

    "상호작용 없음"(loss_max==0) 클래스는 기여값이 0 이라 **그룹 내 최하위로
    고정**된다 — 별도 처리를 두지 않고 이 결합 자체가 처리한다.
    """
    if n_cls == 3:
        p = torch.softmax(logits, -1)
        return (p[:, 1] * (reg * sd + mu) + p[:, 2] * l2) / sd
    return reg


def listnet_loss(score, target, mask, w):
    """그룹 내 top-1 소프트맥스 교차엔트로피 (ListNet top-1 형태).

    점수는 작을수록 좋으므로 `-score` 에 소프트맥스를 건다.  score 는
    표준화 단위(라벨 σ)이므로 온도는 1 로 두고 새 하이퍼파라미터를 만들지
    않는다.
    """
    logits = torch.where(mask, -score, torch.full_like(score, -1e9))
    ce = -(target * torch.log_softmax(logits, -1)).sum(-1)
    return (ce * w).sum() / w.sum().clamp_min(1.0)


def ckpt_path(tag, dirs=('results/s8', 'results/s6_4', 'results/s6_2', 'results/s6')):
    """태그 -> 체크포인트 경로.  최근 단계의 산출물부터 찾는다."""
    import os
    for d in dirs:
        p = f'{d}/model_{tag}.pt'
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'model_{tag}.pt 를 {dirs} 어디에도 못 찾았다')
