"""학습된 후보 평가기(P3-2)를 플래너에 붙이는 얇은 층.

`execute.py` 는 `scorer="nn_filter"` 일 때만 이 모듈을 부른다.  torch 는
그때만 import 되므로, 기본 경로(`scorer="rollout"`)는 torch 없이도 돈다.

**t=0 에서만 쓴다.**  모델은 t=0 상태만 보고 학습됐는데 (P3-1/P3-2), 재계획
시점의 phase 분포는 전혀 다르다 — t>0 에서는 61%가 TO_WP 가 아니고 DONE 은
원핫 슬롯조차 없다.  게다가 `vel` 은 "t=0 에서 상수"라는 이유로 특성에서
빠졌는데 t>0 에서는 상수가 아니다.  그 범위 밖에서 부르면 학습 분포를
벗어난다.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

import numpy as np

_CACHE = {}          # 프로세스당 1회 로딩 (매 replan 마다 로드하지 않는다)

DEP_MODES = ("chain", "fork", "none")


def infer_dep_mode(problem) -> str:
    """`deps` 구조에서 dep_mode 를 되짚는다.  config 가 알려주지 못할 때의 폴백.

    주의: n=2 에서는 chain 과 fork 가 **구조적으로 같다** (둘 다 간선 하나).
    그때는 "chain" 으로 답하므로, 정확한 값이 필요하면 `WMConfig.dep_mode` 로
    넘겨야 한다.
    """
    if not problem.deps:
        return "none"
    out = {}
    for i, _ in problem.deps:
        out[i] = out.get(i, 0) + 1
    return "fork" if max(out.values()) > 1 else "chain"


def _load(path: str, device: str):
    key = (path, device)
    if key in _CACHE:
        return _CACHE[key]
    import torch                                    # 여기서만 필요하다
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if f"{root}/scripts" not in sys.path:
        sys.path.insert(0, f"{root}/scripts")
    from train_wm import ModeScorer
    ck = torch.load(path, map_location=device, weights_only=False)
    norm = json.load(open(os.path.join(os.path.dirname(path), "norm.json")))
    d_agent = len(ck["keep"])
    m = ModeScorer(d_agent, 3, len(norm["global_fields"])).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    _CACHE[key] = (m, ck, norm, torch)
    return _CACHE[key]


def _obstacles(problem) -> np.ndarray:
    """사각형 장애물을 (중심, 외접원 반경) 으로.  build_dataset.py 와 같은 변환."""
    out = []
    for r in problem.world.obstacles:
        out.append((0.5 * (r.x0 + r.x1), 0.5 * (r.y0 + r.y1),
                    0.5 * float(np.hypot(r.x1 - r.x0, r.y1 - r.y0))))
    return np.asarray(out, float).reshape(-1, 3)


def completion_prob(problem, state, modes, costs, cfg) -> Optional[np.ndarray]:
    """각 후보의 **완주 확률**.  모델을 못 쓰면 None.

    특성 구성은 `scripts/build_dataset.py` 와 한 글자도 다르면 안 된다 —
    다르면 학습 때와 다른 입력을 주는 셈이고, 그것은 조용히 틀린다.
    """
    path = cfg.nn_model_path
    if not path or not os.path.exists(path):
        return None
    try:
        m, ck, norm, torch = _load(path, cfg.nn_device)
    except Exception:                                # noqa: BLE001
        return None

    cs = norm["coord_scale"]; vn = norm["v_max_norm"]
    half = 0.5 * problem.world.width
    n = problem.n
    dep = cfg.dep_mode or infer_dep_mode(problem)
    order = np.argsort([c.total for c in costs])     # total_rank: 1 = 가장 싼 것
    rank = np.empty(len(costs), int); rank[order] = np.arange(1, len(costs) + 1)

    wp = np.array([a.waypoint for a in problem.agents], float)
    gl = np.array([a.goal for a in problem.agents], float)
    rho = np.array([a.radius for a in problem.agents], float)
    ob = _obstacles(problem)

    A = np.zeros((len(modes), n, 19), np.float32)
    O = np.zeros((len(modes), max(1, len(ob)), 3), np.float32)
    Om = np.zeros((len(modes), max(1, len(ob))), bool)
    G = np.zeros((len(modes), len(norm["global_fields"])), np.float32)
    for j, md in enumerate(modes):
        for a in range(n):
            f = A[j, a]
            f[0:2] = (state.pos[a] - half) / cs
            f[2:4] = state.vel[a] / vn
            f[4:6] = (wp[a] - half) / cs
            f[6:8] = (gl[a] - half) / cs
            ph = int(state.phase[a])
            if 0 <= ph < norm["n_phase"]:
                f[8 + ph] = 1.0
            f[11] = rho[a] / cs
            r_i = min(int(md.routes[a]), norm["n_route"] - 1)
            f[12 + r_i] = 1.0
            f[16] = md.yield_rank[a] / max(1, n - 1)
            f[17] = 1.0 if md.cautious else 0.0
            f[18] = 1.0 if md.split_side else 0.0
        if len(ob):
            O[j, :len(ob), :2] = (ob[:, :2] - half) / cs
            O[j, :len(ob), 2] = ob[:, 2] / cs
            Om[j, :len(ob)] = True
        G[j, 0] = n / norm["max_agents"]
        G[j, 1 + DEP_MODES.index(dep)] = 1.0
        G[j, 4] = (rank[j] - 1) / max(1, len(costs) - 1)

    keep = ck["keep"]
    with torch.no_grad():
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(cfg.nn_device)
        lg, _ = m(t(A[:, :, keep]),
                  t(np.ones((len(modes), n), bool)),
                  t(O), t(Om), t(G))
        p = torch.sigmoid(lg).cpu().numpy()
    return p.astype(float)
