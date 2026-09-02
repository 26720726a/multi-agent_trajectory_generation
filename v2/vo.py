"""속도 장애물(VO) 필터 — 후보 속도에 "부딪히기까지 걸리는 시간" 비용을 매긴다.

원본에서는 이 계산이 `mahoi/wm/controller.py` 의 `step()` **루프 안에**
박혀 있었다 (`_ttc` 는 286행, 이웃 항 388-401행, 정적 장애물 항 403-411행).
로직은 하나도 바꾸지 않고 위치만 옮겼다.

왜 `add_vo_cost` 가 배열을 **제자리에서** 더하는가
--------------------------------------------------
원본은 이미 만들어져 있는 `cost` 배열에 `cost[hit] += ...` 로 누적한다.
새 배열에 따로 모았다가 마지막에 한 번 더하면 부동소수 덧셈의 **순서**가
달라져 마지막 자리가 어긋날 수 있고, 그러면 `argmin` 이 동률 근처에서 뒤집혀
rollout 전체가 달라진다.  S2 의 게이트가 "원본과 같은 결과"이므로 여기서는
비트 단위 동일성이 필요하다 — 그래서 원본과 똑같이 제자리 누적을 한다.
`tests/test_vo_extraction.py` 가 100 케이스에서 **정확히 같은 값**(allclose 가
아니라 `==`)인지 확인한다.

계층
----
이 모듈은 `planning/` 을 import 하지 않는다.  `problem` 은 타입으로 받지 않고
`n / dt / agents / min_sep` 만 읽는 덕 타이핑으로 다루고, 장애물 기하는 호출부가
이미 계산해 둔 `near` 배열로 받는다 (원본의 `scene.nearest_points(pos[i])`).
"""
from __future__ import annotations

import numpy as np

from config import CONTROLLER
from .phase import PHASE_DONE


def _ttc(w: np.ndarray, u: np.ndarray, R: float) -> np.ndarray:
    """Time-to-collision for relative offset `w` (2,) and velocities `u` (C,2).

    Returns +inf where the candidate never brings the pair within `R`.
    `w` points from the ego agent towards the obstacle/neighbour, so the pair
    closes when the relative velocity `u` points along `w`.
    """
    ww = float(w @ w)
    c = ww - R * R
    a = np.einsum("ij,ij->i", u, u)
    b = u @ w
    if c < 0.0:                       # already overlapping -> immediate
        return np.zeros(len(u))
    disc = b * b - a * c
    out = np.full(len(u), np.inf)
    ok = (disc > 0.0) & (a > 1e-12) & (b > 0.0)
    if np.any(ok):
        t = (b[ok] - np.sqrt(disc[ok])) / a[ok]
        t = np.where(t < 0.0, np.inf, t)
        out[ok] = t
    return out


def add_vo_cost(problem, i: int, pos: np.ndarray, vel: np.ndarray,
                phase: np.ndarray, cand: np.ndarray, near: np.ndarray,
                alpha: np.ndarray, cost: np.ndarray) -> None:
    """`cost` 에 이웃 VO 항과 정적 장애물 항을 **제자리로** 더한다.

    Parameters
    ----------
    problem : `n`, `dt`, `agents[k].v_max/.radius`, `min_sep(i, j)` 만 읽는다.
    near    : (K, 2) 자기 위치에서 본 장애물/벽의 최근접점.  원본의
              `SceneCache.nearest_points(pos[i])` 결과를 그대로 받는다.
    alpha   : (n, n) 책임 분담.  `alpha[i, j]` 가 1 에 가까울수록 i 가 비킨다.
    cost    : (C,) 이미 선호도·부드러움 항이 들어 있는 배열.  **수정된다.**
    """
    n, dt = problem.n, problem.dt
    a = problem.agents[i]

    for j in range(n):
        if j == i:
            continue
        R = problem.min_sep(i, j) + CONTROLLER.vo_soft_margin
        w = pos[j] - pos[i]
        if float(w @ w) > (R + (a.v_max + problem.agents[j].v_max)
                           * CONTROLLER.tau) ** 2:
            continue                          # far away: cannot interact
        aij = float(alpha[i, j])
        if phase[j] == PHASE_DONE:
            aij = 1.0                         # a parked agent will not move
        u = vel[i][None, :] + (cand - vel[i][None, :]) / aij - vel[j][None, :]
        t_c = _ttc(w, u, R)
        hit = t_c < CONTROLLER.tau
        cost[hit] += CONTROLLER.w_ttc / np.maximum(t_c[hit], dt)

    R_o = a.radius + CONTROLLER.obst_pad + 0.5 * CONTROLLER.vo_soft_margin
    for k in range(len(near)):
        w = near[k] - pos[i]
        if float(w @ w) > (R_o + a.v_max * CONTROLLER.tau_obst) ** 2:
            continue
        t_c = _ttc(w, cand, R_o)
        hit = t_c < CONTROLLER.tau_obst
        cost[hit] += CONTROLLER.w_ttc_obst / np.maximum(t_c[hit], dt)
