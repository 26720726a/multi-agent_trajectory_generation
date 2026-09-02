"""상호 안전 투영 — 에이전트 간 충돌을 *구조적으로* 불가능하게 만든다.

원본 `mahoi/wm/controller.py:466-499` 의 `_project_safe` 를 **그대로** 옮긴
것이다.  함수 이름·시그니처·기본값(`iters=12`)·계산 순서 모두 원본과 같다.

이 함수가 안전 계층에 있어야 하는 이유는 원본 controller 의 모듈 docstring 이
말하는 그대로다: 안전은 계획 품질과 경쟁하지 않고, 계획이 무엇을 고르든 그
뒤에 무조건 한 번 돈다.  `tests/test_layering.py (a)` 가 이 방향을 못 박는다.
"""
from __future__ import annotations

import numpy as np

from .geometry import seg_seg_min_dist
from .phase import PHASE_DONE, PHASE_DWELL


def _project_safe(problem, pos: np.ndarray, v: np.ndarray,
                  phase: np.ndarray, iters: int = 12) -> np.ndarray:
    """Shrink velocities until no pair's swept segments violate the hard bound.

    Stationary agents are already at a safe separation (invariant maintained by
    this very function), so scaling everything to zero is a valid fallback and
    the loop is guaranteed to terminate.
    """
    n, dt = problem.n, problem.dt
    v = v.copy()
    for _ in range(iters):
        worst = None
        for i in range(n):
            for j in range(i + 1, n):
                R = problem.min_sep(i, j)
                d = seg_seg_min_dist(pos[i], pos[i] + v[i] * dt,
                                     pos[j], pos[j] + v[j] * dt)
                if d < R + 1e-4:
                    if worst is None or d < worst[0]:
                        worst = (d, i, j)
        if worst is None:
            return v
        _, i, j = worst
        for k in (i, j):
            if phase[k] in (PHASE_DWELL, PHASE_DONE):
                continue                          # cannot be asked to move
            v[k] *= 0.5
            if np.linalg.norm(v[k]) < 1e-3:
                v[k] = 0.0
    # last resort: everyone who is allowed to stop, stops
    for k in range(n):
        if phase[k] not in (PHASE_DWELL, PHASE_DONE):
            v[k] = 0.0
    return v
