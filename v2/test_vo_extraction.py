"""묶음 2 — VO 필터와 안전 투영을 controller 에서 떼어낸 것이 정말 같은 계산인가.

원본에서 이 두 덩어리는 `mahoi/wm/controller.py` 의 `step()` 루프 안에 박혀
있었다.  떼어내면서 값이 한 자리라도 달라지면 `argmin` 이 동률 근처에서 뒤집혀
rollout 전체가 갈라지므로, 판정은 `allclose` 가 아니라 **`==`** 다.

아래 `_reference_*` 는 원본 코드를 **글자 그대로** 옮겨 온 것이며, 상수도
원본 리터럴을 그대로 적었다 (config 를 참조하면 "둘 다 config 를 읽으니 같다"는
동어반복이 되어 대조가 되지 않는다).  원본 저장소를 직접 import 해서 돌린 대조는
`reports/S2_migration.md` §4 에 첨부한 스크래치패드 스크립트 결과다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from safety.phase import PHASE_DONE, PHASE_DWELL
from safety.project import _project_safe
from safety.vo import _ttc, add_vo_cost

# 원본 mahoi/wm/controller.py:41-52 의 리터럴.
TAU, TAU_OBST = 2.6, 1.1
W_TTC, W_TTC_OBST = 1.35, 0.55
OBST_PAD, SOFT_MARGIN = 0.02, 0.22


@dataclass
class _Agent:
    v_max: float
    radius: float


@dataclass
class _Problem:
    """`vo` / `project` 가 실제로 읽는 것만 가진 최소 스텁.

    두 모듈이 `Problem` 을 타입으로 import 하지 않는다는 사실 자체가 여기서
    확인된다 — 계층 규칙(tests/test_layering.py (a))의 실물 증거다.
    """
    agents: List[_Agent]
    dt: float = 0.10
    safety: float = 0.10

    @property
    def n(self) -> int:
        return len(self.agents)

    def min_sep(self, i: int, j: int) -> float:
        return self.agents[i].radius + self.agents[j].radius + self.safety


def _reference_vo(problem, i, pos, vel, phase, cand, near, alpha, cost):
    """controller.py:388-411 을 그대로 옮긴 참조 구현."""
    n, dt = problem.n, problem.dt
    a = problem.agents[i]
    for j in range(n):
        if j == i:
            continue
        R = problem.min_sep(i, j) + SOFT_MARGIN
        w = pos[j] - pos[i]
        if float(w @ w) > (R + (a.v_max + problem.agents[j].v_max) * TAU) ** 2:
            continue
        aij = float(alpha[i, j])
        if phase[j] == PHASE_DONE:
            aij = 1.0
        u = vel[i][None, :] + (cand - vel[i][None, :]) / aij - vel[j][None, :]
        t_c = _reference_ttc(w, u, R)
        hit = t_c < TAU
        cost[hit] += W_TTC / np.maximum(t_c[hit], dt)

    R_o = a.radius + OBST_PAD + 0.5 * SOFT_MARGIN
    for k in range(len(near)):
        w = near[k] - pos[i]
        if float(w @ w) > (R_o + a.v_max * TAU_OBST) ** 2:
            continue
        t_c = _reference_ttc(w, cand, R_o)
        hit = t_c < TAU_OBST
        cost[hit] += W_TTC_OBST / np.maximum(t_c[hit], dt)


def _reference_ttc(w, u, R):
    """controller.py:286-306 을 그대로 옮긴 참조 구현."""
    ww = float(w @ w)
    c = ww - R * R
    a = np.einsum("ij,ij->i", u, u)
    b = u @ w
    if c < 0.0:
        return np.zeros(len(u))
    disc = b * b - a * c
    out = np.full(len(u), np.inf)
    ok = (disc > 0.0) & (a > 1e-12) & (b > 0.0)
    if np.any(ok):
        t = (b[ok] - np.sqrt(disc[ok])) / a[ok]
        t = np.where(t < 0.0, np.inf, t)
        out[ok] = t
    return out


def _reference_project(problem, pos, v, phase, iters=12):
    """controller.py:466-499 을 그대로 옮긴 참조 구현."""
    from safety.geometry import seg_seg_min_dist
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
                continue
            v[k] *= 0.5
            if np.linalg.norm(v[k]) < 1e-3:
                v[k] = 0.0
    for k in range(n):
        if phase[k] not in (PHASE_DWELL, PHASE_DONE):
            v[k] = 0.0
    return v


def _cases(seed: int = 20260829, k: int = 100):
    """서로 가까운 배치를 절반 섞는다 — 전부 원거리 컷오프에 걸리면 대조가 무의미하다."""
    rng = np.random.default_rng(seed)
    for case in range(k):
        n = int(rng.integers(2, 5))
        problem = _Problem([_Agent(1.20, 0.30) for _ in range(n)])
        i = int(rng.integers(0, n))
        pos = rng.uniform(0.5, 9.5, size=(n, 2))
        if case % 2 == 0:
            pos[:] = pos[i] + rng.normal(0.0, 0.6, size=(n, 2))
        yield dict(
            problem=problem, i=i, pos=pos,
            vel=rng.uniform(-1.2, 1.2, size=(n, 2)),
            phase=rng.integers(0, 4, size=n),
            cand=rng.uniform(-1.2, 1.2, size=(65, 2)),
            near=pos[i] + rng.normal(0.0, 1.5, size=(int(rng.integers(1, 9)), 2)),
            alpha=rng.uniform(0.25, 1.0, size=(n, n)),
            base=rng.uniform(0.0, 3.0, size=65),
        )


def test_vo_cost_matches_the_inline_original_bit_for_bit():
    touched = 0
    for c in _cases():
        ref, got = c["base"].copy(), c["base"].copy()
        _reference_vo(c["problem"], c["i"], c["pos"], c["vel"], c["phase"],
                      c["cand"], c["near"], c["alpha"], ref)
        add_vo_cost(c["problem"], c["i"], c["pos"], c["vel"], c["phase"],
                    c["cand"], c["near"], c["alpha"], got)
        assert np.array_equal(ref, got), np.max(np.abs(ref - got))
        touched += int((ref != c["base"]).any())
    assert touched >= 80, f"VO 항이 실제로 붙은 케이스가 {touched} 뿐 — 대조가 헐겁다"


def test_ttc_matches_the_original():
    rng = np.random.default_rng(7)
    for _ in range(100):
        w = rng.normal(0.0, 2.0, size=2)
        u = rng.normal(0.0, 1.5, size=(65, 2))
        R = float(rng.uniform(0.2, 2.0))
        assert np.array_equal(_ttc(w, u, R), _reference_ttc(w, u, R))


def test_project_safe_matches_the_original():
    rng = np.random.default_rng(11)
    shrunk = 0
    for _ in range(100):
        n = int(rng.integers(2, 5))
        problem = _Problem([_Agent(1.20, 0.30) for _ in range(n)])
        pos = rng.uniform(2.0, 8.0, size=2) + rng.normal(0.0, 0.5, size=(n, 2))
        v = rng.uniform(-1.2, 1.2, size=(n, 2))
        phase = rng.integers(0, 4, size=n)
        ref = _reference_project(problem, pos, v.copy(), phase)
        got = _project_safe(problem, pos, v.copy(), phase)
        assert np.array_equal(ref, got)
        shrunk += int(not np.array_equal(ref, v))
    assert shrunk >= 50, f"속도가 실제로 줄어든 케이스가 {shrunk} 뿐 — 대조가 헐겁다"


def test_project_safe_never_moves_a_dwelling_or_finished_agent():
    """안전 투영은 "움직여 달라고 할 수 없는" 에이전트를 건드리지 않는다."""
    problem = _Problem([_Agent(1.20, 0.30), _Agent(1.20, 0.30)])
    pos = np.array([[5.0, 5.0], [5.2, 5.0]])          # 하한(0.70) 안쪽
    v = np.array([[1.0, 0.0], [-1.0, 0.0]])
    out = _project_safe(problem, pos, v.copy(), np.array([PHASE_DWELL, 0]))
    assert np.array_equal(out[0], v[0]), "dwell 중인 에이전트의 속도가 바뀌었다"
    assert np.linalg.norm(out[1]) == 0.0
