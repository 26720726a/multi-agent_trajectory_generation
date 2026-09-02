#!/usr/bin/env python3
"""묶음 2 대조: 원본 controller 의 인라인 VO 블록 vs v2 safety/vo.add_vo_cost.

원본 저장소를 **읽기 전용으로** import 해서 원본 상수(`TAU`, `SOFT_MARGIN`,
`W_TTC`, ...)와 원본 `_ttc` 를 그대로 쓰는 참조 구현을 만들고, 무작위 입력
100 케이스에서 v2 추출본과 값을 대조한다.  판정은 allclose 가 아니라 `==` 다
(safety/vo.py 의 docstring 참조).
"""
import os, sys
import numpy as np

ORIG = "/home/kjs/Desktop/kuaicv/cvpr2027/Mahoi-WM/mahoi-wm"
V2 = "/home/kjs/Desktop/kuaicv/cvpr2027/MAHOI-WM_v2"
sys.path.insert(0, V2)
sys.path.insert(0, ORIG)

import mahoi.wm.controller as ctrl          # 원본
from safety.vo import add_vo_cost           # v2
from safety.project import _project_safe as v2_project


def reference_inline(problem, i, pos, vel, phase, cand, near, alpha, cost):
    """controller.py:388-411 을 원본 상수/함수 그대로 붙여넣은 것."""
    n, dt = problem.n, problem.dt
    a = problem.agents[i]
    for j in range(n):
        if j == i:
            continue
        R = problem.min_sep(i, j) + ctrl.SOFT_MARGIN
        w = pos[j] - pos[i]
        if float(w @ w) > (R + (a.v_max + problem.agents[j].v_max) * ctrl.TAU) ** 2:
            continue
        aij = float(alpha[i, j])
        if phase[j] == ctrl.PHASE_DONE:
            aij = 1.0
        u = vel[i][None, :] + (cand - vel[i][None, :]) / aij - vel[j][None, :]
        t_c = ctrl._ttc(w, u, R)
        hit = t_c < ctrl.TAU
        cost[hit] += ctrl.W_TTC / np.maximum(t_c[hit], dt)

    R_o = a.radius + ctrl.OBST_PAD + 0.5 * ctrl.SOFT_MARGIN
    for k in range(len(near)):
        w = near[k] - pos[i]
        if float(w @ w) > (R_o + a.v_max * ctrl.TAU_OBST) ** 2:
            continue
        t_c = ctrl._ttc(w, cand, R_o)
        hit = t_c < ctrl.TAU_OBST
        cost[hit] += ctrl.W_TTC_OBST / np.maximum(t_c[hit], dt)


def main():
    from mahoi.world import random_problem          # 원본의 진짜 Problem
    rng = np.random.default_rng(20260829)
    bad = 0
    ttc_checked = 0
    for case in range(100):
        problem = random_problem(int(rng.integers(0, 10_000)),
                                 n_agents=int(rng.integers(2, 5)))
        n = problem.n
        i = int(rng.integers(0, n))
        pos = rng.uniform(0.5, 9.5, size=(n, 2))
        # 절반은 서로 아주 가깝게 — 원거리 컷오프에 다 걸리면 대조가 무의미하다
        if case % 2 == 0:
            pos[:] = pos[i] + rng.normal(0.0, 0.6, size=(n, 2))
        vel = rng.uniform(-1.2, 1.2, size=(n, 2))
        phase = rng.integers(0, 4, size=n)
        cand = rng.uniform(-1.2, 1.2, size=(65, 2))
        near = pos[i] + rng.normal(0.0, 1.5, size=(int(rng.integers(1, 9)), 2))
        alpha = rng.uniform(0.25, 1.0, size=(n, n))
        base = rng.uniform(0.0, 3.0, size=65)

        ca, cb = base.copy(), base.copy()
        reference_inline(problem, i, pos, vel, phase, cand, near, alpha, ca)
        add_vo_cost(problem, i, pos, vel, phase, cand, near, alpha, cb)
        if not np.array_equal(ca, cb):
            bad += 1
            print(f"  case {case}: MISMATCH  max|d|={np.max(np.abs(ca - cb)):.3e}")
        if np.isfinite(ca).all() and (ca != base).any():
            ttc_checked += 1

        # _ttc 자체도 직접 대조한다
        w = rng.normal(0, 2, size=2)
        u = rng.normal(0, 1.5, size=(65, 2))
        R = float(rng.uniform(0.2, 2.0))
        if not np.array_equal(ctrl._ttc(w, u, R), __import__("safety.vo", fromlist=["_ttc"])._ttc(w, u, R)):
            bad += 1
            print(f"  case {case}: _ttc MISMATCH")

    # ---- _project_safe 도 같은 방식으로 100 케이스 대조 --------------------- #
    pbad = 0
    for case in range(100):
        problem = random_problem(int(rng.integers(0, 10_000)),
                                 n_agents=int(rng.integers(2, 5)))
        n = problem.n
        c = rng.uniform(2.0, 8.0, size=2)
        pos = c + rng.normal(0.0, 0.5, size=(n, 2))      # 일부러 겹치게 둔다
        v = rng.uniform(-1.2, 1.2, size=(n, 2))
        phase = rng.integers(0, 4, size=n)
        a = ctrl._project_safe(problem, pos, v.copy(), phase)
        b = v2_project(problem, pos, v.copy(), phase)
        if not np.array_equal(a, b):
            pbad += 1
            print(f"  project case {case}: MISMATCH max|d|={np.max(np.abs(a-b)):.3e}")
    print(f"_project_safe cases=100  mismatches={pbad}")
    bad += pbad

    print(f"cases=100  VO 항이 실제로 붙은 케이스={ttc_checked}  mismatches={bad}")
    print("RESULT:", "IDENTICAL" if bad == 0 else "DIFFERENT")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
