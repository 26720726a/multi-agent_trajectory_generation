"""안전 불변식 — 어떤 개선을 하든 여기는 절대 깨지면 안 된다.

세 가지가 걸려 있다.

  1. agent 간 충돌은 `controller._project_safe` 가 매 스텝 속도를 줄여 *구조적으로*
     막는다.  "드물게 일어난다"가 아니라 "일어날 수 없다"가 설계다.  A2 에서 이
     함수를 벡터화할 때 이 성질이 조용히 사라지기 쉬우므로 테스트로 묶어 둔다.
  2. 검증은 `mahoi/validate.py` 가 한다.  플래너 내부를 전혀 보지 않고 실행된
     좌표만 보는 독립 검증기라, 새 플래너가 자기 채점을 하지 않는다.
  3. HOI dwell 과 dependency 순서는 과제 정의(C1 / C4)라 최적화 대상이 아니다.

빠르게 돌도록 mode 수와 시나리오를 줄였다.  전체 회귀는 test_regression.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from mahoi import validate as V
from mahoi.wm.execute import WMConfig, run_wm_planner
from mahoi.world import get_scenario

FAST = dict(horizon_s=4.0, replan_s=1.0, max_modes=4, keep_modes=4,
            k_routes=2, seed=0, time_budget_s=120.0)


@pytest.fixture(scope="module", params=["crossing2", "deadlock2"])
def run(request):
    problem = get_scenario(request.param)
    res = run_wm_planner(problem, WMConfig(**FAST))
    return problem, res, V.validate(problem, res.traj)


def test_independent_validator_passes(run):
    problem, _, rep = run
    assert rep.valid, "독립 검증기 실패:\n  " + "\n  ".join(rep.errors[:6])


def test_no_agent_collisions(run):
    """구조적 보장이 실제로 성립하는지."""
    problem, _, rep = run
    assert rep.n_agent_violations == 0
    assert rep.agent_clearance >= problem.min_sep(0, 1) - 1e-3


def test_no_obstacle_collisions(run):
    problem, _, rep = run
    assert rep.n_obstacle_violations == 0
    assert rep.obstacle_clearance >= min(a.radius for a in problem.agents) - 1e-3


def test_dependency_order_holds(run):
    """C4 — 선행 agent 가 HOI 를 끝낸 뒤에만 후행이 waypoint 에 진입한다."""
    problem, res, rep = run
    assert rep.n_dep_violations == 0
    ws, we = res.traj.event_steps()
    for (i, j) in problem.deps:
        assert ws[j] >= we[i], (
            f"{problem.agents[j].name} 가 {problem.agents[i].name} 보다 먼저 진입")


def test_speed_limit_never_exceeded(run):
    problem, res, _ = run
    step = np.linalg.norm(np.diff(res.traj.pos, axis=0), axis=2)   # (T, n)
    for i, a in enumerate(problem.agents):
        assert step[:, i].max() <= a.v_max * problem.dt + 1e-6


def test_hoi_dwell_is_held(run):
    """C1 — waypoint 에 정확히 도달해 d 초 정지한다.  회피 대상이 아니라 과제다."""
    problem, res, _ = run
    ws, we = res.traj.event_steps()
    for i, a in enumerate(problem.agents):
        need = int(round(a.dwell / problem.dt))
        assert we[i] - ws[i] == need, f"{a.name}: HOI 길이가 {a.dwell}s 가 아님"
        held = res.traj.pos[ws[i]:we[i] + 1, i, :]
        assert np.allclose(held, np.asarray(a.waypoint), atol=1e-3), \
            f"{a.name}: dwell 중에 waypoint 를 벗어남"


def test_everyone_reaches_the_goal(run):
    problem, res, _ = run
    assert res.traj.feasible
    assert np.all(res.traj.done >= 0)
    for i, a in enumerate(problem.agents):
        assert np.linalg.norm(res.traj.pos[-1, i] - np.asarray(a.goal)) < 0.25


def test_deterministic_for_a_fixed_seed():
    """A3 병렬화 뒤에도 같은 seed 는 같은 결과를 내야 한다."""
    problem = get_scenario("crossing2")
    a = run_wm_planner(problem, WMConfig(**FAST)).traj
    b = run_wm_planner(problem, WMConfig(**FAST)).traj
    assert a.pos.shape == b.pos.shape
    assert np.allclose(a.pos, b.pos)
