"""가속도 상한 (S3-A) — 도입과 계측.

원본에는 가속도 제한이 **없었다** (S0 §C-1).  컨트롤러는 매 스텝 후보 팬에서
임의의 속도를 고를 수 있었고, 급변은 `W_TURN` 이 비용으로만 눌렀다.
여기서 확인하는 것은 셋이다.

  1. `a_max = inf` 에서 **아무것도 달라지지 않는다.**  구현이 옳은지 확인하는
     유일한 수단이다 (S3 §A-1).
  2. 유한한 `a_max` 에서 후보가 실제로 갇힌다 — 개수는 유지한 채로.
  3. 계측이 동작을 바꾸지 않는다.  `stats=None` 과 넘겼을 때가 같아야 한다.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from config import CONTROLLER, PHYSICS
from planning.control import _candidates, multistep_free_steps, new_stats

FINITE_A_MAX = pytest.mark.skipif(
    not math.isfinite(PHYSICS.a_max),
    reason="requires a_max=3.0 (or another finite a_max); re-enable the S3 acceleration experiment")
MULTISTEP = pytest.mark.skipif(
    not CONTROLLER.multistep_lookahead,
    reason="requires multistep_lookahead=True; re-enable the S3-B2 experiment")


def _fan(v_pref, v_max, **kw):
    return _candidates(np.asarray(v_pref, float), v_max, **kw)


# --------------------------------------------------------------------------- #
#  1. a_max = inf 는 아무 일도 하지 않는다
# --------------------------------------------------------------------------- #
def test_infinite_dv_max_is_bit_identical_to_the_unlimited_fan():
    rng = np.random.default_rng(3)
    for _ in range(100):
        v_pref = rng.uniform(-1.2, 1.2, size=2)
        v_prev = rng.uniform(-1.2, 1.2, size=2)
        base = _fan(v_pref, 1.20)
        for kw in ({}, {"v_prev": v_prev}, {"v_prev": v_prev, "dv_max": math.inf},
                   {"dv_max": 0.3}):                       # v_prev 없이 dv_max 만
            assert np.array_equal(base, _fan(v_pref, 1.20, **kw)), kw


@FINITE_A_MAX
def test_the_shipped_a_max_is_the_plan_target():
    """S3-A 에서 계획서 §2 목표값 3.0 을 넣었다.

    이 테스트가 빨개지면 값이 또 바뀐 것이다 — 의도한 것인지 확인할 것.
    값 변경이 **눈에 띄는 커밋**으로 남게 하는 장치다.
    """
    assert PHYSICS.a_max == 3.0
    assert PHYSICS.braking_distance == pytest.approx(1.20 ** 2 / (2 * 3.0))


def test_reverting_a_max_to_infinity_restores_the_S2_fan():
    """되돌릴 수 있어야 S2 재현이 계속 가능하다."""
    v_pref, v_prev = np.array([1.0, 0.3]), np.array([-0.4, 0.8])
    assert np.array_equal(
        _fan(v_pref, 1.20, v_prev=v_prev, dv_max=math.inf),
        _fan(v_pref, 1.20))


def test_infinite_dv_max_uses_the_original_one_step_filter_call():
    """B2는 inf에서 기존 `free_steps(pos, pos + v*dt)` 경로를 그대로 쓴다."""
    class Scene:
        def __init__(self):
            self.calls = []

        def free_steps(self, i, p, qs):
            self.calls.append((i, p.copy(), qs.copy()))
            return np.array([True, False])

    scene = Scene()
    pos = np.array([2.0, 3.0])
    cand = np.array([[1.2, 0.0], [0.0, 1.2]])
    keep, calls = multistep_free_steps(scene, 4, pos, cand, 0.1, math.inf)
    assert np.array_equal(keep, [True, False])
    assert calls == len(scene.calls) == 1
    assert np.array_equal(scene.calls[0][2], pos[None, :] + cand * 0.1)


@MULTISTEP
def test_multistep_filter_rejects_a_candidate_that_cannot_stop_before_a_wall():
    """1.2 -> .9 -> .6 -> .3 -> 0의 0.18m 제동 궤적을 끝까지 본다."""
    class Wall:
        def free_steps(self, i, p, qs):
            return qs[:, 0] <= 0.15 + 1e-12

    keep, calls = multistep_free_steps(
        Wall(), 0, np.zeros(2), np.array([[1.2, 0.0], [0.3, 0.0]]), 0.1, 0.3)
    assert np.array_equal(keep, [False, True])
    assert calls == 2, "첫 단계에서 막힌 후보는 뒤 단계를 검사하지 않는다"


# --------------------------------------------------------------------------- #
#  2. 유한한 a_max 에서 후보가 갇힌다
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dv_max", [0.05, 0.3, 1.0])
def test_every_candidate_is_within_dv_max_of_the_previous_velocity(dv_max):
    rng = np.random.default_rng(5)
    for _ in range(60):
        v_pref = rng.uniform(-1.2, 1.2, size=2)
        v_prev = rng.uniform(-1.2, 1.2, size=2)
        cand = _fan(v_pref, 1.20, v_prev=v_prev, dv_max=dv_max)
        d = np.linalg.norm(cand - v_prev[None, :], axis=1)
        assert d.max() <= dv_max * (1 + 1e-9), d.max()


def test_clipping_keeps_the_candidate_count():
    """탈락이 아니라 클리핑을 고른 이유 — 후보 수가 상황에 따라 흔들리면 안 된다.

    하나도 안 남는 경우가 생기면 "기하적으로 막혔다" 와 "가속이 모자랐다" 가
    같은 증상(정지)으로 뭉개진다.
    """
    from config import CONTROLLER
    v_pref, v_prev = np.array([1.0, 0.2]), np.array([-0.9, 0.4])
    for dv in (1e-6, 0.01, 0.3, 10.0, math.inf):
        cand = _fan(v_pref, 1.20, v_prev=v_prev, dv_max=dv)
        assert len(cand) == CONTROLLER.n_candidates == 65


def test_clipping_moves_each_candidate_straight_towards_the_previous_velocity():
    """방향을 유지한 채 크기만 줄인다 — 팬의 각도 다양성이 살아 있어야 한다."""
    v_pref, v_prev = np.array([1.2, 0.0]), np.array([0.0, 0.0])
    full = _fan(v_pref, 1.20, v_prev=v_prev, dv_max=math.inf)
    clip = _fan(v_pref, 1.20, v_prev=v_prev, dv_max=0.1)
    for a, b in zip(full, clip):
        na, nb = np.linalg.norm(a - v_prev), np.linalg.norm(b - v_prev)
        if na <= 0.1 + 1e-12:
            assert np.allclose(a, b)                       # 이미 안쪽이면 그대로
        else:
            assert nb == pytest.approx(0.1)
            cross = (a - v_prev)[0] * (b - v_prev)[1] - (a - v_prev)[1] * (b - v_prev)[0]
            assert abs(cross) < 1e-12, "방향이 틀어졌다"


def test_a_candidate_already_inside_the_limit_is_untouched():
    v_prev = np.array([0.5, 0.5])
    cand = _fan(np.array([0.5, 0.5]), 1.20, v_prev=v_prev, dv_max=5.0)
    assert np.array_equal(cand, _fan(np.array([0.5, 0.5]), 1.20))


# --------------------------------------------------------------------------- #
#  3. 계측
# --------------------------------------------------------------------------- #
def test_measuring_does_not_change_the_trajectory():
    """계측은 관찰이지 개입이 아니다."""
    from planning.control import SceneCache, initial_state, step
    from planning.world import get_scenario
    from planning.worldmodel import WorldModel

    problem = get_scenario("crossing2")
    wm = WorldModel(problem, seed=0)
    mode = wm.sample_modes(max_modes=1)[0]
    routes = [wm.library[i][0] for i in range(problem.n)]
    alpha, ss, side = mode.alpha(problem.n), mode.speed_scale(problem.n), \
        mode.side(problem.n)
    scene = SceneCache(problem)

    a = initial_state(problem)
    b = initial_state(problem)
    stats = new_stats()
    for _ in range(60):
        a = step(problem, scene, routes, a, alpha, ss, side)
        b = step(problem, scene, routes, b, alpha, ss, side, stats)
        assert np.array_equal(a.pos, b.pos) and np.array_equal(a.vel, b.vel)
    assert stats["steps"] == 60 * problem.n


def test_max_accel_is_measured_even_when_a_max_is_infinite():
    """`a_max=inf` 인 기준선에서도 "얼마나 급한가" 는 물어야 한다.

    이 값이 목표 a_max 를 크게 넘으면, 상한을 도입하는 순간 실제로 물릴 것이라는
    뜻이다.  S3-A 를 돌리기 전에 알 수 있는 유일한 신호다.
    """
    from planning.execute import WMConfig, run_wm_planner
    from planning.world import get_scenario
    res = run_wm_planner(get_scenario("crossing2"), WMConfig(seed=0))
    acc = res.traj.extra["accel"]
    assert acc["steps"] > 0
    assert acc["max_accel"] > 0.0
    if math.isinf(PHYSICS.a_max):
        assert acc["n_amax_violations"] == 0, "inf 에서는 위반이 있을 수 없다"


def test_executed_stats_are_summed_over_segments_not_over_imagined_rollouts():
    """상상만 하고 버린 rollout 은 세지 않는다.

    묻는 것은 "실현 불가능한 가속에 **의존했는가**" 이므로, 실제로 실행된 명령만
    답이 된다.  `steps` 가 실행된 궤적 길이 x 에이전트 수와 맞는지로 확인한다.
    """
    from planning.execute import WMConfig, run_wm_planner
    from planning.world import get_scenario
    problem = get_scenario("crossing2")
    res = run_wm_planner(problem, WMConfig(seed=0))
    assert res.traj.extra["accel"]["steps"] == res.traj.T * problem.n
    assert res.traj.extra["accel"]["n_free_steps_calls"] >= \
        res.traj.extra["accel"]["steps"] / problem.n


def test_violation_causes_are_counted_separately_and_partition_the_total():
    """원인이 다르면 대응도 다르다 — 셋은 배타적이고 합이 전체와 같아야 한다.

      stop     HOI dwell 진입 / goal 도착의 순간 정지 (과업 정의라 감속 구간이 없다)
      block    후보 65개가 전부 기하 필터에 걸려 강제로 선 것
      snap     W/G 정확 착지 대입
      project  `_project_safe` 가 넘긴 것  <- S3 §A-2 가 재라고 한 숫자
    """
    from planning.execute import WMConfig, run_wm_planner
    from planning.world import get_scenario

    s = new_stats()
    assert set(s) == {"steps", "n_amax_violations", "n_amax_viol_stop",
                      "n_amax_viol_block", "n_amax_viol_snap",
                      "n_amax_viol_project", "n_project_active",
                      "n_blocked", "n_free_steps_calls", "max_accel"}
    assert all(v == 0.0 for v in s.values())

    for name in ("crossing2", "chain3"):
        acc = run_wm_planner(get_scenario(name), WMConfig(seed=0)).traj.extra["accel"]
        parts = (acc["n_amax_viol_stop"] + acc["n_amax_viol_block"]
                 + acc["n_amax_viol_snap"] + acc["n_amax_viol_project"])
        assert parts == acc["n_amax_violations"], (name, acc)


def test_the_safety_projection_is_measured_but_not_limited():
    """S3 §A-2 의 결정: 안전을 우선하되 **위반을 계측한다**.

    투영을 a_max 로 제한하면 충돌 방지가 깨지므로 제한하지 않는다.  대신
    "얼마나 자주 개입했는가"(n_project_active)와 "그 개입이 a_max 를
    넘겼는가"(n_amax_viol_project)를 따로 센다.
    """
    from planning.execute import WMConfig, run_wm_planner
    from planning.world import get_scenario
    acc = run_wm_planner(get_scenario("crossing2"), WMConfig(seed=0)).traj.extra["accel"]
    assert acc["n_project_active"] > 0, "이 시나리오에서 투영은 반드시 개입한다"
    assert acc["n_project_active"] <= acc["steps"]
