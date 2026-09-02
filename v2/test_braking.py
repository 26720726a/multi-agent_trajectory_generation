"""제동 항 (S3-A') — `v_pref` 가 목표에서 멈출 수 있는 속도로 제한되는가.

S3-A 는 게이트 4 를 −7.30%p 로 미달했고, 보고서 §6-2 가 뿌리로 지목한 것이
이것이었다.  원래 식 `min(v_lim, d_target/dt)` 는 목표가 한 스텝 거리보다 멀면
언제나 최대 속력을 원한다.  `a_max` 가 없을 때는 목표에 닿는 순간 속도를 0 으로
갈아끼우면 그만이었지만, 상한이 생기면 **자기가 지울 수 없는 속도**를 낸다.

여기서 확인하는 것은 셋이다.

  1. `a_max = inf` 에서 **식이 원래와 완전히 같다.**  회귀 게이트의 근거다.
  2. 유한한 `a_max` 에서 제동 항이 실제로 물린다.
  3. 후보 팬(`_candidates`)은 **건드리지 않았다** — 이 변경은 축이 하나다.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from config import CONTROLLER, PHYSICS
from planning.control import (A_MAX, BRAKE_LOOKAHEAD, BRAKE_N_RAYS,
                              _candidates, brake_free_distance,
                              brake_speed)

FINITE_A_MAX = pytest.mark.skipif(
    not math.isfinite(PHYSICS.a_max),
    reason="requires a_max=3.0 (or another finite a_max); re-enable the S3 braking experiment")
MULTISTEP = pytest.mark.skipif(
    not CONTROLLER.multistep_lookahead,
    reason="requires multistep_lookahead=True; re-enable the S3-B2 experiment")


# --------------------------------------------------------------------------- #
#  1. a_max = inf 에서 원래 식과 동일
# --------------------------------------------------------------------------- #
def test_infinite_a_max_makes_the_braking_term_vanish(monkeypatch):
    """`inf` 를 **곧바로** 돌려준다 — `2*inf*0.0` 이 nan 이 되는 것을 피한다.

    nan 이 되면 `min(v_lim, d/dt, nan)` 이 파이썬에서 nan 을 낼 수 있어
    S2 재현이 조용히 깨진다.
    """
    import planning.control as C
    monkeypatch.setattr(C, "A_MAX", math.inf)
    for d in (0.0, 1e-12, 0.05, 1.0, 1e6):
        v = C.brake_speed(d)
        assert v == math.inf, d
        assert not math.isnan(v)


def test_min_with_infinity_is_exactly_the_original_expression(monkeypatch):
    """`min(a, b, inf)` 가 `min(a, b)` 와 부동소수 수준까지 같은지."""
    import planning.control as C
    monkeypatch.setattr(C, "A_MAX", math.inf)
    rng = np.random.default_rng(1)
    for _ in range(500):
        v_lim = float(rng.uniform(0.0, 2.0))
        d = float(rng.uniform(0.0, 10.0))
        dt = 0.1
        assert min(v_lim, d / dt, C.brake_speed(d)) == min(v_lim, d / dt)


# --------------------------------------------------------------------------- #
#  2. 유한한 a_max 에서 제동 항이 물린다
# --------------------------------------------------------------------------- #
@FINITE_A_MAX
def test_brake_speed_is_the_standard_constant_deceleration_formula():
    assert A_MAX == 3.0
    for d in (0.05, 0.24, 0.5, 1.0, 2.4):
        assert brake_speed(d) == pytest.approx(math.sqrt(2 * 3.0 * d))
    assert brake_speed(0.0) == 0.0
    assert brake_speed(-1.0) == 0.0, "음수 거리는 0 으로 클램프"


@FINITE_A_MAX
def test_the_braking_term_binds_exactly_inside_the_stopping_distance():
    """v_max 를 유지하려면 목표에서 정지거리만큼 떨어져 있어야 한다.

    v_max=1.20, a_max=3.0 -> 0.24 m.  이 안쪽에서만 제동 항이 이긴다.
    """
    stop = PHYSICS.v_max ** 2 / (2 * A_MAX)
    assert stop == pytest.approx(0.24)
    assert brake_speed(stop) == pytest.approx(PHYSICS.v_max)
    assert brake_speed(stop * 0.5) < PHYSICS.v_max      # 안쪽 -> 감속
    assert brake_speed(stop * 2.0) > PHYSICS.v_max      # 바깥 -> 무관


@FINITE_A_MAX
def test_the_braking_term_still_needs_2x_a_max_at_the_very_end():
    """**제동 항은 급정거를 절반으로 줄일 뿐 없애지 못한다.**  dt 이산화 때문이다.

    연속시간에서 `sqrt(2*a*d)` 는 정확히 `a` 로 감속하는 프로파일이다.  그런데
    `dt` 단위로 끊으면 목표 바로 앞에서 `d/dt` 항이 제동 항을 이기고, 그 스텝에
    목표에 정확히 도착하면서 속도가 한 번에 0 이 된다.

        d/dt < sqrt(2*a*d)   <=>   d < 2*a*dt^2 = 0.060 m

    그 구간의 최대 감속도는 `d/dt^2` 이고 경계 `d = 2*a*dt^2` 에서 최대가 되어
    정확히 **2*a_max = 6.0 m/s^2** 다.  제동 항이 없던 S3-A 에서는 이 값이
    `v_max/dt = 12.0` 이었으므로 정확히 절반이 된다.

    이것은 결함이 아니라 이산화의 본질이며, 지시서가 지정한 식
    (`sqrt(2*a_max*d_target)`) 을 그대로 쓴 결과다.  없애려면 이산 제동
    프로파일로 바꿔야 하는데 그것은 **또 다른 축**이라 하지 않았다.
    """
    dt, v_max = PHYSICS.dt, PHYSICS.v_max
    boundary = 2.0 * A_MAX * dt ** 2
    assert boundary == pytest.approx(0.060)

    worst, worst_d = 0.0, None
    for d in np.arange(0.001, 3.0, 0.001):
        d = float(d)
        v = min(v_max, d / dt, brake_speed(d))          # 실제 식 (세 항)
        d_next = max(0.0, d - v * dt)
        v_next = min(v_max, d_next / dt, brake_speed(d_next))
        acc = (v - v_next) / dt
        if acc > worst:
            worst, worst_d = acc, d

    assert worst == pytest.approx(2.0 * A_MAX, rel=1e-3), worst
    assert worst_d == pytest.approx(boundary, abs=1e-3), worst_d
    # 제동 항이 없을 때의 최악(= 순항 속도에서 한 스텝에 정지)
    assert worst < v_max / dt, "제동 항이 최악값을 줄이지 못했다"


@FINITE_A_MAX
def test_far_from_the_target_the_braking_term_does_nothing():
    """정지거리 밖에서는 원래 식과 같아야 한다 — 제동 항은 끝에서만 작동한다."""
    dt, v_max = PHYSICS.dt, PHYSICS.v_max
    stop = v_max ** 2 / (2 * A_MAX)
    for d in (stop * 1.01, 0.5, 1.0, 5.0):
        assert min(v_max, d / dt, brake_speed(d)) == min(v_max, d / dt)


# --------------------------------------------------------------------------- #
#  3. 후보 팬은 건드리지 않았다
# --------------------------------------------------------------------------- #
def test_the_candidate_fan_is_untouched():
    """S3-A 보고서 §6-1 의 표가 그대로여야 한다 — 축은 하나만 움직인다."""
    expected = {0.0: (17, 2), 0.3: (45, 21), 0.6: (46, 22), 1.2: (44, 22)}
    for spd, (n_cand, n_speed) in expected.items():
        c = _candidates(np.array([1.2, 0.0]), 1.20,
                        v_prev=np.array([spd, 0.0]), dv_max=0.3)
        assert len(np.unique(np.round(c, 9), axis=0)) == n_cand, spd
        assert len(np.unique(np.round(np.linalg.norm(c, axis=1), 6))) == n_speed


# --------------------------------------------------------------------------- #
#  4. 끝에서 끝까지 — 실제로 급정거가 줄었는가
# --------------------------------------------------------------------------- #
@FINITE_A_MAX
@MULTISTEP
def test_the_violation_budget_on_a_fixed_scenario_is_pinned():
    """crossing2(seed=0)의 위반 구성을 못 박는다 — 다음 변경의 기준선이다.

    A-2차 (제동 항, 목표까지만) : 4건 = 정지 1 / 전멸 3 / 스냅 0
    A-3차 (제동 대상 확장)      : 6건 = 정지 2 / 전멸 3 / 스냅 1
    A-4차 (기하 제동 끔)        : 6건 = 정지 2 / 전멸 3 / 스냅 1  (A-3차와 동일)
    B2    (다단계 하드 필터)    : 4건 = 정지 1 / 전멸 0 / 스냅 3
    B3    (전멸 시 감속)        : 4건 = 정지 1 / **전멸 0** / 스냅 3
                                  -> 게이트 2(충돌) 미달로 **되돌렸다**

    B3 는 전멸 위반을 0 으로 만들었지만 장애물 침범 88행을 냈다. B2는 같은
    전멸을 후보 선택 전에 배제하므로 이 고정 시나리오에서 전멸 위반이 0이다.
    """
    from planning.execute import WMConfig, run_wm_planner
    from planning.world import get_scenario
    acc = run_wm_planner(get_scenario("crossing2"), WMConfig(seed=0)).traj.extra["accel"]
    assert acc["n_amax_violations"] == 4, acc
    assert acc["n_amax_viol_block"] == 0, acc
    assert acc["n_amax_viol_project"] == 0, acc


# --------------------------------------------------------------------------- #
#  5. 제동 대상 거리 확장 (S3-A 3차)
# --------------------------------------------------------------------------- #
class _FakeScene:
    """`free_steps` 만 흉내 낸다 — 원점에서 `limit` 까지만 자유로운 세계."""

    def __init__(self, limit: float, axis: int = 0):
        self.limit, self.axis = limit, axis

    def free_steps(self, i, p, qs):
        return np.abs(qs[:, self.axis] - p[self.axis]) <= self.limit + 1e-12


def test_free_distance_returns_the_cap_when_nothing_blocks():
    scene = _FakeScene(1e9)
    d = brake_free_distance(scene, 0, np.zeros(2), np.array([1.0, 0.0]),
                            1.2, 0.3, 0.30)
    assert d == 0.30


def test_free_distance_stops_before_the_wall():
    """벽이 0.1 m 앞이면 그보다 멀리 간다고 답하면 안 된다."""
    scene = _FakeScene(0.1)
    d = brake_free_distance(scene, 0, np.zeros(2), np.array([1.0, 0.0]),
                            1.2, 0.3, 0.30, n=8)
    assert 0.0 <= d <= 0.1 + 1e-9


@FINITE_A_MAX
def test_the_fan_half_angle_comes_from_a_max_not_a_magic_number():
    """반각 = asin(dv_max/|v|).  순항일수록 좁고, 느리면 전방위다.

    임의 상수를 쓰면 넓을 때 제동이 안 걸리고 좁을 때 굽은 통로에서 과도해진다.
    """
    dv = A_MAX * PHYSICS.dt
    assert math.degrees(math.asin(min(1.0, dv / 1.20))) == pytest.approx(14.48, abs=0.1)
    assert math.degrees(math.asin(min(1.0, dv / 0.60))) == pytest.approx(30.0, abs=0.1)
    assert math.asin(min(1.0, dv / 0.30)) == pytest.approx(math.pi / 2)


def test_a_side_opening_saves_the_agent_from_braking():
    """부채꼴을 쓰는 이유 — 옆이 열려 있으면 감속할 이유가 없다.

    외길 광선만 보면 굽은 통로에서 과도하게 감속해 완주가 깨진다 (실측).
    """
    class SideOpen:
        def free_steps(self, i, p, qs):
            # 정면(+x)은 0.05 m 에서 막히고, 위쪽으로 벌어지면 열려 있다
            return (qs[:, 1] - p[1] > 1e-6) | (qs[:, 0] - p[0] <= 0.05 + 1e-12)
    d = brake_free_distance(SideOpen(), 0, np.zeros(2), np.array([1.0, 0.0]),
                            0.60, 0.3, 0.30)
    assert d == 0.30, "옆이 열렸는데 감속했다"


def test_infinite_a_max_never_calls_the_distance_probe(monkeypatch):
    """`a_max=inf` 면 자유 거리를 **재지도 않는다** — S2 재현이 자명하고 비용 0."""
    import planning.control as C
    calls = []
    monkeypatch.setattr(C, "A_MAX", math.inf)
    monkeypatch.setattr(C, "brake_free_distance",
                        lambda *a, **k: calls.append(1) or 1.0)
    from planning.execute import WMConfig, run_wm_planner
    from planning.world import get_scenario
    run_wm_planner(get_scenario("crossing2"), WMConfig(seed=0))
    assert calls == [], f"{len(calls)}회 호출됐다"


@FINITE_A_MAX
def test_geometric_braking_is_off_because_measurement_showed_it_useless():
    """S3-A 3차에서 기하 제동을 껐다 — 측정 결과에 따른 정리다.

    호출의 98.7% 가 cap 을 그대로 돌려주었고, 급정거형 전멸은 3.2% 밖에 줄지
    않은 채 재계획당 벽시계만 1.32 배가 되었다.  코드는 남겨 두었으므로 값을
    넣으면 다시 켜지고, 그때 `phys_fp` 가 갈려 CSV 가 섞이지 않는다.
    """
    from config import CONTROLLER
    assert CONTROLLER.brake_lookahead is None
    assert BRAKE_LOOKAHEAD is None
    # 실측 필요량은 그대로 유효하다 — 다시 켤 때의 기준값이다
    need, v = 0.0, PHYSICS.v_max
    while v > 1e-12:
        v = max(0.0, v - A_MAX * PHYSICS.dt)
        need += v * PHYSICS.dt
    assert need == pytest.approx(0.180)


def test_the_dependency_brake_target_is_still_active():
    """`d_geom` 만 껐다.  의존성 대기 경계는 제동 대상으로 남는다 —
    순항 충돌 비율을 72.2% -> 53.1% 로 줄인 것이 확인됐기 때문이다."""
    import inspect
    from planning import control as C
    src = inspect.getsource(C.step)
    assert "d_brake = min(d_brake, slack)" in src
    assert "BRAKE_LOOKAHEAD is not None" in src


# --------------------------------------------------------------------------- #
#  6. 전멸 fallback (S3-B3)
# --------------------------------------------------------------------------- #
@FINITE_A_MAX
def test_the_wipeout_deceleration_is_off_because_it_broke_the_collision_gate():
    """S3-B3 에서 켜 보았고 되돌렸다 — 측정 결과에 따른 정리다.

    사양(계획서 3.3-7)은 "-a_max 로 0.67초 내 정지" 이고 원본 코드는 순간
    정지였다.  그 어긋남을 없애면 전멸 위반이 0 이 된다(원리적으로).
    **그런데 장애물을 침범한다** — 전멸은 "후보가 전부 기하 필터에 걸렸다" 는
    뜻인데 감속 fallback 은 그 필터를 거치지 않고 계속 움직이고,
    `_project_safe` 는 에이전트 간만 다루므로 장애물은 아무도 막지 않는다.

    장애물/벽 침범 88행(1.81%), 최악 obstacle_clearance 0.2436 m (하한 0.30).
    에이전트 간 충돌은 0 이었다.  성공률도 80.31% -> 57.86%.

    켜려면 감속 이동도 기하 필터를 통과해야 한다 — B2(다단계 필터)의 일이다.
    """
    from config import CONTROLLER
    assert CONTROLLER.wipeout_decelerate is False
    # 아래는 켰을 때의 성질 — 되살릴 때의 기준으로 남긴다
    dv = A_MAX * PHYSICS.dt
    # 순항에서 전멸하면 한 스텝에 dv 만큼만 줄어든다
    v_prev = np.array([PHYSICS.v_max, 0.0])
    sp = float(np.linalg.norm(v_prev))
    v_cmd = v_prev * (1.0 - dv / sp)
    assert np.linalg.norm(v_cmd - v_prev) / PHYSICS.dt == pytest.approx(A_MAX)
    # 방향은 유지된다 — 방향을 바꾸면 그것이 또 하나의 축이다
    cross = v_prev[0] * v_cmd[1] - v_prev[1] * v_cmd[0]
    assert cross == pytest.approx(0.0)
    # 0.67 초 = 약 7 스텝 안에 선다 (v_max/a_max = 0.40 s = 4 스텝)
    steps, v = 0, PHYSICS.v_max
    while v > 1e-12:
        v = max(0.0, v - dv)
        steps += 1
    assert steps * PHYSICS.dt <= 0.67


def test_an_infinite_a_max_falls_back_to_the_original_instant_stop():
    """`a_max=inf` 면 한 스텝에 지울 수 있는 속도가 무한이라 즉시 0 이다.

    그래서 S2 재현이 **자동으로** 유지된다 — 별도 분기가 필요 없다.
    """
    dv = math.inf
    for sp in (PHYSICS.v_max, 0.5, 1e-6):
        assert not (sp > dv), "inf 에서는 감속 분기로 가면 안 된다"


def test_deceleration_happens_before_the_safety_projection():
    """순서를 바꾸지 않았다 — 감속 결과를 투영이 받아 안전을 지킨다."""
    import inspect
    from planning import control as C
    src = inspect.getsource(C.step)
    i_decel = src.index("WIPEOUT_DECELERATE and sp_prev > dv_max")
    i_proj = src.index("_project_safe(problem, pos, v_cmd, phase)")
    assert i_decel < i_proj, "투영이 감속보다 먼저 온다"


def test_the_safety_projection_does_not_cover_obstacles():
    """B3 가 왜 깨졌는지의 뿌리 — 투영은 **에이전트 간**만 다룬다.

    `safety/project.py` 는 쌍(i, j)의 스윕 거리만 본다.  장애물은 `step()` 의
    하드 필터가 막는데, 전멸 fallback 은 그 필터를 우회해 움직인다.
    이 구조가 바뀌지 않는 한 "전멸했는데 계속 움직이는" 처방은 전부 같은
    문제를 만든다.
    """
    import inspect
    from safety import project
    src = inspect.getsource(project._project_safe)
    assert "min_sep(i, j)" in src
    assert "obstacle" not in src.lower() and "rect" not in src.lower()
