"""침범 탐지 — **지표가 실제로 잡는가** (S3-S §S-4).

이 파일이 없으면 "충돌 0건" 이 *안 일어났다* 인지 *못 잡는다* 인지 구별할 수
없다.  실제로 S3-A~A 3차의 "충돌 0건" 게이트 네 번은 항상 0 이 나오는 열을
보고 통과했다 (S3-B3 §7-1).

그래서 **고의로 침범하는 궤적을 만들어** 검증기가 잡는지 확인한다.
정상 궤적에서 오탐이 없는지도 함께 본다 — 한쪽만으로는 지표가 산다고 할 수 없다.

판정의 단일 출처는 `safety/validate.py` 이고 기준은 다음과 같다 (S3-S §S-2).

    에이전트 쌍   중심거리 >= min_sep(i,j) = r_i + r_j + safety = 0.70
    장애물/벽     중심~경계 >= robot_radius = 0.30

둘 다 **여유 1e-3 m** 를 둔다.  원본 검증기가 쓰던 값이고, dt=0.1 s 에서
v_max=1.2 m/s 로 한 스텝이 0.12 m 이므로 그 1/120 이다 — 수치 잡음보다
크고 물리적으로는 무시할 수 있는 폭이다.
"""
from __future__ import annotations

import numpy as np
import pytest

import safety.validate as V
from config import PHYSICS
from planning.traj import Trajectory
from planning.world import AgentSpec, Problem, World
from safety.geometry import Rect

TOL = 1e-3          # 검증기가 쓰는 여유.  §S-2 에서 확정했다.


def _problem(obstacles=None, n=2):
    """두 대가 서로 지나가는 최소 문제.  waypoint/goal 은 검사에 쓰지 않는다."""
    w = World(10.0, 10.0, obstacles=obstacles or [])
    agents = [AgentSpec(chr(ord("A") + i), start=(1.0 + i, 1.0),
                        waypoint=(5.0 + i, 5.0), goal=(9.0 - i, 9.0))
              for i in range(n)]
    return Problem("t", w, agents, deps=[])


def _traj(problem, pos: np.ndarray) -> Trajectory:
    """(T+1, n, 2) 위치에서 궤적을 만든다.  이벤트는 전부 '완료' 로 둔다 —
    여기서 보려는 것은 충돌 판정뿐이고 dwell/goal 오류는 별개다."""
    T, n = pos.shape[0] - 1, pos.shape[1]
    vel = np.zeros_like(pos)
    vel[:-1] = np.diff(pos, axis=0) / problem.dt
    return Trajectory(method="t", pos=pos, vel=vel, dt=problem.dt,
                      wp_in=np.zeros(n, int), wp_out=np.zeros(n, int),
                      done=np.full(n, T, int),
                      extra={"v_max": [a.v_max for a in problem.agents]})


# --------------------------------------------------------------------------- #
#  1. 에이전트 간 침범을 잡는가
# --------------------------------------------------------------------------- #
def test_agents_half_a_metre_apart_are_flagged():
    """두 대를 0.5 m 거리로 세워 둔다.  하한은 0.70 이므로 명백한 침범이다."""
    problem = _problem()
    pos = np.zeros((5, 2, 2))
    pos[:, 0] = [2.0, 2.0]
    pos[:, 1] = [2.5, 2.0]                       # 0.5 m — 하한 0.70 미만
    rep = V.validate(problem, _traj(problem, pos), check_dep=False)
    assert rep.n_agent_violations > 0, rep.errors
    assert rep.agent_clearance == pytest.approx(0.5)
    assert rep.first_violation_t >= 0
    assert not rep.valid


def test_the_agent_threshold_sits_exactly_at_min_sep():
    """하한 바로 위는 통과, 바로 아래는 걸린다 — 여유는 1e-3 이다."""
    problem = _problem()
    need = problem.min_sep(0, 1)
    assert need == pytest.approx(PHYSICS.min_sep) == pytest.approx(0.70)
    for gap, expect_violation in ((need + 1e-4, False),
                                  (need - TOL * 0.5, False),   # 여유 안쪽
                                  (need - TOL * 2.0, True)):
        pos = np.zeros((3, 2, 2))
        pos[:, 0] = [2.0, 2.0]
        pos[:, 1] = [2.0 + gap, 2.0]
        rep = V.validate(problem, _traj(problem, pos), check_dep=False)
        assert (rep.n_agent_violations > 0) is expect_violation, (gap, rep.errors)


def test_a_fast_crossing_between_samples_is_still_caught():
    """샘플 사이를 스쳐 지나가도 잡아야 한다 — 스윕 검사의 존재 이유다.

    두 대가 서로를 향해 달려 **끝점에서는** 멀지만 스텝 **중간에** 겹친다.
    """
    problem = _problem()
    pos = np.zeros((2, 2, 2))
    pos[0, 0] = [2.0, 2.0];  pos[1, 0] = [2.6, 2.0]
    pos[0, 1] = [2.6, 2.0];  pos[1, 1] = [2.0, 2.0]      # 서로 통과
    ends = min(np.linalg.norm(pos[0, 0] - pos[0, 1]),
               np.linalg.norm(pos[1, 0] - pos[1, 1]))
    assert ends == pytest.approx(0.6)          # 끝점만 보면 0.6
    rep = V.validate(problem, _traj(problem, pos), check_dep=False)
    assert rep.n_agent_violations > 0
    assert rep.agent_clearance < 1e-6, "스윕 최근접이 0 이어야 한다"


# --------------------------------------------------------------------------- #
#  2. 장애물 침범을 잡는가
# --------------------------------------------------------------------------- #
def test_a_path_through_an_obstacle_is_flagged():
    """장애물 한가운데를 지나는 궤적."""
    problem = _problem(obstacles=[Rect(3.0, 1.0, 5.0, 3.0)])
    pos = np.zeros((6, 2, 2))
    pos[:, 0, 0] = np.linspace(2.0, 6.0, 6)      # 사각형을 관통한다
    pos[:, 0, 1] = 2.0
    pos[:, 1] = [9.0, 9.0]                       # 다른 한 대는 멀리
    rep = V.validate(problem, _traj(problem, pos), check_dep=False)
    assert rep.n_obstacle_violations > 0, rep.errors
    assert rep.obstacle_clearance == pytest.approx(0.0)
    assert rep.first_violation_t >= 0


def test_grazing_a_wall_is_flagged():
    """벽에 반경보다 가까이 붙는 경우도 침범이다."""
    problem = _problem()
    pos = np.zeros((4, 2, 2))
    pos[:, 0] = [0.20, 5.0]                      # 왼쪽 벽까지 0.20 < 0.30
    pos[:, 1] = [9.0, 9.0]
    rep = V.validate(problem, _traj(problem, pos), check_dep=False)
    assert rep.n_obstacle_violations > 0
    assert rep.obstacle_clearance == pytest.approx(0.20)


def test_the_obstacle_threshold_sits_exactly_at_the_radius():
    problem = _problem()
    r = problem.agents[0].radius
    assert r == pytest.approx(PHYSICS.robot_radius) == pytest.approx(0.30)
    for d, expect_violation in ((r + 1e-4, False),
                                (r - TOL * 0.5, False),
                                (r - TOL * 2.0, True)):
        pos = np.zeros((3, 2, 2))
        pos[:, 0] = [d, 5.0]
        pos[:, 1] = [9.0, 9.0]
        rep = V.validate(problem, _traj(problem, pos), check_dep=False)
        assert (rep.n_obstacle_violations > 0) is expect_violation, (d, rep.errors)


# --------------------------------------------------------------------------- #
#  3. 정상 궤적에서 오탐이 없는가
# --------------------------------------------------------------------------- #
def test_a_clean_trajectory_raises_nothing():
    """탐지가 살아 있는 것과 오탐이 없는 것은 **둘 다** 필요하다."""
    problem = _problem(obstacles=[Rect(3.0, 1.0, 5.0, 3.0)])
    pos = np.zeros((6, 2, 2))
    pos[:, 0, 0] = np.linspace(1.0, 2.0, 6)
    pos[:, 0, 1] = 6.0
    pos[:, 1, 0] = np.linspace(8.0, 9.0, 6)
    pos[:, 1, 1] = 6.0
    rep = V.validate(problem, _traj(problem, pos), check_dep=False)
    assert rep.n_agent_violations == 0 and rep.n_obstacle_violations == 0
    assert rep.first_violation_t == -1


def test_the_real_planner_output_is_clean():
    """실제 플래너 출력에서도 오탐이 없어야 한다 (5종 중 하나)."""
    from planning.execute import WMConfig, run_wm_planner
    from planning.world import get_scenario
    problem = get_scenario("crossing2")
    res = run_wm_planner(problem, WMConfig(seed=0))
    rep = V.validate(problem, res.traj)
    assert rep.n_agent_violations == 0
    assert rep.n_obstacle_violations == 0
    assert rep.first_violation_t == -1


# --------------------------------------------------------------------------- #
#  4. B3 의 88행이 새 파이프라인에서도 잡히는가 (회귀 고정)
# --------------------------------------------------------------------------- #
def test_the_b3_obstacle_breach_is_still_detected():
    """S3-B3 가 낸 실제 침범을 재현해 잡히는지 확인한다.

    `B3_decel.csv` 의 최악 행(`n2_none_72203b7a`, obstacle_clearance 0.2436)을
    그대로 되살릴 수는 없다 — `wipeout_decelerate` 를 되돌렸기 때문이다.
    대신 **그때 관측된 값**(반경 0.30 에 0.2436 로 접근)을 궤적으로 재현해,
    같은 크기의 침범이 지금 파이프라인에서 잡히는지 못 박는다.
    """
    problem = _problem()
    pos = np.zeros((4, 2, 2))
    pos[:, 0] = [0.2436, 5.0]                    # B3 의 최악값
    pos[:, 1] = [9.0, 9.0]
    rep = V.validate(problem, _traj(problem, pos), check_dep=False)
    assert rep.n_obstacle_violations > 0, "B3 가 낸 크기의 침범을 놓쳤다"
    assert rep.obstacle_clearance == pytest.approx(0.2436)


# --------------------------------------------------------------------------- #
#  5. CSV 열이 status 와 독립적으로 살아남는가 (S-1)
# --------------------------------------------------------------------------- #
def test_violation_columns_survive_the_status_overwrite():
    """`status` 는 완주 실패면 덮어써진다.  판정 열은 그래도 남아야 한다.

    이것이 S3-B3 §7-1 이 드러낸 구멍의 정확한 지점이다.
    """
    from bench.generate import Instance
    from bench.run import FIELDS, _blank, _fill

    for col in ("n_agent_violations", "n_obstacle_violations",
                "min_agent_clearance", "min_obstacle_clearance",
                "first_violation_t"):
        assert col in FIELDS, col

    problem = _problem()
    pos = np.zeros((5, 2, 2))
    pos[:, 0] = [2.0, 2.0]
    pos[:, 1] = [2.5, 2.0]                       # 침범
    traj = _traj(problem, pos)
    rep = V.validate(problem, traj, check_dep=False)

    inst = Instance(instance_seed=0, n_agents=2, dep_mode="chain", size=10.0,
                    n_obstacles=(3, 5), couple_prob=1.0, couple_dist=0.55)
    row = _blank(inst, "wm_planner", 1.0, "abc1234", 0, 300.0)
    _fill(row, problem, traj, rep, 0.1, 1.0)
    assert row["status"] == "invalid"
    row["status"] = "unfinished:deadlock: no team progress"   # 실제로 일어나는 덮어쓰기

    assert int(row["n_agent_violations"]) > 0, "덮어쓰기에 판정이 지워졌다"
    assert float(row["min_agent_clearance"]) == pytest.approx(0.5)
    assert int(row["first_violation_t"]) >= 0
