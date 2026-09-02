"""Phase 0 계약 — Track A 와 Track B 가 주고받는 자료 구조를 동결한다.

두 사람이 병렬로 작업하는 근거는 "World Model 이 Rollout 을 만들고 Planner 가
그것만 읽는다"는 좁은 경계 하나뿐이다.  그 경계가 소리 없이 움직이면 분업이
무너지므로, 여기서 필드 집합을 명시적으로 못박는다.

이 테스트가 깨지면 둘 중 하나다.

  * 정말로 계약을 바꿔야 한다  -> 양쪽이 합의하고 이 파일을 함께 수정한다.
  * 한쪽이 상대 모르게 바꿨다  -> 되돌린다.

새 정보를 실어 나르고 싶을 때는 필드를 추가하지 말고 `Trajectory.extra` /
`Rollout.remaining` 딕셔너리를 쓴다.  그러라고 만들어 둔 자리다.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from planning.control import (PHASE_DONE, PHASE_DWELL, PHASE_TO_GOAL,
                              PHASE_TO_WP, TeamState, initial_state)
from planning.planner import Cost, CostWeights, Planner
from planning.traj import Trajectory
from planning.world import get_scenario
from planning.worldmodel import PlanMode, Rollout, WorldModel


def _fields(cls) -> set:
    return {f.name for f in dataclasses.fields(cls)}


# --------------------------------------------------------------------------- #
#  동결된 스키마
# --------------------------------------------------------------------------- #
def test_trajectory_schema():
    assert _fields(Trajectory) == {
        "method", "pos", "vel", "dt", "wp_in", "wp_out", "done",
        "runtime", "feasible", "note", "extra",
    }


def test_trajectory_readonly_api():
    """validate.py / viz.py 가 Solution 에 기대하는 것과 같은 이름들."""
    for name in ("n", "T", "xy", "K", "team_time", "flow_time",
                 "travel_distance", "total_wait", "mean_speed_ratio",
                 "n_full_stops"):
        assert hasattr(Trajectory, name), name
    for name in ("completion_steps", "event_steps", "truncate"):
        assert callable(getattr(Trajectory, name)), name


def test_planmode_schema():
    assert _fields(PlanMode) == {"routes", "yield_rank", "cautious", "split_side"}


def test_rollout_schema():
    assert _fields(Rollout) == {
        "mode", "traj", "reached_goal", "stalled", "truncated",
        "hit_horizon", "end_state", "remaining",
    }


def test_cost_schema():
    """Planner 가 내놓는 값.  A 는 읽기만, B 는 자유롭게 재설계 가능."""
    assert _fields(Cost) == {
        "total", "feasible",
        "n_agent_collisions", "n_obstacle_collisions", "n_dep_violations",
        "n_unfinished", "stalled",
        "makespan", "flow", "wait", "dist", "clear", "turn", "dev",
        "min_agent_gap", "min_obst_gap", "label",
    }


def test_costweights_keys_match_cost_terms():
    """가중치 이름과 cost 항 이름이 어긋나면 viz 의 breakdown 이 조용히 깨진다."""
    w = _fields(CostWeights) - {"soft_margin"}
    assert w == {"make", "flow", "wait", "dist", "clear", "turn", "dev"}


def test_teamstate_schema():
    assert _fields(TeamState) == {
        "pos", "vel", "phase", "dwell_left", "hint", "t",
        "wp_in", "wp_out", "done",
    }


def test_phase_constants_come_from_the_safety_layer():
    """S2 에서 PHASE_* 를 safety/phase.py 로 내렸다.  planning 이 re-export 할 뿐
    두 벌이 되지 않았는지 확인한다."""
    from safety import phase as SP
    assert (PHASE_TO_WP, PHASE_DWELL, PHASE_TO_GOAL, PHASE_DONE) == \
        (SP.PHASE_TO_WP, SP.PHASE_DWELL, SP.PHASE_TO_GOAL, SP.PHASE_DONE)


def test_phase_constants_are_ordered():
    """remaining_estimate 가 `phase >= PHASE_DWELL` 같은 비교를 쓴다."""
    assert (PHASE_TO_WP, PHASE_DWELL, PHASE_TO_GOAL, PHASE_DONE) == (0, 1, 2, 3)


# --------------------------------------------------------------------------- #
#  경계가 실제로 동작하는가
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def wm_and_rolls():
    problem = get_scenario("crossing2")
    wm = WorldModel(problem, k_routes=2, seed=0)
    modes = wm.sample_modes(max_modes=4)
    rolls = wm.rollouts(initial_state(problem), modes, horizon=25)
    return problem, wm, modes, rolls


def test_worldmodel_public_surface(wm_and_rolls):
    """A 가 내부를 갈아엎어도 이 두 개는 남아야 한다."""
    _, wm, _, _ = wm_and_rolls
    assert callable(wm.sample_modes) and callable(wm.rollouts)


def test_canonical_mode_is_first(wm_and_rolls):
    """아무것도 못 고를 때 돌아갈 자리 — 최단 경로 + 항등 양보 순서."""
    problem, _, modes, _ = wm_and_rolls
    assert modes[0] == PlanMode(tuple([0] * problem.n),
                                tuple(range(problem.n)), False, False)


def test_rollout_shapes(wm_and_rolls):
    problem, _, _, rolls = wm_and_rolls
    for r in rolls:
        t = r.traj
        assert t.pos.ndim == 3 and t.pos.shape[1:] == (problem.n, 2)
        assert t.vel.shape == t.pos.shape
        assert t.pos.shape[0] == t.T + 1
        for ev in (t.wp_in, t.wp_out, t.done):
            assert ev.shape == (problem.n,)
            assert np.all(ev >= -1)
        assert set(r.remaining) >= {"makespan", "flow", "finish", "wp_start"}


def test_planner_selects_and_scores_every_rollout(wm_and_rolls):
    problem, wm, _, rolls = wm_and_rolls
    best, costs = Planner(problem).select(rolls, wm)
    assert len(costs) == len(rolls)
    assert 0 <= best < len(rolls)
    assert costs[best].total == min(c.total for c in costs)


def test_hard_block_is_large_but_finite():
    """모든 미래가 infeasible 해도 순위는 정의되어야 한다 — 예외를 던지면 안 된다."""
    from planning.planner import W_HARD
    assert np.isfinite(W_HARD) and W_HARD >= 1e3


def test_horizon_truncation_is_not_a_violation(wm_and_rolls):
    """horizon 이 끝나 아직 도착 못 한 agent 는 제약 위반이 아니라 미완이다."""
    problem, wm, _, rolls = wm_and_rolls
    planner = Planner(problem)
    for r in rolls:
        if r.hit_horizon:
            assert planner.evaluate(r, wm).n_unfinished == 0
            return
    pytest.skip("이 설정에서는 horizon 에 걸린 rollout 이 없었다")
