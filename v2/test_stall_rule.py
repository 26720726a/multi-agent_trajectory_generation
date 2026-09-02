"""정체 판정이 언제 살아나는가 — S3-C 스윕이 볼 것을 미리 못 박는다.

`config/planner.py` 의 경고가 말하는 대로, `stall_window(45)` 가
`horizon_steps` 이상이면 rollout 은 정체 판정에 도달하기 전에 지평선이 먼저
끝난다.  그러면 `Cost.stalled` 는 **한 번도** True 가 되지 않고, `feasible`
에서 그 항이 통째로 빠진다 (P0-1 §G, P1b 8,100행에서 확인된 상태).

여기서 확인하는 것은 그 관계가 **말이 아니라 실측으로** 성립하는가다.
S3-C 는 h ∈ {4,6,8,10,12,15} 를 도는데, h=4 만 위반이고 나머지는 아니다 —
성능 변화가 그 경계와 겹치는지를 판단하려면 이 숫자가 먼저 믿을 수 있어야 한다.
"""
from __future__ import annotations

import pytest

from config import PLANNER, PHYSICS
from planning.execute import WMConfig, run_wm_planner
from planning.world import get_scenario


@pytest.fixture(scope="module")
def problem():
    return get_scenario("crossing2")


def test_the_shipped_horizon_enables_the_stall_rule(problem):
    """현행 h=10.0 (=100 스텝) > stall_window=45 — 판정이 실제로 돈다."""
    steps = PLANNER.horizon_steps(PHYSICS.dt)
    assert steps is not None and PLANNER.stall_window < steps
    res = run_wm_planner(problem, WMConfig(seed=0))
    assert res.n_evals > 0
    assert res.n_stalled_evals > 0, "발동해야 하는데 한 번도 없었다"


def test_the_stall_rule_comes_alive_once_the_horizon_passes_the_window(problem):
    """h=10.0 (=100 스텝) > 45 — 이제 판정이 실제로 돈다."""
    res = run_wm_planner(problem, WMConfig(seed=0, horizon_s=10.0))
    assert res.n_stalled_evals > 0
    assert res.n_stalled_evals <= res.n_evals


@pytest.mark.parametrize("h,alive", [(4.0, False), (6.0, True), (8.0, True),
                                     (10.0, True), (12.0, True), (15.0, True)])
def test_every_sweep_point_matches_its_predicted_stall_status(problem, h, alive):
    """S3-C 의 6개 값 각각에서 예측과 실측이 맞는지.

    예측은 `horizon_s/dt > stall_window` 하나뿐이다.  실측이 어긋나면 스윕
    결과의 해석이 통째로 틀어진다.
    """
    steps = int(round(h / PHYSICS.dt))
    assert (steps > PLANNER.stall_window) is alive, "예측 자체가 틀렸다"
    res = run_wm_planner(problem, WMConfig(seed=0, horizon_s=h))
    assert (res.n_stalled_evals > 0) is alive, \
        f"h={h}: 예측 {alive}, 실측 {res.n_stalled_evals}/{res.n_evals}"


def test_counting_covers_every_replan_not_just_t0(problem):
    """`first_table` 은 t=0 한 번뿐이라 이 질문에 답할 수 없다.

    재계획마다 후보 전부를 세므로 `n_evals` 는 t=0 의 후보 수보다 크다.
    """
    res = run_wm_planner(problem, WMConfig(seed=0))
    assert res.n_evals > len(res.first_table)
    assert len(res.replan_ms) == len(res.switches) > 0
    assert all(t >= 0.0 for t in res.replan_ms)
