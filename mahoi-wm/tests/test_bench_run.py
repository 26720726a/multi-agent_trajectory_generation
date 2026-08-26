"""bench/run.py 의 계약 — 어떤 행이 남고, 죽었을 때 뭐라고 적히는가.

여기서 지키는 것은 성능이 아니라 **분모**다.  성공률은 `ok 행 / 전체 행` 이므로,
행이 조용히 사라지거나 조용히 복제되면 표는 여전히 그럴듯하게 나오고 틀렸다는
사실만 안 보인다.  가장 어려운 인스턴스가 빠질 때 성공률이 *올라가는* 것이
특히 위험하다 — 정확히 반대로 읽힌다.
"""
from __future__ import annotations

import os
import signal
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.generate import STOCHASTIC_METHODS, Instance
from bench.run import (ALL_METHODS, FIELDS, MAKESPAN_TOL, NOTE_MAX, ROS_PREFIX,
                       classify_death, methods_for_seed, run_instance_guarded,
                       strip_ros_paths, wm_config_without_seed, wm_mode_stats)

TINY = Instance(instance_seed=0, n_agents=2, dep_mode="chain", size=10.0,
                n_obstacles=(3, 5), couple_prob=1.0, couple_dist=0.55)
WM_CFG = {"horizon_s": 4.0, "replan_s": 1.0, "max_modes": 8, "keep_modes": 4,
          "k_routes": 2, "time_budget_s": 60.0}


# --------------------------------------------------------------------------- #
#  planner_seed 를 곱해도 되는 method 는 하나뿐이다
# --------------------------------------------------------------------------- #
def test_deterministic_methods_run_once_regardless_of_seed():
    """결정론적 method 를 seed 마다 돌리면 같은 행이 복제되어 분모만 부푼다."""
    assert methods_for_seed(ALL_METHODS, planner_seed=0) == ALL_METHODS
    for ps in (1, 2, 7):
        assert methods_for_seed(ALL_METHODS, planner_seed=ps) == ["wm_planner"]


def test_stochastic_set_is_not_duplicated_here():
    """어느 method 가 확률적인지는 generate.py 한 곳에만 적혀 있어야 한다."""
    assert methods_for_seed(ALL_METHODS, 1) == [m for m in ALL_METHODS
                                                if m in STOCHASTIC_METHODS]


def test_baselines_survive_a_config_without_seed_zero():
    """planner_seeds 에 0 이 없다고 baseline 이 통째로 사라지면 안 된다."""
    seeds = (5, 6, 7)
    got = [methods_for_seed(ALL_METHODS, ps, deterministic_seed=seeds[0])
           for ps in seeds]
    assert got[0] == ALL_METHODS
    assert got[1] == got[2] == ["wm_planner"]


def test_methods_not_requested_are_never_run():
    assert methods_for_seed(["lower_bound", "wm_planner"], 0) == \
        ["lower_bound", "wm_planner"]
    assert methods_for_seed(["lower_bound"], 1) == []


# --------------------------------------------------------------------------- #
#  wm_config 의 seed 는 planner_seed 를 이기지 못한다
# --------------------------------------------------------------------------- #
def test_wm_config_seed_is_stripped_and_reported():
    clean, dropped = wm_config_without_seed({"horizon_s": 4.0, "seed": 3})
    assert dropped == 3
    assert "seed" not in clean and clean["horizon_s"] == 4.0


def test_wm_config_without_seed_is_left_alone():
    clean, dropped = wm_config_without_seed({"horizon_s": 4.0})
    assert dropped is None and clean == {"horizon_s": 4.0}


def test_wm_config_is_not_mutated_in_place():
    """호출부의 dict 를 건드리면 두 번째 인스턴스가 다른 설정으로 돈다."""
    cfg = {"horizon_s": 4.0, "seed": 3}
    wm_config_without_seed(cfg)
    assert cfg["seed"] == 3


# --------------------------------------------------------------------------- #
#  죽은 이유를 확실한 척 적지 않는다
# --------------------------------------------------------------------------- #
def test_our_own_kill_is_a_timeout():
    status, note = classify_death(over_time=True, exitcode=-9, elapsed=5.0,
                                  timeout_s=5.0, peak_mb=641.0)
    assert status == "timeout"
    assert "5" in note and "641" in note


def test_sigkill_without_evidence_stays_ambiguous():
    """OOM 인지 외부 종료인지 모르면 모른다고 적는다 — oom 이라고 단정하지 않는다."""
    status, note = classify_death(over_time=False, exitcode=-int(signal.SIGKILL),
                                  elapsed=2.0, timeout_s=300.0, peak_mb=12.0)
    assert status == "killed"
    assert "구별할 수 없다" in note


def test_sigkill_with_a_huge_footprint_is_called_oom():
    huge = 10 ** 9                       # 어떤 기계에서도 전체 메모리를 넘는다
    status, note = classify_death(over_time=False, exitcode=-int(signal.SIGKILL),
                                  elapsed=2.0, timeout_s=300.0, peak_mb=huge)
    assert status == "oom"


def test_other_deaths_are_killed_with_the_reason_kept():
    status, note = classify_death(False, -int(signal.SIGSEGV), 1.0, 300.0, 5.0)
    assert status == "killed" and "SIGSEGV" in note
    status, note = classify_death(False, 1, 1.0, 300.0, 5.0)
    assert status == "killed" and "exit code 1" in note


@pytest.mark.parametrize("case", [
    (True, -9, 5.0, 5.0, 641.0),
    (False, -int(signal.SIGKILL), 2.0, 300.0, 12.0),
    (False, 1, 1.0, 300.0, 5.0),
    (False, None, 1.0, 300.0, 0.0),
])
def test_every_death_has_a_status_and_a_short_note(case):
    status, note = classify_death(*case)
    assert status in {"timeout", "oom", "killed"}
    assert note and len(note[:NOTE_MAX]) <= NOTE_MAX


# --------------------------------------------------------------------------- #
#  B2 의 증거 두 열
# --------------------------------------------------------------------------- #
class _C:
    """Cost 의 최소 대역 — 이 계산은 total/feasible/makespan 만 본다."""
    def __init__(self, total, feasible, makespan):
        self.total, self.feasible, self.makespan = total, feasible, makespan


def test_chosen_is_fastest_when_argmin_total_is_also_quickest():
    table = [_C(10.0, True, 5.0), _C(11.0, True, 6.0)]
    out = wm_mode_stats(type("R", (), {"first_table": table})())
    assert out == {"n_modes": 2, "n_feasible_modes": 2,
                   "chosen_is_fastest_feasible": True}


def test_soft_terms_buying_time_is_reported_as_false():
    """total 로는 이겼지만 makespan 은 더 느린 경우 — 이게 B2 가 재려는 것이다."""
    table = [_C(10.0, True, 6.0), _C(10.5, True, 5.0)]
    out = wm_mode_stats(type("R", (), {"first_table": table})())
    assert out["chosen_is_fastest_feasible"] is False


def test_infeasible_rollouts_do_not_set_the_target():
    """더 빠른 rollout 이 있어도 feasible 이 아니면 기준이 될 수 없다."""
    table = [_C(10.0, True, 6.0), _C(99.0, False, 1.0)]
    out = wm_mode_stats(type("R", (), {"first_table": table})())
    assert out["n_feasible_modes"] == 1
    assert out["chosen_is_fastest_feasible"] is True


def test_no_feasible_rollout_leaves_the_verdict_blank():
    """답이 없었던 것과 "빠른 걸 못 골랐다" 는 다른 이야기다."""
    table = [_C(10.0, False, 6.0)]
    out = wm_mode_stats(type("R", (), {"first_table": table})())
    assert out["n_modes"] == 1 and out["n_feasible_modes"] == 0
    assert out["chosen_is_fastest_feasible"] == ""


def test_empty_table_leaves_both_blank():
    out = wm_mode_stats(type("R", (), {"first_table": []})())
    assert out["n_feasible_modes"] == "" and out["chosen_is_fastest_feasible"] == ""


def test_tolerance_matches_check_wm():
    """check_wm.py 의 "(same)" 과 같은 문턱이어야 두 리포트가 서로를 검산한다."""
    table = [_C(10.0, True, 5.0), _C(11.0, True, 5.0 - MAKESPAN_TOL / 2)]
    out = wm_mode_stats(type("R", (), {"first_table": table})())
    assert out["chosen_is_fastest_feasible"] is True


# --------------------------------------------------------------------------- #
#  워커 경로 — 실제로 자식에서 돌려 본다
# --------------------------------------------------------------------------- #
def test_guarded_run_returns_rows_from_the_child():
    rows = run_instance_guarded(TINY, WM_CFG, ["lower_bound", "wm_planner"],
                                commit="test", planner_seed=0, timeout_s=120.0)
    assert [r["method"] for r in rows] == ["lower_bound", "wm_planner"]
    assert all(r["status"] == "ok" for r in rows)
    wm = rows[-1]
    assert wm["planner_seed"] == 0 and wm["timeout_s"] == 120.0
    assert wm["n_modes"] and wm["n_feasible_modes"] != ""


def test_every_row_fits_the_declared_schema():
    """FIELDS 에 없는 키가 섞이면 DictWriter 가 배치 중간에 죽는다."""
    rows = run_instance_guarded(TINY, WM_CFG, ["lower_bound", "wm_planner"],
                                commit="test", planner_seed=0, timeout_s=120.0)
    for r in rows:
        assert set(r) <= set(FIELDS), sorted(set(r) - set(FIELDS))


def test_a_timeout_still_leaves_one_row_per_method():
    """행이 사라지면 그 method 의 분모가 줄어 실패가 성공처럼 보인다."""
    methods = ["lower_bound", "sequential", "wm_planner"]
    rows = run_instance_guarded(TINY, WM_CFG, methods, commit="test",
                                planner_seed=0, timeout_s=0.05)
    assert [r["method"] for r in rows] == methods
    assert all(r["status"] in {"timeout", "killed", "oom"} for r in rows)
    assert all(r["note"] for r in rows)


def test_worker_strips_ros_from_the_path():
    """자식이 /opt/ros 를 물려받으면 import 단계에서 죽고, 그 죽음은 인스턴스의
    난이도와 구별되지 않는다."""
    marker = f"{ROS_PREFIX}/jazzy/lib/python3.12/site-packages"
    sys.path.insert(0, marker)
    old_env = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join([marker, "/keep/me"])
    try:
        removed = strip_ros_paths()
        assert marker in removed
        assert not [p for p in sys.path if p.startswith(ROS_PREFIX)]
        assert os.environ["PYTHONPATH"] == "/keep/me"
    finally:
        if marker in sys.path:
            sys.path.remove(marker)
        if old_env is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_env
