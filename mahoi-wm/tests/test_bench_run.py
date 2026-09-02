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


# =========================================================================== #
#  3단계 — 재개, 병렬, 검산
# =========================================================================== #
import csv                                                          # noqa: E402
import multiprocessing as mp                                        # noqa: E402
import multiprocessing.pool                                         # noqa: E402

from bench.run import (INNER_WORKERS_VAR, KEY, NO_METHOD, THREAD_VARS,
                       SchemaMismatch, _NoDaemonContext, _pool_init,
                       audit_row_count, clean_note, describe_schema_mismatch,
                       pending_methods, rows_per_instance, scan_existing)


def _row(uid: str, method: str, ps: int, **kw) -> dict:
    r = {k: "" for k in FIELDS}
    r.update(uid=uid, method=method, planner_seed=ps, status="ok")
    r.update(kw)
    return r


def _write_csv(path, rows, fields=FIELDS) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------- #
#  note 는 한 줄이어야 재개가 행을 셀 수 있다
# --------------------------------------------------------------------------- #
def test_note_is_folded_to_one_line():
    """줄바꿈이 남으면 한 행이 여러 줄이 되고, 재개가 잘린 행을 알아볼 수 없다."""
    assert "\n" not in clean_note("Traceback\n  line 1\n  line 2")
    assert clean_note("a\r\nb") == "a b"


def test_note_is_truncated_and_never_none():
    assert len(clean_note("x" * 500)) == NOTE_MAX
    assert clean_note(None) == "" and clean_note("") == ""


# --------------------------------------------------------------------------- #
#  기존 CSV 읽기
# --------------------------------------------------------------------------- #
def test_scan_reads_back_every_key(tmp_path):
    p = tmp_path / "a.csv"
    _write_csv(p, [_row("u1", "lower_bound", 0), _row("u1", "wm_planner", 1)])
    done, n_good, n_dropped, end = scan_existing(str(p))
    assert done == {("u1", "lower_bound", "0"), ("u1", "wm_planner", "1")}
    assert (n_good, n_dropped) == (2, 0)
    assert end == p.stat().st_size          # 온전한 파일이면 끝까지가 유효하다


def test_scan_of_a_header_only_file_is_empty(tmp_path):
    p = tmp_path / "a.csv"
    _write_csv(p, [])
    assert scan_existing(str(p)) == (set(), 0, 0, p.stat().st_size)


def test_scan_of_an_empty_file_is_empty(tmp_path):
    p = tmp_path / "a.csv"
    p.write_bytes(b"")
    assert scan_existing(str(p)) == (set(), 0, 0, 0)


def test_a_truncated_last_row_is_dropped_and_the_offset_points_before_it(tmp_path):
    """쓰는 도중에 죽으면 마지막 줄이 잘린다.  거기 이어 쓰면 두 행이 붙는다."""
    p = tmp_path / "a.csv"
    _write_csv(p, [_row("u1", "lower_bound", 0), _row("u2", "wm_planner", 0)])
    whole = p.read_bytes()
    p.write_bytes(whole[:-12])              # 마지막 행을 자른다
    done, n_good, n_dropped, end = scan_existing(str(p))
    assert done == {("u1", "lower_bound", "0")}
    assert (n_good, n_dropped) == (1, 1)
    # 잘라낸 지점까지가 정확히 온전한 행들이다
    assert p.read_bytes()[:end] == whole[:whole.index(b"u2")]


def test_everything_after_a_broken_row_is_distrusted(tmp_path):
    """깨진 지점 뒤가 멀쩡해 보여도, 왜 깨졌는지 모르는 채로 믿을 수 없다."""
    p = tmp_path / "a.csv"
    _write_csv(p, [_row(f"u{i}", "wm_planner", 0) for i in range(4)])
    lines = p.read_bytes().split(b"\r\n")
    lines[2] = b"garbage,row"               # 두 번째 데이터 행을 부순다
    p.write_bytes(b"\r\n".join(lines))
    done, n_good, n_dropped, _ = scan_existing(str(p))
    assert done == {("u0", "wm_planner", "0")}
    assert n_good == 1 and n_dropped >= 1


def test_a_different_header_is_refused(tmp_path):
    p = tmp_path / "a.csv"
    _write_csv(p, [], fields=[f for f in FIELDS if f != "planner_seed"])
    with pytest.raises(SchemaMismatch) as exc:
        scan_existing(str(p))
    assert "planner_seed" in str(exc.value)


def test_the_mismatch_message_names_the_columns():
    msg = describe_schema_mismatch(["a", "b", "seed"], ["a", "b", "planner_seed"])
    assert "planner_seed" in msg and "seed" in msg


def test_a_reordered_header_is_reported_as_order_not_as_missing():
    """열 이름이 같고 순서만 다른 경우를 "빠졌다" 고 하면 엉뚱한 곳을 고치게 된다."""
    msg = describe_schema_mismatch(["b", "a"], ["a", "b"])
    assert "순서" in msg and "0번째" in msg


# --------------------------------------------------------------------------- #
#  무엇을 다시 돌릴 것인가
# --------------------------------------------------------------------------- #
def test_nothing_pending_when_everything_is_done():
    done = {(TINY.uid, m, "0") for m in ALL_METHODS}
    assert pending_methods(TINY, 0, ALL_METHODS, done) == []


def test_only_the_missing_methods_come_back():
    """인스턴스가 행을 쓰는 도중에 죽으면 앞쪽 method 는 이미 파일에 있다."""
    already = ("lower_bound", "sequential")
    done = {(TINY.uid, m, "0") for m in already}
    # ALL_METHODS 에서 파생시킨다 — method 를 추가할 때마다 이 테스트가
    # 깨지는 것은 계약 위반이 아니라 목록이 늘어난 것뿐이다.
    assert pending_methods(TINY, 0, ALL_METHODS, done) == \
        [m for m in ALL_METHODS if m not in already]


def test_pending_respects_the_deterministic_seed_rule():
    assert pending_methods(TINY, 1, ALL_METHODS, set()) == ["wm_planner"]


def test_a_done_key_from_another_seed_does_not_count():
    done = {(TINY.uid, "wm_planner", "0")}
    assert pending_methods(TINY, 1, ALL_METHODS, done) == ["wm_planner"]


# --------------------------------------------------------------------------- #
#  검산
# --------------------------------------------------------------------------- #
def test_rows_per_instance_multiplies_only_the_stochastic_method():
    det = sum(1 for m in ALL_METHODS if m not in STOCHASTIC_METHODS)
    sto = sum(1 for m in ALL_METHODS if m in STOCHASTIC_METHODS)
    assert rows_per_instance(ALL_METHODS, [0]) == det + sto
    assert rows_per_instance(ALL_METHODS, [0, 1, 2]) == det + sto * 3
    assert rows_per_instance(["lower_bound"], [0, 1, 2]) == 1


def test_audit_passes_on_a_complete_file(tmp_path, capsys):
    p = tmp_path / "a.csv"
    rows = [_row(f"u{i}", m, ps)
            for i in range(2)
            for m in ALL_METHODS for ps in ([0, 1] if m == "wm_planner" else [0])]
    _write_csv(p, rows)
    assert audit_row_count(str(p), ALL_METHODS, [0, 1], n_instances=2) is True


def test_audit_notices_a_missing_row(tmp_path, capsys):
    p = tmp_path / "a.csv"
    _write_csv(p, [_row("u0", "lower_bound", 0)])
    assert audit_row_count(str(p), ALL_METHODS, [0, 1], n_instances=2) is False
    assert "기대" in capsys.readouterr().err


def test_audit_notices_a_duplicated_row(tmp_path, capsys):
    """재개가 이미 끝난 것을 또 돌리면 분자와 분모가 같이 부푼다."""
    p = tmp_path / "a.csv"
    _write_csv(p, [_row("u0", "lower_bound", 0)] * 2)
    assert audit_row_count(str(p), ALL_METHODS, [0], n_instances=1) is False
    assert "겹치는 행" in capsys.readouterr().err


def test_ungeneratable_instances_are_excluded_from_the_expectation(tmp_path):
    """생성 실패는 method 와 무관하게 한 줄만 남긴다 — 기대치에서 덜어내야 한다."""
    p = tmp_path / "a.csv"
    rows = [_row("good", m, ps)
            for m in ALL_METHODS for ps in ([0, 1] if m == "wm_planner" else [0])]
    rows.append(_row("bad", NO_METHOD, 0, status="ungeneratable"))
    _write_csv(p, rows)
    assert audit_row_count(str(p), ALL_METHODS, [0, 1], n_instances=2) is True


# --------------------------------------------------------------------------- #
#  워커가 실제로 어떤 환경에서 도는가
# --------------------------------------------------------------------------- #
def _probe(_):
    return {"threads": {v: os.environ.get(v) for v in THREAD_VARS},
            "inner": os.environ.get(INNER_WORKERS_VAR),
            "ros": [p for p in sys.path if p.startswith(ROS_PREFIX)],
            "daemon": mp.current_process().daemon}


def test_pool_workers_are_pinned_and_ros_free():
    """조건 6 — 워커 안에서 스레드가 묶여 있고 /opt/ros 가 없다."""
    pool = mp.pool.Pool(2, initializer=_pool_init, initargs=(3,),
                        context=_NoDaemonContext())
    try:
        got = pool.map(_probe, range(2))
    finally:
        pool.close()
        pool.join()
    for g in got:
        assert all(g["threads"][v] == "1" for v in THREAD_VARS), g["threads"]
        assert g["inner"] == "3"
        assert g["ros"] == []


def test_pool_workers_may_have_children():
    """daemon 워커는 자식을 못 만든다 — 그러면 2단계의 하드 타임아웃이 죽는다."""
    pool = mp.pool.Pool(1, initializer=_pool_init, initargs=(1,),
                        context=_NoDaemonContext())
    try:
        assert pool.map(_probe, [0])[0]["daemon"] is False
    finally:
        pool.close()
        pool.join()


def test_key_columns_exist_in_the_schema():
    assert set(KEY) <= set(FIELDS)
