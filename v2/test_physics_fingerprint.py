"""물리 지문 — 서로 다른 물리로 돈 CSV 가 섞이지 않게 하는 장치.

S0 §G-3 위험 3.  인스턴스 `uid` 는 **문제 정의만** 해싱한다.  `v_max` 를
1.20 에서 2.0 으로 올려도 uid 는 그대로다.  짝지은 비교에는 그 성질이 필요하지만
(같은 인스턴스끼리 비교해야 한다), 그 때문에 **두 물리의 행이 한 CSV 에 섞여도
조인이 성립해 버린다** — 경고 하나 없이 틀린 답이 나온다.  그 구멍을 막는다.
"""
from __future__ import annotations

import csv
import math
import os

import pytest

from config import FINGERPRINT_FIELDS, PHYS_FP, physics_fingerprint
from config.controller import CONTROLLER, Controller
from config.physics import PHYSICS, Physics
from config.planner import PLANNER, Planner


# --------------------------------------------------------------------------- #
#  지문 자체
# --------------------------------------------------------------------------- #
def test_fingerprint_is_stable_and_short():
    assert physics_fingerprint() == physics_fingerprint() == PHYS_FP
    assert len(PHYS_FP) == 12
    assert all(c in "0123456789abcdef" for c in PHYS_FP)


@pytest.mark.parametrize("kw", [
    # 현재 a_max=inf이면 같은 값을 다시 넣는 행은 지문을 바꾸지 않는다.
    {"v_max": 2.0}, {"v_nominal": 1.0}, {"a_max": 5.0},
    pytest.param({"a_max": math.inf}, marks=pytest.mark.skipif(
        math.isinf(PHYSICS.a_max),
        reason="requires shipped a_max=3.0; re-enable the S3 finite-braking configuration")),
    {"dt": 0.05},
    {"robot_radius": 0.35}, {"safety_margin": 0.15}, {"map_size": 12.0},
])
def test_changing_a_physics_value_changes_the_fingerprint(kw):
    assert physics_fingerprint(Physics(**kw), PLANNER, CONTROLLER) != PHYS_FP


@pytest.mark.parametrize("kw", [
    {"horizon_s": 4.0}, {"horizon_s": None}, {"replan_s": 0.5},
    {"stall_window": 30},
])
def test_changing_a_planner_value_changes_the_fingerprint(kw):
    assert physics_fingerprint(PHYSICS, Planner(**kw), CONTROLLER) != PHYS_FP


@pytest.mark.parametrize("kw", [
    {"tau": 3.0}, {"vo_soft_margin": 0.25}, {"w_turn": 0.3},
    {"speed_levels": (0.0, 0.5, 1.0)}, {"lookahead": 0.9},
])
def test_changing_a_controller_value_changes_the_fingerprint(kw):
    assert physics_fingerprint(PHYSICS, PLANNER, Controller(**kw)) != PHYS_FP


def test_reordering_the_field_list_does_not_change_the_fingerprint():
    """나열 순서는 지문의 일부가 아니다.

    필드를 재정렬하는 리팩터링이 **이미 돌려 둔 배치를 무효로 만들면** 안 된다.
    지문이 답해야 하는 질문은 "값이 같은가" 하나뿐이다.
    """
    import config
    original = config.FINGERPRINT_FIELDS
    shuffled = tuple(
        (group, tuple(reversed(names))) for group, names in reversed(original))
    assert shuffled != original
    try:
        config.FINGERPRINT_FIELDS = shuffled
        assert physics_fingerprint() == PHYS_FP
    finally:
        config.FINGERPRINT_FIELDS = original


def test_fingerprint_covers_every_controller_field():
    """컨트롤러 튜너블은 전부 궤적을 바꾼다 — 하나라도 빠지면 구멍이다."""
    import dataclasses
    covered = dict(FINGERPRINT_FIELDS)["controller"]
    assert set(covered) == {f.name for f in dataclasses.fields(Controller)}


def test_run_configuration_is_deliberately_outside_the_fingerprint():
    """`max_modes` / `time_budget_s` 같은 것은 지문에 넣지 않는다.

    bench config 가 인스턴스마다 덮어쓰는 값이라, 넣으면 지문이 실행 설정을
    뒤따라가 "물리가 같은가" 라는 질문에 답하지 못하게 된다.
    """
    covered = dict(FINGERPRINT_FIELDS)["planner"]
    assert set(covered) == {"horizon_s", "replan_s", "stall_window"}
    for name in ("max_modes", "keep_modes", "k_routes", "time_budget_s",
                 "max_sim_s", "stall_patience"):
        assert name not in covered


# --------------------------------------------------------------------------- #
#  bench/run.py 와의 연결
# --------------------------------------------------------------------------- #
def test_fields_has_phys_fp_at_the_end():
    """새 열은 뒤에 붙인다 — 앞쪽이 밀리면 옛 CSV 와 나란히 볼 수 없다."""
    from bench.run import FIELDS
    assert FIELDS[-1] == "phys_fp"


def test_blank_row_carries_the_current_fingerprint():
    from bench.generate import Instance
    from bench.run import _blank
    inst = Instance(instance_seed=0, n_agents=2, dep_mode="chain", size=10.0,
                    n_obstacles=(3, 5), couple_prob=1.0, couple_dist=0.55)
    row = _blank(inst, "wm_planner", 1.0, "abc1234", 0, 300.0)
    assert row["phys_fp"] == PHYS_FP


def _write(path, rows):
    from bench.run import FIELDS
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def _row(uid="u1", method="wm_planner", ps=0, fp=None):
    return {"uid": uid, "method": method, "planner_seed": ps,
            "phys_fp": PHYS_FP if fp is None else fp}


def test_scan_existing_accepts_matching_fingerprints(tmp_path):
    from bench.run import scan_existing
    p = str(tmp_path / "ok.csv")
    _write(p, [_row("u1"), _row("u2")])
    done, n_good, n_dropped, _ = scan_existing(p)
    assert n_good == 2 and n_dropped == 0
    assert ("u1", "wm_planner", "0") in done


def test_scan_existing_refuses_a_foreign_fingerprint(tmp_path):
    """v_max 만 바뀐 행이 섞여도 멈춰야 한다 — uid 는 같으므로 이것 말고는 단서가 없다."""
    from bench.run import PhysicsMismatch, scan_existing
    other = physics_fingerprint(Physics(v_max=2.0), PLANNER, CONTROLLER)
    assert other != PHYS_FP
    p = str(tmp_path / "mixed.csv")
    _write(p, [_row("u1"), _row("u2", fp=other)])
    with pytest.raises(PhysicsMismatch) as exc:
        scan_existing(p)
    msg = str(exc.value)
    assert other in msg and PHYS_FP in msg and "u2" in msg


def test_scan_existing_refuses_an_empty_fingerprint(tmp_path):
    """S2 이전의 CSV 에는 이 열이 빈칸이다.  그것도 "다른 물리" 로 다룬다."""
    from bench.run import PhysicsMismatch, scan_existing
    p = str(tmp_path / "old.csv")
    _write(p, [_row("u1", fp="")])
    with pytest.raises(PhysicsMismatch):
        scan_existing(p)


def test_scan_existing_can_be_asked_about_another_fingerprint(tmp_path):
    """도구가 "이 CSV 는 어느 물리인가" 를 물을 수 있어야 한다."""
    from bench.run import scan_existing
    other = physics_fingerprint(Physics(a_max=3.0), PLANNER, CONTROLLER)
    p = str(tmp_path / "a3.csv")
    _write(p, [_row("u1", fp=other)])
    done, n_good, _, _ = scan_existing(p, phys_fp=other)
    assert n_good == 1 and done
