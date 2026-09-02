"""config 검증 규칙이 실제로 동작하는지.

규칙 (a) 는 **현행 값이 이미 위반**이라 통과 여부가 아니라 "경고가 나오는가"를
확인한다.  (b)(c) 는 현행 값이 통과하므로 일부러 깨뜨려 본다.
"""
from __future__ import annotations

import math
import warnings

import pytest

from config import VIOLATIONS, validate
from config.controller import CONTROLLER
from config.physics import Physics
from config.planner import Planner

FINITE_A_MAX = pytest.mark.skipif(
    not math.isfinite(Physics().a_max),
    reason="requires shipped a_max=3.0; re-enable the S3 finite-braking configuration")


def test_current_config_has_no_validation_warnings():
    """S5-1의 h=10.0에서는 stall 규칙까지 해소돼야 한다."""
    names = {n for n, _ in VIOLATIONS}
    assert names == set(), names


def test_stall_rule_message_names_the_numbers():
    """h=4.0으로 낮췄을 때 경고가 원인을 숫자로 말해야 한다."""
    v = validate(Physics(), Planner(horizon_s=4.0), CONTROLLER)
    msg = dict(v)["stall_window >= horizon_steps"]
    assert "45" in msg and "40" in msg


def test_stall_rule_clears_when_horizon_grows():
    """S3 에서 horizon 을 10.0 으로 올리면 이 위반이 사라진다."""
    v = validate(Physics(), Planner(horizon_s=10.0), CONTROLLER)
    assert not any(n == "stall_window >= horizon_steps" for n, _ in v)


def test_min_sep_rule_fires_when_margin_is_zero():
    v = validate(Physics(safety_margin=0.0), Planner(), CONTROLLER)
    assert any(n == "min_sep <= 2*robot_radius" for n, _ in v)


@FINITE_A_MAX
def test_braking_rule_is_now_actually_checked():
    """S3-A 에서 a_max 가 유한해졌으므로 (c) 가 실제로 돈다.

    S2 까지는 `a_max=inf` 라 이 규칙이 건너뛰어졌다.  지금은 검사되고,
    현행 조합(v_max=1.20, a_max=3.0)의 정지거리는 0.24 m 로 min_sep 0.70 m
    안에 넉넉히 들어온다.
    """
    p = Physics()
    assert math.isfinite(p.a_max)
    assert p.braking_distance == pytest.approx(p.v_max ** 2 / (2 * p.a_max))
    assert p.braking_distance < p.min_sep
    v = validate(p, Planner(), CONTROLLER)
    assert not any(n == "braking_distance > min_sep" for n, _ in v)


def test_an_infinite_a_max_still_skips_the_braking_rule():
    """되돌릴 수 있는 상태로 남겨 둔다 — S2 재현이 이 경로를 쓴다."""
    v = validate(Physics(a_max=math.inf), Planner(), CONTROLLER)
    assert not any(n == "braking_distance > min_sep" for n, _ in v)


def test_braking_rule_fires_on_a_weak_brake():
    """S3 에서 a_max 를 넣을 때 이 규칙이 실제로 걸린다.

    v_max=1.20, a_max=0.5 -> 정지거리 1.44 m > min_sep 0.70 m.
    """
    v = validate(Physics(a_max=0.5), Planner(), CONTROLLER)
    assert any(n == "braking_distance > min_sep" for n, _ in v)


def test_plan_target_a_max_passes():
    """계획서 §2 목표 조합(v_max=2.0, a_max=3.0)이 (c) 를 통과하는지 미리 확인.

    정지거리 = 2.0^2 / (2*3.0) = 0.667 m <= min_sep 0.70 m — 아슬아슬하게 통과한다.
    """
    p = Physics(v_max=2.0, v_nominal=2.0, a_max=3.0)
    assert p.braking_distance <= p.min_sep
    v = validate(p, Planner(), CONTROLLER)
    assert not any(n == "braking_distance > min_sep" for n, _ in v)


def test_import_has_no_config_warning_at_the_shipped_horizon():
    import importlib
    import config
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.reload(config)
    assert not any(issubclass(x.category, config.ConfigWarning) for x in w)
