"""회귀 — 5종 시나리오 수치가 v0 기준선에서 벗어나지 않는지.

`pytest --runslow` 로만 돈다 (약 2분).  각 Phase 를 끝낼 때 main 브랜치에서 한
번씩 돌린다.  나빠졌으면 그 Phase 안에서 잡는다 — 다음으로 넘기면 두 트랙의
변경이 섞여 원인을 못 찾는다.

기준선을 의도적으로 갱신할 때:

    python scripts/run_wm_experiments.py --no-gif --no-ablation
    python tests/test_regression.py --update      # results/baseline_v0.json 갱신

갱신 커밋은 반드시 단독으로, "왜 좋아졌는지"를 메시지에 적는다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

from mahoi import validate as V
from mahoi.wm.execute import WMConfig, run_wm_planner
from mahoi.world import get_scenario

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "results", "baseline_v0.json")


def _load():
    with open(BASELINE, encoding="utf-8") as fh:
        return json.load(fh)


def _config(base) -> WMConfig:
    c = dict(base["config"])
    c.pop("dt", None)
    return WMConfig(**c)


BASE = _load()
SCENARIOS = list(BASE["scenarios"])


@pytest.mark.slow
@pytest.mark.parametrize("name", SCENARIOS)
def test_scenario_has_not_regressed(name):
    ref = BASE["scenarios"][name]["wm_planner"]
    tol = BASE["tolerance"]
    problem = get_scenario(name)

    res = run_wm_planner(problem, _config(BASE))
    rep = V.validate(problem, res.traj)

    # --- 안전은 허용 오차가 없다 ---------------------------------------- #
    assert rep.valid, f"{name}: 검증 실패\n  " + "\n  ".join(rep.errors[:6])
    assert rep.n_agent_violations == 0
    assert rep.n_obstacle_violations == 0
    assert rep.n_dep_violations == 0

    # --- 시간은 허용 오차 안에서만 나빠질 수 있다 ------------------------ #
    got, want = res.traj.team_time, ref["team_time"]
    budget = max(tol["team_time_abs_s"], want * tol["team_time_rel"])
    assert got <= want + budget, (
        f"{name}: team time 회귀 — {want:.2f}s -> {got:.2f}s "
        f"(허용 +{budget:.2f}s)")
    if got < want - budget:
        pytest.skip(f"{name}: {want:.2f}s -> {got:.2f}s 로 개선됨. "
                    f"의도한 것이라면 기준선을 갱신할 것")


@pytest.mark.slow
def test_lower_bound_is_never_beaten():
    """임계경로 하한보다 빠른 해가 나오면 하한 계산이나 시간 집계가 틀린 것이다."""
    from mahoi.coordination import critical_path_bound
    from mahoi.paths import build_prior
    for name in SCENARIOS:
        problem = get_scenario(name)
        tracks, _ = build_prior(problem)
        lb = critical_path_bound(problem, tracks)["lb_team_time"]
        res = run_wm_planner(problem, _config(BASE))
        assert res.traj.team_time >= lb - 1e-6, (
            f"{name}: {res.traj.team_time:.2f}s < 하한 {lb:.2f}s")


# --------------------------------------------------------------------------- #
def _update() -> None:
    base = _load()
    for name in base["scenarios"]:
        problem = get_scenario(name)
        res = run_wm_planner(problem, _config(base))
        rep = V.validate(problem, res.traj)
        base["scenarios"][name]["wm_planner"].update(
            team_time=round(res.traj.team_time, 2),
            flow_time=round(res.traj.flow_time, 2),
            distance=round(res.traj.travel_distance, 2),
            wait=round(res.traj.total_wait, 2),
            valid=bool(rep.valid),
            agent_clearance=round(float(rep.agent_clearance), 3),
            obstacle_clearance=round(float(rep.obstacle_clearance), 3),
            dep_violations=int(rep.n_dep_violations),
            n_switches=res.n_switches,
        )
        print(f"  {name:<10} {res.traj.team_time:6.2f}s  "
              f"{'OK' if rep.valid else 'FAIL'}")
    with open(BASELINE, "w", encoding="utf-8") as fh:
        json.dump(base, fh, indent=2, ensure_ascii=False)
    print(f"\n기준선 갱신 -> {BASELINE}")


if __name__ == "__main__":
    if "--update" in sys.argv:
        _update()
    else:
        print(__doc__)
