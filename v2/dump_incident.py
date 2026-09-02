#!/usr/bin/env python3
"""사고 한 건을 스텝 단위로 뜯어본다 (S3 §B-2).

    python scripts/dump_incident.py --uid n3_chain_1a2b3c4d --csv results/s3/B_vmax2.csv
    python scripts/dump_incident.py --scenario crossing2 --planner-seed 0

무엇을 보이는가 — 사고 직전 `--window` 스텝에서

    쌍마다   중심거리, 하한(min_sep)까지의 여유, **접근 속도**(음수면 멀어짐)
    에이전트마다  속력, |dv|/dt, **안전 투영이 개입했는지**와 그 크기
    이웃마다  VO 가 실제로 발동했는지 (t_c < TAU 인 이웃이 있었는지)

`_project_safe` 개입 여부는 추론이 아니다.  `control.step(trace=...)` 이
투영 **전**과 **후**의 명령을 그대로 남기고, 여기서 그 차이를 읽는다.

사고의 정의 (우선순위 순)
  1. 충돌     — 어느 쌍이든 중심거리가 min_sep 밑으로 내려간 첫 스텝
  2. 최근접   — 충돌이 없으면 가장 가까웠던 스텝
  3. 정지     — 팀 전체가 멈춘 첫 스텝 (deadlock 진단용, `--stall`)
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONTROLLER, PHYSICS                       # noqa: E402
from planning.execute import WMConfig, run_wm_planner        # noqa: E402
from planning.world import SCENARIOS, get_scenario, random_problem  # noqa: E402
from safety.phase import PHASE_DONE, PHASE_DWELL             # noqa: E402
from safety.vo import _ttc                                   # noqa: E402
import safety.validate as V                                  # noqa: E402


# --------------------------------------------------------------------------- #
def problem_from_csv(path: str, uid: str, planner_seed: int = 0):
    """CSV 행의 축 값으로 인스턴스를 되살린다.  `uid` 가 같으면 같은 문제다.

    `planner_seed` 는 문제를 바꾸지 않지만 (bench/generate.py), **어느 행의
    결과를 재현하려는 것인지**는 가른다.  메타데이터를 엉뚱한 시드의 행에서
    읽어 오면 "CSV 는 성공인데 재현은 실패" 처럼 보인다.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["uid"] != uid or r.get("method") != "wm_planner" \
                    or int(r.get("planner_seed", 0) or 0) != planner_seed:
                continue
            lo, hi = (r["n_obstacles"].split("-") + [r["n_obstacles"]])[:2]
            return random_problem(
                seed=int(r["instance_seed"]), n_agents=int(r["n_agents"]),
                size=float(r["size"]), n_obstacles=(int(lo), int(hi)),
                dep_mode=r["dep_mode"], coupled_waypoints=float(r["couple_prob"]),
                couple_dist=float(r["couple_dist"])), r
    raise SystemExit(f"{path} 에 uid={uid}, planner_seed={planner_seed} 인 "
                     f"wm_planner 행이 없다")


def find_incident(problem, traj, stall: bool):
    """(스텝, 사유).  없으면 최근접 스텝을 돌려준다."""
    xy = traj.pos
    n, T = problem.n, traj.T
    worst_t, worst_d = 0, math.inf
    for t in range(T + 1):
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(xy[t, i] - xy[t, j]))
                if d < problem.min_sep(i, j) - 1e-3:
                    return t, f"충돌 (쌍 {i}-{j}, 중심거리 {d:.4f} < 하한 " \
                              f"{problem.min_sep(i, j):.4f})"
                if d < worst_d:
                    worst_t, worst_d = t, d
    if stall:
        sp = np.linalg.norm(traj.vel, axis=2).max(axis=1)      # (T+1,)
        moving = sp > 1e-6
        for t in range(T):
            if not moving[t:].any():
                return t, "정지 (이후 아무도 움직이지 않는다)"
    return worst_t, f"충돌 없음 — 최근접 스텝 (중심거리 {worst_d:.4f})"


def dump(problem, traj, trace, t0: int, window: int, reason: str) -> None:
    n = problem.n
    dt = problem.dt
    lo = max(0, t0 - window)
    names = [a.name for a in problem.agents]

    print(f"\n사고 지점: t = {t0 * dt:.1f}s (스텝 {t0}) — {reason}")
    print(f"창: 스텝 {lo}..{t0}   dt={dt}  a_max={PHYSICS.a_max}  "
          f"v_max={PHYSICS.v_max}  min_sep={PHYSICS.min_sep}  "
          f"TAU={CONTROLLER.tau}")

    by_t = {r["t"]: r for r in trace}
    for t in range(lo, t0 + 1):
        print(f"\n  --- t={t * dt:5.2f}s (스텝 {t}) " + "-" * 46)
        # trace 는 스텝 **전** 상태의 t 를 적고, 그 스텝의 명령이 traj.vel[t] 다.
        rec = by_t.get(t)
        # -- 쌍 --
        for i in range(n):
            for j in range(i + 1, n):
                p_i, p_j = traj.pos[t, i], traj.pos[t, j]
                v_i, v_j = traj.vel[t, i], traj.vel[t, j]
                w = p_j - p_i
                d = float(np.linalg.norm(w))
                need = problem.min_sep(i, j)
                closing = float(-(v_j - v_i) @ w / max(d, 1e-12))
                # VO 가 이 쌍에서 발동했는가 — 실제 명령 속도로 t_c 를 잰다
                R = need + CONTROLLER.vo_soft_margin
                tc = float(_ttc(w, (v_i - v_j)[None, :], R)[0])
                fired = tc < CONTROLLER.tau
                print(f"    쌍 {names[i]}-{names[j]}: 거리 {d:6.3f} "
                      f"(여유 {d - need:+6.3f})  접근속도 {closing:+6.3f} m/s  "
                      f"TTC {'%6.2f' % tc if math.isfinite(tc) else '   inf'}s  "
                      f"VO {'발동' if fired else ' -- '}")
        # -- 에이전트 --
        for i in range(n):
            v_now = traj.vel[t, i]
            v_bef = traj.vel[t - 1, i] if t > 0 else np.zeros(2)
            acc = float(np.linalg.norm(v_now - v_bef)) / dt
            flag = "!! a_max 초과" if acc > PHYSICS.a_max * (1 + 1e-9) else ""
            line = (f"    {names[i]}: 속력 {np.linalg.norm(v_now):5.3f}  "
                    f"|dv|/dt {acc:6.2f} {flag:<13}")
            if rec is not None:
                dproj = float(np.linalg.norm(rec["v_cmd"][i] - rec["v_pre"][i]))
                ph = int(rec["phase"][i])
                blocked = bool(np.linalg.norm(rec["v_pre"][i]) < 1e-15
                               and np.linalg.norm(rec["vel_prev"][i]) > 1e-9
                               and ph not in (PHASE_DWELL, PHASE_DONE))
                line += (f" 투영 {'개입 %.3f' % dproj if dproj > 1e-12 else '  --   '}"
                         f"  phase={ph}{'  후보전멸?' if blocked else ''}")
            print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--uid", help="CSV 에서 이 uid 의 인스턴스를 되살린다")
    g.add_argument("--scenario", choices=list(SCENARIOS))
    ap.add_argument("--csv", help="--uid 와 함께: 축 값을 읽을 CSV")
    ap.add_argument("--planner-seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--stall", action="store_true",
                    help="충돌이 없으면 최근접 대신 '정지' 지점을 찾는다")
    ap.add_argument("--horizon-s", type=float, default=None)
    args = ap.parse_args()

    if args.uid:
        if not args.csv:
            ap.error("--uid 에는 --csv 가 필요하다")
        problem, row = problem_from_csv(args.csv, args.uid, args.planner_seed)
        print(f"uid {args.uid}  (CSV 의 status={row['status']!r}, "
              f"team_time={row['team_time']!r}, phys_fp={row.get('phys_fp')})")
    else:
        problem = get_scenario(args.scenario)
        print(f"시나리오 {args.scenario}")

    kw = {"seed": args.planner_seed}
    if args.horizon_s is not None:
        kw["horizon_s"] = args.horizon_s
    trace: list = []
    res = run_wm_planner(problem, WMConfig(**kw), trace=trace)
    rep = V.validate(problem, res.traj)
    acc = res.traj.extra["accel"]
    print(f"재현: team_time {res.traj.team_time:.2f}s  feasible {res.traj.feasible}  "
          f"note {res.traj.note!r}")
    print(f"  검증: valid={rep.valid}  agent_viol={rep.n_agent_violations}  "
          f"obst_viol={rep.n_obstacle_violations}  dep_viol={rep.n_dep_violations}")
    print(f"  가속도: 위반 {int(acc['n_amax_violations'])} "
          f"(정지 {int(acc['n_amax_viol_stop'])} / 착지 {int(acc['n_amax_viol_snap'])} "
          f"/ 투영 {int(acc['n_amax_viol_project'])}),  "
          f"투영 개입 {int(acc['n_project_active'])}/{int(acc['steps'])},  "
          f"max_accel {acc['max_accel']:.2f}")
    if rep.errors:
        print("  오류:")
        for e in rep.errors[:6]:
            print(f"    {e}")

    t0, reason = find_incident(problem, res.traj, args.stall)
    dump(problem, res.traj, trace, t0, args.window, reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
