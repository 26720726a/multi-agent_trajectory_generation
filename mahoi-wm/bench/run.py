#!/usr/bin/env python3
"""대량 실행 — 난이도 격자 × 방법 을 돌려 한 줄씩 CSV 로 쌓는다.

    python -m bench.run --config bench/configs/smoke.json
    python -m bench.run --config bench/configs/scale.json --out bench/runs/scale.csv

설계 원칙 두 가지.

* **한 인스턴스의 실패가 전체를 죽이지 않는다.**  예외는 잡아서 `status` 열에
  기록한다.  대량 실행에서 중간에 죽으면 몇 시간을 잃는다.
* **모든 행이 스스로를 재현한다.**  seed·축 값·git commit 이 행마다 들어 있어,
  나중에 이상한 점을 발견하면 그 한 줄만 보고 되살릴 수 있다.

  TODO(B1) 지금 부족한 것
    - 실패 원인 분류가 status 4종뿐이다.  deadlock / livelock / mode 집합에 답이
      없음 / cost 오판 을 구분해야 A 에게 넘길 작업 목록이 된다 (B6).
    - 방법별 병렬 실행.  지금은 순차라 1,000 인스턴스가 하룻밤을 넘긴다.
    - coordination A* 는 n>=4 에서 격자가 폭발한다.  확장 가능한 상한 baseline
      으로 교체해야 한다 (B5).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.generate import Instance, axis_from_config, buildable, grid
from mahoi import validate as V
from mahoi.coordination import (critical_path_bound, plan_coordination,
                                plan_sequential)
from mahoi.paths import build_prior
from mahoi.wm.execute import WMConfig, run_wm_planner

FIELDS = [
    "uid", "seed", "n_agents", "dep_mode", "size", "n_obstacles",
    "coupled_waypoints", "method", "status", "team_time", "flow_time",
    "distance", "wait", "valid", "agent_clearance", "obstacle_clearance",
    "dep_violations", "n_switches", "runtime_s", "lower_bound", "ratio_to_bound",
    "git_commit",
]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:                                    # noqa: BLE001
        return "unknown"


def _blank(inst: Instance, method: str, lb: Optional[float],
           commit: str) -> Dict[str, object]:
    row = {k: "" for k in FIELDS}
    row.update(inst.row())
    row.update(method=method, status="ok", git_commit=commit,
               lower_bound=None if lb is None else round(lb, 3))
    return row


def _fill(row: Dict[str, object], problem, sol, rep, runtime: float,
          lb: Optional[float]) -> Dict[str, object]:
    row.update(
        team_time=round(sol.team_time, 3), flow_time=round(sol.flow_time, 3),
        distance=round(sol.travel_distance, 3), wait=round(sol.total_wait, 3),
        runtime_s=round(runtime, 3))
    if rep is not None:
        row.update(valid=bool(rep.valid),
                   agent_clearance=round(float(rep.agent_clearance), 4),
                   obstacle_clearance=round(float(rep.obstacle_clearance), 4),
                   dep_violations=int(rep.n_dep_violations))
        if not rep.valid:
            row["status"] = "invalid"
    if lb:
        row["ratio_to_bound"] = round(sol.team_time / lb, 4)
    return row


def run_instance(inst: Instance, wm_cfg: Dict, methods: List[str],
                 commit: str, astar_max_agents: int = 3) -> List[Dict[str, object]]:
    problem = buildable(inst)
    if problem is None:
        r = _blank(inst, "-", None, commit)
        r["status"] = "ungeneratable"
        return [r]

    try:
        tracks, _ = build_prior(problem)
        lb = critical_path_bound(problem, tracks)["lb_team_time"]
    except Exception as exc:                             # noqa: BLE001
        r = _blank(inst, "-", None, commit)
        r["status"] = f"no_prior:{type(exc).__name__}"
        return [r]

    rows: List[Dict[str, object]] = []

    if "lower_bound" in methods:
        r = _blank(inst, "lower_bound", lb, commit)
        r.update(team_time=round(lb, 3), ratio_to_bound=1.0, runtime_s=0.0)
        rows.append(r)

    if "sequential" in methods:
        r = _blank(inst, "sequential", lb, commit)
        t0 = time.perf_counter()
        try:
            sol = plan_sequential(problem, tracks)
            if sol.feasible:
                _fill(r, problem, sol, V.validate(problem, sol),
                      time.perf_counter() - t0, lb)
            else:
                r["status"] = "no_solution"
        except Exception as exc:                         # noqa: BLE001
            r["status"] = f"error:{type(exc).__name__}"
        rows.append(r)

    if "coordination_astar" in methods:
        r = _blank(inst, "coordination_astar", lb, commit)
        if problem.n > astar_max_agents:
            r["status"] = "skipped:lattice_too_large"    # B5 가 해결할 자리
        else:
            t0 = time.perf_counter()
            try:
                sol = plan_coordination(problem, tracks)
                if sol.feasible:
                    _fill(r, problem, sol, V.validate(problem, sol),
                          time.perf_counter() - t0, lb)
                else:
                    r["status"] = "no_solution"
            except MemoryError:
                r["status"] = "skipped:lattice_too_large"
            except Exception as exc:                     # noqa: BLE001
                r["status"] = f"error:{type(exc).__name__}"
        rows.append(r)

    if "wm_planner" in methods:
        r = _blank(inst, "wm_planner", lb, commit)
        t0 = time.perf_counter()
        try:
            res = run_wm_planner(problem, WMConfig(**wm_cfg))
            rep = V.validate(problem, res.traj)
            _fill(r, problem, res.traj, rep, time.perf_counter() - t0, lb)
            r["n_switches"] = res.n_switches
            if not res.traj.feasible:
                # note 에 "livelock" / "wall-clock" 등이 들어 있다
                r["status"] = f"unfinished:{res.note or 'unknown'}"[:60]
        except Exception as exc:                         # noqa: BLE001
            r["status"] = f"error:{type(exc).__name__}"
        rows.append(r)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="앞에서 N개 인스턴스만 (설정 확인용)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    axis = axis_from_config(cfg["axis"])
    methods = cfg.get("methods",
                      ["lower_bound", "sequential", "coordination_astar", "wm_planner"])
    wm_cfg = cfg.get("wm_config", {})

    out = args.out or os.path.join("bench", "runs", f"{cfg.get('name', 'run')}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    commit = git_commit()

    insts = list(grid(axis))
    if args.limit:
        insts = insts[:args.limit]
    print(f"{cfg.get('name', 'run')}: {len(insts)} instances x {len(methods)} methods"
          f"  (commit {commit})\n-> {out}\n", flush=True)

    t_start = time.perf_counter()
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for k, inst in enumerate(insts, 1):
            t0 = time.perf_counter()
            rows = run_instance(inst, wm_cfg, methods, commit)
            for r in rows:
                w.writerow(r)
            fh.flush()                        # 중간에 죽어도 여기까지는 남는다
            wm = next((r for r in rows if r["method"] == "wm_planner"), None)
            tag = f"{wm['team_time']}s {wm['status']}" if wm else rows[0]["status"]
            print(f"  [{k:>4}/{len(insts)}] {inst.uid:<34} {tag}"
                  f"   ({time.perf_counter() - t0:.1f}s)", flush=True)

    print(f"\n완료: {time.perf_counter() - t_start:.1f}s  ->  {out}")
    print(f"분석: python -m bench.analyze --csv {out}")


if __name__ == "__main__":
    main()
