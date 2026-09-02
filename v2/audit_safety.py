#!/usr/bin/env python3
"""과거 배치를 **하나의 검증기로** 다시 판정한다 (S3-S §S-3).

    python scripts/audit_safety.py --traj-dir <npz 들이 있는 곳>

두 방법을 대조한다.

  (a) CSV 의 `agent_clearance` / `obstacle_clearance` 열 기준
      — S3-B3 §7-1 이 쓴 방법.  **그 열 자체가 옳은지는 검증된 적이 없다.**
  (b) 궤적을 **그 배치의 코드 상태에서** 다시 굴리고,
      **현재 트리의** `safety/validate.py` 를 직접 호출한 결과

궤적 생성은 배치마다 다른 코드로, 판정은 하나의 검증기로 — 그래야 "코드가
달라서" 와 "검증기가 달라서" 가 섞이지 않는다.  궤적 덤프는 워크트리에서
따로 만들고(`dump_traj.py`), 여기서는 그것을 읽기만 한다.

어긋난 사례가 핵심 산출물이다.  어느 방향으로 어긋났는지(a 가 놓쳤나,
b 가 오탐인가)와 원인을 함께 낸다.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import safety.validate as V                                   # noqa: E402
from config import PHYSICS                                    # noqa: E402
from planning.traj import Trajectory                          # noqa: E402
from planning.world import random_problem                     # noqa: E402

#: `safety/validate.py` 가 쓰는 여유.  safety/README.md §5 참조.
TOL = 1e-3

BATCHES = ["baseline_S2", "A_amax3", "A2_braking",
           "A3_brakedist", "A4_nogeom", "B3_decel"]


def rebuild(row: dict):
    lo, hi = (row["n_obstacles"].split("-") + [row["n_obstacles"]])[:2]
    return random_problem(
        seed=int(row["instance_seed"]), n_agents=int(row["n_agents"]),
        size=float(row["size"]), n_obstacles=(int(lo), int(hi)),
        dep_mode=row["dep_mode"], coupled_waypoints=float(row["couple_prob"]),
        couple_dist=float(row["couple_dist"]))


def fnum(row: dict, col: str) -> Optional[float]:
    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        return None


def col_verdict(row: dict) -> bool:
    """(a) CSV clearance 열만 보고 내리는 판정."""
    a = fnum(row, "agent_clearance")
    o = fnum(row, "obstacle_clearance")
    return ((a is not None and a < PHYSICS.min_sep - TOL)
            or (o is not None and o < PHYSICS.robot_radius - TOL))


def audit(name: str, traj_dir: str) -> dict:
    path = f"results/s3/{name}.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        rows = {(r["uid"], r["planner_seed"]): r
                for r in csv.DictReader(fh) if r["method"] == "wm_planner"}
    npz = np.load(os.path.join(traj_dir, f"traj_{name}.npz"))
    keys = sorted({k.rsplit("|", 1)[0] for k in npz.files})

    checked, hits, mism = 0, [], []
    for key in keys:
        uid, ps = key.split("|")
        row = rows.get((uid, ps))
        if row is None:
            continue
        problem = rebuild(row)
        traj = Trajectory(
            method="replay", pos=npz[key + "|pos"], vel=npz[key + "|vel"],
            dt=problem.dt, wp_in=npz[key + "|wp_in"],
            wp_out=npz[key + "|wp_out"], done=npz[key + "|done"],
            extra={"v_max": [a.v_max for a in problem.agents]})
        rep = V.validate(problem, traj)
        checked += 1
        live = (rep.n_agent_violations > 0) or (rep.n_obstacle_violations > 0)
        from_col = col_verdict(row)
        if live:
            hits.append((row, rep))
        if live != from_col:
            mism.append((row, rep, live, from_col))
    all_col = [r for r in rows.values() if col_verdict(r)]
    return {"name": name, "n_rows": len(rows), "checked": checked,
            "col_all": all_col, "hits": hits, "mism": mism}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True)
    args = ap.parse_args()

    print("=" * 100)
    print(f"S3-S §S-3 재판정   판정 = safety/validate.py (여유 {TOL} m)")
    print(f"  에이전트 쌍 >= min_sep {PHYSICS.min_sep},  "
          f"장애물/벽 >= radius {PHYSICS.robot_radius}")
    print("  궤적은 **각 배치의 코드 상태**에서 다시 굴린 것, 판정은 **현재 검증기** 하나")
    print("=" * 100)
    print(f"\n{'배치':<15} {'전체행':>7} {'표본':>6} {'(a)CSV열 전체':>13} "
          f"{'(b)재실행 침범':>14} {'불일치':>7}")
    out = []
    for n in BATCHES:
        r = audit(n, args.traj_dir)
        out.append(r)
        print(f"  {n:<13} {r['n_rows']:>7} {r['checked']:>6} "
              f"{len(r['col_all']):>13} {len(r['hits']):>14} {len(r['mism']):>7}",
              flush=True)

    total_mism = sum(len(r["mism"]) for r in out)
    print(f"\n불일치 총계: {total_mism}건 / 표본 {sum(r['checked'] for r in out)}행")

    for r in out:
        if not r["mism"]:
            continue
        print(f"\n### 불일치 — {r['name']} ({len(r['mism'])}건)")
        for row, rep, live, from_col in r["mism"][:12]:
            direction = ("(a)가 놓쳤다" if live and not from_col
                         else "(a)만 침범이라 했다")
            print(f"  {row['uid']} ps={row['planner_seed']}  {direction}")
            print(f"     CSV  agent={row.get('agent_clearance')} "
                  f"obst={row.get('obstacle_clearance')}")
            print(f"     검증 agent_viol={rep.n_agent_violations} "
                  f"obst_viol={rep.n_obstacle_violations}  "
                  f"agent_clr={rep.agent_clearance:.6f} "
                  f"obst_clr={rep.obstacle_clearance:.6f}  "
                  f"first_t={rep.first_violation_t}")

    for r in out:
        if not r["hits"]:
            continue
        print(f"\n### 재실행에서 침범 — {r['name']} ({len(r['hits'])}건)")
        for row, rep in sorted(r["hits"],
                               key=lambda x: min(x[1].agent_clearance,
                                                 x[1].obstacle_clearance))[:8]:
            print(f"  {row['uid']} ps={row['planner_seed']}  "
                  f"agent_viol={rep.n_agent_violations} "
                  f"obst_viol={rep.n_obstacle_violations}  "
                  f"agent_clr={rep.agent_clearance:.6f} "
                  f"obst_clr={rep.obstacle_clearance:.6f}  "
                  f"first_t={rep.first_violation_t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
