#!/usr/bin/env python3
"""전멸 뒤 감속 구간에서 안전 여유가 어떻게 되는가 (S3-B3 §5).

    python scripts/diag_decel.py results/s3/B3_decel.csv --limit 60

B3 는 전멸했을 때 **즉시 정지 대신 `a_max` 로 감속**한다.  그러면 서는 데
최대 4 스텝이 걸리고 그동안 계속 움직인다.  **그 움직임이 위험한가**를 재는
것이 이 도구다 — 게이트 2(충돌 0건)가 왜 통과했는지(혹은 깨졌는지)를 설명한다.

각 전멸 사건에서 **그 스텝부터 속도가 0 이 될 때까지**를 감속 구간으로 보고,
구간 안의 최소 여유를 잰다.

    이웃 여유   min(중심거리 - min_sep)      < 0 이면 충돌
    장애물 여유 min(최근접거리 - (r + pad))  < 0 이면 침범

`--baseline` 을 주면 같은 인스턴스를 **다른 트리**(예: 즉시 정지 버전)에서도
돌려 짝지어 비교한다.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONTROLLER, PHYSICS                       # noqa: E402
from planning.control import A_MAX, SceneCache               # noqa: E402
from planning.execute import WMConfig, run_wm_planner        # noqa: E402
from planning.world import random_problem                    # noqa: E402


def rebuild(row: dict):
    lo, hi = (row["n_obstacles"].split("-") + [row["n_obstacles"]])[:2]
    return random_problem(
        seed=int(row["instance_seed"]), n_agents=int(row["n_agents"]),
        size=float(row["size"]), n_obstacles=(int(lo), int(hi)),
        dep_mode=row["dep_mode"], coupled_waypoints=float(row["couple_prob"]),
        couple_dist=float(row["couple_dist"]))


def episodes(row: dict, max_len: int = 8) -> List[dict]:
    """전멸 사건마다 감속 구간의 최소 여유를 잰다."""
    problem = rebuild(row)
    scene = SceneCache(problem)
    trace: List[dict] = []
    run_wm_planner(problem, WMConfig(
        seed=int(row["planner_seed"]), horizon_s=4.0, max_modes=16,
        keep_modes=8, k_routes=2, time_budget_s=120.0), trace=trace)
    by_t = {r["t"]: r for r in trace}
    pad = PHYSICS.robot_radius + CONTROLLER.obst_pad
    out = []
    for rec in trace:
        for i in np.flatnonzero(rec["forced"]):
            i = int(i)
            v0 = float(np.linalg.norm(rec["vel_prev"][i]))
            if v0 <= 1e-9:
                continue                       # 이미 서 있었다 — 감속 구간 없음
            steps, agap, ogap = 0, math.inf, math.inf
            for k in range(max_len):
                cur = by_t.get(rec["t"] + k)
                if cur is None:
                    break
                p = cur["pos"]
                for j in range(problem.n):
                    if j == i:
                        continue
                    agap = min(agap, float(np.linalg.norm(p[i] - p[j]))
                               - problem.min_sep(i, j))
                near = scene.nearest_points(p[i])
                ogap = min(ogap, float(np.min(
                    np.linalg.norm(near - p[i][None, :], axis=1))) - pad)
                steps = k + 1
                if float(np.linalg.norm(cur["v_cmd"][i])) <= 1e-9:
                    break
            out.append({"uid": row["uid"], "seed": int(row["planner_seed"]),
                        "t": int(rec["t"]), "agent": i, "v0": v0,
                        "steps_to_stop": steps,
                        "agent_gap": agap, "obst_gap": ogap})
    return out


def q(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * (len(s) - 1)))] if s else float("nan")


def report(name: str, eps: List[dict]) -> None:
    if not eps:
        print(f"[{name}] 감속 구간이 있는 전멸 사건 없음")
        return
    print(f"\n[{name}]  움직이는 전멸(|v_prev|>0) {len(eps)}건")
    s = [e["steps_to_stop"] for e in eps]
    print(f"  정지까지 스텝: 중앙값 {q(s,.5):.0f}  p90 {q(s,.9):.0f}  최대 {max(s)}")
    for key, lab, hard in (("agent_gap", "이웃 여유 (하한 대비)", 0.0),
                           ("obst_gap", "장애물 여유 (팽창 대비)", 0.0)):
        v = [e[key] for e in eps if math.isfinite(e[key])]
        if not v:
            continue
        bad = sum(1 for x in v if x < hard)
        print(f"  {lab}: 최소 {min(v):+.4f}  p10 {q(v,.10):+.4f}  "
              f"중앙값 {q(v,.5):+.4f} m   음수(침범) {bad}/{len(v)}")
    v0 = [e["v0"] for e in eps]
    print(f"  전멸 시점 속력: 중앙값 {q(v0,.5):.3f}  최대 {max(v0):.3f} m/s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["method"] == "wm_planner"]

    def nb(r):
        try:
            return float(r.get("n_blocked") or 0)
        except ValueError:
            return 0.0

    have = [r for r in rows if nb(r) > 0]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(have), size=min(args.limit, len(have)), replace=False)
    picked = [have[int(k)] for k in sorted(idx)]
    print(f"{args.csv}: 전멸이 있는 {len(have)}행 중 무작위 {len(picked)}행 재실행")
    print(f"  a_max={A_MAX}  dv_max={A_MAX*PHYSICS.dt:.2f} m/s  "
          f"min_sep={PHYSICS.min_sep}  팽창 pad={PHYSICS.robot_radius+CONTROLLER.obst_pad:.2f}")

    eps: List[dict] = []
    for k, r in enumerate(picked, 1):
        try:
            eps += episodes(r)
        except Exception as exc:                              # noqa: BLE001
            print(f"  [{k}] {r['uid']} 실패: {exc}", file=sys.stderr)
        if k % 20 == 0:
            print(f"  ... {k}/{len(picked)}, 사건 {len(eps)}건", flush=True)
    report(os.path.basename(args.csv), eps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
