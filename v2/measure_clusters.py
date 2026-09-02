#!/usr/bin/env python3
"""S5-2: n=2/3/4 층화 100개에서 관측 전용 클러스터를 재측정한다."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.generate import axis_from_config, buildable, grid
from planning.execute import WMConfig, run_wm_planner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="bench/configs/difficulty.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    by_n = {2: [], 3: [], 4: []}
    for inst in grid(axis_from_config(cfg["axis"])):
        if inst.n_agents in by_n:
            by_n[inst.n_agents].append(inst)
    take = {2: 34, 3: 33, 4: 33}
    selected = [x for n in (2, 3, 4) for x in by_n[n][:take[n]]]
    rows = []
    for i, inst in enumerate(selected, 1):
        problem = buildable(inst)
        if problem is None:
            raise RuntimeError(f"ungeneratable selected instance {inst.uid}")
        res = run_wm_planner(problem, WMConfig(seed=args.seed))
        rows.append({"uid": inst.uid, "n_agents": inst.n_agents,
                     "planner_seed": args.seed, "status": "ok" if res.traj.feasible else res.traj.note,
                     "replan_ms": res.replan_ms,
                     "records": [{"t": r["t"], "phase": r["phase"].tolist(),
                                  "pos": r["pos"].tolist()} for r in res.cluster_records]})
        print(f"[{i:3}/100] {inst.uid} {rows[-1]['status']}", flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"radius_m": 3.0, "sample": {str(n): [x.uid for x in by_n[n][:take[n]]]
                  for n in take}, "rows": rows}, fh)


if __name__ == "__main__":
    main()
