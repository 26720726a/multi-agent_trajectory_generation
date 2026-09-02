#!/usr/bin/env python3
"""배치의 표본 행을 **그 배치의 코드 상태에서** 다시 굴려 궤적만 npz 로 뺀다.

검증은 하지 않는다 — 현재 트리의 `safety/validate.py` 로 나중에 한 번에 한다.
그래야 궤적 생성은 각 시점 코드, 판정은 **하나의 검증기**가 된다 (S3-S §S-3).
"""
import csv, os, sys, json
import multiprocessing as mp
import numpy as np

ROOT = os.environ["WT_ROOT"]
sys.path.insert(0, ROOT)
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[v] = "1"

from planning.execute import WMConfig, run_wm_planner
from planning.world import random_problem


def rebuild(row):
    lo, hi = (row["n_obstacles"].split("-") + [row["n_obstacles"]])[:2]
    return random_problem(seed=int(row["instance_seed"]), n_agents=int(row["n_agents"]),
        size=float(row["size"]), n_obstacles=(int(lo), int(hi)), dep_mode=row["dep_mode"],
        coupled_waypoints=float(row["couple_prob"]), couple_dist=float(row["couple_dist"]))


def one(row):
    try:
        p = rebuild(row)
        res = run_wm_planner(p, WMConfig(seed=int(row["planner_seed"]), horizon_s=4.0,
            max_modes=16, keep_modes=8, k_routes=2, time_budget_s=120.0))
        return (row["uid"], row["planner_seed"], res.traj.pos, res.traj.vel,
                res.traj.wp_in, res.traj.wp_out, res.traj.done, "")
    except Exception as exc:                                   # noqa: BLE001
        return (row["uid"], row["planner_seed"], None, None, None, None, None,
                f"{type(exc).__name__}: {exc}")


def main():
    csv_path, keys_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    want = {tuple(k) for k in json.load(open(keys_path))}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["method"] == "wm_planner"
                and (r["uid"], r["planner_seed"]) in want]
    with mp.Pool(14) as pool:
        res = pool.map(one, rows, chunksize=1)
    d, errs = {}, {}
    for uid, ps, pos, vel, wi, wo, dn, err in res:
        k = f"{uid}|{ps}"
        if err:
            errs[k] = err; continue
        d[k + "|pos"] = pos; d[k + "|vel"] = vel
        d[k + "|wp_in"] = wi; d[k + "|wp_out"] = wo; d[k + "|done"] = dn
    np.savez_compressed(out, **d)
    json.dump(errs, open(out + ".errs.json", "w"))
    print(f"{os.path.basename(csv_path)}: {len(rows)}행 요청, "
          f"{len(d)//5} 성공, {len(errs)} 오류 -> {out}")


if __name__ == "__main__":
    main()
