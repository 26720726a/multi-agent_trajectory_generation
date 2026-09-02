#!/usr/bin/env python3
"""S3-F 실패 원인 재분류: 기존 CSV의 baseline 성공 -> A4 실패를 측정한다.

알고리즘에는 손대지 않는다. A4 재실행은 현재 설정 그대로, baseline 재실행은
과거 `a_max=inf` 동작을 보기 위해 실행 중인 프로세스의 control.A_MAX만 잠시
`inf`로 바꾼다. 이는 후보 필터의 inf 경로를 재현하는 계측용 override이며,
파일/전역 config를 쓰지 않는다.

예:
  python3 scripts/diag_failure_taxonomy.py \
    results/s3/baseline_S2.csv results/s3/A4_nogeom.csv \
    results/s3/B2_multistep.csv --sample 60 --workers 15 --out results/s3/S3F
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONTROLLER, PHYSICS  # noqa: E402
from planning import control as C  # noqa: E402
from planning.execute import WMConfig, run_wm_planner  # noqa: E402
from planning.world import random_problem  # noqa: E402


KEY = ("uid", "planner_seed")
CFG = dict(horizon_s=4.0, replan_s=1.0, max_modes=16, keep_modes=8,
           k_routes=2, time_budget_s=120.0)


def load(path: str) -> Dict[Tuple[str, str], dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {tuple(r[k] for k in KEY): r for r in csv.DictReader(fh)
                if r["method"] == "wm_planner"}


def rebuild(r: dict):
    lo, hi = (r["n_obstacles"].split("-") + [r["n_obstacles"]])[:2]
    return random_problem(seed=int(r["instance_seed"]), n_agents=int(r["n_agents"]),
                          size=float(r["size"]), n_obstacles=(int(lo), int(hi)),
                          dep_mode=r["dep_mode"],
                          coupled_waypoints=float(r["couple_prob"]),
                          couple_dist=float(r["couple_dist"]))


def run(row: dict, inf: bool, trace: bool = False):
    """한 프로세스 안에서 baseline/A4를 나란히 재실행한다."""
    old = C.A_MAX
    try:
        C.A_MAX = math.inf if inf else old
        tr: List[dict] = []
        res = run_wm_planner(rebuild(row), WMConfig(seed=int(row["planner_seed"]),
                                                     **CFG), trace=tr if trace else None)
        return res, tr
    finally:
        C.A_MAX = old


def terminal_state(problem, res, trace: List[dict]) -> dict:
    """실패 직전 100스텝의 정지 형태를 수치로 남긴다. 유형 이름은 여기서 정하지 않는다."""
    traj = res.traj
    T = traj.T
    lo = max(0, T - 100)
    speed = np.linalg.norm(traj.vel[lo:T], axis=2)
    still = speed < 0.03
    durations = []
    for i in range(problem.n):
        n = 0
        for x in still[::-1, i]:
            if not x:
                break
            n += 1
        durations.append(n)
    final = traj.pos[T]
    pairs = []
    mutual = 0
    for i in range(problem.n):
        for j in range(i + 1, problem.n):
            dvec = final[j] - final[i]
            d = float(np.linalg.norm(dvec))
            u = dvec / max(d, 1e-12)
            vi, vj = traj.vel[max(T - 1, 0), i], traj.vel[max(T - 1, 0), j]
            toward = float(vi @ u) > 0.02 and float(vj @ -u) > 0.02
            near = d < problem.min_sep(i, j) + 0.35
            mutual += int(toward and near)
            pairs.append({"pair": [i, j], "distance": d, "margin": d - problem.min_sep(i, j),
                          "toward_each_other": toward})
    tail = trace[-100:] if trace else []
    forced = np.zeros(problem.n, int)
    dep = np.zeros(problem.n, int)
    pref0 = np.zeros(problem.n, int)
    low_cmd = np.zeros(problem.n, int)
    for rec in tail:
        forced += rec["forced"].astype(int)
        dep += rec["dep_hold"].astype(int)
        pref0 += (np.linalg.norm(rec["v_pref"], axis=1) < 0.03).astype(int)
        low_cmd += (np.linalg.norm(rec["v_cmd"], axis=1) < 0.03).astype(int)
    agents = []
    for i, a in enumerate(problem.agents):
        # 원인은 수치로만 기록한다. 명명은 F-1의 60개를 본 뒤 F-2에서 한다.
        agents.append({"agent": i, "phase": int(trace[-1]["phase"][i]) if trace else None,
                       "pos": final[i].round(4).tolist(),
                       "speed": float(np.linalg.norm(traj.vel[max(T - 1, 0), i])),
                       "still_steps": durations[i], "forced_tail": int(forced[i]),
                       "dep_tail": int(dep[i]), "pref0_tail": int(pref0[i]),
                       "low_cmd_tail": int(low_cmd[i])})
    return {"T": T, "window_start": lo, "n_still": int(np.count_nonzero(np.array(durations) >= 20)),
            "mutual_near_toward_pairs": mutual, "pairs": pairs, "agents": agents}


def divergence(problem, base, a4) -> dict:
    """동일 t에서 위치 차이가 처음 0.15m/0.30m을 넘는 시점과 속도 차이를 낸다."""
    m = min(base.traj.T, a4.traj.T)
    d = np.max(np.linalg.norm(base.traj.pos[:m + 1] - a4.traj.pos[:m + 1], axis=2), axis=1)
    def first(x):
        hit = np.flatnonzero(d >= x)
        return int(hit[0]) if len(hit) else None
    t = first(0.30)
    if t is None:
        t = first(0.15)
    if t is None:
        t = m
    lo, hi = max(0, t - 3), min(m, t + 3)
    return {"first_015": first(0.15), "first_030": first(0.30), "chosen_t": t,
            "max_position_delta": float(d.max()),
            "window": [{"t": k, "pos_delta_max": float(d[k]),
                        "baseline_speed": np.linalg.norm(base.traj.vel[k], axis=1).round(3).tolist(),
                        "a4_speed": np.linalg.norm(a4.traj.vel[k], axis=1).round(3).tolist()}
                       for k in range(lo, hi + 1)]}


def one_f1(row: dict) -> dict:
    problem = rebuild(row)
    base, _ = run(row, inf=True, trace=False)
    a4, trace = run(row, inf=False, trace=True)
    return {"uid": row["uid"], "planner_seed": int(row["planner_seed"]),
            "baseline": {"feasible": base.traj.feasible, "T": base.traj.T,
                         "team_time": base.traj.team_time},
            "a4": {"feasible": a4.traj.feasible, "T": a4.traj.T,
                   "team_time": a4.traj.team_time, "note": a4.traj.note,
                   "accel": a4.traj.extra["accel"]},
            "terminal": terminal_state(problem, a4, trace),
            "divergence": divergence(problem, base, a4)}


def q(xs: Iterable[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1)))] if xs else float("nan")


def static_features(row: dict) -> dict:
    """F-3의 난이도 대리변수. route library는 만들지 않고 문제 기하에서 직접 잰다."""
    p = rebuild(row)
    pts = np.array([x for a in p.agents for x in (a.start, a.waypoint, a.goal)])
    pairs = [float(np.linalg.norm(pts[i] - pts[j])) for i in range(len(pts))
             for j in range(i + 1, len(pts))]
    straight = [float(np.linalg.norm(np.asarray(a.goal) - np.asarray(a.start))) for a in p.agents]
    via_wp = [float(np.linalg.norm(np.asarray(a.waypoint) - np.asarray(a.start)) +
                    np.linalg.norm(np.asarray(a.goal) - np.asarray(a.waypoint))) for a in p.agents]
    area = sum((r.x1-r.x0)*(r.y1-r.y0) for r in p.world.obstacles) / (p.world.width*p.world.height)
    # 장애물/벽까지의 최근접 거리를 S/W/G에서 재어 "좁음" 대리변수로 쓴다.
    clear = []
    for x in pts:
        wall = min(x[0], x[1], p.world.width-x[0], p.world.height-x[1])
        obs = []
        for r in p.world.obstacles:
            dx = max(r.x0-x[0], 0.0, x[0]-r.x1)
            dy = max(r.y0-x[1], 0.0, x[1]-r.y1)
            obs.append(math.hypot(dx, dy))
        clear.append(min([wall] + obs))
    return {"path_via_over_straight": float(np.mean(np.divide(via_wp, np.maximum(straight, 1e-9)))),
            "obstacle_density": area, "min_point_clearance": min(clear),
            "min_task_point_distance": min(pairs), "n_agents": p.n,
            "size": p.world.width, "obstacles": row["n_obstacles"]}


def f3(baseline: Dict, a4: Dict, b2: Dict) -> dict:
    keys = sorted(set(baseline) & set(a4) & set(b2))
    lost = [k for k in keys if baseline[k]["status"] == "ok" and a4[k]["status"] != "ok"]
    both = [k for k in keys if baseline[k]["status"] == a4[k]["status"] == "ok"]
    # CSV의 기하축으로 우선 층화하고, 새 대리변수는 군 평균/분위수로 비교한다.
    def rows(ks):
        out = []
        for k in ks:
            x = static_features(a4[k])
            x.update(group="lost" if k in set(lost) else "both",
                     wipe=float(a4[k].get("n_amax_viol_block") or 0),
                     b2_wipe=float(b2[k].get("n_amax_viol_block") or 0),
                     a4_ok=a4[k]["status"] == "ok", b2_ok=b2[k]["status"] == "ok")
            out.append(x)
        return out
    r_lost, r_both = rows(lost), rows(both)
    features = ("path_via_over_straight", "obstacle_density", "min_point_clearance", "min_task_point_distance")
    summary = {g: {f: {"median": q((x[f] for x in rs), .5), "p10": q((x[f] for x in rs), .1),
                         "p90": q((x[f] for x in rs), .9)} for f in features}
               for g, rs in (("lost", r_lost), ("both", r_both))}
    # A4 실패/성공을 동일 (n,size,obst)층 안에서 전멸 있음/없음으로 비교한다.
    strata = defaultdict(list)
    for k in keys:
        r = a4[k]
        strata[(r["n_agents"], r["size"], r["n_obstacles"])].append(k)
    effects = []
    for label, ks in strata.items():
        yes = [a4[k]["status"] == "ok" for k in ks if float(a4[k].get("n_amax_viol_block") or 0) > 0]
        no = [a4[k]["status"] == "ok" for k in ks if float(a4[k].get("n_amax_viol_block") or 0) == 0]
        if yes and no:
            effects.append({"stratum": label, "n_wipe": len(yes), "n_none": len(no),
                            "success_wipe": sum(yes)/len(yes), "success_none": sum(no)/len(no),
                            "delta": sum(yes)/len(yes)-sum(no)/len(no)})
    reduced = []
    for k in keys:
        aw, bw = float(a4[k].get("n_amax_viol_block") or 0), float(b2[k].get("n_amax_viol_block") or 0)
        if bw < aw:
            reduced.append({"a4_ok": a4[k]["status"] == "ok", "b2_ok": b2[k]["status"] == "ok",
                            "reduction": aw-bw})
    return {"n_lost": len(lost), "n_both": len(both), "feature_summary": summary,
            "strata": effects, "reduced": {"n": len(reduced),
                "still_failed": sum(not x["b2_ok"] for x in reduced),
                "a4_failed_b2_failed": sum(not x["a4_ok"] and not x["b2_ok"] for x in reduced),
                "a4_failed_b2_ok": sum(not x["a4_ok"] and x["b2_ok"] for x in reduced)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("a4")
    ap.add_argument("b2")
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True, help="출력 디렉터리")
    args = ap.parse_args()
    base, a4, b2 = load(args.baseline), load(args.a4), load(args.b2)
    lost = [a4[k] for k in sorted(set(base) & set(a4))
            if base[k]["status"] == "ok" and a4[k]["status"] != "ok"]
    rng = np.random.default_rng(args.seed)
    chosen = [lost[int(i)] for i in sorted(rng.choice(len(lost), size=min(args.sample, len(lost)), replace=False))]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"F-1/F-4: baseline 성공 -> A4 실패 {len(lost)}건 중 무작위 {len(chosen)}건")
    if args.workers == 1:
        sample = [one_f1(r) for r in chosen]
    else:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            sample = list(pool.imap(one_f1, chosen))
    (out / "f1_sample.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "f3.json").write_text(json.dumps(f3(base, a4, b2), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out/'f1_sample.json'}\n-> {out/'f3.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
