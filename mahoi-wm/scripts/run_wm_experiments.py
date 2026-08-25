#!/usr/bin/env python3
"""World Model + Planner experiments on the five 8/18 scenarios.

    python scripts/run_wm_experiments.py                 # everything
    python scripts/run_wm_experiments.py --only crossing2 corridor2
    python scripts/run_wm_experiments.py --no-gif --no-ablation

For every scenario it runs, in one place, the four methods that are directly
comparable -- the critical-path lower bound, the sequential baseline, the fixed
route coordination A* from 8/18, and the new World Model + Planner -- validates
each with the *independent* checker, and writes the figures and tables.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mahoi import validate as V
from mahoi import viz
from mahoi.coordination import (critical_path_bound, plan_coordination,
                                plan_independent, plan_sequential)
from mahoi.paths import build_prior
from mahoi.world import SCENARIOS, get_scenario
from mahoi.wm import viz as wviz
from mahoi.wm.execute import WMConfig, run_wm_planner
from mahoi.wm.planner import CostWeights, cost_table

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "wm")

#: The headline configuration.  Four seconds of imagination is short enough to
#: stay cheap and long enough to cover a typical crossing; the plan is revisited
#: every second, which is what makes the scheme *reactive* rather than a
#: one-shot open-loop plan.
MAIN = dict(horizon_s=4.0, replan_s=1.0, max_modes=16, keep_modes=8, k_routes=2)

ABLATIONS = [
    ("WM+P  H=8s",       dict(MAIN, horizon_s=8.0)),
    ("WM+P  full view",  dict(MAIN, horizon_s=None)),
    ("WM+P  no switching", dict(MAIN, horizon_s=None, switching=False)),
]


# --------------------------------------------------------------------------- #
def _metrics(problem, sol, rep) -> Dict[str, object]:
    return {
        "team_time": round(sol.team_time, 2) if sol.feasible else None,
        "flow_time": round(sol.flow_time, 2) if sol.feasible else None,
        "distance": round(sol.travel_distance, 2) if sol.feasible else None,
        "wait": round(sol.total_wait, 2) if sol.feasible else None,
        "runtime_s": round(sol.runtime, 2),
        "valid": bool(rep.valid) if rep is not None else None,
        "agent_clearance": round(float(rep.agent_clearance), 3) if rep else None,
        "obstacle_clearance": round(float(rep.obstacle_clearance), 3) if rep else None,
        "dep_violations": int(rep.n_dep_violations) if rep else None,
        "errors": (rep.errors[:4] if rep else []),
    }


def run_scenario(name: str, gif: bool = True, ablation: bool = True,
                 verbose: bool = True) -> Dict[str, object]:
    problem = get_scenario(name)
    outdir = os.path.join(OUT, name)
    os.makedirs(outdir, exist_ok=True)
    print(f"\n{'=' * 78}\n{name}   ({problem.n} agents, deps {problem.dep_str()})"
          f"\n  {problem.description}\n{'=' * 78}", flush=True)

    row: Dict[str, object] = {"scenario": name, "n_agents": problem.n,
                              "deps": problem.dep_str(),
                              "description": problem.description}

    # ---- references from the 8/18 pipeline -------------------------------- #
    tracks, _ = build_prior(problem)
    bound = critical_path_bound(problem, tracks)
    row["lower_bound"] = round(bound["lb_team_time"], 2)
    print(f"  critical-path lower bound      {row['lower_bound']:6.2f} s")

    prior = plan_independent(problem, tracks)
    rep_prior = V.validate(problem, prior)
    row["independent_prior"] = _metrics(problem, prior, rep_prior)

    t0 = time.perf_counter()
    seq = plan_sequential(problem, tracks)
    rep_seq = V.validate(problem, seq) if seq.feasible else None
    row["sequential"] = _metrics(problem, seq, rep_seq)
    print(f"  sequential baseline            "
          f"{row['sequential']['team_time'] or 'no solution':>9} s "
          f"({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    try:
        coord = plan_coordination(problem, tracks)
    except MemoryError as e:
        coord = None
        print(f"  coordination A*                skipped ({e})")
    rep_coord = V.validate(problem, coord) if coord is not None and coord.feasible else None
    row["coordination_astar"] = _metrics(problem, coord, rep_coord) if coord else None
    if coord is not None:
        print(f"  coordination A* (fixed routes) "
              f"{row['coordination_astar']['team_time'] or 'no solution':>9} s "
              f"({time.perf_counter() - t0:.1f}s)")

    # ---- the new method --------------------------------------------------- #
    print(f"\n  --- World Model + Planner "
          f"(imagine {MAIN['horizon_s']}s, replan every {MAIN['replan_s']}s, "
          f"{MAIN['max_modes']} futures) ---", flush=True)
    weights = CostWeights()
    res = run_wm_planner(problem, WMConfig(**MAIN), weights, verbose=verbose)
    rep_wm = V.validate(problem, res.traj)
    row["wm_planner"] = _metrics(problem, res.traj, rep_wm)
    row["wm_planner"].update(
        n_decisions=len(res.switches), n_switches=res.n_switches,
        n_modes=len(res.modes),
        n_feasible_at_t0=int(sum(1 for c in res.first_table if c.feasible)))
    print(f"\n  {rep_wm.summary()}   team {res.traj.team_time:.2f}s  "
          f"({res.runtime:.1f}s cpu, {res.n_switches} switches)")
    for e in rep_wm.errors[:4]:
        print(f"    ! {e}")

    with open(os.path.join(outdir, "cost_table_t0.txt"), "w") as fh:
        fh.write(f"# {name}: Planner cost table at t = 0\n")
        fh.write(cost_table(res.first_table) + "\n")

    # ---- figures ---------------------------------------------------------- #
    # The executor imagines only `horizon_s` ahead, which is the right thing for
    # planning but shows almost nothing in a picture.  For the illustration we
    # re-roll the same mode set to completion, so the fan displays the futures
    # the World Model can actually express.
    wm, planner_w = res.world_model, weights
    full0 = wm.rollouts(res.states[0], res.modes, horizon=None)
    from mahoi.wm.planner import Planner as _P
    best0, costs0 = _P(problem, weights).select(full0, wm)
    wviz.rollout_fan(problem, full0, best0,
                     path=os.path.join(outdir, "fan_t0.png"),
                     title=f"{name}: {len(full0)} futures imagined at t = 0")
    wviz.cost_bars(costs0, weights,
                   path=os.path.join(outdir, "cost_t0.png"))
    with open(os.path.join(outdir, "cost_table_t0_fullview.txt"), "w") as fh:
        fh.write(f"# {name}: cost table at t = 0, rollouts run to completion\n")
        fh.write(cost_table(costs0) + "\n")

    # ...and again from part-way through, to show that the imagined futures --
    # and therefore the choice -- move with the state.
    mid = min(len(res.states) - 1, max(1, len(res.states) // 3))
    if mid >= 1:
        full_m = wm.rollouts(res.states[mid], res.live_modes or res.modes,
                             horizon=None)
        best_m, _ = _P(problem, weights).select(full_m, wm)
        wviz.rollout_fan(problem, full_m, best_m,
                         path=os.path.join(outdir, "fan_mid.png"),
                         title=f"{name}: futures re-imagined at "
                               f"t = {res.states[mid].t * problem.dt:.1f}s")
    wviz.switch_timeline(problem, res.switches, res.traj,
                         path=os.path.join(outdir, "switches.png"))

    sols, titles, colours = [prior], ["independent prior\n(infeasible)"], ["#B00000"]
    if coord is not None and coord.feasible:
        sols.append(coord)
        titles.append(f"coordination A* (fixed routes)\n{coord.team_time:.1f}s")
        colours.append("#3C6BB0")
    sols.append(res.traj)
    titles.append(f"world model + planner\n{res.traj.team_time:.1f}s")
    colours.append("#1a7f37")

    t_show = int(np.argmin([
        min(np.linalg.norm(prior.xy[t, i] - prior.xy[t, j])
            for i in range(problem.n) for j in range(i + 1, problem.n))
        for t in range(prior.T + 1)])) if problem.n > 1 else 0
    viz.comparison_figure(problem, sols, titles, colours, t=t_show,
                          path=os.path.join(outdir, "compare.png"),
                          suptitle=f"{name} -- at t = {t_show * problem.dt:.1f}s "
                                   f"(closest approach of the prior)")
    viz.gantt(problem, sols, [t.split("\n")[0] for t in titles],
              path=os.path.join(outdir, "gantt.png"))
    wviz.speed_trace(problem, [res.traj], ["world model + planner"],
                     path=os.path.join(outdir, "speed.png"))
    if gif:
        viz.animate_comparison(problem, sols, titles, colours,
                               path=os.path.join(outdir, f"anim_{name}.gif"))

    # ---- ablations -------------------------------------------------------- #
    if ablation:
        print("\n  --- ablations ---", flush=True)
        row["ablations"] = {}
        for label, kw in ABLATIONS:
            try:
                r = run_wm_planner(problem, WMConfig(**kw), weights, verbose=False)
                rp = V.validate(problem, r.traj)
                m = _metrics(problem, r.traj, rp)
                m.update(n_switches=r.n_switches)
                row["ablations"][label] = m
                print(f"    {label:<20} team={m['team_time'] or 'x':>6} s  "
                      f"flow={m['flow_time']:>6} s  switches={r.n_switches:<3} "
                      f"{'OK' if rp.valid else 'FAIL'}  ({r.runtime:.1f}s cpu)")
            except Exception as exc:                       # noqa: BLE001
                row["ablations"][label] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"    {label:<20} FAILED: {exc}")

    return row


# --------------------------------------------------------------------------- #
def summary_markdown(rows: List[Dict[str, object]]) -> str:
    def cell(d: Optional[Dict], key: str = "team_time") -> str:
        if not d or d.get(key) is None:
            return "no solution"
        return f"{d[key]:.1f}"

    L = ["# World Model + Planner -- results", "",
         "Team completion time in seconds.  `lower bound` is the dependency-aware",
         "critical path with collisions relaxed away, so no method can beat it.", "",
         "| scenario | agents | lower bound | sequential | coordination A* "
         "(fixed routes) | **world model + planner** | vs bound | vs A\\* |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lb = r["lower_bound"]
        wm = r["wm_planner"]["team_time"]
        ca = (r["coordination_astar"] or {}).get("team_time")
        gap_lb = f"+{100 * (wm - lb) / lb:.1f} %" if wm else "-"
        gap_ca = f"{100 * (wm - ca) / ca:+.1f} %" if (wm and ca) else "-"
        L.append(f"| {r['scenario']} | {r['n_agents']} | {lb:.1f} | "
                 f"{cell(r['sequential'])} | {cell(r['coordination_astar'])} | "
                 f"**{cell(r['wm_planner'])}** | {gap_lb} | {gap_ca} |")

    L += ["", "## Constraint satisfaction (independent validator)", "",
          "| scenario | valid | min agent gap (m) | required (m) | "
          "min obstacle clearance (m) | dependency violations |", "|---|---|---|---|---|---|"]
    for r in rows:
        w = r["wm_planner"]
        prob = get_scenario(r["scenario"])
        L.append(f"| {r['scenario']} | {'OK' if w['valid'] else 'FAIL'} | "
                 f"{w['agent_clearance']} | {prob.min_sep(0, 1):.2f} | "
                 f"{w['obstacle_clearance']} | {w['dep_violations']} |")

    L += ["", "## Planner activity", "",
          "| scenario | futures imagined | feasible at t=0 | decisions | "
          "mid-flight switches | cpu (s) |", "|---|---|---|---|---|---|"]
    for r in rows:
        w = r["wm_planner"]
        L.append(f"| {r['scenario']} | {w['n_modes']} | {w['n_feasible_at_t0']} | "
                 f"{w['n_decisions']} | {w['n_switches']} | {w['runtime_s']} |")

    if any("ablations" in r for r in rows):
        labels = [lab for lab, _ in ABLATIONS]
        L += ["", "## Ablations (team completion time, s)", "",
              "| scenario | " + " | ".join(["WM+P  main (H=4s)"] + labels) + " |",
              "|---" * (len(labels) + 2) + "|"]
        for r in rows:
            cells = [cell(r["wm_planner"])]
            for lab in labels:
                cells.append(cell((r.get("ablations") or {}).get(lab)))
            L.append(f"| {r['scenario']} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, choices=list(SCENARIOS))
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    names = args.only or list(SCENARIOS)
    rows = [run_scenario(n, gif=not args.no_gif, ablation=not args.no_ablation,
                         verbose=not args.quiet) for n in names]

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    md = summary_markdown(rows)
    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write(md)

    flat = []
    for r in rows:
        flat.append({
            "scenario": r["scenario"], "lower_bound": r["lower_bound"],
            "sequential": (r["sequential"] or {}).get("team_time"),
            "coordination A*": (r["coordination_astar"] or {}).get("team_time"),
            "world model + planner": r["wm_planner"]["team_time"]})
    wviz.method_bars(flat,
                     ["sequential", "coordination A*", "world model + planner"],
                     ["#B0B7BE", "#3C6BB0", "#1a7f37"],
                     path=os.path.join(OUT, "team_time.png"))

    print("\n" + md)
    print(f"outputs -> {OUT}")


if __name__ == "__main__":
    main()
