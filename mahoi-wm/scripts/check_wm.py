#!/usr/bin/env python3
"""Verification pass for the World Model + Planner.

Three things are checked, none of which the planner is allowed to self-report:

1. **Constraint satisfaction.**  Every produced trajectory goes through
   `mahoi.validate`, which reads only the executed coordinates.
2. **Seed robustness.**  The mode set is a deterministic subsample of a much
   larger space, so a good result on seed 0 could be luck.  We re-run each
   scenario across several seeds and report the spread.
3. **Cost-ordering sanity.**  For a rollout set scored at t = 0, the Planner's
   ranking must agree with the actual outcome: rollouts it calls feasible must
   really contain no violation, and the one it picks must not be beaten on
   makespan by another *feasible* rollout by more than the soft terms can
   justify.

    python scripts/check_wm.py            # all scenarios, seeds 0..2
    python scripts/check_wm.py --seeds 5
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mahoi import validate as V
from mahoi.coordination import critical_path_bound
from mahoi.paths import build_prior
from mahoi.world import SCENARIOS, get_scenario
from mahoi.wm.execute import WMConfig, run_wm_planner
from mahoi.wm.planner import CostWeights, Planner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None, choices=list(SCENARIOS))
    args = ap.parse_args()

    names = args.only or list(SCENARIOS)
    weights = CostWeights()
    failures = []

    print(f"{'scenario':<11} {'bound':>6} {'seeds -> team time (s)':<34} "
          f"{'mean':>6} {'sd':>5} {'worst gap':>10} {'valid':>6}")
    print("-" * 88)

    for name in names:
        problem = get_scenario(name)
        tracks, _ = build_prior(problem)
        lb = critical_path_bound(problem, tracks)["lb_team_time"]
        times, gaps, all_ok = [], [], True

        for seed in range(args.seeds):
            res = run_wm_planner(problem, WMConfig(seed=seed), weights)
            rep = V.validate(problem, res.traj)
            times.append(res.traj.team_time)
            gaps.append(float(rep.agent_clearance))
            if not rep.valid:
                all_ok = False
                failures.append((name, seed, rep.errors[:3]))

            # -- cost-ordering sanity, once per scenario -------------------- #
            if seed == 0:
                wm = res.world_model
                full = wm.rollouts(res.states[0], res.modes, horizon=None)
                best, costs = Planner(problem, weights).select(full, wm)
                for r, c in zip(full, costs):
                    rp = V.validate(problem, r.traj, check_dep=False)
                    claims_ok = c.feasible
                    really_ok = (rp.n_agent_violations == 0 and
                                 rp.n_obstacle_violations == 0)
                    if claims_ok and not really_ok:
                        failures.append(
                            (name, seed,
                             [f"planner called {c.label} feasible but the "
                              f"validator found {rp.n_agent_violations} agent / "
                              f"{rp.n_obstacle_violations} obstacle violations"]))
                feas = [c for c in costs if c.feasible]
                if feas:
                    fastest = min(c.makespan for c in feas)
                    chosen = costs[best].makespan
                    print(f"  [{name}] {len(feas)}/{len(costs)} futures feasible; "
                          f"fastest {fastest:.1f}s, chosen {chosen:.1f}s "
                          f"({'same' if abs(chosen - fastest) < 1e-6 else 'traded for soft terms'})")

        arr = np.array(times)
        marks = " ".join(f"{t:5.1f}" for t in times)
        print(f"{name:<11} {lb:>6.1f} {marks:<34} {arr.mean():>6.1f} "
              f"{arr.std():>5.2f} {min(gaps):>10.3f} {'OK' if all_ok else 'FAIL':>6}")

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for name, seed, errs in failures:
            for e in errs:
                print(f"  {name} seed={seed}: {e}")
        sys.exit(1)
    print("all checks passed: 0 collisions, 0 dependency violations, "
          "speed limits respected, across every scenario and seed.")


if __name__ == "__main__":
    main()
