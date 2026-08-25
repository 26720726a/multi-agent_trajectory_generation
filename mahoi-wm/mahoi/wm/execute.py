"""Executor: run the World Model + Planner loop and produce the real trajectory.

    t = 0
    build the rollout library (routes x yield orders x speed/side variants)
    repeat until everybody is home:
        World Model : roll every live mode forward from the *actual* state
        Planner     : score them, pick the cheapest future
        execute     : commit only the first `replan` seconds of that future
        (the committed part is a re-simulation of the same deterministic
         controller, so what is executed is exactly the head of the future
         that was chosen -- no plan/execution mismatch is swept under a rug)

Why re-selecting mid-flight can change anything
-----------------------------------------------
The rollouts are deterministic, so if the World Model always imagined the whole
future the choice made at t = 0 would never be improved on -- replanning would
be theatre.  It is not, for two reasons, and both are switched on by default:

* **Bounded imagination.**  With `horizon_s` set, each rollout stops after a few
  seconds and the Planner adds a critical-path *lower bound* for the remainder.
  The ranking under that truncated view is not the ranking under the full view,
  so as the horizon slides forward the Planner genuinely learns something and
  can change its mind.  This is ordinary MPC.
* **The state moves off the library.**  Reactive avoidance pushes agents off
  their reference routes, so at replan time the same mode label denotes a
  slightly different future than it did at t = 0.

`switching=False` collapses the loop to "imagine everything once, commit to it"
-- the ablation that isolates how much the mid-flight re-selection is worth.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..world import Problem
from .controller import PHASE_DONE, TeamState, initial_state
from .planner import Cost, CostWeights, Planner, cost_table
from .traj import Trajectory, concat
from .worldmodel import PlanMode, Rollout, WorldModel


@dataclass
class WMConfig:
    horizon_s: Optional[float] = 4.0   # imagination depth; None = to completion
    replan_s: float = 1.0              # how much of the chosen future we commit
    max_modes: int = 16                # rollouts generated at t = 0
    keep_modes: int = 8                # live pool after the first selection
    k_routes: int = 2                  # geometric route candidates per leg
    switching: bool = True             # re-select mid-flight?
    seed: int = 0
    max_sim_s: float = 90.0            # simulated-time cap (safety net)
    time_budget_s: float = 240.0       # wall-clock cap
    stall_patience: int = 60           # executed steps without progress -> give up
    switch_margin_rel: float = 0.01    # hysteresis: relative gain needed to switch
    switch_margin_abs: float = 0.05    # hysteresis: absolute gain needed to switch


@dataclass
class WMResult:
    traj: Trajectory
    switches: List[Dict[str, object]] = field(default_factory=list)
    first_table: List[Cost] = field(default_factory=list)
    first_rollouts: List[Rollout] = field(default_factory=list)
    modes: List[PlanMode] = field(default_factory=list)
    states: List[TeamState] = field(default_factory=list)
    live_modes: List[PlanMode] = field(default_factory=list)
    world_model: Optional[WorldModel] = None
    runtime: float = 0.0
    note: str = ""

    @property
    def n_switches(self) -> int:
        return sum(1 for s in self.switches if s["switched"])


def run_wm_planner(problem: Problem, cfg: Optional[WMConfig] = None,
                   weights: Optional[CostWeights] = None,
                   verbose: bool = False) -> WMResult:
    cfg = cfg or WMConfig()
    t0 = time.perf_counter()
    dt = problem.dt

    wm = WorldModel(problem, k_routes=cfg.k_routes, seed=cfg.seed)
    planner = Planner(problem, weights)
    modes = wm.sample_modes(max_modes=cfg.max_modes)

    horizon = None if cfg.horizon_s is None else max(1, int(round(cfg.horizon_s / dt)))
    exec_steps = max(1, int(round(cfg.replan_s / dt)))
    max_steps = int(round(cfg.max_sim_s / dt))

    st: TeamState = initial_state(problem)
    live: List[PlanMode] = list(modes)
    segments: List[Trajectory] = []
    switches: List[Dict[str, object]] = []
    first_table: List[Cost] = []
    first_rolls: List[Rollout] = []
    prev_mode: Optional[PlanMode] = None
    best_togo, since_progress = np.inf, 0
    note = ""

    states: List[TeamState] = []
    while not st.all_done and st.t < max_steps:
        states.append(st.copy())
        if time.perf_counter() - t0 > cfg.time_budget_s:
            note = "wall-clock budget exhausted"
            break

        rolls = wm.rollouts(st, live, horizon=horizon)
        k, costs = planner.select(rolls, wm)

        # Hysteresis: near-ties between modes are common (many of them share
        # their first few seconds), and flipping between them every replan is
        # churn, not coordination.  Stay put unless the challenger is clearly
        # better -- this keeps the switch log meaningful as a *record of
        # decisions that mattered*.
        if prev_mode is not None and prev_mode in live:
            kp = live.index(prev_mode)
            margin = cfg.switch_margin_rel * abs(costs[kp].total) + \
                cfg.switch_margin_abs
            if costs[k].total > costs[kp].total - margin:
                k = kp
        chosen = live[k]

        if not first_table:
            first_table, first_rolls = costs, rolls
            if cfg.keep_modes and len(live) > cfg.keep_modes:
                order = np.argsort([c.total for c in costs])[:cfg.keep_modes]
                live = [live[int(i)] for i in sorted(order)]

        switches.append({
            "t": st.t * dt, "mode": chosen.label(problem),
            "switched": prev_mode is not None and chosen != prev_mode,
            "J": costs[k].total, "feasible": costs[k].feasible,
            "makespan_est": costs[k].makespan,
        })
        if verbose:
            flag = " *" if switches[-1]["switched"] else "  "
            print(f"  t={st.t * dt:5.1f}s{flag} -> {chosen.label(problem):<18} "
                  f"J={costs[k].total:9.2f}  est_makespan={costs[k].makespan:5.2f}s"
                  f"{'' if costs[k].feasible else '   [infeasible view]'}")
        prev_mode = chosen

        if not cfg.switching:
            full = wm.rollout(st, chosen, horizon=None, max_steps=max_steps)
            segments.append(full.traj)
            st = full.end_state
            break

        seg = wm.rollout(st, chosen, horizon=exec_steps, max_steps=max_steps)
        if seg.traj.T == 0:
            note = "controller could not advance"
            break
        segments.append(seg.traj)
        st = seg.end_state

        togo = wm._to_go([wm.library[i][min(chosen.routes[i],
                                            len(wm.library[i]) - 1)]
                          for i in range(problem.n)], st)
        if togo < best_togo - 0.02:
            best_togo, since_progress = togo, 0
        else:
            since_progress += exec_steps
            if since_progress >= cfg.stall_patience:
                note = "livelock: no team progress"
                break

    if not segments:
        raise RuntimeError("executor produced no motion")

    traj = concat(segments, method="world model + planner")
    traj.extra["v_max"] = [a.v_max for a in problem.agents]
    traj.runtime = time.perf_counter() - t0
    traj.feasible = bool(np.all(st.phase == PHASE_DONE))
    traj.note = note or (f"{len(switches)} decisions, "
                         f"{sum(1 for s in switches if s['switched'])} switches")

    return WMResult(traj=traj, switches=switches, first_table=first_table,
                    first_rollouts=first_rolls, modes=modes, states=states,
                    live_modes=live, world_model=wm,
                    runtime=traj.runtime, note=note)


# --------------------------------------------------------------------------- #
#  Named configurations used by the experiments
# --------------------------------------------------------------------------- #
def cfg_receding(**kw) -> WMConfig:
    """Default: bounded imagination + mid-flight re-selection."""
    return WMConfig(horizon_s=4.0, replan_s=1.0, switching=True, **kw)


def cfg_reselect_full(**kw) -> WMConfig:
    """Ablation: imagination runs to completion, but still re-selects."""
    return WMConfig(horizon_s=None, replan_s=1.0, switching=True, **kw)


def cfg_static(**kw) -> WMConfig:
    """Ablation: imagine every future once at t = 0, then commit to one."""
    return WMConfig(horizon_s=None, replan_s=1.0, switching=False, **kw)
