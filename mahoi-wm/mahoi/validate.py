"""Independent checker.

Deliberately written against the *raw executed positions* only -- it never
looks at planner internals -- so it can catch bugs in the planners rather than
repeating their assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .coordination import Solution
from .geometry import point_rect_dist, seg_seg_min_dist
from .world import Problem

TOL = 1e-6


@dataclass
class Report:
    valid: bool
    errors: List[str] = field(default_factory=list)
    obstacle_clearance: float = np.inf     # min over time/agents (m)
    agent_clearance: float = np.inf        # min pairwise distance (m)
    n_obstacle_violations: int = 0
    n_agent_violations: int = 0
    n_dep_violations: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        head = "OK " if self.valid else "FAIL"
        return (f"[{head}] obst_clr={self.obstacle_clearance:.3f}m  "
                f"agent_clr={self.agent_clearance:.3f}m  "
                f"dep_viol={self.n_dep_violations}")


def _densify(xy: np.ndarray, sub: int = 4) -> np.ndarray:
    """(T+1, n, 2) -> (sub*T+1, n, 2) by linear interpolation between timesteps."""
    T = xy.shape[0] - 1
    if T == 0:
        return xy.copy()
    w = np.linspace(0.0, 1.0, sub, endpoint=False)[:, None, None]
    seg = (1 - w)[None] * xy[:-1, None] + w[None] * xy[1:, None]
    out = seg.reshape(-1, xy.shape[1], 2)
    return np.concatenate([out, xy[-1:]], axis=0)


def validate(problem: Problem, sol: Solution,
             check_dep: bool = True) -> Report:
    rep = Report(valid=True)
    if sol.K.shape[0] < 2:
        rep.valid = False
        rep.errors.append("empty plan")
        return rep

    xy = sol.xy                       # (T+1, n, 2)
    T, n = sol.T, sol.n
    dense = _densify(xy)

    # ---- 1. endpoints -----------------------------------------------------
    for i, a in enumerate(problem.agents):
        if np.linalg.norm(xy[0, i] - np.asarray(a.start)) > 0.25:
            rep.errors.append(f"{a.name}: does not begin at Start")
        if np.linalg.norm(xy[-1, i] - np.asarray(a.goal)) > 0.25:
            rep.errors.append(f"{a.name}: does not end at Goal")

    # ---- 2. speed limit ---------------------------------------------------
    for i, a in enumerate(problem.agents):
        step = np.linalg.norm(np.diff(xy[:, i], axis=0), axis=1)
        if step.size and step.max() > a.v_max * problem.dt + 1e-6:
            rep.errors.append(f"{a.name}: speed limit exceeded "
                              f"({step.max() / problem.dt:.2f} > {a.v_max} m/s)")

    # ---- 3. static obstacles + room bounds --------------------------------
    for i, a in enumerate(problem.agents):
        pts = dense[:, i, :]
        clr = np.inf
        for r in problem.world.obstacles:
            clr = min(clr, float(point_rect_dist(pts, r).min()))
        wall = np.minimum.reduce([
            pts[:, 0], pts[:, 1],
            problem.world.width - pts[:, 0], problem.world.height - pts[:, 1]])
        clr = min(clr, float(wall.min()))
        rep.obstacle_clearance = min(rep.obstacle_clearance, clr)
        if clr < a.radius - 1e-3:
            rep.n_obstacle_violations += 1
            rep.errors.append(f"{a.name}: obstacle/wall clearance "
                              f"{clr:.3f} < radius {a.radius:.3f}")

    # ---- 4. agent-agent collisions (exact, continuous in time) ------------
    for i in range(n):
        for j in range(i + 1, n):
            need = problem.min_sep(i, j)
            worst = np.inf
            for t in range(T):
                d = seg_seg_min_dist(xy[t, i], xy[t + 1, i], xy[t, j], xy[t + 1, j])
                worst = min(worst, d)
            worst = min(worst, float(np.linalg.norm(xy[-1, i] - xy[-1, j])))
            rep.agent_clearance = min(rep.agent_clearance, worst)
            if worst < need - 1e-3:
                rep.n_agent_violations += 1
                rep.errors.append(
                    f"{problem.agents[i].name}-{problem.agents[j].name}: "
                    f"min distance {worst:.3f} < required {need:.3f}")

    # ---- 5. waypoint actually visited, dwell respected --------------------
    ws, we = sol.event_steps()
    for i, a in enumerate(problem.agents):
        wp = np.asarray(a.waypoint)
        d = np.linalg.norm(xy[:, i] - wp[None, :], axis=1)
        near = d <= 0.20
        if not near.any():
            rep.errors.append(f"{a.name}: never reaches its Waypoint")
            continue
        need = int(round(a.dwell / problem.dt))
        seg = near[ws[i]:we[i] + 1]
        if len(seg) < need + 1 or not seg.all():
            rep.errors.append(f"{a.name}: HOI dwell not held at the Waypoint")

    # ---- 6. dependencies --------------------------------------------------
    if check_dep:
        gap_steps = int(round(problem.gap / problem.dt))
        for (i, j) in problem.deps:
            if ws[j] < we[i] + gap_steps:
                rep.n_dep_violations += 1
                rep.errors.append(
                    f"dependency {problem.agents[i].name}->{problem.agents[j].name}: "
                    f"{problem.agents[j].name} starts its HOI at "
                    f"t={ws[j] * problem.dt:.1f}s but "
                    f"{problem.agents[i].name} only finishes at "
                    f"t={we[i] * problem.dt:.1f}s")

    rep.valid = len(rep.errors) == 0
    rep.metrics = {
        "team_time": sol.team_time,
        "flow_time": sol.flow_time,
        "travel_distance": sol.travel_distance,
        "total_wait": sol.total_wait,
        "runtime": sol.runtime,
    }
    return rep
