"""The Planner: score every imagined future with an explicit cost, pick one.

No reinforcement learning, no value network, no reward shaping -- the objective
is written down in full below and every term is computed directly from the
rollout's positions.  That is deliberate: the point of the experiment is to see
which *cost* makes a team behave well, and a hand-written cost can be read,
argued about, and ablated one term at a time.

    J(rollout) = W_HARD * (  n_agent_collisions
                           + n_obstacle_collisions
                           + n_dependency_violations
                           + n_agents_that_never_finish
                           + stalled )
               + w_make  * makespan_estimate      [s]
               + w_flow  * flowtime_estimate      [s]
               + w_wait  * speed_given_up         [s]
               + w_dist  * total_travel_distance  [m]
               + w_clear * safety_margin_deficit  [m^2 s]
               + w_turn  * command_roughness      [m^2/s]
               + w_dev   * reference_deviation    [m s]

The first block is the *feasibility* block.  Its weight is enormous but finite,
so the ranking is always defined: when no rollout is feasible -- which happens
transiently in a narrow corridor -- the Planner still returns the least-broken
future instead of raising, and the executor gets a chance to recover on the
next replan.  Among feasible rollouts the block is exactly zero, so the choice
is made purely by the soft terms.

`makespan_estimate` is the rollout's own makespan when the rollout runs to
completion.  Under a finite horizon it is (time simulated) + (critical-path
lower bound on the remainder), supplied by `WorldModel.remaining_estimate`.
Because the remainder term relaxes collisions away it can only *under*-estimate,
which is the right direction for a terminal cost: a truncated future is never
flattered relative to one that actually finished.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geometry import point_rect_dist, seg_seg_min_dist
from ..world import Problem
from .traj import Trajectory
from .worldmodel import Rollout

W_HARD = 1.0e4


@dataclass
class CostWeights:
    make: float = 1.00        # team completion time -- the headline objective
    flow: float = 0.06        # break ties towards finishing individuals early
    wait: float = 0.14        # discourage yielding that buys nothing
    dist: float = 0.03        # discourage long detours
    clear: float = 0.60       # keep a comfortable margin, not just a legal one
    turn: float = 0.02        # discourage oscillation (a livelock precursor)
    dev: float = 0.015        # respect the reference route unless it costs time
    soft_margin: float = 0.25  # m of separation we would *like* to keep


@dataclass
class Cost:
    """Full breakdown for one rollout.  Everything here is reported, not just J."""
    total: float
    feasible: bool
    n_agent_collisions: int = 0
    n_obstacle_collisions: int = 0
    n_dep_violations: int = 0
    n_unfinished: int = 0
    stalled: bool = False
    makespan: float = 0.0
    flow: float = 0.0
    wait: float = 0.0
    dist: float = 0.0
    clear: float = 0.0
    turn: float = 0.0
    dev: float = 0.0
    min_agent_gap: float = np.inf
    min_obst_gap: float = np.inf
    label: str = ""

    def hard_count(self) -> int:
        return (self.n_agent_collisions + self.n_obstacle_collisions +
                self.n_dep_violations + self.n_unfinished + int(self.stalled))

    def row(self) -> Dict[str, object]:
        return {
            "mode": self.label, "J": round(self.total, 3),
            "feasible": self.feasible,
            "makespan": round(self.makespan, 2), "flow": round(self.flow, 2),
            "wait": round(self.wait, 2), "dist": round(self.dist, 2),
            "clear": round(self.clear, 3), "turn": round(self.turn, 3),
            "dev": round(self.dev, 2),
            "coll_a": self.n_agent_collisions,
            "coll_o": self.n_obstacle_collisions,
            "dep": self.n_dep_violations, "unfin": self.n_unfinished,
            "stall": int(self.stalled),
            "min_gap": round(float(self.min_agent_gap), 3),
        }


# --------------------------------------------------------------------------- #
#  Cost terms
# --------------------------------------------------------------------------- #
def _swept_gaps(xy: np.ndarray, i: int, j: int) -> np.ndarray:
    """(T,) minimum distance between agents i and j *during* each timestep.

    Both agents move linearly within a step, so their relative offset is linear
    too and the closest approach is the distance from the origin to a segment --
    the same computation as `geometry.seg_seg_min_dist`, vectorised over time.
    Checking the swept volume rather than the end points is what stops a fast
    crossing from slipping between two samples.
    """
    a = xy[:-1, i, :] - xy[:-1, j, :]                      # (T, 2)
    b = xy[1:, i, :] - xy[1:, j, :]
    d = b - a
    denom = np.einsum("ij,ij->i", d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.clip(-np.einsum("ij,ij->i", a, d) / denom, 0.0, 1.0)
    s = np.where(denom < 1e-12, 0.0, s)
    return np.linalg.norm(a + s[:, None] * d, axis=1)


def _pair_terms(problem: Problem, xy: np.ndarray,
                soft_margin: float) -> Tuple[int, float, float]:
    """(#violating timesteps, worst gap, integrated margin deficit)."""
    n = xy.shape[1]
    n_bad, worst, deficit = 0, np.inf, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            R = problem.min_sep(i, j)
            soft = R + soft_margin
            g = _swept_gaps(xy, i, j)
            d_end = float(np.linalg.norm(xy[-1, i] - xy[-1, j]))
            g_all = np.append(g, d_end)
            worst = min(worst, float(g_all.min()))
            n_bad += int(np.count_nonzero(g_all < R - 1e-3))
            short = np.clip(soft - g, 0.0, None)
            deficit += float(short @ short)
    return n_bad, float(worst), deficit * problem.dt


def _obstacle_terms(problem: Problem, xy: np.ndarray) -> Tuple[int, float]:
    n_bad, worst = 0, np.inf
    W, H = problem.world.width, problem.world.height
    for i, a in enumerate(problem.agents):
        pts = xy[:, i, :]
        clr = np.full(len(pts), np.inf)
        for r in problem.world.obstacles:
            clr = np.minimum(clr, point_rect_dist(pts, r))
        clr = np.minimum.reduce([clr, pts[:, 0], pts[:, 1],
                                 W - pts[:, 0], H - pts[:, 1]])
        worst = min(worst, float(clr.min()))
        n_bad += int(np.count_nonzero(clr < a.radius - 1e-3))
    return n_bad, float(worst)


def _dep_violations(problem: Problem, traj: Trajectory,
                    only_settled: bool = True) -> int:
    """Dependency pairs whose ordering is already broken inside the window.

    A pair is only judged once the successor has actually reached its waypoint;
    a rollout that simply has not got there yet is not a violation, it is
    incomplete, and `n_unfinished` already accounts for that.
    """
    gap_steps = int(round(problem.gap / problem.dt))
    bad = 0
    for (i, j) in problem.deps:
        ws_j = int(traj.wp_in[j])
        we_i = int(traj.wp_out[i])
        if ws_j < 0:
            continue                        # successor has not started its HOI
        if we_i < 0 or ws_j < we_i + gap_steps:
            bad += 1
        elif not only_settled and we_i < 0:
            bad += 1
    return bad


def _deviation(traj: Trajectory, roll: Rollout, world_model) -> float:
    """Integrated distance from each agent to its assigned reference route."""
    routes = [world_model.library[i][min(roll.mode.routes[i],
                                         len(world_model.library[i]) - 1)]
              for i in range(traj.n)]
    total = 0.0
    stride = max(1, traj.T // 60)           # subsample: this is a soft term
    for i in range(traj.n):
        pts = np.vstack([routes[i].leg1.pts, routes[i].leg2.pts])
        p = traj.pos[::stride, i, :]
        d = np.linalg.norm(p[:, None, :] - pts[None, :, :], axis=2).min(axis=1)
        total += float(d.sum()) * stride * traj.dt
    return total


# --------------------------------------------------------------------------- #
#  Planner
# --------------------------------------------------------------------------- #
class Planner:
    def __init__(self, problem: Problem, weights: Optional[CostWeights] = None):
        self.problem = problem
        self.w = weights or CostWeights()

    # -- scoring ----------------------------------------------------------- #
    def evaluate(self, roll: Rollout, world_model) -> Cost:
        p, w = self.problem, self.w
        traj = roll.traj
        xy = traj.pos

        n_pair, min_gap, clear = _pair_terms(p, xy, w.soft_margin)
        n_obst, min_obst = _obstacle_terms(p, xy)
        n_dep = _dep_violations(p, traj)
        # An agent that is merely still on the road when the imagination horizon
        # expires is NOT a constraint violation -- the terminal critical-path
        # term already prices the rest of its journey.  Only a rollout that ran
        # to its natural end (or livelocked) without everybody arriving counts.
        n_unfin = 0 if roll.hit_horizon else \
            int(np.count_nonzero(~roll.reached_goal))

        # -- completion time: simulated part + critical-path remainder ------- #
        elapsed = traj.T * p.dt
        rem = roll.remaining
        finish = np.asarray(rem["finish"], float)
        done_at = np.where(traj.done >= 0, traj.done * p.dt, elapsed + finish)
        makespan = float(done_at.max())
        flow = float(done_at.sum())

        dv = np.diff(traj.vel[:-1], axis=0) if traj.T > 1 else np.zeros((1, p.n, 2))
        turn = float(np.sum(np.linalg.norm(dv, axis=2) ** 2)) * p.dt
        dev = _deviation(traj, roll, world_model)

        hard = (n_pair + n_obst + n_dep + n_unfin + int(roll.stalled))
        total = (W_HARD * hard
                 + w.make * makespan + w.flow * flow
                 + w.wait * traj.total_wait + w.dist * traj.travel_distance
                 + w.clear * clear + w.turn * turn + w.dev * dev)

        return Cost(total=total, feasible=(hard == 0),
                    n_agent_collisions=n_pair, n_obstacle_collisions=n_obst,
                    n_dep_violations=n_dep, n_unfinished=n_unfin,
                    stalled=roll.stalled, makespan=makespan, flow=flow,
                    wait=traj.total_wait, dist=traj.travel_distance,
                    clear=clear, turn=turn, dev=dev,
                    min_agent_gap=min_gap, min_obst_gap=min_obst,
                    label=roll.mode.label(p))

    # -- selection --------------------------------------------------------- #
    def select(self, rolls: Sequence[Rollout], world_model
               ) -> Tuple[int, List[Cost]]:
        """Index of the best future, plus the full cost table (for reporting)."""
        costs = [self.evaluate(r, world_model) for r in rolls]
        best = int(np.argmin([c.total for c in costs]))
        return best, costs


def cost_table(costs: Sequence[Cost], top: Optional[int] = None) -> str:
    """Human-readable ranking, cheapest first."""
    order = np.argsort([c.total for c in costs])
    if top:
        order = order[:top]
    head = (f"{'rank':>4} {'mode':<18} {'J':>10} {'mkspn':>7} {'flow':>7} "
            f"{'wait':>6} {'dist':>7} {'clr':>6} {'dev':>6} "
            f"{'cA':>3} {'cO':>3} {'dep':>3} {'unf':>3} {'stl':>3} {'gap':>6}")
    lines = [head, "-" * len(head)]
    for rank, k in enumerate(order):
        c = costs[int(k)]
        lines.append(
            f"{rank:>4} {c.label:<18} {c.total:>10.2f} {c.makespan:>7.2f} "
            f"{c.flow:>7.2f} {c.wait:>6.2f} {c.dist:>7.2f} {c.clear:>6.2f} "
            f"{c.dev:>6.2f} {c.n_agent_collisions:>3} {c.n_obstacle_collisions:>3} "
            f"{c.n_dep_violations:>3} {c.n_unfinished:>3} {int(c.stalled):>3} "
            f"{c.min_agent_gap:>6.2f}")
    return "\n".join(lines)
