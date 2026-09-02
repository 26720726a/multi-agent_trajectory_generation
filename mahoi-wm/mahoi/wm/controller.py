"""Reactive per-step controller -- the *dynamics* the World Model rolls out.

This is where method B lives.  Every agent, at every timestep, reads a
synchronous snapshot of the shared board (neighbour positions + velocities +
"predecessor finished its HOI" flags), forms a preferred velocity by pure
pursuit along its assigned reference route, and then corrects that velocity with
a sampling-based **generalized reciprocal velocity obstacle**.  The route is a
*guide*, not a rail: the agent leaves it whenever avoiding a neighbour or an
obstacle requires it, and pure pursuit pulls it back afterwards.

Two things differ from a plain RVO simulator, and both are what make the World
Model able to produce genuinely *different* futures from the same state:

* **Asymmetric responsibility.**  Classic RVO splits avoidance 50/50, which is
  what makes it deterministic and therefore single-future.  Here the share
  `alpha_ij` comes from a *yield order* supplied by the caller, so "A goes
  first, B swings wide" and "B goes first, A swings wide" are two different
  rollouts of the same scene, differing in both path and timing.
* **Structural safety.**  After every agent has picked a velocity, a reciprocal
  projection pass shrinks the velocities of any pair whose swept segments would
  come closer than `r_i + r_j + safety`.  Because standing still is always safe
  from a safe configuration, this pass always terminates and agent-agent
  collisions become impossible rather than merely unlikely.  The price of the
  projection shows up as lost speed, which the Planner's cost already charges
  for -- so safety never silently competes with the objective.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..geometry import Rect, seg_hits_rect, seg_seg_min_dist
from ..world import Problem

# --------------------------------------------------------------------------- #
#  Tunables (all explicit -- no learned parameters anywhere in this file)
# --------------------------------------------------------------------------- #
LOOKAHEAD = 0.85          # m, pure-pursuit lookahead along the reference route
TAU = 2.6                 # s, velocity-obstacle time horizon
TAU_OBST = 1.1            # s, shorter horizon for static geometry
W_TTC = 1.35              # weight on 1/time-to-collision (neighbours)
W_TTC_OBST = 0.55         # weight on 1/time-to-collision (obstacles)
W_TURN = 0.22             # anti-oscillation: penalise changing the command
W_SIDE = 0.16             # strength of the mode's left/right passing bias
OBST_PAD = 0.02           # extra clearance kept from inflated obstacles (m)
SOFT_MARGIN = 0.22        # m, preferred extra separation beyond the hard bound
SNAP_TOL = 0.02           # m, distance at which we snap onto W or G
DEP_STANDOFF = 0.45       # m, where a blocked successor holds before its W

SPEED_LEVELS = (0.0, 0.35, 0.70, 1.0)
ANGLE_LEVELS = np.deg2rad([0, 12, -12, 25, -25, 42, -42, 62, -62,
                           85, -85, 115, -115, 150, -150, 180])


# --------------------------------------------------------------------------- #
#  Reference route ("guide")
# --------------------------------------------------------------------------- #
class Guide:
    """A dense polyline with arclength, supporting windowed projection.

    Used for the two legs of an agent's route (Start->Waypoint, Waypoint->Goal).
    """

    def __init__(self, poly: np.ndarray, ds: float = 0.05):
        poly = np.asarray(poly, float)
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        m = max(2, int(np.ceil(total / ds)) + 1)
        s = np.linspace(0.0, total, m)
        self.pts = np.stack([np.interp(s, cum, poly[:, 0]),
                             np.interp(s, cum, poly[:, 1])], axis=1)
        self.s = s
        self.length = total

    def project(self, p: np.ndarray, hint: int = 0, window: int = 48) -> int:
        """Index of the closest sample, searched in a window around `hint`.

        The window keeps the search O(1) inside a rollout and, more importantly,
        stops an agent that has been pushed sideways from "snapping" onto a
        later part of its own route and skipping ground it never covered.
        """
        lo = max(0, hint - 6)
        hi = min(len(self.pts), hint + window)
        d = np.linalg.norm(self.pts[lo:hi] - p[None, :], axis=1)
        return int(lo + np.argmin(d))

    def project_full(self, p: np.ndarray) -> int:
        """Closest sample over the whole polyline.

        Used when re-anchoring a state onto a *different* route than the one it
        was produced on -- the World Model does exactly that every time it
        rolls out an alternative mode from the current position.
        """
        return int(np.argmin(np.linalg.norm(self.pts - p[None, :], axis=1)))

    def clamp(self, idx: int) -> int:
        return int(min(max(idx, 0), len(self.pts) - 1))

    def lookahead(self, idx: int, dist: float) -> np.ndarray:
        target_s = self.s[self.clamp(idx)] + dist
        j = int(np.searchsorted(self.s, target_s))
        return self.pts[min(j, len(self.pts) - 1)]

    def remaining(self, idx: int) -> float:
        return float(self.length - self.s[self.clamp(idx)])


@dataclass
class Route:
    """One agent's Start -> Waypoint -> Goal guide pair."""
    leg1: Guide
    leg2: Guide
    label: str = ""

    @property
    def length(self) -> float:
        return self.leg1.length + self.leg2.length


# --------------------------------------------------------------------------- #
#  Rollout state
# --------------------------------------------------------------------------- #
PHASE_TO_WP, PHASE_DWELL, PHASE_TO_GOAL, PHASE_DONE = 0, 1, 2, 3


@dataclass
class TeamState:
    """Everything the rollout needs; cheap to copy so branches stay independent."""
    pos: np.ndarray                  # (n, 2)
    vel: np.ndarray                  # (n, 2)
    phase: np.ndarray                # (n,) int
    dwell_left: np.ndarray           # (n,) int, timesteps of HOI remaining
    hint: np.ndarray                 # (n,) int, projection hint per agent
    t: int = 0
    wp_in: np.ndarray = None         # (n,) int, -1 = not yet
    wp_out: np.ndarray = None
    done: np.ndarray = None

    def copy(self) -> "TeamState":
        return TeamState(pos=self.pos.copy(), vel=self.vel.copy(),
                         phase=self.phase.copy(), dwell_left=self.dwell_left.copy(),
                         hint=self.hint.copy(), t=self.t,
                         wp_in=self.wp_in.copy(), wp_out=self.wp_out.copy(),
                         done=self.done.copy())

    @property
    def all_done(self) -> bool:
        return bool(np.all(self.phase == PHASE_DONE))


def initial_state(problem: Problem) -> TeamState:
    n = problem.n
    return TeamState(
        pos=np.array([a.start for a in problem.agents], float),
        vel=np.zeros((n, 2)),
        phase=np.zeros(n, int),
        dwell_left=np.array([int(round(a.dwell / problem.dt))
                             for a in problem.agents], int),
        hint=np.zeros(n, int),
        t=0,
        wp_in=np.full(n, -1, int), wp_out=np.full(n, -1, int),
        done=np.full(n, -1, int),
    )


# --------------------------------------------------------------------------- #
#  Scene geometry cache
# --------------------------------------------------------------------------- #
class SceneCache:
    """Pre-inflated obstacles and walls, per agent radius.

    The per-agent inflated rectangles are also stored as one `(R, 4)` array so
    the swept-segment test can be run for every candidate velocity at once --
    that test is the innermost loop of the whole experiment and doing it one
    Python call at a time dominated the runtime.
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        self.rects = list(problem.world.obstacles)
        self.blocked: List[List[Rect]] = []
        self.blocked_arr: List[np.ndarray] = []
        for a in problem.agents:
            pad = a.radius + OBST_PAD
            infl = [r.inflate(pad) for r in self.rects]
            self.blocked.append(infl)
            self.blocked_arr.append(
                np.array([[r.x0, r.y0, r.x1, r.y1] for r in infl], float)
                if infl else np.zeros((0, 4)))
        self.W = problem.world.width
        self.H = problem.world.height

    def nearest_points(self, p: np.ndarray) -> np.ndarray:
        """Closest point of every obstacle (and of the four walls) to `p`."""
        out = np.empty((len(self.rects) + 4, 2))
        for k, r in enumerate(self.rects):
            out[k] = (np.clip(p[0], r.x0, r.x1), np.clip(p[1], r.y0, r.y1))
        m = len(self.rects)
        out[m + 0] = (0.0, p[1])
        out[m + 1] = (self.W, p[1])
        out[m + 2] = (p[0], 0.0)
        out[m + 3] = (p[0], self.H)
        return out

    def free_step(self, i: int, p: np.ndarray, q: np.ndarray) -> bool:
        """True if agent `i` may move along the segment p->q."""
        return bool(self.free_steps(i, p, np.atleast_2d(q))[0])

    def free_steps(self, i: int, p: np.ndarray, qs: np.ndarray) -> np.ndarray:
        """(C,) bool: which of the candidate end points agent `i` may move to.

        Vectorised Liang-Barsky clipping of the segment `p -> qs[c]` against
        every inflated obstacle, plus an end-point containment test and the
        room bounds.  A candidate survives only if the whole swept segment
        stays clear, which is what the independent validator later re-checks on
        a 4x-densified trajectory.
        """
        pad = self.problem.agents[i].radius + OBST_PAD
        ok = ((qs[:, 0] >= pad) & (qs[:, 0] <= self.W - pad) &
              (qs[:, 1] >= pad) & (qs[:, 1] <= self.H - pad))
        R = self.blocked_arr[i]
        if len(R) == 0 or not np.any(ok):
            return ok

        # end point strictly inside an inflated rectangle -> reject
        inside = ((qs[:, None, 0] > R[None, :, 0]) & (qs[:, None, 0] < R[None, :, 2]) &
                  (qs[:, None, 1] > R[None, :, 1]) & (qs[:, None, 1] < R[None, :, 3]))
        ok &= ~inside.any(axis=1)

        eps = 1e-7
        x0, y0 = R[:, 0] + eps, R[:, 1] + eps
        x1, y1 = R[:, 2] - eps, R[:, 3] - eps
        proper = (x0 < x1) & (y0 < y1)                       # (R,)
        d = qs - p[None, :]                                  # (C, 2)
        t0 = np.zeros((len(qs), len(R)))
        t1 = np.ones((len(qs), len(R)))
        alive = np.ones((len(qs), len(R)), bool)
        for num_r, den_c in ((p[0] - x0, -d[:, 0]), (x1 - p[0], d[:, 0]),
                             (p[1] - y0, -d[:, 1]), (y1 - p[1], d[:, 1])):
            num = num_r[None, :]                             # (1, R)
            den = den_c[:, None]                             # (C, 1)
            zero = np.abs(den) < 1e-15
            alive &= ~(zero & (num < 0.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                t = num / den
            t = np.where(np.isfinite(t), t, 0.0)
            neg = (den < 0.0) & ~zero
            pos = (den > 0.0) & ~zero
            t0 = np.where(neg, np.maximum(t0, t), t0)
            t1 = np.where(pos, np.minimum(t1, t), t1)
        hit = alive & (t0 < t1) & proper[None, :]
        return ok & ~hit.any(axis=1)


# --------------------------------------------------------------------------- #
#  Candidate velocities
# --------------------------------------------------------------------------- #
def ctrl_snap(dt: float, v_lim: float) -> float:
    """How close to W / G counts as "close enough to finish the approach".

    Kept below one timestep of travel so substituting the exact-arrival velocity
    can never produce a step longer than `v_lim * dt`.
    """
    return min(SNAP_TOL, 0.45 * v_lim * dt)


def _candidates(v_pref: np.ndarray, v_max: float) -> np.ndarray:
    """Polar fan of candidate velocities around the preferred direction."""
    nrm = float(np.linalg.norm(v_pref))
    if nrm < 1e-9:
        base = np.array([1.0, 0.0])
    else:
        base = v_pref / nrm
    ca, sa = np.cos(ANGLE_LEVELS), np.sin(ANGLE_LEVELS)
    dirs = np.stack([base[0] * ca - base[1] * sa,
                     base[0] * sa + base[1] * ca], axis=1)          # (A, 2)
    sp = np.asarray(SPEED_LEVELS) * v_max                            # (S,)
    cand = (dirs[None, :, :] * sp[:, None, None]).reshape(-1, 2)     # (S*A, 2)
    cand = np.vstack([v_pref[None, :], cand])
    return cand


def _ttc(w: np.ndarray, u: np.ndarray, R: float) -> np.ndarray:
    """Time-to-collision for relative offset `w` (2,) and velocities `u` (C,2).

    Returns +inf where the candidate never brings the pair within `R`.
    `w` points from the ego agent towards the obstacle/neighbour, so the pair
    closes when the relative velocity `u` points along `w`.
    """
    ww = float(w @ w)
    c = ww - R * R
    a = np.einsum("ij,ij->i", u, u)
    b = u @ w
    if c < 0.0:                       # already overlapping -> immediate
        return np.zeros(len(u))
    disc = b * b - a * c
    out = np.full(len(u), np.inf)
    ok = (disc > 0.0) & (a > 1e-12) & (b > 0.0)
    if np.any(ok):
        t = (b[ok] - np.sqrt(disc[ok])) / a[ok]
        t = np.where(t < 0.0, np.inf, t)
        out[ok] = t
    return out


# --------------------------------------------------------------------------- #
#  One synchronous timestep of the whole team
# --------------------------------------------------------------------------- #
def step(problem: Problem, scene: SceneCache, routes: Sequence[Route],
         st: TeamState, alpha: np.ndarray, speed_scale: np.ndarray,
         side_bias: np.ndarray) -> TeamState:
    """Advance the team by one `dt`.

    Parameters
    ----------
    alpha       : (n, n) responsibility share.  `alpha[i, j]` in (0, 1] is how
                  much of the i-j avoidance agent `i` takes on.  0.5/0.5 is
                  classic RVO; larger means "I get out of the way".
    speed_scale : (n,) multiplier on each agent's preferred speed.
    side_bias   : (n,) in {-1, 0, +1}; which side the agent prefers to pass on.
    """
    n, dt = problem.n, problem.dt
    nxt = st.copy()
    nxt.t = st.t + 1

    # ---- board snapshot: everybody reads the *same* state (D7) ------------- #
    pos, vel, phase = st.pos, st.vel, st.phase
    pred_done = np.ones(n, bool)
    for (i, j) in problem.deps:
        if st.wp_out[i] < 0:
            pred_done[j] = False

    v_cmd = np.zeros((n, 2))

    for i in range(n):
        a = problem.agents[i]
        if phase[i] in (PHASE_DWELL, PHASE_DONE):
            continue

        # -- preferred velocity: pure pursuit along the guide (D4) ---------- #
        leg = routes[i].leg1 if phase[i] == PHASE_TO_WP else routes[i].leg2
        target_pt = np.asarray(a.waypoint if phase[i] == PHASE_TO_WP else a.goal,
                               float)
        idx = leg.project(pos[i], hint=int(st.hint[i]))
        nxt.hint[i] = idx
        v_lim = a.v_max * float(speed_scale[i])

        d_target = float(np.linalg.norm(target_pt - pos[i]))
        if d_target <= v_lim * dt + SNAP_TOL or leg.remaining(idx) <= LOOKAHEAD:
            aim = target_pt                       # home in exactly on W / G
        else:
            aim = leg.lookahead(idx, LOOKAHEAD)
        d_aim = np.linalg.norm(aim - pos[i])
        if d_aim < 1e-9:
            v_pref = np.zeros(2)
        else:
            v_pref = (aim - pos[i]) / d_aim * min(v_lim, d_target / dt)

        # -- dependency event-hold (D5): stop short of my own waypoint ------ #
        blocked = (phase[i] == PHASE_TO_WP) and not pred_done[i]
        if blocked:
            slack = max(0.0, d_target - DEP_STANDOFF)
            if slack < v_lim * dt:
                v_pref = v_pref / max(np.linalg.norm(v_pref), 1e-9) * (slack / dt) \
                    if slack > 1e-9 else np.zeros(2)

        cand = _candidates(v_pref, v_lim)
        step_pts = pos[i][None, :] + cand * dt

        # -- hard filter: static geometry + dependency stand-off ------------ #
        keep = scene.free_steps(i, pos[i], step_pts)
        if blocked:
            keep &= np.linalg.norm(step_pts - target_pt[None, :], axis=1) >= \
                DEP_STANDOFF - 1e-9
        if not np.any(keep):
            v_cmd[i] = 0.0
            continue
        cand = cand[keep]
        step_pts = step_pts[keep]

        # -- soft cost: preference + reciprocal VO + obstacles + smoothness -- #
        cost = np.linalg.norm(cand - v_pref[None, :], axis=1)
        cost += W_TURN * np.linalg.norm(cand - vel[i][None, :], axis=1)

        for j in range(n):
            if j == i:
                continue
            R = problem.min_sep(i, j) + SOFT_MARGIN
            w = pos[j] - pos[i]
            if float(w @ w) > (R + (a.v_max + problem.agents[j].v_max) * TAU) ** 2:
                continue                          # far away: cannot interact
            aij = float(alpha[i, j])
            if phase[j] == PHASE_DONE:
                aij = 1.0                         # a parked agent will not move
            u = vel[i][None, :] + (cand - vel[i][None, :]) / aij - vel[j][None, :]
            t_c = _ttc(w, u, R)
            hit = t_c < TAU
            cost[hit] += W_TTC / np.maximum(t_c[hit], dt)

        near = scene.nearest_points(pos[i])
        R_o = a.radius + OBST_PAD + 0.5 * SOFT_MARGIN
        for k in range(len(near)):
            w = near[k] - pos[i]
            if float(w @ w) > (R_o + a.v_max * TAU_OBST) ** 2:
                continue
            t_c = _ttc(w, cand, R_o)
            hit = t_c < TAU_OBST
            cost[hit] += W_TTC_OBST / np.maximum(t_c[hit], dt)

        if side_bias[i] != 0 and np.linalg.norm(v_pref) > 1e-6:
            d0 = v_pref / np.linalg.norm(v_pref)
            cross = d0[0] * cand[:, 1] - d0[1] * cand[:, 0]
            cost -= W_SIDE * float(side_bias[i]) * cross / max(v_lim, 1e-9)

        v_cmd[i] = cand[int(np.argmin(cost))]

        # -- exact arrival: land *on* W / G rather than near it -------------- #
        # Snapping after integration would teleport the agent by up to SNAP_TOL
        # and show up as a speed-limit violation, so instead we substitute the
        # exact-arrival velocity here and let the safety pass veto it if unsafe.
        if d_target <= v_lim * dt + 1e-12 and \
                np.linalg.norm(pos[i] + v_cmd[i] * dt - target_pt) < ctrl_snap(dt, v_lim):
            v_cmd[i] = (target_pt - pos[i]) / dt

    # ---- reciprocal safety projection: agent-agent collisions -> impossible - #
    v_cmd = _project_safe(problem, pos, v_cmd, phase)
    # numerical guard: never report a step longer than v_max * dt
    for i in range(n):
        sp = float(np.linalg.norm(v_cmd[i]))
        cap = problem.agents[i].v_max * (1.0 - 1e-9)
        if sp > cap:
            v_cmd[i] *= cap / sp

    # ---- integrate + advance the task state ------------------------------- #
    nxt.pos = pos + v_cmd * dt
    nxt.vel = v_cmd
    for i in range(n):
        a = problem.agents[i]
        if phase[i] == PHASE_DWELL:
            nxt.dwell_left[i] = st.dwell_left[i] - 1
            if nxt.dwell_left[i] <= 0:
                nxt.phase[i] = PHASE_TO_GOAL
                nxt.wp_out[i] = nxt.t
                nxt.hint[i] = 0
        elif phase[i] == PHASE_TO_WP:
            if np.linalg.norm(nxt.pos[i] - np.asarray(a.waypoint)) < 1e-6:
                nxt.pos[i] = np.asarray(a.waypoint, float)
                nxt.phase[i] = PHASE_DWELL
                nxt.wp_in[i] = nxt.t
                if nxt.dwell_left[i] <= 0:        # zero-length HOI
                    nxt.phase[i] = PHASE_TO_GOAL
                    nxt.wp_out[i] = nxt.t
                    nxt.hint[i] = 0
        elif phase[i] == PHASE_TO_GOAL:
            if np.linalg.norm(nxt.pos[i] - np.asarray(a.goal)) < 1e-6:
                nxt.pos[i] = np.asarray(a.goal, float)
                nxt.phase[i] = PHASE_DONE
                nxt.done[i] = nxt.t
                nxt.vel[i] = 0.0
    return nxt


def _project_safe(problem: Problem, pos: np.ndarray, v: np.ndarray,
                  phase: np.ndarray, iters: int = 12) -> np.ndarray:
    """Shrink velocities until no pair's swept segments violate the hard bound.

    Stationary agents are already at a safe separation (invariant maintained by
    this very function), so scaling everything to zero is a valid fallback and
    the loop is guaranteed to terminate.
    """
    n, dt = problem.n, problem.dt
    v = v.copy()
    for _ in range(iters):
        worst = None
        for i in range(n):
            for j in range(i + 1, n):
                R = problem.min_sep(i, j)
                d = seg_seg_min_dist(pos[i], pos[i] + v[i] * dt,
                                     pos[j], pos[j] + v[j] * dt)
                if d < R + 1e-4:
                    if worst is None or d < worst[0]:
                        worst = (d, i, j)
        if worst is None:
            return v
        _, i, j = worst
        for k in (i, j):
            if phase[k] in (PHASE_DWELL, PHASE_DONE):
                continue                          # cannot be asked to move
            v[k] *= 0.5
            if np.linalg.norm(v[k]) < 1e-3:
                v[k] = 0.0
    # last resort: everyone who is allowed to stop, stops
    for k in range(n):
        if phase[k] not in (PHASE_DWELL, PHASE_DONE):
            v[k] = 0.0
    return v
