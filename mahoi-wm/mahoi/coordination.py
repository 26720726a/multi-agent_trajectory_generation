"""Stage 2: turn the independent prior into a feasible, *fast* team plan.

Key idea (path-velocity decomposition / coordination diagram)
------------------------------------------------------------
Freeze each agent's geometric route from `paths.py` and keep only one decision
variable per agent: how far along its own route it has travelled.  With the
route discretised at ds = v_max * dt, agent i's state is an integer index
k_i in [0, n_i] and in one timestep it may advance by 0 (wait) or 1 (go).

The joint state (k_0, ..., k_{n-1}) lives on a lattice -- the *coordination
diagram* -- and every requirement of the task becomes a forbidden region in it:

  * agent-agent collision -> {k : ||p_i(k_i) - p_j(k_j)|| < r_i + r_j + safety}
  * dependency i -> j     -> {k : k_j >= wpStart_j  and  k_i < wpEnd_i + gap}
                             (agent j may not reach its waypoint while agent i
                              has not finished its own -- and because a lattice
                              path is monotone, this is exactly the temporal
                              ordering constraint)
  * obstacles / waypoint / goal -> satisfied by construction of the route

Each lattice step costs one timestep, so a shortest monotone path from
(0,...,0) to (n_0,...,n_{n-1}) is a *minimum-team-time* plan -- the "GT" of the
reference figure.  A* with h(k) = max_i (n_i - k_i) is admissible and tight.
"""
from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .paths import AgentTrack, VisibilityGraph, build_route_options
from .world import Problem

EPS_FLOW = 1e-4          # tie-break: among equally fast plans prefer early finishers


# --------------------------------------------------------------------------- #
#  Solution
# --------------------------------------------------------------------------- #
@dataclass
class Solution:
    method: str
    tracks: List[AgentTrack]
    K: np.ndarray                     # (T+1, n) index along each agent's track
    dt: float
    runtime: float = 0.0
    feasible: bool = True
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.K.shape[1]

    @property
    def T(self) -> int:
        return self.K.shape[0] - 1

    @property
    def xy(self) -> np.ndarray:
        """(T+1, n, 2) world positions."""
        out = np.empty((self.T + 1, self.n, 2))
        for i, tr in enumerate(self.tracks):
            out[:, i, :] = tr.pts[self.K[:, i]]
        return out

    def completion_steps(self) -> np.ndarray:
        out = np.empty(self.n, int)
        for i, tr in enumerate(self.tracks):
            done = np.flatnonzero(self.K[:, i] >= tr.n)
            out[i] = done[0] if len(done) else self.T
        return out

    def event_steps(self) -> Tuple[np.ndarray, np.ndarray]:
        """(waypoint-start, waypoint-end) timestep of each agent."""
        s = np.empty(self.n, int)
        e = np.empty(self.n, int)
        for i, tr in enumerate(self.tracks):
            a = np.flatnonzero(self.K[:, i] >= tr.wp_start)
            b = np.flatnonzero(self.K[:, i] >= tr.wp_end)
            s[i] = a[0] if len(a) else self.T
            e[i] = b[0] if len(b) else self.T
        return s, e

    @property
    def team_time(self) -> float:
        return float(self.completion_steps().max()) * self.dt

    @property
    def flow_time(self) -> float:
        return float(self.completion_steps().sum()) * self.dt

    @property
    def travel_distance(self) -> float:
        return float(np.linalg.norm(np.diff(self.xy, axis=0), axis=2).sum())

    @property
    def total_wait(self) -> float:
        """Seconds lost to yielding: time spent below full speed while still
        travelling (outside the HOI dwell, before reaching the goal).

        With K = 1 this is exactly the number of full stops; with K > 1 a
        partial slowdown contributes its fractional share.
        """
        lost = 0.0
        for i, tr in enumerate(self.tracks):
            k = self.K[:, i]
            adv = np.diff(k)
            inside = (k[:-1] >= tr.wp_start) & (k[:-1] < tr.wp_end)
            done = k[:-1] >= tr.n
            active = ~inside & ~done
            lost += float(np.sum(np.clip(tr.K - adv[active], 0, tr.K))) / tr.K
        return lost * self.dt

    @property
    def mean_speed_ratio(self) -> float:
        """Average travel speed as a fraction of v_max (HOI dwell excluded)."""
        num = den = 0.0
        for i, tr in enumerate(self.tracks):
            k = self.K[:, i]
            adv = np.diff(k)
            inside = (k[:-1] >= tr.wp_start) & (k[:-1] < tr.wp_end)
            done = k[:-1] >= tr.n
            active = ~inside & ~done
            num += float(np.sum(adv[active])) / tr.K
            den += float(np.count_nonzero(active))
        return num / den if den else 1.0


def solution_from_K(problem: Problem, tracks: List[AgentTrack], K, method: str,
                    runtime: float, feasible: bool = True,
                    note: str = "") -> Solution:
    return Solution(method=method, tracks=tracks, K=np.asarray(K, int),
                    dt=problem.dt, runtime=runtime, feasible=feasible, note=note)


# --------------------------------------------------------------------------- #
#  Forbidden regions of the coordination lattice
# --------------------------------------------------------------------------- #
def _pair_threshold(problem: Problem, i: int, j: int) -> float:
    """Node-check threshold that also certifies the continuous motion.

    If both endpoints of the relative-motion segment are at distance
    >= sqrt(R^2 + L^2/4), every point of the segment is at distance >= R,
    where L is the largest change of the relative offset in one timestep.
    """
    R = problem.min_sep(i, j)
    L = (problem.agents[i].v_max + problem.agents[j].v_max) * problem.dt
    return float(np.sqrt(R * R + 0.25 * L * L))


def pair_free_table(problem: Problem, tracks: List[AgentTrack],
                    i: int, j: int) -> np.ndarray:
    """(n_i+1, n_j+1) boolean: True where agents i and j may coexist."""
    d = np.linalg.norm(tracks[i].pts[:, None, :] - tracks[j].pts[None, :, :], axis=2)
    return d >= _pair_threshold(problem, i, j)


def dep_free_table(problem: Problem, tracks: List[AgentTrack],
                   i: int, j: int) -> np.ndarray:
    """(n_i+1, n_j+1) boolean for the single dependency i -> j."""
    gap_idx = int(round(problem.gap / problem.dt)) * tracks[i].K
    ki = np.arange(tracks[i].n + 1)[:, None]
    kj = np.arange(tracks[j].n + 1)[None, :]
    bad = (kj >= tracks[j].wp_start) & (ki < tracks[i].wp_end + gap_idx)
    return ~bad


# --------------------------------------------------------------------------- #
#  Critical-path lower bound  (dependency-aware, collision-free)
# --------------------------------------------------------------------------- #
def critical_path_bound(problem: Problem, tracks: List[AgentTrack]) -> Dict[str, object]:
    """Tightest easily-computed lower bound on the team completion time.

    Each agent travels at full speed on its shortest route, agents are allowed
    to pass through each other, but the dependency ordering is still enforced.
    Propagating the earliest waypoint times along the dependency DAG gives the
    *critical path* -- no feasible plan can beat it.

    This is much tighter than the independent prior, which ignores the
    dependencies entirely, and it is what makes "our plan is X% above optimal"
    a meaningful statement.
    """
    gap_steps = int(round(problem.gap / problem.dt))
    n = problem.n
    wp_free = [int(np.ceil(tr.wp_start / tr.K)) for tr in tracks]        # S -> W
    hoi = [tr.wp_end // tr.K - tr.wp_start // tr.K for tr in tracks]     # HOI length
    tail = [int(np.ceil((tr.n - tr.wp_end) / tr.K)) for tr in tracks]    # W -> G

    est_start = [0] * n
    est_end = [0] * n
    for i in problem.topo_order():
        t = wp_free[i]
        for (a, b) in problem.deps:
            if b == i:
                t = max(t, est_end[a] + gap_steps)
        est_start[i] = t
        est_end[i] = t + hoi[i]
    finish = [est_end[i] + tail[i] for i in range(n)]

    return {
        "lb_team_time": max(finish) * problem.dt,
        "lb_flow_time": sum(finish) * problem.dt,
        "wp_start": [t * problem.dt for t in est_start],
        "finish": [t * problem.dt for t in finish],
        "critical_agent": problem.agents[int(np.argmax(finish))].name,
    }


def build_free_lattice(problem: Problem, tracks: List[AgentTrack],
                       max_cells: float = 6e7) -> np.ndarray:
    """Boolean array over the coordination lattice.  True = configuration allowed."""
    n = len(tracks)
    shape = tuple(tr.n + 1 for tr in tracks)
    cells = int(np.prod(shape))
    if cells > max_cells:
        raise MemoryError(f"coordination lattice too large: {shape} = {cells} cells")

    free = np.ones(shape, dtype=bool)

    def _apply(tbl: np.ndarray, i: int, j: int):
        # `tbl` is indexed [k_i, k_j]; broadcasting needs axis order i < j,
        # so transpose first when the pair is given the other way round.
        if i > j:
            tbl, i, j = tbl.T, j, i
        bshape = [1] * n
        bshape[i] = tbl.shape[0]
        bshape[j] = tbl.shape[1]
        np.logical_and(free, np.ascontiguousarray(tbl).reshape(bshape), out=free)

    for i in range(n):
        for j in range(i + 1, n):
            _apply(pair_free_table(problem, tracks, i, j), i, j)
    for (i, j) in problem.deps:
        _apply(dep_free_table(problem, tracks, i, j), i, j)
    return free


# --------------------------------------------------------------------------- #
#  Planner 0: independent prior (reference / lower bound, NOT feasible)
# --------------------------------------------------------------------------- #
def plan_independent(problem: Problem, tracks: List[AgentTrack]) -> Solution:
    t0 = time.perf_counter()
    T = max(tr.free_flow_steps for tr in tracks)
    K = np.empty((T + 1, len(tracks)), int)
    for i, tr in enumerate(tracks):
        K[:, i] = np.minimum(np.arange(T + 1) * tr.K, tr.n)
    return solution_from_K(problem, tracks, K, "independent prior",
                           time.perf_counter() - t0, feasible=False,
                           note="each agent planned alone; ignores the others")


# --------------------------------------------------------------------------- #
#  1-D velocity search (sequential / prioritized baselines)
# --------------------------------------------------------------------------- #
def _velocity_search(problem: Problem, tracks: List[AgentTrack], me: int,
                     others: Sequence[int], others_xy: np.ndarray,
                     release: int, earliest_start: int,
                     horizon: int) -> Optional[np.ndarray]:
    """Fastest speed profile for agent `me`, treating `others` as moving obstacles.

    others_xy : (To+1, len(others), 2); positions are held after To.
    release   : agent may not reach wp_start before this timestep.
    earliest_start : agent may not move at all before this timestep.
    """
    tr = tracks[me]
    N = tr.n
    To = max(len(others_xy) - 1, 0)
    thresh = np.array([_pair_threshold(problem, me, o) for o in others]) \
        if len(others) else np.zeros(0)

    def ok(k: int, t: int) -> bool:
        if not len(others):
            return True
        d = np.linalg.norm(others_xy[min(t, To)] - tr.pts[k][None, :], axis=1)
        return bool(np.all(d >= thresh))

    if not ok(0, 0):
        return None
    KL = tr.K
    seen = {(0, 0)}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    pq: List[Tuple[int, int, Tuple[int, int]]] = [(N, 0, (0, 0))]
    goal_state = None
    while pq:
        _, _, (k, t) = heapq.heappop(pq)
        if k >= N and t >= To:
            goal_state = (k, t)
            break
        if t >= horizon:
            continue
        if tr.wp_start <= k < tr.wp_end:
            moves = (KL,)                    # the HOI runs at a fixed rate
        else:
            moves = tuple(range(KL, -1, -1))
        for dk in moves:
            nk = k + dk
            if nk > N:
                continue
            if k < tr.wp_start < nk:
                continue                     # must land exactly on the waypoint
            if dk > 0 and t < earliest_start:
                continue
            if nk >= tr.wp_start and t + 1 < release:
                continue                     # dependency release time
            nt = t + 1
            if (nk, nt) in seen or not ok(nk, nt):
                continue
            seen.add((nk, nt))
            parent[(nk, nt)] = (k, t)
            h = max(-(-(N - nk) // KL), To - nt)
            heapq.heappush(pq, (nt + h, nk, (nk, nt)))
    if goal_state is None:
        return None
    chain = [goal_state]
    while chain[-1] in parent:
        chain.append(parent[chain[-1]])
    chain.reverse()
    return np.asarray([k for k, _ in chain], int)


def _pad(K: np.ndarray, T: int) -> np.ndarray:
    if len(K) >= T + 1:
        return K[:T + 1]
    return np.concatenate([K, np.repeat(K[-1:], T + 1 - len(K), axis=0)], axis=0)


def _assemble(profiles: List[np.ndarray]) -> np.ndarray:
    T = max(len(p) for p in profiles) - 1
    return np.stack([_pad(p, T) for p in profiles], axis=1)


def _others_xy(tracks: List[AgentTrack], profiles: Dict[int, np.ndarray],
               planned: List[int]) -> np.ndarray:
    if not planned:
        return np.zeros((1, 0, 2))
    T = max(len(profiles[o]) for o in planned) - 1
    return np.stack([tracks[o].pts[_pad(profiles[o], T)] for o in planned], axis=1)


# --------------------------------------------------------------------------- #
#  Planner 1: fully sequential baseline ("just wait until the other is done")
# --------------------------------------------------------------------------- #
def plan_sequential(problem: Problem, tracks: List[AgentTrack]) -> Solution:
    """The lazy solution the brief warns about: strictly one agent at a time,
    in a dependency-consistent order.  Always feasible, always slow."""
    t0 = time.perf_counter()
    horizon = sum(tr.n for tr in tracks) + 400
    profiles: Dict[int, np.ndarray] = {}
    planned: List[int] = []
    cursor = 0
    for me in problem.topo_order():
        prof = _velocity_search(problem, tracks, me, planned,
                                _others_xy(tracks, profiles, planned),
                                release=0, earliest_start=cursor, horizon=horizon)
        if prof is None:
            return solution_from_K(problem, tracks, np.zeros((1, problem.n), int),
                                   "sequential", time.perf_counter() - t0,
                                   feasible=False, note="no sequential solution")
        profiles[me] = prof
        planned.append(me)
        cursor = int(np.flatnonzero(prof >= tracks[me].n)[0])
    K = _assemble([profiles[i] for i in range(problem.n)])
    return solution_from_K(problem, tracks, K, "sequential",
                           time.perf_counter() - t0,
                           note="one agent at a time (dependency order)")


# --------------------------------------------------------------------------- #
#  Planner 2: prioritized velocity tuning
# --------------------------------------------------------------------------- #
def _topological_orders(problem: Problem, limit: int = 24) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    for perm in itertools.permutations(range(problem.n)):
        pos = {a: k for k, a in enumerate(perm)}
        if all(pos[i] < pos[j] for i, j in problem.deps):
            out.append(perm)
        if len(out) >= limit:
            break
    return out


def _plan_prioritized_order(problem: Problem, tracks: List[AgentTrack],
                            order: Sequence[int]) -> Optional[np.ndarray]:
    horizon = sum(tr.n for tr in tracks) + 400
    gap_steps = int(round(problem.gap / problem.dt))
    profiles: Dict[int, np.ndarray] = {}
    planned: List[int] = []
    for me in order:
        release = 0
        for (i, j) in problem.deps:
            if j == me and i in profiles:
                done = np.flatnonzero(profiles[i] >= tracks[i].wp_end)
                te = int(done[0]) if len(done) else len(profiles[i]) - 1
                release = max(release, te + gap_steps)
        prof = _velocity_search(problem, tracks, me, planned,
                                _others_xy(tracks, profiles, planned),
                                release=release, earliest_start=0, horizon=horizon)
        if prof is None:
            return None
        profiles[me] = prof
        planned.append(me)
    return _assemble([profiles[i] for i in range(problem.n)])


def plan_prioritized(problem: Problem, tracks: List[AgentTrack]) -> Solution:
    """Plan agents one by one; each avoids the ones already planned.  Every
    dependency-consistent priority order is tried and the fastest kept."""
    t0 = time.perf_counter()
    best_K, best_cost, best_order = None, None, None
    for order in _topological_orders(problem):
        K = _plan_prioritized_order(problem, tracks, order)
        if K is None:
            continue
        s = solution_from_K(problem, tracks, K, "tmp", 0.0)
        cost = (s.team_time, s.flow_time)
        if best_cost is None or cost < best_cost:
            best_K, best_cost, best_order = K, cost, order
    if best_K is None:
        return solution_from_K(problem, tracks, np.zeros((1, problem.n), int),
                               "prioritized", time.perf_counter() - t0,
                               feasible=False, note="no priority order worked")
    names = " < ".join(problem.agents[i].name for i in best_order)
    return solution_from_K(problem, tracks, best_K, "prioritized",
                           time.perf_counter() - t0,
                           note=f"best priority order: {names}")


# --------------------------------------------------------------------------- #
#  Planner 3: joint coordination-lattice A*   (main method)
# --------------------------------------------------------------------------- #
def plan_coordination(problem: Problem, tracks: List[AgentTrack],
                      free: Optional[np.ndarray] = None,
                      keep_lattice: bool = False,
                      flow_weight: float = EPS_FLOW,
                      stop_weight: float = 1e-3) -> Solution:
    """Joint minimum-team-time plan on fixed routes.

    `flow_weight` w defines the objective  makespan + w * flowtime (in steps).
    With the default w ~ 1e-4 this is lexicographic: minimise team time first,
    then break ties by finishing individual agents early.  Larger w trades a
    little team time for a lot of idle-waiting.
    """
    t0 = time.perf_counter()
    n = len(tracks)
    shape = tuple(tr.n + 1 for tr in tracks)
    if free is None:
        free = build_free_lattice(problem, tracks)

    # flat, byte-indexed occupancy for a fast pure-python inner loop
    flat_free = np.ascontiguousarray(free).reshape(-1).tobytes()
    strides: List[int] = [1] * n
    for a in range(n - 2, -1, -1):
        strides[a] = strides[a + 1] * shape[a + 1]

    N = [tr.n for tr in tracks]
    wpS = [tr.wp_start for tr in tracks]
    wpE = [tr.wp_end for tr in tracks]
    KL = [tr.K for tr in tracks]

    deltas = [d for d in itertools.product(*[range(k + 1) for k in KL]) if any(d)]
    offsets = [sum(d[a] * strides[a] for a in range(n)) for d in deltas]

    start_flat = 0
    goal_flat = sum(N[a] * strides[a] for a in range(n))

    def fail(msg: str) -> Solution:
        return solution_from_K(problem, tracks, np.zeros((1, n), int),
                               "coordination A*", time.perf_counter() - t0,
                               feasible=False, note=msg)

    if not flat_free[start_flat]:
        return fail("start configuration already violates a constraint")
    if not flat_free[goal_flat]:
        return fail("goal configuration violates a constraint")

    def decode(flat: int) -> List[int]:
        k, rem = [0] * n, flat
        for a in range(n):
            k[a] = rem // strides[a]
            rem -= k[a] * strides[a]
        return k

    g: Dict[int, float] = {start_flat: 0.0}
    parent: Dict[int, int] = {}
    closed = set()
    pq: List[Tuple[float, int]] = [(float(max(N)), start_flat)]
    n_exp = 0
    found = False
    while pq:
        _, cur = heapq.heappop(pq)
        if cur in closed:
            continue
        closed.add(cur)
        n_exp += 1
        if cur == goal_flat:
            found = True
            break
        k = decode(cur)
        gc = g[cur]
        for d, off in zip(deltas, offsets):
            valid = True
            for a in range(n):
                nk = k[a] + d[a]
                if nk > N[a]:
                    valid = False
                    break
                if wpS[a] <= k[a] < wpE[a]:
                    if d[a] != KL[a]:       # the HOI runs at a fixed rate
                        valid = False
                        break
                elif k[a] < wpS[a] < nk:
                    valid = False           # must land exactly on the waypoint
                    break
            if not valid:
                continue
            nxt = cur + off
            if nxt in closed or not flat_free[nxt]:
                continue
            unfinished = 0
            stops = 0
            h = 0
            for a in range(n):
                left = N[a] - k[a] - d[a]
                if left > 0:
                    unfinished += 1
                    if d[a] == 0:
                        stops += 1          # a full stop, not a slowdown
                    steps = -(-left // KL[a])
                    if steps > h:
                        h = steps
            ng = gc + 1.0 + flow_weight * unfinished + stop_weight * stops
            if ng < g.get(nxt, np.inf):
                g[nxt] = ng
                parent[nxt] = cur
                heapq.heappush(pq, (ng + h, nxt))

    rt = time.perf_counter() - t0
    if not found:
        return fail("deadlock: no feasible coordination on these fixed routes")

    chain = [goal_flat]
    while chain[-1] in parent:
        chain.append(parent[chain[-1]])
    chain.reverse()
    K = np.asarray([decode(c) for c in chain], int)
    sol = solution_from_K(problem, tracks, K, "coordination A*", rt,
                          note="minimum team time on the prior routes")
    sol.extra.update(expanded=n_exp, lattice=shape,
                     lattice_cells=int(np.prod(shape)))
    if keep_lattice:
        sol.extra["free"] = free
    return sol


# --------------------------------------------------------------------------- #
#  Planner 4: coordination A* over alternative geometric routes
# --------------------------------------------------------------------------- #
def plan_coordination_multiroute(problem: Problem, k_routes: int = 2,
                                 margin: float = 0.12) -> Solution:
    """If pure speed tuning deadlocks or wastes time waiting, let each agent
    also pick a different route (homotopy class).  Small exhaustive search."""
    t0 = time.perf_counter()
    options: List[List[AgentTrack]] = []
    for i in range(problem.n):
        vg = VisibilityGraph(problem.world,
                             clearance=problem.agents[i].radius + margin)
        opts = build_route_options(problem, i, vg, k=k_routes)
        if not opts:
            raise RuntimeError(f"no route options for agent {problem.agents[i].name}")
        options.append(opts)

    best: Optional[Solution] = None
    tried = 0
    for combo in itertools.product(*[range(len(o)) for o in options]):
        tracks = [options[i][c] for i, c in enumerate(combo)]
        try:
            sol = plan_coordination(problem, tracks)
        except MemoryError:
            continue
        tried += 1
        if not sol.feasible:
            continue
        key = (sol.team_time, sol.flow_time, round(sol.travel_distance, 3))
        if best is None or key < best.extra["key"]:
            sol.extra["key"] = key
            sol.extra["routes"] = combo
            best = sol
    if best is None:
        raise RuntimeError("no feasible route combination")
    best.method = "coordination A* + route choice"
    best.runtime = time.perf_counter() - t0
    best.note = f"best of {tried} route combinations"
    return best
