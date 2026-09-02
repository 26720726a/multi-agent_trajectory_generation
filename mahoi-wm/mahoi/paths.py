"""Stage 1 of the pipeline: the *independent prior*.

Each agent is routed Start -> Waypoint -> Goal on its own, ignoring the other
agents.  We use a visibility graph over obstacles inflated by the agent radius,
which gives the shortest collision-free polyline; corners are then filleted so
the result looks (and differentiates) like a smooth trajectory.

The output of this module is an `AgentTrack`: a fixed geometric path resampled
at uniform arclength `ds = v_max * dt`, with the waypoint dwell inserted as a
run of repeated samples.  Index k along the track therefore corresponds to
"k timesteps of travel at full speed", which is what makes the coordination
search in `coordination.py` a clean unit-cost search.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import (Rect, resample_polyline, round_corners, seg_hits_rect,
                       point_rect_dist, polyline_length)
from .world import AgentSpec, Problem, World

Point = Tuple[float, float]


# --------------------------------------------------------------------------- #
#  Visibility graph
# --------------------------------------------------------------------------- #
class VisibilityGraph:
    """Visibility graph over obstacles inflated by `clearance`."""

    def __init__(self, world: World, clearance: float, corner_eps: float = 1e-3):
        self.world = world
        self.clearance = clearance
        self.inflated = [r.inflate(clearance) for r in world.obstacles]
        lo, hi_x, hi_y = clearance, world.width - clearance, world.height - clearance

        nodes: List[Point] = []
        for rect in self.inflated:
            for c in rect.corners(corner_eps):
                if not (lo <= c[0] <= hi_x and lo <= c[1] <= hi_y):
                    continue
                if any(o.contains(c, eps=1e-6) for o in self.inflated):
                    continue
                nodes.append((float(c[0]), float(c[1])))
        self.base_nodes = nodes

    # -- helpers ------------------------------------------------------------
    def blocked(self, p: Point, q: Point) -> bool:
        return any(seg_hits_rect(p, q, r) for r in self.inflated)

    def free_point(self, p: Point) -> bool:
        lo = self.clearance
        if not (lo <= p[0] <= self.world.width - lo and
                lo <= p[1] <= self.world.height - lo):
            return False
        return not any(r.contains(p, eps=1e-9) for r in self.inflated)

    def project_free(self, p: Point) -> Point:
        """Nudge a point out of an inflated obstacle / off the wall."""
        if self.free_point(p):
            return p
        lo = self.clearance
        x = float(np.clip(p[0], lo, self.world.width - lo))
        y = float(np.clip(p[1], lo, self.world.height - lo))
        for _ in range(64):
            hit = next((r for r in self.inflated if r.contains((x, y), eps=1e-9)), None)
            if hit is None:
                break
            cand = [(hit.x0 - 1e-3, y), (hit.x1 + 1e-3, y),
                    (x, hit.y0 - 1e-3), (x, hit.y1 + 1e-3)]
            cand.sort(key=lambda c: (c[0] - x) ** 2 + (c[1] - y) ** 2)
            x, y = cand[0]
            x = float(np.clip(x, lo, self.world.width - lo))
            y = float(np.clip(y, lo, self.world.height - lo))
        return (x, y)

    # -- search -------------------------------------------------------------
    def shortest_path(self, src: Point, dst: Point,
                      edge_penalty: Optional[Dict[Tuple[int, int], float]] = None
                      ) -> Optional[np.ndarray]:
        """Euclidean shortest path from src to dst as a polyline (M,2)."""
        src, dst = self.project_free(src), self.project_free(dst)
        nodes = [src, dst] + self.base_nodes
        m = len(nodes)
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(m)]
        for a in range(m):
            for b in range(a + 1, m):
                if self.blocked(nodes[a], nodes[b]):
                    continue
                w = float(np.hypot(nodes[a][0] - nodes[b][0],
                                   nodes[a][1] - nodes[b][1]))
                if edge_penalty:
                    w += edge_penalty.get((a, b), 0.0) + edge_penalty.get((b, a), 0.0)
                adj[a].append((b, w))
                adj[b].append((a, w))

        dist = [np.inf] * m
        prev = [-1] * m
        dist[0] = 0.0
        pq = [(0.0, 0)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u] + 1e-12:
                continue
            if u == 1:
                break
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v] - 1e-12:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if not np.isfinite(dist[1]):
            return None
        chain, cur = [], 1
        while cur != -1:
            chain.append(nodes[cur])
            cur = prev[cur]
        return np.asarray(chain[::-1], float)

    def k_paths(self, src: Point, dst: Point, k: int = 3,
                penalty: float = 1.6) -> List[np.ndarray]:
        """A few *diverse* routes, obtained by penalising already-used edges.

        Used by the multi-route variant of the coordination planner: sometimes
        the cheapest way out of a deadlock is for one agent to take a slightly
        longer, topologically different route rather than to wait.
        """
        out: List[np.ndarray] = []
        pen: Dict[Tuple[int, int], float] = {}
        for _ in range(k):
            p = self.shortest_path(src, dst, edge_penalty=pen)
            if p is None:
                break
            if not any(len(p) == len(q) and np.allclose(p, q) for q in out):
                out.append(p)
            # penalise the interior vertices of this route so the next search
            # is pushed into a different corridor
            nodes = [self.project_free(src), self.project_free(dst)] + self.base_nodes
            idx = {tuple(np.round(n, 6)): i for i, n in enumerate(nodes)}
            ids = [idx.get(tuple(np.round(v, 6)), -1) for v in p]
            for a, b in zip(ids[:-1], ids[1:]):
                if a >= 0 and b >= 0:
                    pen[(a, b)] = pen.get((a, b), 0.0) + penalty
        return out


# --------------------------------------------------------------------------- #
#  Agent track
# --------------------------------------------------------------------------- #
@dataclass
class AgentTrack:
    """A fixed geometric route, discretised into equal arclength steps.

    The route is sampled every ds = v_max * dt / K metres, so advancing by K
    indices in one timestep means full speed and advancing by d < K means the
    agent moves at speed (d / K) * v_max.  K = 1 reduces to the original
    "go or wait" model; K > 1 lets an agent slow down instead of stopping dead.

    A run of identical samples in [wp_start, wp_end] encodes the waypoint dwell;
    that run is exactly K * (dwell / dt) samples long and is traversed at a
    fixed rate of K per timestep, so the HOI always takes `dwell` seconds.
    """
    pts: np.ndarray               # (n+1, 2)
    wp_start: int                 # index at which the agent reaches its waypoint
    wp_end: int                   # index at which the HOI is finished
    smooth: np.ndarray            # dense polyline, for plotting only
    length: float                 # geometric path length (m)
    K: int = 1                    # speed levels per timestep

    @property
    def n(self) -> int:
        return len(self.pts) - 1

    @property
    def free_flow_steps(self) -> int:
        """Timesteps needed to run the whole route at full speed."""
        return int(np.ceil(self.n / self.K))

    def pos(self, k: int) -> np.ndarray:
        return self.pts[min(max(k, 0), self.n)]


def build_track(problem: Problem, i: int, vg: VisibilityGraph,
                route: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                speed_levels: int = 1) -> Optional[AgentTrack]:
    """Independent prior route for agent `i`, discretised into an AgentTrack."""
    a = problem.agents[i]
    K = max(1, int(speed_levels))
    ds = a.v_max * problem.dt / K

    if route is None:
        leg1 = vg.shortest_path(a.start, a.waypoint)
        leg2 = vg.shortest_path(a.waypoint, a.goal)
    else:
        leg1, leg2 = route
    if leg1 is None or leg2 is None:
        return None

    # Fillet the corners, but keep S / W / G as exact stationary points.
    s1 = _safe_round(leg1, vg, a.radius)
    s2 = _safe_round(leg2, vg, a.radius)

    r1 = resample_polyline(s1, ds)
    r2 = resample_polyline(s2, ds)
    dwell_samples = int(round(a.dwell / problem.dt)) * K

    pts = [r1]
    wp_start = len(r1) - 1
    if dwell_samples > 0:
        pts.append(np.repeat(r1[-1:], dwell_samples, axis=0))
    wp_end = wp_start + dwell_samples
    pts.append(r2[1:])
    pts = np.concatenate(pts, axis=0)

    smooth = np.concatenate([s1, s2[1:]], axis=0)
    return AgentTrack(pts=pts, wp_start=wp_start, wp_end=wp_end, K=K,
                      smooth=smooth, length=polyline_length(s1) + polyline_length(s2))


def _safe_round(poly: np.ndarray, vg: VisibilityGraph, radius: float,
                slack: float = 0.04) -> np.ndarray:
    """Fillet corners, shrinking the fillet until clearance is preserved.

    `slack` absorbs the sagitta lost when the filleted curve is later resampled
    at ds-long chords, so the *executed* trajectory still clears the obstacle.
    """
    for cut in (0.35, 0.22, 0.12, 0.05):
        cand = round_corners(poly, cut_ratio=cut, max_cut=max(0.15, vg.clearance * 3))
        if _clearance_ok(cand, vg.world.obstacles, radius + slack):
            return cand
    return poly


def _clearance_ok(poly: np.ndarray, obstacles: Sequence[Rect], radius: float,
                  ds: float = 0.03) -> bool:
    dense = resample_polyline(poly, ds)
    for r in obstacles:
        if np.any(point_rect_dist(dense, r) < radius - 1e-6):
            return False
    return True


def build_prior(problem: Problem, margin: float = 0.12, speed_levels: int = 1
                ) -> Tuple[List[AgentTrack], List[VisibilityGraph]]:
    """Independent prior for every agent (the left panel of the reference fig)."""
    tracks, vgs = [], []
    for i, a in enumerate(problem.agents):
        vg = VisibilityGraph(problem.world, clearance=a.radius + margin)
        tr = build_track(problem, i, vg, speed_levels=speed_levels)
        if tr is None:
            raise RuntimeError(f"no route for agent {a.name}")
        tracks.append(tr)
        vgs.append(vg)
    return tracks, vgs


def build_route_options(problem: Problem, i: int, vg: VisibilityGraph,
                        k: int = 2, speed_levels: int = 1) -> List[AgentTrack]:
    """Alternative geometric routes for agent `i` (shortest first)."""
    a = problem.agents[i]
    legs1 = vg.k_paths(a.start, a.waypoint, k=k)
    legs2 = vg.k_paths(a.waypoint, a.goal, k=k)
    out: List[AgentTrack] = []
    for l1 in legs1:
        for l2 in legs2:
            tr = build_track(problem, i, vg, route=(l1, l2),
                             speed_levels=speed_levels)
            if tr is not None:
                out.append(tr)
    out.sort(key=lambda t: t.n)
    # de-duplicate by length
    uniq, seen = [], set()
    for t in out:
        key = round(t.length, 3)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)
    return uniq[:k * k]
