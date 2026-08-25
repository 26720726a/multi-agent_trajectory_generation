"""Problem definition for the Multi-Agent HOI toy experiment.

Abstraction from the project brief
----------------------------------
    each agent's HOI task      ->  Start / Waypoint / Goal
    dependency between tasks   ->  order in which waypoints are *used*
    interaction between agents ->  collision avoidance (circular footprints)
    efficient cooperation      ->  team completion time (makespan)

Everything lives in continuous 2D metres.  Time is discretised at `dt`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .geometry import Rect

Point = Tuple[float, float]

PALETTE = ["#C43F3F", "#3C6BB0", "#3E9E62", "#8A5FBF"]


@dataclass
class AgentSpec:
    """One agent = one simplified HOI task chain S -> W -> G."""
    name: str
    start: Point
    waypoint: Point
    goal: Point
    dwell: float = 1.0            # seconds spent at the waypoint (the "HOI")
    radius: float = 0.30          # physical footprint
    v_max: float = 1.20           # m/s
    interact_radius: float = 0.55  # visual/semantic radius of the HOI region
    color: str = PALETTE[0]


@dataclass
class World:
    """Rectangular room with axis-aligned rectangular obstacles."""
    width: float
    height: float
    obstacles: List[Rect] = field(default_factory=list)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (0.0, 0.0, self.width, self.height)

    def wall_rects(self, pad: float = 10.0) -> List[Rect]:
        """The room walls expressed as four big rectangles (used for inflation)."""
        w, h = self.width, self.height
        return [Rect(-pad, -pad, 0.0, h + pad),
                Rect(w, -pad, w + pad, h + pad),
                Rect(-pad, -pad, w + pad, 0.0),
                Rect(-pad, h, w + pad, h + pad)]


@dataclass
class Problem:
    """A full instance.

    `deps` holds (i, j) pairs meaning
        "agent i must be *finished* with its waypoint before agent j may
         *start* its waypoint",
    i.e.  t_wp_start[j] >= t_wp_start[i] + dwell[i] + gap.
    Together the pairs form a DAG over the agents' waypoint events.
    """
    name: str
    world: World
    agents: List[AgentSpec]
    deps: List[Tuple[int, int]] = field(default_factory=list)
    gap: float = 0.0              # extra separation required by the dependency (s)
    safety: float = 0.10          # extra clearance between agent footprints (m)
    dt: float = 0.10              # simulation timestep (s)
    description: str = ""

    @property
    def n(self) -> int:
        return len(self.agents)

    def min_sep(self, i: int, j: int) -> float:
        return self.agents[i].radius + self.agents[j].radius + self.safety

    def is_dag(self) -> bool:
        indeg = [0] * self.n
        for _, j in self.deps:
            indeg[j] += 1
        stack = [i for i in range(self.n) if indeg[i] == 0]
        seen = 0
        while stack:
            i = stack.pop()
            seen += 1
            for a, b in self.deps:
                if a == i:
                    indeg[b] -= 1
                    if indeg[b] == 0:
                        stack.append(b)
        return seen == self.n

    def topo_order(self) -> List[int]:
        indeg = [0] * self.n
        for _, j in self.deps:
            indeg[j] += 1
        ready = sorted(i for i in range(self.n) if indeg[i] == 0)
        out: List[int] = []
        while ready:
            i = ready.pop(0)
            out.append(i)
            for a, b in self.deps:
                if a == i:
                    indeg[b] -= 1
                    if indeg[b] == 0:
                        ready.append(b)
            ready.sort()
        return out

    def dep_str(self) -> str:
        if not self.deps:
            return "(none)"
        return ", ".join(f"{self.agents[i].name}->{self.agents[j].name}"
                         for i, j in self.deps)


# --------------------------------------------------------------------------- #
#  Scenarios
# --------------------------------------------------------------------------- #
def scenario_crossing() -> Problem:
    """Two agents with crossing routes and *adjacent* waypoints -- modelled on
    the reference figure.  The two HOI regions are closer than the sum of the
    agent radii, so the agents cannot be there at the same time, and the
    dependency says A must finish first.  The independent prior puts both of
    them there simultaneously, so it violates both requirements at once."""
    w = World(10.0, 10.0, obstacles=[
        Rect(2.4, 5.2, 5.6, 7.8),
        Rect(2.2, 2.4, 5.6, 4.2),
        Rect(6.8, 6.6, 8.9, 8.3),
        Rect(0.7, 3.0, 1.6, 4.6),
    ])
    agents = [
        AgentSpec("A", start=(1.1, 9.0), waypoint=(6.55, 5.45), goal=(9.0, 1.2),
                  dwell=1.4, color=PALETTE[0]),
        AgentSpec("B", start=(1.6, 1.1), waypoint=(6.30, 4.85), goal=(8.6, 9.2),
                  dwell=1.4, color=PALETTE[1]),
    ]
    return Problem("crossing2", w, agents, deps=[(0, 1)], gap=0.0,
                   description="crossing routes, adjacent HOI regions, A before B")


def scenario_deadlock() -> Problem:
    """A case where *speed tuning alone is not enough*: agent B's shortest route
    runs straight over agent A's goal, so once A parks there B is stuck forever.
    Sequential / prioritized planning fails; the joint planner survives only by
    letting B pass first, and the multi-route variant fixes it outright."""
    w = World(10.0, 10.0, obstacles=[
        Rect(2.6, 5.0, 5.4, 7.6),
        Rect(2.6, 3.3, 5.6, 5.0),
        Rect(1.4, 2.6, 2.8, 4.0),
        Rect(6.0, 2.6, 8.2, 3.9),
    ])
    agents = [
        AgentSpec("A", start=(1.6, 8.4), waypoint=(6.55, 5.55), goal=(4.9, 1.4),
                  dwell=1.4, color=PALETTE[0]),
        AgentSpec("B", start=(2.1, 1.2), waypoint=(6.35, 4.75), goal=(8.5, 8.6),
                  dwell=1.4, color=PALETTE[1]),
    ]
    return Problem("deadlock2", w, agents, deps=[(0, 1)], gap=0.0,
                   description="B's route crosses A's goal: speed tuning alone can fail")


def scenario_corridor() -> Problem:
    """Two agents that must both squeeze through one doorway, in opposite
    directions, while their waypoints sit on opposite sides of the wall.
    Pure "wait for the other" is feasible but very slow."""
    w = World(10.0, 10.0, obstacles=[
        Rect(0.0, 4.6, 4.1, 5.6),
        Rect(5.9, 4.6, 10.0, 5.6),
        Rect(1.2, 7.2, 3.4, 8.6),
        Rect(6.6, 1.4, 8.8, 2.8),
    ])
    agents = [
        AgentSpec("A", start=(1.0, 1.0), waypoint=(8.4, 8.4), goal=(1.2, 9.0),
                  dwell=1.2, color=PALETTE[0]),
        AgentSpec("B", start=(9.0, 9.0), waypoint=(1.6, 1.6), goal=(9.0, 1.0),
                  dwell=1.2, color=PALETTE[1]),
    ]
    return Problem("corridor2", w, agents, deps=[(0, 1)], gap=0.0,
                   description="single doorway, opposite directions, A before B")


def scenario_chain3() -> Problem:
    """Three agents with a chained dependency A -> B -> C in a cluttered room."""
    w = World(12.0, 12.0, obstacles=[
        Rect(2.2, 6.6, 6.4, 7.8),
        Rect(7.6, 3.0, 9.6, 8.2),
        Rect(2.4, 2.6, 5.0, 4.2),
        Rect(0.0, 9.0, 4.4, 10.0),
    ])
    agents = [
        AgentSpec("A", start=(1.0, 1.0), waypoint=(6.6, 5.3), goal=(11.0, 11.0),
                  dwell=1.2, color=PALETTE[0]),
        AgentSpec("B", start=(11.0, 1.0), waypoint=(6.6, 9.2), goal=(1.0, 6.0),
                  dwell=1.2, color=PALETTE[1]),
        AgentSpec("C", start=(1.0, 11.2), waypoint=(6.9, 2.0), goal=(11.0, 6.4),
                  dwell=1.2, color=PALETTE[2]),
    ]
    return Problem("chain3", w, agents, deps=[(0, 1), (1, 2)], gap=0.0,
                   description="cluttered room, chained dependency A->B->C")


def scenario_fork3() -> Problem:
    """Three agents with a fork dependency A -> B and A -> C.  B and C are
    independent of each other, so a good planner runs them in parallel while a
    sequential schedule serialises all three."""
    w = World(12.0, 12.0, obstacles=[
        Rect(3.0, 4.8, 5.2, 7.4),
        Rect(6.8, 4.8, 9.0, 7.4),
        Rect(3.0, 1.4, 9.0, 2.6),
        Rect(3.0, 9.6, 9.0, 10.8),
    ])
    agents = [
        AgentSpec("A", start=(6.0, 0.7), waypoint=(6.0, 6.1), goal=(6.0, 11.4),
                  dwell=1.2, color=PALETTE[0]),
        AgentSpec("B", start=(1.0, 1.0), waypoint=(11.0, 3.6), goal=(11.0, 11.2),
                  dwell=1.2, color=PALETTE[1]),
        AgentSpec("C", start=(11.0, 1.0), waypoint=(1.0, 8.6), goal=(1.0, 11.2),
                  dwell=1.2, color=PALETTE[2]),
    ]
    return Problem("fork3", w, agents, deps=[(0, 1), (0, 2)], gap=0.0,
                   description="fork dependency A->{B,C}, B and C can overlap")


SCENARIOS: Dict[str, callable] = {
    "crossing2": scenario_crossing,
    "corridor2": scenario_corridor,
    "deadlock2": scenario_deadlock,
    "chain3": scenario_chain3,
    "fork3": scenario_fork3,
}


def get_scenario(name: str) -> Problem:
    return SCENARIOS[name]()


# --------------------------------------------------------------------------- #
#  Random instance generator (for GT dataset construction)
# --------------------------------------------------------------------------- #
def random_problem(seed: int, n_agents: int = 2, size: float = 10.0,
                   n_obstacles: Tuple[int, int] = (3, 5),
                   dep_mode: str = "chain",
                   coupled_waypoints: float = 0.5) -> Problem:
    """Random room + random S/W/G + a random dependency DAG.

    `coupled_waypoints` is the probability that two agents' waypoints are
    deliberately placed close together, which is what makes the instance
    interesting (the HOI regions then physically exclude each other).
    """
    rng = np.random.default_rng(seed)
    radius, margin = 0.30, 0.45

    # --- obstacles: non-overlapping axis-aligned rectangles ---------------
    rects: List[Rect] = []
    for _ in range(int(rng.integers(n_obstacles[0], n_obstacles[1] + 1))):
        for _try in range(60):
            w_ = float(rng.uniform(1.2, 3.4))
            h_ = float(rng.uniform(1.2, 3.4))
            x0 = float(rng.uniform(0.9, size - 0.9 - w_))
            y0 = float(rng.uniform(0.9, size - 0.9 - h_))
            cand = Rect(x0, y0, x0 + w_, y0 + h_)
            grown = cand.inflate(2 * (radius + margin))
            if any(not (grown.x1 < r.x0 or grown.x0 > r.x1 or
                        grown.y1 < r.y0 or grown.y0 > r.y1) for r in rects):
                continue
            rects.append(cand)
            break
    world = World(size, size, rects)

    def free_pt() -> Point:
        for _ in range(4000):
            p = (float(rng.uniform(radius + 0.2, size - radius - 0.2)),
                 float(rng.uniform(radius + 0.2, size - radius - 0.2)))
            if all(not r.inflate(radius + 0.15).contains(p) for r in rects):
                return p
        raise RuntimeError("cannot sample a free point")

    def far_pts(k: int, dmin: float) -> List[Point]:
        pts: List[Point] = []
        while len(pts) < k:
            p = free_pt()
            if all(np.hypot(p[0] - q[0], p[1] - q[1]) > dmin for q in pts):
                pts.append(p)
        return pts

    starts = far_pts(n_agents, 2.5)
    goals = far_pts(n_agents, 2.5)
    wps = far_pts(n_agents, 2.5)
    # optionally pull one pair of waypoints together so they physically conflict
    if n_agents >= 2 and rng.random() < coupled_waypoints:
        i, j = rng.choice(n_agents, size=2, replace=False)
        ang = rng.uniform(0, 2 * np.pi)
        d = 0.55
        cand = (wps[i][0] + d * np.cos(ang), wps[i][1] + d * np.sin(ang))
        if all(not r.inflate(radius + 0.15).contains(cand) for r in rects) and \
           radius + 0.2 < cand[0] < size - radius - 0.2 and \
           radius + 0.2 < cand[1] < size - radius - 0.2:
            wps[j] = (float(cand[0]), float(cand[1]))

    agents = [AgentSpec(chr(ord("A") + i), starts[i], wps[i], goals[i],
                        dwell=float(rng.choice([0.8, 1.2, 1.6])),
                        radius=radius, color=PALETTE[i % len(PALETTE)])
              for i in range(n_agents)]

    perm = [int(x) for x in rng.permutation(n_agents)]
    if dep_mode == "chain":
        deps = [(perm[k], perm[k + 1]) for k in range(n_agents - 1)]
    elif dep_mode == "fork":
        deps = [(perm[0], perm[k]) for k in range(1, n_agents)]
    elif dep_mode == "none":
        deps = []
    else:
        deps = [(perm[0], perm[1])] if n_agents >= 2 else []
    return Problem(f"rand{seed}", world, agents, deps=deps, gap=0.0,
                   description=f"random instance (seed={seed}, {dep_mode})")
