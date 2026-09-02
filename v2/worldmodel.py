"""The World Model: "from *this* state, here are N ways the future could go."

Interface
---------
    wm = WorldModel(problem)
    modes  = wm.sample_modes(max_modes=16)
    rolls  = wm.rollouts(state, modes, horizon=None)   # -> List[Rollout]

Every rollout is a *joint* multi-agent future: all agents advance together under
the reactive controller of `controller.py`.  Two rollouts of the same state can
differ in

  * **space**  -- each agent is assigned one of several geometrically distinct
                  routes (different homotopy class: left of the pillar vs right
                  of it), obtained by penalising already-used visibility-graph
                  edges;
  * **time**   -- the *yield order* decides who gives way to whom, which changes
                  both who arrives first and how far each agent detours; a
                  cautious variant additionally scales down the yielding
                  agents' preferred speed.

This is deliberately a **procedural** generator, not a learned one.  The GT set
from the previous round is biased (instances the fixed-route planner could not
solve were discarded, 40 % of 2-agent and 80 % of 3-agent cases), so training a
generative model on it now would bake that bias into the World Model.  The
`WorldModel` API is kept narrow -- `sample_modes` + `rollouts` -- precisely so a
learned generator can replace it later without touching the Planner.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import PLANNER
from . import control as ctrl
from .control import (PHASE_DONE, PHASE_DWELL, PHASE_TO_GOAL, PHASE_TO_WP,
                      Guide, Route, SceneCache, TeamState, initial_state)
from .paths import VisibilityGraph, _safe_round
from .traj import Trajectory
from .world import Problem

# Responsibility shares implied by a yield order (see controller.step).
#: robot_radius=0.30 과 값만 우연히 같다 — 이쪽은 무차원 책임 분담 비율이다.
#: tests/test_layering.py 의 WAIVERS 에 그 사유로 등록되어 있다.
ALPHA_PRIVILEGED = 0.30
ALPHA_YIELDING = 0.85
ALPHA_NEUTRAL = 0.50


# --------------------------------------------------------------------------- #
#  A plan mode = one "way the team could organise itself"
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlanMode:
    routes: Tuple[int, ...]          # index into each agent's route library
    yield_rank: Tuple[int, ...]      # yield_rank[i] = priority of agent i (0 = first)
    cautious: bool = False           # yielding agents move at 75 % preferred speed
    split_side: bool = False         # privileged/yielding agents pass on opposite sides

    def label(self, problem: Problem) -> str:
        order = sorted(range(len(self.yield_rank)), key=lambda i: self.yield_rank[i])
        names = ">".join(problem.agents[i].name for i in order)
        tag = "".join(["c" if self.cautious else "", "s" if self.split_side else ""])
        return f"r{''.join(str(r) for r in self.routes)}|{names}" + (f"|{tag}" if tag else "")

    def alpha(self, n: int) -> np.ndarray:
        a = np.full((n, n), ALPHA_NEUTRAL)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.yield_rank[i] < self.yield_rank[j]:
                    a[i, j] = ALPHA_PRIVILEGED
                elif self.yield_rank[i] > self.yield_rank[j]:
                    a[i, j] = ALPHA_YIELDING
        return a

    def speed_scale(self, n: int) -> np.ndarray:
        if not self.cautious:
            return np.ones(n)
        med = (n - 1) / 2.0
        return np.array([0.75 if self.yield_rank[i] > med else 1.0
                         for i in range(n)])

    def side(self, n: int) -> np.ndarray:
        if not self.split_side:
            return np.zeros(n, int)
        med = (n - 1) / 2.0
        return np.array([1 if self.yield_rank[i] <= med else -1
                         for i in range(n)], int)


@dataclass
class Rollout:
    """One imagined future plus the bookkeeping the Planner needs."""
    mode: PlanMode
    traj: Trajectory
    reached_goal: np.ndarray         # (n,) bool -- inside the simulated window
    stalled: bool                    # no team progress for `stall_window` steps
    truncated: bool                  # stopped before everybody was home
    hit_horizon: bool                # ...because the *imagination* horizon ran
                                     # out, not because anything went wrong
    end_state: TeamState
    remaining: Dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  World model
# --------------------------------------------------------------------------- #
class WorldModel:
    def __init__(self, problem: Problem, k_routes: int = PLANNER.k_routes,
                 margin: float = 0.12, seed: int = 0):
        self.problem = problem
        self.scene = SceneCache(problem)
        self.rng = np.random.default_rng(seed)
        self.library: List[List[Route]] = [
            self._routes_for(i, k_routes, margin) for i in range(problem.n)]

    # -- route library ----------------------------------------------------- #
    def _routes_for(self, i: int, k: int, margin: float) -> List[Route]:
        a = self.problem.agents[i]
        vg = VisibilityGraph(self.problem.world, clearance=a.radius + margin)
        legs1 = vg.k_paths(a.start, a.waypoint, k=k) or []
        legs2 = vg.k_paths(a.waypoint, a.goal, k=k) or []
        if not legs1 or not legs2:
            raise RuntimeError(f"no route for agent {a.name}")
        out: List[Route] = []
        seen = set()
        for l1 in legs1:
            for l2 in legs2:
                s1 = _safe_round(l1, vg, a.radius)
                s2 = _safe_round(l2, vg, a.radius)
                g1, g2 = Guide(s1), Guide(s2)
                key = (round(g1.length, 2), round(g2.length, 2))
                if key in seen:
                    continue
                seen.add(key)
                out.append(Route(g1, g2, label=f"{a.name}#{len(out)}"))
        out.sort(key=lambda r: r.length)
        return out[:max(1, k * k)]

    # -- mode set ---------------------------------------------------------- #
    def sample_modes(self, max_modes: int = PLANNER.max_modes,
                     allow_cautious: bool = True) -> List[PlanMode]:
        """Deterministic, reproducible subsample of the mode space.

        The canonical mode -- shortest route for everybody, identity yield
        order, no speed scaling -- is always first, so the planner degrades
        gracefully to "everyone just drives" when nothing else is affordable.
        """
        n = self.problem.n
        # NOTE(S2): 아래는 전체 조합을 **먼저 다 만들고** 그 다음에 잘라낸다.
        # agent 6명이면 138,240개, 8명이면 30,965,760개라 여기서 서거나 OOM 이
        # 난다 (bench/run.py 모듈 docstring 이 별도 프로세스+타임아웃으로 감싸는
        # 이유).  고치지 않는다 — S4(Phase 1) 의 일이다.
        route_combos = list(itertools.product(
            *[range(len(lib)) for lib in self.library]))
        route_combos.sort(key=lambda c: (sum(c), c))
        perms = list(itertools.permutations(range(n)))
        variants = [(False, False), (False, True)]
        if allow_cautious:
            variants += [(True, True)]

        modes: List[PlanMode] = []
        for combo in route_combos:
            for perm in perms:
                rank = tuple(perm.index(i) for i in range(n))
                for cautious, split in variants:
                    modes.append(PlanMode(tuple(combo), rank, cautious, split))

        canonical = PlanMode(tuple([0] * n), tuple(range(n)), False, False)
        rest = [m for m in modes if m != canonical]
        if len(rest) > max_modes - 1:
            idx = self.rng.choice(len(rest), size=max_modes - 1, replace=False)
            rest = [rest[int(k)] for k in sorted(idx)]
        return [canonical] + rest

    # -- rollouts ---------------------------------------------------------- #
    def rollouts(self, state: TeamState, modes: Sequence[PlanMode],
                 horizon: Optional[int] = None, max_steps: int = 1200,
                 stall_window: int = PLANNER.stall_window) -> List[Rollout]:
        return [self.rollout(state, m, horizon, max_steps, stall_window)
                for m in modes]

    def rollout(self, state: TeamState, mode: PlanMode,
                horizon: Optional[int] = None, max_steps: int = 1200,
                stall_window: int = PLANNER.stall_window,
                trace: Optional[List[dict]] = None) -> Rollout:
        """Forward-simulate the team from `state` under `mode`.

        `horizon` (in timesteps) caps the imagined future; `None` means "until
        everybody is done".  A capped rollout is what makes this a receding
        horizon scheme -- the Planner then adds a critical-path estimate of what
        happens after the horizon.
        """
        p = self.problem
        routes = [self.library[i][min(mode.routes[i], len(self.library[i]) - 1)]
                  for i in range(p.n)]
        alpha = mode.alpha(p.n)
        sscale = mode.speed_scale(p.n)
        side = mode.side(p.n)

        budget = max_steps if horizon is None else min(max_steps, horizon)
        st = state.copy()
        # Re-anchor onto *this* mode's routes: the incoming hint indexes whatever
        # guide the state was produced on, which is a different polyline.
        for i in range(p.n):
            if st.phase[i] == PHASE_TO_WP:
                st.hint[i] = routes[i].leg1.project_full(st.pos[i])
            elif st.phase[i] == PHASE_TO_GOAL:
                st.hint[i] = routes[i].leg2.project_full(st.pos[i])
            else:
                st.hint[i] = 0
        pos_log = [st.pos.copy()]
        vel_log: List[np.ndarray] = []

        best_togo = self._to_go(routes, st)
        since_progress = 0
        stalled = False
        steps = 0
        # S3-A: 가속도 계측.  `Trajectory.extra` 로 나른다 — 계약이 필드 추가
        # 대신 그 딕셔너리를 쓰라고 정해 두었다 (tests/test_contract.py).
        stats = ctrl.new_stats()
        while steps < budget and not st.all_done:
            st = ctrl.step(p, self.scene, routes, st, alpha, sscale, side,
                           stats, trace)
            pos_log.append(st.pos.copy())
            vel_log.append(st.vel.copy())
            steps += 1
            togo = self._to_go(routes, st)
            if togo < best_togo - 0.02:
                best_togo = togo
                since_progress = 0
            else:
                since_progress += 1
                if since_progress >= stall_window:
                    stalled = True
                    break

        pos = np.stack(pos_log, axis=0)
        vel = np.stack(vel_log + [np.zeros((p.n, 2))], axis=0) if vel_log \
            else np.zeros_like(pos)
        base = state.t

        def rel(ev: np.ndarray) -> np.ndarray:
            """Global event timestep -> index inside this rollout window.

            Events that already happened before the window opened are clamped to
            0; `traj.concat` keeps the *first* occurrence, so the true (earlier)
            timestep recorded by a previous segment always wins.
            """
            ev = np.asarray(ev, int)
            return np.where(ev < 0, -1, np.maximum(ev - base, 0))

        traj = Trajectory(
            method=f"world-model rollout {mode.label(p)}",
            pos=pos, vel=vel, dt=p.dt,
            wp_in=rel(st.wp_in), wp_out=rel(st.wp_out), done=rel(st.done),
            extra={"v_max": [a.v_max for a in p.agents], "mode": mode,
                   "accel": stats})

        return Rollout(
            mode=mode, traj=traj,
            reached_goal=(st.phase == PHASE_DONE),
            stalled=stalled,
            truncated=(not st.all_done) and not stalled,
            hit_horizon=(horizon is not None and steps >= horizon
                         and not st.all_done and not stalled),
            end_state=st,
            remaining=self.remaining_estimate(routes, st))

    # -- helpers ----------------------------------------------------------- #
    def _to_go(self, routes: Sequence[Route], st: TeamState) -> float:
        """Scalar team progress measure (metres left).  Used for stall detection."""
        tot = 0.0
        for i in range(self.problem.n):
            ph = st.phase[i]
            if ph == PHASE_DONE:
                continue
            r = routes[i]
            if ph == PHASE_TO_WP:
                tot += r.leg1.remaining(int(st.hint[i])) + r.leg2.length
            elif ph == PHASE_DWELL:
                tot += r.leg2.length
            else:
                tot += r.leg2.remaining(int(st.hint[i]))
        return tot

    def remaining_estimate(self, routes: Sequence[Route],
                           st: TeamState) -> Dict[str, object]:
        """Critical-path estimate of the time still needed after `st`.

        Everybody travels its remaining route at full speed, agents may pass
        through each other, but the dependency ordering is still enforced --
        the same relaxation `coordination.critical_path_bound` uses, restarted
        from a mid-execution state.  Because collisions are relaxed away this
        is a *lower* bound on the true remainder, which is what a terminal cost
        should be: it never makes a rollout look better than it can be.
        """
        p = self.problem
        n = p.n
        dt = p.dt
        gap = p.gap

        t_to_wp = np.zeros(n)
        hoi = np.zeros(n)
        tail = np.zeros(n)
        for i in range(n):
            a = p.agents[i]
            r = routes[i]
            ph = st.phase[i]
            if ph == PHASE_TO_WP:
                t_to_wp[i] = r.leg1.remaining(int(st.hint[i])) / a.v_max
                hoi[i] = st.dwell_left[i] * dt
                tail[i] = r.leg2.length / a.v_max
            elif ph == PHASE_DWELL:
                hoi[i] = st.dwell_left[i] * dt
                tail[i] = r.leg2.length / a.v_max
            elif ph == PHASE_TO_GOAL:
                tail[i] = r.leg2.remaining(int(st.hint[i])) / a.v_max

        est_start = np.zeros(n)
        est_end = np.zeros(n)
        for i in p.topo_order():
            if st.phase[i] >= PHASE_DWELL:
                # the waypoint is already reached: nothing can delay it any more
                est_start[i] = 0.0
            else:
                t = t_to_wp[i]
                for (u, v) in p.deps:
                    if v == i:
                        t = max(t, est_end[u] + gap)
                est_start[i] = t
            est_end[i] = est_start[i] + hoi[i]
        finish = est_end + tail
        finish[st.phase == PHASE_DONE] = 0.0
        return {
            "makespan": float(finish.max()) if n else 0.0,
            "flow": float(finish.sum()),
            "finish": finish,
            "wp_start": est_start,
        }
