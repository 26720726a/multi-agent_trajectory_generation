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

S2 이관 메모
------------
원본 `mahoi/wm/controller.py`(499 LOC)에서 **안전에 해당하는 부분만** 아래로
내렸다.  로직은 하나도 바꾸지 않았다.

    _project_safe        -> safety/project.py
    _ttc + VO 필터 루프  -> safety/vo.py   (`add_vo_cost`)
    PHASE_*              -> safety/phase.py

원본의 import 경로가 그대로 살아 있도록 세 가지 모두 여기서 re-export 한다
(`tests/test_contract.py` 가 `PHASE_*` 와 `TeamState` 를 이 모듈에서 가져온다).
"Tunables" 블록은 `config/controller.py` 참조로 바꿨고 **값은 전부 동일**하다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import CONTROLLER, PHYSICS
from safety.geometry import Rect
from safety.phase import (PHASE_DONE, PHASE_DWELL, PHASE_TO_GOAL,  # noqa: F401
                          PHASE_TO_WP)
from safety.project import _project_safe                           # noqa: F401
from safety.vo import _ttc, add_vo_cost                            # noqa: F401
from .world import Problem

# --------------------------------------------------------------------------- #
#  Tunables (all explicit -- no learned parameters anywhere in this file)
# --------------------------------------------------------------------------- #
#  원본 controller.py:37-54 의 블록.  값은 그대로이고 출처만 config 로 옮겼다.
#  이름을 남겨 두는 이유는 원본 코드가 이 이름으로 읽히기 때문이다 — 아래 본문의
#  `TAU`, `SOFT_MARGIN` 같은 표현이 원본과 글자까지 같아야 diff 가 읽힌다.
LOOKAHEAD = CONTROLLER.lookahead      # m, pure-pursuit lookahead along the route
TAU = CONTROLLER.tau                  # s, velocity-obstacle time horizon
TAU_OBST = CONTROLLER.tau_obst        # s, shorter horizon for static geometry
W_TTC = CONTROLLER.w_ttc              # weight on 1/time-to-collision (neighbours)
W_TTC_OBST = CONTROLLER.w_ttc_obst    # weight on 1/time-to-collision (obstacles)
W_TURN = CONTROLLER.w_turn            # anti-oscillation: penalise changing the command
W_SIDE = CONTROLLER.w_side            # strength of the mode's left/right passing bias
OBST_PAD = CONTROLLER.obst_pad        # extra clearance kept from inflated obstacles (m)
SOFT_MARGIN = CONTROLLER.vo_soft_margin  # m, preferred extra separation
SNAP_TOL = CONTROLLER.snap_tol        # m, distance at which we snap onto W or G
DEP_STANDOFF = CONTROLLER.dep_standoff   # m, where a blocked successor holds

#: 가속도 상한 (m/s^2).  **원본에는 없던 제한이다** (S0 §C-1) — 원본은 매 스텝
#: 후보 팬에서 임의의 속도를 고를 수 있었고 급변은 `W_TURN` 으로 비용만 매겼다.
#: `inf` 면 `_candidates` 의 제한이 통째로 건너뛰어져 S2 동작을 비트 단위로
#: 재현한다 (tests/test_amax.py).
A_MAX = PHYSICS.a_max

#: 전멸(후보 65개 전멸) 시 즉시 정지 대신 a_max 로 감속할 것인가.
WIPEOUT_DECELERATE = CONTROLLER.wipeout_decelerate

#: S3-B2: 후보를 고른 뒤 최대 감속으로 설 때까지의 정적 기하 검사.
MULTISTEP_LOOKAHEAD = CONTROLLER.multistep_lookahead

#: 제동 항이 진행 방향으로 내다보는 거리 (m).  `None` 이면 기하 제동을 끈다 —
#: S3-A 3차에서 측정으로 무효가 확인됐다.  근거는 config/controller.py.
BRAKE_LOOKAHEAD = CONTROLLER.brake_lookahead

SPEED_LEVELS = CONTROLLER.speed_levels
#: 원본은 `np.deg2rad([...])` 였다.  config 는 `math.radians` 로 같은 값을 만든다
#: — 두 결과가 비트 단위로 같은 것을 확인했다 (reports/S2_migration.md §3).
ANGLE_LEVELS = np.asarray(CONTROLLER.angle_levels, float)


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
#  PHASE_TO_WP / PHASE_DWELL / PHASE_TO_GOAL / PHASE_DONE 은 safety/phase.py 에
#  있다 (안전 계층이 "움직여 달라고 할 수 없는 에이전트"를 알아야 한다).
#  위에서 re-export 하므로 원본과 같은 이름으로 여기서 가져다 쓸 수 있다.


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
        room bounds. `p` may be one `(2,)` common start or one `(C, 2)` start
        per candidate; the latter is used by S3-B2's deceleration trajectory.
        A candidate survives only if its whole swept segment stays clear.
        """
        p = np.asarray(p, float)
        if p.ndim == 1:
            p = np.broadcast_to(p, qs.shape)
        if p.shape != qs.shape:
            raise ValueError("p must be shape (2,) or match qs")
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
        d = qs - p                                           # (C, 2)
        t0 = np.zeros((len(qs), len(R)))
        t1 = np.ones((len(qs), len(R)))
        alive = np.ones((len(qs), len(R)), bool)
        for num_c, den_c in ((p[:, 0, None] - x0[None, :], -d[:, 0]),
                             (x1[None, :] - p[:, 0, None], d[:, 0]),
                             (p[:, 1, None] - y0[None, :], -d[:, 1]),
                             (y1[None, :] - p[:, 1, None], d[:, 1])):
            num = num_c                                      # (C, R)
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


#: 제동 자유 거리를 잴 때 쏘는 광선의 개수 (부채꼴을 몇 갈래로 나눌지).
#: 각도 자체는 고정하지 않는다 — `brake_free_distance` 가 속도에서 유도한다.
BRAKE_N_RAYS = 5


def brake_free_distance(scene: "SceneCache", i: int, pos_i: np.ndarray,
                        direction: np.ndarray, speed: float, dv_max: float,
                        cap: float, n: int = 8) -> float:
    """**어느 방향으로든** 자유롭게 갈 수 있는 최대 거리 (S3-A 3차).

    진행 방향을 중심으로 `BRAKE_RAYS` 부채꼴로 광선을 쏘고, 각 광선에서 처음
    막히는 지점 직전까지를 구한 뒤 **최댓값**을 돌려준다.  `free_steps` 는
    **스윕 세그먼트**를 보므로 "여기까지 자유" 는 "그 사이가 전부 자유" 다.

    **왜 최댓값인가.**  막으려는 것은 후보 전멸이고, 전멸은 **모든** 방향이
    막힐 때만 일어난다.  한 방향이라도 멀리 갈 수 있으면 감속할 이유가 없다.

    **왜 부채꼴인가** (외길 광선이 아니라).  진행 방향 하나만 보면 굽은 통로
    에서 과도하게 감속한다 — 실제로 5종 시나리오 중 둘이 완주에 실패했다.
    지시서가 미리 경고한 경우이고, 실측으로 확인해 부채꼴로 넓혔다.
    반각은 임의로 고르지 않고 `asin(dv_max/|v|)` 로 **`a_max` 에서 유도**한다.

    **왜 반경 검색이 아닌가.**  `scene.nearest_points()` 는 방향을 보지 않아
    옆으로 스쳐 지나가는 벽까지 세므로 통로 안에서 항상 기어가게 된다.
    광선은 `step()` 의 하드 필터와 **정확히 같은 판정**(`free_steps`)을 쓰므로
    무엇이 기각될지를 그대로 예측한다.

    비용은 `free_steps` 호출 한 번(끝점 `len(BRAKE_RAYS) * n`개)이다.
    """
    # 부채꼴의 반각은 **다음 스텝에 실제로 갈 수 있는 방향 범위**다.
    # 횡으로 쓸 수 있는 속도 변화가 `dv_max` 뿐이므로 `asin(dv_max/|v|)` 이고,
    # `|v| <= dv_max` 면 어느 방향으로든 갈 수 있어 반각이 90° 다.
    # 임의로 고른 각도가 아니라 `a_max` 에서 유도된 값이라는 점이 중요하다 —
    # 넓게 잡으면 제동이 걸리지 않고, 좁게 잡으면 굽은 통로에서 과도해진다.
    half = math.asin(min(1.0, dv_max / max(speed, 1e-12)))
    angles = np.linspace(-half, half, BRAKE_N_RAYS)
    ca, sa = np.cos(angles), np.sin(angles)
    dirs = np.stack([direction[0] * ca - direction[1] * sa,
                     direction[0] * sa + direction[1] * ca], axis=1)   # (R, 2)
    ds = np.linspace(cap / n, cap, n)                                   # (n,)
    qs = (pos_i[None, None, :]
          + dirs[:, None, :] * ds[None, :, None]).reshape(-1, 2)        # (R*n, 2)
    ok = scene.free_steps(i, pos_i, qs).reshape(len(dirs), n)
    best = 0.0
    for r in range(len(dirs)):
        row = ok[r]
        if row.all():
            return cap
        first_bad = int(np.argmin(row))
        if first_bad > 0:
            best = max(best, float(ds[first_bad - 1]))
    return best


def brake_speed(d_target: float) -> float:
    """`d_target` 앞에서 멈추려면 지금 낼 수 있는 최대 속력 (S3-A').

        v <= sqrt(2 * a_max * d)

    등가속 제동의 표준식이다.  `a_max` 가 유한하지 않으면 **곧바로 inf 를
    돌려준다** — `2 * inf * 0.0` 이 `nan` 이 되는 것을 피하려는 것이기도 하고,
    그래야 `min(v_lim, d/dt, inf)` 가 원래 식 `min(v_lim, d/dt)` 와 부동소수
    수준에서까지 같아져 S2 비트 단위 재현이 성립하기 때문이다.

    **왜 필요한가** (S3-A 보고서 §6-2): 원래 식은 `min(v_lim, d_target/dt)` 라
    목표가 한 스텝 거리보다 멀면 언제나 최대 속력을 원했다.  `a_max` 가 없을
    때는 목표에 닿는 순간 속도를 0 으로 갈아끼우면 그만이었지만, 상한이 생기면
    **자기가 지울 수 없는 속도**를 내고 달리다가 급정거하게 된다.  S3-A 에서
    위반의 71.4% 가 정지(40.0%)와 착지 스냅(31.4%)이었던 것이 그 결과다.

    **인자 이름이 `d_target` 이지만 목표까지의 거리만 받는 것은 아니다.**
    S3-A 3차부터 호출부가 `min(d_target, d_geom, d_dep)` 를 넘긴다 — 가장 가까운
    하드 제약까지의 자유 거리다.  A' 는 목표만 봐서 기하·의존성 대기 앞에서의
    급정거를 전혀 막지 못했다 (S3-A" §5-4).
    """
    if not math.isfinite(A_MAX):
        return math.inf
    return math.sqrt(2.0 * A_MAX * max(d_target, 0.0))


def _candidates(v_pref: np.ndarray, v_max: float,
                v_prev: Optional[np.ndarray] = None,
                dv_max: float = math.inf) -> np.ndarray:
    """Polar fan of candidate velocities around the preferred direction.

    `v_prev` / `dv_max` (S3-A): 가속도 상한을 후보 집합 자체에 넣는다.
    한 스텝에 바꿀 수 있는 속도는 `dv_max = a_max * dt` 까지다.

    **클리핑을 고른 이유** (탈락시키지 않는 이유):

    * 탈락시키면 남는 후보 수가 상황마다 달라지고, 최악의 경우 `v_pref` 하나만
      남거나 **하나도 안 남는다**.  하나도 안 남으면 `step` 의 하드 필터와
      구별할 수 없는 정지가 되어, "기하적으로 막혔다" 와 "가속이 모자랐다" 가
      같은 증상으로 뭉개진다.
    * 클리핑은 후보 수(65)를 상황과 무관하게 유지한다.  방향의 다양성이 그대로
      남아 `argmin` 이 고를 것이 항상 있고, 비용 지형이 스텝마다 크기를 바꾸지
      않는다.
    * 대가는 서로 다른 후보가 같은 값으로 뭉갤 수 있다는 것이다.  뭉개진 후보는
      비용도 같으므로 `argmin` 은 그중 첫 번째를 고른다 — 결정론적이고, 중복을
      제거하지 않으므로 인덱스 대응도 흐트러지지 않는다.

    `dv_max` 가 유한하지 않으면 **아무것도 하지 않는다.**  그래서 `a_max=inf`
    에서 S2 결과가 비트 단위로 재현된다 — 구현이 옳은지 확인하는 유일한 수단이다.
    """
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
    if v_prev is not None and math.isfinite(dv_max):
        d = cand - v_prev[None, :]
        nrm_d = np.linalg.norm(d, axis=1)
        over = nrm_d > dv_max
        # 넘는 것만 손댄다.  `v_prev + (c - v_prev)` 는 부동소수에서 `c` 와
        # 정확히 같지 않으므로, 전부를 다시 조립하면 제한에 걸리지도 않은
        # 후보까지 마지막 자리가 흔들린다.
        if np.any(over):
            scaled = v_prev[None, :] + d * (dv_max / np.maximum(nrm_d, 1e-30))[:, None]
            cand = np.where(over[:, None], scaled, cand)
    return cand


def multistep_free_steps(scene: SceneCache, i: int, pos_i: np.ndarray,
                         cand: np.ndarray, dt: float, dv_max: float,
                         enabled: bool = MULTISTEP_LOOKAHEAD) -> Tuple[np.ndarray, int]:
    """정지 가능한 감속 궤적이 전부 자유로운 후보만 남긴다 (S3-B2).

    첫 세그먼트는 현재 후보 `v_0`로, 이후에는 매 스텝 방향을 유지한 채
    `dv_max`만큼 감속한 `v_k`로 검사한다. 방향 전환 가능성을 이용하지 않는
    보수적 근사다. `a_max=inf` (따라서 `dv_max=inf`)에서는 기존의 정확히 한
    번 `free_steps` 호출로 돌아가므로 S2 재현 경로를 바꾸지 않는다.

    반환값의 둘째 항은 실제 `free_steps` 호출 횟수다. 배치 계측에서 비용을
    분리해 보기 위한 관측치이며 제어에는 되먹이지 않는다.
    """
    step_pts = pos_i[None, :] + cand * dt
    if not enabled or not math.isfinite(dv_max):
        return scene.free_steps(i, pos_i, step_pts), 1

    speed = np.linalg.norm(cand, axis=1)
    # 속력이 0인 후보도 현행처럼 한 스텝(제자리 세그먼트)을 검사한다.
    depth = np.maximum(1, np.ceil(speed / dv_max).astype(int))
    keep = np.ones(len(cand), bool)
    seg_start = np.broadcast_to(pos_i, cand.shape).copy()
    seg_vel = cand.copy()
    calls = 0
    for k in range(int(depth.max(initial=1))):
        active = keep & (depth > k)
        if not np.any(active):
            break
        end = seg_start[active] + seg_vel[active] * dt
        ok = scene.free_steps(i, seg_start[active], end)
        calls += 1
        active_idx = np.flatnonzero(active)
        keep[active_idx[~ok]] = False
        survived = active_idx[ok]
        if len(survived):
            seg_start[survived] = end[ok]
            sp = np.linalg.norm(seg_vel[survived], axis=1)
            scale = np.maximum(0.0, 1.0 - dv_max / np.maximum(sp, 1e-30))
            seg_vel[survived] *= scale[:, None]
    return keep, calls


# --------------------------------------------------------------------------- #
#  One synchronous timestep of the whole team
# --------------------------------------------------------------------------- #
def new_stats() -> Dict[str, float]:
    """`step` 이 채우는 계측 누산기 (S3-A).

    `TeamState` 에 필드를 더하지 않는 이유는 그 스키마가 계약으로 동결되어
    있기 때문이다 (tests/test_contract.py).  계약 문서가 지시하는 대로 별도의
    딕셔너리로 실어 나른다.

        steps                (에이전트 x 스텝) 표본 수
        n_amax_violations    |dv|/dt 가 a_max 를 넘은 표본 수
        max_accel            관측된 최대 |dv|/dt.  a_max=inf 에서도 의미가 있다 —
                             "지금 컨트롤러가 실제로 얼마나 급하게 움직이는가"
        n_project_active     안전 투영이 명령을 **바꾼** 표본 수.  "안전이 계획을
                             얼마나 자주 덮어쓰는가" 에 답한다

        n_blocked            후보 65개가 **전부** 기하 필터에 걸려 강제로 선
                             표본 수.  "가속 제한 때문에 갈 곳이 없어졌는가" 를
                             본다 — a_max 를 넣으면 후보가 이전 속도 주변으로
                             모이므로 이 값이 늘 수 있다.

    위반은 **원인별로 갈라 센다** (넷은 서로 배타적이고 합이 n_amax_violations).
    원인이 다르면 대응도 다르기 때문이다.

        n_amax_viol_stop     dwell 진입 / 도착으로 속도가 0 으로 강제된 것.
                             HOI 는 "waypoint 에서 d 초 정지" 라는 **과업 정의**라
                             감속 구간이 없다.  회피 실패가 아니다.
        n_amax_viol_block    후보가 전멸해 강제로 선 것.  기하가 원인이지
                             컨트롤러의 선택이 아니다.
        n_amax_viol_snap     도착 직전의 정확 착지 대입이 넘긴 것.
                             W/G 에 정확히 내려앉으려고 속도를 갈아끼우는 자리다.
        n_amax_viol_project  `_project_safe` 가 넘긴 것.  **이것이 S3 §A-2 가
                             재라고 한 숫자다** — 투영은 일부러 a_max 로 제한하지
                             않으므로 (제한하면 충돌 방지가 깨진다) 그 대가가
                             여기 쌓인다.  크면 "물리적으로 실현 불가능한 회피에
                             의존하고 있다" 는 뜻이다.
    """
    return {"steps": 0.0, "n_amax_violations": 0.0,
            "n_amax_viol_stop": 0.0, "n_amax_viol_block": 0.0,
            "n_amax_viol_snap": 0.0, "n_amax_viol_project": 0.0,
            "n_project_active": 0.0, "n_blocked": 0.0,
            "n_free_steps_calls": 0.0,
            "max_accel": 0.0}


def step(problem: Problem, scene: SceneCache, routes: Sequence[Route],
         st: TeamState, alpha: np.ndarray, speed_scale: np.ndarray,
         side_bias: np.ndarray,
         stats: Optional[Dict[str, float]] = None,
         trace: Optional[List[dict]] = None) -> TeamState:
    """Advance the team by one `dt`.

    Parameters
    ----------
    alpha       : (n, n) responsibility share.  `alpha[i, j]` in (0, 1] is how
                  much of the i-j avoidance agent `i` takes on.  0.5/0.5 is
                  classic RVO; larger means "I get out of the way".
    speed_scale : (n,) multiplier on each agent's preferred speed.
    side_bias   : (n,) in {-1, 0, +1}; which side the agent prefers to pass on.
    stats       : 주면 가속도 계측을 누적한다 (S3-A).  `new_stats()` 참조.
                  `None` 이면 아무 일도 하지 않는다 — 계측이 동작을 바꾸지 않는다.
    trace       : 주면 스텝마다 관측을 그대로 append 한다 — 투영 전/후 명령,
                  각 에이전트의 `v_pref` / `v_lim`, 후보가 전멸했는지(`forced`),
                  의존성 대기에 걸려 있었는지(`dep_hold`).  사고 한 건을 뜯어볼
                  때만 쓴다 (`scripts/dump_incident.py`, `scripts/diag_wipeout.py`).
                  배치에서는 항상 `None` 이라 비용이 0 이고, 기록은 **관측일 뿐
                  제어에 되먹임되지 않는다** (tests/test_amax.py 가 확인).
    """
    n, dt = problem.n, problem.dt
    dv_max = A_MAX * dt
    #: 후보가 전멸해 강제로 선 에이전트.  위반의 원인을 가르는 데 쓴다.
    forced = np.zeros(n, bool)
    if trace is not None:
        # 계측용 기록장.  `trace` 가 None 이면 만들지도 않는다.
        tr_pref = np.full((n, 2), np.nan)
        tr_vlim = np.full(n, np.nan)
        tr_hold = np.zeros(n, bool)
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
        # -- dependency event-hold (D5): 이 두 줄을 v_pref 앞으로 올렸다.
        # 값은 `phase` 와 `d_target` 만 쓰므로 순서를 바꿔도 결과가 같고, 아래
        # 제동 항이 `slack` 을 필요로 한다 (S3-A 3차).
        blocked = (phase[i] == PHASE_TO_WP) and not pred_done[i]
        slack = max(0.0, d_target - DEP_STANDOFF) if blocked else math.inf

        # -- 제동 대상 거리: 가장 가까운 하드 제약까지 (S3-A 3차) ------------ #
        # A' 는 `d_target`(waypoint/goal)만 봤다.  그래서 목표 근처 감속은
        # 고쳤지만 장애물과 standoff 경계 앞에서의 급정거는 그대로였다 —
        # 급정거형 전멸의 100% 가 제동에 필요한 여유(0.180 m)보다 좁은 곳에서
        # 일어났다 (S3-A" §5-3).
        #
        # `a_max=inf` 면 `brake_speed` 가 곧바로 inf 를 내므로 자유 거리를
        # **재지도 않는다** — S2 재현이 자명하고 비용도 0 이다.
        d_brake = d_target
        if math.isfinite(A_MAX):
            if BRAKE_LOOKAHEAD is not None:
                # 기하 제동.  S3-A 3차에서 무효로 측정되어 기본값은 꺼져 있다.
                # 제동은 **지금 속도**를 지우는 것이므로 진행 방향으로 본다.
                sp_prev = float(np.linalg.norm(vel[i]))
                if sp_prev > 1e-9:
                    d_brake = min(d_brake, brake_free_distance(
                        scene, i, pos[i], vel[i] / sp_prev, sp_prev, dv_max,
                        BRAKE_LOOKAHEAD))
            d_brake = min(d_brake, slack)

        d_aim = np.linalg.norm(aim - pos[i])
        if d_aim < 1e-9:
            v_pref = np.zeros(2)
        else:
            # 세 번째 항이 제동 항이다.  `a_max=inf` 면 inf 라 원래 식과 완전히
            # 같다.  `d_target/dt` 는 그대로 둔다 — 그것은 오버슛 방지이지
            # 제동이 아니다.
            v_pref = (aim - pos[i]) / d_aim * min(
                v_lim, d_target / dt, brake_speed(d_brake))

        # 기존 한 스텝 감속.  제동 항이 이 역할을 흡수하므로 **중복**이지만,
        # 둘 중 작은 쪽이 이기므로 동작은 안전한 방향이다.  제거는 별도 축이라
        # 그대로 둔다 (S3-A 3차 지시서).
        if blocked:
            if slack < v_lim * dt:
                v_pref = v_pref / max(np.linalg.norm(v_pref), 1e-9) * (slack / dt) \
                    if slack > 1e-9 else np.zeros(2)

        if trace is not None:
            tr_pref[i] = v_pref
            tr_vlim[i] = v_lim
            tr_hold[i] = blocked

        # S3-A: 후보를 이전 속도 주변 `dv_max` 안으로 클리핑한다.  `a_max=inf`
        # 면 `_candidates` 가 이 인자를 통째로 무시하므로 S2 와 완전히 같다.
        cand = _candidates(v_pref, v_lim, v_prev=vel[i], dv_max=dv_max)
        step_pts = pos[i][None, :] + cand * dt

        # -- hard filter: static geometry + dependency stand-off ------------ #
        # S3-B2: 정적 기하는 "이 후보로 들어간 뒤 최대 감속으로 설 수 있는가"를
        # 묻는다. 의존성 standoff 는 이 단계에서 한 스텝 판정을 유지한다.
        keep, free_steps_calls = multistep_free_steps(
            scene, i, pos[i], cand, dt, dv_max)
        if stats is not None:
            stats["n_free_steps_calls"] += free_steps_calls
        if blocked:
            keep &= np.linalg.norm(step_pts - target_pt[None, :], axis=1) >= \
                DEP_STANDOFF - 1e-9
        if not np.any(keep):
            # 후보가 전멸했다.  원본은 여기서 `v_cmd = 0` — 순항 1.20 m/s 에서
            # 12 m/s^2 이라 a_max 를 4배 위반한다.  계획서 3.3-7 의 fallback 은
            # "-a_max 로 0.67초 내 정지" 이므로 **사양과 구현이 어긋나 있었다**.
            #
            # 방향은 유지하고 크기만 `dv_max` 줄인다.  방향까지 바꾸면 그것이
            # 또 하나의 축이 된다.  `|v_prev| <= dv_max` 면 한 스텝에 설 수
            # 있으므로 0 이고, `a_max=inf` 면 항상 그 경우라 원본과 같다.
            #
            # 여러 스텝 연속으로 전멸이면 매 스텝 `dv_max` 씩 줄어 최대 4 스텝에
            # 선다.  그동안 계속 움직이는 것이 의도다 — 그 움직임의 안전은
            # 아래 `_project_safe` 가 받는다 (순서를 바꾸지 않았다).
            sp_prev = float(np.linalg.norm(vel[i]))
            if WIPEOUT_DECELERATE and sp_prev > dv_max:
                v_cmd[i] = vel[i] * (1.0 - dv_max / sp_prev)
            else:
                v_cmd[i] = 0.0
            forced[i] = True
            continue
        cand = cand[keep]
        step_pts = step_pts[keep]

        # -- soft cost: preference + reciprocal VO + obstacles + smoothness -- #
        cost = np.linalg.norm(cand - v_pref[None, :], axis=1)
        cost += W_TURN * np.linalg.norm(cand - vel[i][None, :], axis=1)

        # 이웃 VO + 정적 장애물 항은 safety/vo.py 로 내렸다.  `cost` 를 제자리에서
        # 갱신하는 것까지 원본과 같다 — 따로 모았다 더하면 부동소수 덧셈 순서가
        # 달라져 아래 argmin 이 동률 근처에서 뒤집힌다.
        add_vo_cost(problem, i, pos, vel, phase, cand,
                    scene.nearest_points(pos[i]), alpha, cost)

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
    v_pre = v_cmd.copy() if (stats is not None or trace is not None) else None
    v_cmd = _project_safe(problem, pos, v_cmd, phase)
    # numerical guard: never report a step longer than v_max * dt
    for i in range(n):
        sp = float(np.linalg.norm(v_cmd[i]))
        cap = problem.agents[i].v_max * (1.0 - 1e-9)
        if sp > cap:
            v_cmd[i] *= cap / sp

    # ---- a_max 계측 (S3-A) ------------------------------------------------- #
    # 여기서 재는 이유: `_project_safe` 와 속도 캡을 **모두 지난 최종 명령**이
    # 실제로 실행되는 값이다.  안전 투영은 일부러 a_max 로 제한하지 않으므로
    # (그렇게 하면 충돌 방지가 깨진다), 그 초과분은 숨기지 말고 세어야 한다.
    # 도착 직전의 정확 착지 대입(위 "exact arrival")도 a_max 를 우회할 수 있고,
    # 그것 역시 여기 잡힌다.
    if stats is not None:
        lim = A_MAX * (1.0 + 1e-9)
        dv = np.linalg.norm(v_cmd - vel, axis=1) / dt                # 최종
        dv_pre = np.linalg.norm(v_pre - vel, axis=1) / dt            # 투영 전
        over = dv > lim
        # 원인을 배타적으로 가른다 (정지 -> 착지 스냅 -> 안전 투영).
        # `_candidates` 의 클리핑 때문에 argmin 이 고른 후보 자체는 항상 a_max
        # 안에 있다.  따라서 투영 **전**에 넘었다면 원인은 정지 아니면 스냅이다.
        stopping = (phase == PHASE_DWELL) | (phase == PHASE_DONE)
        v_stop = over & stopping
        v_block = over & ~stopping & forced
        v_snap = over & ~stopping & ~forced & (dv_pre > lim)
        v_proj = over & ~stopping & ~forced & ~(dv_pre > lim)
        stats["steps"] += n
        stats["n_blocked"] += float(np.count_nonzero(forced))
        stats["n_amax_violations"] += float(np.count_nonzero(over))
        stats["n_amax_viol_stop"] += float(np.count_nonzero(v_stop))
        stats["n_amax_viol_block"] += float(np.count_nonzero(v_block))
        stats["n_amax_viol_snap"] += float(np.count_nonzero(v_snap))
        stats["n_amax_viol_project"] += float(np.count_nonzero(v_proj))
        stats["n_project_active"] += float(np.count_nonzero(
            np.linalg.norm(v_cmd - v_pre, axis=1) > 1e-12))
        if n:
            stats["max_accel"] = max(stats["max_accel"], float(dv.max()))

    if trace is not None:
        trace.append({"t": st.t, "pos": pos.copy(), "vel_prev": vel.copy(),
                      "v_pre": v_pre.copy(), "v_cmd": v_cmd.copy(),
                      "phase": phase.copy(), "forced": forced.copy(),
                      "v_pref": tr_pref, "v_lim": tr_vlim, "dep_hold": tr_hold})

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
