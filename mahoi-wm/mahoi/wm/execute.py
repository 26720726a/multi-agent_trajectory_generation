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

import csv
import os

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

    # -- 후보 단위 로깅 (기본 꺼짐) ------------------------------------------ #
    #: 켜면 재계획마다 live mode 전부를 한 행씩 남긴다.  끄면 루프 안에서
    #: `if cfg.log_modes:` 한 번만 보고 지나가므로 기존 동작·성능과 동일하다.
    log_modes: bool = False
    #: 로그를 쓸 디렉터리.  하위에 `wm_log/` 를 만들고 **프로세스마다 따로** 쓴다
    #: — 8워커가 한 파일에 append 하면 행이 섞인다.  합치는 것은 배치 뒤 별도 단계.
    log_dir: Optional[str] = None
    #: 조인 키.  호출부(bench/run.py)가 인스턴스 uid 를 넣어 준다.
    log_uid: str = ""

    # -- 후보 채점기 (P3-4) --------------------------------------------------- #
    #: "rollout" = 현행 (16개 전부 굴려 cost argmin).
    #: "nn_filter" = 굴린 뒤, **t=0 에서만** 학습된 분류기로 막힐 후보를 빼고
    #:   남은 것 중에서 고른다.  후보 수를 늘리지 않으므로 rollout 비용은 그대로다.
    #:   t=0 으로 제한하는 이유는 scorer.py 의 모듈 docstring 참조.
    scorer: str = "rollout"
    nn_model_path: str = ""
    nn_threshold: float = 0.5
    nn_device: str = "cpu"
    #: dep_mode 는 Problem 에 남지 않는다.  n=2 에서 chain/fork 가 구조적으로
    #: 같아 deps 로는 되짚을 수 없으므로 호출부가 알려주는 편이 정확하다.
    dep_mode: str = ""


#: 후보 단위 로그의 열.  `makespan_est` 는 **추정치**다 — 잘린 rollout 에
#: critical-path 나머지를 더한 값이라(worldmodel.py 의 rollout docstring) 실측
#: 완료 시각이 아니다.  실측은 `outcomes` 쪽 `team_time` 뿐이고, 둘의 차이를
#: 재는 것이 이 로그의 목적 중 하나다.
MODE_LOG_FIELDS = [
    "uid", "planner_seed", "replan_idx", "t", "mode_idx", "is_chosen",
    "mode_label", "routes", "yield_rank", "cautious", "split_side",
    "feasible", "makespan_est", "stalled", "total",
    "pos", "vel", "phase",
]

#: 인스턴스 단위 로그.  `(uid, planner_seed)` 로 위와 조인된다.
OUTCOME_LOG_FIELDS = [
    "uid", "planner_seed", "team_time", "status", "n_replans", "note",
]


def _log_paths(cfg: "WMConfig") -> Tuple[str, str]:
    """프로세스별 로그 파일 경로.  워커가 서로의 행을 덮어쓰지 않게 pid 로 가른다."""
    d = os.path.join(cfg.log_dir or ".", "wm_log")
    os.makedirs(d, exist_ok=True)
    pid = os.getpid()
    return (os.path.join(d, f"modes.{pid}.csv"),
            os.path.join(d, f"outcomes.{pid}.csv"))


def _append_rows(path: str, fields: Sequence[str], rows: Sequence[Sequence]) -> None:
    """인스턴스 하나 분량을 한 번에 덧붙인다.  파일이 없으면 헤더부터 쓴다."""
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(fields)
        w.writerows(rows)


def _fmt_xy(a) -> str:
    return ";".join(f"{x:.3f},{y:.3f}" for x, y in a)


def _fmt_i(a) -> str:
    return ",".join(str(int(v)) for v in a)


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
    #: nn_filter 가 t=0 에서 뺀 후보들 (label, 완주확률).  비어 있으면 안 뺐다.
    nn_filtered: List[Dict[str, object]] = field(default_factory=list)
    nn_kept: int = 0

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
    mode_log: List[List] = []          # 인스턴스 하나 분량만 들고 있는다
    replan_idx = 0
    nn_filtered: List[Dict[str, object]] = []
    nn_kept = 0
    while not st.all_done and st.t < max_steps:
        states.append(st.copy())
        if time.perf_counter() - t0 > cfg.time_budget_s:
            note = "wall-clock budget exhausted"
            break

        rolls = wm.rollouts(st, live, horizon=horizon)
        k, costs = planner.select(rolls, wm)

        # -- 학습된 분류기로 "막힐 후보" 빼기 (t=0 에서만) --------------------- #
        # rollout 은 이미 다 돌렸으므로 비용이 늘지 않고, cost 순위(total_rank)도
        # 여기서는 이미 알려져 있다 — 모델이 학습 때 본 것과 같은 입력이 된다.
        if cfg.scorer == "nn_filter" and replan_idx == 0:
            from .scorer import completion_prob
            prob = completion_prob(problem, st, live, costs, cfg)
            if prob is not None:
                ok = prob >= cfg.nn_threshold
                # 전부 걸러지면 필터를 무시한다 — 고를 것이 없어지면 안 된다.
                if ok.any() and not ok.all():
                    tot = np.array([c.total for c in costs], float)
                    k = int(np.argmin(np.where(ok, tot, np.inf)))
                    nn_filtered = [{"mode": live[j].label(problem),
                                    "p": round(float(prob[j]), 4)}
                                   for j in np.flatnonzero(~ok)]
                    nn_kept = int(ok.sum())

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

        # -- 후보 단위 로그 -------------------------------------------------- #
        # 여기여야 하는 이유가 둘 있다.  (1) 위 히스테리시스가 k 를 되돌릴 수
        # 있으므로 그 전에 적으면 `is_chosen` 이 틀린다.  (2) 바로 아래에서
        # `live` 가 keep_modes 로 잘리는데, `costs` 는 자르기 **전**의 live 와
        # 1:1 이다 — 자른 뒤에 적으면 mode 와 cost 가 어긋난다.
        if cfg.log_modes:
            pos, vel, phase = _fmt_xy(st.pos), _fmt_xy(st.vel), _fmt_i(st.phase)
            for j, m in enumerate(live):
                c = costs[j]
                mode_log.append([
                    cfg.log_uid, cfg.seed, replan_idx, st.t * dt, j, j == k,
                    m.label(problem), _fmt_i(m.routes), _fmt_i(m.yield_rank),
                    m.cautious, m.split_side,
                    c.feasible, round(c.makespan, 4), c.stalled,
                    round(c.total, 4), pos, vel, phase,
                ])
        replan_idx += 1

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
                # deadlock 이라고 부른다.  실제 메커니즘은 후보 속도가 전부
                # 기각되어(controller.py:378-380) _project_safe 의 최종 fallback
                # 인 "움직일 수 있는 전원 정지"(controller.py:495-498)로 수렴하는
                # 것이다 — 서로 양보하며 계속 움직이는 livelock 이 아니다.
                note = "deadlock: no team progress"
                break

    if not segments:
        raise RuntimeError("executor produced no motion")

    traj = concat(segments, method="world model + planner")
    traj.extra["v_max"] = [a.v_max for a in problem.agents]
    traj.runtime = time.perf_counter() - t0
    traj.feasible = bool(np.all(st.phase == PHASE_DONE))
    traj.note = note or (f"{len(switches)} decisions, "
                         f"{sum(1 for s in switches if s['switched'])} switches")

    if cfg.log_modes:
        modes_path, out_path = _log_paths(cfg)
        _append_rows(modes_path, MODE_LOG_FIELDS, mode_log)
        _append_rows(out_path, OUTCOME_LOG_FIELDS, [[
            cfg.log_uid, cfg.seed, round(traj.team_time, 3),
            "ok" if traj.feasible else f"unfinished:{note or '?'}"[:60],
            replan_idx, (note or "")[:120],
        ]])

    return WMResult(traj=traj, switches=switches, first_table=first_table,
                    first_rollouts=first_rolls, modes=modes, states=states,
                    live_modes=live, world_model=wm,
                    runtime=traj.runtime, note=note,
                    nn_filtered=nn_filtered, nn_kept=nn_kept)


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
