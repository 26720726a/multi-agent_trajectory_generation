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

from config import PHYSICS, PLANNER
from .control import PHASE_DONE, TeamState, initial_state
from .planner import Cost, CostWeights, Planner, cost_table  # noqa: F401
from .traj import Trajectory, concat
from .world import Problem
from .worldmodel import PlanMode, Rollout, WorldModel


@dataclass
class WMConfig:
    # 기본값의 출처는 config/planner.py 한 곳이다.  원본에서는 같은 숫자가
    # 여기(51-62)와 프리셋 3종(320-330), scripts/run_wm_experiments.py:43 에
    # 흩어져 있었다 (S0 §C).  값은 전부 원본 그대로.
    horizon_s: Optional[float] = PLANNER.horizon_s  # imagination depth; None = to completion
    replan_s: float = PLANNER.replan_s          # how much of the chosen future we commit
    max_modes: int = PLANNER.max_modes          # rollouts generated at t = 0
    keep_modes: int = PLANNER.keep_modes        # live pool after the first selection
    k_routes: int = PLANNER.k_routes            # geometric route candidates per leg
    switching: bool = True             # re-select mid-flight?
    seed: int = 0
    max_sim_s: float = PLANNER.max_sim_s        # simulated-time cap (safety net)
    time_budget_s: float = PLANNER.time_budget_s  # wall-clock cap
    stall_patience: int = PLANNER.stall_patience  # executed steps without progress
    switch_margin_rel: float = PLANNER.switch_margin_rel  # hysteresis: relative gain
    switch_margin_abs: float = PLANNER.switch_margin_abs  # hysteresis: absolute gain

    # -- 후보 단위 로깅 (기본 꺼짐) ------------------------------------------ #
    #: 켜면 재계획마다 live mode 전부를 한 행씩 남긴다.  끄면 루프 안에서
    #: `if cfg.log_modes:` 한 번만 보고 지나가므로 기존 동작·성능과 동일하다.
    log_modes: bool = False
    #: 로그를 쓸 디렉터리.  하위에 `wm_log/` 를 만들고 **프로세스마다 따로** 쓴다
    #: — 8워커가 한 파일에 append 하면 행이 섞인다.  합치는 것은 배치 뒤 별도 단계.
    log_dir: Optional[str] = None
    #: 조인 키.  호출부(bench/run.py)가 인스턴스 uid 를 넣어 준다.
    log_uid: str = ""

    # -- 후보 채점기 -------------------------------------------------------- #
    # 원본에는 `scorer` / `nn_model_path` / `nn_threshold` / `nn_device` 가 있어
    # `scorer="nn_filter"` 일 때 `mahoi/wm/scorer.py`(torch)를 불렀다.  S2 는
    # scorer.py 를 이관하지 않으므로(미확인 분류) 그 네 필드와 호출부를 함께
    # 제거했다.  남은 것은 현행 기본 경로인 "rollout" — 16개를 전부 굴려
    # cost argmin — 하나뿐이며, 기본값으로 돌던 동작은 바뀌지 않는다.
    # 자세한 것은 reports/S2_migration.md §3 (묶음 4).
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
    #: S3-C 용.  **모든 재계획**에서 채점한 후보 수와, 그중 `Cost.stalled` 가
    #: True 였던 수.  `first_table` 은 t=0 한 번뿐이라 "정체 판정이 언제부터
    #: 살아나는가" 를 볼 수 없다 — 그 질문에 답하려면 루프 전체를 세야 한다.
    #:
    #: horizon_steps <= stall_window 면 rollout 이 정체 판정에 도달하기 전에
    #: 지평선이 먼저 끝나므로 이 값은 **구조적으로 0** 이다 (h=4.0 -> 40 스텝
    #: < 45).  h 를 올리면 0 이 아니게 되고, 그 시점이 성능 변화와 겹치는지가
    #: S3-C 가 보려는 것이다.
    n_evals: int = 0
    n_stalled_evals: int = 0
    #: S5-1: 각 재계획의 후보 생성·rollout·비용·선택 구간만 잰 시간(ms).
    #: 경로 라이브러리 생성, 실행 구간 재시뮬레이션, 안전 검증은 포함하지 않는다.
    replan_ms: List[float] = field(default_factory=list)
    #: S5-2: 후보 생성 전 현재 상태에서 계산한 관측 전용 클러스터 기록.
    cluster_records: List[Dict[str, object]] = field(default_factory=list)

    @property
    def n_switches(self) -> int:
        return sum(1 for s in self.switches if s["switched"])


def _sum_accel(segments: Sequence[Trajectory]) -> Dict[str, float]:
    """실행 구간들의 가속도 계측을 합친다.  누계는 더하고, 최대는 최대를 취한다."""
    from .control import new_stats
    out = new_stats()
    for seg in segments:
        st = seg.extra.get("accel")
        if not st:
            continue
        for k in ("steps", "n_amax_violations", "n_amax_viol_stop",
                  "n_amax_viol_block", "n_amax_viol_snap",
                  "n_amax_viol_project", "n_project_active", "n_blocked",
                  "n_free_steps_calls"):
            out[k] += float(st.get(k, 0.0))
        out["max_accel"] = max(out["max_accel"], float(st.get("max_accel", 0.0)))
    return out


def interaction_clusters(pos: np.ndarray, phase: np.ndarray, radius: float) -> List[List[int]]:
    """현재 위치의 열린 반경 그래프 연결 성분.

    완료 agent는 움직이는 상호작용 대상이 아니므로 제외한다. 다만 controller의
    기하 필터에는 그대로 남아 물리적 장애물 역할을 계속한다.
    """
    live = [i for i, p in enumerate(phase) if p != PHASE_DONE]
    seen, out = set(), []
    for root in live:
        if root in seen:
            continue
        comp, todo = [], [root]
        seen.add(root)
        while todo:
            i = todo.pop()
            comp.append(i)
            for j in live:
                if j not in seen and np.linalg.norm(pos[i] - pos[j]) < radius:
                    seen.add(j); todo.append(j)
        out.append(sorted(comp))
    return out


def run_wm_planner(problem: Problem, cfg: Optional[WMConfig] = None,
                   weights: Optional[CostWeights] = None,
                   verbose: bool = False,
                   trace: Optional[List[dict]] = None) -> WMResult:
    """`trace` 를 주면 **실행에 옮긴 구간만** 스텝별로 기록한다.

    상상만 하고 버린 rollout 은 기록하지 않는다 — 사고를 뜯어볼 때 알고 싶은
    것은 "실제로 무엇이 실행됐는가" 이고, 후보 16개를 전부 남기면 그 답이
    파묻힌다.  `scripts/dump_incident.py` 가 이것을 쓴다.
    """
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
    n_evals = n_stalled_evals = 0
    replan_ms: List[float] = []
    cluster_records: List[Dict[str, object]] = []
    while not st.all_done and st.t < max_steps:
        states.append(st.copy())
        if time.perf_counter() - t0 > cfg.time_budget_s:
            note = "wall-clock budget exhausted"
            break

        # S5-2: 후보를 만들기 전의 현재 위치만 쓴다. 결과는 관측용이며,
        # live/mode/state 어느 것도 바꾸지 않아 계획 동작은 불변이다.
        clusters = interaction_clusters(st.pos, st.phase, PHYSICS.interact_cluster_radius)
        cluster_records.append({"t": st.t * dt, "phase": st.phase.copy(),
                                "pos": st.pos.copy(), "clusters": clusters})

        # S5-1: 이 경계는 후보 생성부터 mode 선택까지다. 초기 A* 경로 계획은
        # while 전에 끝났고, 아래 실행 rollout/검증도 경계 밖이라 포함하지 않는다.
        t_replan = time.perf_counter()
        rolls = wm.rollouts(st, live, horizon=horizon)
        k, costs = planner.select(rolls, wm)
        n_evals += len(costs)
        n_stalled_evals += sum(1 for c in costs if c.stalled)

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
        replan_ms.append((time.perf_counter() - t_replan) * 1e3)

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
            full = wm.rollout(st, chosen, horizon=None, max_steps=max_steps,
                              trace=trace)
            segments.append(full.traj)
            st = full.end_state
            break

        seg = wm.rollout(st, chosen, horizon=exec_steps, max_steps=max_steps,
                         trace=trace)
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
    # S3-A: **실행된** 구간의 가속도 계측만 합친다.  상상만 하고 버린 rollout 은
    # 세지 않는다 — 실현 불가능한 가속에 의존했는지를 묻는 질문이므로 실제로
    # 실행된 명령만이 답이다.  `concat` 은 마지막 조각의 extra 만 남기므로
    # (traj.py) 여기서 명시적으로 더한다.
    traj.extra["accel"] = _sum_accel(segments)
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
                    n_evals=n_evals, n_stalled_evals=n_stalled_evals,
                    replan_ms=replan_ms, cluster_records=cluster_records)


# --------------------------------------------------------------------------- #
#  Named configurations used by the experiments
# --------------------------------------------------------------------------- #
def cfg_receding(**kw) -> WMConfig:
    """Default: bounded imagination + mid-flight re-selection."""
    return WMConfig(horizon_s=PLANNER.horizon_s, replan_s=PLANNER.replan_s,
                    switching=True, **kw)


def cfg_reselect_full(**kw) -> WMConfig:
    """Ablation: imagination runs to completion, but still re-selects."""
    return WMConfig(horizon_s=None, replan_s=PLANNER.replan_s,
                    switching=True, **kw)


def cfg_static(**kw) -> WMConfig:
    """Ablation: imagine every future once at t = 0, then commit to one."""
    return WMConfig(horizon_s=None, replan_s=PLANNER.replan_s,
                    switching=False, **kw)
