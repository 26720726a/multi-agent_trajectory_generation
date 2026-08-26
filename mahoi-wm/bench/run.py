#!/usr/bin/env python3
"""대량 실행 — 난이도 격자 × planner seed × 방법 을 돌려 한 줄씩 CSV 로 쌓는다.

    python -m bench.run --config bench/configs/smoke.json
    python -m bench.run --config bench/configs/scale.json --timeout 600

설계 원칙 세 가지.

* **한 인스턴스의 실패가 전체를 죽이지 않는다.**  예외는 잡아서 `status` 열에
  기록한다.  대량 실행에서 중간에 죽으면 몇 시간을 잃는다.
* **한 인스턴스의 *멈춤* 도 전체를 죽이지 않는다.**  예외로 잡히지 않는 실패가
  있다 — `WMConfig.time_budget_s` 는 `execute.py` 의 메인 루프 *안에서만*
  검사되는데, `wm.sample_modes()` 는 그 루프에 들어가기 전에 전체 mode 조합을
  리스트로 만든다.  agent 6명이면 138,240개, 8명이면 30,965,760개다.  그 안에서
  멈추면 wall-clock cap 이 발동할 기회조차 없이 서거나 OOM 으로 죽는다.  그래서
  인스턴스를 **별도 프로세스**에서 돌리고 `--timeout` 초를 넘기면 죽인다.
  프로세스를 죽이는 것 말고는 확실한 방법이 없다.
* **모든 행이 스스로를 재현한다.**  seed·축 값·git commit·적용된 타임아웃이
  행마다 들어 있어, 나중에 이상한 점을 발견하면 그 한 줄만 보고 되살릴 수 있다.
  죽은 인스턴스도 행을 남긴다 — 조용히 사라지면 성공률의 분모가 줄어 실패가
  성공처럼 보인다.

두 개의 seed
------------
`instance_seed` 는 문제를, `planner_seed` 는 플래너의 모드 샘플링만 바꾼다
(`bench/generate.py` 의 모듈 docstring).  결정론적 method 에 planner_seed 를
곱하면 완전히 같은 행이 복제되어 분모만 부풀므로, 기준 seed 에서 한 번만 돈다.

  TODO(B1) 지금 부족한 것
    - 병렬 실행과 재개(append).  지금은 순차라 1,000 인스턴스가 하룻밤을 넘긴다
      (3단계).
    - 실패 원인 분류.  `unfinished:*` 를 deadlock / livelock / mode 집합에 답이
      없음 / cost 오판 으로 쪼개야 A 에게 넘길 작업 목록이 된다 (B6).  이번엔
      `res.note` 를 note 열에 그대로 실어 재료만 남긴다.
    - coordination A* 는 n>=4 에서 격자가 폭발한다.  확장 가능한 상한 baseline
      으로 교체해야 한다 (B5).
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.generate import (STOCHASTIC_METHODS, Axis, Instance,
                            axis_from_config, buildable, grid)
from mahoi import validate as V
from mahoi.coordination import (critical_path_bound, plan_coordination,
                                plan_sequential)
from mahoi.paths import build_prior
from mahoi.wm.execute import WMConfig, run_wm_planner

#: 실행 순서이자 CSV 안에서의 method 순서.
ALL_METHODS = ["lower_bound", "sequential", "coordination_astar", "wm_planner"]

FIELDS = [
    "uid", "instance_seed", "n_agents", "dep_mode", "size", "n_obstacles",
    "couple_prob", "couple_dist", "method", "status", "team_time", "flow_time",
    "distance", "wait", "valid", "agent_clearance", "obstacle_clearance",
    "dep_violations", "n_switches", "runtime_s", "lower_bound", "ratio_to_bound",
    "git_commit",
    # -- 2단계에서 추가.  기존 열 뒤에 붙인다 — 앞쪽이 밀리면 옛 CSV 와 나란히
    #    놓고 볼 수 없다. --
    "planner_seed", "timeout_s", "n_modes", "n_feasible_modes",
    "chosen_is_fastest_feasible", "note",
]

#: note 열의 상한.  자유 텍스트라 길이를 안 막으면 CSV 한 줄이 화면을 덮는다.
NOTE_MAX = 120

#: `chosen_is_fastest_feasible` 의 허용 오차.  check_wm.py 의 "(same)" 판정과
#: 같은 값이어야 두 리포트가 서로를 검산할 수 있다.
MAKESPAN_TOL = 1e-6

#: ROS 는 PYTHONPATH 에 자기 site-packages 를 밀어 넣는다.  거기 있는 패키지가
#: 프로젝트의 것을 가리면(대표적으로 yaml) 자식은 import 단계에서 죽고, 그 죽음은
#: "인스턴스가 어렵다" 와 구별되지 않는다.  워커는 항상 이걸 먼저 턴다.
ROS_PREFIX = "/opt/ros"

#: fork 는 이미 import 된 모듈을 물려받아 인스턴스당 재기동 비용이 없다.  이
#: 파일은 워커를 띄우기 전에 스레드를 만들지 않으므로 fork 가 안전하다.
_MP_START = "fork" if "fork" in mp.get_all_start_methods() else "spawn"

_POLL_S = 0.2          # 자식을 들여다보는 주기 (RSS 표본 간격이기도 하다)
_DRAIN_S = 0.5         # 죽기 직전에 큐에 넣었을 수 있으니 한 번 더 기다린다
_JOIN_S = 5.0          # kill 뒤 정리를 기다리는 시간

#: SIGKILL 로 죽었을 때 OOM 으로 볼 최고 RSS 의 기준 (전체 메모리 대비).
#: 이보다 낮으면 원인을 단정하지 않고 "killed" 로 두고 사유를 note 에 남긴다.
OOM_RSS_FRACTION = 0.5


def strip_ros_paths() -> List[str]:
    """`sys.path` 와 `PYTHONPATH` 에서 /opt/ros 항목을 제거하고, 제거한 것을 돌려준다.

    자식 프로세스는 부모의 `sys.path` 와 환경변수를 그대로 물려받는다.  환경변수
    까지 터는 이유는 3단계의 Pool 워커(및 그 자식)가 다시 물려받지 않게 하기
    위해서다.
    """
    removed = [p for p in sys.path if p.startswith(ROS_PREFIX)]
    if removed:
        sys.path[:] = [p for p in sys.path if not p.startswith(ROS_PREFIX)]
    env = os.environ.get("PYTHONPATH", "")
    if env:
        keep = [p for p in env.split(os.pathsep)
                if p and not p.startswith(ROS_PREFIX)]
        if keep:
            os.environ["PYTHONPATH"] = os.pathsep.join(keep)
        else:
            os.environ.pop("PYTHONPATH", None)
    return removed


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:                                    # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- #
#  행 만들기
# --------------------------------------------------------------------------- #
def _blank(inst: Instance, method: str, lb: Optional[float], commit: str,
           planner_seed: int, timeout_s: Optional[float]) -> Dict[str, object]:
    row = {k: "" for k in FIELDS}
    row.update(inst.row())
    row.update(method=method, status="ok", git_commit=commit,
               planner_seed=planner_seed,
               timeout_s="" if timeout_s is None else round(float(timeout_s), 1),
               lower_bound=None if lb is None else round(lb, 3))
    return row


def _fill(row: Dict[str, object], problem, sol, rep, runtime: float,
          lb: Optional[float]) -> Dict[str, object]:
    row.update(
        team_time=round(sol.team_time, 3), flow_time=round(sol.flow_time, 3),
        distance=round(sol.travel_distance, 3), wait=round(sol.total_wait, 3),
        runtime_s=round(runtime, 3))
    if rep is not None:
        row.update(valid=bool(rep.valid),
                   agent_clearance=round(float(rep.agent_clearance), 4),
                   obstacle_clearance=round(float(rep.obstacle_clearance), 4),
                   dep_violations=int(rep.n_dep_violations))
        if not rep.valid:
            row["status"] = "invalid"
    if lb:
        row["ratio_to_bound"] = round(sol.team_time / lb, 4)
    return row


def wm_mode_stats(res) -> Dict[str, object]:
    """t=0 의 cost 표에서 B2 의 증거 두 개를 뽑는다.

    `first_table` 은 플래너가 **첫 선택을 할 때 실제로 본** 표다.  거기서

      * `n_feasible_modes` : `feasible=True` 인 rollout 수
      * chosen             : `total` 이 최소인 것 — `Planner.select()` 가
                             `argmin(total)` 이므로 플래너가 실제로 고른 것과 같다
      * fastest_feasible   : feasible 중 `makespan` 이 최소인 것

    을 구하고, 둘의 makespan 이 `MAKESPAN_TOL` 이내로 같으면 "빠른 걸 골랐다"
    이다.  다르면 soft 항(clearance·deviation 등)이 시간을 사고 있다는 뜻이고,
    그게 바로 B2 가 재려는 것이다.

    `scripts/check_wm.py` 가 5종 시나리오에서 찍는 "(same)" 과 같은 계산이다.
    거기서는 5개 시나리오뿐이지만, 이 열은 격자 전체에서 같은 질문에 답한다.
    """
    table = list(getattr(res, "first_table", ()) or ())
    out: Dict[str, object] = {"n_modes": len(table), "n_feasible_modes": "",
                              "chosen_is_fastest_feasible": ""}
    if not table:
        return out
    feas = [c for c in table if c.feasible]
    out["n_feasible_modes"] = len(feas)
    if feas:
        chosen = min(table, key=lambda c: c.total)
        fastest = min(c.makespan for c in feas)
        out["chosen_is_fastest_feasible"] = bool(
            abs(chosen.makespan - fastest) < MAKESPAN_TOL)
    return out


def methods_for_seed(methods: Sequence[str], planner_seed: int,
                     deterministic_seed: int = 0) -> List[str]:
    """이 planner_seed 에서 실제로 돌릴 method.

    `lower_bound` / `sequential` / `coordination_astar` 는 결정론적이라
    planner_seed 를 곱하면 완전히 같은 행이 복제된다.  성공률의 분모만 부풀고
    분자도 같이 부풀어, 실패율이 seed 개수만큼 희석된다.  그래서 기준 seed 에서
    한 번만 돈다.  어느 method 가 확률적인지는 `generate.STOCHASTIC_METHODS` 가
    정한다 — 표를 두 군데 두면 언젠가 어긋난다.

    기준 seed 를 0 이 아니라 인자로 받는 이유: `planner_seeds` 에 0 이 없는
    config 를 주면 baseline 이 통째로 사라진다.  호출부는 첫 seed 를 넘긴다.
    """
    return [m for m in ALL_METHODS
            if m in methods
            and (m in STOCHASTIC_METHODS or planner_seed == deterministic_seed)]


def wm_config_without_seed(wm_cfg: Dict) -> Tuple[Dict, Optional[int]]:
    """`wm_config` 에서 "seed" 를 떼어낸다.  떼어낸 값을 함께 돌려준다.

    플래너 seed 는 `axis.planner_seeds` 가 정한다.  config 에도 seed 가 있으면
    어느 쪽이 이겼는지 CSV 만 보고는 알 수 없다 — 그러면 "seed 편차" 열이
    무엇의 편차인지 말할 수 없게 된다.
    """
    clean = dict(wm_cfg)
    return clean, clean.pop("seed", None)


# --------------------------------------------------------------------------- #
#  인스턴스 하나 (워커의 본체 — picklable 한 최상위 함수여야 한다)
# --------------------------------------------------------------------------- #
def run_instance(inst: Instance, wm_cfg: Dict, methods: List[str], commit: str,
                 planner_seed: int = 0, astar_max_agents: int = 3,
                 deterministic_seed: int = 0,
                 timeout_s: Optional[float] = None) -> List[Dict[str, object]]:
    """인스턴스 하나 × method 들을 돌려 행 목록을 만든다.

    인자와 반환값이 전부 picklable 하다 (3단계에서 `multiprocessing.Pool` 에
    그대로 넘긴다).  `timeout_s` 는 여기서 쓰이지 않고 행에 기록만 된다 — 실제
    타임아웃은 `run_instance_guarded()` 가 프로세스 밖에서 건다.
    """
    wm_cfg, ignored = wm_config_without_seed(wm_cfg)
    if ignored is not None:
        print(f'[bench.run] 경고: wm_config 의 "seed": {ignored} 를 무시한다 — '
              f"플래너 seed 는 planner_seeds 가 정한다 (지금 {planner_seed}).",
              file=sys.stderr)

    todo = methods_for_seed(methods, planner_seed, deterministic_seed)

    problem = buildable(inst)
    if problem is None:
        # 인스턴스가 애초에 존재하지 않는다.  method 와 무관하므로 한 줄만
        # 남기고, planner_seed 마다 복제하지는 않는다.
        if planner_seed != deterministic_seed:
            return []
        r = _blank(inst, "-", None, commit, planner_seed, timeout_s)
        r["status"] = "ungeneratable"
        return [r]

    try:
        tracks, _ = build_prior(problem)
        lb = critical_path_bound(problem, tracks)["lb_team_time"]
    except Exception as exc:                             # noqa: BLE001
        if planner_seed != deterministic_seed:
            return []
        r = _blank(inst, "-", None, commit, planner_seed, timeout_s)
        r["status"] = f"no_prior:{type(exc).__name__}"
        r["note"] = str(exc)[:NOTE_MAX]
        return [r]

    rows: List[Dict[str, object]] = []

    if "lower_bound" in todo:
        r = _blank(inst, "lower_bound", lb, commit, planner_seed, timeout_s)
        r.update(team_time=round(lb, 3), ratio_to_bound=1.0, runtime_s=0.0)
        rows.append(r)

    if "sequential" in todo:
        r = _blank(inst, "sequential", lb, commit, planner_seed, timeout_s)
        t0 = time.perf_counter()
        try:
            sol = plan_sequential(problem, tracks)
            if sol.feasible:
                _fill(r, problem, sol, V.validate(problem, sol),
                      time.perf_counter() - t0, lb)
            else:
                r["status"] = "no_solution"
        except Exception as exc:                         # noqa: BLE001
            r["status"] = f"error:{type(exc).__name__}"
            r["note"] = str(exc)[:NOTE_MAX]
        rows.append(r)

    if "coordination_astar" in todo:
        r = _blank(inst, "coordination_astar", lb, commit, planner_seed, timeout_s)
        if problem.n > astar_max_agents:
            r["status"] = "skipped:lattice_too_large"    # B5 가 해결할 자리
        else:
            t0 = time.perf_counter()
            try:
                sol = plan_coordination(problem, tracks)
                if sol.feasible:
                    _fill(r, problem, sol, V.validate(problem, sol),
                          time.perf_counter() - t0, lb)
                else:
                    r["status"] = "no_solution"
            except MemoryError:
                r["status"] = "skipped:lattice_too_large"
            except Exception as exc:                     # noqa: BLE001
                r["status"] = f"error:{type(exc).__name__}"
                r["note"] = str(exc)[:NOTE_MAX]
        rows.append(r)

    if "wm_planner" in todo:
        r = _blank(inst, "wm_planner", lb, commit, planner_seed, timeout_s)
        t0 = time.perf_counter()
        try:
            res = run_wm_planner(problem, WMConfig(**wm_cfg, seed=planner_seed))
            rep = V.validate(problem, res.traj)
            _fill(r, problem, res.traj, rep, time.perf_counter() - t0, lb)
            r["n_switches"] = res.n_switches
            r.update(wm_mode_stats(res))
            # B6 이 분류할 재료.  "livelock" / "wall-clock" 등이 들어 있다.
            r["note"] = (res.note or "")[:NOTE_MAX]
            if not res.traj.feasible:
                r["status"] = f"unfinished:{res.note or 'unknown'}"[:60]
        except Exception as exc:                         # noqa: BLE001
            r["status"] = f"error:{type(exc).__name__}"
            r["note"] = str(exc)[:NOTE_MAX]
        rows.append(r)

    return rows


# --------------------------------------------------------------------------- #
#  하드 타임아웃 — 프로세스를 죽이는 것 말고는 확실한 방법이 없다
# --------------------------------------------------------------------------- #
def _worker(q, inst, wm_cfg, methods, commit, planner_seed, astar_max_agents,
            deterministic_seed, timeout_s) -> None:
    """자식 프로세스의 진입점.  결과를 큐에 하나 넣고 끝난다."""
    strip_ros_paths()
    try:
        rows = run_instance(inst, wm_cfg, methods, commit,
                            planner_seed=planner_seed,
                            astar_max_agents=astar_max_agents,
                            deterministic_seed=deterministic_seed,
                            timeout_s=timeout_s)
        q.put(("ok", rows))
    except BaseException as exc:                         # noqa: BLE001
        # run_instance 가 method 별로 이미 잡는다.  여기까지 온 것은 import 실패나
        # MemoryError 처럼 인스턴스 하나를 통째로 무너뜨린 것이다.
        q.put(("error", f"{type(exc).__name__}: {exc}"))


def _peak_rss_mb(pid: int) -> float:
    """자식의 최고 RSS (MB).  못 읽으면 0.0 — "모른다" 는 뜻으로만 쓴다."""
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def _total_mem_mb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 20
    except (ValueError, OSError, AttributeError):
        return 0.0


def _signal_name(num: int) -> str:
    try:
        return signal.Signals(num).name
    except ValueError:
        return f"signal {num}"


def classify_death(over_time: bool, exitcode: Optional[int], elapsed: float,
                   timeout_s: float, peak_mb: float) -> Tuple[str, str]:
    """자식이 결과 없이 끝난 이유를 status 와 note 로 옮긴다.

    우리가 죽였으면 timeout 이다 — 이건 확실하다.  자기 혼자 SIGKILL 로 죽었으면
    OOM killer 가 유력하지만, 외부에서 죽였을 수도 있다.  그래서 최고 RSS 가
    전체 메모리의 `OOM_RSS_FRACTION` 을 넘었을 때만 "oom" 이라고 단정하고,
    아니면 "killed" 로 두되 사유를 note 에 적는다.  애매한 것을 확실한 척
    적으면, 나중에 그 열을 근거로 잘못된 결론을 내리게 된다.
    """
    if over_time:
        return "timeout", (f"{timeout_s:g}s 하드 타임아웃 초과 "
                           f"(경과 {elapsed:.1f}s, 최고 RSS {peak_mb:.0f}MB)")
    if exitcode is not None and exitcode < 0:
        sig = -exitcode
        if sig == int(signal.SIGKILL):
            total = _total_mem_mb()
            if total and peak_mb >= OOM_RSS_FRACTION * total:
                return "oom", (f"SIGKILL, 최고 RSS {peak_mb:.0f}MB / "
                               f"전체 {total:.0f}MB — OOM killer 로 본다")
            return "killed", (f"SIGKILL (최고 RSS {peak_mb:.0f}MB, "
                              f"경과 {elapsed:.1f}s) — OOM 인지 외부 종료인지 "
                              f"구별할 수 없다")
        return "killed", (f"{_signal_name(sig)} 로 죽었다 "
                          f"(경과 {elapsed:.1f}s, 최고 RSS {peak_mb:.0f}MB)")
    return "killed", (f"결과 없이 exit code {exitcode} 로 끝났다 "
                      f"(경과 {elapsed:.1f}s, 최고 RSS {peak_mb:.0f}MB)")


def run_instance_guarded(inst: Instance, wm_cfg: Dict, methods: List[str],
                         commit: str, planner_seed: int, timeout_s: float,
                         astar_max_agents: int = 3,
                         deterministic_seed: int = 0) -> List[Dict[str, object]]:
    """`run_instance` 를 별도 프로세스에서 돌리고, 넘치면 강제 종료한다.

    스레드나 시그널로는 안 된다.  `sample_modes()` 가 30,965,760개짜리 리스트를
    만드는 동안에는 파이썬 레벨에서 정중하게 끼어들 지점이 없고, 애초에 메모리를
    이미 다 먹은 뒤라 되돌릴 것도 없다.

    죽었을 때도 **돌릴 예정이던 method 마다 한 줄씩** 남긴다.  행을 안 남기면
    그 method 의 성공률 분모가 조용히 줄어, 가장 어려운 인스턴스가 통계에서
    빠지면서 성공률이 올라간다 — 정확히 반대로 읽힌다.
    """
    ctx = mp.get_context(_MP_START)
    q = ctx.Queue()
    proc = ctx.Process(
        target=_worker,
        args=(q, inst, wm_cfg, methods, commit, planner_seed, astar_max_agents,
              deterministic_seed, timeout_s),
        daemon=True)

    t0 = time.perf_counter()
    proc.start()

    payload, peak_mb = None, 0.0
    while True:
        try:
            payload = q.get(timeout=_POLL_S)
            break
        except queue.Empty:
            pass
        peak_mb = max(peak_mb, _peak_rss_mb(proc.pid))
        if not proc.is_alive():
            try:                       # 죽기 직전에 넣었을 수도 있다
                payload = q.get(timeout=_DRAIN_S)
            except queue.Empty:
                pass
            break
        if time.perf_counter() - t0 >= timeout_s:
            break

    over_time = payload is None and proc.is_alive()
    if proc.is_alive():
        proc.kill()
    proc.join(_JOIN_S)
    elapsed = time.perf_counter() - t0
    exitcode = proc.exitcode
    # 죽은 자식이 큐 락을 쥔 채 사라졌을 수 있다.  더 읽지 않으므로 붙잡지 않고 뗀다.
    q.cancel_join_thread()
    q.close()

    if payload is not None and payload[0] == "ok":
        return payload[1]

    if payload is not None:            # 워커가 스스로 보고한 파국
        status, note = "error:worker", str(payload[1])
    else:
        status, note = classify_death(over_time, exitcode, elapsed, timeout_s,
                                      peak_mb)

    todo = methods_for_seed(methods, planner_seed, deterministic_seed) or ["-"]
    rows = []
    for m in todo:
        r = _blank(inst, m, None, commit, planner_seed, timeout_s)
        r.update(status=status, runtime_s=round(elapsed, 3), note=note[:NOTE_MAX])
        rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
#  배치
# --------------------------------------------------------------------------- #
def planner_seeds_of(axes: Sequence[Axis]) -> Tuple[int, ...]:
    """격자들의 planner_seeds.  블록마다 다르면 합집합을 쓰고 그 사실을 알린다.

    `grid()` 가 여러 격자를 이어붙인 뒤에는 인스턴스가 어느 블록에서 왔는지
    알 수 없다.  블록마다 다른 seed 수를 쓰고 싶다면 그건 격자를 따로 돌릴
    일이다 — 여기서 조용히 한쪽을 고르면 절반이 안 돈다.
    """
    per_block = [tuple(ax.planner_seeds) for ax in axes]
    if len(set(per_block)) > 1:
        merged = tuple(sorted({s for blk in per_block for s in blk}))
        print(f"[bench.run] 경고: 격자마다 planner_seeds 가 다르다 {per_block}. "
              f"합집합 {merged} 로 전부 돌린다.", file=sys.stderr)
        return merged
    return per_block[0] if per_block else (0,)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="앞에서 N개 인스턴스만 (설정 확인용)")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="인스턴스 하나의 하드 상한(초).  넘기면 자식을 죽이고 "
                         "status=timeout 행을 남긴다 (기본 300)")
    args = ap.parse_args()

    strip_ros_paths()

    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    axes = axis_from_config(cfg["axis"])
    methods = cfg.get("methods", list(ALL_METHODS))
    wm_cfg, ignored_seed = wm_config_without_seed(cfg.get("wm_config", {}))
    planner_seeds = planner_seeds_of(axes)
    deterministic_seed = planner_seeds[0]

    if ignored_seed is not None:
        print(f'[bench.run] 경고: wm_config 의 "seed": {ignored_seed} 를 무시한다. '
              f"플래너 seed 는 axis.planner_seeds 가 정한다 "
              f"(이번 실행: {list(planner_seeds)}).", file=sys.stderr)

    out = args.out or os.path.join("bench", "runs", f"{cfg.get('name', 'run')}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    commit = git_commit()

    insts = list(grid(axes))
    if args.limit:
        insts = insts[:args.limit]
    total = len(insts) * len(planner_seeds)
    det = [m for m in ALL_METHODS if m in methods and m not in STOCHASTIC_METHODS]
    print(f"{cfg.get('name', 'run')}: {len(insts)} instances x "
          f"{len(planner_seeds)} planner seeds = {total} runs  (commit {commit})\n"
          f"  방법 {methods}\n"
          f"  결정론적 {det} 는 planner_seed={deterministic_seed} 에서만 돈다\n"
          f"  하드 타임아웃 {args.timeout:g}s / 인스턴스\n-> {out}\n", flush=True)

    t_start = time.perf_counter()
    n_rows = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        k = 0
        for inst in insts:
            for ps in planner_seeds:
                k += 1
                t0 = time.perf_counter()
                rows = run_instance_guarded(
                    inst, wm_cfg, methods, commit, planner_seed=ps,
                    timeout_s=args.timeout, deterministic_seed=deterministic_seed)
                for r in rows:
                    w.writerow(r)
                    n_rows += 1
                fh.flush()                # 중간에 죽어도 여기까지는 남는다
                wm = next((r for r in rows if r["method"] == "wm_planner"), None)
                tag = (f"{wm['team_time']}s {wm['status']}" if wm
                       else (rows[0]["status"] if rows else "-"))
                print(f"  [{k:>4}/{total}] {inst.uid:<34} ps={ps} {tag}"
                      f"   ({time.perf_counter() - t0:.1f}s)", flush=True)

    print(f"\n완료: {time.perf_counter() - t_start:.1f}s, {n_rows} rows  ->  {out}")
    print(f"분석: python -m bench.analyze --csv {out}")


if __name__ == "__main__":
    main()
