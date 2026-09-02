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
import multiprocessing.pool
import os
import queue
import signal
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

#: BLAS/OpenMP 는 **import 시점에** 코어 수만큼 스레드 풀을 만든다.  워커마다 그게
#: 생기면 `--workers N` 은 N배 빨라지는 대신 N×코어 개의 스레드가 서로를 밀어내
#: 오히려 느려진다.  그래서 numpy 가 딸려 오는 아래 mahoi import 보다 **위에서**
#: 묶는다 — 나중에 환경변수를 고쳐봐야 이미 만들어진 풀은 줄지 않는다.
#:
#: spawn 워커도 이 모듈을 다시 import 하며 이 줄을 지나므로 같은 보장을 받는다.
#: fork 워커는 이미 묶인 부모를 물려받는다.  `_pool_init` 이 한 번 더 세우는 것은
#: 확인용이지 이 줄을 대신하지는 못한다.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for _v in THREAD_VARS:
    os.environ[_v] = "1"

#: Track A(A3) 가 rollout 을 자기 Pool 로 병렬화할 때 읽기로 한 변수.  바깥(B)과
#: 안쪽(A) 이 각자 코어 수만큼 프로세스를 만들면 곱해져서 코어를 초과 구독하고,
#: 최악에는 서로의 완료를 기다리며 멈춘다.  바깥이 몇 개를 쓰는지 아는 쪽은
#: 여기뿐이므로 안쪽 몫을 여기서 정해 내려보낸다.  기본 1 = 안쪽은 병렬화하지 않는다.
INNER_WORKERS_VAR = "MAHOI_INNER_WORKERS"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.generate import (STOCHASTIC_METHODS, Axis, Instance,
                            axis_from_config, buildable, expected_count, grid)
from mahoi import validate as V
from mahoi.coordination import (critical_path_bound, plan_coordination,
                                plan_prioritized, plan_sequential)
from mahoi.paths import build_prior
from mahoi.wm.execute import WMConfig, run_wm_planner

#: 실행 순서이자 CSV 안에서의 method 순서.
ALL_METHODS = ["lower_bound", "sequential", "prioritized",
               "coordination_astar", "wm_planner"]

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


def clean_note(text: object) -> str:
    """note 를 한 줄로 접고 자른다.

    줄바꿈을 그대로 두면 CSV 한 행이 여러 줄이 되고, 그러면 재개가 "잘린 마지막
    행" 을 줄 단위로 알아볼 수 없게 된다.  예외 메시지에는 줄바꿈이 흔하다.
    """
    return " ".join(str(text or "").split())[:NOTE_MAX]

#: `chosen_is_fastest_feasible` 의 허용 오차.  check_wm.py 의 "(same)" 판정과
#: 같은 값이어야 두 리포트가 서로를 검산할 수 있다.
MAKESPAN_TOL = 1e-6

#: ROS 는 PYTHONPATH 에 자기 site-packages 를 밀어 넣는다.  거기 있는 패키지가
#: 프로젝트의 것을 가리면(대표적으로 yaml) 자식은 import 단계에서 죽고, 그 죽음은
#: "인스턴스가 어렵다" 와 구별되지 않는다.  워커는 항상 이걸 먼저 턴다.
ROS_PREFIX = "/opt/ros"

_BASE_CTX = mp.get_context(
    "fork" if "fork" in mp.get_all_start_methods() else "spawn")

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
        r["note"] = clean_note(exc)
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
            r["note"] = clean_note(exc)
        rows.append(r)

    if "prioritized" in todo:
        # sequential 과 같은 고정 경로를 쓰되 동시 이동을 허용하고, 의존성에
        # 맞는 우선순위 순서를 최대 24개까지 시도한다 (coordination.py).
        # sequential 이 43% 인 격자에서 74% 라, 고정 경로 baseline 의 상한에
        # 더 가깝다.
        r = _blank(inst, "prioritized", lb, commit, planner_seed, timeout_s)
        t0 = time.perf_counter()
        try:
            sol = plan_prioritized(problem, tracks)
            if sol.feasible:
                _fill(r, problem, sol, V.validate(problem, sol),
                      time.perf_counter() - t0, lb)
            else:
                r["status"] = "no_solution"
        except Exception as exc:                         # noqa: BLE001
            r["status"] = f"error:{type(exc).__name__}"
            r["note"] = clean_note(exc)
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
                r["note"] = clean_note(exc)
        rows.append(r)

    if "wm_planner" in todo:
        r = _blank(inst, "wm_planner", lb, commit, planner_seed, timeout_s)
        t0 = time.perf_counter()
        try:
            kw = dict(wm_cfg, seed=planner_seed)
            if kw.get("log_modes"):
                kw["log_uid"] = inst.uid      # 조인 키.  log_dir 은 main() 이 넣는다
            res = run_wm_planner(problem, WMConfig(**kw))
            rep = V.validate(problem, res.traj)
            _fill(r, problem, res.traj, rep, time.perf_counter() - t0, lb)
            r["n_switches"] = res.n_switches
            r.update(wm_mode_stats(res))
            # B6 이 분류할 재료.  "deadlock" / "wall-clock" 등이 들어 있다.
            r["note"] = clean_note(res.note)
            if not res.traj.feasible:
                r["status"] = f"unfinished:{res.note or 'unknown'}"[:60]
        except Exception as exc:                         # noqa: BLE001
            r["status"] = f"error:{type(exc).__name__}"
            r["note"] = clean_note(exc)
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
        r.update(status=status, runtime_s=round(elapsed, 3), note=clean_note(note))
        rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
#  재개 — 기존 CSV 를 읽어 "이미 끝난 것" 을 알아낸다
# --------------------------------------------------------------------------- #
#: 행 하나를 가리키는 열쇠.  분석은 uid 로 짝을 짓고, 한 인스턴스 안에서는
#: method 와 planner_seed 가 행을 가른다.  이 셋이 곧 "작업 단위" 다.
KEY = ("uid", "method", "planner_seed")

#: `buildable()` 이 실패했거나 prior 를 못 만든 인스턴스가 남기는 method 이름.
#: method 와 무관한 실패이므로 한 줄만 남고, 재개할 때도 통째로 건너뛴다.
NO_METHOD = "-"


class SchemaMismatch(RuntimeError):
    """기존 CSV 의 헤더가 지금의 FIELDS 와 다르다."""


def _key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return tuple(str(row.get(k, "")) for k in KEY)          # type: ignore[return-value]


def _parse_line(raw: bytes) -> Optional[List[str]]:
    try:
        return next(csv.reader([raw.decode("utf-8")]))
    except (UnicodeDecodeError, StopIteration, csv.Error):
        return None


def describe_schema_mismatch(found: Sequence[str], want: Sequence[str]) -> str:
    """어느 열이 어떻게 다른지 사람이 읽을 수 있게 적는다.

    "헤더가 다르다" 만으로는 무엇을 해야 할지 알 수 없다.  빠진 열·남는 열·순서만
    다른 경우를 구분해서 보여준다.
    """
    missing = [c for c in want if c not in found]
    extra = [c for c in found if c not in want]
    lines = [f"  기존 {len(found)}열, 현재 FIELDS {len(want)}열"]
    if missing:
        lines.append(f"  현재에만 있는 열 (기존 CSV 에 없음): {missing}")
    if extra:
        lines.append(f"  기존에만 있는 열 (지금은 안 씀):     {extra}")
    if not missing and not extra:
        lines.append("  열 이름은 같고 **순서**가 다르다:")
        for i, (a, b) in enumerate(zip(found, want)):
            if a != b:
                lines.append(f"    {i}번째: 기존 {a!r} != 현재 {b!r}")
    return "\n".join(lines)


def scan_existing(path: str, fields: Sequence[str] = FIELDS
                  ) -> Tuple[Set[Tuple[str, str, str]], int, int, int]:
    """기존 CSV 에서 완료된 열쇠를 읽는다.

    반환 `(done, n_good, n_dropped, good_end)`.  `good_end` 는 **마지막 온전한
    행이 끝나는 바이트 위치**다.  쓰는 도중에 죽으면 마지막 줄이 잘린 채 남는데,
    거기에 그냥 이어 쓰면 잘린 줄과 새 줄이 한 줄로 붙어 CSV 가 조용히 망가진다.
    그래서 호출부는 이 위치까지 파일을 잘라내고 append 한다.

    첫 번째로 깨진 줄에서 멈춘다.  그 뒤의 줄은 온전해 보여도 믿지 않는다 —
    깨진 지점 뒤를 신뢰하려면 무엇이 왜 깨졌는지 알아야 하는데, 모른다.

    `note` 는 `clean_note()` 가 한 줄로 접어 두므로 "한 줄 == 한 행" 이 성립한다.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    lines = raw.splitlines(keepends=True)
    if not lines:
        return set(), 0, 0, 0

    header = _parse_line(lines[0])
    if header != list(fields):
        raise SchemaMismatch(describe_schema_mismatch(header or [], fields))

    done: Set[Tuple[str, str, str]] = set()
    good_end, n_good = len(lines[0]), 0
    for i, ln in enumerate(lines[1:], start=1):
        row = _parse_line(ln) if ln.endswith((b"\r\n", b"\n")) else None
        if row is None or len(row) != len(fields):
            return done, n_good, len(lines) - i, good_end
        done.add(_key(dict(zip(fields, row))))
        good_end += len(ln)
        n_good += 1
    return done, n_good, 0, good_end


# --------------------------------------------------------------------------- #
#  작업 단위
# --------------------------------------------------------------------------- #
def pending_methods(inst: Instance, planner_seed: int, methods: Sequence[str],
                    done: Set[Tuple[str, str, str]],
                    deterministic_seed: int = 0) -> List[str]:
    """이 (인스턴스, planner_seed) 에서 **아직 안 끝난** method.

    재개를 method 단위로 보는 이유: 인스턴스 하나가 여러 행을 쓰는 도중에 죽으면
    앞쪽 method 는 이미 파일에 있다.  단위를 인스턴스로 잡으면 그걸 다시 돌려
    같은 행을 두 번 쓰게 된다.
    """
    ps = str(planner_seed)
    return [m for m in methods_for_seed(methods, planner_seed, deterministic_seed)
            if (inst.uid, m, ps) not in done]


def _pool_init(inner_workers: int) -> None:
    """Pool 워커의 첫 숨.

    * `/opt/ros` 를 판다.  fork 면 부모에서 이미 털렸지만 spawn 이면 아니다 —
      물려받으면 import 단계에서 죽고, 그 죽음은 "인스턴스가 어렵다" 와 구별되지
      않는다.  initializer 로 명시하는 편이 시작 방식에 안 기댄다.
    * 스레드 수를 다시 못 박는다 (모듈 상단 참조).
    * Track A 에게 안쪽 몫을 알린다.
    * SIGINT 를 무시한다.  Ctrl-C 는 부모가 받아서 정리해야 한다 — 워커가 같이
      받으면 트레이스백 N개가 쏟아지고 부모의 정리 순서가 엉킨다.
    """
    strip_ros_paths()
    for var in THREAD_VARS:
        os.environ[var] = "1"
    os.environ[INNER_WORKERS_VAR] = str(inner_workers)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


#: Pool 워커는 기본이 daemon 이고, daemon 프로세스는 자식을 만들 수 없다
#: ("daemonic processes are not allowed to have children").  그런데 2단계의 하드
#: 타임아웃은 인스턴스마다 손자 프로세스를 띄워 죽이는 방식이다 — `sample_modes()`
#: 가 3천만 조합을 만드는 동안에는 파이썬 레벨에서 끼어들 지점이 없어서, 프로세스를
#: 죽이는 것 말고는 확실한 방법이 없다.  그래서 워커의 daemon 을 끈다.
#:
#: 대가: 부모가 SIGKILL 로 죽으면 워커가 고아로 남는다.  Ctrl-C 와 정상 종료는
#: `pool.terminate()` / `pool.join()` 이 거둔다.
class _NoDaemonProcess(_BASE_CTX.Process):                   # type: ignore[name-defined]
    @property
    def daemon(self) -> bool:
        return False

    @daemon.setter
    def daemon(self, value: bool) -> None:
        pass


class _NoDaemonContext(type(_BASE_CTX)):                     # type: ignore[misc]
    Process = _NoDaemonProcess


def _run_unit(task):
    """Pool 에 넘어가는 작업 하나.  인자·반환값이 전부 picklable 해야 한다."""
    inst, wm_cfg, methods, commit, ps, timeout_s, det_seed = task
    try:
        rows = run_instance_guarded(inst, wm_cfg, methods, commit,
                                    planner_seed=ps, timeout_s=timeout_s,
                                    deterministic_seed=det_seed)
        return inst, ps, rows, None
    except BaseException as exc:                             # noqa: BLE001
        return inst, ps, [], f"{type(exc).__name__}: {exc}"


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


def rows_per_instance(methods: Sequence[str], planner_seeds: Sequence[int]) -> int:
    """`buildable()` 이 성공한 인스턴스 하나가 남겨야 할 행 수.

    결정론적 method 는 한 번, 확률적 method 는 planner_seed 마다 한 번.
    """
    det = sum(1 for m in ALL_METHODS
              if m in methods and m not in STOCHASTIC_METHODS)
    stoch = sum(1 for m in ALL_METHODS
                if m in methods and m in STOCHASTIC_METHODS)
    return det + stoch * len(planner_seeds)


def audit_row_count(path: str, methods: Sequence[str],
                    planner_seeds: Sequence[int], n_instances: int) -> bool:
    """최종 CSV 행 수가 기대치와 맞는가.

    `buildable()` 이 실패한 인스턴스는 method 와 무관하게 한 줄만 남기므로,
    그 수를 세어 기대치에서 덜어낸다.  맞지 않으면 어딘가에서 행이 사라졌거나
    두 번 쓰였다는 뜻이고, 둘 다 성공률을 조용히 틀리게 만든다.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n_bad = len({r["uid"] for r in rows if r["method"] == NO_METHOD})
    want = (n_instances - n_bad) * rows_per_instance(methods, planner_seeds) + n_bad
    got = len(rows)
    dupes = len(rows) - len({_key(r) for r in rows})
    print(f"\n검산: {got} rows (기대 {want})"
          f"  = (인스턴스 {n_instances} - 생성실패 {n_bad}) x "
          f"{rows_per_instance(methods, planner_seeds)} + {n_bad}")
    if dupes:
        print(f"[bench.run] 경고: (uid, method, planner_seed) 가 겹치는 행 {dupes}건. "
              f"재개가 이미 끝난 것을 다시 돌렸다는 뜻이다.", file=sys.stderr)
    if got != want:
        print(f"[bench.run] 경고: 행 수가 기대치와 다르다 ({got} != {want}). "
              f"중간에 죽었다면 다시 실행하면 채워진다.", file=sys.stderr)
        return False
    return not dupes


def _hms(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="앞에서 N개 인스턴스만 (설정 확인용)")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="인스턴스 하나의 하드 상한(초).  넘기면 자식을 죽이고 "
                         "status=timeout 행을 남긴다 (기본 300)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="인스턴스를 나눠 돌릴 프로세스 수 (기본: 코어 수 - 1)")
    ap.add_argument("--inner-workers", type=int, default=1,
                    help=f"워커 안쪽(Track A rollout) 이 쓸 프로세스 수. "
                         f"{INNER_WORKERS_VAR} 로 전달된다 (기본 1 = 병렬화 안 함)")
    ap.add_argument("--fresh", action="store_true",
                    help="기존 출력 파일을 이어받지 않고 덮어쓴다")
    args = ap.parse_args()

    strip_ros_paths()

    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    axes = axis_from_config(cfg["axis"])
    methods = cfg.get("methods", list(ALL_METHODS))
    wm_cfg, ignored_seed = wm_config_without_seed(cfg.get("wm_config", {}))
    planner_seeds = planner_seeds_of(axes)
    det_seed = planner_seeds[0]

    if ignored_seed is not None:
        print(f'[bench.run] 경고: wm_config 의 "seed": {ignored_seed} 를 무시한다. '
              f"플래너 seed 는 axis.planner_seeds 가 정한다 "
              f"(이번 실행: {list(planner_seeds)}).", file=sys.stderr)

    out = args.out or os.path.join("bench", "runs", f"{cfg.get('name', 'run')}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if wm_cfg.get("log_modes"):
        # 워커가 프로세스별 파일로 쓴다 (execute.py 의 _log_paths).  합치는 것은
        # 배치가 끝난 뒤 별도 단계 — 실행 중에 합치면 부분 파일을 읽게 된다.
        wm_cfg["log_dir"] = os.path.dirname(out) or "."
    commit = git_commit()

    insts = list(grid(axes))
    n_grid = expected_count(axes)
    if len(insts) != n_grid:
        print(f"[bench.run] 알림: 격자가 겹쳐 {n_grid - len(insts)}개가 합쳐졌다 "
              f"(축의 곱 {n_grid} -> 인스턴스 {len(insts)}).", file=sys.stderr)
    if args.limit:
        insts = insts[:args.limit]

    # -- 재개 --------------------------------------------------------------- #
    done: Set[Tuple[str, str, str]] = set()
    resuming = os.path.exists(out) and not args.fresh
    if resuming:
        try:
            done, n_good, n_dropped, good_end = scan_existing(out)
        except SchemaMismatch as exc:
            print(f"\n기존 CSV 의 헤더가 지금의 FIELDS 와 다르다: {out}\n{exc}\n\n"
                  f"스키마가 섞인 CSV 는 나중에 분석할 수 없다.  다른 --out 을 쓰거나, "
                  f"기존 결과를 버려도 된다면 --fresh 를 붙인다.", file=sys.stderr)
            raise SystemExit(2)
        if n_dropped:
            print(f"[bench.run] 기존 CSV 의 마지막 {n_dropped}행이 잘려 있어 버린다 "
                  f"(쓰는 도중에 죽은 흔적).  해당 조합은 다시 돌린다.",
                  file=sys.stderr)
            with open(out, "r+b") as fh:
                fh.truncate(good_end)
        print(f"재개: {out} 에서 {n_good}행을 이어받는다.")
    elif os.path.exists(out):
        print(f"--fresh: {out} 를 덮어쓴다 ({os.path.getsize(out)} bytes 를 버린다).")

    tasks = [(inst, wm_cfg, todo, commit, ps, args.timeout, det_seed)
             for inst in insts for ps in planner_seeds
             if (inst.uid, NO_METHOD, str(det_seed)) not in done
             and (todo := pending_methods(inst, ps, methods, done, det_seed))]

    total_units = len(insts) * len(planner_seeds)
    workers = max(1, args.workers)
    det = [m for m in ALL_METHODS if m in methods and m not in STOCHASTIC_METHODS]
    print(f"{cfg.get('name', 'run')}: 격자 {n_grid}, 인스턴스 {len(insts)} x "
          f"{len(planner_seeds)} planner seeds = {total_units} units\n"
          f"  기존 완료 {total_units - len(tasks)} units, 이번 실행 대상 "
          f"{len(tasks)} units\n"
          f"  방법 {methods}\n"
          f"  결정론적 {det} 는 planner_seed={det_seed} 에서만 돈다\n"
          f"  워커 {workers} (안쪽 {args.inner_workers}), "
          f"하드 타임아웃 {args.timeout:g}s / 인스턴스\n"
          f"  commit {commit}\n-> {out}\n", flush=True)

    if not tasks:
        print("0 remaining — 이미 다 끝났다.")
        audit_row_count(out, methods, planner_seeds, len(insts))
        return

    os.environ[INNER_WORKERS_VAR] = str(args.inner_workers)
    mode = "a" if resuming and os.path.exists(out) else "w"
    t_start = time.perf_counter()
    n_done = n_rows = 0
    interrupted = False

    with open(out, mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if mode == "w" or fh.tell() == 0:
            w.writeheader()
            fh.flush()

        pool = mp.pool.Pool(workers, initializer=_pool_init,
                            initargs=(args.inner_workers,),
                            context=_NoDaemonContext())
        try:
            for inst, ps, rows, err in pool.imap_unordered(_run_unit, tasks):
                n_done += 1
                for r in rows:
                    w.writerow(r)
                    n_rows += 1
                fh.flush()               # 결과가 오는 대로 즉시 durable 하게

                elapsed = time.perf_counter() - t_start
                eta = elapsed / n_done * (len(tasks) - n_done)
                if err:                  # 워커 자체가 무너진 경우 (행이 없다)
                    print(f"[bench.run] 워커 실패: {inst.uid} ps={ps}: {err}",
                          file=sys.stderr, flush=True)
                wm = next((r for r in rows if r["method"] == "wm_planner"), None)
                tag = (f"{wm['team_time']}s {wm['status']}" if wm
                       else (rows[0]["status"] if rows else f"실패:{err}"))
                # 병렬이라 줄 순서가 뒤섞인다.  uid 를 항상 실어 어느 인스턴스의
                # 이야기인지 한 줄만 보고 알 수 있게 한다.
                print(f"  [{n_done:>5}/{len(tasks)}] {inst.uid:<34} ps={ps} "
                      f"{tag:<24} {_hms(elapsed)} 경과, 남은 ~{_hms(eta)}",
                      flush=True)
            pool.close()
        except KeyboardInterrupt:
            interrupted = True
            print("\n중단 (Ctrl-C).  워커를 정리한다...", file=sys.stderr, flush=True)
            pool.terminate()
        finally:
            pool.join()

    dt = time.perf_counter() - t_start
    if interrupted:
        print(f"\n중단: {n_done}/{len(tasks)} units, {n_rows} rows 를 {out} 에 남겼다. "
              f"({_hms(dt)})\n같은 명령을 다시 실행하면 남은 것만 돈다.")
        raise SystemExit(130)

    print(f"\n완료: {_hms(dt)}, {n_done} units / {n_rows} rows  ->  {out}")
    audit_row_count(out, methods, planner_seeds, len(insts))
    print(f"분석: python -m bench.analyze --csv {out}")


if __name__ == "__main__":
    main()
