#!/usr/bin/env python3
"""후보 전멸(candidate wipeout) 진단 — S3-A" (W1/W2/W3).

    python scripts/diag_wipeout.py results/s3/A2_braking.csv --limit 100

**측정만 한다.**  알고리즘을 바꾸지 않고, 현재 궤적도 바꾸지 않는다.  전멸이
일어난 스텝에서 *그 시점의 상태 그대로* 다른 후보 집합을 `free_steps` 에
물어볼 뿐이다.

가르려는 가설
-------------
  H1 (팬 해상도)  도달 가능 원반 안에 안전한 속도가 있는데 65개 팬이 성글어
                  놓쳤다.            -> SPEED_LEVELS 상대화(A2)가 맞다
  H2 (도달 불가)  도달 가능 원반 전체가 막혀 있다.  한 스텝 앞만 보는 필터가
                  빠져나올 수 없는 상태를 미리 못 막았다.
                                     -> 다단계 전방 검사가 맞다

W2 의 판정 규칙은 **결과를 보기 전에** 정해져 있다 (S3-A" 지시서).

    b(원반 조밀 표본)에 자유 후보가 전멸 사건의 20% 이상에 존재  -> H1 우세
    b 가 거의 항상 비어 있고(<5%) d(즉시 정지)는 자유          -> H2 확정
    그 사이                                                    -> 혼합
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONTROLLER, PHYSICS                          # noqa: E402
from planning.control import (DEP_STANDOFF, SceneCache, _candidates,  # noqa: E402
                              A_MAX)
from planning.execute import WMConfig, run_wm_planner           # noqa: E402
from planning.world import random_problem                       # noqa: E402
from safety.phase import PHASE_DWELL, PHASE_DONE                # noqa: E402

PHASE_NAME = {0: "TO_WP", 1: "DWELL", 2: "TO_GOAL", 3: "DONE"}


# --------------------------------------------------------------------------- #
#  후보 집합 넷
# --------------------------------------------------------------------------- #
def disk_samples(v_prev: np.ndarray, dv_max: float, v_lim: float,
                 n_ring: int = 20, n_rad: int = 25) -> np.ndarray:
    """도달 가능 원반 `|v - v_prev| <= dv_max` 의 조밀 표본 (약 500점).

    속력 상한 `v_lim` 도 함께 지킨다 — 그것을 넘는 속도는 애초에 낼 수 없으므로
    "팬이 놓쳤다" 의 후보가 될 수 없다.  중심(= 이전 속도 유지)도 포함한다.
    """
    ang = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
    rad = np.linspace(0.0, dv_max, n_rad)
    r, a = np.meshgrid(rad, ang, indexing="ij")
    pts = np.stack([r * np.cos(a), r * np.sin(a)], axis=-1).reshape(-1, 2)
    cand = v_prev[None, :] + pts
    keep = np.linalg.norm(cand, axis=1) <= v_lim + 1e-12
    cand = cand[keep]
    return cand if len(cand) else np.zeros((0, 2))


def free_mask(scene: SceneCache, problem, i: int, pos_i: np.ndarray,
              cand: np.ndarray, dep_hold: bool,
              target_pt: np.ndarray) -> np.ndarray:
    """`step()` 의 하드 필터와 **같은 계산**.  기하 + 의존성 대기."""
    if not len(cand):
        return np.zeros(0, bool)
    step_pts = pos_i[None, :] + cand * problem.dt
    keep = scene.free_steps(i, pos_i, step_pts)
    if dep_hold:
        keep = keep & (np.linalg.norm(step_pts - target_pt[None, :], axis=1)
                       >= DEP_STANDOFF - 1e-9)
    return keep


def blame(scene: SceneCache, problem, i: int, pos_i: np.ndarray,
          cand: np.ndarray, dep_hold: bool, target_pt: np.ndarray) -> str:
    """전멸을 만든 것이 기하인가 의존성 대기인가.

    `step()` 의 하드 필터는 **정적 기하(장애물+벽)와 의존성 대기 두 가지뿐**
    이다.  이웃 에이전트는 VO(soft cost)와 `_project_safe` 가 다루므로 후보를
    기각하지 못한다 — 즉 **전멸의 직접 원인이 될 수 없다.**
    """
    geo = scene.free_steps(i, pos_i, pos_i[None, :] + cand * problem.dt)
    if not dep_hold:
        return "기하(장애물/벽)"
    both = free_mask(scene, problem, i, pos_i, cand, True, target_pt)
    if geo.any() and not both.any():
        return "의존성 대기"
    if not geo.any():
        return "기하(장애물/벽)"
    return "(막히지 않음)"


# --------------------------------------------------------------------------- #
#  인스턴스 하나
# --------------------------------------------------------------------------- #
def rebuild(row: dict):
    lo, hi = (row["n_obstacles"].split("-") + [row["n_obstacles"]])[:2]
    return random_problem(
        seed=int(row["instance_seed"]), n_agents=int(row["n_agents"]),
        size=float(row["size"]), n_obstacles=(int(lo), int(hi)),
        dep_mode=row["dep_mode"], coupled_waypoints=float(row["couple_prob"]),
        couple_dist=float(row["couple_dist"]))


def analyse(row: dict, rewind: int) -> List[dict]:
    problem = rebuild(row)
    scene = SceneCache(problem)
    dt = problem.dt
    dv_max = A_MAX * dt
    trace: List[dict] = []
    res = run_wm_planner(problem, WMConfig(
        seed=int(row["planner_seed"]), horizon_s=4.0, max_modes=16,
        keep_modes=8, k_routes=2, time_budget_s=120.0), trace=trace)

    by_t = {r["t"]: r for r in trace}
    out: List[dict] = []
    for rec in trace:
        for i in np.flatnonzero(rec["forced"]):
            i = int(i)
            pos_i = rec["pos"][i]
            v_prev = rec["vel_prev"][i]
            v_pref = rec["v_pref"][i]
            v_lim = float(rec["v_lim"][i])
            hold = bool(rec["dep_hold"][i])
            a = problem.agents[i]
            ph = int(rec["phase"][i])
            target = np.asarray(a.waypoint if ph == 0 else a.goal, float)

            # -- 이웃/장애물 거리 (맥락) ------------------------------------ #
            others = [j for j in range(problem.n) if j != i]
            d_agent = min((float(np.linalg.norm(rec["pos"][j] - pos_i))
                           for j in others), default=float("inf"))
            near = scene.nearest_points(pos_i)
            d_obst = float(np.min(np.linalg.norm(near - pos_i[None, :], axis=1)))

            # -- W2: 후보 집합 넷 ------------------------------------------- #
            c_a = _candidates(v_pref, v_lim, v_prev=v_prev, dv_max=dv_max)
            c_b = disk_samples(v_prev, dv_max, v_lim)
            c_c = _candidates(v_pref, v_lim)                    # 클리핑 없음
            c_d = np.zeros((1, 2))                              # 즉시 정지

            f_a = free_mask(scene, problem, i, pos_i, c_a, hold, target)
            f_b = free_mask(scene, problem, i, pos_i, c_b, hold, target)
            f_c = free_mask(scene, problem, i, pos_i, c_c, hold, target)
            f_d = free_mask(scene, problem, i, pos_i, c_d, hold, target)

            # -- W3: 되감기 — 원반이 언제부터 비었나 ------------------------ #
            hist = []
            for k in range(1, rewind + 1):
                prev = by_t.get(rec["t"] - k)
                if prev is None:
                    break
                pv = prev["vel_prev"][i]
                pl = prev["v_lim"][i]
                if not np.isfinite(pl):          # dwell/done 이라 안 움직였다
                    hist.append(None)
                    continue
                pk = disk_samples(pv, dv_max, float(pl))
                pt = np.asarray(a.waypoint if int(prev["phase"][i]) == 0
                                else a.goal, float)
                fk = free_mask(scene, problem, i, prev["pos"][i], pk,
                               bool(prev["dep_hold"][i]), pt)
                hist.append(float(fk.mean()) if len(pk) else 0.0)

            out.append({
                "uid": row["uid"], "planner_seed": int(row["planner_seed"]),
                "t": int(rec["t"]), "agent": i, "phase": PHASE_NAME.get(ph, ph),
                "speed_prev": float(np.linalg.norm(v_prev)),
                "v_lim": v_lim, "dep_hold": hold,
                "d_agent": d_agent, "d_obst": d_obst,
                "blame": blame(scene, problem, i, pos_i, c_b, hold, target),
                "free_a": int(f_a.sum()), "n_a": len(c_a),
                "free_b": int(f_b.sum()), "n_b": len(c_b),
                "free_c": int(f_c.sum()), "n_c": len(c_c),
                "free_d": int(f_d.sum()),
                "rewind_free_frac": hist,
            })
    return out


# --------------------------------------------------------------------------- #
def hist_line(label: str, count: int, total: int, width: int = 40) -> str:
    bar = "#" * int(round(width * count / max(total, 1)))
    return f"    {label:<22} {count:>6} ({100.0*count/max(total,1):5.1f}%) {bar}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--limit", type=int, default=100,
                    help="재실행할 인스턴스(행) 수.  전멸이 많은 순서로 고른다")
    ap.add_argument("--rewind", type=int, default=15)
    ap.add_argument("--out", default=None, help="사건 전부를 JSON 으로 저장")
    ap.add_argument("--sample", choices=("top", "random"), default="top",
                    help="top=전멸이 많은 순 (지시서 기본), random=무작위")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["method"] == "wm_planner"]

    def nb(r):
        try:
            return float(r.get("n_blocked") or 0)
        except ValueError:
            return 0.0

    have = [r for r in rows if nb(r) > 0]
    if args.sample == "top":
        have.sort(key=nb, reverse=True)
        picked = have[:args.limit]
        how = "전멸이 많은 순 (표본이 그쪽으로 치우친다 — 무작위 표본과 함께 볼 것)"
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(have), size=min(args.limit, len(have)),
                         replace=False)
        picked = [have[int(k)] for k in sorted(idx)]
        how = "무작위"
    print(f"{args.csv}: wm_planner {len(rows)}행 중 전멸이 있는 행 {len(have)} "
          f"({100.0*len(have)/max(len(rows),1):.1f}%)")
    print(f"  {len(picked)}행을 계측 모드로 재실행한다 — {how}")
    print(f"  (n_blocked {nb(picked[0]):.0f} ~ {nb(picked[-1]):.0f})\n")

    events: List[dict] = []
    for k, r in enumerate(picked, 1):
        try:
            events += analyse(r, args.rewind)
        except Exception as exc:                              # noqa: BLE001
            print(f"  [{k}/{len(picked)}] {r['uid']} 실패: {exc}", file=sys.stderr)
        if k % 20 == 0:
            print(f"  ... {k}/{len(picked)} 인스턴스, 사건 {len(events)}건",
                  flush=True)

    if not events:
        print("전멸 사건을 하나도 재현하지 못했다.")
        return 1
    n = len(events)
    #: `v_cmd` 가 0 으로 강제되므로 |dv|/dt = |v_prev|/dt 다.  그것이 a_max 를
    #: 넘는 전멸만이 "급정거" 이고, 나머지는 **이미 서 있던 것**이다.
    #: 성능에 영향을 주는 것은 앞쪽뿐이므로 갈라서 본다.
    lim = A_MAX * PHYSICS.dt
    for e in events:
        e["harmful"] = e["speed_prev"] > lim * (1 + 1e-9)
    harm = [e for e in events if e["harmful"]]
    idle = [e for e in events if not e["harmful"]]

    print(f"\n{'='*74}\nW1 — 전멸 사건 인벤토리   총 {n}건 "
          f"({len(picked)} 인스턴스 재실행)\n{'='*74}")
    print(f"\n  **부류 나누기** (|v_cmd| 는 0 으로 강제되므로 |dv|/dt = |v_prev|/dt)")
    print(f"    급정거형  |v_prev| > a_max*dt = {lim:.2f} m/s   "
          f"{len(harm):>6}/{n} ({100.0*len(harm)/n:5.1f}%)  <- a_max 를 깬다")
    print(f"    대기형    |v_prev| <= {lim:.2f} m/s (거의 정지)  "
          f"{len(idle):>6}/{n} ({100.0*len(idle)/n:5.1f}%)  <- 이미 서 있었다")

    print("\n  |v_prev| 분포")
    bins = [(0.0, 1e-9, "정지 (=0)"), (1e-9, 0.3, "0 < v <= 0.3"),
            (0.3, 0.6, "0.3 < v <= 0.6"), (0.6, 0.9, "0.6 < v <= 0.9"),
            (0.9, 1.2 + 1e-9, "0.9 < v <= 1.2  (순항)")]
    for lo, hi, lab in bins:
        c = sum(1 for e in events if lo < e["speed_prev"] <= hi
                or (lo == 0.0 and e["speed_prev"] <= hi))
        print(hist_line(lab, c, n))

    print("\n  phase 분포")
    for k_, c in Counter(e["phase"] for e in events).most_common():
        print(hist_line(str(k_), c, n))

    print("\n  무엇이 막았는가 — 전체")
    for k_, c in Counter(e["blame"] for e in events).most_common():
        print(hist_line(str(k_), c, n))
    print("    (이웃 에이전트는 하드 필터에 없다 — VO 는 soft cost 이고 "
          "_project_safe 는 나중이다.  즉 전멸의 직접 원인이 될 수 없다)")
    if harm:
        print("\n  무엇이 막았는가 — **급정거형만**")
        for k_, c in Counter(e["blame"] for e in harm).most_common():
            print(hist_line(str(k_), c, len(harm)))
        print("\n  급정거형의 |v_prev| 분포")
        for lo, hi, lab in bins[1:]:
            c = sum(1 for e in harm if lo < e["speed_prev"] <= hi)
            print(hist_line(lab, c, len(harm)))

    print("\n  최근접 거리 (중앙값)")
    for key, lab in (("d_agent", "이웃 에이전트"), ("d_obst", "장애물/벽")):
        v = sorted(e[key] for e in events if math.isfinite(e[key]))
        if v:
            print(f"    {lab:<16} {v[len(v)//2]:6.3f} m   "
                  f"(p10 {v[len(v)//10]:6.3f}, p90 {v[9*len(v)//10]:6.3f})")

    # -- W2 --------------------------------------------------------------- #
    print(f"\n{'='*74}\nW2 — 반사실 후보 평가 (H1 vs H2)\n{'='*74}")

    def w2(group: List[dict], label: str) -> Optional[float]:
        m = len(group)
        if not m:
            print(f"\n  [{label}] 표본 없음")
            return None
        a_bad = sum(1 for e in group if e["free_a"] == 0)
        b_any = sum(1 for e in group if e["free_b"] > 0)
        c_any = sum(1 for e in group if e["free_c"] > 0)
        d_any = sum(1 for e in group if e["free_d"] > 0)
        print(f"\n  [{label}]  {m}건")
        print(f"    a) 현행 클리핑 65개가 전부 막힘        {a_bad:>6}/{m} "
              f"({100.0*a_bad/m:5.1f}%)   <- 정의상 100% (검산)")
        print(f"    b) 도달 가능 원반 조밀 표본에 자유 후보 {b_any:>6}/{m} "
              f"({100.0*b_any/m:5.1f}%)   <- 판정의 핵심")
        print(f"    c) 클리핑 없는 65개에 자유 후보        {c_any:>6}/{m} "
              f"({100.0*c_any/m:5.1f}%)   <- a_max 자체가 제약인가")
        print(f"    d) 즉시 정지(v=0)가 자유               {d_any:>6}/{m} "
              f"({100.0*d_any/m:5.1f}%)")
        if b_any:
            fr = sorted(100.0 * e["free_b"] / max(e["n_b"], 1)
                        for e in group if e["free_b"] > 0)
            print(f"       b 가 비어 있지 않을 때 자유 비율: 중앙값 "
                  f"{fr[len(fr)//2]:.2f}%, p90 {fr[9*len(fr)//10]:.2f}%, "
                  f"최대 {fr[-1]:.2f}%")
        pct = 100.0 * b_any / m
        if pct >= 20.0:
            v = "H1 우세 — 팬 해상도.  SPEED_LEVELS 상대화(A2)로 간다"
        elif pct < 5.0 and d_any > 0:
            v = "H2 확정 — 도달 불가.  다단계 전방 검사로 간다"
        elif pct < 5.0:
            v = "H2 쪽이지만 d 도 막혔다 — 즉시 정지조차 불가능한 상태다"
        else:
            v = f"혼합 (b={pct:.1f}%, 20% 와 5% 사이) — 둘 다 기여한다"
        print(f"    판정: b = {pct:.1f}%  ->  **{v}**")
        return pct

    print("\n  판정 규칙은 결과를 보기 전에 정해졌다 (지시서 W2).")
    w2(events, "전체")
    w2(harm, "급정거형 — a_max 를 깨는 것.  성능과 직결된다")
    w2(idle, "대기형 — 이미 서 있던 것")

    # -- W3 --------------------------------------------------------------- #
    print(f"\n{'='*74}\nW3 — 언제부터 갇혔는가 (되감기 {args.rewind} 스텝)\n"
          f"      **급정거형만** 본다 — 대기형은 갇힌 것이 아니라 기다리는 것이다\n"
          f"{'='*74}")
    last_free, never = [], 0
    n = max(len(harm), 1)
    for e in harm:
        h = e["rewind_free_frac"]
        k_free = None
        for k, f in enumerate(h, 1):
            if f is not None and f > 0.0:
                k_free = k
                break
        if k_free is None:
            never += 1
        else:
            last_free.append(k_free)
    if last_free:
        s = sorted(last_free)
        print(f"\n  마지막으로 원반에 탈출구가 있었던 시점까지의 스텝 수 k")
        print(f"    중앙값 {s[len(s)//2]}, p90 {s[9*len(s)//10]}, 최대 {s[-1]}  "
              f"(표본 {len(s)})")
        for k in range(1, min(args.rewind, 8) + 1):
            print(hist_line(f"k = {k}", sum(1 for x in last_free if x == k),
                            len(last_free)))
    print(f"  되감기 {args.rewind} 스텝 내내 원반이 비어 있던 사건: {never}/{n} "
          f"({100.0*never/n:5.1f}%)")
    brake_steps = PHYSICS.v_max / A_MAX / PHYSICS.dt
    print(f"\n  비교: v_max/a_max = {PHYSICS.v_max/A_MAX:.2f}s = "
          f"{brake_steps:.0f} 스텝 (순항에서 완전 정지에 필요한 시간)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(events, fh, ensure_ascii=False)
        print(f"\n사건 {n}건 -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
