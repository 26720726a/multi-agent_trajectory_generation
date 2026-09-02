#!/usr/bin/env python3
"""속도 부호 반전을 세어 "진동성 livelock" 추론을 검증한다 (S3-A 보고서 §6-3).

    V2_ROOT=<a_max=inf 인 트리> python scripts/count_reversals.py \
        results/s3/baseline_S2.csv results/s3/A2_braking.csv

S3-A 에서 예상 밖의 일이 있었다.  가속 상한을 넣었더니 성공률은 −7.30%p 인데
**209건은 새로 풀렸다.**  당시의 추론은 이랬다 — 기준선 컨트롤러는 매 스텝
임의의 속도를 고를 수 있어 마주 본 두 에이전트가 왕복 진동에 빠질 수 있고,
`W_TURN=0.22` 는 그것을 비용으로만 누른다.  `a_max` 는 그 진폭을 **구조적으로**
제한하므로 진동이 원인이던 인스턴스는 오히려 풀린다.

그 추론이 맞다면, **기준선 궤적에서** 다음이 성립해야 한다.

    (A 에서 실패했다가 B 에서 성공한 것)의 반전 횟수
        >  (양쪽 다 성공한 것)의 반전 횟수

반전의 정의: 한 에이전트의 속도 벡터가 **연속한 두 이동 스텝**에서 서로
둔각을 이루는 것 (`v[t] · v[t+1] < 0`).  정지 스텝은 건너뛴다 — 서 있는 것은
진동이 아니다.  건너뛰므로 "멈췄다가 반대로 출발" 도 한 번으로 잡힌다.

**기준선 물리로 재실행해야 한다.**  묻는 것이 "기준선에서 진동하고 있었는가"
이므로, `V2_ROOT` 로 `a_max=inf` 인 트리를 가리켜야 한다.  그 트리에 제동 항이
들어 있어도 무방하다 — `a_max=inf` 면 제동 항은 무효다.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.environ.get(
    "V2_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from config import PHYSICS                                    # noqa: E402
from planning.execute import WMConfig, run_wm_planner         # noqa: E402
from planning.world import random_problem                     # noqa: E402


def load(path: str) -> Dict[Tuple[str, str], dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {(r["uid"], r["planner_seed"]): r
                for r in csv.DictReader(fh) if r["method"] == "wm_planner"}


def rebuild(row: dict):
    lo, hi = (row["n_obstacles"].split("-") + [row["n_obstacles"]])[:2]
    return random_problem(
        seed=int(row["instance_seed"]), n_agents=int(row["n_agents"]),
        size=float(row["size"]), n_obstacles=(int(lo), int(hi)),
        dep_mode=row["dep_mode"], coupled_waypoints=float(row["couple_prob"]),
        couple_dist=float(row["couple_dist"]))


def reversals(vel: np.ndarray, eps: float = 1e-6) -> int:
    """(T+1, n, 2) 명령 속도에서 부호 반전 횟수 (전 에이전트 합)."""
    total = 0
    for i in range(vel.shape[1]):
        v = vel[:, i, :]
        moving = v[np.linalg.norm(v, axis=1) > eps]
        if len(moving) < 2:
            continue
        dots = np.einsum("ij,ij->i", moving[:-1], moving[1:])
        total += int(np.count_nonzero(dots < 0.0))
    return total


def measure(rows: List[dict], limit: int, horizon_s: float) -> List[float]:
    out = []
    for r in rows[:limit]:
        problem = rebuild(r)
        res = run_wm_planner(problem, WMConfig(
            seed=int(r["planner_seed"]), horizon_s=horizon_s,
            max_modes=16, keep_modes=8, k_routes=2, time_budget_s=120.0))
        # 스텝 수로 정규화한다 — 오래 도는 궤적이 자동으로 많아지면 안 된다
        steps = max(res.traj.T * problem.n, 1)
        out.append(100.0 * reversals(res.traj.vel) / steps)
    return out


def describe(name: str, xs: List[float]) -> str:
    if not xs:
        return f"  {name:<28} 표본 없음"
    s = sorted(xs)
    return (f"  {name:<28} n={len(xs):<4} 중앙값 {st.median(s):6.3f}  "
            f"평균 {st.mean(s):6.3f}  p90 {s[int(0.9*(len(s)-1))]:6.3f}  "
            f"최대 {s[-1]:6.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="기준선 CSV (여기 물리로 재실행한다)")
    ap.add_argument("b", help="비교 대상 CSV")
    ap.add_argument("--limit", type=int, default=60,
                    help="각 군에서 재실행할 최대 인스턴스 수")
    ap.add_argument("--horizon-s", type=float, default=4.0)
    ap.add_argument("--min-samples", type=int, default=15,
                    help="이보다 적으면 판정하지 않고 건수만 보고한다")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    keys = sorted(set(A) & set(B))
    gained = [A[k] for k in keys if A[k]["status"] != "ok" and B[k]["status"] == "ok"]
    both = [A[k] for k in keys if A[k]["status"] == "ok" and B[k]["status"] == "ok"]
    lost = [A[k] for k in keys if A[k]["status"] == "ok" and B[k]["status"] != "ok"]

    print(f"재실행 트리: {ROOT}")
    print(f"  a_max = {PHYSICS.a_max}   (기준선 물리여야 한다: inf)")
    if np.isfinite(PHYSICS.a_max):
        print("  !! a_max 가 유한하다.  기준선에서의 진동을 재는 것이 목적이므로 "
              "V2_ROOT 로 a_max=inf 인 트리를 가리킬 것.", file=sys.stderr)
    print(f"\n군 크기:  얻음 {len(gained)},  양쪽성공 {len(both)},  잃음 {len(lost)}")

    if len(gained) < args.min_samples:
        print(f"\n얻은 건수가 {len(gained)} 뿐이라 판정하지 않는다 "
              f"(기준 {args.min_samples}).  건수만 보고한다.")
        return 0

    print(f"\n기준선 궤적의 부호 반전 (100 (에이전트x스텝) 당), "
          f"각 군 최대 {args.limit}개 재실행\n")
    g = measure(gained, args.limit, args.horizon_s)
    b = measure(both, args.limit, args.horizon_s)
    l = measure(lost, args.limit, args.horizon_s)
    print(describe("얻음 (실패 -> 성공)", g))
    print(describe("양쪽 성공", b))
    print(describe("잃음 (성공 -> 실패)", l))

    if g and b:
        mg, mb = st.median(g), st.median(b)
        print(f"\n판정: 얻음 중앙값 {mg:.3f} vs 양쪽성공 {mb:.3f}  ->  "
              + ("추론과 **같은 방향** (얻은 쪽이 더 많이 진동했다)"
                 if mg > mb else
                 "추론과 **반대 방향** — 진동 가설로는 설명되지 않는다"))
        print("  (표본이 작고 짝지어지지 않은 비교다.  방향만 본다.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
