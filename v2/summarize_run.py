#!/usr/bin/env python3
"""배치 CSV 하나를 계획서 §7.1 의 KPI 로 요약한다.

    python scripts/summarize_run.py results/s3/baseline_S2.csv

`compare_runs.py` 가 **두 실행의 차이**를 보는 도구라면, 이쪽은 **한 실행의
절대 수치**를 본다.  게이트 판정(충돌 0건 / 성공률 / n_amax_violations)에
쓰는 숫자가 전부 여기서 나온다.

계획서 §7.4 를 따라 **충돌률은 언제나 deadlock 비율과 짝으로** 낸다.  충돌만
보면 "전부 세워 두면 0%" 라는 퇴화 해가 만점을 받는다.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PHYSICS                                     # noqa: E402

#: 침범 판정 여유 (m).  `safety/validate.py` 와 같은 값이어야 한다.
#: 근거는 safety/README.md §5 — dt=0.1s 에서 한 스텝 0.12 m 의 1/120 이다.
VIOL_TOL = 1e-3


def fnum(row: dict, col: str) -> Optional[float]:
    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        return None


def quantile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


#: `status` / `note` 를 실패 원인으로 가른다.  계획서 §7.1 의 실패 분류다.
def classify(row: dict) -> str:
    st = row.get("status", "")
    note = (row.get("note") or "").lower()
    if st == "ok":
        return "ok"
    if st == "invalid":
        return "invalid(검증 실패)"
    if st == "ungeneratable":
        return "기하적 불가(인스턴스 생성 실패)"
    if st.startswith("unfinished:"):
        if "deadlock" in note or "deadlock" in st:
            return "deadlock"
        if "wall-clock" in note or "budget" in st:
            return "timeout(플래너 예산)"
        if "could not advance" in note or "advance" in st:
            return "정지(컨트롤러가 전진 못함)"
        return "미완(기타)"
    if st in ("timeout", "killed") or st.startswith("error"):
        return f"하네스({st})"
    return st or "(빈칸)"


def summarize(path: str, method: str = "wm_planner") -> List[str]:
    with open(path, newline="", encoding="utf-8") as fh:
        allrows = list(csv.DictReader(fh))
    rows = [r for r in allrows if r.get("method") == method]
    out: List[str] = []
    n = len(rows)
    if not n:
        return [f"{path}: method={method} 행이 없다"]

    fps = Counter(r.get("phys_fp", "") for r in allrows)
    uids = {r["uid"] for r in rows}
    seeds = sorted({r.get("planner_seed", "") for r in rows})

    out += [
        "=" * 74,
        f"{os.path.basename(path)}   method={method}",
        "=" * 74,
        f"  행 {n},  인스턴스 {len(uids)},  planner_seed {seeds}",
        f"  물리 지문: " + ", ".join(f"{k or '(없음)'}({v}행)"
                                  for k, v in fps.most_common()),
    ]
    if len(fps) > 1:
        out.append("  !! 한 파일에 지문이 섞여 있다 — 이 CSV 로는 결론지을 수 없다")

    # -- 성공/실패 ---------------------------------------------------------- #
    cls = Counter(classify(r) for r in rows)
    n_ok = cls["ok"]
    out += ["", f"  성공률  {100.0 * n_ok / n:6.2f}%   ({n_ok}/{n})", "",
            "  실패 분류"]
    for k, v in cls.most_common():
        if k == "ok":
            continue
        out.append(f"    {k:<32} {v:>6}  ({100.0*v/n:5.2f}%)")
    if len(cls) == 1:
        out.append("    (없음)")

    # -- 충돌 (반드시 deadlock 과 짝으로) ------------------------------------ #
    #
    # **`status == "invalid"` 를 쓰면 안 된다.**  `bench/run.py` 가 `_fill` 에서
    # 검증 실패를 그렇게 표시한 뒤, 바로 아래에서 완주 실패면
    # `status = "unfinished:..."` 로 **덮어쓴다**.  그래서 이 값은 실제 배치에서
    # 항상 0 이고, 그것을 충돌률로 읽으면 **아무것도 검사하지 않는 지표**가 된다
    # (S3-B3 에서 발견 — 그 전까지의 "충돌 0건" 판정이 전부 이 열을 봤다).
    #
    # 대신 독립 검증기가 남긴 여유를 직접 본다.  `validate` 의 판정과 같은 식이다:
    #   에이전트 간  agent_clearance    < min_sep(i,j) - 1e-3
    #   장애물/벽    obstacle_clearance < radius       - 1e-3
    # 여기서는 전 에이전트가 같은 반경이라 상수로 비교할 수 있다.
    # S3-S §S-6: 판정은 **검증기가 적어 둔 열**로 한다.
    #     n_agent_violations == 0 AND n_obstacle_violations == 0
    # 그 열이 없는 옛 CSV 는 clearance 로 물러나되 그 사실을 밝힌다.
    a_lim = PHYSICS.min_sep - VIOL_TOL
    o_lim = PHYSICS.robot_radius - VIOL_TOL
    has_cols = any(fnum(r, "n_agent_violations") is not None for r in rows)
    if has_cols:
        hit_a = [r for r in rows if (fnum(r, "n_agent_violations") or 0) > 0]
        hit_o = [r for r in rows if (fnum(r, "n_obstacle_violations") or 0) > 0]
        basis = "n_agent_violations / n_obstacle_violations (검증기 직접)"
    else:
        hit_a = [r for r in rows
                 if (v := fnum(r, "agent_clearance")) is not None and v < a_lim]
        hit_o = [r for r in rows
                 if (v := fnum(r, "obstacle_clearance")) is not None and v < o_lim]
        basis = (f"clearance 열 (여유 {VIOL_TOL}) — 이 CSV 에는 판정 열이 없다 "
                 f"(S3-S 이전 스키마)")
    invalid = list({id(r): r for r in hit_a + hit_o}.values())
    dep = [d for d in (fnum(r, "dep_violations") for r in rows) if d]
    clr = [c for c in (fnum(r, "agent_clearance") for r in rows) if c is not None]
    oclr = [c for c in (fnum(r, "obstacle_clearance") for r in rows) if c is not None]
    dead = cls["deadlock"]
    out += ["",
            f"  충돌률   {100.0*len(invalid)/n:6.2f}%   ({len(invalid)}행)"
            f"      <- deadlock {100.0*dead/n:6.2f}% ({dead}행) 과 함께 읽을 것",
            f"    에이전트 간 {len(hit_a)}행,  장애물/벽 {len(hit_o)}행",
            f"    판정 근거: {basis}",
            f"    dep_violations 합계 {int(sum(dep))}",
            (f"    agent_clearance    최소 {min(clr):.4f} m  (하한 {PHYSICS.min_sep:.2f}),  "
             f"p10 {quantile(clr, .10):.4f}" if clr else "    agent_clearance 기록 없음"),
            (f"    obstacle_clearance 최소 {min(oclr):.4f} m  "
             f"(하한 {PHYSICS.robot_radius:.2f}),  p10 {quantile(oclr, .10):.4f}"
             if oclr else "    obstacle_clearance 기록 없음")]

    okrows = [r for r in rows if r.get("status") == "ok"]

    # -- makespan / 하한 대비 ------------------------------------------------ #
    tt = [v for v in (fnum(r, "team_time") for r in okrows) if v is not None]
    rb = [v for v in (fnum(r, "ratio_to_bound") for r in okrows) if v is not None]
    if tt:
        out += ["", "  성공한 행에서만",
                f"    makespan      중앙값 {quantile(tt, .50):7.2f}s  "
                f"p10 {quantile(tt, .10):6.2f}  p90 {quantile(tt, .90):6.2f}"]
    if rb:
        out.append(f"    ratio_to_bound 중앙값 {quantile(rb, .50):7.3f}   "
                   f"p90 {quantile(rb, .90):6.3f}")

    # -- 후보 선택의 질 ------------------------------------------------------ #
    cf = [r.get("chosen_is_fastest_feasible", "") for r in okrows]
    yes = sum(1 for v in cf if v == "True")
    known = sum(1 for v in cf if v in ("True", "False"))
    if known:
        out.append(f"    chosen_is_fastest_feasible  {100.0*yes/known:6.2f}% "
                   f"({yes}/{known})")
    nf = [v for v in (fnum(r, "n_feasible_modes") for r in okrows) if v is not None]
    nm = [v for v in (fnum(r, "n_modes") for r in okrows) if v is not None]
    if nf and nm:
        out.append(f"    t=0 의 feasible 후보  중앙값 {quantile(nf, .50):.1f}"
                   f" / {quantile(nm, .50):.0f}")

    # -- 가속도 (S3-A 에서 추가된 열) ---------------------------------------- #
    def col(name):
        return [v for v in (fnum(r, name) for r in okrows) if v is not None]

    av, ma = col("n_amax_violations"), col("max_accel")
    if ma:
        out += ["", "  가속도",
                f"    max_accel  중앙값 {quantile(ma, .50):6.2f}  "
                f"p90 {quantile(ma, .90):6.2f}  최대 {max(ma):6.2f} m/s^2"]
    if av:
        nz = [v for v in av if v > 0]
        tot = sum(av)
        out.append(f"    n_amax_violations 합계 {int(tot)},  "
                   f"0이 아닌 행 {len(nz)}/{len(av)} "
                   f"({100.0*len(nz)/len(av):.1f}%),  행당 최대 {int(max(av))}")
        if tot:
            out.append("      원인 분해 (배타적, 합 = 위 총계)")
            for key, label in (("n_amax_viol_stop",    "정지 (HOI dwell/도착)"),
                               ("n_amax_viol_block",   "후보 전멸 (기하)"),
                               ("n_amax_viol_snap",    "착지 스냅"),
                               ("n_amax_viol_project", "안전 투영  <- §A-2")):
                c = col(key)
                if c:
                    out.append(f"        {label:<24} {int(sum(c)):>7}  "
                               f"({100.0*sum(c)/tot:5.1f}%)")
        else:
            out.append("      (위반 0건)")
    pa, st_, bl = col("n_project_active"), col("steps"), col("n_blocked")
    fsc = col("n_free_steps_calls")
    steps_tot = sum(col("steps")) if col("steps") else 0.0
    if pa:
        out.append(f"    안전 투영 개입 {int(sum(pa))} 표본"
                   + (f",  후보 전멸 {int(sum(bl))} 표본" if bl else ""))
    if fsc:
        out.append(f"    free_steps 호출 {int(sum(fsc))}회"
                   + (f" ({sum(fsc) / max(sum(st_), 1.0):.2f}/agent-step)"
                      if st_ else ""))
    elif not ma:
        out.append("\n  가속도 열 없음 (S3-A 이전 스키마의 CSV)")

    # -- 플래너 시드 간 편차 -------------------------------------------------- #
    per_uid: Dict[str, List[float]] = defaultdict(list)
    for r in okrows:
        v = fnum(r, "team_time")
        if v is not None:
            per_uid[r["uid"]].append(v)
    spreads = [(max(v) - min(v)) / (quantile(v, .50) or 1.0)
               for v in per_uid.values() if len(v) == len(seeds)]
    if spreads:
        out += ["", f"  플래너 시드 간 makespan 상대 편차 "
                    f"((max-min)/중앙값), 세 시드 모두 성공한 {len(spreads)} 인스턴스",
                f"    중앙값 {100*quantile(spreads, .50):6.2f}%   "
                f"p90 {100*quantile(spreads, .90):6.2f}%   "
                f"최대 {100*max(spreads):6.2f}%"]

    # -- 재계획 주기 (1Hz 준수) ---------------------------------------------- #
    # **추론**: CSV 에 재계획 횟수 열이 없다.  재계획은 `replan_s`(=1.0s) 마다
    # 일어나므로 team_time / 1.0 을 횟수의 근사로 쓴다.  실측이 아니다.
    per_replan = [fnum(r, "runtime_s") / max(fnum(r, "team_time"), 1e-9)
                  for r in okrows
                  if fnum(r, "runtime_s") is not None and fnum(r, "team_time")]
    if per_replan:
        out += ["", "  재계획 1회당 벽시계 (추론: 재계획 횟수 ~= team_time/replan_s=1.0s)",
                f"    중앙값 {quantile(per_replan, .50):6.3f}s   "
                f"p90 {quantile(per_replan, .90):6.3f}s   "
                f"최대 {max(per_replan):6.3f}s",
                f"    1Hz(1.0s) 안에 든 비율 "
                f"{100.0*sum(1 for v in per_replan if v <= 1.0)/len(per_replan):.1f}%"]

    # -- 축별 성공률 ---------------------------------------------------------- #
    for axis in ("n_agents", "dep_mode", "size", "couple_prob"):
        if axis not in rows[0]:
            continue
        by: Dict[str, List[int]] = defaultdict(list)
        for r in rows:
            by[r[axis]].append(int(r.get("status") == "ok"))
        out.append("")
        out.append(f"  {axis} 별 성공률")
        for k in sorted(by, key=lambda x: (len(x), x)):
            v = by[k]
            out.append(f"    {k:<8} {100.0*sum(v)/len(v):6.2f}%  ({sum(v)}/{len(v)})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--method", default="wm_planner")
    args = ap.parse_args()
    for p in args.csv:
        print("\n".join(summarize(p, args.method)))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
