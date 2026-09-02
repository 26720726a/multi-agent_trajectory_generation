#!/usr/bin/env python3
"""배치 CSV 두 개를 **짝지어** 비교한다.

    python scripts/compare_runs.py A.csv B.csv
    python scripts/compare_runs.py A.csv B.csv --method wm_planner --csv out.csv

계획서 §7.4 "총계가 아니라 짝지은 개별 대조" 를 도구로 못박는 것이다.
성공률의 차이만 보면 **어느 인스턴스가 좋아지고 어느 것이 나빠졌는지** 알 수
없고, 그 둘이 상쇄되면 "변화 없음" 으로 보인다.  그래서

  1. `(uid, method, planner_seed)` 로 조인하고,
  2. 성공/실패는 **전이표**로 (양쪽성공 / A만 / B만 / 양쪽실패),
  3. makespan 은 **양쪽 성공한 부분집합에서만** 짝지어 차이를 내고,
  4. 부호검정으로 win/loss 의 비대칭이 우연인지 보고,
  5. 중앙값이 0 이어도 분포(p10/p50/p90 + 히스토그램)를 함께 낸다.

`phys_fp` 도 함께 확인한다 (S0 §G-3 위험 3).  한 CSV 안에서 지문이 갈리면
그 파일 자체가 오염이므로 **거부하고 멈춘다**.  두 CSV 의 지문이 다른 것은
정상이다 — 물리를 바꿔 비교하는 것이 이 도구의 용도다.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

KEY = ("uid", "method", "planner_seed")


class MixedPhysics(RuntimeError):
    """한 CSV 안에 서로 다른 `phys_fp` 가 있다."""


# --------------------------------------------------------------------------- #
#  읽기
# --------------------------------------------------------------------------- #
def load(path: str) -> Tuple[Dict[Tuple[str, str, str], dict], str, List[str]]:
    """`(행 딕셔너리, 그 파일의 phys_fp, 그 파일의 git_commit 목록)`.

    지문이 갈리면 `MixedPhysics`.  옛 CSV 처럼 열이 아예 없으면 `"(없음)"` 을
    돌려주되 막지는 않는다 — 그런 파일과의 비교는 **가능하지만 위험하다**는 것을
    보고서에 적어야 하고, 그 판단은 사람 몫이다.
    """
    rows: Dict[Tuple[str, str, str], dict] = {}
    fps: Counter = Counter()
    commits: Counter = Counter()
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[tuple(r.get(k, "") for k in KEY)] = r     # type: ignore[index]
            fps[r.get("phys_fp", "")] += 1
            commits[r.get("git_commit", "")] += 1
    if not fps:
        return rows, "(빈 파일)", []
    if len(fps) > 1:
        raise MixedPhysics(
            f"{path}: 한 파일에 phys_fp 가 {len(fps)}개 섞여 있다 — "
            + ", ".join(f"{fp or '(빈칸)'}: {n}행" for fp, n in fps.most_common()))
    fp = next(iter(fps))
    return rows, fp or "(없음)", [c for c, _ in commits.most_common()]


def ok(row: dict) -> bool:
    """성공 = 독립 검증기까지 통과.

    `bench/run.py::_fill` 이 `rep.valid` 가 False 면 status 를 `invalid` 로
    바꾸므로, `status == "ok"` 는 "완주했고 검증도 통과" 와 같다.
    """
    return row.get("status", "") == "ok"


def fnum(row: dict, col: str) -> Optional[float]:
    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
#  통계
# --------------------------------------------------------------------------- #
def sign_test(wins: int, losses: int) -> float:
    """양측 부호검정 p-value.  동률은 세지 않는다 (전통적인 처리).

    정확 이항검정이다.  n 이 커도 파이썬 정수라 정밀도 손실이 없다.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    # 로그 공간에서 나눈다.  격자 전체(n ~ 4000)에서는 `2 ** n` 도 `tail` 도
    # float 로 못 옮기는 정수라, 곧이곧대로 나누면 OverflowError 가 난다.
    # `math.log` 는 큰 정수를 그대로 받는다.
    log_p = math.log(2.0) + math.log(tail) - n * math.log(2.0)
    return min(1.0, math.exp(log_p)) if log_p < 0.0 else 1.0


def quantile(xs: Sequence[float], q: float) -> float:
    """선형 보간 분위수.  numpy 없이 — 이 스크립트는 CSV 만 있으면 돌아야 한다."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def histogram(xs: Sequence[float], bins: int = 11, width: int = 44) -> List[str]:
    """중앙값이 0 이어도 분포는 볼 수 있어야 한다.

    "차이 없음" 과 "크게 좋아진 것과 크게 나빠진 것이 상쇄됨" 은 전혀 다른
    이야기인데, 중앙값만으로는 구별되지 않는다.
    """
    if not xs:
        return ["  (표본 없음)"]
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [f"  전부 {lo:+.3f} ({len(xs)}개)"]
    step = (hi - lo) / bins
    counts = [0] * bins
    for x in xs:
        counts[min(bins - 1, int((x - lo) / step))] += 1
    top = max(counts)
    out = []
    for i, c in enumerate(counts):
        a, b = lo + i * step, lo + (i + 1) * step
        bar = "#" * int(round(width * c / top)) if c else ""
        out.append(f"  [{a:+7.2f}, {b:+7.2f})  {c:>5}  {bar}")
    return out


# --------------------------------------------------------------------------- #
#  비교
# --------------------------------------------------------------------------- #
def transitions(a: dict, b: dict, keys: Sequence[tuple]) -> Counter:
    t: Counter = Counter()
    for k in keys:
        ra, rb = a[k], b[k]
        t[(ok(ra), ok(rb))] += 1
    return t


def paired(a: dict, b: dict, keys: Sequence[tuple], col: str,
           tol: float) -> Tuple[List[float], List[tuple]]:
    """양쪽 성공한 열쇠에서만 `b - a`.  (차이 목록, 그 열쇠들)"""
    diffs, used = [], []
    for k in keys:
        ra, rb = a[k], b[k]
        if not (ok(ra) and ok(rb)):
            continue
        va, vb = fnum(ra, col), fnum(rb, col)
        if va is None or vb is None:
            continue
        diffs.append(vb - va)
        used.append(k)
    return diffs, used


def report_method(name: str, a: dict, b: dict, keys: Sequence[tuple],
                  tol: float, top: int) -> List[str]:
    out = [f"\n### method = {name}   (짝지어진 열쇠 {len(keys)}개)"]

    # -- 1. 성공/실패 전이표 ------------------------------------------------ #
    t = transitions(a, b, keys)
    both = t[(True, True)]
    only_a = t[(True, False)]
    only_b = t[(False, True)]
    neither = t[(False, False)]
    n = len(keys) or 1
    out += [
        "",
        "  성공/실패 전이  (행=A, 열=B)",
        f"    {'':<12} {'B 성공':>8} {'B 실패':>8}",
        f"    {'A 성공':<12} {both:>8} {only_a:>8}",
        f"    {'A 실패':<12} {only_b:>8} {neither:>8}",
        "",
        f"    성공률  A {100*(both+only_a)/n:6.2f}%  ->  "
        f"B {100*(both+only_b)/n:6.2f}%   "
        f"({100*(only_b-only_a)/n:+.2f}%p)",
        f"    전이한 것만  A만성공 {only_a}  /  B만성공 {only_b}   "
        f"부호검정 p = {sign_test(only_b, only_a):.4g}",
    ]
    if only_a or only_b:
        out.append("    (성공률 차이는 이 두 숫자의 차이일 뿐이다 — 총계가 같아도 "
                   "서로 다른 인스턴스가 뒤바뀌었을 수 있다)")

    # -- 2. 양쪽 성공에서의 makespan 짝지은 차이 ---------------------------- #
    diffs, used = paired(a, b, keys, "team_time", tol)
    out.append("")
    out.append(f"  makespan 짝지은 차이 (B - A), 양쪽 성공한 {len(diffs)}개에서만")
    if not diffs:
        out.append("    표본 없음")
        return out
    wins = sum(1 for d in diffs if d < -tol)      # B 가 더 빠르다
    losses = sum(1 for d in diffs if d > tol)
    ties = len(diffs) - wins - losses
    p = sign_test(wins, losses)
    out += [
        f"    B 승 {wins}  /  A 승 {losses}  /  동률 {ties}      "
        f"부호검정 p = {p:.4g}",
        f"    평균 {sum(diffs)/len(diffs):+.3f}s   "
        f"p10 {quantile(diffs, .10):+.3f}   중앙값 {quantile(diffs, .50):+.3f}   "
        f"p90 {quantile(diffs, .90):+.3f}",
        f"    최선 {min(diffs):+.3f}s   최악 {max(diffs):+.3f}s",
        "",
        "  분포",
    ]
    out += histogram(diffs)
    if abs(quantile(diffs, .50)) <= tol:
        out.append("    (중앙값이 0 이다 — 위 분포와 p10/p90 을 함께 읽을 것.  "
                   "'변화 없음' 과 '상쇄' 는 다르다)")

    if top:
        order = sorted(zip(diffs, used), key=lambda x: x[0])
        out.append("")
        out.append(f"  가장 크게 좋아진 {top}개 / 나빠진 {top}개")
        for d, k in order[:top]:
            out.append(f"    {d:+8.3f}s  {k[0]} ps={k[2]}")
        for d, k in order[-top:][::-1]:
            out.append(f"    {d:+8.3f}s  {k[0]} ps={k[2]}")
    return out


def summarise(label: str, rows: dict, fp: str) -> List[str]:
    """한 CSV 만 보고 알 수 있는 것 — 게이트 판정에 쓰는 절대 수치들."""
    by_status: Counter = Counter(r.get("status", "") for r in rows.values())
    invalid = [r for r in rows.values() if r.get("status") == "invalid"]
    clr = [c for c in (fnum(r, "agent_clearance") for r in rows.values())
           if c is not None]
    dep = [d for d in (fnum(r, "dep_violations") for r in rows.values())
           if d is not None]
    amax = [v for v in (fnum(r, "n_amax_violations") for r in rows.values())
            if v is not None]
    out = [f"  {label}: {len(rows)}행,  phys_fp = {fp}",
           "    status: " + ", ".join(f"{k or '(빈칸)'}={v}"
                                      for k, v in by_status.most_common())]
    out.append(f"    검증 실패(invalid) {len(invalid)}행"
               + ("  <-- 충돌/위반 후보" if invalid else ""))
    if clr:
        out.append(f"    agent_clearance 최소 {min(clr):.4f} m"
                   f"  (기록된 행 {len(clr)})")
    if dep:
        out.append(f"    dep_violations 합계 {int(sum(dep))}")
    if amax:
        nz = [v for v in amax if v > 0]
        out.append(f"    n_amax_violations 합계 {int(sum(amax))}, "
                   f"0이 아닌 행 {len(nz)}/{len(amax)}, 최대 {int(max(amax))}")
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="기준선 CSV")
    ap.add_argument("b", help="비교 대상 CSV")
    ap.add_argument("--method", default="wm_planner",
                    help="비교할 method.  'all' 이면 공통 method 전부")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="동률로 볼 makespan 차이 (기본 1e-9)")
    ap.add_argument("--top", type=int, default=5,
                    help="가장 크게 변한 인스턴스를 몇 개까지 보일지 (0=끔)")
    ap.add_argument("--csv", default=None,
                    help="짝지은 차이를 이 경로에 CSV 로 쓴다")
    args = ap.parse_args()

    try:
        A, fp_a, ca = load(args.a)
        B, fp_b, cb = load(args.b)
    except MixedPhysics as exc:
        print(f"\n거부: {exc}\n\n한 파일 안에서 물리가 갈리면 그 파일로는 아무것도 "
              f"결론지을 수 없다.", file=sys.stderr)
        return 2

    print("=" * 78)
    print(f"A = {args.a}")
    print(f"B = {args.b}")
    print("=" * 78)
    print("\n## 각 CSV 단독 요약")
    print("\n".join(summarise("A", A, fp_a)))
    print("\n".join(summarise("B", B, fp_b)))
    print(f"\n  물리 지문: A {fp_a}  vs  B {fp_b}   "
          + ("**같다** (config 값이 같다)" if fp_a == fp_b
             else "다르다 (물리를 바꾼 비교 — 의도한 것인지 확인할 것)"))
    print(f"  git commit: A {', '.join(ca) or '(없음)'}  vs  "
          f"B {', '.join(cb) or '(없음)'}")
    if fp_a == fp_b and ca and cb and set(ca) != set(cb):
        print("\n  " + "!" * 68)
        print("  !! 물리 지문은 같은데 **코드가 다르다.**")
        print("  !! `phys_fp` 는 config 값만 해싱하므로 알고리즘 변경을 잡지 못한다")
        print("  !! (S3-A\' §9-4).  두 실행의 차이는 물리가 아니라 **코드**다 —")
        print("  !! 위의 '같다' 를 '같은 조건' 으로 읽으면 안 된다.")
        print("  " + "!" * 68)
    elif fp_a != fp_b and ca and cb and set(ca) != set(cb):
        print("  (주의: 물리도 코드도 둘 다 다르다 — 축이 하나가 아니다)")

    common = sorted(set(A) & set(B))
    only_a, only_b = len(A) - len(common), len(B) - len(common)
    print(f"\n## 조인\n  공통 열쇠 {len(common)},  A 에만 {only_a},  B 에만 {only_b}")
    if only_a or only_b:
        print("  (한쪽에만 있는 행은 짝지은 비교에서 빠진다 — 격자가 다르다는 뜻이므로 "
              "왜 다른지 확인할 것)")

    methods = ([args.method] if args.method != "all"
               else sorted({k[1] for k in common}))
    lines: List[str] = []
    for m in methods:
        keys = [k for k in common if k[1] == m]
        if not keys:
            lines.append(f"\n### method = {m}: 공통 행 없음")
            continue
        lines += report_method(m, A, B, keys, args.tol, args.top)
    print("\n".join(lines))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["uid", "method", "planner_seed", "ok_a", "ok_b",
                        "team_time_a", "team_time_b", "diff"])
            for k in common:
                ra, rb = A[k], B[k]
                va, vb = fnum(ra, "team_time"), fnum(rb, "team_time")
                w.writerow([k[0], k[1], k[2], ok(ra), ok(rb), va, vb,
                            None if (va is None or vb is None) else vb - va])
        print(f"\n짝지은 표 -> {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
