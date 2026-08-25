#!/usr/bin/env python3
"""집계 — run.py 가 쌓은 CSV 를 읽어 요약 표와 곡선을 낸다.

    python -m bench.analyze --csv bench/runs/scale.csv
    python -m bench.analyze --csv bench/runs/difficulty.csv --by coupled_waypoints

의도적으로 얇게 만들어 두었다.  무엇을 그릴지가 곧 B1 의 판단이고, 여기에
미리 답을 박아두면 그 판단을 빼앗는다.  지금 있는 것은 CSV 스키마가 실제로
쓸 만한지 확인하는 최소한이다.

  TODO(B1)
    - ratio_to_bound 의 *분포* (상자그림/바이올린).  평균만 보면 chain3 처럼
      seed 에 따라 25.2s 가 튀는 경우가 묻힌다.
    - 방법 간 짝지은 비교 (같은 인스턴스에서의 차이).  독립 평균 비교는 인스턴스
      난이도 편차에 묻힌다.
    - status 를 원인별로 쪼갠 실패 분해 (B6).
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np

METHOD_ORDER = ["lower_bound", "sequential", "coordination_astar", "wm_planner"]
COLOURS = {"lower_bound": "#111111", "sequential": "#B0B7BE",
           "coordination_astar": "#3C6BB0", "wm_planner": "#1a7f37"}


def load(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row: Dict[str, str], key: str) -> Optional[float]:
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def summarise(rows: List[Dict[str, str]], by: str) -> None:
    """method × <by> 격자로 성공률과 하한 대비 비율을 낸다."""
    keys = sorted({r[by] for r in rows if r.get(by)},
                  key=lambda s: (len(s), s))
    print(f"\n{'method':<20}{by:>8}{'n':>6}{'성공률':>9}"
          f"{'ratio 중앙':>12}{'ratio 평균':>12}{'최악':>8}{'cpu(s)':>9}")
    print("-" * 84)
    for m in METHOD_ORDER:
        for k in keys:
            sel = [r for r in rows if r["method"] == m and r.get(by) == k]
            if not sel:
                continue
            ok = [r for r in sel if r["status"] == "ok"]
            ratios = [x for x in (_f(r, "ratio_to_bound") for r in ok) if x]
            cpu = [x for x in (_f(r, "runtime_s") for r in ok) if x is not None]
            rate = f"{100 * len(ok) / len(sel):5.1f} %"
            if ratios:
                print(f"{m:<20}{k:>8}{len(sel):>6}{rate:>9}"
                      f"{np.median(ratios):>12.3f}{np.mean(ratios):>12.3f}"
                      f"{max(ratios):>8.3f}{np.mean(cpu):>9.2f}")
            else:
                print(f"{m:<20}{k:>8}{len(sel):>6}{rate:>9}"
                      f"{'-':>12}{'-':>12}{'-':>8}{'-':>9}")


def failures(rows: List[Dict[str, str]]) -> None:
    bad = [r for r in rows if r["status"] != "ok"]
    if not bad:
        print("\n실패 없음.")
        return
    print(f"\n실패 {len(bad)} / {len(rows)}")
    for (m, st), n in sorted(Counter((r["method"], r["status"]) for r in bad).items(),
                             key=lambda kv: -kv[1]):
        print(f"  {n:>5}  {m:<20} {st}")
    print("\n  재현 예시:")
    seen = set()
    for r in bad:
        key = (r["method"], r["status"])
        if key in seen:
            continue
        seen.add(key)
        print(f"    {r['uid']}  ({r['method']}, {r['status']})")


def curve(rows: List[Dict[str, str]], by: str, path: str) -> None:
    """<by> 를 x 축으로 한 성공률 곡선."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = sorted({r[by] for r in rows if r.get(by)}, key=lambda s: float(s))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for m in METHOD_ORDER:
        rate, ratio, xs = [], [], []
        for k in keys:
            sel = [r for r in rows if r["method"] == m and r.get(by) == k]
            if not sel:
                continue
            ok = [r for r in sel if r["status"] == "ok"]
            xs.append(float(k))
            rate.append(100 * len(ok) / len(sel))
            rs = [x for x in (_f(r, "ratio_to_bound") for r in ok) if x]
            ratio.append(np.median(rs) if rs else np.nan)
        if not xs:
            continue
        ax0.plot(xs, rate, "-o", ms=5, lw=1.8, color=COLOURS.get(m), label=m)
        if m != "lower_bound":
            ax1.plot(xs, ratio, "-o", ms=5, lw=1.8, color=COLOURS.get(m), label=m)

    # 그림의 글자는 영문으로 둔다 — matplotlib 기본 폰트에 한글이 없는 환경이
    # 흔하고, 곡선 하나 보려고 폰트를 설치하게 만들 이유가 없다.
    ax0.set_xlabel(by); ax0.set_ylabel("success rate (%)"); ax0.set_ylim(-4, 104)
    ax0.set_title("success rate", fontsize=11, loc="left")
    ax1.set_xlabel(by); ax1.set_ylabel("team time / lower bound")
    ax1.axhline(1.0, color="#111", lw=1.0, ls="--")
    ax1.set_title("ratio to bound (median)", fontsize=11, loc="left")
    for ax in (ax0, ax1):
        ax.grid(alpha=0.25); ax.set_axisbelow(True); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n곡선 -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--by", default="n_agents",
                    help="집계 축 (n_agents, dep_mode, coupled_waypoints, size, ...)")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    rows = load(args.csv)
    print(f"{args.csv}: {len(rows)} rows  "
          f"(commit {rows[0].get('git_commit', '?') if rows else '?'})")
    summarise(rows, args.by)
    failures(rows)
    if not args.no_fig:
        try:
            curve(rows, args.by,
                  os.path.splitext(args.csv)[0] + f"_{args.by}.png")
        except ValueError:
            print(f"\n(곡선 생략: '{args.by}' 가 수치 축이 아님)")


if __name__ == "__main__":
    main()
