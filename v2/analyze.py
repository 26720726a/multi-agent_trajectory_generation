#!/usr/bin/env python3
"""집계 — run.py 가 쌓은 CSV 를 읽어 **커밋해서 diff 를 뜰 수 있는** 보고서를 낸다.

    python -m bench.analyze --csv bench/runs/difficulty.csv
    python -m bench.analyze --csv bench/runs/difficulty.csv > results/bench_v0.txt
    python -m bench.analyze --csv bench/runs/difficulty.csv --by dep_mode --fig

주 산출물은 **텍스트**다 (`results/check_wm_v0.txt` 와 같은 성격).  PNG 는
`--fig` 를 줄 때만 만드는 부수 산출물이다.

왜 텍스트가 주인가
------------------
A1(mode 열거 제거) 과 A4(샘플링 개선) 의 성공지표를 B 가 재기로 되어 있는데,
그림으로는 "좋아졌다" 를 리뷰에서 보일 수 없다.  같은 명령으로 텍스트를 다시
뽑아 `git diff` 를 뜨면 개선이 **줄 단위로** 드러난다.

그래서 이 파일의 제1 제약은 **같은 CSV -> 같은 바이트** 다:

* 실행 시각·절대 경로·소요 시간을 본문에 넣지 않는다.  파일명은 basename 만.
* 모든 정렬을 명시한다.  dict 순회 순서나 groupby 결과 순서에 기대지 않는다.
* 부동소수는 자릿수를 고정한다 (비율 3자리, 초 2자리, 퍼센트 1자리, ms 1자리).
* 행이 없는 섹션도 제목과 "해당 없음" 을 남긴다.  섹션이 통째로 사라지면
  그 뒤가 전부 밀려 diff 가 못 쓰게 된다.

  TODO(B2) [6] 의 손실 분포는 CSV 에 열이 없어 아직 못 낸다.  자세한 것은
    `SECTION6_MISSING` 주석 참조.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

METHOD_ORDER = ["lower_bound", "sequential", "coordination_astar", "wm_planner"]
COLOURS = {"lower_bound": "#111111", "sequential": "#B0B7BE",
           "coordination_astar": "#3C6BB0", "wm_planner": "#1a7f37"}

#: 보고서가 읽는 열.  없으면 시작도 하지 않는다 — 절반쯤 그린 표를 내놓느니
#: 무엇이 없는지 말하고 멈추는 편이 낫다.
REQUIRED = ("uid", "method", "status", "team_time", "runtime_s",
            "ratio_to_bound", "git_commit", "planner_seed")

#: 방법의 실패가 아니라 계산 자원의 한계.  성공률과 섞어 읽으면 안 된다.
RESOURCE_STATUSES = ("killed", "oom", "timeout")

#: [6] 이 필요로 하지만 run.py 가 아직 쓰지 않는 열.
#:
#: `chosen_is_fastest_feasible` 은 참/거짓만 말한다.  그런데 2단계에서 실제로
#: 관측된 False 는 total 차이 0.001 에 makespan 손실 0.0062s 였다 — soft 항이
#: 선택을 "바꿨다" 와 "거의 동점이라 반올림 수준 차이가 승부를 갈랐다" 가
#: 구별되지 않는다.  구별하려면 그 손실(chosen.makespan - fastest.makespan) 을
#: run.py 가 행에 적어야 한다.  열이 생기면 이 이름으로 읽어 자동으로 채운다.
LOSS_COLUMN = "makespan_loss_s"

#: 손실이 이보다 크면 "동점 처리" 로 보기 어렵다 (초).
LOSS_NOTABLE_S = 0.010

#: 짝지은 비교에서 무승부로 볼 차이 (초).  team_time 은 CSV 에 3자리로 적힌다.
TIE_TOL_S = 1e-6

#: [3] 의 기준선.  A4 가 이 숫자를 낮추기로 되어 있다.
SEED_SPREAD_BASELINE = ("chain3 에서 2.71s (results/check_wm_v0.txt), "
                        "A4 목표 0.5s 이하")


# --------------------------------------------------------------------------- #
#  읽기
# --------------------------------------------------------------------------- #
class MissingColumns(RuntimeError):
    """CSV 에 보고서가 읽어야 할 열이 없다."""


def load(path: str) -> Tuple[List[Dict[str, str]], int]:
    """CSV 를 읽는다.  반환 `(온전한 행, 버린 행 수)`.

    배치가 **도는 중에도** 읽을 수 있어야 한다 (17시간짜리 실행의 중간 점검).
    그때 마지막 줄은 쓰이는 중이라 잘려 있다.  `DictReader` 는 모자란 칸을
    None 으로 채우는데, 그걸 그대로 집계하면 method=None 같은 행이 표에 섞인다.
    그래서 필수 열이 하나라도 비어 있는 행은 버리고, **몇 행을 버렸는지 적는다**
    — 조용히 버리면 보고서의 rows 와 CSV 의 줄 수가 안 맞는 이유를 알 수 없다.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, restkey="__extra__")
        header = reader.fieldnames or []
        raw = list(reader)
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        raise MissingColumns(
            f"이 CSV 에는 보고서가 읽어야 할 열이 없다: {missing}\n"
            f"  있는 열 ({len(header)}개): {header}\n"
            f"  bench/run.py 2단계 이전에 만들어진 CSV 로 보인다. "
            f"같은 config 를 지금 코드로 다시 돌려야 한다.")
    rows = [r for r in raw
            if "__extra__" not in r and all(r.get(c) is not None
                                            for c in REQUIRED)]
    return rows, len(raw) - len(rows)


def num(row: Dict[str, str], key: str) -> Optional[float]:
    """CSV 칸을 float 로.  빈 칸·결측·NaN·무한대는 전부 None 이다.

    `if x:` 로 거르지 않는 이유: 그러면 0.0 이 결측과 함께 사라진다.
    ratio_to_bound 에서 그건 "하한과 정확히 같았다" 를 "잰 적 없다" 로
    바꿔치기하는 일이고, 분포의 왼쪽 끝이 조용히 잘린다.
    """
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def is_ok(row: Dict[str, str]) -> bool:
    return row.get("status") == "ok"


# --------------------------------------------------------------------------- #
#  서식 — 한글은 두 칸을 차지한다
# --------------------------------------------------------------------------- #
def _w(s: str) -> int:
    """터미널에서 차지하는 칸 수.  한글·기호는 2칸이다."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, width: int, right: bool = True) -> str:
    gap = " " * max(0, width - _w(s))
    return gap + s if right else s + gap


def table(headers: Sequence[str], widths: Sequence[int],
          rights: Sequence[bool], rows: Sequence[Sequence[str]]) -> List[str]:
    """고정폭 표.  머리글 + 대시줄 + 본문."""
    head = " ".join(_pad(h, w, r) for h, w, r in zip(headers, widths, rights))
    out = [head, "-" * _w(head)]
    for row in rows:
        out.append(" ".join(_pad(str(c), w, r)
                            for c, w, r in zip(row, widths, rights)).rstrip())
    return out


def fpct(x: float) -> str:
    return f"{x:.1f} %"


def fratio(x: float) -> str:
    return f"{x:.3f}"


def fsec(x: float) -> str:
    return f"{x:.2f}"


def fms(x: float) -> str:
    return f"{1000.0 * x:.1f}"


def quant(values: Sequence[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), p))


def axis_key(value: str) -> Tuple[int, float, str]:
    """축 값의 정렬 순서.  숫자는 숫자로, 나머지는 사전순, 빈 값은 맨 뒤."""
    if value == "":
        return (2, 0.0, "")
    try:
        return (0, float(value), value)
    except ValueError:
        return (1, 0.0, value)


def axis_values(rows: Sequence[Dict[str, str]], by: str) -> List[str]:
    return sorted({r.get(by, "") for r in rows}, key=axis_key)


def methods_present(rows: Sequence[Dict[str, str]]) -> List[str]:
    seen = {r["method"] for r in rows}
    known = [m for m in METHOD_ORDER if m in seen]
    return known + sorted(seen - set(METHOD_ORDER) - {"-"})


def uid_sample(uids: Sequence[str], k: int) -> str:
    """재현용 uid 몇 개.  정렬해서 뽑아야 같은 CSV 에 같은 목록이 나온다."""
    picked = sorted(set(uids))[:k]
    return " ".join(picked) if picked else "-"


# --------------------------------------------------------------------------- #
#  머리말
# --------------------------------------------------------------------------- #
def header_lines(rows: Sequence[Dict[str, str]], csv_name: str,
                 dropped: int = 0) -> List[str]:
    commits = sorted({r["git_commit"] for r in rows if r.get("git_commit")})
    tag = commits[0] if len(commits) == 1 else f"{len(commits)}개 섞임"
    out = [f"Mahoi-WM bench report   csv={csv_name}  commit={tag}  "
           f"rows={len(rows)}"]
    if dropped:
        out.append("")
        out.append(f"  참고: 잘린 행 {dropped}개를 뺐다 "
                   f"(배치가 도는 중에 읽으면 마지막 줄이 쓰이는 중이다).")
    if len(commits) > 1:
        out.append("")
        out.append(f"  경고: 한 CSV 에 커밋이 {len(commits)}개 섞여 있다. "
                   f"배치 도중 코드가 바뀌었다는 뜻이므로,")
        out.append("        아래 숫자는 단일 버전의 성능이 아니다.")
        for c in commits:
            n = sum(1 for r in rows if r.get("git_commit") == c)
            out.append(f"          {c}  {n} rows")
    return out


# --------------------------------------------------------------------------- #
#  [1] 방법별 요약
# --------------------------------------------------------------------------- #
def section_summary(rows: Sequence[Dict[str, str]]) -> Tuple[List[str], Dict]:
    out = ["[1] 방법별 요약",
           "  ratio = team_time / lower_bound.  평균이 아니라 분위수로 적는다 —",
           "  평균은 chain3 의 19.4/19.5/25.2 를 21.4 하나로 뭉개 튐을 숨긴다."]
    body, excluded, facts = [], 0, {}
    for m in methods_present(rows):
        sel = [r for r in rows if r["method"] == m]
        ok = [r for r in sel if is_ok(r)]
        # num() 이 None 을 돌려준 행만 뺀다.  falsy 검사로 거르면 ratio==0.0
        # (하한과 정확히 같았다) 이 결측과 함께 사라진다.
        ratios = [v for v in (num(r, "ratio_to_bound") for r in ok)
                  if v is not None]
        miss = len(ok) - len(ratios)
        excluded += miss
        cpu = [v for v in (num(r, "runtime_s") for r in ok) if v is not None]
        rate = 100.0 * len(ok) / len(sel) if sel else 0.0
        if m == "wm_planner":
            facts = {"rate": rate, "p50": quant(ratios, 50) if ratios else None}
        body.append([
            m, str(len(sel)), fpct(rate),
            fratio(quant(ratios, 50)) if ratios else "-",
            fratio(quant(ratios, 25)) if ratios else "-",
            fratio(quant(ratios, 75)) if ratios else "-",
            fratio(max(ratios)) if ratios else "-",
            fsec(float(np.mean(cpu))) if cpu else "-",
        ])
    if not body:
        out.append("  해당 없음")
        return out, facts
    out += table(["method", "n", "성공률", "ratio p50", "p25", "p75", "최악",
                  "cpu(s)"],
                 [18, 6, 8, 9, 7, 7, 7, 8],
                 [False, True, True, True, True, True, True, True], body)
    out.append(f"  ratio 결측으로 분포에서 뺀 ok 행: {excluded}")
    return out, facts


# --------------------------------------------------------------------------- #
#  [2] 난이도 축별 성공률
# --------------------------------------------------------------------------- #
def section_by_axis(rows: Sequence[Dict[str, str]], by: str) -> List[str]:
    out = [f"[2] 난이도 축별 성공률  (축: {by})",
           "  어디서 무너지는가.  이 표가 논문의 '적용 범위' 문장이 된다."]
    body = []
    for m in methods_present(rows):
        for v in axis_values(rows, by):
            sel = [r for r in rows if r["method"] == m and r.get(by, "") == v]
            if not sel:
                continue
            ok = [r for r in sel if is_ok(r)]
            ratios = [x for x in (num(r, "ratio_to_bound") for r in ok)
                      if x is not None]
            body.append([m, v or "-", str(len(sel)),
                         fpct(100.0 * len(ok) / len(sel)),
                         fratio(quant(ratios, 50)) if ratios else "-"])
    if not body:
        out.append("  해당 없음")
        return out
    out += table(["method", by, "n", "성공률", "ratio p50"],
                 [18, 10, 6, 8, 9],
                 [False, True, True, True, True], body)
    return out


# --------------------------------------------------------------------------- #
#  [3] 플래너 시드 편차 — A4 의 계기판
# --------------------------------------------------------------------------- #
def section_seed_spread(rows: Sequence[Dict[str, str]], by: str
                        ) -> Tuple[List[str], Optional[float]]:
    out = ["[3] 플래너 시드 편차  — A4 의 계기판",
           "  같은 문제를 놓고 planner_seed 만 바꿨을 때 team_time 이 얼마나 흔들리나.",
           f"  기준선: {SEED_SPREAD_BASELINE}",
           "  표준편차는 모집단 기준(ddof=0) — check_wm.py 의 2.71s 와 같은 계산이다."]

    wm = [r for r in rows if r["method"] == "wm_planner"]
    seeds = {r["planner_seed"] for r in wm}
    if len(seeds) < 2:
        out.append(f"  planner_seeds 가 {len(seeds)}개라 산출 불가.")
        return out, None

    by_uid: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in wm:
        by_uid[r["uid"]].append(r)

    spreads: Dict[str, Tuple[float, str]] = {}
    partial, all_failed = [], []
    for uid in sorted(by_uid):
        group = by_uid[uid]
        ok = [r for r in group if is_ok(r)]
        if not ok:
            all_failed.append(uid)
            continue
        if len(ok) != len(group):
            partial.append(uid)
            continue                    # 편차가 아니라 성공률 문제다
        times = [x for x in (num(r, "team_time") for r in ok) if x is not None]
        if len(times) < 2:
            continue
        spreads[uid] = (float(np.std(times)), group[0].get(by, ""))

    if not spreads:
        out.append("  모든 시드가 성공한 인스턴스가 없어 산출 불가.")
    else:
        body = []
        for v in axis_values(rows, by):
            vals = sorted(s for s, av in spreads.values() if av == v)
            if not vals:
                continue
            body.append([v or "-", str(len(vals)), fsec(quant(vals, 50)),
                         fsec(quant(vals, 90)), fsec(max(vals))])
        allv = sorted(s for s, _ in spreads.values())
        body.append(["(전체)", str(len(allv)), fsec(quant(allv, 50)),
                     fsec(quant(allv, 90)), fsec(max(allv))])
        out += table([by, "n", "sd p50", "sd p90", "sd 최대"],
                     [10, 6, 8, 8, 8], [True, True, True, True, True], body)

    out.append("")
    out.append("  편차 상위 5 — Track A 가 재현할 대상")
    top = sorted(spreads.items(), key=lambda kv: (-kv[1][0], kv[0]))[:5]
    if not top:
        out.append("    해당 없음")
    else:
        for uid, (sd, av) in top:
            out.append(f"    {uid:<34} sd={fsec(sd)}s  {by}={av or '-'}")
    out.append(f"  일부 시드만 성공한 인스턴스: {len(partial)}  "
               f"(편차가 아니라 성공률 문제이므로 위 분포에서 뺐다)")
    out.append(f"  모든 시드가 실패한 인스턴스: {len(all_failed)}")
    p90 = quant(sorted(s for s, _ in spreads.values()), 90) if spreads else None
    return out, p90


# --------------------------------------------------------------------------- #
#  [4] 짝지은 비교
# --------------------------------------------------------------------------- #
def section_paired(rows: Sequence[Dict[str, str]]) -> List[str]:
    out = ["[4] 짝지은 비교  (wm_planner - baseline, 같은 uid)",
           "  독립 평균 비교는 인스턴스 난이도 편차에 묻힌다.  같은 문제끼리만 뺀다.",
           "  wm_planner 가 planner_seed 여러 개면 그 인스턴스의 team_time",
           "  **중앙값**을 대표값으로 쓴다.",
           "  양쪽 다 status=ok 인 쌍만 센다."]

    wm_by_uid: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r["method"] == "wm_planner" and is_ok(r):
            v = num(r, "team_time")
            if v is not None:
                wm_by_uid[r["uid"]].append(v)
    wm_rep = {u: float(np.median(v)) for u, v in wm_by_uid.items() if v}

    body = []
    for base in [m for m in METHOD_ORDER if m != "wm_planner"]:
        base_rows = [r for r in rows if r["method"] == base]
        if not base_rows:
            continue
        diffs, win, tie, loss = [], 0, 0, 0
        for r in sorted(base_rows, key=lambda r: r["uid"]):
            if not is_ok(r) or r["uid"] not in wm_rep:
                continue
            b = num(r, "team_time")
            if b is None:
                continue
            d = wm_rep[r["uid"]] - b
            diffs.append(d)
            if d < -TIE_TOL_S:
                win += 1
            elif d > TIE_TOL_S:
                loss += 1
            else:
                tie += 1
        if not diffs:
            body.append([base, "0", "-", "-", "-", "-"])
            continue
        body.append([base, str(len(diffs)), f"{win}/{tie}/{loss}",
                     fsec(float(np.median(diffs))),
                     fsec(quant(diffs, 25)), fsec(quant(diffs, 75))])
    if not body:
        out.append("  해당 없음")
        return out
    out += table(["baseline", "쌍", "승/무/패", "Δ 중앙(s)", "Δ p25", "Δ p75"],
                 [18, 6, 12, 10, 8, 8],
                 [False, True, True, True, True, True], body)
    out.append("  승 = wm_planner 가 더 빠름.")

    # -- 최적 baseline 과의 비교는 따로 강조한다 -------------------------- #
    out.append("")
    out.append("  coordination_astar 대비 — 유일한 **최적** baseline 이므로,")
    out.append("  '최적 대비 몇 % 손해' 는 여기서만 나온다.")
    astar = [r for r in rows if r["method"] == "coordination_astar"]
    if not astar:
        out.append("    해당 없음 (CSV 에 coordination_astar 행이 없다)")
        return out
    skipped = sum(1 for r in astar if r["status"] == "skipped:lattice_too_large")
    pens = []
    for r in sorted(astar, key=lambda r: r["uid"]):
        if not is_ok(r) or r["uid"] not in wm_rep:
            continue
        b = num(r, "team_time")
        if b is None or b <= 0:
            continue
        pens.append(100.0 * (wm_rep[r["uid"]] - b) / b)
    if not pens:
        out.append("    비교 가능한 쌍이 없다.")
    else:
        out.append(f"    쌍 {len(pens)}  손해 p50 {fpct(quant(pens, 50))}  "
                   f"p90 {fpct(quant(pens, 90))}  최악 {fpct(max(pens))}")
    out.append(f"    격자 폭발로 빠진 인스턴스(skipped:lattice_too_large): {skipped}")
    return out


# --------------------------------------------------------------------------- #
#  [5] 실패 분해
# --------------------------------------------------------------------------- #
def section_failures(rows: Sequence[Dict[str, str]], by: str) -> List[str]:
    out = ["[5] 실패 분해",
           "  (method, status) x 난이도 축.  재현용 uid 가 Track A 의 작업 목록이다.",
           "  status 는 알파벳순으로 고정한다."]
    bad = [r for r in rows if not is_ok(r)]
    if not bad:
        out.append("  해당 없음 (실패 0건)")
    else:
        buckets: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        for r in bad:
            buckets[(r["method"], r["status"], r.get(by, ""))].append(r["uid"])
        body = []
        for m in methods_present(bad):
            keys = sorted((k for k in buckets if k[0] == m),
                          key=lambda k: (k[1], axis_key(k[2])))
            for k in keys:
                body.append([m, k[1], k[2] or "-", str(len(buckets[k])),
                             uid_sample(buckets[k], 3)])
        out += table(["method", "status", by, "n", "재현 uid (최대 3)"],
                     [18, 38, 8, 6, 40],
                     [False, False, True, True, False], body)

    out.append("")
    out.append("  자원 한계 — 방법의 실패가 아니다.  성공률과 섞어 읽지 말 것.")
    res = [r for r in rows if r["status"] in RESOURCE_STATUSES]
    if not res:
        out.append("    해당 없음")
    else:
        for st in RESOURCE_STATUSES:
            sel = [r for r in res if r["status"] == st]
            if sel:
                out.append(f"    {st:<10} {len(sel):>5}  "
                           f"{uid_sample([r['uid'] for r in sel], 3)}")
    return out


# --------------------------------------------------------------------------- #
#  [6] cost 항이 선택에 관여했는가 — B2
# --------------------------------------------------------------------------- #
def section_cost_terms(rows: Sequence[Dict[str, str]], by: str
                       ) -> Tuple[List[str], Optional[float], Optional[float]]:
    out = ["[6] cost 항이 선택에 관여했는가  — B2",
           "  chosen_is_fastest_feasible=False 는 soft 항 6개가 makespan 을 사고",
           "  다른 것을 샀다는 뜻이다.  다만 비율만으로는 '선택을 바꿨다' 와",
           "  '거의 동점이라 반올림 수준 차이가 승부를 갈랐다' 가 구별되지 않는다.",
           "  해석 기준: 손실이 전부 수 ms 수준이면 그 6개 항은 실질적으로",
           "  동점 처리기이며, 그것은 '검증되지 않았다' 보다 강한 결론이다."]

    wm = [r for r in rows if r["method"] == "wm_planner"]
    judged = [r for r in wm if r.get("chosen_is_fastest_feasible") in
              ("True", "False")]
    if not judged:
        out.append("  해당 없음 (chosen_is_fastest_feasible 이 채워진 행이 없다)")
        return out, None, None

    n_false = sum(1 for r in judged
                  if r["chosen_is_fastest_feasible"] == "False")
    overall = 100.0 * n_false / len(judged)
    body = []
    for v in axis_values(judged, by):
        sel = [r for r in judged if r.get(by, "") == v]
        if not sel:
            continue
        f = sum(1 for r in sel if r["chosen_is_fastest_feasible"] == "False")
        body.append([v or "-", str(len(sel)), str(f),
                     fpct(100.0 * f / len(sel))])
    body.append(["(전체)", str(len(judged)), str(n_false), fpct(overall)])
    out += table([by, "판정된 행", "False", "관여율"],
                 [10, 10, 8, 8], [True, True, True, True], body)

    out.append("")
    out.append("  False 인 행의 makespan 손실 (chosen - fastest)")
    losses = [x for x in (num(r, LOSS_COLUMN) for r in judged
                          if r["chosen_is_fastest_feasible"] == "False")
              if x is not None]
    if not losses:
        # SECTION6_MISSING — 열이 없으면 여기서 멈춘다.  비율만으로 결론을
        # 내리면 정확히 위에서 경고한 혼동을 저지르게 된다.
        out.append(f"    산출 불가 — CSV 에 '{LOSS_COLUMN}' 열이 없다.")
        out.append("    chosen_is_fastest_feasible 은 참/거짓만 말하므로,")
        out.append("    손실이 6ms 인지 6s 인지 이 CSV 만으로는 알 수 없다.")
        out.append("    run.py 가 그 차이를 행에 적어야 채워진다.")
        return out, overall, None

    out.append(f"    n={len(losses)}  p50 {fms(quant(losses, 50))} ms  "
               f"p90 {fms(quant(losses, 90))} ms  최대 {fms(max(losses))} ms")
    notable = sorted({r["uid"] for r in judged
                      if r["chosen_is_fastest_feasible"] == "False"
                      and (num(r, LOSS_COLUMN) or 0.0) > LOSS_NOTABLE_S})[:5]
    out.append(f"    손실 {1000 * LOSS_NOTABLE_S:.0f}ms 초과 사례: "
               f"{' '.join(notable) if notable else '없음'}")
    return out, overall, quant(losses, 90)


# --------------------------------------------------------------------------- #
#  보고서
# --------------------------------------------------------------------------- #
def build_report(rows: Sequence[Dict[str, str]], by: str, csv_name: str,
                 dropped: int = 0) -> str:
    out: List[str] = []
    out += header_lines(rows, csv_name, dropped)
    out.append("")

    s1, facts = section_summary(rows)
    out += s1 + [""]
    out += section_by_axis(rows, by) + [""]
    s3, sd_p90 = section_seed_spread(rows, by)
    out += s3 + [""]
    out += section_paired(rows) + [""]
    out += section_failures(rows, by) + [""]
    s6, involved, loss_p90 = section_cost_terms(rows, by)
    out += s6 + [""]

    rate = fpct(facts["rate"]) if facts.get("rate") is not None else "n/a"
    p50 = fratio(facts["p50"]) if facts.get("p50") is not None else "n/a"
    sd = f"{fsec(sd_p90)}s" if sd_p90 is not None else "n/a"
    inv = fpct(involved) if involved is not None else "n/a"
    loss = f" (손실 p90 {fms(loss_p90)}ms)" if loss_p90 is not None \
        else " (손실 미기록)"
    out.append(f"{len(rows)} rows: wm_planner {rate} 성공, ratio p50 {p50}, "
               f"시드 편차 p90 {sd}, cost 항 관여 {inv}{loss}")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
#  그림 (부수 산출물)
# --------------------------------------------------------------------------- #
def curve(rows: Sequence[Dict[str, str]], by: str, path: str) -> None:
    """<by> 를 x 축으로 한 성공률 곡선."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [v for v in axis_values(rows, by) if v != ""]
    try:
        xs_all = [float(k) for k in keys]
    except ValueError:
        xs_all = list(range(len(keys)))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for m in methods_present(rows):
        rate, ratio, xs = [], [], []
        for k, x in zip(keys, xs_all):
            sel = [r for r in rows if r["method"] == m and r.get(by) == k]
            if not sel:
                continue
            ok = [r for r in sel if is_ok(r)]
            xs.append(x)
            rate.append(100.0 * len(ok) / len(sel))
            rs = [v for v in (num(r, "ratio_to_bound") for r in ok)
                  if v is not None]
            ratio.append(float(np.median(rs)) if rs else np.nan)
        if not xs:
            continue
        ax0.plot(xs, rate, "-o", ms=5, lw=1.8, color=COLOURS.get(m), label=m)
        if m != "lower_bound":
            ax1.plot(xs, ratio, "-o", ms=5, lw=1.8, color=COLOURS.get(m),
                     label=m)

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--by", default="n_agents",
                    help="집계 축 (n_agents, dep_mode, size, couple_prob, ...)")
    ap.add_argument("--out", default=None,
                    help="텍스트 경로 (기본: CSV 옆 <이름>_report.txt)")
    ap.add_argument("--fig", action="store_true",
                    help="PNG 도 만든다 (기본은 만들지 않는다)")
    args = ap.parse_args()

    try:
        rows, dropped = load(args.csv)
    except MissingColumns as exc:
        print(f"\n{args.csv}:\n{exc}", file=sys.stderr)
        raise SystemExit(2)

    if rows and args.by not in rows[0]:
        print(f"\n--by {args.by} 는 이 CSV 에 없는 열이다.\n"
              f"  있는 열: {sorted(rows[0])}", file=sys.stderr)
        raise SystemExit(2)

    report = build_report(rows, args.by, os.path.basename(args.csv), dropped)
    sys.stdout.write(report)                 # 리다이렉트해도 이게 전부다

    out = args.out or os.path.splitext(args.csv)[0] + "_report.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report)
    # 안내는 stderr 로 — stdout 은 보고서만 담아야 diff 가 깨끗하다.
    print(f"-> {out}", file=sys.stderr)

    if args.fig:
        png = os.path.splitext(out)[0] + ".png"
        curve(rows, args.by, png)
        print(f"-> {png}", file=sys.stderr)


if __name__ == "__main__":
    main()
