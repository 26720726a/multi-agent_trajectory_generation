#!/usr/bin/env python3
"""묶음 6(b) 대조: 같은 smoke.json 으로 돌린 원본/v2 배치 CSV 를 행 단위로 맞춘다.

키는 (uid, method, planner_seed).  판정 기준은 S2 지시서 그대로다 —
성공/실패 플래그는 **완전 일치**, makespan(team_time) 차이는 **< 1e-9**.
`runtime_s`(벽시계)와 `git_commit`(저장소가 다르다)은 재현 대상이 아니라 제외한다.
"""
import csv, sys

IGNORE = {"runtime_s", "git_commit"}
TOL = 1e-9


def load(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return {(r["uid"], r["method"], r["planner_seed"]): r
                for r in csv.DictReader(fh)}


a, b = load(sys.argv[1]), load(sys.argv[2])
print(f"원본 {len(a)} 행 / v2 {len(b)} 행")
if set(a) != set(b):
    print("키 집합이 다르다:", set(a) ^ set(b)); sys.exit(1)

bad, rows = 0, []
for k in sorted(a):
    ra, rb = a[k], b[k]
    diffs = []
    for col in ra:
        if col in IGNORE:
            continue
        va, vb = ra[col], rb[col]
        if va == vb:
            continue
        try:
            if abs(float(va) - float(vb)) < TOL:
                continue
        except (TypeError, ValueError):
            pass
        diffs.append(f"{col}: {va!r} != {vb!r}")
    bad += bool(diffs)
    rows.append((k, ra, rb, diffs))

w = max(len(k[0]) for k in a)
print(f"\n{'uid':<{w}} {'method':<18} {'ps':>2} {'status':<10} "
      f"{'team_time(orig)':>15} {'(v2)':>9} {'|d|':>8} {'valid':>6} 판정")
print("-" * (w + 80))
for k, ra, rb, diffs in rows:
    ta, tb = ra["team_time"], rb["team_time"]
    d = abs(float(ta) - float(tb)) if ta and tb else 0.0
    print(f"{k[0]:<{w}} {k[1]:<18} {k[2]:>2} "
          f"{(ra['status'] + '/' + rb['status']) if ra['status'] != rb['status'] else ra['status']:<10} "
          f"{ta or '-':>15} {tb or '-':>9} {d:>8.1e} "
          f"{(ra['valid'] or '-'):>6} {'OK' if not diffs else 'DIFF ' + '; '.join(diffs)}")

print(f"\n행 {len(a)}개 중 불일치 {bad}개 (비교 제외 열: {sorted(IGNORE)})")
print("RESULT:", "IDENTICAL" if bad == 0 else "DIFFERENT")
sys.exit(0 if bad == 0 else 1)
