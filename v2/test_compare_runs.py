"""짝지은 비교 도구 — 계획서 §7.4 "총계가 아니라 짝지은 개별 대조" 의 구현.

여기서 확인하는 것은 셋이다.

  1. **같은 CSV 두 개는 전부 tie 여야 한다.**  S3-0 게이트의 문언 그대로다.
     이게 깨지면 이후 모든 단계의 판정을 믿을 수 없다.
  2. 총계가 같아도 **뒤바뀐 인스턴스**가 있으면 전이표가 그것을 드러내야 한다.
     성공률만 보면 놓치는 바로 그 경우다.
  3. 물리가 섞인 CSV 는 거부한다.
"""
from __future__ import annotations

import csv
import importlib.util
import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_tool():
    path = os.path.join(ROOT, "scripts", "compare_runs.py")
    spec = importlib.util.spec_from_file_location("compare_runs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CR = _load_tool()

COLS = ["uid", "method", "planner_seed", "status", "team_time", "phys_fp",
        "git_commit", "agent_clearance", "dep_violations"]


def write(path, rows, fp="fp0", commit="c0"):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            base = {"method": "wm_planner", "planner_seed": 0, "status": "ok",
                    "phys_fp": fp, "git_commit": commit,
                    "agent_clearance": 0.9, "dep_violations": 0}
            base.update(r)
            w.writerow(base)
    return str(path)


# --------------------------------------------------------------------------- #
#  통계 조각
# --------------------------------------------------------------------------- #
def test_sign_test_is_the_exact_two_sided_binomial():
    assert CR.sign_test(0, 0) == 1.0
    assert CR.sign_test(5, 5) == 1.0
    # 10전 10승: 2 * (1/2)^10
    assert CR.sign_test(10, 0) == pytest.approx(2 * 0.5 ** 10)
    # 24승 1패 (P0-3 의 실제 수치) — 계획서가 p<0.0001 이라 적은 그 값
    p = CR.sign_test(24, 1)
    assert p < 1e-4
    assert CR.sign_test(24, 1) == CR.sign_test(1, 24), "양측이면 대칭이어야 한다"


def test_sign_test_survives_the_full_grid_size():
    """격자 전체는 n ~ 4000 이다.  `2 ** n` 을 float 로 옮기면 OverflowError.

    실제로 한 번 터졌다 (baseline vs A 비교).  작은 n 으로만 테스트해서
    놓쳤던 자리다.
    """
    from fractions import Fraction

    def exact(w, l):
        """같은 정의를 유리수로 — float 를 전혀 거치지 않는 기준값."""
        n, k = w + l, min(w, l)
        return Fraction(2 * sum(math.comb(n, i) for i in range(k + 1)),
                        2 ** n)

    for w, l in ((2000, 1900), (600, 700), (120, 80)):
        assert CR.sign_test(w, l) == pytest.approx(float(exact(w, l)), rel=1e-9)
    assert CR.sign_test(3000, 600) < 1e-300
    assert CR.sign_test(2000, 2000) == 1.0
    for n in (500, 4000, 20000):
        assert 0.0 <= CR.sign_test(n // 2, n - n // 2) <= 1.0


def test_quantile_matches_linear_interpolation():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert CR.quantile(xs, 0.0) == 0.0
    assert CR.quantile(xs, 0.5) == 2.0
    assert CR.quantile(xs, 1.0) == 4.0
    assert CR.quantile(xs, 0.25) == 1.0
    assert CR.quantile([7.0], 0.9) == 7.0


def test_histogram_shows_a_bimodal_split_that_a_median_would_hide():
    """중앙값 0 이 "변화 없음" 인지 "상쇄" 인지는 분포만이 답한다."""
    xs = [-5.0] * 10 + [5.0] * 10
    assert CR.quantile(xs, 0.5) == pytest.approx(0.0)
    lines = CR.histogram(xs, bins=4)
    filled = [ln for ln in lines if "#" in ln]
    assert len(filled) == 2, "양 끝 두 봉우리가 보여야 한다"


# --------------------------------------------------------------------------- #
#  게이트: 같은 CSV 두 개는 전부 tie
# --------------------------------------------------------------------------- #
def test_identical_csvs_produce_only_ties(tmp_path, capsys):
    rows = [{"uid": f"u{i}", "team_time": 10.0 + i} for i in range(8)]
    p = write(tmp_path / "a.csv", rows)
    keys = sorted(CR.load(p)[0])
    A, _, _ = CR.load(p)
    B, _, _ = CR.load(p)

    diffs, _ = CR.paired(A, B, keys, "team_time", 1e-9)
    assert len(diffs) == 8
    assert all(d == 0.0 for d in diffs)
    wins = sum(1 for d in diffs if d < -1e-9)
    losses = sum(1 for d in diffs if d > 1e-9)
    assert (wins, losses) == (0, 0), "전부 tie 여야 한다"

    t = CR.transitions(A, B, keys)
    assert t[(True, True)] == 8
    assert t[(True, False)] == t[(False, True)] == t[(False, False)] == 0


def test_main_on_identical_files_reports_all_ties(tmp_path, capsys):
    """도구를 CLI 그대로 돌렸을 때의 출력까지 확인한다 (S3-0 게이트의 문언)."""
    rows = [{"uid": f"u{i}", "team_time": 10.0 + i} for i in range(5)]
    p = write(tmp_path / "a.csv", rows)
    argv = sys.argv
    try:
        sys.argv = ["compare_runs.py", p, p, "--top", "0"]
        assert CR.main() == 0
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "동률 5" in out
    assert "B 승 0" in out and "A 승 0" in out
    assert "**같다**" in out


# --------------------------------------------------------------------------- #
#  전이표가 상쇄를 드러내는가
# --------------------------------------------------------------------------- #
def test_transitions_expose_a_swap_that_totals_would_hide(tmp_path):
    """성공률은 같은데 성공한 인스턴스가 뒤바뀐 경우.

    총계만 보면 "변화 없음" 이다.  전이표는 4건이 뒤집혔다고 말한다.
    """
    A, _, _ = CR.load(write(tmp_path / "a.csv", [
        {"uid": f"u{i}", "status": "ok" if i < 4 else "unfinished:deadlock",
         "team_time": 10.0} for i in range(8)]))
    B, _, _ = CR.load(write(tmp_path / "b.csv", [
        {"uid": f"u{i}", "status": "ok" if i >= 4 else "unfinished:deadlock",
         "team_time": 10.0} for i in range(8)]))
    keys = sorted(set(A) & set(B))
    t = CR.transitions(A, B, keys)
    assert t[(True, True)] == 0
    assert t[(True, False)] == 4 and t[(False, True)] == 4
    # 성공률은 같다 — 그래서 총계로는 아무 일도 없었던 것처럼 보인다
    assert sum(1 for k in keys if CR.ok(A[k])) == sum(1 for k in keys if CR.ok(B[k]))


def test_paired_diffs_use_only_rows_where_both_succeeded(tmp_path):
    """한쪽이 실패한 인스턴스의 makespan 을 섞으면 안 된다.

    실패한 쪽은 team_time 이 비어 있거나 중간에 끊긴 값이라, 넣으면 "실패할수록
    빨라 보이는" 편향이 생긴다.
    """
    A, _, _ = CR.load(write(tmp_path / "a.csv", [
        {"uid": "u0", "team_time": 10.0},
        {"uid": "u1", "team_time": 20.0},
        {"uid": "u2", "status": "unfinished:deadlock", "team_time": 3.0}]))
    B, _, _ = CR.load(write(tmp_path / "b.csv", [
        {"uid": "u0", "team_time": 9.0},
        {"uid": "u1", "status": "timeout", "team_time": 5.0},
        {"uid": "u2", "team_time": 25.0}]))
    diffs, used = CR.paired(A, B, sorted(set(A) & set(B)), "team_time", 1e-9)
    assert diffs == [-1.0]
    assert [k[0] for k in used] == ["u0"]


# --------------------------------------------------------------------------- #
#  물리 지문
# --------------------------------------------------------------------------- #
def test_a_csv_with_mixed_physics_is_refused(tmp_path):
    p = write(tmp_path / "mixed.csv", [{"uid": "u0", "team_time": 1.0}])
    with open(p, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=COLS).writerow(
            {"uid": "u1", "method": "wm_planner", "planner_seed": 0,
             "status": "ok", "team_time": 1.0, "phys_fp": "OTHER",
             "git_commit": "c0", "agent_clearance": 0.9, "dep_violations": 0})
    with pytest.raises(CR.MixedPhysics) as exc:
        CR.load(p)
    assert "OTHER" in str(exc.value) and "fp0" in str(exc.value)


def test_same_physics_but_different_code_is_loudly_flagged(tmp_path, capsys):
    """S3-A' §9-4 의 구멍.  `phys_fp` 는 config 값만 해싱하므로 알고리즘 변경을
    잡지 못한다.  제동 항을 넣은 A' 와 넣지 않은 A 의 지문이 실제로 같았다.

    지문이 같은데 `git_commit` 이 다르면 **눈에 띄게** 알려야 한다 — 안 그러면
    "같은 물리끼리의 비교" 라는 출력이 "같은 조건" 으로 읽힌다.
    """
    a = write(tmp_path / "a.csv", [{"uid": "u0", "team_time": 10.0}],
              fp="same", commit="aaa1111")
    b = write(tmp_path / "b.csv", [{"uid": "u0", "team_time": 9.0}],
              fp="same", commit="bbb2222")
    argv = sys.argv
    try:
        sys.argv = ["compare_runs.py", a, b, "--top", "0"]
        assert CR.main() == 0
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "aaa1111" in out and "bbb2222" in out
    assert "코드가 다르다" in out
    assert "!!" in out


def test_identical_code_and_physics_says_nothing_extra(tmp_path, capsys):
    a = write(tmp_path / "a.csv", [{"uid": "u0", "team_time": 10.0}])
    argv = sys.argv
    try:
        sys.argv = ["compare_runs.py", a, a, "--top", "0"]
        assert CR.main() == 0
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "코드가 다르다" not in out


def test_two_csvs_with_different_physics_are_allowed_but_flagged(tmp_path, capsys):
    """물리를 바꿔 비교하는 것이 이 도구의 용도다 — 막지 않고 표시한다."""
    a = write(tmp_path / "a.csv", [{"uid": "u0", "team_time": 10.0}], fp="fpA")
    b = write(tmp_path / "b.csv", [{"uid": "u0", "team_time": 9.0}], fp="fpB")
    argv = sys.argv
    try:
        sys.argv = ["compare_runs.py", a, b, "--top", "0"]
        assert CR.main() == 0
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "fpA" in out and "fpB" in out
    assert "다르다" in out


# --------------------------------------------------------------------------- #
#  bench 쪽 — 지문이 실행 설정을 반영하는가
# --------------------------------------------------------------------------- #
def test_effective_fingerprint_follows_the_wm_config_override():
    """S3-C 는 `wm_config.horizon_s` 만 바꾸며 6번 돈다.

    config 의 `PLANNER.horizon_s` 를 그대로 지문으로 쓰면 그 6개 CSV 가 전부
    같은 지문을 달게 되어, 이 열이 거짓말을 한다.
    """
    from bench.run import effective_fingerprint
    from config import PHYS_FP, PLANNER
    assert effective_fingerprint({}) == PHYS_FP
    assert effective_fingerprint({"horizon_s": PLANNER.horizon_s}) == PHYS_FP
    assert effective_fingerprint({"max_modes": 4}) == PHYS_FP, \
        "실행 설정(max_modes)은 물리가 아니다"
    fps = {effective_fingerprint({"horizon_s": h})
           for h in (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, None)}
    assert len(fps) == 7, "지평선 값마다 지문이 갈려야 한다"
