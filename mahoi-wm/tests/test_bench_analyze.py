"""bench/analyze.py 의 계약 — 같은 CSV 는 같은 바이트를 낸다.

이 보고서는 커밋해서 `git diff` 로 before/after 를 비교하려고 만든다
(`results/check_wm_v0.txt` 와 같은 성격).  그러니 실행할 때마다 한 글자라도
달라지면 목적 자체가 사라진다 — 개선이 아니라 잡음이 diff 를 채운다.

두 번째 계약은 **섹션이 사라지지 않는다** 는 것이다.  행이 없다고 섹션을
빼면 그 뒤가 전부 밀려서, 한 줄 바뀐 것과 한 섹션 사라진 것이 같은 크기의
diff 로 보인다.
"""
from __future__ import annotations

import csv
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.analyze import (LOSS_COLUMN, REQUIRED, MissingColumns, axis_key,
                           build_report, load, num, section_cost_terms,
                           section_failures, section_seed_spread)
from bench.run import FIELDS

SECTIONS = ("[1]", "[2]", "[3]", "[4]", "[5]", "[6]")


def _row(uid, method, ps, status="ok", **kw):
    r = {k: "" for k in FIELDS}
    r.update(uid=uid, method=method, planner_seed=ps, status=status,
             n_agents="2", dep_mode="chain", git_commit="abc1234",
             team_time="10.0", runtime_s="0.5", ratio_to_bound="1.100")
    r.update(kw)
    return r


def _write(path, rows, fields=FIELDS):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _report(tmp_path, rows, by="n_agents"):
    p = tmp_path / "x.csv"
    _write(p, rows)
    loaded, dropped = load(str(p))
    return build_report(loaded, by, "x.csv", dropped)


# --------------------------------------------------------------------------- #
#  결정성 — 이 파일의 존재 이유
# --------------------------------------------------------------------------- #
def test_the_same_csv_gives_the_same_bytes(tmp_path):
    rows = [_row(f"u{i}", m, ps)
            for i in range(3) for m in ("lower_bound", "wm_planner")
            for ps in (("0",) if m == "lower_bound" else ("0", "1"))]
    a = _report(tmp_path, rows)
    b = _report(tmp_path, rows)
    assert a == b


def test_no_timestamp_or_absolute_path_leaks_in(tmp_path):
    """시각·절대경로·소요시간이 본문에 들어가면 매 실행 diff 가 생긴다."""
    text = _report(tmp_path, [_row("u0", "wm_planner", "0")])
    assert str(tmp_path) not in text
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)          # 날짜
    assert not re.search(r"\d{2}:\d{2}:\d{2}", text)          # 시각
    assert "x.csv" in text                                    # basename 은 남는다


def test_row_order_in_the_csv_does_not_change_the_report(tmp_path):
    """dict 순회나 groupby 순서에 기대면 CSV 순서가 보고서를 흔든다."""
    rows = [_row(f"u{i}", m, ps)
            for i in range(4) for m in ("sequential", "wm_planner")
            for ps in (("0",) if m == "sequential" else ("0", "1"))]
    assert _report(tmp_path, rows) == _report(tmp_path, list(reversed(rows)))


def test_floats_are_written_with_fixed_widths(tmp_path):
    text = _report(tmp_path, [_row("u0", "wm_planner", "0")])
    assert re.search(r"\d\.\d{3}", text)          # 비율 3자리
    assert re.search(r"\d+\.\d %", text)          # 퍼센트 1자리


# --------------------------------------------------------------------------- #
#  섹션은 비어도 남는다
# --------------------------------------------------------------------------- #
def test_every_section_survives_an_empty_result(tmp_path):
    text = _report(tmp_path, [_row("u0", "wm_planner", "0")])
    for tag in SECTIONS:
        assert tag in text, tag


def test_a_clean_run_still_prints_the_failure_section(tmp_path):
    """조건 5 — 실패 0건이어도 [5] 가 사라지면 안 된다."""
    text = _report(tmp_path, [_row("u0", "wm_planner", "0")])
    body = text.split("[5]")[1].split("[6]")[0]
    assert "해당 없음" in body


def test_a_single_planner_seed_says_so_instead_of_vanishing(tmp_path):
    """조건 1 — planner_seed 가 1개여도 [3] 은 제목과 사유를 남긴다."""
    rows = [_row(f"u{i}", "wm_planner", "0") for i in range(3)]
    lines, p90 = section_seed_spread(rows, "n_agents")
    assert lines[0].startswith("[3]")
    assert any("산출 불가" in ln for ln in lines)
    assert p90 is None


def test_the_verdict_is_one_line_and_last(tmp_path):
    text = _report(tmp_path, [_row("u0", "wm_planner", "0")])
    last = [ln for ln in text.splitlines() if ln.strip()][-1]
    assert last.startswith("1 rows:") and "wm_planner" in last


# --------------------------------------------------------------------------- #
#  결측을 falsy 로 거르지 않는다
# --------------------------------------------------------------------------- #
def test_zero_is_a_value_but_blank_is_not():
    assert num({"a": "0"}, "a") == 0.0            # falsy 지만 결측이 아니다
    assert num({"a": ""}, "a") is None
    assert num({}, "a") is None
    assert num({"a": "nan"}, "a") is None
    assert num({"a": "inf"}, "a") is None
    assert num({"a": "없음"}, "a") is None


def test_missing_ratio_rows_are_counted_not_hidden(tmp_path):
    rows = [_row("u0", "wm_planner", "0"),
            _row("u1", "wm_planner", "0", ratio_to_bound="")]
    text = _report(tmp_path, rows)
    assert "ratio 결측으로 분포에서 뺀 ok 행: 1" in text


# --------------------------------------------------------------------------- #
#  정렬은 전부 명시된다
# --------------------------------------------------------------------------- #
def test_numeric_axis_values_sort_numerically():
    assert sorted(["10", "9", "2"], key=axis_key) == ["2", "9", "10"]


def test_non_numeric_and_blank_axis_values_have_a_fixed_place():
    assert sorted(["chain", "2", ""], key=axis_key) == ["2", "chain", ""]


def test_failure_statuses_are_alphabetical(tmp_path):
    rows = [_row("u0", "wm_planner", "0", status="no_solution"),
            _row("u1", "wm_planner", "0", status="invalid"),
            _row("u2", "wm_planner", "0", status="error:X")]
    lines = section_failures(rows, "n_agents")
    seen = [s for s in ("error:X", "invalid", "no_solution")
            if any(s in ln for ln in lines)]
    order = [min(i for i, ln in enumerate(lines) if s in ln) for s in seen]
    assert order == sorted(order)


# --------------------------------------------------------------------------- #
#  자원 한계는 방법의 실패와 섞지 않는다
# --------------------------------------------------------------------------- #
def test_timeout_rows_get_their_own_heading(tmp_path):
    rows = [_row("u0", "wm_planner", "0", status="timeout"),
            _row("u1", "wm_planner", "0", status="oom")]
    lines = section_failures(rows, "n_agents")
    tail = "\n".join(lines).split("자원 한계")[1]
    assert "timeout" in tail and "oom" in tail


# --------------------------------------------------------------------------- #
#  [6] — 열이 없으면 없다고 말한다
# --------------------------------------------------------------------------- #
def test_cost_section_reports_the_rate_without_the_loss_column():
    rows = [_row("u0", "wm_planner", "0", chosen_is_fastest_feasible="True"),
            _row("u1", "wm_planner", "0", chosen_is_fastest_feasible="False")]
    lines, involved, loss_p90 = section_cost_terms(rows, "n_agents")
    assert involved == 50.0 and loss_p90 is None
    assert any("산출 불가" in ln and LOSS_COLUMN in ln for ln in lines)


def test_cost_section_uses_the_loss_column_when_it_appears():
    """열이 생기면 코드를 안 고쳐도 채워져야 한다.

    값은 2단계에서 실제로 관측된 것이다 — n2_chain_e2f46c6a ps=1 에서
    chosen 12.7402 - fastest 12.7340 = 0.0062s.  통과 조건 4 가 이 서식을
    그대로 요구한다.
    """
    rows = [_row("u0", "wm_planner", "1", chosen_is_fastest_feasible="False")]
    rows[0][LOSS_COLUMN] = "0.0062"
    lines, involved, loss_p90 = section_cost_terms(rows, "n_agents")
    assert involved == 100.0
    assert loss_p90 == pytest.approx(0.0062)
    body = "\n".join(lines)
    assert "p50 6.2 ms" in body and "최대 6.2 ms" in body


def test_only_losses_above_the_threshold_are_named():
    """수 ms 짜리를 전부 나열하면 목록이 신호가 아니라 잡음이 된다."""
    rows = [_row("small", "wm_planner", "0", chosen_is_fastest_feasible="False"),
            _row("big", "wm_planner", "0", chosen_is_fastest_feasible="False")]
    rows[0][LOSS_COLUMN] = "0.0062"                  # 6.2ms — 동점 처리 수준
    rows[1][LOSS_COLUMN] = "0.2500"                  # 250ms — 진짜 거래
    body = "\n".join(section_cost_terms(rows, "n_agents")[0])
    assert "초과 사례: big" in body and "small" not in body.split("초과 사례:")[1]


def test_blank_verdicts_are_not_counted_as_false():
    """빈 칸은 '판정하지 않았다' 이지 '빠른 걸 못 골랐다' 가 아니다."""
    rows = [_row("u0", "wm_planner", "0", chosen_is_fastest_feasible=""),
            _row("u1", "wm_planner", "0", chosen_is_fastest_feasible="False")]
    _, involved, _ = section_cost_terms(rows, "n_agents")
    assert involved == 100.0                      # 판정된 1행 중 1행이 False


# --------------------------------------------------------------------------- #
#  입력 거부
# --------------------------------------------------------------------------- #
def test_an_old_csv_is_refused_by_name(tmp_path):
    p = tmp_path / "old.csv"
    _write(p, [], fields=[f for f in FIELDS if f != "planner_seed"])
    with pytest.raises(MissingColumns) as exc:
        load(str(p))
    assert "planner_seed" in str(exc.value)


def test_required_columns_all_exist_in_run_py():
    """run.py 가 안 쓰는 열을 여기서 요구하면 모든 CSV 가 거부된다."""
    assert set(REQUIRED) <= set(FIELDS)


def test_a_half_written_last_row_is_dropped_and_reported(tmp_path):
    """17시간짜리 배치를 도는 중에 읽으면 마지막 줄이 쓰이는 중이다."""
    p = tmp_path / "live.csv"
    _write(p, [_row("u0", "wm_planner", "0"), _row("u1", "wm_planner", "1")])
    whole = p.read_bytes()
    p.write_bytes(whole[:-20])
    rows, dropped = load(str(p))
    assert len(rows) == 1 and dropped == 1
    assert "잘린 행 1개" in build_report(rows, "n_agents", "live.csv", dropped)


def test_mixed_commits_are_all_listed(tmp_path):
    """배치 도중 코드가 커밋되면 숫자가 단일 버전의 성능이 아니다."""
    rows = [_row("u0", "wm_planner", "0"),
            _row("u1", "wm_planner", "0", git_commit="def5678")]
    text = _report(tmp_path, rows)
    assert "2개 섞임" in text and "abc1234" in text and "def5678" in text


def test_a_single_commit_is_named_in_the_header(tmp_path):
    text = _report(tmp_path, [_row("u0", "wm_planner", "0")])
    assert "commit=abc1234" in text and "섞임" not in text
