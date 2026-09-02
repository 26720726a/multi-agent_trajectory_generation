"""bench/generate.py 의 계약 — uid 의 유일성과 spec 의 왕복.

여기서 지키는 것은 성능이 아니라 **정체성**이다.  B2(cost 항 ablation) 와
B6(실패 재현) 은 uid 로 같은 인스턴스의 행들을 짝짓는다.  uid 가 두 인스턴스를
뭉개거나 실행마다 달라지면 그 짝짓기가 조용히 틀린다 — 표는 여전히 그럴듯하게
나오고, 틀렸다는 사실만 안 보인다.  그래서 여기에 못을 박아 둔다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import fields, replace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.generate import (_COERCE, _seed_tuple, Axis, GRID_AXES, Instance,
                            UID_EXCLUDE, axes_from_config, axis_from_config,
                            expected_count, grid)

# 축마다 값을 둘 이상 주고, size 에는 int() 로 뭉개지던 10.0 / 10.5 를 넣는다.
DENSE = Axis(
    n_agents=(2, 3),
    dep_mode=("chain", "fork", "none"),
    size=(10.0, 10.5),
    n_obstacles=((2, 3), (3, 5)),
    couple_prob=(0.0, 0.5, 1.0),
    couple_dist=(0.35, 0.55),
    instance_seeds=(0, 1, 2),
)


def _ser(p) -> str:
    """Problem 을 좌표까지 포함해 직렬화한다 (동일성 비교용)."""
    return json.dumps({
        "name": p.name,
        "world": [p.world.width, p.world.height],
        "obstacles": [[r.x0, r.y0, r.x1, r.y1] for r in p.world.obstacles],
        "agents": [[a.name, list(a.start), list(a.waypoint), list(a.goal),
                    a.dwell, a.radius, a.v_max, a.interact_radius, a.color]
                   for a in p.agents],
        "deps": [list(d) for d in p.deps],
        "gap": p.gap, "safety": p.safety, "dt": p.dt,
    }, sort_keys=True)


def test_uid_unique_across_the_whole_grid():
    """조건 3 — 격자 전체에서 uid 중복 0건 (size 10.0 vs 10.5 포함)."""
    insts = list(grid(DENSE))
    assert len(insts) == 2 * 3 * 2 * 2 * 3 * 2 * 3
    uids = [i.uid for i in insts]
    assert len(set(uids)) == len(uids)


def test_uid_separates_sizes_that_int_would_collapse():
    """옛 uid 는 `s{int(size)}` 라 10.0 과 10.5 가 같은 이름이었다."""
    a = Instance(0, 2, "chain", 10.0, (3, 5), 0.5, 0.55)
    b = Instance(0, 2, "chain", 10.5, (3, 5), 0.5, 0.55)
    assert a.uid != b.uid
    assert _ser(a.build()) != _ser(b.build())


def test_uid_is_a_function_of_every_field():
    """축을 하나 흔들 때마다 uid 가 움직여야 조용한 중복이 안 생긴다."""
    base = Instance(0, 2, "chain", 10.0, (3, 5), 0.5, 0.55)
    variants = [
        Instance(1, 2, "chain", 10.0, (3, 5), 0.5, 0.55),
        Instance(0, 3, "chain", 10.0, (3, 5), 0.5, 0.55),
        Instance(0, 2, "fork", 10.0, (3, 5), 0.5, 0.55),
        Instance(0, 2, "chain", 12.0, (3, 5), 0.5, 0.55),
        Instance(0, 2, "chain", 10.0, (4, 6), 0.5, 0.55),
        Instance(0, 2, "chain", 10.0, (3, 5), 1.0, 0.55),
        Instance(0, 2, "chain", 10.0, (3, 5), 0.5, 0.85),
    ]
    assert len({base.uid, *(v.uid for v in variants)}) == len(variants) + 1


def test_uid_prefix_is_readable():
    assert Instance(0, 3, "chain", 10.0, (3, 5), 0.5, 0.55).uid.startswith("n3_chain_")


def test_uid_is_stable_across_processes():
    """`hash()` 는 PYTHONHASHSEED 마다 달라진다.  hashlib 은 안 달라진다."""
    code = textwrap.dedent("""
        import sys
        sys.path.insert(0, %r)
        from bench.generate import Instance
        print(Instance(0, 3, "chain", 10.5, (3, 5), 0.5, 0.55).uid)
    """) % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outs = set()
    for hashseed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        outs.add(subprocess.check_output([sys.executable, "-c", code],
                                         env=env, text=True).strip())
    assert len(outs) == 1
    assert outs.pop() == Instance(0, 3, "chain", 10.5, (3, 5), 0.5, 0.55).uid


def test_same_spec_gives_the_same_problem_down_to_coordinates():
    """조건 4 — 같은 spec -> 같은 uid, 같은 Problem."""
    for inst in list(grid(DENSE))[::7]:
        twin = Instance(**{**inst.spec,
                           "n_obstacles": tuple(inst.spec["n_obstacles"])})
        assert twin.uid == inst.uid
        assert _ser(twin.build()) == _ser(inst.build())


def test_json_round_trip_rebuilds_the_same_problem():
    """조건 5 — to_json -> from_json -> build() 가 원본과 동일."""
    for inst in list(grid(DENSE))[::5]:
        back = Instance.from_json(inst.to_json())
        assert back == inst
        assert back.uid == inst.uid
        assert _ser(back.build()) == _ser(inst.build())


def test_from_json_rejects_a_spec_it_does_not_understand():
    """축이 하나 늘어난 CSV 를 옛 코드로 읽으면 uid 는 같은데 문제는 다르다."""
    inst = Instance(0, 2, "chain", 10.0, (3, 5), 0.5, 0.55)
    d = json.loads(inst.to_json())
    with pytest.raises(ValueError):
        Instance.from_json({**d, "room_kind": "corridor"})
    with pytest.raises(ValueError):
        Instance.from_json({k: v for k, v in d.items() if k != "couple_dist"})


def test_planner_seed_is_not_part_of_the_instance():
    """planner_seed 를 결정론적 method 에 곱하면 성공률 분모만 부푼다."""
    assert "planner_seed" not in Instance(0, 2, "chain", 10.0, (3, 5), 0.5, 0.55).spec
    assert Axis().planner_seeds == (0,)


def test_axis_accepts_a_list_of_grids_and_a_bare_dict():
    """조건부 스펙 — dict 하나(옛 형식) 도 dict 의 리스트도 받는다."""
    single = {"n_agents": [2], "dep_mode": ["chain"], "instance_seeds": [0, 1, 2]}
    assert len(axes_from_config(single)) == 1
    assert len(axes_from_config([single, {**single, "n_agents": [3]}])) == 2

    joined = list(grid(axes_from_config([single, {**single, "n_agents": [3]}])))
    assert {i.n_agents for i in joined} == {2, 3}
    assert len({i.uid for i in joined}) == len(joined)


def test_overlapping_grids_are_not_counted_twice():
    """겹치는 격자는 같은 인스턴스다.  두 번 세면 성공률이 그만큼 틀어진다."""
    a = {"n_agents": [2, 3], "dep_mode": ["chain"], "instance_seeds": [0, 1, 2]}
    b = {"n_agents": [3, 4], "dep_mode": ["chain"], "instance_seeds": [0, 1, 2]}
    insts = list(grid(axes_from_config([a, b])))
    assert {i.n_agents for i in insts} == {2, 3, 4}
    assert len({i.uid for i in insts}) == len(insts) == 9


def test_old_config_key_names_still_work():
    old = {"n_agents": [2], "coupled_waypoints": [0.0, 1.0],
           "seeds": {"range": [0, 4]}}
    ax = axis_from_config(old)[0]
    assert ax.couple_prob == (0.0, 1.0)
    assert ax.instance_seeds == (0, 1, 2, 3)
    assert ax.couple_dist == (0.55,)          # world.py 의 동결된 기본값


def test_shipped_configs_parse():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_dir = os.path.join(here, "bench", "configs")
    for name in sorted(os.listdir(cfg_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(cfg_dir, name), encoding="utf-8") as fh:
            cfg = json.load(fh)
        insts = list(grid(axes_from_config(cfg["axis"])))
        assert insts, name
        assert len({i.uid for i in insts}) == len(insts), name


# --------------------------------------------------------------------------- #
#  시드 표기 — 리스트는 값 목록이고, 범위는 범위라고 적는다
#
#  `instance_seeds: [0, 1]` 이 조용히 `range(0, 1)` 로 읽히면서 시드 1 의
#  인스턴스가 통째로 사라졌다.  길이가 2 일 때만 의미가 바뀌니 config 만 봐서는
#  알 수 없었고, `grid()` 도 만들어지지 않은 것을 셀 수는 없었다.
# --------------------------------------------------------------------------- #
def test_bare_list_is_a_literal_list_of_seeds():
    assert _seed_tuple([0, 3]) == (0, 3)          # 범위가 아니다
    assert _seed_tuple([0, 1]) == (0, 1)
    assert _seed_tuple([5]) == (5,)
    assert _seed_tuple([0, 1, 2]) == (0, 1, 2)


def test_range_must_be_spelled_out():
    assert _seed_tuple({"range": [0, 3]}) == (0, 1, 2)
    assert _seed_tuple({"range": [3]}) == (0, 1, 2)
    assert _seed_tuple({"range": [0, 6, 2]}) == (0, 2, 4)


def test_malformed_seed_spec_is_refused_not_guessed():
    with pytest.raises(ValueError):
        _seed_tuple({"rnge": [0, 3]})             # 오타를 조용히 무시하지 않는다
    with pytest.raises(ValueError):
        _seed_tuple({"range": []})
    with pytest.raises(ValueError):
        _seed_tuple({"range": [0, 1, 2, 3]})


def test_two_element_seed_list_is_two_seeds_not_a_range():
    """재현 케이스 — (3, seed=1) 이 사라지지 않는다."""
    axes = axes_from_config([{"n_agents": [2], "instance_seeds": [0]},
                             {"n_agents": [3], "instance_seeds": [0, 1]}])
    got = [(i.n_agents, i.instance_seed) for i in grid(axes)]
    assert got == [(2, 0), (3, 0), (3, 1)]


def test_config_path_and_hand_built_axis_agree():
    """이 케이스를 안 보고 있어서 버그를 놓쳤다.

    config 를 거친 Axis 는 `float()` 를 통과해 10.0 을, 손으로 만든 Axis 는 10 을
    담는다.  둘은 같은 문제이므로 같은 uid 여야 한다.
    """
    cfg = {"n_agents": [2, 3], "dep_mode": ["chain", "fork"], "size": [10, 10.5],
           "n_obstacles": [[3, 5]], "couple_prob": [0, 1], "couple_dist": [0.55],
           "instance_seeds": [0, 1, 2], "planner_seeds": [0]}
    hand = Axis(n_agents=(2, 3), dep_mode=("chain", "fork"), size=(10, 10.5),
                n_obstacles=((3, 5),), couple_prob=(0, 1), couple_dist=(0.55,),
                instance_seeds=(0, 1, 2), planner_seeds=(0,))
    assert axes_from_config(cfg) == [hand]
    assert [i.uid for i in grid(axes_from_config(cfg))] == [i.uid for i in grid(hand)]


# --------------------------------------------------------------------------- #
#  uid 는 스펙 전체의 함수다 — 필드를 나열하지 않고 자동으로 검사한다
# --------------------------------------------------------------------------- #
#: 필드 -> 그 필드를 "다른 값으로" 옮기는 함수.  필드가 늘면 여기도 늘어야 하고,
#: 안 늘면 바로 아래 테스트가 잡는다.
_BUMP = {
    "instance_seed": lambda v: v + 1,
    "n_agents": lambda v: v + 1,
    "dep_mode": lambda v: "fork" if v == "chain" else "chain",
    "size": lambda v: v + 0.5,
    "n_obstacles": lambda v: (v[0] + 1, v[1] + 1),
    "couple_prob": lambda v: (v + 0.25) % 1.0,
    "couple_dist": lambda v: v + 0.3,
}

BASE = Instance(0, 2, "chain", 10.0, (3, 5), 0.5, 0.55)


def test_tables_cover_every_field():
    """필드를 추가하고 표 하나를 잊으면 여기서 걸린다 — uid 에서가 아니라."""
    names = {f.name for f in fields(Instance)}
    assert set(_BUMP) == names
    assert set(_COERCE) == names


def test_uid_exclude_is_empty():
    """지금은 모든 필드가 문제를 결정한다.  비우는 건 의식적인 선택이어야 한다."""
    assert UID_EXCLUDE == frozenset()


def test_uid_moves_when_any_single_field_moves():
    """dataclasses.fields 를 순회해 자동으로 검사한다 — 손으로 나열하지 않는다."""
    uids = {BASE.uid: "(기준)"}
    for f in fields(Instance):
        other = replace(BASE, **{f.name: _BUMP[f.name](getattr(BASE, f.name))})
        assert other != BASE, f.name
        assert other.uid not in uids, \
            f"{f.name} 만 바꿨는데 uid 가 {uids[other.uid]} 와 같다"
        uids[other.uid] = f.name
    assert len(uids) == len(fields(Instance)) + 1


def test_uid_separates_instances_that_differ_only_in_seed():
    """이번 버그가 의심했던 지점.  instance_seed 는 문제 자체를 결정한다."""
    a = replace(BASE, instance_seed=0)
    b = replace(BASE, instance_seed=1)
    assert a.uid != b.uid
    assert _ser(a.build()) != _ser(b.build())


def test_uid_survives_a_float_that_does_not_round_trip():
    """repr 대신 자릿수 고정.  0.1+0.2 는 0.30000000000000004 로 적히지 않는다."""
    assert replace(BASE, couple_dist=0.1 + 0.2).uid == \
        replace(BASE, couple_dist=0.3).uid


def test_uid_does_not_confuse_an_int_with_a_float_of_another_value():
    assert replace(BASE, size=10).uid == replace(BASE, size=10.0).uid
    assert replace(BASE, size=10).uid != replace(BASE, size=10.5).uid


# --------------------------------------------------------------------------- #
#  expected_count — grid() 가 몇 개를 내야 하는지 따로 셀 수 있어야 한다
# --------------------------------------------------------------------------- #
def test_expected_count_matches_grid_on_a_disjoint_grid(capsys):
    """통과 조건 4 — 겹치지 않는 축에서는 개수가 같고 경고가 0건이다."""
    ax = Axis(n_agents=(2, 3, 4), size=(10.0, 10.5),
              dep_mode=("chain", "fork", "none"), instance_seeds=(0, 1, 2))
    insts = list(grid(ax))
    assert expected_count(ax) == 3 * 3 * 2 * 1 * 1 * 1 * 3 == 54
    assert len(insts) == expected_count(ax)
    assert len({i.uid for i in insts}) == len(insts)
    assert capsys.readouterr().err == ""


def test_expected_count_uses_the_same_axis_table_as_grid():
    """축을 추가했을 때 둘이 어긋나면 검산이 검산이 아니게 된다."""
    assert {a for a, _ in GRID_AXES} <= {f.name for f in fields(Axis)}
    assert {f for _, f in GRID_AXES} == {f.name for f in fields(Instance)}


def test_expected_count_sums_over_blocks():
    """리스트 형식 axis 두 블록 -> 두 블록의 합."""
    a = {"n_agents": [2], "dep_mode": ["chain"], "instance_seeds": [0, 1, 2]}
    b = {"n_agents": [3, 4], "dep_mode": ["chain"], "instance_seeds": [0, 1]}
    axes = axes_from_config([a, b])
    assert expected_count(axes) == 3 + 4
    insts = list(grid(axes))
    assert len(insts) == expected_count(axes) == 7
    assert {i.n_agents for i in insts} == {2, 3, 4}


def test_expected_count_accepts_a_bare_axis_and_a_list():
    ax = Axis(n_agents=(2, 3), instance_seeds=(0, 1))
    assert expected_count(ax) == expected_count([ax]) == 4
    assert expected_count([ax, ax]) == 8          # 겹침은 세지 않는다


# --------------------------------------------------------------------------- #
#  겹침은 여전히 한 번만 세지만, 이제 조용하지 않다
# --------------------------------------------------------------------------- #
def test_overlap_is_reported_on_stderr(capsys):
    a = {"n_agents": [2, 3], "dep_mode": ["chain"], "instance_seeds": [0, 1, 2]}
    b = {"n_agents": [3, 4], "dep_mode": ["chain"], "instance_seeds": [0, 1, 2]}
    axes = axes_from_config([a, b])
    insts = list(grid(axes))

    err = capsys.readouterr().err
    assert len(insts) == 9
    assert expected_count(axes) == 12
    assert err, "겹침을 말없이 건너뛰면 안 된다"
    assert "3" in err and "12" in err and "9" in err        # 건수와 양쪽 개수
    assert err.count("n3_chain_") >= 1                      # 겹친 건 n=3 쪽이다


def test_overlap_warning_can_be_silenced(capsys):
    a = {"n_agents": [2, 3], "dep_mode": ["chain"], "instance_seeds": [0, 1, 2]}
    insts = list(grid(axes_from_config([a, a]), warn=False))
    assert len(insts) == 6
    assert capsys.readouterr().err == ""


def test_no_warning_when_nothing_is_dropped(capsys):
    list(grid(Axis(n_agents=(2, 3), instance_seeds=(0, 1, 2))))
    assert capsys.readouterr().err == ""


def test_shipped_configs_have_no_hidden_overlap(capsys):
    """coupling.json 은 격자를 일부러 쪼갠다 — 그래도 겹치면 안 된다."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_dir = os.path.join(here, "bench", "configs")
    for name in sorted(os.listdir(cfg_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(cfg_dir, name), encoding="utf-8") as fh:
            cfg = json.load(fh)
        axes = axes_from_config(cfg["axis"])
        insts = list(grid(axes))
        assert len(insts) == expected_count(axes), name
        assert capsys.readouterr().err == "", name


def test_shipped_configs_keep_their_seed_sets():
    """config 문법을 바꿔도 시드 집합은 그대로여야 한다 (결과가 비교 가능하도록)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_dir = os.path.join(here, "bench", "configs")
    expect = {
        "coupling.json":   [(tuple(range(10)), (0, 1, 2))] * 2,
        "difficulty.json": [(tuple(range(10)), (0, 1, 2))],
        "scale.json":      [(tuple(range(5)), (0, 1, 2))],
        "smoke.json":      [((0, 1, 2), (0, 1))],
    }
    for name, want in expect.items():
        with open(os.path.join(cfg_dir, name), encoding="utf-8") as fh:
            cfg = json.load(fh)
        axes = axes_from_config(cfg["axis"])
        assert [(ax.instance_seeds, ax.planner_seeds) for ax in axes] == want, name
