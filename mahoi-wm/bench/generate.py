"""인스턴스 생성 — 난이도 축을 따라 무작위 문제를 찍어낸다.

`world.random_problem()` 은 이미 `n_agents` 와 `dep_mode` 를 받는다.  여기서 하는
일은 그것을 **난이도 축의 격자**로 감싸서, "어떤 조건에서 되고 어떤 조건에서
안 되는가"를 물을 수 있게 만드는 것뿐이다.

두 개의 seed 는 서로 다른 것을 잰다
-----------------------------------
* `Axis.instance_seeds` -> `random_problem(seed=)`.  **인스턴스 자체**를 바꾼다.
  방·장애물·S/W/G 좌표가 전부 달라지므로, 이 축의 편차는 "문제가 얼마나
  어려운가"의 편차다.
* `Axis.planner_seeds` -> `WMConfig.seed`.  같은 문제를 놓고 **플래너의 모드
  샘플링**만 바꾼다.  계획서가 A4 의 성공지표로 삼은 "chain3 seed 편차 2.71s"
  (`scripts/check_wm.py`) 는 이쪽이다.  둘을 한 축으로 섞으면 어느 쪽 편차를
  보고 있는지 알 수 없게 된다.

`Instance` 는 planner_seed 를 **갖지 않는다**.  결정론적 방법
(lower_bound / sequential / coordination_astar) 에 planner_seed 를 곱하면 같은
행이 그대로 복제되어 성공률의 분모만 부풀 뿐이다.  어느 method 에 곱할지는
`STOCHASTIC_METHODS` 를 보고 run.py 가 판단한다 (아래 참조).

  TODO(B1) 아직 없는 축
    - 통로 폭 / room_kind(open/corridor) : 조건부 하위 축이라 `axis` 를 dict 의
      리스트로 받게 열어만 두었다.  구현은 이번 단계 밖.
    - dependency 밀도 : chain/fork 외에 join, 부분 순서, 무제약
    - 방 크기 대비 agent 수 : 혼잡도 (n_agents / size^2)
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass, fields
from typing import (Callable, Dict, FrozenSet, Iterator, List, Optional,
                    Sequence, Tuple, Union)

from mahoi.world import Problem, random_problem

#: planner_seed 를 곱할 가치가 있는 method.  나머지는 결정론적이라 곱하면
#: 동일한 행이 중복될 뿐이다.  이 표는 **자료**일 뿐이고, 실제로 몇 번 돌릴지는
#: run.py 가 정한다:
#:
#:     seeds = axis.planner_seeds if m in STOCHASTIC_METHODS else (0,)
STOCHASTIC_METHODS = frozenset({"wm_planner"})

#: uid 해시에서 **뺄** 필드.  지금은 비어 있어야 한다 — `Instance` 의 모든 필드가
#: 문제를 결정하기 때문이다.  나열을 "넣을 것" 이 아니라 "뺄 것" 쪽에 두는 이유:
#: 새 필드를 추가하고 이 표를 고치는 걸 잊으면, 사고가 "조용한 중복" 이 아니라
#: "uid 가 바뀜" 이 된다.  전자는 표가 그럴듯하게 틀리고 후자는 눈에 띈다.
UID_EXCLUDE: FrozenSet[str] = frozenset()

#: 경고에 실어 보낼 중복 예시의 개수.
_WARN_EXAMPLES = 3


def _norm(v: object) -> str:
    """해시에 넣을 값의 정규 표기.

    * float 는 `repr` 이 아니라 유효숫자 6자리로 적는다.  repr 은 0.1+0.2 같은
      값에서 파이썬 버전·플랫폼에 따라 흔들릴 수 있고, 그러면 같은 인스턴스가
      어제와 다른 uid 를 갖는다.
    * 정수와 같은 값의 float(10.0) 은 정수로 적는다.  config 를 거친 Axis 는
      `float()` 를 통과해 10.0 을, 손으로 만든 `Axis(size=(10,))` 는 10 을
      담는데, 둘은 `random_problem` 에 같은 문제를 만든다.  표기를 맞춰야 두
      경로가 같은 uid 를 낸다.
    * 타입마다 표기를 다르게 두어 `2` 와 `"2"` 가 섞이지 않게 한다.
    """
    if isinstance(v, bool):                       # bool 은 int 보다 먼저 본다
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isfinite(v) and v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.6g}"
    if isinstance(v, str):
        return json.dumps(v)                      # 구분자가 값에 섞여도 안전하다
    if isinstance(v, (tuple, list)):
        return "[" + ",".join(_norm(x) for x in v) + "]"
    raise TypeError(f"uid 해시에 넣을 수 없는 값: {v!r} ({type(v).__name__})")


#: `Instance` 필드 -> 값 정규화 함수.  `from_json()` 과 `grid()` 가 **같은** 표를
#: 쓴다.  따로 두면 한쪽만 고쳐져서 JSON 을 거친 인스턴스와 격자가 만든 인스턴스가
#: 미묘하게 달라진다.  필드를 추가하면 여기도 채워야 하고, 안 채우면
#: `test_coerce_table_covers_every_field` 가 잡는다.
_COERCE: Dict[str, Callable[[object], object]] = {
    "instance_seed": int,
    "n_agents": int,
    "dep_mode": str,
    "size": float,
    "n_obstacles": lambda v: (int(v[0]), int(v[1])),
    "couple_prob": float,
    "couple_dist": float,
}

#: (Axis 의 축 필드, 그 값이 채우는 Instance 필드).  `grid()` 와
#: `expected_count()` 가 이 하나를 공유하므로 축을 추가해도 둘이 어긋나지 않는다.
#: 순서는 곧 `grid()` 의 산출 순서다 (run.py 의 CSV 행 순서).
GRID_AXES: Tuple[Tuple[str, str], ...] = (
    ("n_agents", "n_agents"),
    ("dep_mode", "dep_mode"),
    ("size", "size"),
    ("n_obstacles", "n_obstacles"),
    ("couple_prob", "couple_prob"),
    ("couple_dist", "couple_dist"),
    ("instance_seeds", "instance_seed"),
)


@dataclass(frozen=True)
class Axis:
    """난이도 격자의 한 축.  값 목록의 데카르트 곱이 인스턴스 집합이 된다."""
    n_agents: Tuple[int, ...] = (2, 3)
    dep_mode: Tuple[str, ...] = ("chain",)
    size: Tuple[float, ...] = (10.0,)
    n_obstacles: Tuple[Tuple[int, int], ...] = ((3, 5),)
    couple_prob: Tuple[float, ...] = (0.5,)      # 커플링이 *일어날 확률*
    couple_dist: Tuple[float, ...] = (0.55,)     # 일어났을 때의 *거리* (m)
    instance_seeds: Tuple[int, ...] = (0, 1, 2)  # 문제를 바꾼다
    planner_seeds: Tuple[int, ...] = (0,)        # 플래너만 바꾼다 (곱하지 않는다)


@dataclass(frozen=True)
class Instance:
    """하나의 문제 + 그것을 정확히 재현하는 데 필요한 전부.

    planner_seed 는 여기에 없다 (모듈 docstring 참조).  즉 이 dataclass 의 필드
    전체가 곧 `build()` 의 입력이고, uid 는 그 전체의 함수다.
    """
    instance_seed: int
    n_agents: int
    dep_mode: str
    size: float
    n_obstacles: Tuple[int, int]
    couple_prob: float
    couple_dist: float

    # -- 정체성 ---------------------------------------------------------- #
    @property
    def spec(self) -> Dict[str, object]:
        """직렬화 가능한 스펙 전체.  JSON 왕복과 해시가 모두 이것만 본다."""
        d = asdict(self)
        d["n_obstacles"] = list(self.n_obstacles)
        return d

    @property
    def uid(self) -> str:
        """사람이 읽는 접두사 + 스펙 전체의 안정 해시 8자리.

        해시는 `dataclasses.fields(self)` 를 돌며 만든다 — 필드 이름을 손으로
        나열하지 않는다.  나열하면 축이 하나 늘 때마다 여기를 같이 고쳐야 하고,
        잊으면 서로 다른 두 인스턴스가 같은 uid 를 갖는다.  그러면 `grid()` 의
        중복 제거가 멀쩡한 인스턴스를 지우고, B2/B6 이 남남인 행을 짝짓는다.
        빼야 할 필드가 생기면 `UID_EXCLUDE` 에 적는다.

        값 표기는 `_norm` 이 맡는다 (float 자릿수 고정).  필드 이름도 블롭에
        넣어서, 두 필드가 값을 맞바꿔도 해시가 달라지게 한다.

        필드가 아니라 property 인 이유: 필드였다면 스펙과 uid 가 따로 놀 수 있다.

        `hash()` 는 실행마다 달라지므로 쓰지 않는다 (PYTHONHASHSEED).  여기서
        만든 uid 는 프로세스·머신·날짜를 넘어 같다 — B2(cost 항 ablation) 와
        B6(실패 재현) 이 이것으로 같은 인스턴스의 행들을 짝짓는다.
        """
        blob = ";".join(
            f"{f.name}={_norm(getattr(self, f.name))}"
            for f in sorted(fields(self), key=lambda f: f.name)
            if f.name not in UID_EXCLUDE)
        h = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]
        return f"n{self.n_agents}_{self.dep_mode}_{h}"

    # -- 왕복 ------------------------------------------------------------- #
    def to_json(self) -> str:
        return json.dumps(self.spec, sort_keys=True)

    @classmethod
    def from_json(cls, blob: Union[str, Dict[str, object]]) -> "Instance":
        """`to_json()` 의 역.  spec 만으로 완전 복원된다 (uid 는 다시 계산된다).

        모르는 키는 거부한다.  조용히 무시하면 축이 하나 늘어난 CSV 를 옛 코드로
        읽었을 때 uid 는 같은데 문제는 다른 상황이 생긴다.
        """
        d = json.loads(blob) if isinstance(blob, str) else dict(blob)
        d.pop("uid", None)                      # 파생값이므로 입력으로 받지 않는다
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"알 수 없는 Instance 필드: {sorted(unknown)}")
        missing = known - set(d)
        if missing:
            raise ValueError(f"빠진 Instance 필드: {sorted(missing)}")
        return cls(**{k: _COERCE[k](d[k]) for k in known})

    # -- 실체화 ----------------------------------------------------------- #
    def build(self) -> Problem:
        return random_problem(
            seed=self.instance_seed, n_agents=self.n_agents, size=self.size,
            n_obstacles=self.n_obstacles, dep_mode=self.dep_mode,
            coupled_waypoints=self.couple_prob, couple_dist=self.couple_dist)

    def row(self) -> Dict[str, object]:
        d = asdict(self)
        d["uid"] = self.uid
        d["n_obstacles"] = f"{self.n_obstacles[0]}-{self.n_obstacles[1]}"
        return d


def grid(axis: Union[Axis, Sequence[Axis]], *, warn: bool = True) -> Iterator[Instance]:
    """난이도 격자 전체를 순회한다.  uid 는 축 값으로만 결정되므로 재현 가능하다.

    `axis` 가 여러 개면 각각을 독립된 격자로 돌리고 이어붙인다 (조건부 축).
    격자끼리 겹치는 부분은 같은 spec == 같은 인스턴스이므로 uid 로 한 번만
    내보낸다 — 안 그러면 겹친 인스턴스만 성공률에서 두 번 세어진다.

    다만 **말없이** 건너뛰지는 않는다.  예전에는 그냥 `continue` 였고, 그래서
    인스턴스가 사라져도 아무 데도 흔적이 남지 않았다.  겹침은 대개 의도지만
    (`coupling.json` 의 두 격자), 의도가 아닐 때는 축이 통째로 접힌 것이므로
    반드시 눈에 띄어야 한다.  건너뛴 게 있으면 다 돌고 나서 stderr 에 몇 건이
    왜 겹쳤는지 적는다.  `warn=False` 로 끌 수 있다.

    `expected_count()` 와 비교하면 겹침 여부를 미리 알 수 있다.
    """
    axes = [axis] if isinstance(axis, Axis) else list(axis)
    seen: Dict[str, Instance] = {}
    dups: List[Instance] = []
    for ax in axes:
        for combo in itertools.product(*(getattr(ax, a) for a, _ in GRID_AXES)):
            inst = Instance(**{fld: _COERCE[fld](v)
                               for (_, fld), v in zip(GRID_AXES, combo)})
            if inst.uid in seen:
                dups.append(inst)
                continue
            seen[inst.uid] = inst
            yield inst
    if dups and warn:
        _warn_duplicates(dups, produced=len(seen), expected=expected_count(axes))


def _warn_duplicates(dups: Sequence[Instance], produced: int, expected: int) -> None:
    """겹쳐서 버린 인스턴스를 stderr 에 요약한다.  건별로 찍으면 너무 시끄럽다."""
    print(f"[bench.generate] 경고: 격자가 겹쳐 {len(dups)}건을 건너뛰었다 "
          f"(축의 곱 {expected} -> 인스턴스 {produced}). "
          f"의도한 겹침이 아니라면 축 하나가 접힌 것이다.", file=sys.stderr)
    for inst in dups[:_WARN_EXAMPLES]:
        spec = json.dumps(inst.spec, sort_keys=True, ensure_ascii=False)
        print(f"  {inst.uid}  {spec}", file=sys.stderr)
    if len(dups) > _WARN_EXAMPLES:
        print(f"  ... 그 밖에 {len(dups) - _WARN_EXAMPLES}건", file=sys.stderr)


def expected_count(axis: Union[Axis, Sequence[Axis]]) -> int:
    """겹침을 무시했을 때 나와야 할 인스턴스 수 — 축 크기의 곱 (격자가 여럿이면 합).

    `grid()` 와 `GRID_AXES` 를 공유하므로 축을 추가해도 둘이 어긋나지 않는다.
    `len(list(grid(ax))) < expected_count(ax)` 이면 격자가 겹쳤다는 뜻이고,
    그 차이가 곧 `grid()` 가 stderr 에 적은 건수다.  run.py 가 이걸로 검산한다.
    """
    axes = [axis] if isinstance(axis, Axis) else list(axis)
    total = 0
    for ax in axes:
        n = 1
        for a, _ in GRID_AXES:
            n *= len(getattr(ax, a))
        total += n
    return total


def buildable(inst: Instance) -> Optional[Problem]:
    """생성 자체가 실패하는 인스턴스는 조용히 버린다.

    좁은 방에 agent 를 많이 넣으면 자유 지점 샘플링이 실패한다.  이건 방법의
    실패가 아니라 인스턴스가 애초에 존재하지 않는 것이므로, 성공률 분모에
    넣으면 안 된다.
    """
    try:
        p = inst.build()
    except Exception:                                    # noqa: BLE001
        return None
    return p if p.is_dag() else None


# --------------------------------------------------------------------------- #
#  config -> Axis
# --------------------------------------------------------------------------- #
def _one_axis(cfg: Dict) -> Axis:
    """dict 하나를 Axis 하나로.  옛 키 이름도 계속 받는다 (하위호환).

        seeds              -> instance_seeds
        coupled_waypoints  -> couple_prob
    """
    obs = cfg.get("n_obstacles", [[3, 5]])
    raw_seeds = cfg.get("instance_seeds", cfg.get("seeds", [0, 1, 2]))
    raw_couple = cfg.get("couple_prob", cfg.get("coupled_waypoints", [0.5]))
    return Axis(
        n_agents=tuple(int(x) for x in cfg.get("n_agents", [2, 3])),
        dep_mode=tuple(str(x) for x in cfg.get("dep_mode", ["chain"])),
        size=tuple(float(s) for s in cfg.get("size", [10.0])),
        n_obstacles=tuple((int(a), int(b)) for a, b in obs),
        couple_prob=tuple(float(c) for c in raw_couple),
        couple_dist=tuple(float(c) for c in cfg.get("couple_dist", [0.55])),
        instance_seeds=_seed_tuple(raw_seeds),
        planner_seeds=_seed_tuple(cfg.get("planner_seeds", [0])),
    )


def _seed_tuple(raw) -> Tuple[int, ...]:
    """시드 목록.  리스트는 **있는 그대로** 값 목록이다.

        [0, 1]              -> (0, 1)          값 두 개
        [5]                 -> (5,)
        {"range": [0, 10]}  -> (0, 1, ..., 9)  range() 의 인자 그대로

    예전에는 두 개짜리 리스트만 몰래 범위로 읽었다.  그래서
    `instance_seeds: [0, 1]` 이 `range(0, 1)` == 시드 하나로 접혔고, 시드 1 의
    인스턴스는 애초에 만들어지지도 않았다 — 길이가 2 일 때만 의미가 바뀌니
    config 를 읽어서는 알아챌 수가 없었다.  범위를 쓰려면 이제 range 라고
    적어야 한다.
    """
    if isinstance(raw, dict):
        unknown = set(raw) - {"range"}
        if unknown:
            raise ValueError(f"알 수 없는 시드 스펙 키: {sorted(unknown)}")
        if "range" not in raw:
            raise ValueError('시드를 dict 로 줄 때는 {"range": [...]} 형식이다')
        args = [int(x) for x in raw["range"]]
        if not 1 <= len(args) <= 3:
            raise ValueError(f'"range" 의 인자는 1~3 개다: {raw["range"]}')
        return tuple(range(*args))
    if isinstance(raw, int):
        return (int(raw),)
    return tuple(int(x) for x in raw)


def axes_from_config(spec: Union[Dict, Sequence[Dict]]) -> List[Axis]:
    """config 의 `"axis"` 를 Axis 목록으로.

    dict 하나(기존 형식)면 격자 하나, dict 의 리스트면 **독립적인 격자 여러 개**
    다.  후자가 필요한 이유: 앞으로 들어올 room_kind(open/corridor) 처럼 특정
    값일 때만 의미 있는 하위 축은 순수한 곱으로 두면 무의미한 조합이 대량
    생긴다 (통로 폭이 있는 open room 같은 것).  격자를 쪼개면 각 격자 안에서는
    곱이 전부 의미를 갖는다.
    """
    if isinstance(spec, dict):
        return [_one_axis(spec)]
    return [_one_axis(s) for s in spec]


#: 옛 이름.  단일 dict 를 주던 호출부가 그대로 동작하도록 남겨 둔다.
#: 반환은 항상 List[Axis] 이고, `grid()` 가 둘 다 받는다.
axis_from_config = axes_from_config
