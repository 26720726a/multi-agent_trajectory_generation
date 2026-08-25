"""인스턴스 생성 — 난이도 축을 따라 무작위 문제를 찍어낸다.

`world.random_problem()` 은 이미 `n_agents` 와 `dep_mode` 를 받는다.  여기서 하는
일은 그것을 **난이도 축의 격자**로 감싸서, "어떤 조건에서 되고 어떤 조건에서
안 되는가"를 물을 수 있게 만드는 것뿐이다.

  TODO(B1) 지금 노출된 축은 네 개뿐이다.  아래를 추가해야 곡선이 의미를 갖는다.
    - 통로 폭 / 장애물 밀도   : random_problem 의 n_obstacles 와 크기 분포
    - waypoint 근접도         : coupled_waypoints 와 결합 거리 d (지금 0.55 고정)
    - dependency 밀도         : chain/fork 외에 join, 부분 순서, 무제약
    - 방 크기 대비 agent 수   : 혼잡도 (n_agents / size^2)
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from mahoi.world import Problem, random_problem


@dataclass(frozen=True)
class Axis:
    """난이도 격자의 한 축.  값 목록의 데카르트 곱이 인스턴스 집합이 된다."""
    n_agents: Tuple[int, ...] = (2, 3)
    dep_mode: Tuple[str, ...] = ("chain",)
    size: Tuple[float, ...] = (10.0,)
    n_obstacles: Tuple[Tuple[int, int], ...] = ((3, 5),)
    coupled_waypoints: Tuple[float, ...] = (0.5,)
    seeds: Tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class Instance:
    """하나의 문제 + 그것을 정확히 재현하는 데 필요한 전부."""
    uid: str
    seed: int
    n_agents: int
    dep_mode: str
    size: float
    n_obstacles: Tuple[int, int]
    coupled_waypoints: float

    def build(self) -> Problem:
        return random_problem(
            seed=self.seed, n_agents=self.n_agents, size=self.size,
            n_obstacles=self.n_obstacles, dep_mode=self.dep_mode,
            coupled_waypoints=self.coupled_waypoints)

    def row(self) -> Dict[str, object]:
        d = asdict(self)
        d["n_obstacles"] = f"{self.n_obstacles[0]}-{self.n_obstacles[1]}"
        return d


def grid(axis: Axis) -> Iterator[Instance]:
    """난이도 격자 전체를 순회한다.  uid 는 축 값으로만 결정되므로 재현 가능하다."""
    combos = itertools.product(axis.n_agents, axis.dep_mode, axis.size,
                               axis.n_obstacles, axis.coupled_waypoints,
                               axis.seeds)
    for n, dep, size, obs, cw, seed in combos:
        uid = f"n{n}_{dep}_s{int(size)}_o{obs[0]}{obs[1]}_c{int(cw * 100)}_seed{seed}"
        yield Instance(uid=uid, seed=seed, n_agents=n, dep_mode=dep,
                       size=size, n_obstacles=obs, coupled_waypoints=cw)


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


def axis_from_config(cfg: Dict) -> Axis:
    obs = cfg.get("n_obstacles", [[3, 5]])
    return Axis(
        n_agents=tuple(cfg.get("n_agents", [2, 3])),
        dep_mode=tuple(cfg.get("dep_mode", ["chain"])),
        size=tuple(float(s) for s in cfg.get("size", [10.0])),
        n_obstacles=tuple((int(a), int(b)) for a, b in obs),
        coupled_waypoints=tuple(float(c) for c in cfg.get("coupled_waypoints", [0.5])),
        seeds=tuple(range(*cfg["seeds"]) if isinstance(cfg.get("seeds"), list)
                    and len(cfg["seeds"]) == 2 else cfg.get("seeds", [0, 1, 2])),
    )
