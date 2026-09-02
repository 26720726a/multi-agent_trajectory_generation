"""config 단일 출처 — 로더와 검증.

    from config import PHYSICS, PLANNER, CONTROLLER

이 패키지는 **어떤 상위 계층도 import 하지 않는다** (`safety/`, `planning/`
모두). 의존 방향이 한쪽이어야 상수가 코드에 끌려다니지 않는다.
`tests/test_layering.py` 가 이것을 못 박는다.

검증은 import 시점에 한 번 돈다.  규칙마다 위반의 성격이 달라 처리도 다르다 —
아래 `validate()` 참조.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import warnings
from typing import List, Tuple

from .controller import CONTROLLER, Controller
from .physics import PHYSICS, Physics
from .planner import PLANNER, Planner

__all__ = ["PHYSICS", "PLANNER", "CONTROLLER", "Physics", "Planner",
           "Controller", "validate", "ConfigWarning", "VIOLATIONS",
           "physics_fingerprint", "FINGERPRINT_FIELDS", "PHYS_FP"]


#: 지문에 들어가는 필드.  `(묶음 이름, 필드 이름들)`.
#:
#: 왜 필요한가 (S0 §G-3 위험 3): 인스턴스 `uid` 는 **문제 정의만** 해싱한다.
#: `v_max` 나 `a_max` 를 바꿔도 uid 는 그대로다.  짝지은 비교에는 그게 필요하지만
#: (같은 인스턴스끼리 비교해야 하므로), 서로 **다른 물리로 돈 CSV 가 조용히
#: 섞이는 것**은 막아야 한다.  그 구분자가 이 지문이다.
#:
#: `CONTROLLER` 는 전 필드를 넣는다 — 컨트롤러 튜너블은 전부 궤적을 바꾼다.
#: `PLANNER` 는 궤적을 바꾸는 세 개만 넣는다.  `max_modes` 나 `time_budget_s`
#: 같은 것은 **bench config 가 인스턴스마다 덮어쓰므로** 여기 넣으면 지문이
#: 실행 설정을 뒤따라가 버려 "물리가 같은가" 라는 질문에 답하지 못한다.
FINGERPRINT_FIELDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("physics", ("v_max", "v_nominal", "a_max", "dt", "robot_radius",
                 "safety_margin", "interact_cluster_radius", "map_size")),
    ("planner", ("horizon_s", "replan_s", "stall_window")),
    ("controller", tuple(f.name for f in dataclasses.fields(Controller))),
)


def _fp_items(physics: Physics, planner: Planner,
              controller: Controller) -> List[str]:
    objs = {"physics": physics, "planner": planner, "controller": controller}
    out: List[str] = []
    for group, names in FINGERPRINT_FIELDS:
        obj = objs[group]
        for name in names:
            # repr 은 float 를 왕복 가능한 형태로 낸다 (inf 는 'inf').
            out.append(f"{group}.{name}={getattr(obj, name)!r}")
    return out


def physics_fingerprint(physics: Physics = PHYSICS, planner: Planner = PLANNER,
                        controller: Controller = CONTROLLER) -> str:
    """이 설정으로 나온 궤적이 어느 물리에서 나왔는지 가리키는 12자 지문.

    **정렬한 뒤** 해싱한다.  그래서 `FINGERPRINT_FIELDS` 의 나열 순서를 바꿔도
    지문은 그대로이고, 값이 하나라도 바뀌면 달라진다.  전자는 리팩터링이
    데이터를 무효로 만들지 않게 하기 위해서, 후자가 이 함수의 존재 이유다.
    """
    body = "\n".join(sorted(_fp_items(physics, planner, controller)))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]


class ConfigWarning(UserWarning):
    """설정 조합이 이상하지만 import 를 막지는 않는 경우."""


def validate(physics: Physics = PHYSICS, planner: Planner = PLANNER,
             controller: Controller = CONTROLLER) -> List[Tuple[str, str]]:
    """설정을 검사하고 **위반 목록**을 돌려준다.

    반환은 `(규칙 이름, 설명)` 의 리스트다.  비어 있으면 문제가 없다.

    규칙마다 강도가 다른 이유:

    * (a) `stall_window < horizon_steps` 는 **현행 값이 이미 위반**이다
      (45 > 40).  이것은 실수가 아니라 원본의 상태이고, 그 결과 rollout 의
      정체 판정이 한 번도 발동하지 않아 P1b 전 8,100행에서 feasible 이
      100% 였다 (P0-1 §G).  import 를 막으면 **S2 의 회귀 재현이 불가능해진다**
      — 원본과 같은 결과를 내는 것이 이관 게이트이기 때문이다.
      그래서 경고만 내고 플래그로 남긴다.  해소는 S3.

    * (b), (c) 는 물리적으로 성립할 수 없는 조합이라 발견 즉시 고쳐야 하지만,
      **현행 값은 둘 다 통과**하므로 지금은 경고 경로가 돌지 않는다.
      (c) 는 `a_max` 가 유한할 때만 의미가 있고 현행은 inf 다 — S3 에서
      3.0 을 넣는 순간 실제로 검사된다.
    """
    out: List[Tuple[str, str]] = []

    # (a) 정체 판정 창이 상상 지평선보다 길면 그 판정은 발동할 수 없다
    steps = planner.horizon_steps(physics.dt)
    if steps is not None and planner.stall_window >= steps:
        out.append((
            "stall_window >= horizon_steps",
            f"stall_window={planner.stall_window} 가 horizon "
            f"{planner.horizon_s}s = {steps} 스텝 이상이라, rollout 의 정체 판정이 "
            f"발동할 수 없다.  Cost.stalled 는 항상 False 가 되고 feasible 은 "
            f"항상 True 가 된다 (P0-1 §G, P1b 8,100행에서 확인).  "
            f"현행 값을 재현하려는 의도라면 정상이며, 해소는 S3 이다."))

    # (b) 하한이 두 발자국보다 좁으면 애초에 겹친다
    if physics.min_sep <= 2.0 * physics.robot_radius:
        out.append((
            "min_sep <= 2*robot_radius",
            f"min_sep={physics.min_sep} 가 2*radius={2*physics.robot_radius} 이하다. "
            f"safety_margin={physics.safety_margin} 이 0 이하라는 뜻이다."))

    # (c) 정지거리가 하한을 넘으면 제동으로 충돌을 피할 수 없다 (계획서 §2.3)
    if math.isfinite(physics.a_max):
        d = physics.braking_distance
        if d > physics.min_sep:
            out.append((
                "braking_distance > min_sep",
                f"v_max={physics.v_max} 에서 a_max={physics.a_max} 로 멈추려면 "
                f"{d:.3f} m 가 필요한데 min_sep 은 {physics.min_sep:.3f} m 다. "
                f"정면으로 마주 오면 제동만으로는 못 피한다."))
    return out


VIOLATIONS: List[Tuple[str, str]] = validate()

#: 지금 설정의 지문.  `bench/run.py` 가 CSV 의 `phys_fp` 열에 이 값을 적는다.
PHYS_FP: str = physics_fingerprint()

for _name, _msg in VIOLATIONS:
    warnings.warn(f"[config] {_name}: {_msg}", ConfigWarning, stacklevel=2)
