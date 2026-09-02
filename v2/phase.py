"""에이전트 과업 단계 상수.

원본에서는 `mahoi/wm/controller.py:126` 에 있었다.  여기로 내린 이유는 안전
계층이 이 값을 **필요로 하기 때문**이다 — `project.py` 는 "움직여 달라고 할 수
없는 에이전트"(dwell 중이거나 이미 도착한)를 알아야 하고, `vo.py` 는 "주차된
에이전트는 비켜 주지 않는다"를 알아야 한다.  planning/ 에 두면 safety -> planning
방향의 import 가 생겨 계층 규칙(tests/test_layering.py (a))을 깬다.

`planning/control.py` 가 같은 이름으로 re-export 하므로 원본의 import 경로
(`from ...controller import PHASE_DONE`) 는 그대로 살아 있다.  값도 순서도
원본과 같다 — `worldmodel.remaining_estimate` 가 `phase >= PHASE_DWELL` 같은
대소 비교를 쓰기 때문에 순서가 계약의 일부다 (tests/test_contract.py).
"""
from __future__ import annotations

PHASE_TO_WP, PHASE_DWELL, PHASE_TO_GOAL, PHASE_DONE = 0, 1, 2, 3
