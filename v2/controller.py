"""컨트롤러(안전+추종) 상수 — 단일 출처.

원본 `mahoi/wm/controller.py:37-59` 의 "Tunables" 블록을 그대로 옮긴 것이다.
그 블록의 주석대로 **학습된 파라미터는 하나도 없다** — 전부 명시적 상수다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import math


def _deg(*d: float) -> Tuple[float, ...]:
    return tuple(math.radians(x) for x in d)


@dataclass(frozen=True)
class Controller:
    # -- 경로 추종 ------------------------------------------------------------ #
    #: pure-pursuit 전방주시 거리 (m).  원본 controller.py:40.
    lookahead: float = 0.85
    #: W/G 에 스냅하는 거리 (m).  원본 controller.py:48.
    snap_tol: float = 0.02
    #: 의존성에 막힌 후속 에이전트가 자기 W 앞에서 대기하는 거리 (m).
    #: 원본 controller.py:49.
    dep_standoff: float = 0.45

    # -- 속도 장애물 (VO) ------------------------------------------------------ #
    #: 에이전트 간 VO 시간 지평선 (s).  원본 controller.py:41.
    tau: float = 2.6
    #: 정적 기하에 대한 짧은 지평선 (s).  원본 controller.py:42.
    tau_obst: float = 1.1
    #: 하한 위에 더 두고 싶은 여유 (m).  원본 controller.py:47 `SOFT_MARGIN`.
    #: **planner 의 `cost_soft_margin`(0.25) 과 다른 값이다** — 자세한 것은
    #: `config/planner.py` 의 해당 주석 참조.
    vo_soft_margin: float = 0.22
    #: 부풀린 장애물에서 더 두는 여유 (m).  원본 controller.py:46.
    obst_pad: float = 0.02

    # -- 후보 속도 비용 가중치 ------------------------------------------------- #
    #: 1/TTC 가중치 (이웃 / 장애물).  원본 controller.py:43,44.
    w_ttc: float = 1.35
    w_ttc_obst: float = 0.55
    #: 명령 급변 억제.  원본 controller.py:45.
    #: **a_max 가 없는 현행 구조에서 가속도를 억제하는 유일한 항이다**
    #: (S0 §C-1) — 상한 강제가 아니라 비용이다.
    w_turn: float = 0.22
    #: mode 의 좌/우 통과 편향 세기.  원본 controller.py:46 `W_SIDE`.
    w_side: float = 0.16

    # -- 제동 (S3-A 3차) ------------------------------------------------------ #
    #: 제동 항이 진행 방향으로 내다보는 거리 (m).
    #:
    #: 순항 v_max=1.20 에서 a_max=3.0 으로 완전히 서려면 4 스텝 동안 0.180 m 를
    #: 더 간다 (S3-A" 보고서 5-3 에서 실측).  그 1.7 배로 잡았다 — 여유를 두되
    #: 무한정 보지는 않는다.  무한정 보면 통로 안에서 항상 기어가게 된다.
    #:
    #: 한 스텝 하드 필터가 보는 거리는 v_max*dt = 0.120 m 로 필요량의 2/3 다.
    #: 그 부족분이 급정거형 전멸의 직접 원인이었다 — 기하로 막힌 사건의 100% 가
    #: 0.180 m 보다 좁은 여유(중앙값 0.018 m)에서 일어났다.
    #: **S3-A 3차에서 측정으로 무효가 확인되어 껐다** (`None`).
    #: 호출의 98.7% 가 cap 을 그대로 돌려주었고 — 부채꼴 안 한 방향이라도
    #: 열려 있으면 감속하지 않으므로 — 급정거형 전멸은 3.2% 밖에 줄지 않은 채
    #: 재계획당 벽시계만 1.32배가 되었다 (reports/S3A4_brake_distance.md).
    #:
    #: 코드(`planning/control.py::brake_free_distance`)는 남겨 두었다.  값을
    #: 넣으면 다시 켜지고, 그때 `phys_fp` 가 갈려 CSV 가 섞이지 않는다.
    #: 껐을 때 남는 제동 대상은 목표(waypoint/goal)와 의존성 대기 경계 둘이며,
    #: 후자는 효과가 확인되었다 (순항 충돌 비율 72.2% -> 53.1%).
    brake_lookahead: Optional[float] = None

    #: 후보 65개가 전부 기하 필터에 걸렸을 때(=전멸) 무엇을 할 것인가.
    #:
    #: `False` = 원본 동작.  `v_cmd = 0` 으로 **한 스텝에** 세운다.  순항
    #: 1.20 m/s 에서는 12 m/s^2 이라 a_max 를 4배 위반한다.
    #: `True`  = `a_max` 로 최대 감속.  방향은 유지하고 크기만 `a_max*dt` 줄인다.
    #:
    #: 계획서 3.3-7 이 명시한 fallback 은 "-3.0 m/s^2, 0.67초 내 정지" 인데
    #: 원본 코드는 순간 정지였다 — **사양과 구현이 어긋나 있었다.**  이 값이
    #: 그 어긋남을 없앤다.  전멸 자체를 줄이지는 않는다 (S3-B3).
    #:
    #: `a_max=inf` 면 한 스텝에 지울 수 있는 속도가 무한이라 두 경로가 같은
    #: 결과를 낸다 — S2 재현이 자동으로 유지된다.
    #:
    #: **S3-B3 에서 켜 보았고 게이트 2(충돌 0건)를 깨서 되돌렸다.**
    #: 전멸은 "후보 65개가 **전부 기하 필터에 걸렸다**" 는 뜻인데, 감속
    #: fallback 은 그 필터를 거치지 않고 `v_prev` 방향으로 계속 움직인다.
    #: `_project_safe` 는 **에이전트 간**만 다루므로 장애물은 아무도 막지 않는다.
    #: 결과: 장애물/벽 침범 88행 (1.81%), 최악 `obstacle_clearance` 0.2436 m
    #: (하한 0.30).  에이전트 간 충돌은 0 이었다.
    #: 성공률도 80.31% -> 57.86% 로 무너졌다 (reports/S3B3_decel_fallback.md).
    #:
    #: 켜려면 감속 이동도 기하 필터를 통과하게 만들어야 한다 — 그것은 B2
    #: (다단계 필터)의 일이다.
    wipeout_decelerate: bool = False

    #: 후보를 고른 뒤 최대 감속으로 설 때까지의 궤적을 하드 필터로 검사한다.
    #:
    #: 유한한 `a_max` 에서는 후보 속력마다 `ceil(|v| / (a_max*dt))` 스텝을
    #: 검사한다. 순항 1.20 m/s, a_max=3.0, dt=0.10이면 최대 4스텝이다.
    #: S3-B2에서 전멸 위반은 절반으로 줄었지만 성공률·비용 게이트를 모두
    #: 통과하지 못했다. S4-A의 원본 물리 사양에서는 False로 복원한다. 다시
    #: 켜려면 별도 지문과 전수 안전/성능 검증이 필요하다.
    multistep_lookahead: bool = False

    # -- 후보 속도 팬 --------------------------------------------------------- #
    #: v_max 에 곱할 속력 배율.  원본 controller.py:57.
    speed_levels: Tuple[float, ...] = (0.0, 0.35, 0.70, 1.0)
    #: 선호 방향에서 벌리는 각도 (rad).  원본 controller.py:58-59.
    angle_levels: Tuple[float, ...] = field(default_factory=lambda: _deg(
        0, 12, -12, 25, -25, 42, -42, 62, -62, 85, -85, 115, -115, 150, -150, 180))

    @property
    def n_candidates(self) -> int:
        """후보 속도 개수.

        원본 `_candidates`(controller.py:270-283)는 `len(speed) x len(angle)`
        격자를 만든 뒤 `v_pref` 자신을 **앞에 하나 더** 붙인다.
        따라서 4 x 16 + 1 = **65** 다.

        (S1 지시서는 "속도 6 x 횡오프셋 6 = 36" 으로 적었으나, S0 §C 에서
        코드를 확인한 결과 실제 구성은 위와 같다.)
        """
        return len(self.speed_levels) * len(self.angle_levels) + 1


CONTROLLER = Controller()
