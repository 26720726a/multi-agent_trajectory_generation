"""물리 상수 — 단일 출처.

원본 MAHOI-WM 에서는 같은 개념이 여러 곳에 흩어져 있었고 값이 어긋난 곳도
있었다 (S0 §C-2):

    v_max        world.py:35 에 1.20,  build_dataset.py:44 에 1.5
    robot_radius world.py:34 의 AgentSpec 기본값과 world.py:265 의 재선언

여기 있는 값이 유일한 정의다.  값을 바꾸면 git diff 에 남는다 — yaml 로더를
두지 않은 이유이기도 하다 (의존 최소화 + 변경 이력의 가시성).

S1/S2 에서는 원본의 현행 값을 그대로 옮겼다.  계획서 §2 목표값으로의 변경은
**S3 에서 한 축씩** 진행한다 — 한 번에 둘을 바꾸면 어느 쪽이 무엇을 했는지
가릴 수 없기 때문이다.

    S3-A   a_max  inf -> 3.0     (시도; S4-A에서 S2 사양으로 복원)
    S3-B   v_max  1.20 -> 2.0
    S3-C   horizon_s  스윕으로 결정
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Physics:
    # -- 속도 ---------------------------------------------------------------- #
    #: 최대 속도.  원본 world.py:35 `AgentSpec.v_max`.
    #: 계획서 §2 목표는 2.0 이며 S3 에서 올린다.
    v_max: float = 1.20

    #: 순항 속도.  **원본에는 없던 개념이다** (S0 §C: "코드에 없음").
    #: 원본 컨트롤러는 `SPEED_LEVELS` 배율을 `v_max` 에 곱해 쓸 뿐 별도의
    #: 순항 속도를 두지 않았다.  현행 동작을 재현하려면 v_max 와 같아야 하므로
    #: 같은 값으로 둔다.  둘을 분리해 쓸지는 S2 에서 controller 를 옮길 때
    #: 확정한다 — 지금 다른 값을 넣으면 이관 게이트가 깨진다.
    v_nominal: float = 1.20

    #: 최대 가속도 (m/s^2).  **원본에는 제한이 없었다** (S0 §C-1) — 컨트롤러는
    #: 매 스텝 `_candidates` 팬에서 임의의 속도를 고를 수 있었고, 급변은
    #: `W_TURN` 으로 **비용으로만** 억제되었다.
    #:
    #: S3-A에서 계획서 §2 목표값 3.0 을 시험했으나, 후보 집합이 이전 속도
    #: 주변 반경으로 좁아져 성공률을 회복하지 못했다. S4-A의 기본값은 원본
    #: 사양인 `math.inf`다. 제한을 다시 시험하려면 S3-A/B2의 안전 게이트와
    #: `a_max=inf` 회귀 게이트를 함께 다시 세워야 한다.
    #:
    #: 유한값일 때 제한은
    #: `planning/control.py::_candidates` 가 후보를 이전 속도 주변
    #: `a_max * dt` 안으로 클리핑하는 방식으로 건다.  `math.inf` 로 되돌리면
    #: 그 클리핑이 통째로 건너뛰어져 S2 동작을 비트 단위로 재현한다 —
    #: 구현이 옳은지 확인하는 수단이고, `tests/test_amax.py` 가 지키고 있다.
    #:
    #: 기준선(a_max=inf)에서 실제로 관측된 가속도는 중앙값 13.55, p90 18.41,
    #: 최대 24.00 m/s^2 였다 (results/s3/baseline_S2.summary.txt).  즉 3.0은
    #: 느슨한 상한이 아니라 현행 동작의 4~8배를 자르는 제약이었다.
    a_max: float = math.inf

    # -- 시간 ---------------------------------------------------------------- #
    #: 시뮬레이션 타임스텝.  원본 world.py:76 `Problem.dt`.
    #: 제어 주기도 이 값이다 — 원본에 별도 상수가 없고 `controller.step` 이
    #: 정확히 1 dt 를 전진시킨다 (S0 §C: "제어 주기 = dt, 별도 정의 없음").
    dt: float = 0.10

    # -- 공간 ---------------------------------------------------------------- #
    #: 에이전트 반경.  원본 world.py:34.
    robot_radius: float = 0.30

    #: 두 발자국 사이에 더 두는 여유.  원본 world.py:75 `Problem.safety`.
    safety_margin: float = 0.10

    #: HOI 영역의 시각적/의미적 반경.  원본 world.py:36 `interact_radius`.
    #: 이웃 판정에는 쓰이지 않는다 — 그쪽은 `min_sep + vo_soft_margin` 이다
    #: (S0 §C 참조).
    interact_radius: float = 0.55

    #: S5-2 상호작용 클러스터의 중심거리 반경(m). 연결 성분으로 묶기 위한
    #: 관측용 물리 사양이며, 아직 후보 생성·선택에는 사용하지 않는다.
    interact_cluster_radius: float = 3.0

    #: 기본 방 한 변 (m).  원본 world.py:250 `random_problem(size=10.0)`.
    #: 난이도 격자는 8/10/12 를 쓰며 그것은 bench config 가 정한다.
    map_size: float = 10.0

    # -- 과업 ---------------------------------------------------------------- #
    #: waypoint 에 머무는 시간 (HOI).  원본 world.py:33 `AgentSpec.dwell`.
    #: 원본의 고정 시나리오들은 1.2~1.4 를 개별 지정했다 (world.py:143~226,
    #: 11곳).  그것들은 시나리오 정의이지 전역 상수가 아니므로 여기 오지 않는다.
    dwell_s: float = 1.0

    # -- 인스턴스 생성 -------------------------------------------------------- #
    #: `random_problem` 이 장애물끼리(그리고 장애물과 통로 사이) 벌려 두는 여유.
    #: 원본 world.py:265 의 `radius, margin = 0.30, 0.45` 중 `margin` 이다.
    #:
    #: **이것은 S0 §C-2 가 지목한 "재선언" 이 아니다** — 같은 줄의 `radius` 는
    #: `AgentSpec.radius` 의 재선언이 맞지만, 이 0.45 는 다른 어디에도 없는
    #: 생성기 전용 값이다.  `paths.build_prior(margin=0.12)`(가시성 그래프의
    #: 여유)와도 다른 개념이니 이름을 붙여 구분해 둔다.  값은 원본 그대로다.
    gen_obstacle_margin: float = 0.45

    # -- 파생 ---------------------------------------------------------------- #
    @property
    def min_sep(self) -> float:
        """두 에이전트 중심 사이의 하한.  원본 world.py:83-84 와 같은 식.

        원본은 `Problem.min_sep(i, j)` 로 **쌍마다** 계산했다 (반경이 다를 수
        있으므로).  현행 데이터에서는 전 에이전트가 0.30 이라 상수 0.70 이다.
        쌍별 반경을 도입하면 이 property 는 기본값 역할만 하게 된다.
        """
        return 2.0 * self.robot_radius + self.safety_margin

    @property
    def control_dt(self) -> float:
        """제어 주기.  원본에 별도 상수가 없어 dt 와 동일하다."""
        return self.dt

    @property
    def braking_distance(self) -> float:
        """v_max 에서 정지까지의 거리.  a_max=inf 면 0 이다.

        계획서 §2.3 이 이 값을 `min_sep` 이하로 요구한다 — 검증은
        `config/__init__.py` 가 한다.
        """
        return 0.0 if math.isinf(self.a_max) else self.v_max ** 2 / (2.0 * self.a_max)


PHYSICS = Physics()
