"""계획 계층 상수 — 단일 출처.

원본에서는 `horizon_s`·`replan_s` 가 네 곳에 재기입되어 있었다 (S0 §C):
`execute.py:51,52`(기본값), `execute.py:320,325,330`(프리셋 3종),
`scripts/run_wm_experiments.py:43`(MAIN).  값을 바꾸려면 네 곳을 다 고쳐야 했다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Planner:
    # -- 상상 지평선 ---------------------------------------------------------- #
    #: rollout 을 몇 초까지 굴려 볼 것인가.  None 이면 완주까지 (단 max_sim_s 상한).
    #: 원본 execute.py:51 기본값.
    #:
    #: **P0-3 에서 10.0 이 4.0 보다 유의하게 낫다고 측정됐다** (전수 100 인스턴스
    #: 짝지은 24승 1패, p<0.0001, 성공률 +7.6%p, cpu 1.88x).  그럼에도 4.0 을
    #: 유지하는 이유는 S2 이관 게이트 때문이다 — 원본과 같은 결과가 나오는지
    #: 확인해야 하는데, 여기서 10.0 으로 바꾸면 그 비교가 불가능해진다.
    #: S4-A의 S2 재현 게이트가 통과한 뒤, P0-3 근거대로 S5-1에서 10.0 s로
    #: 승격했다.  100 step 이므로 stall_window=45보다 길어 Cost.stalled가
    #: 실제 후보 판정에 다시 참여한다.
    horizon_s: Optional[float] = 10.0

    #: 고른 미래 중 실제로 실행에 옮기는 길이.  원본 execute.py:52.
    replan_s: float = 1.0

    # -- 정체 판정 ------------------------------------------------------------ #
    #: rollout **내부**의 정체 판정.  이만큼 연속으로 진전이 없으면 그 rollout 을
    #: stalled 로 표시하고 끊는다.  원본 worldmodel.py:178,184 의 **인자 기본값**
    #: 이라 config 로 조절할 수 없었다 (P0-2 STEP 1 에서 확인).
    #:
    #: **주의: horizon 과 결합되어 있다.**  horizon_s=4.0 이면 40 스텝인데
    #: stall_window 가 45 라 이 판정은 **한 번도 발동하지 않는다** (P0-1 §G).
    #: 그 결과 P1b 전 8,100행에서 feasible 비율이 100% 였다.  현행 동작을
    #: 재현하려면 이 위반 상태를 그대로 두어야 하므로, `config/__init__.py` 의
    #: 검증은 import 실패가 아니라 경고를 낸다.
    stall_window: int = 45

    #: MPC 루프 **바깥**의 정체 판정.  실행된 스텝 기준.  원본 execute.py:60.
    stall_patience: int = 60

    # -- 후보 집합 ------------------------------------------------------------ #
    #: t=0 에 만들 rollout 수.  원본 execute.py:53.
    max_modes: int = 16
    #: 첫 선택 이후 남기는 후보 수.  원본 execute.py:54.
    keep_modes: int = 8
    #: 다리(leg)마다 만들 기하 경로 후보 수.  원본 execute.py:55.
    k_routes: int = 2

    # -- 상한 ----------------------------------------------------------------- #
    #: 시뮬레이션 시간 상한.  원본 execute.py:58.
    max_sim_s: float = 90.0
    #: 벽시계 상한.  원본 execute.py:59.  bench config 는 120.0 을 쓴다.
    time_budget_s: float = 240.0

    # -- 히스테리시스 --------------------------------------------------------- #
    #: 재선택에 필요한 상대/절대 이득.  원본 execute.py:61,62.
    switch_margin_rel: float = 0.01
    switch_margin_abs: float = 0.05

    # -- 비용 가중치 (원본 planner.py:48,52-60) -------------------------------- #
    #: 하드 위반 하나당 붙는 벌점.  원본 planner.py:48 `W_HARD`.
    w_hard: float = 1.0e4
    w_make: float = 1.00
    w_flow: float = 0.06
    w_wait: float = 0.14
    w_dist: float = 0.03
    w_clear: float = 0.60
    w_turn: float = 0.02
    w_dev: float = 0.015

    #: **controller 의 `vo_soft_margin`(0.22) 과 다른 값이다.**  원본에서
    #: `CostWeights.soft_margin=0.25`(planner.py:60) 와
    #: `controller.SOFT_MARGIN=0.22`(controller.py:47) 가 같은 이름으로 다른
    #: 값을 갖고 공존해왔다 (S0 §C-2).  어느 쪽이 어디에 쓰이는지는 S2 에서
    #: 확정하고, 지금은 **둘 다 다른 이름으로 보존**한다 — 통일하면 rollout 이
    #: 달라져 S2 게이트가 깨진다.
    cost_soft_margin: float = 0.25

    # -- 파생 ----------------------------------------------------------------- #
    def horizon_steps(self, dt: float) -> Optional[int]:
        """지평선을 스텝 수로.  원본 execute.py:91 과 같은 식."""
        if self.horizon_s is None:
            return None
        return max(1, int(round(self.horizon_s / dt)))

    def exec_steps(self, dt: float) -> int:
        """한 번에 실행에 옮기는 스텝 수.  원본 execute.py:92."""
        return max(1, int(round(self.replan_s / dt)))


PLANNER = Planner()
