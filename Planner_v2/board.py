"""board.py — 공유 blackboard (B0: 이상적, 지연/드롭/노이즈 없음).

각 에이전트는 이웃의 '정확한 현재 위치·속도'만 읽는다. 미래 궤적은 없다.
동기 스텝을 보장하기 위해 더블 버퍼를 쓴다:
  - snapshot(): 이번 스텝 계산에 쓸 '읽기 전용' 상태 (스텝 시작 시점 고정)
  - commit(): 각 에이전트가 자기 다음 상태를 '쓰기 버퍼'에 기록
  - swap(): 모든 에이전트가 계산을 마친 뒤 한 번에 버퍼를 교체
    → 먼저 계산된 에이전트가 나중 에이전트에게 유리/불리해지는 순서 편향이 없다.

dependency 는 예측이 아니라 '이벤트 대기'로 구현한다: 선행 에이전트가 자기
경로 위에서 특정 지점(arc-length)을 실제로 지나는 순간 board 에 이벤트 플래그를
세우고, 후행 에이전트는 그 플래그만 확인한다.
"""
from dataclasses import dataclass, field
import copy
import numpy as np


@dataclass
class AgentPublicState:
    id: int
    pos: np.ndarray
    vel: np.ndarray
    radius: float
    arrived: bool = False


class Board:
    def __init__(self, agents):
        self.state = {
            a.id: AgentPublicState(a.id, np.asarray(a.start, float),
                                    np.zeros(2), a.radius, False)
            for a in agents
        }
        self.passed = set()          # {(agent_id, wp_name), ...} 이벤트 플래그
        self._next_state = {}
        self._next_passed = set()

    def snapshot(self):
        """이번 스텝 동안 모든 에이전트가 공통으로 읽을 고정 스냅샷."""
        return copy.deepcopy(self.state), set(self.passed)

    def commit(self, agent_id, pos, vel, radius, arrived):
        self._next_state[agent_id] = AgentPublicState(
            agent_id, np.asarray(pos, float), np.asarray(vel, float),
            radius, bool(arrived))

    def announce(self, agent_id, wp_name):
        self._next_passed.add((agent_id, wp_name))

    def swap(self):
        self.state.update(self._next_state)
        self.passed |= self._next_passed
        self._next_state = {}
        self._next_passed = set()
