"""planner.py — 설계 문서 A 의 코어: 중앙집중 순차 계획 (prioritized planning)

각 agent 를 우선순위대로 처리한다:
  1. 기하 경로는 이미 확정 (path.py) — 여기서는 절대 수정하지 않는다
  2. 상위 agent 들의 '확정된 궤적'을 space-time 장애물로 본다
  3. (s, t) 2차원에서 {전진, 대기} 만으로 goal 도달 최소 시간을 탐색
     → 속도 프로파일 s(t) 만 결정하는 1D 스케줄링

시간/공간 이산화:
  dt 마다 한 스텝, 한 스텝에 전진하면 ds = v_max * dt 만큼 arc-length 진행.
  충돌 판정: 두 원판 중심 거리 < 2r + delta 면 금지.
  swap(교차) 방지: t→t+1 사이 중간 시점도 함께 검사.

한계(문서 §실패 모드): prioritized planning 은 불완전 — 우선순위 순서가
나쁘면 해가 있어도 못 찾을 수 있다.  toy 에서는 순서 휴리스틱(긴 경로 우선,
ID tie-break)으로 충분하다.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Agent:
    aid: int
    name: str
    color: str
    start: tuple
    goal: tuple
    path: object = None          # path.Path
    schedule: np.ndarray = None  # s[t] — 시간 인덱스별 arc-length


@dataclass
class Params:
    v_max: float = 1.0     # m/s
    dt: float = 0.25       # s / step
    radius: float = 0.28   # 로봇 반지름 (m)
    delta: float = 0.06    # 안전 마진 (m)
    t_max_steps: int = 400

    @property
    def min_sep(self):
        return 2 * self.radius + self.delta

    @property
    def ds(self):
        return self.v_max * self.dt


def positions_of(agent: Agent, prm: Params):
    """확정된 agent 의 시간별 위치 배열 (t_max 까지, goal 도달 후엔 goal 에 정지)."""
    T = prm.t_max_steps + 1
    out = np.empty((T, 2))
    sch = agent.schedule
    for t in range(T):
        s = sch[min(t, len(sch) - 1)]
        out[t] = agent.path.pos(s)
    return out


def _conflict(p, q, min_sep):
    return np.hypot(p[0] - q[0], p[1] - q[1]) < min_sep


def schedule_speed(agent: Agent, committed_pos: list, prm: Params) -> np.ndarray:
    """(s,t) 격자에서 도달 최소 시간 스케줄.  committed_pos: 상위 agent 위치 배열들."""
    L = agent.path.length
    N = int(np.ceil(L / prm.ds))               # s 인덱스 0..N (N = goal)
    s_of = lambda i: min(i * prm.ds, L)
    # 각 s 인덱스의 좌표 미리 계산
    xy = np.array([agent.path.pos(s_of(i)) for i in range(N + 1)])

    def free(i, t):
        p = xy[i]
        for other in committed_pos:
            if _conflict(p, other[min(t, len(other) - 1)], prm.min_sep):
                return False
        return True

    def edge_free(i, j, t):
        """t→t+1 에 i→j 로 갈 때 중간 시점(반스텝) 충돌 검사 — 스침/스왑 방지."""
        pm = 0.5 * (xy[i] + xy[j])
        for other in committed_pos:
            o0 = other[min(t, len(other) - 1)]
            o1 = other[min(t + 1, len(other) - 1)]
            om = 0.5 * (o0 + o1)
            if _conflict(pm, om, prm.min_sep):
                return False
        return True

    # BFS: 시간층을 앞으로 전개.  reach[t] = 도달 가능한 s 인덱스 집합
    if not free(0, 0):
        raise RuntimeError(f"{agent.name}: t=0 에 시작점이 이미 충돌")
    parent = {(0, 0): None}
    frontier = {0}
    for t in range(prm.t_max_steps):
        nxt = set()
        for i in frontier:
            for j in (i, i + 1):             # 대기 or 전진
                if j > N or (j, t + 1) in parent:
                    continue
                if free(j, t + 1) and edge_free(i, j, t):
                    parent[(j, t + 1)] = (i, t)
                    nxt.add(j)
        if N in nxt:
            # 역추적
            sch = np.empty(t + 2)
            node = (N, t + 1)
            while node is not None:
                j, tt = node
                sch[tt] = s_of(j)
                node = parent[node]
            return sch
        if not nxt:
            raise RuntimeError(f"{agent.name}: (s,t) 탐색 실패 — 우선순위 재배치 필요")
        frontier = nxt
    raise RuntimeError(f"{agent.name}: t_max 내 goal 미도달")


def plan_all(agents: list, prm: Params) -> list:
    """A 전체: 우선순위 정렬 → 순차 확정.

    기본 휴리스틱 = 긴 경로 우선, ID tie-break.
    prioritized planning 은 불완전(문서 §실패 모드)하므로, 실패 시
    다른 우선순위 순서를 시도하는 fallback 을 둔다.  agent 수가 작은
    toy 에서는 전 순열 시도가 현실적이다.
    """
    from itertools import permutations
    heuristic = tuple(sorted(agents, key=lambda a: (-a.path.length, a.aid)))
    orders = [heuristic] + [p for p in permutations(agents) if p != heuristic]
    last_err = None
    for order in orders:
        try:
            committed_pos = []
            for ag in order:
                ag.schedule = schedule_speed(ag, committed_pos, prm)
                committed_pos.append(positions_of(ag, prm))
            if order != heuristic:
                print("[plan_all] 휴리스틱 순서 실패 → 재배치 성공:",
                      " > ".join(a.name for a in order))
            return list(order)
        except RuntimeError as e:
            last_err = e
            for ag in agents:
                ag.schedule = None
    raise RuntimeError(
        f"모든 우선순위 순서 실패 — 속도 조절만으로는 해소 불가한 인스턴스"
        f" (예: 단일 차선 corridor 를 반대 방향으로 공유). 마지막 오류: {last_err}")
