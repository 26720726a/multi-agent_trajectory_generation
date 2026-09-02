"""scenarios.py — S1~S4 정의 (§6).

각 scenario 함수는 (scene, agents) 를 돌려준다. agents 는 이미 Stage 1
(reference path 확정 + dependency 배선)까지 끝난 상태.
"""
from dataclasses import dataclass
import numpy as np

from scene import Scene, Rect
from path import build_path
from agent import Agent, link_dependency


@dataclass
class Params:
    dt: float = 0.25
    v_max: float = 1.0
    radius: float = 0.28
    tau: float = 1.5
    delta: float = 0.0          # 안전 마진(옵션). 기본 off.
    lookahead: float = 0.5
    slow_radius: float = 0.5
    arrive_tol: float = 0.08
    hold_margin: float = 0.9
    n_dir: int = 16
    n_speed: int = 5
    w_col: float = 1.0
    w_dev: float = 1.0
    progress_window: int = 40
    progress_eps: float = None   # None → sim.run 이 반지름 기반 기본값 사용
    max_steps: int = 4000
    tie_break: bool = True


def _plan(scene, agents, prm: Params, inflate=None):
    inflate = prm.radius + 0.08 if inflate is None else inflate
    free = scene.build_grid(inflate)
    for a in agents:
        a.path = build_path(scene, free, tuple(a.start), tuple(a.goal), inflate)
    return free


def make_agent(aid, name, start, goal, prm: Params, yield_dir=None):
    return Agent(aid, name, start, goal, radius=prm.radius, v_max=prm.v_max,
                 lookahead=prm.lookahead, slow_radius=prm.slow_radius,
                 arrive_tol=prm.arrive_tol, hold_margin=prm.hold_margin,
                 yield_dir=yield_dir)


# ---------------------------------------------------------------------------
# S1 — 2 에이전트, 중앙 교차, dependency 없음 → 회피가 창발하는가
# ---------------------------------------------------------------------------
def scenario_s1(prm: Params = None):
    prm = prm or Params()
    scene = Scene(width=10.0, height=10.0, obstacles=[])
    scene.add_boundary_walls()
    a0 = make_agent(0, "red", (1.0, 5.0), (9.0, 5.0), prm)
    a1 = make_agent(1, "blue", (5.0, 1.0), (5.0, 9.0), prm)
    agents = [a0, a1]
    _plan(scene, agents, prm)
    return scene, agents, prm


# ---------------------------------------------------------------------------
# S2 — 2 에이전트, A ≺ B (event-hold 확인)
# ---------------------------------------------------------------------------
def scenario_s2(prm: Params = None):
    prm = prm or Params()
    scene = Scene(width=10.0, height=10.0, obstacles=[])
    scene.add_boundary_walls()
    a0 = make_agent(0, "red", (1.0, 5.0), (9.0, 5.0), prm)    # predecessor A
    a1 = make_agent(1, "blue", (5.0, 1.0), (5.0, 9.0), prm)   # successor B
    agents = [a0, a1]
    _plan(scene, agents, prm)
    link_dependency(successor=a1, predecessor=a0, point_xy=(5.0, 5.0), wp_name="cross")
    return scene, agents, prm


# ---------------------------------------------------------------------------
# S3 — 3 에이전트, 체인 1≺2, {1,2}≺3 (분산 확장)
# ---------------------------------------------------------------------------
def scenario_s3(prm: Params = None):
    prm = prm or Params()
    scene = Scene(width=10.0, height=10.0, obstacles=[])
    scene.add_boundary_walls()
    a1 = make_agent(1, "a1", (1.0, 1.0), (9.0, 9.0), prm)
    a2 = make_agent(2, "a2", (9.0, 1.0), (1.0, 9.0), prm)
    a3 = make_agent(3, "a3", (5.0, 0.5), (5.0, 9.5), prm)
    agents = [a1, a2, a3]
    _plan(scene, agents, prm)
    P = (5.0, 5.0)
    link_dependency(successor=a2, predecessor=a1, point_xy=P, wp_name="p12")
    link_dependency(successor=a3, predecessor=a1, point_xy=P, wp_name="p13")
    link_dependency(successor=a3, predecessor=a2, point_xy=P, wp_name="p23")
    return scene, agents, prm


# ---------------------------------------------------------------------------
# S4 — 좁은 통로 정면 대치 (deadlock/livelock 테스트).
# 통로 유효폭은 둘이 '나란히 스쳐 지나갈' 정도는 되지만(완전히 외길은 아님),
# 두 에이전트가 완벽히 대칭으로 정면에서 마주보면 reciprocity 만으로는 안
# 풀린다: 서 있는 것(dev_cost 최소)이 옆으로 비키는 것보다 늘 더 싸 보이는
# 대칭 지점이 생겨 둘 다 그 자리에 굳는다(관측된 실패 모드). tie-break(§2.5c,
# ID 낮은 쪽이 먼저 옆으로 붙는 방향 편향)을 켜면 그 대칭이 깨지면서 바로
# 비스듬히 스쳐 지나간다.
# ---------------------------------------------------------------------------
def scenario_s4(prm: Params = None):
    prm = prm or Params()
    # 벽 안쪽 유효 통행폭 = 1.5 - 2*0.28 = 0.94 (2*radius=0.56 보다는 넓어서
    # '비켜 서기'가 물리적으로 가능하지만, 열린 공간(S1)보다는 훨씬 빡빡하다.
    scene = Scene(width=8.0, height=2.0, obstacles=[
        Rect(1.0, 0.0, 7.0, 0.25),
        Rect(1.0, 1.75, 7.0, 2.0),
    ])
    scene.add_boundary_walls()
    a0 = make_agent(0, "red", (0.5, 1.0), (7.5, 1.0), prm)
    a1 = make_agent(1, "blue", (7.5, 1.0), (0.5, 1.0), prm)
    agents = [a0, a1]
    _plan(scene, agents, prm, inflate=prm.radius + 0.03)
    return scene, agents, prm
