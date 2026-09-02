"""sim.py — 동기 시뮬레이션 루프 (방법 B, 탈중앙 협동).

매 스텝: 모든 에이전트가 '같은' board 스냅샷을 읽고 → 각자 계산 → commit →
(모두 끝난 뒤) swap. 순서 편향이 없도록 board.py 의 더블 버퍼를 그대로 쓴다.

deadlock/livelock 감지: 최근 progress_window 스텝 동안 아직 도착하지 않은
모든 에이전트의 누적 변위가 progress_eps 미만이면 '진행 없음'으로 보고 종료.
"""
from dataclasses import dataclass, field
import numpy as np

from board import Board


@dataclass
class SimResult:
    success: bool
    steps: int
    dt: float
    history: list                       # [{agent_id: np.array2, ...}, ...] (step 0 = 초기상태)
    arrival_step: dict                  # agent_id -> step (도착 못하면 없음)
    collisions_agent: int
    collisions_obstacle: int
    dependency_violations: int
    deadlock: bool
    deadlock_agents: list
    yields_per_agent: dict
    tie_break_events: int
    path_length: dict                   # agent_id -> 실제 이동 거리(m)
    unsafe_steps: int


def _circle_circle_overlap(pA, rA, pB, rB, eps=1e-6):
    d = np.linalg.norm(pA - pB)
    return d < (rA + rB) - eps


def run(scene, agents, dt=0.25, max_steps=4000, tau=1.5, delta=0.0,
        tie_break=True, w_col=1.0, w_dev=1.0, n_dir=16, n_speed=5,
        progress_window=40, progress_eps=None, verbose=False):
    board = Board(agents)
    ids = [a.id for a in agents]
    by_id = {a.id: a for a in agents}

    history = [{aid: board.state[aid].pos.copy() for aid in ids}]
    path_length = {aid: 0.0 for aid in ids}
    yields = {aid: 0 for aid in ids}
    tie_events = 0
    collisions_agent = 0
    collisions_obstacle = 0
    dependency_violations = 0
    unsafe_steps = 0
    arrival_step = {}
    deadlock = False
    deadlock_agents = []

    if progress_eps is None:
        progress_eps = 0.5 * min(a.radius for a in agents)

    step_i = 0
    while step_i < max_steps:
        snap_states, snap_passed = board.snapshot()

        # dependency 위반 사전 스냅샷 체크 (실제로 hold 지점을 넘었는지)
        for a in agents:
            if snap_states[a.id].arrived:
                continue
            s_now = a.path.nearest_s(snap_states[a.id].pos)
            for h in a.hold:
                if s_now > h.own_s + 1e-6 and (h.predecessor_id, h.wp_name) not in snap_passed:
                    dependency_violations += 1

        infos = {}
        for a in agents:
            other_ids = [oid for oid in ids if oid != a.id]
            infos[a.id] = a.step(board, snap_states, snap_passed, dt, scene,
                                  other_ids, tau=tau, delta=delta,
                                  tie_break=tie_break, w_col=w_col, w_dev=w_dev,
                                  n_dir=n_dir, n_speed=n_speed)
        board.swap()
        step_i += 1

        step_pos = {}
        for aid in ids:
            st = board.state[aid]
            step_pos[aid] = st.pos.copy()
            moved = np.linalg.norm(st.pos - history[-1][aid])
            path_length[aid] += moved
            if infos[aid]["yielded"]:
                yields[aid] += 1
            if infos[aid]["tie_break_triggered"]:
                tie_events += 1
            if infos[aid]["unsafe"]:
                unsafe_steps += 1
            if aid not in arrival_step and st.arrived:
                arrival_step[aid] = step_i
        history.append(step_pos)

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a1, a2 = board.state[ids[i]], board.state[ids[j]]
                if _circle_circle_overlap(a1.pos, a1.radius, a2.pos, a2.radius):
                    collisions_agent += 1
        for aid in ids:
            st = board.state[aid]
            if scene.circle_hits_obstacle(st.pos[0], st.pos[1], st.radius):
                collisions_obstacle += 1

        if len(arrival_step) == len(agents):
            if verbose:
                print(f"[sim] all arrived at step {step_i}")
            break

        if step_i >= progress_window:
            active = [aid for aid in ids if aid not in arrival_step]
            disps = []
            for aid in active:
                p_now = history[step_i][aid]
                p_then = history[step_i - progress_window][aid]
                disps.append(np.linalg.norm(p_now - p_then))
            if active and max(disps) < progress_eps:
                deadlock = True
                deadlock_agents = active
                if verbose:
                    print(f"[sim] deadlock/livelock at step {step_i}: "
                          f"stuck agents = {[by_id[a].name for a in active]}")
                break

    success = (len(arrival_step) == len(agents))
    return SimResult(
        success=success, steps=step_i, dt=dt, history=history,
        arrival_step=arrival_step, collisions_agent=collisions_agent,
        collisions_obstacle=collisions_obstacle,
        dependency_violations=dependency_violations,
        deadlock=deadlock, deadlock_agents=deadlock_agents,
        yields_per_agent=yields, tie_break_events=tie_events,
        path_length=path_length, unsafe_steps=unsafe_steps,
    )
