"""baseline.py — strawman: 완전 직렬화(한 명씩, 이전 에이전트 완전 종료 후 다음 시작).

방법 B("국소 양보")와 비교할 대상. 협동이 전혀 없는 대신 안전은 자명하게
보장된다 (같은 시각에 두 에이전트가 동시에 움직이는 일이 없음).
dependency 도 위상순서로 자동 만족된다(선행이 먼저 끝나야 후행 순번이 옴).
"""
import numpy as np

from board import Board


def _topo_order(agents):
    preds = {a.id: {h.predecessor_id for h in a.hold} for a in agents}
    order, done = [], set()
    remaining = list(agents)
    while remaining:
        progressed = False
        for a in list(remaining):
            if preds[a.id] <= done:
                order.append(a)
                done.add(a.id)
                remaining.remove(a)
                progressed = True
        if not progressed:
            order.extend(remaining)          # 순환 등 예외 상황 방어적 fallback
            break
    return order


def _run_alone(scene, agent, dt, max_steps):
    """다른 동적 에이전트 없이 혼자 reference 경로를 따라간다(정적 장애물만 회피).
    완전 직렬화 순서상 이 에이전트 차례가 왔다는 건 자기 predecessor 들이
    이미 '전부' 끝났다는 뜻이므로, dependency hold 는 처음부터 전부 충족된
    것으로 놓는다 — 그렇지 않으면 이 에이전트만 있는 board에는 predecessor가
    아예 없어서 hold 이벤트가 영원히 안 서고(=영원히 대기) makespan 이
    부풀어 오르는 버그가 생긴다.
    """
    board = Board([agent])
    board.passed |= {(h.predecessor_id, h.wp_name) for h in agent.hold}
    path_len = 0.0
    prev = board.state[agent.id].pos.copy()
    positions = [prev.copy()]
    for step_i in range(1, max_steps + 1):
        snap_states, snap_passed = board.snapshot()
        info = agent.step(board, snap_states, snap_passed, dt, scene, [],
                           tau=1.5, delta=0.0, tie_break=False)
        board.swap()
        cur = board.state[agent.id].pos.copy()
        path_len += float(np.linalg.norm(cur - prev))
        prev = cur
        positions.append(cur.copy())
        if board.state[agent.id].arrived:
            return step_i, path_len, positions
    return max_steps, path_len, positions   # 도착 못함(이례적)


def run(scene, agents, dt=0.25, max_steps=4000):
    order = _topo_order(agents)
    per_agent_steps = {}
    per_agent_path_len = {}
    per_agent_positions = {}
    for a in order:
        steps, plen, positions = _run_alone(scene, a, dt, max_steps)
        per_agent_steps[a.id] = steps
        per_agent_path_len[a.id] = plen
        per_agent_positions[a.id] = positions

    makespan = sum(per_agent_steps.values())

    # 시각화용 결합 history: 자기 차례 전엔 start 에, 차례 지나면 goal 에 머무름.
    total_steps = makespan
    history = []
    offset = 0
    offsets = {}
    for a in order:
        offsets[a.id] = offset
        offset += per_agent_steps[a.id]
    for t in range(total_steps + 1):
        frame = {}
        for a in order:
            local_t = t - offsets[a.id]
            pos_list = per_agent_positions[a.id]
            if local_t < 0:
                frame[a.id] = np.asarray(a.start, float)
            elif local_t >= len(pos_list):
                frame[a.id] = np.asarray(pos_list[-1], float)
            else:
                frame[a.id] = np.asarray(pos_list[local_t], float)
        history.append(frame)

    return {
        "order": [a.id for a in order],
        "makespan": makespan,
        "per_agent_steps": per_agent_steps,
        "per_agent_path_len": per_agent_path_len,
        "history": history,
        "dt": dt,
    }
