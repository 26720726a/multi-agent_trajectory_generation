"""sim.py — 재생 + 검증 (충돌 0 확인, makespan 등 지표)"""
import numpy as np
from planner import Params, Agent


def replay(agents, prm: Params):
    """schedule 을 시간별 위치 행렬로.  반환: pos[t, agent, 2], makespan_steps"""
    horizon = max(len(a.schedule) for a in agents)
    P = np.empty((horizon, len(agents), 2))
    for k, a in enumerate(agents):
        for t in range(horizon):
            s = a.schedule[min(t, len(a.schedule) - 1)]
            P[t, k] = a.path.pos(s)
    return P, horizon - 1


def independent_replay(agents, prm: Params):
    """비교용 baseline: 협동 없이 전원 v_max 로 동시에 출발."""
    horizon = max(int(np.ceil(a.path.length / prm.ds)) for a in agents) + 1
    P = np.empty((horizon, len(agents), 2))
    for k, a in enumerate(agents):
        for t in range(horizon):
            P[t, k] = a.path.pos(min(t * prm.ds, a.path.length))
    return P, horizon - 1


def verify(P, prm: Params, scene):
    """agent-agent / agent-장애물 충돌 검사 + 최소 이격 거리."""
    T, n, _ = P.shape
    min_sep = np.inf
    t_min = 0
    aa = 0
    for t in range(T):
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(P[t, i] - P[t, j])
                if d < min_sep:
                    min_sep, t_min = d, t
                if d < 2 * prm.radius:
                    aa += 1
    ao = 0
    for t in range(T):
        for i in range(n):
            x, y = P[t, i]
            for r in scene.obstacles:
                if r.contains(x, y, pad=prm.radius):
                    ao += 1
    return dict(agent_agent=aa, agent_obstacle=ao,
                min_sep=min_sep, t_min_sep=t_min)


def metrics(agents, P, last, prm: Params, check: dict):
    per = {a.name: (len(a.schedule) - 1) * prm.dt for a in agents}
    return dict(
        makespan=last * prm.dt,
        per_agent_time=per,
        path_lengths={a.name: a.path.length for a in agents},
        collisions_agent=check["agent_agent"],
        collisions_obstacle=check["agent_obstacle"],
        min_separation=check["min_sep"],
    )
