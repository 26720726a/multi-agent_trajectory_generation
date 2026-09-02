"""astar.py — inflate 된 grid 위 8-connected A*, 꺾임 최소화 tie-break.

각 에이전트가 "자기 reference 경로"를 만드는 데만 쓴다 (B0: 정적 맵은 안다).
동적 협동(이웃 회피)은 전혀 다루지 않는다 — 그건 rvo.py 의 몫이다.
"""
import heapq
import numpy as np

SQRT2 = 1.4142135623730951
_NBRS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def _octile(a, b):
    dy = abs(a[0] - b[0]); dx = abs(a[1] - b[1])
    return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)


def astar(free: np.ndarray, start_rc, goal_rc):
    """free[r,c]=True 인 셀만 통과. 대각 이동 코너 커팅 금지.
    동률 f-value 는 '직전 이동 방향 유지'를 선호해 꺾임을 줄인다.
    """
    if not free[start_rc] or not free[goal_rc]:
        raise ValueError("start/goal 이 inflate 된 장애물 안에 있음")
    ny, nx = free.shape
    g = {start_rc: 0.0}
    parent = {}
    dir_in = {start_rc: (0, 0)}
    # heap item: (f, turn_penalty, tie_seq, g, cell)
    counter = 0
    pq = [(_octile(start_rc, goal_rc), 0.0, counter, 0.0, start_rc)]
    closed = set()
    while pq:
        f, _tp, _seq, gc, cur = heapq.heappop(pq)
        if cur in closed:
            continue
        if cur == goal_rc:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        closed.add(cur)
        r, c = cur
        pd = dir_in[cur]
        for dr, dc, w in _NBRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < ny and 0 <= nc < nx) or not free[nr, nc]:
                continue
            if dr and dc and not (free[r, nc] and free[nr, c]):
                continue
            ng = gc + w
            nxt = (nr, nc)
            if ng < g.get(nxt, np.inf) - 1e-9:
                g[nxt] = ng
                parent[nxt] = cur
                dir_in[nxt] = (dr, dc)
                turn_penalty = 0.0 if (dr, dc) == pd or pd == (0, 0) else 1e-3
                counter += 1
                heapq.heappush(pq, (ng + _octile(nxt, goal_rc) + turn_penalty,
                                     turn_penalty, counter, ng, nxt))
    raise RuntimeError("A*: 경로 없음")
