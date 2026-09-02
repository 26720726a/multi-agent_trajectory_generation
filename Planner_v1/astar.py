"""astar.py — inflate 된 grid 위 8-connected A* (설계 문서 A §Stage1, B와 공유)"""
import heapq
import numpy as np

SQRT2 = 1.4142135623730951
_NBRS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def _octile(a, b):
    dy = abs(a[0] - b[0]); dx = abs(a[1] - b[1])
    return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)


def astar(free: np.ndarray, start_rc, goal_rc):
    """free[r,c]=True 인 셀만 통과.  대각 이동 시 코너 커팅 금지."""
    if not free[start_rc] or not free[goal_rc]:
        raise ValueError("start/goal 이 inflate 된 장애물 안에 있음")
    ny, nx = free.shape
    g = {start_rc: 0.0}
    parent = {}
    pq = [(_octile(start_rc, goal_rc), 0.0, start_rc)]
    closed = set()
    while pq:
        f, gc, cur = heapq.heappop(pq)
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
        for dr, dc, w in _NBRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < ny and 0 <= nc < nx) or not free[nr, nc]:
                continue
            if dr and dc and not (free[r, nc] and free[nr, c]):
                continue                      # corner cutting 방지
            ng = gc + w
            nxt = (nr, nc)
            if ng < g.get(nxt, np.inf):
                g[nxt] = ng
                parent[nxt] = cur
                heapq.heappush(pq, (ng + _octile(nxt, goal_rc), ng, nxt))
    raise RuntimeError("A*: 경로 없음")
