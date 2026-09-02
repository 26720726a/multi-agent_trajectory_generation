"""path.py — 폴리라인 + arc-length 파라미터화.

계약: 기하 경로(reference path)는 여기서 한 번 확정되면 시뮬레이션 내내 바뀌지
않는다. 매 스텝 바뀌는 건 그 경로 위에서의 속도뿐 (agent.py 가 결정).
"""
import numpy as np


def shortcut(points, scene, inflate):
    """line-of-sight 되는 구간은 직선으로 당긴다 (string pulling)."""
    pts = [np.asarray(p, float) for p in points]
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not scene.segment_free(pts[i], pts[j], inflate):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def chaikin(points, iters, scene, inflate):
    """모서리 둥글리기 — clearance 가 깨지는 단계는 버린다."""
    pts = [np.asarray(p, float) for p in points]
    for _ in range(iters):
        if len(pts) < 3:
            break
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            new.append(0.75 * a + 0.25 * b)
            new.append(0.25 * a + 0.75 * b)
        new.append(pts[-1])
        ok = all(scene.segment_free(p, q, inflate)
                 for p, q in zip(new[:-1], new[1:]))
        if not ok:
            break
        pts = new
    return pts


class Path:
    """arc-length s ∈ [0, L] ↔ (x, y). 웨이포인트의 s 값(s_wp)도 조회 가능."""

    def __init__(self, points):
        self.pts = np.asarray(points, float)
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        seg = np.maximum(seg, 1e-12)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.s[-1])

    def pos(self, s):
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.searchsorted(self.s, s, side="right") - 1)
        i = min(max(i, 0), len(self.pts) - 2)
        ds = self.s[i + 1] - self.s[i]
        t = 0.0 if ds < 1e-12 else (s - self.s[i]) / ds
        return self.pts[i] + t * (self.pts[i + 1] - self.pts[i])

    def nearest_s(self, point):
        """point 를 폴리라인에 투영해 가장 가까운 arc-length 를 돌려준다.
        dependency 웨이포인트(s_wp)나 에이전트의 현재 진행도(s) 계산에 쓴다.
        """
        p = np.asarray(point, float)
        best_s, best_d2 = 0.0, np.inf
        for i in range(len(self.pts) - 1):
            a, b = self.pts[i], self.pts[i + 1]
            ab = b - a
            L2 = float(ab @ ab)
            t = 0.0 if L2 < 1e-12 else float((p - a) @ ab / L2)
            t = min(max(t, 0.0), 1.0)
            proj = a + t * ab
            d2 = float((p - proj) @ (p - proj))
            if d2 < best_d2:
                best_d2 = d2
                best_s = self.s[i] + t * (self.s[i + 1] - self.s[i])
        return best_s


def build_path(scene, free, start_xy, goal_xy, inflate, smooth_iters=3):
    from astar import astar
    cells = astar(free, scene.to_cell(*start_xy), scene.to_cell(*goal_xy))
    pts = [scene.to_xy(rc) for rc in cells]
    pts[0] = start_xy; pts[-1] = goal_xy
    pts = shortcut(pts, scene, inflate)
    pts = chaikin(pts, smooth_iters, scene, inflate)
    return Path(pts)
