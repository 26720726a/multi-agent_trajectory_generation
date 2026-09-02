"""path.py — 경로 후처리 + arc-length 파라미터화 (설계 문서 A §Stage1, B와 공유)

A 의 핵심 계약: '기하 경로는 여기서 확정되고 이후 절대 안 바뀐다.'
planner 는 이 경로 위의 s(t) — 속도 프로파일 — 만 결정한다.
"""
import numpy as np


def shortcut(points, scene, inflate):
    """line-of-sight 가 되는 구간은 직선으로 당긴다 (string pulling)."""
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
    """모서리 둥글리기 — clearance 가 깨지면 그 단계는 버린다."""
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
    """arc-length s ∈ [0, L] → (x, y).  s_wp(웨이포인트 s값)도 기록."""

    def __init__(self, points):
        self.pts = np.asarray(points, float)
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.s[-1])

    def pos(self, s):
        s = np.clip(s, 0.0, self.length)
        i = int(np.searchsorted(self.s, s, side="right") - 1)
        i = min(i, len(self.pts) - 2)
        ds = self.s[i + 1] - self.s[i]
        t = 0.0 if ds < 1e-12 else (s - self.s[i]) / ds
        return self.pts[i] + t * (self.pts[i + 1] - self.pts[i])


def build_path(scene, free, start_xy, goal_xy, inflate, smooth_iters=3):
    from astar import astar
    cells = astar(free, scene.to_cell(*start_xy), scene.to_cell(*goal_xy))
    pts = [scene.to_xy(rc) for rc in cells]
    pts[0] = start_xy; pts[-1] = goal_xy
    pts = shortcut(pts, scene, inflate)
    pts = chaikin(pts, smooth_iters, scene, inflate)
    return Path(pts)
