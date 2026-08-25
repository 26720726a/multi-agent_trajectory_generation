"""Minimal 2D geometry helpers: axis-aligned rectangles, segments, clearances."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle obstacle."""
    x0: float
    y0: float
    x1: float
    y1: float

    def inflate(self, r: float) -> "Rect":
        return Rect(self.x0 - r, self.y0 - r, self.x1 + r, self.y1 + r)

    def contains(self, p: Point, eps: float = 0.0) -> bool:
        return (self.x0 + eps <= p[0] <= self.x1 - eps and
                self.y0 + eps <= p[1] <= self.y1 - eps)

    def corners(self, eps: float = 0.0) -> List[Point]:
        return [(self.x0 - eps, self.y0 - eps), (self.x1 + eps, self.y0 - eps),
                (self.x1 + eps, self.y1 + eps), (self.x0 - eps, self.y1 + eps)]

    @property
    def patch_xy(self):
        return (self.x0, self.y0), self.x1 - self.x0, self.y1 - self.y0


# --------------------------------------------------------------------------- #
def seg_hits_rect(p: Point, q: Point, rect: Rect, eps: float = 1e-7) -> bool:
    """True if segment p-q passes through the *interior* of `rect`.

    Liang-Barsky clipping.  Touching the boundary does not count, which is what
    we want for visibility graphs whose nodes sit on (inflated) corners.
    """
    x0, y0 = rect.x0 + eps, rect.y0 + eps
    x1, y1 = rect.x1 - eps, rect.y1 - eps
    if x0 >= x1 or y0 >= y1:
        return False
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for num, den in ((p[0] - x0, -dx), (x1 - p[0], dx),
                     (p[1] - y0, -dy), (y1 - p[1], dy)):
        if den == 0.0:
            if num < 0.0:
                return False
        else:
            t = num / den
            if den < 0.0:
                if t > t1:
                    return False
                t0 = max(t0, t)
            else:
                if t < t0:
                    return False
                t1 = min(t1, t)
    return t0 < t1


def point_rect_dist(pts: np.ndarray, rect: Rect) -> np.ndarray:
    """Distance from each row of `pts` (N,2) to the rectangle (0 inside)."""
    dx = np.maximum(np.maximum(rect.x0 - pts[:, 0], pts[:, 0] - rect.x1), 0.0)
    dy = np.maximum(np.maximum(rect.y0 - pts[:, 1], pts[:, 1] - rect.y1), 0.0)
    return np.hypot(dx, dy)


def seg_seg_min_dist(p0: np.ndarray, p1: np.ndarray,
                     q0: np.ndarray, q1: np.ndarray) -> float:
    """Minimum distance between two moving points over one timestep.

    Both agents move linearly from p0->p1 and q0->q1 during the same interval,
    so the relative offset r(s) = (p0-q0) + s*((p1-p0)-(q1-q0)) is linear and we
    only need the distance from the origin to that segment.
    """
    a = np.asarray(p0, float) - np.asarray(q0, float)
    b = np.asarray(p1, float) - np.asarray(q1, float)
    d = b - a
    denom = float(d @ d)
    if denom < 1e-12:
        return float(np.linalg.norm(a))
    s = float(np.clip(-(a @ d) / denom, 0.0, 1.0))
    return float(np.linalg.norm(a + s * d))


# --------------------------------------------------------------------------- #
def polyline_length(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def resample_polyline(pts: np.ndarray, ds: float) -> np.ndarray:
    """Resample a polyline to (approximately) uniform arclength spacing `ds`.

    The first and last vertices are preserved exactly.
    """
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-9:
        return pts[:1].copy()
    n = max(1, int(np.ceil(total / ds)))
    targets = np.linspace(0.0, total, n + 1)
    out = np.empty((n + 1, 2), float)
    out[:, 0] = np.interp(targets, cum, pts[:, 0])
    out[:, 1] = np.interp(targets, cum, pts[:, 1])
    return out


def round_corners(pts: np.ndarray, cut_ratio: float = 0.35,
                  max_cut: float = 0.8, n_arc: int = 8) -> np.ndarray:
    """Replace each interior vertex by a quadratic Bezier fillet.

    Produces the smooth, curved look of the reference figure while keeping the
    path homotopic to the shortest polyline.
    """
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts.copy()
    out = [pts[0]]
    for k in range(1, len(pts) - 1):
        a, b, c = pts[k - 1], pts[k], pts[k + 1]
        v0, v1 = a - b, c - b
        l0, l1 = np.linalg.norm(v0), np.linalg.norm(v1)
        if l0 < 1e-9 or l1 < 1e-9:
            continue
        cut = min(cut_ratio * l0, cut_ratio * l1, max_cut)
        p_in = b + v0 / l0 * cut
        p_out = b + v1 / l1 * cut
        ts = np.linspace(0.0, 1.0, n_arc)[:, None]
        arc = (1 - ts) ** 2 * p_in + 2 * (1 - ts) * ts * b + ts ** 2 * p_out
        out.extend(arc)
    out.append(pts[-1])
    out = np.asarray(out, float)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(out, axis=0), axis=1) > 1e-9])
    return out[keep]
