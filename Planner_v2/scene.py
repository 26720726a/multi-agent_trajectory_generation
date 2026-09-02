"""scene.py — 정적 맵 + 장애물 정의 (B0 가정: 정적 맵은 모든 에이전트가 안다)

연속 좌표 [0,W]x[0,H] (m) 위의 장애물 목록. 사각형(Rect)과 원(Circle) 둘 다
지원한다(§ scenario.json obstacles.type).
A* 용 occupancy grid 는 여기서 inflate 해 만들고,
런타임 RVO 회피용으로는 raw(미inflate) 장애물을 그대로 "속도 0인 이웃"으로
취급한다 (rvo.py 가 로봇 반지름만큼 Minkowski-inflate 해서 처리 — 회피 몫은
전부 rvo.py 쪽에서 매 스텝 계산하고, 여기서는 기하만 제공한다).
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x, y, pad=0.0):
        return (self.x0 - pad <= x <= self.x1 + pad and
                self.y0 - pad <= y <= self.y1 + pad)


@dataclass
class Circle:
    cx: float
    cy: float
    r: float

    def contains(self, x, y, pad=0.0):
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= (self.r + pad) ** 2


@dataclass
class Scene:
    width: float = 10.0
    height: float = 10.0
    res: float = 0.05                      # A* grid 해상도 (m/cell)
    obstacles: list = field(default_factory=list)   # Rect | Circle 혼합 리스트

    def build_grid(self, inflate: float) -> np.ndarray:
        """True = free. inflate 만큼 부풀린 occupancy grid (A* 계획용)."""
        nx = int(round(self.width / self.res))
        ny = int(round(self.height / self.res))
        free = np.ones((ny, nx), dtype=bool)
        xs = (np.arange(nx) + 0.5) * self.res
        ys = (np.arange(ny) + 0.5) * self.res
        X, Y = np.meshgrid(xs, ys)
        for obs in self.obstacles:
            if isinstance(obs, Rect):
                mask = ((X >= obs.x0 - inflate) & (X <= obs.x1 + inflate) &
                        (Y >= obs.y0 - inflate) & (Y <= obs.y1 + inflate))
            else:  # Circle
                mask = (X - obs.cx) ** 2 + (Y - obs.cy) ** 2 <= (obs.r + inflate) ** 2
            free[mask] = False
        b = int(np.ceil(inflate / self.res))
        if b > 0:
            free[:b, :] = False; free[-b:, :] = False
            free[:, :b] = False; free[:, -b:] = False
        return free

    def to_cell(self, x, y):
        return (int(y / self.res), int(x / self.res))

    def to_xy(self, rc):
        r, c = rc
        return ((c + 0.5) * self.res, (r + 0.5) * self.res)

    def segment_free(self, p, q, inflate, n=None) -> bool:
        p = np.asarray(p, float); q = np.asarray(q, float)
        if n is None:
            n = max(2, int(np.linalg.norm(q - p) / (self.res * 0.5)))
        for t in np.linspace(0.0, 1.0, n):
            x, y = p + t * (q - p)
            if not (inflate <= x <= self.width - inflate and
                    inflate <= y <= self.height - inflate):
                return False
            for obs in self.obstacles:
                if obs.contains(x, y, pad=inflate):
                    return False
        return True

    def add_boundary_walls(self, thickness: float = 1.0):
        """세계 경계 밖에 얇은 벽을 둘러 RVO 가 경계도 '정적 장애물'로 보게 한다.
        (A* 는 build_grid 의 테두리 inflate 로 이미 경계를 막지만, 런타임 RVO
        샘플링은 장애물 목록만 보므로 별도로 넣어줘야 한다.)
        """
        w, h, t = self.width, self.height, thickness
        self.obstacles += [
            Rect(-t, -t, w + t, 0.0),        # 아래
            Rect(-t, h, w + t, h + t),       # 위
            Rect(-t, -t, 0.0, h + t),        # 왼쪽
            Rect(w, -t, w + t, h + t),       # 오른쪽
        ]

    def circle_hits_obstacle(self, x, y, radius) -> bool:
        """agent–obstacle 충돌 검사(사후 검증용)."""
        for obs in self.obstacles:
            if isinstance(obs, Rect):
                cx = min(max(x, obs.x0), obs.x1)
                cy = min(max(y, obs.y0), obs.y1)
                if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                    return True
            else:  # Circle
                d = ((x - obs.cx) ** 2 + (y - obs.cy) ** 2) ** 0.5
                if d < radius + obs.r:
                    return True
        return False
