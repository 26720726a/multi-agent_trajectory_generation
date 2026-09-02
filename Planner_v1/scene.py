"""scene.py — 정적 맵 + 장애물 inflate (설계 문서 A §Stage1, B와 공유)

월드: 연속 좌표 [0,W]x[0,H] (m).  A* 는 이산 grid 위에서 돈다.
inflate: 로봇 반지름 r + 여유 margin 만큼 장애물을 부풀려서
         "grid 상 free 셀만 지나면 벽에 안 닿음"을 by-construction 으로 보장.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Rect:
    x0: float; y0: float; x1: float; y1: float

    def contains(self, x, y, pad=0.0):
        return (self.x0 - pad <= x <= self.x1 + pad and
                self.y0 - pad <= y <= self.y1 + pad)


@dataclass
class Scene:
    width: float = 10.0
    height: float = 10.0
    res: float = 0.05                      # grid 해상도 (m/cell)
    obstacles: list = field(default_factory=list)

    # ---- grid ----
    def build_grid(self, inflate: float) -> np.ndarray:
        """True = free.  inflate 만큼 부풀린 occupancy grid."""
        nx = int(round(self.width / self.res))
        ny = int(round(self.height / self.res))
        free = np.ones((ny, nx), dtype=bool)
        xs = (np.arange(nx) + 0.5) * self.res
        ys = (np.arange(ny) + 0.5) * self.res
        X, Y = np.meshgrid(xs, ys)
        for r in self.obstacles:
            mask = ((X >= r.x0 - inflate) & (X <= r.x1 + inflate) &
                    (Y >= r.y0 - inflate) & (Y <= r.y1 + inflate))
            free[mask] = False
        # 경계도 inflate
        b = int(np.ceil(inflate / self.res))
        if b > 0:
            free[:b, :] = False; free[-b:, :] = False
            free[:, :b] = False; free[:, -b:] = False
        return free

    def to_cell(self, x, y):
        return (int(y / self.res), int(x / self.res))     # (row, col)

    def to_xy(self, rc):
        r, c = rc
        return ((c + 0.5) * self.res, (r + 0.5) * self.res)

    def segment_free(self, p, q, inflate, n=None) -> bool:
        """선분 p-q 가 inflate 된 장애물과 안 겹치는지 (shortcut 검사용)."""
        p = np.asarray(p, float); q = np.asarray(q, float)
        if n is None:
            n = max(2, int(np.linalg.norm(q - p) / (self.res * 0.5)))
        for t in np.linspace(0.0, 1.0, n):
            x, y = p + t * (q - p)
            if not (inflate <= x <= self.width - inflate and
                    inflate <= y <= self.height - inflate):
                return False
            for r in self.obstacles:
                if r.contains(x, y, pad=inflate):
                    return False
        return True


def default_scene() -> Scene:
    """참고 그림과 유사한 배치 — 중앙 건물 클러스터 + 우측 별채."""
    s = Scene()
    s.obstacles = [
        Rect(3.2, 5.0, 5.4, 7.2),   # 중앙 상단 큰 건물
        Rect(3.0, 3.8, 5.6, 5.0),   # 중앙 하단 넓은 건물
        Rect(2.6, 2.8, 3.4, 3.9),   # 좌하단 소형
        Rect(6.0, 3.0, 7.6, 4.0),   # 우측 별채
    ]
    return s
