"""draw_common.py — viz.py 와 editor.py 가 같이 쓰는 순수 그리기 헬퍼.

여기서는 절대 matplotlib.use(...) 를 호출하지 않는다(백엔드를 안 건드림).
viz.py 는 headless 저장(gif/png)을 위해 자기 모듈에서 Agg 를 강제하고,
editor.py 는 마우스 이벤트를 받아야 하니 인터랙티브 백엔드가 필요하다 —
이 파일을 통해서만 그리기 코드를 공유하면 어느 쪽이 먼저 import 되어도
서로의 백엔드 선택을 건드리지 않는다.
"""
import matplotlib
import matplotlib.font_manager    # matplotlib.font_manager 속성을 확실히 붙여둠
from matplotlib.patches import Rectangle, Circle as CirclePatch

from scene import Rect, Circle

# 한글 라벨이 기본 폰트(DejaVu Sans)엔 없어서 네모로 깨진다. Windows 기본
# 한글 폰트로 바꿔 시도하고, 없으면 조용히 기본 폰트로 넘어간다(백엔드
# 선택과는 무관한 설정이라 여기서 한 번만 해두면 viz.py/editor.py 둘 다 쓴다).
for _fname in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if _fname in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        matplotlib.rcParams["font.family"] = _fname
        break
matplotlib.rcParams["axes.unicode_minus"] = False

COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def draw_obstacle(ax, obs, color="#444444", zorder=1, dashed_pad=None,
                   dashed_color="#999999"):
    """obs 하나(Rect|Circle)를 그린다. dashed_pad 가 주어지면 그만큼 부풀린
    점선 윤곽도 같이 그린다(inflate 시각화). 그려진 patch 들을 돌려준다
    (editor.py 가 드래그 중 geometry 만 빠르게 갱신할 때 씀)."""
    if isinstance(obs, Rect):
        body = Rectangle((obs.x0, obs.y0), obs.x1 - obs.x0, obs.y1 - obs.y0,
                          color=color, zorder=zorder)
        ax.add_patch(body)
        dashed = None
        if dashed_pad is not None:
            dashed = Rectangle(
                (obs.x0 - dashed_pad, obs.y0 - dashed_pad),
                (obs.x1 - obs.x0) + 2 * dashed_pad, (obs.y1 - obs.y0) + 2 * dashed_pad,
                fill=False, linestyle="--", edgecolor=dashed_color, zorder=zorder)
            ax.add_patch(dashed)
    else:  # Circle
        body = CirclePatch((obs.cx, obs.cy), obs.r, color=color, zorder=zorder)
        ax.add_patch(body)
        dashed = None
        if dashed_pad is not None:
            dashed = CirclePatch((obs.cx, obs.cy), obs.r + dashed_pad,
                                  fill=False, linestyle="--",
                                  edgecolor=dashed_color, zorder=zorder)
            ax.add_patch(dashed)
    return body, dashed
