"""viz.py — 참고 그림 스타일의 시각화

좌: independent prior (협동 없음, 충돌 순간 스냅샷)
우: A — prioritized space-time (min team time)
스타일: 흰 배경, 회색 건물, 옅은 전체 경로, 시작=작은 원, 현재=큰 원+점선 링,
       goal=별, 좌상단 모노스페이스 't=..  d=.. m' 주석.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

FACE = {"red": "#d62839", "blue": "#1f6fd6"}
PALE = {"red": "#f2b8c0", "blue": "#b9d2f0"}


def _draw_scene(ax, scene):
    ax.set_xlim(0, scene.width); ax.set_ylim(0, scene.height)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#333333")
    for r in scene.obstacles:
        ax.add_patch(Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0,
                               facecolor="#c9c9c9", edgecolor="#a8a8a8", lw=0.8))


def _draw_agents(ax, agents, P, t, prm, show_ring=True):
    t = min(t, len(P) - 1)
    for k, a in enumerate(agents):
        ax.plot(a.path.pts[:, 0], a.path.pts[:, 1],
                color=PALE[a.color], lw=1.6, zorder=1, solid_capstyle="round")
        ax.plot(*a.start, marker="o", ms=6, color=FACE[a.color], zorder=3)
        ax.plot(*a.goal, marker="*", ms=15, color=FACE[a.color],
                markeredgecolor="#222222", markeredgewidth=0.5, zorder=3)
        x, y = P[t, k]
        ax.add_patch(Circle((x, y), prm.radius * 0.6, facecolor=FACE[a.color],
                            edgecolor="#222222", lw=0.7, zorder=4))
        if show_ring:
            ax.add_patch(Circle((x, y), prm.radius, fill=False, ls=(0, (2, 2)),
                                edgecolor=FACE[a.color], lw=0.9, zorder=4))
        # 대기(hold) 표시: 직전 스텝과 같은 위치 + 아직 goal 미도달
        gx, gy = a.goal
        arrived = np.hypot(x - gx, y - gy) < 1e-6
        if t > 0 and not arrived and np.allclose(P[t, k], P[t - 1, k]):
            ax.text(x, y + prm.radius + 0.14, "hold", ha="center", va="bottom",
                    fontsize=8, family="monospace", color=FACE[a.color],
                    fontweight="bold", zorder=5)


def _annot(ax, t_sec, d):
    ax.text(0.03, 0.965, f"t={t_sec:4.1f}s  d={d:.2f} m",
            transform=ax.transAxes, family="monospace", fontsize=10,
            ha="left", va="top", color="#111111")


def _pair_dist(P, t):
    t = min(t, len(P) - 1)
    return float(np.linalg.norm(P[t, 0] - P[t, 1]))


def snapshot_figure(scene, agents, P_ind, P_a, t_ind, t_a, prm, out_png,
                    makespan_a, ind_collides):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    ax = axes[0]
    _draw_scene(ax, scene)
    _draw_agents(ax, agents, P_ind, t_ind, prm)
    _annot(ax, t_ind * prm.dt, _pair_dist(P_ind, t_ind))
    sub = "closest approach — COLLISION" if ind_collides else "closest approach"
    ax.set_title(f"independent prior\n({sub})", color="#c1272d", fontsize=11)

    ax = axes[1]
    _draw_scene(ax, scene)
    _draw_agents(ax, agents, P_a, t_a, prm)
    _annot(ax, t_a * prm.dt, _pair_dist(P_a, t_a))
    ax.set_title(f"A — prioritized space-time\n(min team time = {makespan_a:.1f}s)",
                 color="#1a7a2e", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def animate(scene, agents, P_ind, P_a, prm, out_gif, stride=1):
    """양쪽 패널 동시 재생 GIF (PIL 로 프레임 합성)."""
    from PIL import Image
    import io
    T = max(len(P_ind), len(P_a))
    frames = []
    for t in range(0, T, stride):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
        for ax, P, ttl, col in (
                (axes[0], P_ind, "independent prior", "#c1272d"),
                (axes[1], P_a, "A — prioritized space-time", "#1a7a2e")):
            _draw_scene(ax, scene)
            _draw_agents(ax, agents, P, t, prm)
            _annot(ax, t * prm.dt, _pair_dist(P, t))
            ax.set_title(ttl, color=col, fontsize=10)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=int(prm.dt * 1000 * stride * 0.6), loop=0)