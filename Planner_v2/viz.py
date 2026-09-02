"""viz.py — matplotlib 애니메이션(gif) + 정적 미리보기(preview) 이미지."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as CirclePatch
from matplotlib.animation import FuncAnimation, PillowWriter

from draw_common import COLORS, draw_obstacle as _draw_obstacle


def animate(scene, agents, history, dt, out_path, title="", stride=2,
            show_velocity=False, extra_note=""):
    fig, ax = plt.subplots(figsize=(6, 6 * scene.height / scene.width))
    ax.set_xlim(0, scene.width)
    ax.set_ylim(0, scene.height)
    ax.set_aspect("equal")
    ax.set_title(title)

    for obs in scene.obstacles:
        _draw_obstacle(ax, obs)

    for i, a in enumerate(agents):
        c = COLORS[i % len(COLORS)]
        if a.path is not None:
            ax.plot(a.path.pts[:, 0], a.path.pts[:, 1], "--", color=c,
                     alpha=0.25, linewidth=1.2, zorder=2)
        ax.plot(*a.goal, marker="*", color=c, markersize=14, zorder=3)

    circles = []
    labels = []
    for i, a in enumerate(agents):
        c = COLORS[i % len(COLORS)]
        circ = CirclePatch(history[0][a.id], a.radius, color=c, alpha=0.85, zorder=4)
        ax.add_patch(circ)
        circles.append(circ)
        lbl = ax.text(*history[0][a.id], a.name, fontsize=8, ha="center",
                       va="center", zorder=5, color="white", weight="bold")
        labels.append(lbl)

    note = ax.text(0.02, 0.98, extra_note, transform=ax.transAxes, fontsize=8,
                    va="top", ha="left")
    frame_idx = list(range(0, len(history), stride))
    if frame_idx[-1] != len(history) - 1:
        frame_idx.append(len(history) - 1)

    def update(k):
        fi = frame_idx[k]
        frame = history[fi]
        for i, a in enumerate(agents):
            p = frame[a.id]
            circles[i].center = p
            labels[i].set_position(p)
        note.set_text(f"{extra_note}  t = {fi * dt:.1f}s")
        return circles + labels + [note]

    anim = FuncAnimation(fig, update, frames=len(frame_idx), interval=60, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=15))
    plt.close(fig)


def preview(scene, agents, out_path, title=""):
    """정적 미리보기: 원본 장애물 + inflate(점선) + start(○)/goal(△) + reference 경로.
    JSON 을 고치고 시뮬레이션 없이 결과부터 눈으로 확인하는 용도(§8 --preview).
    inflate 점선은 에이전트들 중 가장 큰 반지름 기준(대표값)이다 — 에이전트마다
    반지름이 다르면 실제 통과 가능 여부는 그 에이전트 자신의 반지름 기준임에
    유의.
    """
    fig, ax = plt.subplots(figsize=(6, 6 * scene.height / scene.width))
    ax.set_xlim(0, scene.width)
    ax.set_ylim(0, scene.height)
    ax.set_aspect("equal")
    ax.set_title(title)

    inflate_r = max((a.radius for a in agents), default=0.3)
    for obs in scene.obstacles:
        _draw_obstacle(ax, obs, dashed_pad=inflate_r)

    for i, a in enumerate(agents):
        c = COLORS[i % len(COLORS)]
        if a.path is not None:
            ax.plot(a.path.pts[:, 0], a.path.pts[:, 1], "-", color=c,
                     linewidth=1.6, alpha=0.9, zorder=2)
        ax.plot(*a.start, marker="o", color=c, markersize=10, zorder=3,
                markeredgecolor="black", markeredgewidth=0.6)
        ax.plot(*a.goal, marker="^", color=c, markersize=10, zorder=3,
                markeredgecolor="black", markeredgewidth=0.6)
        ax.annotate(f"{a.id}:{a.name}", a.start, textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color=c, weight="bold")
        for h in a.hold:
            wp = a.path.pos(h.own_s)
            ax.plot(*wp, marker="x", color=c, markersize=9, zorder=4)

    ax.text(0.02, 0.98,
            f"○ start   △ goal   x hold-지점   점선 = 반지름 {inflate_r:.2f} inflate(대표값)",
            transform=ax.transAxes, fontsize=7, va="top", ha="left")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
