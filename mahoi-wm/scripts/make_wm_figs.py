#!/usr/bin/env python3
"""Diagram figures for the World Model + Planner deck (no simulation needed)."""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# these diagrams carry Korean labels, so pin a CJK-capable family
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "wm")

CRIM = "#9E1B32"
TINT = "#FAEEF0"
CARD = "#F1F2F4"
INK = "#1F2328"
MUTED = "#6E737B"
LINE = "#D8DBDF"


def _box(ax, x, y, w, h, title, body, fill=CARD, edge=LINE, tc=INK, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=11.5,
            fontweight="bold", color=tc, zorder=3)
    if body:
        ax.text(x + w / 2, y + h - 0.42, body, ha="center", va="top", fontsize=9.0,
                color=MUTED, zorder=3, linespacing=1.5)


def _arrow(ax, p, q, color=CRIM, lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13,
                                 color=color, linewidth=lw, zorder=4,
                                 shrinkA=2, shrinkB=2))


# --------------------------------------------------------------------------- #
def pipeline(path: str) -> None:
    """The World Model + Planner loop, drawn once, used on the overview slide."""
    fig, ax = plt.subplots(figsize=(11.9, 3.9))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 3.9)
    ax.axis("off")

    Y = 2.40                                   # vertical centre of the main row
    _box(ax, 0.10, Y - 0.80, 2.15, 1.60, "현재 상태",
         "Agent 위치 · 속도\nWaypoint / Goal\nScene / Obstacle\nDependency")
    _box(ax, 2.72, Y - 0.90, 2.45, 1.80, "World Model",
         "경로 후보 ×\n양보 순서 ×\n속도 / 회피 변형", fill=TINT, edge=CRIM, tc=CRIM, lw=1.4)
    _box(ax, 5.64, Y - 1.10, 1.95, 2.20, "Future\nRollout × N", "")
    _box(ax, 8.06, Y - 0.90, 2.20, 1.80, "Planner",
         "hard  충돌·의존성\n         ·미완·livelock\nsoft   makespan·flow\n         ·대기·거리·여유",
         fill=TINT, edge=CRIM, tc=CRIM, lw=1.4)

    for dy in np.linspace(Y - 0.88, Y + 0.28, 5):
        ax.plot([5.82, 7.42], [dy, dy], "-", color="#B8BDC4", lw=1.0, zorder=3)
    ax.text(6.62, Y - 1.34, "경로와 timing이\n함께 다른 미래들", ha="center", fontsize=8.8,
            color=MUTED, style="italic", linespacing=1.4)

    for a, b in ((2.28, 2.69), (5.20, 5.61), (7.62, 8.03)):
        _arrow(ax, (a, Y), (b, Y))
    _arrow(ax, (10.29, Y), (10.92, Y))
    ax.text(11.00, Y, "Best\nFuture", ha="left", va="center", fontsize=11.5,
            fontweight="bold", color=CRIM, linespacing=1.4)

    # feedback loop, routed underneath so it never crosses a box
    yl = 0.62
    ax.plot([11.55, 11.55, 1.18], [Y - 0.34, yl, yl], "-", color=CRIM, lw=1.6,
            linestyle=(0, (5, 3)), zorder=1, solid_capstyle="round")
    _arrow(ax, (1.18, yl), (1.18, Y - 0.78), lw=1.6)
    ax.text(6.3, yl + 0.14,
            "1초만 실행 → 실제 상태 갱신 → 다시 상상   (receding horizon 4초)",
            ha="center", va="bottom", fontsize=10.5, color=CRIM, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="none"))
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def rvo_alpha(path: str) -> None:
    """Why asymmetric responsibility is what makes futures differ."""
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
    cases = [(0.30, 0.85, "A 우선 · B 양보", "#B03030", "#5B77C4"),
             (0.50, 0.50, "고전 RVO (50:50)", "#8A8A8A", "#8A8A8A"),
             (0.85, 0.30, "B 우선 · A 양보", "#5B77C4", "#B03030")]
    for ax, (aa, ab, title, ca, cb) in zip(axes, cases):
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-1.7, 1.7)
        ax.set_aspect("equal")
        ax.axis("off")
        # A travels left->right, B right->left; deviation ~ responsibility share
        t = np.linspace(-2.0, 2.0, 200)
        ax.plot(t, 0.95 * aa * np.exp(-(t ** 2) / 0.5), "-", color="#B03030", lw=2.4)
        ax.plot(t, -0.95 * ab * np.exp(-(t ** 2) / 0.5), "-", color="#5B77C4", lw=2.4)
        ax.plot([-2.0], [0], "o", ms=9, color="#B03030")
        ax.plot([2.0], [0], "o", ms=9, color="#5B77C4")
        ax.text(0, 1.40, title, ha="center", fontsize=11, fontweight="bold",
                color=INK)
        ax.text(0, -1.52, f"$\\alpha_A$={aa:.2f}   $\\alpha_B$={ab:.2f}", ha="center",
                fontsize=10, color=MUTED)
    fig.suptitle("같은 상태 · 같은 규칙 — 책임 분담 $\\alpha$ 하나로 서로 다른 미래가 된다",
                 fontsize=12, y=1.02, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def ablation_bars(path: str) -> None:
    """How much the mid-flight re-selection is actually worth."""
    names = ["crossing2", "corridor2", "deadlock2", "chain3", "fork3"]
    main = [16.4, 18.1, 15.5, 19.4, 19.4]
    nosw = [16.4, 18.2, 15.5, 20.5, 20.7]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.bar(x - 0.19, nosw, width=0.36, color="#B8BDC4",
           label="한 번 상상하고 그대로 실행", edgecolor="white", lw=0.6)
    ax.bar(x + 0.19, main, width=0.36, color=CRIM,
           label="매초 재선택 (제안 방식)", edgecolor="white", lw=0.6)
    for xi, (a, b) in enumerate(zip(nosw, main)):
        ax.text(xi - 0.19, a + 0.25, f"{a:.1f}", ha="center", fontsize=8.5, color=MUTED)
        ax.text(xi + 0.19, b + 0.25, f"{b:.1f}", ha="center", fontsize=8.5, color=CRIM,
                fontweight="bold")
        if b < a - 0.05:
            ax.text(xi + 0.19, b - 1.6, f"−{a - b:.1f}s", ha="center", fontsize=10,
                    color=CRIM, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("team completion time (s)")
    ax.set_ylim(0, 24)
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pipeline(os.path.join(OUT, "wm_pipeline.png"))
    rvo_alpha(os.path.join(OUT, "wm_alpha.png"))
    ablation_bars(os.path.join(OUT, "wm_ablation.png"))
    print(f"wrote wm_pipeline.png, wm_alpha.png, wm_ablation.png -> {OUT}")
