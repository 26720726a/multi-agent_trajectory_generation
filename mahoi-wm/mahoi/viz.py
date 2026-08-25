"""Figures in the style of the reference image (prior vs. GT), plus the
coordination diagram and an event Gantt chart."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np

from .coordination import Solution
from .world import Problem

OBST_FC = "#C9C9C9"
OBST_EC = "#9A9A9A"


# --------------------------------------------------------------------------- #
def _draw_scene(ax, problem: Problem):
    w, h = problem.world.width, problem.world.height
    ax.set_xlim(-0.15, w + 0.15)
    ax.set_ylim(-0.15, h + 0.15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.1)
        s.set_color("#222222")
    for r in problem.world.obstacles:
        xy, ww, hh = r.patch_xy
        ax.add_patch(Rectangle(xy, ww, hh, facecolor=OBST_FC,
                               edgecolor=OBST_EC, linewidth=0.8, zorder=1))


def _draw_markers(ax, problem: Problem, alpha: float = 1.0):
    for a in problem.agents:
        ax.plot(*a.start, "o", ms=9, mfc=a.color, mec="#333333", mew=1.1,
                alpha=alpha, zorder=5)
        ax.plot(*a.waypoint, "s", ms=7, mfc=a.color, mec="#333333", mew=0.9,
                alpha=alpha, zorder=5)
        ax.add_patch(Circle(a.waypoint, a.interact_radius, fill=False,
                            ec=a.color, ls=(0, (1.5, 1.5)), lw=1.0,
                            alpha=0.85 * alpha, zorder=4))
        ax.plot(*a.goal, "*", ms=13, mfc=a.color, mec="#333333", mew=0.8,
                alpha=alpha, zorder=5)


def _draw_traj(ax, problem: Problem, sol: Solution, alpha: float = 0.32,
               lw: float = 1.6):
    xy = sol.xy
    for i, a in enumerate(problem.agents):
        ax.plot(xy[:, i, 0], xy[:, i, 1], "-", color=a.color, lw=lw,
                alpha=alpha, zorder=3, solid_capstyle="round")


def _min_pair_dist(xy_t: np.ndarray) -> float:
    n = len(xy_t)
    if n < 2:
        return np.inf
    return min(float(np.linalg.norm(xy_t[i] - xy_t[j]))
               for i in range(n) for j in range(i + 1, n))


# --------------------------------------------------------------------------- #
def panel(ax, problem: Problem, sol: Solution, t: int, title: str = "",
          title_color: str = "#333333", show_dist: bool = True,
          violation: bool = False):
    _draw_scene(ax, problem)
    _draw_traj(ax, problem, sol)
    _draw_markers(ax, problem)
    xy = sol.xy
    tt = min(t, sol.T)
    pos = xy[tt]
    d = _min_pair_dist(pos)
    for i, a in enumerate(problem.agents):
        clash = violation and d < problem.min_sep(0, min(1, problem.n - 1)) - 1e-3
        ax.add_patch(Circle(pos[i], a.radius, facecolor=a.color, alpha=0.85,
                            edgecolor="#B00000" if clash else "#333333",
                            lw=1.8 if clash else 1.0, zorder=6))
    label = f"t={tt * problem.dt:5.1f}s"
    if show_dist:
        label += f"    d={d:.2f} m"
    ax.text(0.03, 0.965, label, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, family="monospace", color="#222222")
    if title:
        ax.set_title(title, fontsize=11, color=title_color, pad=7)


def comparison_figure(problem: Problem, sols: Sequence[Solution],
                      titles: Sequence[str], colors: Sequence[str],
                      t: int = 0, path: Optional[str] = None,
                      suptitle: str = "", figsize_per: float = 4.1):
    k = len(sols)
    fig, axes = plt.subplots(1, k, figsize=(figsize_per * k, figsize_per + 0.5))
    if k == 1:
        axes = [axes]
    for ax, s, ti, co in zip(axes, sols, titles, colors):
        panel(ax, problem, s, t, ti, co, violation=not s.feasible)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=0.995)
        # reserve room so the figure title cannot collide with two-line panel
        # titles (which the world-model comparison uses)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    else:
        fig.tight_layout()
    if path:
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return fig


def legend_figure(problem: Problem, path: str):
    handles = [
        Line2D([], [], marker="o", ls="", mfc="#888", mec="#333", ms=9, label="Start"),
        Line2D([], [], marker="s", ls="", mfc="#888", mec="#333", ms=7, label="Waypoint (HOI)"),
        Line2D([], [], marker="*", ls="", mfc="#888", mec="#333", ms=13, label="Goal"),
        Line2D([], [], ls=(0, (1.5, 1.5)), color="#888", label="HOI region"),
        Line2D([], [], ls="-", color="#888", lw=2, alpha=0.4, label="trajectory"),
    ]
    fig, ax = plt.subplots(figsize=(6, 0.7))
    ax.axis("off")
    ax.legend(handles=handles, ncol=5, loc="center", frameon=False, fontsize=9)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def animate_comparison(problem: Problem, sols: Sequence[Solution],
                       titles: Sequence[str], colors: Sequence[str],
                       path: str, fps: int = 12, stride: int = 2,
                       tail: int = 12):
    k = len(sols)
    T = max(s.T for s in sols)
    frames = list(range(0, T + 1, stride)) + [T] * tail
    fig, axes = plt.subplots(1, k, figsize=(4.1 * k, 4.6))
    if k == 1:
        axes = [axes]

    def draw(t):
        for ax, s, ti, co in zip(axes, sols, titles, colors):
            ax.clear()
            panel(ax, problem, s, t, ti, co, violation=not s.feasible)
        fig.tight_layout()
        return []

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=110,
              savefig_kwargs={"facecolor": "white"})
    plt.close(fig)


# --------------------------------------------------------------------------- #
def coordination_diagram(problem: Problem, sol: Solution, free: np.ndarray,
                         path: Optional[str] = None, other: Optional[Solution] = None):
    """2-agent coordination diagram: forbidden regions + the chosen lattice path."""
    assert problem.n == 2, "coordination diagram is only drawn for 2 agents"
    tr0, tr1 = sol.tracks
    dt = problem.dt

    coll = ~np.asarray(
        np.linalg.norm(tr0.pts[:, None, :] - tr1.pts[None, :, :], axis=2)
        >= problem.min_sep(0, 1))
    dep = ~free & ~coll

    img = np.zeros(free.shape + (3,), float)
    img[...] = 1.0
    img[coll] = np.array([0.90, 0.45, 0.45])
    img[dep] = np.array([0.55, 0.65, 0.90])

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower",
              extent=[0, tr0.n * dt, 0, tr1.n * dt], aspect="auto",
              interpolation="nearest")
    K = sol.K
    ax.plot(K[:, 0] * dt, K[:, 1] * dt, "-", color="#1a7f37", lw=2.4,
            label=f"coordination A*  ({sol.team_time:.1f}s)")
    if other is not None:
        ax.plot(other.K[:, 0] * dt, other.K[:, 1] * dt, "--", color="#444444",
                lw=1.8, label=f"{other.method}  ({other.team_time:.1f}s)")
    ax.plot([0], [0], "ko", ms=6)
    ax.plot([tr0.n * dt], [tr1.n * dt], "k*", ms=13)
    ax.set_xlabel(f"progress of {problem.agents[0].name}  (s of travel)")
    ax.set_ylabel(f"progress of {problem.agents[1].name}  (s of travel)")
    ax.set_title("Coordination diagram", fontsize=11)
    handles = [Line2D([], [], color="#E67373", lw=8, label="collision"),
               Line2D([], [], color="#8CA6E6", lw=8, label="dependency")]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0],
              fontsize=8, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
def speed_profiles(problem: Problem, sols: Sequence[Solution],
                   labels: Sequence[str], path: Optional[str] = None,
                   figsize: Tuple[float, float] = (9.6, 3.2)):
    """Speed of every agent over time, one panel per plan.

    Shows what the extra speed levels buy: the same coordination expressed as a
    gentle slowdown instead of a dead stop.
    """
    fig, axes = plt.subplots(1, len(sols), figsize=figsize, sharey=True)
    if len(sols) == 1:
        axes = [axes]
    tmax = max(s.T for s in sols) * problem.dt
    for ax, sol, lab in zip(axes, sols, labels):
        stops = 0
        for i, tr in enumerate(sol.tracks):
            k = sol.K[:, i]
            v = np.diff(k) / tr.K
            t = np.arange(len(v)) * problem.dt
            ax.step(t, v, where="post", color=problem.agents[i].color,
                    lw=1.6, alpha=0.9, label=problem.agents[i].name)
            active = ~((k[:-1] >= tr.wp_start) & (k[:-1] < tr.wp_end)) & (k[:-1] < tr.n)
            stops += int(np.sum(v[active] == 0))
        ax.set_ylim(-0.08, 1.12)
        ax.set_xlim(0, tmax * 1.02)
        ax.set_xlabel("time (s)")
        ax.set_title(f"{lab}\nteam {sol.team_time:.1f}s · full stops {stops}",
                     fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("speed / $v_{max}$")
    axes[0].legend(fontsize=8, loc="lower right", ncol=len(problem.agents))
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return fig


def gantt(problem: Problem, sols: Sequence[Solution], labels: Sequence[str],
          path: Optional[str] = None, figsize: Optional[Tuple[float, float]] = None):
    """Timeline of each agent: travel / HOI / done, one row block per method."""
    if figsize is None:
        figsize = (7.6, 1.05 * len(sols) * problem.n + 1.0)
    fig, axes = plt.subplots(len(sols), 1, figsize=figsize, sharex=True)
    if len(sols) == 1:
        axes = [axes]
    tmax = max(s.T for s in sols) * problem.dt
    for ax, sol, lab in zip(axes, sols, labels):
        ws, we = sol.event_steps()
        done = sol.completion_steps()
        for i, a in enumerate(problem.agents):
            y = problem.n - 1 - i
            ax.barh(y, done[i] * problem.dt, height=0.5, color=a.color, alpha=0.28)
            ax.barh(y, (we[i] - ws[i]) * problem.dt, left=ws[i] * problem.dt,
                    height=0.5, color=a.color, alpha=0.95)
            ax.text(done[i] * problem.dt + 0.12, y, f"{done[i] * problem.dt:.1f}s",
                    va="center", fontsize=8, color="#333")
        ax.set_yticks(range(problem.n))
        ax.set_yticklabels([a.name for a in problem.agents][::-1], fontsize=9)
        ax.set_xlim(0, tmax * 1.12)
        ax.set_title(f"{lab}   team time = {sol.team_time:.1f}s", fontsize=10,
                     loc="left", pad=4)
        ax.grid(axis="x", alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return fig
