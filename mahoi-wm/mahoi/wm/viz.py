"""Figures specific to the World-Model + Planner experiment.

The scene drawing helpers from `mahoi.viz` are reused verbatim so the new
figures sit next to the 8/18 ones without a style break.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

from ..viz import _draw_markers, _draw_scene
from ..world import Problem
from .planner import Cost
from .traj import Trajectory
from .worldmodel import Rollout

GREY = "#9AA0A6"
PICK = "#1a7f37"


# --------------------------------------------------------------------------- #
def rollout_fan(problem: Problem, rolls: Sequence[Rollout], best: int,
                path: Optional[str] = None, title: str = "",
                max_show: int = 24, figsize: Tuple[float, float] = (5.6, 5.4)):
    """Every future the World Model imagined, with the Planner's pick on top.

    This is the figure that carries the whole idea: from one state the model
    produces many *spatially and temporally different* multi-agent futures, and
    the Planner's explicit cost -- not a learned value -- chooses among them.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _draw_scene(ax, problem)

    shown = list(range(len(rolls)))[:max_show]
    for k in shown:
        if k == best:
            continue
        xy = rolls[k].traj.pos
        for i, a in enumerate(problem.agents):
            ax.plot(xy[:, i, 0], xy[:, i, 1], "-", color=a.color, lw=1.1,
                    alpha=0.30, zorder=2, solid_capstyle="round")

    xy = rolls[best].traj.pos
    for i, a in enumerate(problem.agents):
        # white casing makes the selected future readable on top of the fan
        ax.plot(xy[:, i, 0], xy[:, i, 1], "-", color="white", lw=4.4,
                alpha=0.95, zorder=3, solid_capstyle="round")
        ax.plot(xy[:, i, 0], xy[:, i, 1], "-", color=a.color, lw=2.4,
                alpha=1.0, zorder=4, solid_capstyle="round")
        ax.add_patch(Circle(xy[0, i], a.radius, facecolor=a.color, alpha=0.55,
                            edgecolor="#333", lw=1.0, zorder=5))
    _draw_markers(ax, problem)

    n_other = max(0, len(shown) - 1)
    handles = [Line2D([], [], color=GREY, lw=1.2, alpha=0.6,
                      label=f"{n_other} rejected futures"),
               Line2D([], [], color="#333", lw=2.4, path_effects=None,
                      label=f"selected: {rolls[best].mode.label(problem)}")]
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.95)
    ax.set_title(title or "World Model rollouts -> Planner choice",
                 fontsize=11, pad=7)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
#: (display name, Cost attribute, CostWeights attribute, colour)
_TERMS = [("makespan", "makespan", "make", "#C43F3F"),
          ("flow time", "flow", "flow", "#E08A5B"),
          ("yielding", "wait", "wait", "#3C6BB0"),
          ("distance", "dist", "dist", "#5FA8D3"),
          ("clearance", "clear", "clear", "#3E9E62"),
          ("roughness", "turn", "turn", "#8A5FBF"),
          ("route dev.", "dev", "dev", "#B0B7BE")]


def cost_bars(costs: Sequence[Cost], weights, path: Optional[str] = None,
              top: int = 8, figsize: Tuple[float, float] = (8.4, 4.2)):
    """Stacked breakdown of J for the cheapest `top` modes.

    Infeasible modes are drawn hatched with their violation count, so it is
    visible at a glance whether the Planner is choosing between good options or
    picking the least-bad one.
    """
    order = np.argsort([c.total for c in costs])[:top]
    sel = [costs[int(k)] for k in order]
    labels = [c.label for c in sel]
    y = np.arange(len(sel))[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    left = np.zeros(len(sel))
    for name, ckey, wkey, colour in _TERMS:
        w = getattr(weights, wkey)
        val = np.array([getattr(c, ckey) for c in sel]) * w
        ax.barh(y, val, left=left, height=0.62, color=colour, label=name,
                edgecolor="white", linewidth=0.6)
        left = left + val
    for k, c in enumerate(sel):
        tag = f"  J={c.total:.2f}"
        if not c.feasible:
            tag += f"  [{c.hard_count()} violation(s)]"
            ax.barh(y[k], left[k], height=0.62, facecolor="none",
                    edgecolor="#B00000", hatch="///", linewidth=1.0)
        ax.text(left[k] * 1.01, y[k], tag, va="center", fontsize=8, color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, family="monospace")
    ax.set_xlim(0, left.max() * 1.22)
    ax.set_xlabel("weighted cost contribution")
    ax.set_title("Planner cost breakdown  (cheapest future at the top)",
                 fontsize=11, loc="left", pad=6)
    ax.legend(fontsize=8, ncol=4, loc="lower right", framealpha=0.95)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
def switch_timeline(problem: Problem, switches: Sequence[Dict[str, object]],
                    traj: Trajectory, path: Optional[str] = None,
                    figsize: Tuple[float, float] = (9.2, 3.6)):
    """Which future was being followed at each moment, and what it promised."""
    modes = []
    for s in switches:
        if s["mode"] not in modes:
            modes.append(s["mode"])
    idx = [modes.index(s["mode"]) for s in switches]
    t = [float(s["t"]) for s in switches]
    est = [float(s["makespan_est"]) for s in switches]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw={"height_ratios": [1.35, 1.0]})
    ax0.step(t, idx, where="post", color="#333", lw=1.8)
    for k, s in enumerate(switches):
        colour = PICK if s["feasible"] else "#B00000"
        ax0.plot(t[k], idx[k], "o", ms=5.5, color=colour, zorder=3)
        if s["switched"]:
            ax0.axvline(t[k], color="#C43F3F", lw=1.0, ls=":", alpha=0.8)
    ax0.set_yticks(range(len(modes)))
    ax0.set_yticklabels(modes, fontsize=8, family="monospace")
    ax0.set_ylabel("active future")
    ax0.grid(axis="x", alpha=0.25)
    ax0.set_axisbelow(True)
    n_sw = sum(1 for s in switches if s["switched"])
    ax0.set_title(f"Plan selected over time   ({len(switches)} decisions, "
                  f"{n_sw} switches)", fontsize=11, loc="left", pad=6)

    ax1.plot(t, est, "-o", color="#3C6BB0", ms=4, lw=1.6,
             label="predicted time still needed")
    ax1.axhline(0, color="#888", lw=0.8)
    ax1.plot(t, [traj.team_time - x for x in t], "--", color="#C43F3F", lw=1.4,
             label="actual time still needed")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("seconds")
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax1.grid(alpha=0.25)
    ax1.set_axisbelow(True)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
def speed_trace(problem: Problem, trajs: Sequence[Trajectory],
                labels: Sequence[str], path: Optional[str] = None,
                figsize: Optional[Tuple[float, float]] = None):
    """Speed of every agent over time -- shows yielding as slowdown vs stop."""
    if figsize is None:
        figsize = (4.6 * len(trajs), 3.2)
    fig, axes = plt.subplots(1, len(trajs), figsize=figsize, sharey=True)
    if len(trajs) == 1:
        axes = [axes]
    tmax = max(t.T for t in trajs) * problem.dt
    for ax, tr, lab in zip(axes, trajs, labels):
        active = tr._active_mask()
        stops = 0
        for i, a in enumerate(problem.agents):
            v = np.linalg.norm(tr.vel[:-1, i, :], axis=1) / a.v_max
            t = np.arange(len(v)) * problem.dt
            ax.plot(t, v, "-", color=a.color, lw=1.5, alpha=0.9, label=a.name)
            stops += int(np.count_nonzero((v < 1e-6) & (active[:, i] > 0)))
        ax.set_ylim(-0.08, 1.12)
        ax.set_xlim(0, tmax * 1.02)
        ax.set_xlabel("time (s)")
        ax.set_title(f"{lab}\nteam {tr.team_time:.1f}s · full stops {stops}",
                     fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("speed / $v_{max}$")
    axes[0].legend(fontsize=8, loc="lower right", ncol=problem.n)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
def method_bars(rows: Sequence[Dict[str, object]], methods: Sequence[str],
                colours: Sequence[str], path: Optional[str] = None,
                figsize: Tuple[float, float] = (9.6, 4.2)):
    """Team completion time per scenario, one bar group per method.

    The critical-path lower bound is drawn as a tick so every bar can be read
    as "how far above the best conceivable schedule".
    """
    names = [r["scenario"] for r in rows]
    x = np.arange(len(names))
    w = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=figsize)
    for k, (m, c) in enumerate(zip(methods, colours)):
        vals = [r.get(m) for r in rows]
        xs = x - 0.4 + w * (k + 0.5)
        good = [(xi, v) for xi, v in zip(xs, vals) if v is not None and v == v]
        ax.bar([g[0] for g in good], [g[1] for g in good], width=w * 0.92,
               color=c, label=m, edgecolor="white", linewidth=0.6)
        for xi, v in zip(xs, vals):
            if v is None or v != v:
                ax.text(xi, 1.0, "x", ha="center", va="bottom", fontsize=11,
                        color="#B00000", fontweight="bold")
            else:
                ax.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=7.5,
                        color="#333")
    for xi, r in zip(x, rows):
        ax.plot([xi - 0.44, xi + 0.44], [r["lower_bound"]] * 2, "-",
                color="#111", lw=1.6, zorder=5)
    ax.plot([], [], "-", color="#111", lw=1.6, label="critical-path bound")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("team completion time (s)")
    ax.set_title("Team completion time by method   (x = no solution found)",
                 fontsize=11, loc="left", pad=6)
    ax.legend(fontsize=8.5, ncol=3, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return fig
