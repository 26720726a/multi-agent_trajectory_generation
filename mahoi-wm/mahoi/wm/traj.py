"""Free-form multi-agent trajectory.

Stage-2 of the 8/18 pipeline represented a plan as an index `k_i` along a *fixed*
route, so a plan was fully described by the integer lattice path `K`.  The World
Model produces trajectories that leave their reference route whenever avoiding a
neighbour requires it, so positions can no longer be recovered from an index.

`Trajectory` therefore stores world positions directly, while exposing exactly
the read-only interface that `mahoi.validate.validate` and `mahoi.viz` expect
from a `coordination.Solution`.  That lets the *independent* validator -- the one
that already caught a lattice bug in the previous round -- be reused verbatim on
the new planner's output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Trajectory:
    """Executed (or predicted) motion of the whole team.

    Attributes
    ----------
    pos     : (T+1, n, 2) world positions, sampled every `dt` seconds.
    vel     : (T+1, n, 2) commanded velocities (vel[T] is zero padding).
    wp_in   : (n,) timestep at which each agent *arrived* at its waypoint
              (== the HOI start).  `-1` while it has not happened.
    wp_out  : (n,) timestep at which the HOI dwell finished.  `-1` if pending.
    done    : (n,) timestep at which the agent reached its goal.  `-1` if pending.
    """
    method: str
    pos: np.ndarray
    vel: np.ndarray
    dt: float
    wp_in: np.ndarray
    wp_out: np.ndarray
    done: np.ndarray
    runtime: float = 0.0
    feasible: bool = True
    note: str = ""
    extra: dict = field(default_factory=dict)

    # -- shape ------------------------------------------------------------- #
    @property
    def n(self) -> int:
        return self.pos.shape[1]

    @property
    def T(self) -> int:
        return self.pos.shape[0] - 1

    @property
    def xy(self) -> np.ndarray:
        """(T+1, n, 2) -- the name `validate` / `viz` use."""
        return self.pos

    @property
    def K(self) -> np.ndarray:
        """Compatibility shim.

        `validate` only inspects `K.shape[0]` to reject empty plans, and `viz`
        never touches it.  We expose cumulative arclength (in units of one
        full-speed step) so the array is still meaningful if inspected.
        """
        step = np.linalg.norm(np.diff(self.pos, axis=0), axis=2)      # (T, n)
        cum = np.vstack([np.zeros((1, self.n)), np.cumsum(step, axis=0)])
        return cum

    # -- events ------------------------------------------------------------ #
    def completion_steps(self) -> np.ndarray:
        out = self.done.copy()
        out[out < 0] = self.T
        return out.astype(int)

    def event_steps(self) -> Tuple[np.ndarray, np.ndarray]:
        s = self.wp_in.copy()
        e = self.wp_out.copy()
        s[s < 0] = self.T
        e[e < 0] = self.T
        return s.astype(int), e.astype(int)

    # -- metrics ----------------------------------------------------------- #
    @property
    def team_time(self) -> float:
        """Makespan: when the *last* agent reaches its goal."""
        return float(self.completion_steps().max()) * self.dt

    @property
    def flow_time(self) -> float:
        return float(self.completion_steps().sum()) * self.dt

    @property
    def travel_distance(self) -> float:
        return float(np.linalg.norm(np.diff(self.pos, axis=0), axis=2).sum())

    @property
    def total_wait(self) -> float:
        """Seconds of speed *given up* while travelling.

        Mirrors `Solution.total_wait`: for every timestep in which an agent is
        en route (not dwelling at its waypoint, not finished) we accumulate the
        fraction of `v_max` it did not use.  A full stop contributes `dt`, a
        half-speed step contributes `dt/2`.
        """
        vmax = self.extra.get("v_max")
        if vmax is None:
            return 0.0
        vmax = np.asarray(vmax, float)                                # (n,)
        speed = np.linalg.norm(self.vel[:-1], axis=2)                 # (T, n)
        lost = np.clip(1.0 - speed / vmax[None, :], 0.0, 1.0)
        return float(np.sum(lost * self._active_mask())) * self.dt

    @property
    def mean_speed_ratio(self) -> float:
        vmax = self.extra.get("v_max")
        if vmax is None:
            return 1.0
        vmax = np.asarray(vmax, float)
        speed = np.linalg.norm(self.vel[:-1], axis=2)
        active = self._active_mask()
        den = float(active.sum())
        if den <= 0:
            return 1.0
        return float(np.sum((speed / vmax[None, :]) * active)) / den

    @property
    def n_full_stops(self) -> int:
        """Timesteps an agent stood completely still while en route."""
        speed = np.linalg.norm(self.vel[:-1], axis=2)
        return int(np.sum((speed < 1e-6) & (self._active_mask() > 0)))

    def _active_mask(self) -> np.ndarray:
        """(T, n) 1.0 where the agent is travelling (not dwelling, not done)."""
        T, n = self.T, self.n
        t = np.arange(T)[:, None]
        wi = np.where(self.wp_in < 0, T + 1, self.wp_in)[None, :]
        wo = np.where(self.wp_out < 0, T + 1, self.wp_out)[None, :]
        dn = np.where(self.done < 0, T + 1, self.done)[None, :]
        dwelling = (t >= wi) & (t < wo)
        finished = t >= dn
        return (~dwelling & ~finished).astype(float)

    # -- helpers ----------------------------------------------------------- #
    def truncate(self, T: int) -> "Trajectory":
        """First `T` timesteps as a new Trajectory (events beyond T become -1)."""
        T = int(min(T, self.T))
        cut = lambda a: np.where(a > T, -1, a)                        # noqa: E731
        return Trajectory(
            method=self.method, pos=self.pos[:T + 1].copy(),
            vel=self.vel[:T + 1].copy(), dt=self.dt,
            wp_in=cut(self.wp_in.copy()), wp_out=cut(self.wp_out.copy()),
            done=cut(self.done.copy()), runtime=self.runtime,
            feasible=self.feasible, note=self.note, extra=dict(self.extra))


def concat(parts: Sequence[Trajectory], method: str = "") -> Trajectory:
    """Glue consecutive execution segments into one trajectory.

    Each part starts where the previous one ended, so the duplicated boundary
    sample is dropped.  Event timesteps are shifted by the running offset and
    the *first* occurrence wins.
    """
    parts = [p for p in parts if p.T >= 0]
    if not parts:
        raise ValueError("nothing to concatenate")
    n = parts[0].n
    pos = [parts[0].pos]
    vel = [parts[0].vel[:-1]] if parts[0].T > 0 else []
    wp_in = parts[0].wp_in.astype(float).copy()
    wp_out = parts[0].wp_out.astype(float).copy()
    done = parts[0].done.astype(float).copy()
    off = parts[0].T

    for p in parts[1:]:
        if p.T <= 0:
            continue
        pos.append(p.pos[1:])
        vel.append(p.vel[:-1])
        for tgt, src in ((wp_in, p.wp_in), (wp_out, p.wp_out), (done, p.done)):
            fresh = (tgt < 0) & (np.asarray(src) >= 0)
            tgt[fresh] = np.asarray(src, float)[fresh] + off
        off += p.T

    pos = np.concatenate(pos, axis=0)
    vel = np.concatenate(vel + [np.zeros((1, n, 2))], axis=0) if vel else \
        np.zeros_like(pos)
    return Trajectory(method=method or parts[0].method, pos=pos, vel=vel,
                      dt=parts[0].dt, wp_in=wp_in.astype(int),
                      wp_out=wp_out.astype(int), done=done.astype(int),
                      runtime=sum(p.runtime for p in parts),
                      extra=dict(parts[-1].extra))
