"""agent.py — per-agent 스텝: preferred velocity + dependency hold + RVO 보정.

각 에이전트는 board 의 '이번 스텝 스냅샷'만 보고 독립적으로 자기 다음 속도를
계산한다 (중앙 계획자 없음). 매 스텝:
  1. 자기 reference 경로 위 현재 진행도 s 를 (현재 위치 투영으로) 구한다.
  2. s + lookahead 지점을 향하는 preferred velocity 를 만든다.
  3. dependency: 자기 hold-지점 근처인데 선행이 아직 안 지났으면 v_pref=0 (정지 대기).
     — 미래를 예측하지 않고 board 의 '통과 이벤트' 플래그만 본다.
  4. rvo.choose_velocity 로 이웃/장애물 충돌을 피하는 실제 속도로 보정한다.
  5. 한 스텝 이동하고, 자기 다음 상태를 board 에 commit. 자기 announce 지점을
     지났으면 이벤트 플래그를 세운다.
"""
from dataclasses import dataclass, field
import numpy as np

import rvo


@dataclass
class HoldSpec:
    predecessor_id: int
    wp_name: str
    own_s: float          # 이 에이전트 '자신의' 경로 위에서, 이 지점의 arc-length


@dataclass
class AnnounceSpec:
    wp_name: str
    own_s: float


class Agent:
    def __init__(self, agent_id, name, start, goal, radius, v_max,
                 lookahead=0.5, slow_radius=0.5, arrive_tol=0.08,
                 hold_margin=0.35, yield_dir=None, yield_cooldown_steps=16,
                 yield_pullover_dist=0.4, yield_speed_frac=0.6):
        self.id = agent_id
        self.name = name
        self.start = np.asarray(start, float)
        self.goal = np.asarray(goal, float)
        self.radius = radius
        self.v_max = v_max
        self.lookahead = lookahead
        self.slow_radius = slow_radius
        self.arrive_tol = arrive_tol
        self.hold_margin = hold_margin
        self.yield_dir = yield_dir
        self.yield_cooldown_steps = yield_cooldown_steps
        self.yield_pullover_dist = yield_pullover_dist
        self.yield_speed_frac = yield_speed_frac

        self.path = None                  # path.Path, build_path() 로 채움
        self.hold = []                    # list[HoldSpec] — 내가 기다려야 할 것
        self.announce = []                # list[AnnounceSpec] — 내가 알려야 할 것
        self._yield_cooldown = 0          # §2.5c: 한번 양보를 시작하면 몇 스텝은
                                           # 유지한다 — 안 그러면 서로 '한 스텝
                                           # 전' 상태만 보고 판단이 매 스텝 뒤집혀
                                           # 왕복 진동에 빠질 수 있다(관측된 버그).
        self._yield_target = None         # 양보 중 목표로 삼는 '대피 지점'(고정)

    # ---- dependency 배선 ----
    def add_hold(self, predecessor_id, own_s, wp_name):
        self.hold.append(HoldSpec(predecessor_id, wp_name, own_s))

    def add_announce(self, own_s, wp_name):
        self.announce.append(AnnounceSpec(wp_name, own_s))

    def preferred_velocity(self, pos, s):
        goal_dist = float(np.linalg.norm(self.goal - pos))
        if goal_dist < self.arrive_tol:
            return np.zeros(2), True
        speed = self.v_max
        if goal_dist < self.slow_radius:
            speed = self.v_max * max(goal_dist / self.slow_radius, 0.15)
        look = self.path.pos(s + self.lookahead)
        d = look - pos
        nd = np.linalg.norm(d)
        if nd < 1e-6:
            d = self.goal - pos
            nd = np.linalg.norm(d)
        v_pref = (d / nd) * speed if nd > 1e-6 else np.zeros(2)
        return v_pref, False

    def step(self, board, snap_states, snap_passed, dt, scene, other_ids,
              tau=1.5, delta=0.0, tie_break=True, w_col=1.0, w_dev=1.0,
              n_dir=16, n_speed=5):
        self_state = snap_states[self.id]
        pos = self_state.pos
        if self_state.arrived:
            board.commit(self.id, pos, np.zeros(2), self.radius, True)
            return {"yielded": False, "tie_break_triggered": False,
                    "held": False, "unsafe": False}

        s = self.path.nearest_s(pos)
        v_pref, at_goal = self.preferred_velocity(pos, s)

        held = False
        if not at_goal:
            for h in self.hold:
                if s >= h.own_s - self.hold_margin:
                    if (h.predecessor_id, h.wp_name) not in snap_passed:
                        v_pref = np.zeros(2)
                        held = True

        neighbors = [
            {"id": oid, "pos": snap_states[oid].pos, "vel": snap_states[oid].vel,
             "radius": snap_states[oid].radius, "arrived": snap_states[oid].arrived}
            for oid in other_ids
        ]
        path_dir = None
        look = self.path.pos(min(s + self.lookahead, self.path.length))
        dvec = look - pos
        if np.linalg.norm(dvec) > 1e-6:
            path_dir = dvec / np.linalg.norm(dvec)

        # §2.5c 대칭 깨기: 쿨다운 중이면 그대로 유지, 아니면 이번 스텝 기하로
        # 새로 트리거할지 판정한다. '양보 중'이라는 결정 자체는 한번 내리면
        # cooldown 동안 뒤집지 않는다 — 그래야 두 에이전트가 서로 '한 스텝 전'
        # 상태만 보고 매 스텝 마음을 바꿔 왕복 진동에 빠지는 걸 막는다.
        force_yield = False
        if tie_break and not held and not at_goal:
            if self._yield_cooldown > 0:
                self._yield_cooldown -= 1
                force_yield = True
            else:
                force_yield = rvo.detect_imminent_conflict(
                    self.id, pos, v_pref, self.radius, neighbors, delta=delta, tau=tau)
                if force_yield:
                    self._yield_cooldown = self.yield_cooldown_steps
                    yd = self.yield_dir
                    if yd is None:
                        yd = rvo.right_hand_dir(v_pref, path_dir if path_dir is not None
                                                 else np.array([1.0, 0.0]))
                    yd = np.asarray(yd, float)
                    n = np.linalg.norm(yd)
                    yd = yd / n if n > 1e-9 else np.array([0.0, -1.0])
                    self._yield_target = pos + yd * self.yield_pullover_dist

        if force_yield and self._yield_target is not None:
            # 양보 중엔 '내 경로를 계속 따라가기'가 아니라 '통로 옆 대피
            # 지점으로 붙어서 기다리기'가 목표 속도가 된다 — 도착하면 그냥
            # 정지(v_pref=0)해서 대기. 실제 충돌 회피는 여전히 rvo 가 담당.
            d = self._yield_target - pos
            nd = np.linalg.norm(d)
            if nd < 0.08:
                v_pref = np.zeros(2)
            else:
                v_pref = (d / nd) * min(self.v_max * self.yield_speed_frac, nd / dt)

        v_chosen, info = rvo.choose_velocity(
            self.id, pos, v_pref, self.radius, self.v_max, neighbors,
            scene.obstacles, tau=tau, delta=delta, dt=dt,
            w_col=w_col, w_dev=w_dev, v_prev=self_state.vel,
            n_dir=n_dir, n_speed=n_speed, force_yield=force_yield,
        )

        new_pos = pos + v_chosen * dt
        arrived = at_goal
        if not arrived and np.linalg.norm(self.goal - new_pos) < self.arrive_tol:
            arrived = True
            new_pos = self.goal.copy()
            v_chosen = np.zeros(2)

        board.commit(self.id, new_pos, v_chosen, self.radius, arrived)

        new_s = self.path.nearest_s(new_pos)
        for a in self.announce:
            if new_s >= a.own_s - 1e-6:
                board.announce(self.id, a.wp_name)

        info["held"] = held
        return info


def link_dependency(successor: Agent, predecessor: Agent, point_xy, wp_name):
    """successor 는 predecessor 가 point_xy 를 지날 때까지 자기 WP 직전에서 대기."""
    succ_s = successor.path.nearest_s(point_xy)
    pred_s = predecessor.path.nearest_s(point_xy)
    successor.add_hold(predecessor.id, succ_s, wp_name)
    predecessor.add_announce(pred_s, wp_name)
