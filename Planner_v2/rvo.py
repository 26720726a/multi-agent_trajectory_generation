"""rvo.py — sampling 기반 velocity-obstacle 회피 (직접 구현, 라이브러리 미사용).

핵심 아이디어:
  1. preferred velocity 주변에서 후보 절대속도들을 샘플한다.
  2. 각 후보에 대해, 모든 이웃(동적) + 정적 장애물(속도 0인 '이웃'으로 취급)과의
     time-to-collision(ttc) 을 계산한다.
  3. ttc 가 horizon(tau) 보다 작으면 페널티를 주는 목적함수를 최소화하는 후보를 고른다.

reciprocity 자체는 모든 에이전트가 이 동일한 규칙을 동시에 돌리는 데서 저절로
나온다 (양쪽 다 절반씩 회피 부담) — apex shift 같은 정식 RVO 트릭은 쓰지 않는다.

대칭 깨기(§2.5c, ID tie-break)는 이 파일이 비용함수 항으로 흥정하지 않는다.
detect_imminent_conflict() 로 '지금 내가 낮은 ID 라 양보할 상황인지'만 순수하게
판정하고, 실제 양보 행동(옆으로 비켜서기)은 agent.py 가 v_pref 자체를 바꿔서
표현한다 — 그러면 choose_velocity() 는 항상 '지금 원하는 속도로 안전하게
가라'는 하나의 목적만 풀면 된다.
"""
import numpy as np

from scene import Rect, Circle

HARD_PENALTY = 1.0e5
PENETRATION_W = 2.0e4   # 이미 겹친 상태에서도 '벌어지는' 후보를 상대적으로 선호


# ---------------------------------------------------------------------------
# time-to-collision
# ---------------------------------------------------------------------------

def ttc_circle(pA, v_cand, pB, vB, R):
    """원(A, 후보속도) vs 원(B, 속도 vB) 사이 최소 양의 충돌 시각.
    이미 겹쳐 있으면 0.0, 미래에 충돌 없으면 inf.
    """
    q = pA - pB                       # B 기준 A 의 상대 위치
    if float(q @ q) < R * R:
        return 0.0
    u = v_cand - vB                   # B 기준 A 의 상대 속도
    a = float(u @ u)
    if a < 1e-12:
        return np.inf                 # 상대속도 0 → 미래에 거리 안 변함(현재 안 겹침 확인됨) → 안전
    b = 2.0 * float(q @ u)
    c = float(q @ q) - R * R
    disc = b * b - 4 * a * c
    if disc < 0.0:
        return np.inf
    sq = disc ** 0.5
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    roots = [t for t in (t1, t2) if t > 1e-9]
    return min(roots) if roots else np.inf


def nearest_point_on_rect(pA, rect):
    cx = min(max(pA[0], rect.x0), rect.x1)
    cy = min(max(pA[1], rect.y0), rect.y1)
    return np.array([cx, cy])


def ttc_rect(pA, v_cand, radius, rect):
    """원(A, 반지름 radius, 후보속도) vs 정적 사각형 장애물.
    사각형 위의 '현재 가장 가까운 점'을 반지름 0 인 정적 이웃으로 보고
    ttc_circle 과 똑같은 원-원 공식을 적용한다 (= rect ⊕ disk 의 Minkowski
    합, 모서리는 자연히 둥글게 처리됨). 사각형을 그대로 AABB 로 inflate하면
    모서리 바깥까지 네모나게 부풀어 '통로 입구에 가까이 갈수록 벽에 다가가는
    것으로 오판'하는 문제가 생기는데(관측된 버그), 이 방식은 그 문제가 없다.
    """
    if pA[0] >= rect.x0 and pA[0] <= rect.x1 and pA[1] >= rect.y0 and pA[1] <= rect.y1:
        return 0.0
    target = nearest_point_on_rect(pA, rect)
    return ttc_circle(pA, v_cand, target, np.zeros(2), radius)


def ttc_obstacle(pA, v_cand, radius, obs):
    """정적 장애물(Rect|Circle) 하나에 대한 ttc — 타입별로 위 함수에 위임."""
    if isinstance(obs, Rect):
        return ttc_rect(pA, v_cand, radius, obs)
    # Circle: 장애물 자신의 반지름을 R 에 더한다(뾰족한 점이 아니라 실제 원이므로).
    return ttc_circle(pA, v_cand, np.array([obs.cx, obs.cy]), np.zeros(2), radius + obs.r)


def obstacle_outward(pA, obs):
    """이미 겹친 상태에서 '벗어나는' 방향(단위벡터) — Rect/Circle 공용."""
    if isinstance(obs, Rect):
        q = pA - nearest_point_on_rect(pA, obs)
    else:
        q = pA - np.array([obs.cx, obs.cy])
    dist = float(np.linalg.norm(q))
    return q / dist if dist > 1e-6 else np.array([1.0, 0.0])


# ---------------------------------------------------------------------------
# candidate sampling
# ---------------------------------------------------------------------------

def sample_candidates(v_pref, v_max, n_dir=16, n_speed=5):
    cands = []
    speeds = np.linspace(v_max / n_speed, v_max, n_speed)
    dirs = np.linspace(0.0, 2 * np.pi, n_dir, endpoint=False)
    for th in dirs:
        d = np.array([np.cos(th), np.sin(th)])
        for sp in speeds:
            cands.append(d * sp)
    cands.append(np.array(v_pref, float))
    cands.append(np.zeros(2))
    return cands


# ---------------------------------------------------------------------------
# main decision
# ---------------------------------------------------------------------------

def _forward_dir(v_pref, fallback):
    n = np.linalg.norm(v_pref)
    if n > 1e-6:
        return v_pref / n
    return fallback


def detect_imminent_conflict(self_id, pA, v_pref, radius, neighbors, delta=0.0,
                              tau=1.5, imminent_frac=0.6, headon_deg=60.0):
    """이 스텝의 (스냅샷) 기하만으로 '임박 + 거의 마주보는' 이웃이 있는지,
    있다면 그 이웃보다 내가 낮은 ID 라 양보 역할인지 판정한다 (순수 함수,
    상태 없음 — 언제 새로 트리거할지 판단할 때만 쓰고, 한번 트리거된 뒤
    '양보를 얼마나 유지할지'는 agent.py 가 자기 상태로 기억한다. §2.5c).
    """
    for nb in neighbors:
        if nb.get("arrived"):
            continue              # 도착해 정지한 이웃은 경쟁 상대가 아니다.
        R = radius + nb["radius"] + delta
        ttc_pref = ttc_circle(pA, v_pref, nb["pos"], nb["vel"], R)
        if ttc_pref < tau * imminent_frac:
            q = pA - nb["pos"]
            u = v_pref - nb["vel"]
            nu = np.linalg.norm(u); nq = np.linalg.norm(q)
            if nu > 1e-6 and nq > 1e-6:
                cos_a = float((u @ (-q)) / (nu * nq))
                ang = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
                if ang < headon_deg and self_id < nb["id"]:
                    return True
    return False


def right_hand_dir(v_pref, fallback):
    """진행방향 기준 오른쪽(시계방향 90도) 단위벡터 — tie-break 기본 양보 방향."""
    fwd = _forward_dir(v_pref, fallback)
    return np.array([fwd[1], -fwd[0]])


def choose_velocity(self_id, pA, v_pref, radius, v_max, neighbors, obstacles,
                     tau=1.5, tau_obs=None, delta=0.0, dt=0.25,
                     w_col=1.0, w_dev=1.0, v_prev=None, w_smooth=0.3,
                     n_dir=16, n_speed=5, force_yield=False):
    # 정적 장애물은 위치가 100% 확실하니(미래에 안 움직임) 동적 이웃만큼 긴
    # horizon 으로 미리 겁먹을 필요가 없다 — 좁은 통로 입구처럼 '벽에 바짝
    # 붙어야 지나가는' 상황에서 tau 를 그대로 쓰면 '벽 쪽으로 다가가는 것'
    # 자체가 매 스텝 계속 비용으로 잡혀 제자리에 서 있는 게 더 싸지는 함정에
    # 빠진다. 그래서 정적 장애물엔 훨씬 짧은 tau_obs(한두 스텝 앞만 봄)를 쓴다.
    if tau_obs is None:
        tau_obs = min(tau, 4.0 * dt)
    """이웃/장애물을 피하면서 v_pref 에 가장 가까운 안전한 속도를 고른다.
    tie-break(§2.5c) 는 여기서 비용 항으로 흥정하지 않는다 — agent.py 가
    '양보 중'이면 v_pref 자체를 통로 옆 대피 지점으로 바꿔서 넘겨준다. 그러면
    이 함수는 항상 '지금 원하는 속도로 안전하게 가라'는 하나의 규칙만 풀면
    되고, 두 목적(회피/양보)이 한 비용함수 안에서 서로 밀고 당기며 진동하는
    문제(관측된 버그)가 안 생긴다. force_yield 는 info 표기에만 쓴다.

    반환: (chosen_v, info)
      info = {"yielded": bool, "tie_break_triggered": bool, "unsafe": bool}
    """
    candidates = sample_candidates(v_pref, v_max, n_dir, n_speed)
    v_pref_speed = float(np.linalg.norm(v_pref))

    best_v, best_cost = None, np.inf
    best_unsafe = True
    for v in candidates:
        col = 0.0
        min_ttc = np.inf
        for nb in neighbors:
            R = radius + nb["radius"] + delta
            ttc = ttc_circle(pA, v, nb["pos"], nb["vel"], R)
            min_ttc = min(min_ttc, ttc)
            if ttc <= 1e-9:
                # 이미 겹친 상태: 그래도 '벌어지는' 후보가 '더 파고드는' 후보보다
                # 항상 싸도록 — flat penalty 만 주면 회복 방향이 안 생겨 계속
                # 파고들 수 있다(관측된 버그).
                q = pA - nb["pos"]
                dist = float(np.linalg.norm(q))
                outward = q / dist if dist > 1e-6 else np.array([1.0, 0.0])
                separating = float((v - nb["vel"]) @ outward)
                col += HARD_PENALTY - PENETRATION_W * separating
            elif ttc < tau:
                col += w_col * (1.0 / ttc - 1.0 / tau)
        for obs in obstacles:
            ttc = ttc_obstacle(pA, v, radius + delta, obs)
            min_ttc = min(min_ttc, ttc)
            if ttc <= 1e-9:
                outward = obstacle_outward(pA, obs)
                separating = float(v @ outward)
                col += HARD_PENALTY - PENETRATION_W * separating
            elif ttc < tau_obs:
                col += w_col * (1.0 / ttc - 1.0 / tau_obs)

        dev = w_dev * float(np.linalg.norm(v - v_pref))
        smooth = 0.0
        if v_prev is not None:
            # 직전 속도와 너무 다른 후보에 작은 페널티 — 두 에이전트가 서로의
            # '한 스텝 전' 속도만 보고 동시에 판단하는 구조라, 감쇠가 없으면
            # (멀 때는 둘 다 전속 돌진, 가까워지면 둘 다 급브레이크) 왕복 진동에
            # 빠질 수 있다(관측된 버그). 관성 항 하나로 그 진동을 죽인다.
            smooth = w_smooth * float(np.linalg.norm(v - v_prev))

        cost = col + dev + smooth
        if cost < best_cost:
            best_cost = cost
            best_v = v
            best_unsafe = col >= HARD_PENALTY * 0.5

    chosen_speed = float(np.linalg.norm(best_v))
    yielded = force_yield or (v_pref_speed > 1e-6 and chosen_speed < 0.5 * v_pref_speed)
    info = {
        "yielded": bool(yielded),
        "tie_break_triggered": bool(force_yield),
        "unsafe": bool(best_unsafe),
    }
    return best_v, info
