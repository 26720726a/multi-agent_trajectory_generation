#!/usr/bin/env python3
"""P3-1 — P2-1 라벨을 학습용 텐서 데이터셋으로 만든다.

    python scripts/build_dataset.py                    # 기본 (augment 꺼짐)
    python scripts/build_dataset.py --augment-rot      # 회전 augmentation 켜기

입력
  bench/runs/P2-1_labels.csv   라벨 (completed / makespan_est)
  bench/runs/P1b_modes.csv     t=0 상태 (pos/vel/phase)
  인스턴스 스펙에서 재구성한 Problem  (waypoint/goal/장애물 — 로그에 없다)

출력
  bench/runs/dataset/{train,val,test}.npz
  bench/runs/dataset/norm.json          정규화 상수 (추론 때 같은 값을 써야 한다)
  results/P3-1_split.json               uid 분할

설계에서 중요한 두 가지
  * **positional encoding 을 넣지 않는다.** 에이전트 토큰은 그 에이전트의
    물리량과 이 mode 가 그에게 할당한 것만으로 만들어진다.  순열 정보는
    `yield_rank` 스칼라 하나로만 들어간다 — 그래야 에이전트 순서를 바꿔도
    같은 상황이 같은 예측을 낳는다.
  * **분할은 uid 단위다.** 행 단위로 섞으면 같은 인스턴스의 다른 mode 가
    train 과 val 에 흩어져 검증 곡선이 낙관적으로 망가진다.  조용히 실패하므로
    사후에 알아채기 어렵다.
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.generate import Instance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = f"{ROOT}/bench/runs/P2-1_labels.csv"
MODES  = f"{ROOT}/bench/runs/P1b_modes.csv"
OUTDIR = f"{ROOT}/bench/runs/dataset"
SPLIT_JSON = f"{ROOT}/results/P3-1_split.json"

# -- 정규화 상수.  추론 때 반드시 같은 값을 써야 하므로 norm.json 에 함께 쓴다 -- #
COORD_SCALE = 10.0     # 20m 를 [-1, 1] 로.  방은 8~12m 이므로 여유가 있다
V_MAX_NORM  = 1.5      # 실제 v_max 는 1.20 이라 정규화 속도는 ±0.8 을 넘지 않는다
MAX_AGENTS  = 8        # 지금 데이터는 2~4.  8까지 자리를 잡아 둔다
N_PHASE     = 3        # TO_WP / DWELL / TO_GOAL  (DONE 은 t=0 에 나오지 않는다)
N_ROUTE     = 4        # route library 최대 크기 (k_routes=2 -> k*k=4)
DEP_MODES   = ("chain", "fork", "none")

AGENT_FIELDS = (["pos_x","pos_y","vel_x","vel_y","wp_x","wp_y","goal_x","goal_y"]
                + [f"phase_{i}" for i in range(N_PHASE)] + ["rho"]
                + [f"route_{i}" for i in range(N_ROUTE)]
                + ["yield_rank_norm","cautious","split_side"])
OBST_FIELDS  = ["cx","cy","radius"]
GLOBAL_FIELDS = ["n_agents_norm"] + [f"dep_{d}" for d in DEP_MODES] + ["total_rank_norm"]


# --------------------------------------------------------------------------- #
#  입력
# --------------------------------------------------------------------------- #
def load_labels():
    """라벨 로드 + (uid, mode_label) 중복 제거.

    P2-1b 에서 확인: 두 seed 에 모두 나온 조합 4,413개는 completed 와
    team_time_actual 이 100% 동일하다.  `follow` 는 switching=False 라
    (문제, mode) 가 같으면 결정론적이고 planner_seed 가 결과에 영향을 주지
    않는다.  남기는 쪽은 **planner_seed 가 작은 행** 으로 고정한다 (결정론적
    규칙이어야 재실행 시 같은 데이터셋이 나온다).
    """
    rows = list(csv.DictReader(open(LABELS, newline="", encoding="utf-8")))
    best = {}
    for r in rows:
        k = (r["uid"], r["mode_label"])
        if k not in best or int(r["planner_seed"]) < int(best[k]["planner_seed"]):
            best[k] = r
    keep = [best[k] for k in sorted(best)]
    return rows, keep


def load_states():
    """t=0 의 pos/vel/phase.  (uid, planner_seed, mode_idx) 로 라벨과 조인된다."""
    st = {}
    for r in csv.DictReader(open(MODES, newline="", encoding="utf-8")):
        if r["replan_idx"] != "0":
            continue
        st[(r["uid"], r["planner_seed"], r["mode_idx"])] = r
    return st


def load_specs():
    """uid -> Instance 스펙.  waypoint/goal/장애물은 로그에 없어 재구성해야 한다."""
    spec = {}
    for r in csv.DictReader(open(f"{ROOT}/bench/runs/P1b.csv", newline="", encoding="utf-8")):
        if r["uid"] in spec:
            continue
        lo, hi = r["n_obstacles"].split("-")
        spec[r["uid"]] = (int(r["instance_seed"]), int(r["n_agents"]), r["dep_mode"],
                          float(r["size"]), (int(lo), int(hi)),
                          float(r["couple_prob"]), float(r["couple_dist"]))
    return spec


def static_features(spec):
    """인스턴스마다 한 번만: 시작/waypoint/goal, 장애물, 방 크기."""
    out = {}
    for uid, s in spec.items():
        p = Instance(*s).build()
        obs = []
        for r in p.world.obstacles:
            cx, cy = 0.5 * (r.x0 + r.x1), 0.5 * (r.y0 + r.y1)
            # 사각형을 (중심, 외접원 반경) 으로 옮긴다.  가로세로 비가 큰
            # 장애물은 실제보다 크게 잡힌다 — 보고서의 한계 항목 참조.
            rad = 0.5 * float(np.hypot(r.x1 - r.x0, r.y1 - r.y0))
            obs.append((cx, cy, rad))
        out[uid] = dict(
            size=p.world.width,
            wp=np.array([a.waypoint for a in p.agents], float),
            goal=np.array([a.goal for a in p.agents], float),
            rho=np.array([a.radius for a in p.agents], float),
            obst=np.array(obs, float).reshape(-1, 3))
    return out


# --------------------------------------------------------------------------- #
#  특성
# --------------------------------------------------------------------------- #
def _xy(s):
    return np.array([[float(v) for v in p.split(",")] for p in s.split(";")], float)


def build_rows(keep, states, stat, n_modes, rot=None):
    """(N, ...) 텐서.  rot 이 주어지면 그 각도만큼 회전시킨다 (augmentation)."""
    N = len(keep)
    A = np.zeros((N, MAX_AGENTS, len(AGENT_FIELDS)), np.float32)
    Am = np.zeros((N, MAX_AGENTS), bool)
    O_max = max(len(stat[r["uid"]]["obst"]) for r in keep)
    O = np.zeros((N, O_max, 3), np.float32)
    Om = np.zeros((N, O_max), bool)
    G = np.zeros((N, len(GLOBAL_FIELDS)), np.float32)
    y_cls = np.zeros(N, np.uint8)
    y_reg = np.zeros(N, np.float32)          # makespan_est (증류 타깃)
    y_act = np.full(N, np.nan, np.float32)   # team_time_actual (실측, 참고용)
    reg_mask = np.zeros(N, bool)
    uids = []

    if rot is not None:
        c, s = float(np.cos(rot)), float(np.sin(rot))
        R = np.array([[c, -s], [s, c]])

    for i, r in enumerate(keep):
        uid = r["uid"]; uids.append(uid)
        sf = stat[uid]; half = 0.5 * sf["size"]
        stt = states[(uid, r["planner_seed"], r["mode_idx"])]
        pos = _xy(stt["pos"]); vel = _xy(stt["vel"])
        phase = np.array([int(x) for x in stt["phase"].split(",")], int)
        routes = [int(x) for x in r["routes"].split(",")]
        yrank = [int(x) for x in r["yield_rank"].split(",")]
        n = len(pos)

        # 좌표를 방 중심 기준으로 옮기고 스케일한다
        def norm_xy(a):
            a = (a - half) / COORD_SCALE
            return a @ R.T if rot is not None else a
        P, W, Gl = norm_xy(pos), norm_xy(sf["wp"]), norm_xy(sf["goal"])
        V = vel / V_MAX_NORM
        if rot is not None:
            V = V @ R.T

        for a in range(n):
            f = np.zeros(len(AGENT_FIELDS), np.float32)
            f[0:2] = P[a]; f[2:4] = V[a]; f[4:6] = W[a]; f[6:8] = Gl[a]
            if 0 <= phase[a] < N_PHASE:
                f[8 + phase[a]] = 1.0
            f[8 + N_PHASE] = sf["rho"][a] / COORD_SCALE
            base = 9 + N_PHASE
            if 0 <= routes[a] < N_ROUTE:
                f[base + routes[a]] = 1.0
            f[base + N_ROUTE] = yrank[a] / max(1, n - 1)
            f[base + N_ROUTE + 1] = 1.0 if r["cautious"] == "True" else 0.0
            f[base + N_ROUTE + 2] = 1.0 if r["split_side"] == "True" else 0.0
            A[i, a] = f
        Am[i, :n] = True

        ob = sf["obst"]
        if len(ob):
            cxy = (ob[:, :2] - half) / COORD_SCALE
            if rot is not None:
                cxy = cxy @ R.T
            O[i, :len(ob), :2] = cxy
            O[i, :len(ob), 2] = ob[:, 2] / COORD_SCALE   # 반경은 회전 불변
            Om[i, :len(ob)] = True

        G[i, 0] = n / MAX_AGENTS
        G[i, 1 + DEP_MODES.index(r["dep_mode"])] = 1.0
        nm = n_modes[(uid, r["planner_seed"])]
        G[i, 4] = (int(r["total_rank"]) - 1) / max(1, nm - 1)

        y_cls[i] = 1 if r["completed"] == "True" else 0
        y_reg[i] = float(r["makespan_est"])
        if y_cls[i]:
            reg_mask[i] = True
            if r["team_time_actual"]:
                y_act[i] = float(r["team_time_actual"])
    return dict(agent=A, agent_mask=Am, obst=O, obst_mask=Om, glob=G,
                y_cls=y_cls, y_reg=y_reg, y_actual=y_act, reg_mask=reg_mask,
                uid=np.array(uids))


# --------------------------------------------------------------------------- #
#  분할 — uid 단위, n x dep_mode 층화
# --------------------------------------------------------------------------- #
def split_uids(keep, seed=20260829, frac=(0.70, 0.15, 0.15)):
    cell = defaultdict(set)
    for r in keep:
        cell[(r["n_agents"], r["dep_mode"])].add(r["uid"])
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for c in sorted(cell):
        us = sorted(cell[c])
        rng.shuffle(us)
        n = len(us); a = int(round(frac[0] * n)); b = a + int(round(frac[1] * n))
        out["train"] += us[:a]; out["val"] += us[a:b]; out["test"] += us[b:]
    return {k: sorted(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment-rot", action="store_true",
                    help="회전 augmentation (기본 꺼짐).  P3-2 에서 유무를 비교할 것")
    ap.add_argument("--n-rot", type=int, default=3, help="train 에 붙일 회전 사본 수")
    args = ap.parse_args()

    t0 = time.perf_counter()
    raw, keep = load_labels()
    print(f"라벨 {len(raw):,} -> 중복 제거 후 {len(keep):,}")
    states = load_states()
    spec = load_specs()
    stat = static_features(spec)
    n_modes = defaultdict(int)
    for r in raw:
        n_modes[(r["uid"], r["planner_seed"])] += 1
    print(f"인스턴스 {len(stat):,} 재구성  ({time.perf_counter()-t0:.0f}s)")

    data = build_rows(keep, states, stat, n_modes)
    sp = split_uids(keep)
    json.dump({k: v for k, v in sp.items()}, open(SPLIT_JSON, "w"), indent=1)

    os.makedirs(OUTDIR, exist_ok=True)
    idx = {u: s for s, us in sp.items() for u in us}
    where = np.array([idx[u] for u in data["uid"]])
    for name in ("train", "val", "test"):
        m = where == name
        d = {k: v[m] for k, v in data.items()}
        if name == "train" and args.augment_rot:
            parts = [d]
            for j in range(args.n_rot):
                ang = 2 * np.pi * (j + 1) / (args.n_rot + 1)
                rd = build_rows([keep[i] for i in np.flatnonzero(m)],
                                states, stat, n_modes, rot=ang)
                parts.append(rd)
            d = {k: np.concatenate([p[k] for p in parts], 0) for k in d}
        np.savez_compressed(f"{OUTDIR}/{name}.npz", **d)
        print(f"  {name:<6} {m.sum():>7,} rows / {len(sp[name]):>5} uid"
              f"  -> {os.path.getsize(f'{OUTDIR}/{name}.npz')/2**20:.1f} MB")

    json.dump(dict(coord_scale=COORD_SCALE, v_max_norm=V_MAX_NORM,
                   max_agents=MAX_AGENTS, n_phase=N_PHASE, n_route=N_ROUTE,
                   dep_modes=list(DEP_MODES), agent_fields=AGENT_FIELDS,
                   obst_fields=OBST_FIELDS, global_fields=GLOBAL_FIELDS,
                   coord_center="방 중심 (size/2, size/2)",
                   note="추론 시 반드시 같은 상수를 쓸 것"),
              open(f"{OUTDIR}/norm.json", "w"), indent=1, ensure_ascii=False)
    print(f"완료 {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
