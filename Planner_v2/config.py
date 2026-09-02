"""config.py — scenario.json 로더 + 검증기.

장애물·에이전트·dependency·파라미터를 코드 밖(JSON)으로 뺀다. reference
경로는 저장하지 않는다 — 로드할 때마다 (현재 장애물 + start/waypoint/goal)
로부터 A*를 다시 돌려서 만든다(`path_override` 가 있는 에이전트는 예외).
그래야 장애물이나 anchor 를 고치면 경로가 자동으로 따라온다.

좌표 규약: 원점(0,0) 좌하단, +x 오른쪽, +y 위. `params.grid_w`/`grid_h` 는
곧 세계의 폭·높이(m) 이기도 하다(셀 1개 = 1 m). rect 장애물은 셀 단위
(x,y,w,h), circle 장애물과 에이전트 좌표는 이 프레임의 실수 [x,y].

검증 실패는 무엇이·어디서(id/인덱스)·왜 인지 모아서 한 번에 ScenarioError 로
던진다(첫 에러에서 바로 죽지 않고, 최대한 모아서 사용자가 한 번에 고치게 함).
"""
import json

import numpy as np

from scene import Scene, Rect, Circle
from path import build_path, Path
from agent import Agent, link_dependency
from scenarios import Params


class ScenarioError(Exception):
    """시나리오 JSON 로드/검증 실패. 메시지에 무엇이·어디서·왜 를 담는다."""


DEFAULT_PARAMS = {
    "grid_w": 20,
    "grid_h": 20,
    "tau": 1.5,
    "dt": 0.25,
    "default_radius": 0.28,
    "default_v_max": 1.0,
    "lookahead": 0.5,
    "safety_margin_delta": 0.0,
    "samples_dir": 16,
    "samples_speed": 5,
    "progress_window": 40,
    "progress_eps": 0.05,      # 반지름의 이 비율만큼 window 동안 안 움직이면 정지로 판정
    "tiebreak_enabled": True,
    "max_steps": 4000,
    "slow_radius": 0.5,
    "arrive_tol": 0.08,
    "hold_margin": 0.9,
    "astar_inflate_margin": 0.08,   # A* 는 radius + 이 값만큼 부풀린 grid 위에서 계획
                                     # (좁은 통로라 A*가 경로를 못 찾으면 이 값을 줄인다)
}


# ---------------------------------------------------------------------------
# JSON 파싱 (._comment 무시)
# ---------------------------------------------------------------------------

def _strip_underscore_keys(obj):
    if isinstance(obj, dict):
        return {k: _strip_underscore_keys(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, list):
        return [_strip_underscore_keys(v) for v in obj]
    return obj


def load_raw(path):
    try:
        # utf-8-sig: BOM 있으면 벗기고, 없어도 그냥 utf-8 로 읽는다
        # (Windows 에디터로 JSON 을 저장하면 BOM이 붙는 경우가 흔하다).
        with open(path, "r", encoding="utf-8-sig") as f:
            text = f.read()
    except OSError as e:
        raise ScenarioError(f"파일을 열 수 없음: {path} ({e})")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ScenarioError(
            f"JSON 파싱 실패: {path}:{e.lineno}:{e.colno} — {e.msg}")
    if not isinstance(data, dict):
        raise ScenarioError(f"{path}: 최상위는 object 여야 함")
    return _strip_underscore_keys(data)


def _is_xy(v):
    return (isinstance(v, (list, tuple)) and len(v) == 2 and
            all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in v))


def _raise_if_any(errors, path):
    if errors:
        lines = "\n".join(f"  - {e}" for e in errors)
        raise ScenarioError(f"시나리오 검증 실패: {path}\n{lines}")


# ---------------------------------------------------------------------------
# 검증 + 빌드
# ---------------------------------------------------------------------------

def validate_and_build(data, path="<scenario>"):
    errors = []

    params = dict(DEFAULT_PARAMS)
    params.update(data.get("params", {}) or {})
    grid_w, grid_h = params.get("grid_w"), params.get("grid_h")
    if not (isinstance(grid_w, (int, float)) and grid_w > 0):
        errors.append(f"params.grid_w 는 양수여야 함 (got {grid_w!r})")
        grid_w = DEFAULT_PARAMS["grid_w"]
    if not (isinstance(grid_h, (int, float)) and grid_h > 0):
        errors.append(f"params.grid_h 는 양수여야 함 (got {grid_h!r})")
        grid_h = DEFAULT_PARAMS["grid_h"]

    # ---- obstacles ----
    obstacles_raw = data.get("obstacles", []) or []
    obstacle_ids = set()
    obstacles = []
    for i, o in enumerate(obstacles_raw):
        oid = o.get("id")
        tag = f"obstacles[{i}]" if oid is None else f"obstacles[id={oid}]"
        if not (isinstance(oid, int) and not isinstance(oid, bool) and oid > 0):
            errors.append(f"{tag}: id 는 양의 정수여야 함 (got {oid!r})")
        elif oid in obstacle_ids:
            errors.append(f"{tag}: id 중복")
        else:
            obstacle_ids.add(oid)

        t = o.get("type")
        if t == "rect":
            x, y, w, h = o.get("x"), o.get("y"), o.get("w"), o.get("h")
            if any(v is None or not isinstance(v, (int, float)) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                errors.append(f"{tag}: rect 는 x,y,w,h(w,h>0 실수) 가 필요함")
                continue
            if not (0 <= x and x + w <= grid_w and 0 <= y and y + h <= grid_h):
                errors.append(f"{tag}: rect [{x},{y},{w},{h}] 가 grid({grid_w}x{grid_h}) 범위를 벗어남")
            obstacles.append(Rect(float(x), float(y), float(x + w), float(y + h)))
        elif t == "circle":
            cx, cy, r = o.get("cx"), o.get("cy"), o.get("r")
            if any(v is None or not isinstance(v, (int, float)) for v in (cx, cy, r)) or r <= 0:
                errors.append(f"{tag}: circle 은 cx,cy,r(r>0 실수) 가 필요함")
                continue
            if not (0 <= cx <= grid_w and 0 <= cy <= grid_h):
                errors.append(f"{tag}: circle 중심 [{cx},{cy}] 이 grid({grid_w}x{grid_h}) 범위를 벗어남")
            obstacles.append(Circle(float(cx), float(cy), float(r)))
        else:
            errors.append(f"{tag}: type 은 'rect' 또는 'circle' 이어야 함 (got {t!r})")

    # ---- agents ----
    agents_raw = data.get("agents", []) or []
    if len(agents_raw) == 0:
        errors.append("agents: 최소 1개 필요")
    agent_ids = set()
    specs = []
    for i, a in enumerate(agents_raw):
        aid = a.get("id")
        tag = f"agents[{i}]" if aid is None else f"agents[id={aid}]"
        if not (isinstance(aid, int) and not isinstance(aid, bool) and aid > 0):
            errors.append(f"{tag}: id 는 양의 정수여야 함 (got {aid!r})")
        elif aid in agent_ids:
            errors.append(f"{tag}: id 중복")
        else:
            agent_ids.add(aid)

        start, goal, wp = a.get("start"), a.get("goal"), a.get("waypoint")
        if not _is_xy(start):
            errors.append(f"{tag}: start 는 [x,y] 여야 함 (got {start!r})")
        if not _is_xy(goal):
            errors.append(f"{tag}: goal 은 [x,y] 여야 함 (got {goal!r})")
        if wp is not None and not _is_xy(wp):
            errors.append(f"{tag}: waypoint 는 [x,y] 또는 null 이어야 함 (got {wp!r})")

        route_wps = a.get("route_waypoints", []) or []
        if not isinstance(route_wps, list) or any(not _is_xy(p) for p in route_wps):
            errors.append(f"{tag}: route_waypoints 는 [[x,y],...] 형식이어야 함")
            route_wps = []

        override = a.get("path_override")
        if override is not None:
            if not (isinstance(override, list) and len(override) >= 2 and
                    all(_is_xy(p) for p in override)):
                errors.append(f"{tag}: path_override 는 [[x,y],...] (2점 이상) 또는 null 이어야 함")
                override = None

        radius = a.get("radius", params["default_radius"])
        v_max = a.get("v_max", params["default_v_max"])
        if not (isinstance(radius, (int, float)) and radius > 0):
            errors.append(f"{tag}: radius 는 양수여야 함 (got {radius!r})")
            radius = params["default_radius"]
        if not (isinstance(v_max, (int, float)) and v_max > 0):
            errors.append(f"{tag}: v_max 는 양수여야 함 (got {v_max!r})")
            v_max = params["default_v_max"]

        for label, pt in (("start", start), ("goal", goal), ("waypoint", wp)):
            if _is_xy(pt) and not (0 <= pt[0] <= grid_w and 0 <= pt[1] <= grid_h):
                errors.append(f"{tag}: {label} {pt} 가 grid({grid_w}x{grid_h}) 범위를 벗어남")
        for label, pts in (("route_waypoints", route_wps),):
            for j, pt in enumerate(pts):
                if _is_xy(pt) and not (0 <= pt[0] <= grid_w and 0 <= pt[1] <= grid_h):
                    errors.append(f"{tag}: {label}[{j}] {pt} 가 grid 범위를 벗어남")

        specs.append(dict(id=aid, name=a.get("name", f"agent{aid}"),
                           start=start, goal=goal, waypoint=wp,
                           radius=float(radius), v_max=float(v_max),
                           route_waypoints=route_wps, path_override=override))

    _raise_if_any(errors, path)   # 여기서 끊음 — 이후 A* 등은 유효한 데이터로만 시도

    # ---- scene ----
    scene = Scene(width=float(grid_w), height=float(grid_h), res=0.05, obstacles=obstacles)
    scene.add_boundary_walls()

    # ---- per-agent reference path (§1: 자기 반지름으로 inflate) ----
    agents = []
    id_to_spec = {}
    for spec in specs:
        inflate = spec["radius"] + params["astar_inflate_margin"]
        tag = f"agents[id={spec['id']}]"
        ag = Agent(spec["id"], spec["name"], spec["start"], spec["goal"],
                   radius=spec["radius"], v_max=spec["v_max"],
                   lookahead=params["lookahead"], slow_radius=params["slow_radius"],
                   arrive_tol=params["arrive_tol"], hold_margin=params["hold_margin"])
        try:
            if spec["path_override"] is not None:
                ag.path = Path([np.array(p, float) for p in spec["path_override"]])
            else:
                free = scene.build_grid(inflate)
                anchors = [spec["start"]]
                if spec["waypoint"] is not None:
                    anchors.append(spec["waypoint"])
                anchors += spec["route_waypoints"]
                anchors.append(spec["goal"])
                ag.path = _build_chain(scene, free, anchors, inflate)
        except ValueError:
            errors.append(f"{tag}: start/waypoint/goal 중 하나가 반지름 {spec['radius']:.2f}(+여유)로 "
                           f"inflate된 장애물 안/경계 밖에 있어 A* 시작·끝점을 잡을 수 없음")
            continue
        except RuntimeError:
            errors.append(f"{tag}: 장애물 때문에 경로를 찾을 수 없음(start→waypoint→...→goal 구간 확인)")
            continue
        agents.append(ag)
        id_to_spec[spec["id"]] = spec

    _raise_if_any(errors, path)

    agents_by_id = {a.id: a for a in agents}

    # ---- dependencies ----
    deps_raw = data.get("dependencies", []) or []
    edges = []
    for i, d in enumerate(deps_raw):
        pred, succ = d.get("predecessor"), d.get("successor")
        tag = f"dependencies[{i}]"
        if pred not in agents_by_id:
            errors.append(f"{tag}: predecessor={pred!r} 는 존재하는 agent id 가 아님")
            continue
        if succ not in agents_by_id:
            errors.append(f"{tag}: successor={succ!r} 는 존재하는 agent id 가 아님")
            continue
        if pred == succ:
            errors.append(f"{tag}: predecessor==successor(={pred}) self-dependency 금지")
            continue
        if id_to_spec[succ]["waypoint"] is None:
            errors.append(f"{tag}: successor(id={succ}) 에 dependency 체크포인트로 쓸 "
                           f"waypoint 가 없음 — agents[id={succ}].waypoint 를 채워야 함")
            continue
        edges.append((pred, succ))

    if edges and not errors and _has_cycle(agent_ids, edges):
        errors.append(f"dependencies: 순환(cycle) 이 있어 순서를 정할 수 없음 — {edges}")

    _raise_if_any(errors, path)

    for pred, succ in edges:
        link_dependency(agents_by_id[succ], agents_by_id[pred],
                         id_to_spec[succ]["waypoint"], f"{pred}->{succ}")

    prm = Params(
        dt=float(params["dt"]), v_max=float(params["default_v_max"]),
        radius=float(params["default_radius"]), tau=float(params["tau"]),
        delta=float(params["safety_margin_delta"]), lookahead=float(params["lookahead"]),
        slow_radius=float(params["slow_radius"]), arrive_tol=float(params["arrive_tol"]),
        hold_margin=float(params["hold_margin"]), n_dir=int(params["samples_dir"]),
        n_speed=int(params["samples_speed"]), progress_window=int(params["progress_window"]),
        progress_eps=float(params["progress_eps"]) * float(params["default_radius"]),
        max_steps=int(params["max_steps"]), tie_break=bool(params["tiebreak_enabled"]),
    )
    return scene, agents, prm


def _build_chain(scene, free, anchors, inflate):
    """Start → waypoint → route_waypoints... → Goal 을 이어서 하나의 Path로."""
    pts_all = []
    for i in range(len(anchors) - 1):
        seg = build_path(scene, free, tuple(anchors[i]), tuple(anchors[i + 1]), inflate)
        seg_pts = seg.pts
        if pts_all:
            seg_pts = seg_pts[1:]          # 이어붙일 때 경계점 중복 제거
        pts_all.extend(seg_pts.tolist())
    return Path(pts_all)


def _has_cycle(ids, edges):
    from collections import defaultdict, deque
    indeg = {i: 0 for i in ids}
    adj = defaultdict(list)
    for pred, succ in edges:
        adj[pred].append(succ)
        indeg[succ] = indeg.get(succ, 0) + 1
    q = deque([i for i in ids if indeg.get(i, 0) == 0])
    seen = 0
    while q:
        u = q.popleft(); seen += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return seen != len(ids)


def load_scenario(path):
    data = load_raw(path)
    return validate_and_build(data, path)
