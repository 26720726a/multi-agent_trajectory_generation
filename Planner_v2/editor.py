"""editor.py — scenario.json 을 마우스로 편집하는 인터랙티브 에디터.

    python editor.py --scenario scenarios\\S4.json
    python editor.py                                  # 새 시나리오로 시작

에디터는 scenario.json 의 시각 프론트엔드일 뿐이다 — 장면 표현을 새로 만들지
않고 기존 scene.py/path.py/agent.py/config.py 자료구조를 그대로 읽고 쓴다.
편집 중 화면 갱신은 `config.py`의 엄격한 검증을 통과 못 해도(예: start 가
장애물 안) 최대한 보여주는 '느슨한' 빌드(`Editor._lenient_build`)를 쓰고,
Validate/Run 버튼을 눌렀을 때만 `config.validate_and_build`의 엄격한 검증이
돈다.

matplotlib 백엔드를 강제하지 않는다(마우스 이벤트를 받으려면 인터랙티브
백엔드가 필요) — `viz.py`는 headless 저장을 위해 자기 모듈에서 Agg 를
강제하므로, 이 파일은 `viz.py`를 import 하지 않고 `draw_common.py`(백엔드
비의존)만 공유해서 쓴다. Run 버튼은 시뮬레이터를 이 프로세스 안에서 다시
구현하지 않고 `cli.py`를 **별도 프로세스**로 띄운다.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict, deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RadioButtons, Button, CheckButtons, Slider, TextBox
import matplotlib.widgets as _mpl_widgets

# matplotlib(3.11.1 최신판 포함) 자체 버그 우회: TextBox._resize 가 마우스
# 이벤트 전용 재부모화 데코레이터(_call_with_reparented_event)에 감싸여 있어,
# 창을 리사이즈하면 ResizeEvent(에는 .inaxes 가 없음)가 그 데코레이터를 타고
# 들어가 "AttributeError: 'ResizeEvent' object has no attribute 'inaxes'" 가
# 난다(콜백 예외라 죽지는 않지만 매번 traceback 노이즈가 뜸). 우리 코드가
# resize_event 를 마우스 핸들러와 공유하는 게 아니라 matplotlib 내부 위젯
# 배선 문제라, TextBox._resize 의 원본 함수(= self.stop_typing() 만 함, 애초에
# event.inaxes 를 안 씀)를 데코레이터 없이 직접 바인딩해서 우회한다.
if hasattr(_mpl_widgets.TextBox._resize, "__wrapped__"):
    _mpl_widgets.TextBox._resize = _mpl_widgets.TextBox._resize.__wrapped__

from scene import Scene, Rect, Circle
from agent import Agent, link_dependency
import config
from draw_common import COLORS, draw_obstacle

MODES = ["장애물", "start", "waypoint", "goal", "dependency"]
MIN_SIZE = 0.1
REBUILD_DEBOUNCE_MS = 200   # 슬라이더 조작이 끝나고 이만큼(ms) 잠잠하면 A* 재계산


def _is_xy(v):
    return (isinstance(v, (list, tuple)) and len(v) == 2 and
            all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in v))


def _has_cycle(ids, edges):
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


class Editor:
    def __init__(self, path):
        self.path = path or "scenarios/untitled.json"
        self.data = self._load_or_default(path)

        self.mode = "장애물"
        self.selected_obstacle_id = None
        self.selected_agent_id = None
        self.grid_snap = False
        self.drag = None            # 진행 중인 드래그 상태
        self._dep_first_click = None
        self._handle_artists = []
        self._obstacle_artists = {}
        self._agents = []
        self.placing = None         # {"id": int, "clicks": [...]}, None 이면 배치 모드 아님
        self._placing_artists = []  # 배치 중 미리보기 마커(확정 전)

        self._build_figure()
        self._refresh_agent_radio()
        self._rebuild()

    # =====================================================================
    # 데이터 로드/저장
    # =====================================================================
    def _load_or_default(self, path):
        data = {}
        if path and os.path.exists(path):
            try:
                data = config.load_raw(path)
            except config.ScenarioError as e:
                print(str(e), file=sys.stderr)
                data = {}
        data.setdefault("params", {})
        data.setdefault("obstacles", [])
        data.setdefault("agents", [])
        data.setdefault("dependencies", [])
        merged = dict(config.DEFAULT_PARAMS)
        merged.update(data["params"])
        data["params"] = merged
        return data

    def _exported_data(self):
        return {"params": self.data["params"], "obstacles": self.data["obstacles"],
                "agents": self.data["agents"], "dependencies": self.data["dependencies"]}

    def _current_path(self):
        return self.textbox_path.text.strip()

    def _next_id(self, key):
        ids = [o.get("id") for o in self.data[key] if isinstance(o.get("id"), int)]
        return (max(ids) + 1) if ids else 1

    def _find_agent_spec(self, aid):
        return next((a for a in self.data["agents"] if a.get("id") == aid), None)

    def _find_obstacle_spec(self, oid):
        return next((o for o in self.data["obstacles"] if o.get("id") == oid), None)

    # =====================================================================
    # '느슨한' 빌드 — Validate 만큼 엄격하지 않게, 가능한 만큼만 지어서 편집
    # 중에도 뭔가 보여준다. scene.py/agent.py/config._build_chain 을 그대로 씀.
    # =====================================================================
    def _obs_dict_to_shape(self, o):
        try:
            if o.get("type") == "rect":
                return Rect(float(o["x"]), float(o["y"]),
                            float(o["x"]) + float(o["w"]), float(o["y"]) + float(o["h"]))
            if o.get("type") == "circle":
                return Circle(float(o["cx"]), float(o["cy"]), float(o["r"]))
        except (KeyError, TypeError, ValueError):
            return None
        return None

    def _lenient_build(self):
        params = self.data["params"]
        grid_w = params.get("grid_w") or 20
        grid_h = params.get("grid_h") or 20

        obstacles = []
        for o in self.data["obstacles"]:
            shape = self._obs_dict_to_shape(o)
            if shape is not None:
                obstacles.append(shape)
        scene = Scene(width=float(grid_w), height=float(grid_h), res=0.1, obstacles=obstacles)

        agents = []
        for a in self.data["agents"]:
            aid = a.get("id")
            radius = a.get("radius") or params.get("default_radius", 0.28)
            v_max = a.get("v_max") or params.get("default_v_max", 1.0)
            start, goal, wp = a.get("start"), a.get("goal"), a.get("waypoint")
            ag = Agent(aid, a.get("name", f"agent{aid}"),
                       start if _is_xy(start) else [0.0, 0.0],
                       goal if _is_xy(goal) else [0.0, 0.0],
                       radius=radius, v_max=v_max)
            ag.problem = None
            if not (_is_xy(start) and _is_xy(goal)):
                ag.problem = "start/goal 미지정"
                ag.path = None
                agents.append(ag)
                continue
            try:
                inflate = radius + params.get("astar_inflate_margin", 0.08)
                free = scene.build_grid(inflate)
                anchors = [start]
                if _is_xy(wp):
                    anchors.append(wp)
                for p in (a.get("route_waypoints") or []):
                    if _is_xy(p):
                        anchors.append(p)
                anchors.append(goal)
                ag.path = config._build_chain(scene, free, anchors, inflate)
            except Exception:
                ag.problem = "경로 없음(장애물에 막힘)"
                ag.path = None
            agents.append(ag)

        by_id = {ag.id: ag for ag in agents}
        wp_by_id = {a.get("id"): a.get("waypoint") for a in self.data["agents"]}
        for d in self.data["dependencies"]:
            pred, succ = d.get("predecessor"), d.get("successor")
            if pred in by_id and succ in by_id:
                sa, pa = by_id[succ], by_id[pred]
                wp = wp_by_id.get(succ)
                if _is_xy(wp) and sa.path is not None and pa.path is not None:
                    try:
                        link_dependency(sa, pa, wp, f"{pred}->{succ}")
                    except Exception:
                        pass
        return scene, agents

    def _current_inflate_r(self):
        radii = [a.get("radius") for a in self.data["agents"] if a.get("radius")]
        return max(radii) if radii else self.data["params"].get("default_radius", 0.28)

    # =====================================================================
    # 그림 창 구성
    # =====================================================================
    def _build_figure(self):
        self.fig = plt.figure(figsize=(12.5, 7.5))
        try:
            self.fig.canvas.manager.set_window_title("scenario 에디터")
        except Exception:
            pass
        self.ax = self.fig.add_axes([0.05, 0.08, 0.60, 0.85])
        self.ax.set_aspect("equal")

        ax_path_label = self.fig.add_axes([0.05, 0.955, 0.60, 0.035]); ax_path_label.axis("off")
        self.textbox_path = TextBox(self.fig.add_axes([0.13, 0.955, 0.52, 0.035]),
                                     "파일 ", initial=self.path)

        ax_mode = self.fig.add_axes([0.69, 0.74, 0.29, 0.20])
        ax_mode.set_title("모드", fontsize=9)
        self.radio_mode = RadioButtons(ax_mode, MODES)
        self.radio_mode.on_clicked(self._on_mode_change)

        self.ax_agent = self.fig.add_axes([0.69, 0.52, 0.29, 0.18])
        self.radio_agent = None

        ax_snap = self.fig.add_axes([0.69, 0.465, 0.29, 0.045])
        self.check_snap = CheckButtons(ax_snap, ["grid-snap"], [self.grid_snap])
        self.check_snap.on_clicked(self._on_snap_toggle)

        ax_w = self.fig.add_axes([0.72, 0.40, 0.24, 0.025])
        ax_h = self.fig.add_axes([0.72, 0.36, 0.24, 0.025])
        self.slider_w = Slider(ax_w, "w/r", MIN_SIZE, 20.0, valinit=1.0)
        self.slider_h = Slider(ax_h, "h", MIN_SIZE, 20.0, valinit=1.0)
        # 콜백 id 를 보관해뒀다가 프로그램이 값을 세팅할 때 잠깐 끊는다
        # (아래 _set_slider_silently) — bool 플래그로 "지금은 무시해" 하는 방식은
        # set_val 이 유발하는 on_changed 가 지연 실행되면(TkAgg 에서 실제로
        # 그럴 수 있음) 플래그가 이미 풀린 뒤 실행돼 무력화된다. disconnect 로
        # 물리적으로 끊어두면 타이밍과 무관하게 절대 안 불린다.
        self._cid_slider_w = self.slider_w.on_changed(self._on_slider_w)
        self._cid_slider_h = self.slider_h.on_changed(self._on_slider_h)
        ax_w.set_visible(False); ax_h.set_visible(False)

        # 슬라이더/드래그 중엔 A*(=_rebuild) 를 매 tick 마다 돌리지 않고, 마지막
        # 변경으로부터 REBUILD_DEBOUNCE_MS 동안 잠잠하면 한 번만 돈다. 안 그러면
        # 빠른 드래그 한 번에 A* 재계산 + 전체 redraw 가 수십 번씩 쌓여 Tk
        # 이벤트 루프가 밀리면서 창이 "응답 없음"이 된다(관측된 버그).
        self._rebuild_timer = self.fig.canvas.new_timer(interval=REBUILD_DEBOUNCE_MS)
        self._rebuild_timer.single_shot = True
        self._rebuild_timer.add_callback(self._debounced_rebuild)

        specs = [
            ("에이전트 추가", self._on_add_agent),
            ("에이전트 삭제", self._on_del_agent),
            ("장애물 추가", self._on_add_obstacle),
            ("장애물 삭제", self._on_del_obstacle),
            ("Validate", self._on_validate),
            ("Save", self._on_save),
            ("Load", self._on_load),
            ("Run", self._on_run),
        ]
        self.buttons = []
        y = 0.315
        for label, cb in specs:
            bax = self.fig.add_axes([0.69, y, 0.29, 0.035])
            b = Button(bax, label)
            b.on_clicked(cb)
            self.buttons.append(b)
            y -= 0.043

        self.status_text = self.fig.text(0.05, 0.015, "", fontsize=8.5, color="#333333",
                                          va="bottom", ha="left")
        self.help_text = self.fig.text(
            0.69, y - 0.01,
            "장애물: 빈곳 드래그=생성, 몸통 드래그=이동,\n"
            "핸들 드래그=크기조절, Delete=삭제.\n"
            "start/waypoint/goal: 캔버스 클릭으로 배치.\n"
            "dependency: 에이전트 두 번 클릭\n(predecessor→successor).\n"
            "'에이전트 추가': 캔버스 3클릭으로\n"
            "start→waypoint→goal 순서 배치(Esc=취소).",
            fontsize=7.3, va="top", ha="left", color="#555555")

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _refresh_agent_radio(self):
        self.ax_agent.clear()
        self.ax_agent.set_title("현재 에이전트", fontsize=9)
        ids = [a.get("id") for a in self.data["agents"]]
        if not ids:
            self.ax_agent.text(0.05, 0.5, "(없음 — '에이전트 추가')", fontsize=7.5)
            self.radio_agent = None
            self.selected_agent_id = None
            self.fig.canvas.draw_idle()
            return
        if self.selected_agent_id not in ids:
            self.selected_agent_id = ids[0]
        labels = [f"{a.get('id')}:{a.get('name', '')}" for a in self.data["agents"]]
        active = ids.index(self.selected_agent_id)
        self.radio_agent = RadioButtons(self.ax_agent, labels, active=active)
        self.radio_agent.on_clicked(self._on_agent_radio)
        self.fig.canvas.draw_idle()

    def _on_agent_radio(self, label):
        self.selected_agent_id = int(label.split(":")[0])
        self._rebuild()

    # =====================================================================
    # 전체 다시 그리기 (A* 재계산 포함 — release/버튼 액션에서만 호출)
    # =====================================================================
    def _rebuild(self):
        scene, agents = self._lenient_build()
        self._scene = scene
        self._agents = agents

        self.ax.clear()
        self._handle_artists = []
        gw, gh = scene.width, scene.height
        pad = max(0.5, 0.05 * max(gw, gh))
        self.ax.set_xlim(-pad, gw + pad)
        self.ax.set_ylim(-pad, gh + pad)
        self.ax.set_aspect("equal")
        self.ax.set_title(os.path.basename(self._current_path()) if hasattr(self, "textbox_path") else "")
        self.ax.add_patch(Rectangle((0, 0), gw, gh, fill=False, edgecolor="#bbbbbb", linewidth=1, zorder=0))

        inflate_r = self._current_inflate_r()
        self._obstacle_artists = {}
        for o in self.data["obstacles"]:
            shape = self._obs_dict_to_shape(o)
            if shape is None:
                continue
            oid = o.get("id")
            selected = (oid == self.selected_obstacle_id)
            color = "#c0392b" if selected else "#444444"
            body, dashed = draw_obstacle(self.ax, shape, color=color, dashed_pad=inflate_r)
            self._obstacle_artists[oid] = {"patch": body, "dashed": dashed, "shape": shape}
            if selected:
                self._draw_handles(shape)

        agents_by_id = {a.id: a for a in agents}
        for i, a in enumerate(self.data["agents"]):
            aid = a.get("id")
            c = COLORS[i % len(COLORS)]
            ag = agents_by_id.get(aid)
            selected = (aid == self.selected_agent_id)
            invalid = ag is None or ag.problem is not None
            if ag is not None and ag.path is not None:
                self.ax.plot(ag.path.pts[:, 0], ag.path.pts[:, 1], "-", color=c,
                             linewidth=2.2 if selected else 1.4, alpha=0.9, zorder=2)
                for h in ag.hold:
                    wp_pt = ag.path.pos(h.own_s)
                    self.ax.plot(*wp_pt, marker="x", color=c, markersize=9, zorder=4)
            start, goal, wp = a.get("start"), a.get("goal"), a.get("waypoint")
            for pt, marker in ((start, "o"), (goal, "^"), (wp, "D")):
                if not _is_xy(pt):
                    continue
                mcolor = "#e74c3c" if invalid else c
                self.ax.plot(*pt, marker=marker, color=mcolor,
                             markersize=12 if selected else 9, zorder=5,
                             markeredgecolor="black", markeredgewidth=1.6 if selected else 0.7)
            label_pt = start if _is_xy(start) else (goal if _is_xy(goal) else None)
            if label_pt is not None:
                self.ax.annotate(f"{aid}:{a.get('name', '')}", label_pt, textcoords="offset points",
                                  xytext=(6, 6), fontsize=8, color=c, weight="bold")

        self._update_status()
        self.fig.canvas.draw_idle()

    def _update_status(self):
        problems = [f"{a.id}:{a.problem}" for a in self._agents if a.problem]
        base = (f"모드: {self.mode}   장애물 {len(self.data['obstacles'])}개   "
                f"에이전트 {len(self.data['agents'])}개   grid-snap: {self.grid_snap}")
        if problems:
            base += "   ⚠ " + ", ".join(problems)
        self.status_text.set_text(base)

    def _set_status(self, msg):
        self.status_text.set_text(msg)
        self.fig.canvas.draw_idle()

    # =====================================================================
    # 장애물 핸들(코너/에지) — (name, x, y, moves_x0, moves_x1, moves_y0, moves_y1)
    # =====================================================================
    def _handle_points(self, shape):
        if isinstance(shape, Rect):
            x0, y0, x1, y1 = shape.x0, shape.y0, shape.x1, shape.y1
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            return [
                ("ll", x0, y0, True, False, True, False),
                ("lr", x1, y0, False, True, True, False),
                ("ul", x0, y1, True, False, False, True),
                ("ur", x1, y1, False, True, False, True),
                ("b", mx, y0, False, False, True, False),
                ("t", mx, y1, False, False, False, True),
                ("l", x0, my, True, False, False, False),
                ("r", x1, my, False, True, False, False),
            ]
        return [("r", shape.cx + shape.r, shape.cy, False, False, False, False)]

    def _draw_handles(self, shape):
        self._handle_artists = []
        for _name, x, y, *_ in self._handle_points(shape):
            ln, = self.ax.plot(x, y, marker="s", color="white",
                                markeredgecolor="#c0392b", markersize=7, zorder=6)
            self._handle_artists.append(ln)

    def _resize_shape(self, shape, handle_name, x, y):
        if isinstance(shape, Rect):
            x0, y0, x1, y1 = shape.x0, shape.y0, shape.x1, shape.y1
            for name, _hx, _hy, mx0, mx1, my0, my1 in self._handle_points(shape):
                if name == handle_name:
                    if mx0: x0 = min(x, x1 - MIN_SIZE)
                    if mx1: x1 = max(x, x0 + MIN_SIZE)
                    if my0: y0 = min(y, y1 - MIN_SIZE)
                    if my1: y1 = max(y, y0 + MIN_SIZE)
                    break
            return Rect(x0, y0, x1, y1)
        r = max(MIN_SIZE, float(np.hypot(x - shape.cx, y - shape.cy)))
        return Circle(shape.cx, shape.cy, r)

    def _snap_shape(self, shape):
        if not self.grid_snap:
            return shape
        if isinstance(shape, Rect):
            x0, y0 = round(shape.x0), round(shape.y0)
            x1, y1 = round(shape.x1), round(shape.y1)
            if x1 <= x0: x1 = x0 + 1
            if y1 <= y0: y1 = y0 + 1
            return Rect(float(x0), float(y0), float(x1), float(y1))
        return Circle(float(round(shape.cx)), float(round(shape.cy)),
                      max(0.5, round(shape.r * 2) / 2))

    def _snap_point(self, x, y):
        if self.grid_snap:
            return (float(round(x)), float(round(y)))
        return (x, y)

    def _tol(self):
        gw = self.data["params"].get("grid_w") or 20
        gh = self.data["params"].get("grid_h") or 20
        return 0.035 * max(gw, gh)

    def _hit_test(self, x, y):
        tol = self._tol()
        if self.selected_obstacle_id is not None:
            art = self._obstacle_artists.get(self.selected_obstacle_id)
            if art is not None:
                for name, hx, hy, *_ in self._handle_points(art["shape"]):
                    if abs(hx - x) <= tol and abs(hy - y) <= tol:
                        return ("handle", self.selected_obstacle_id, name)
        for o in self.data["obstacles"]:
            shape = self._obs_dict_to_shape(o)
            if shape is not None and shape.contains(x, y):
                return ("body", o.get("id"), None)
        return ("empty", None, None)

    def _push_obstacle_shape(self, oid, shape):
        """드래그 중 빠른 경로: A* 없이 data + patch geometry 만 갱신."""
        o = self._find_obstacle_spec(oid)
        if o is None:
            return
        if isinstance(shape, Rect):
            o["type"] = "rect"
            o["x"], o["y"] = shape.x0, shape.y0
            o["w"], o["h"] = shape.x1 - shape.x0, shape.y1 - shape.y0
            for k in ("cx", "cy", "r"):
                o.pop(k, None)
        else:
            o["type"] = "circle"
            o["cx"], o["cy"], o["r"] = shape.cx, shape.cy, shape.r
            for k in ("x", "y", "w", "h"):
                o.pop(k, None)

        art = self._obstacle_artists.get(oid)
        if art is not None:
            art["shape"] = shape
            inflate_r = self._current_inflate_r()
            if isinstance(shape, Rect):
                art["patch"].set_xy((shape.x0, shape.y0))
                art["patch"].set_width(shape.x1 - shape.x0)
                art["patch"].set_height(shape.y1 - shape.y0)
                if art["dashed"] is not None:
                    art["dashed"].set_xy((shape.x0 - inflate_r, shape.y0 - inflate_r))
                    art["dashed"].set_width((shape.x1 - shape.x0) + 2 * inflate_r)
                    art["dashed"].set_height((shape.y1 - shape.y0) + 2 * inflate_r)
            else:
                art["patch"].center = (shape.cx, shape.cy)
                art["patch"].set_radius(shape.r)
                if art["dashed"] is not None:
                    art["dashed"].center = (shape.cx, shape.cy)
                    art["dashed"].set_radius(shape.r + inflate_r)
            if oid == self.selected_obstacle_id:
                pts = self._handle_points(shape)
                for ln, (name, hx, hy, *_r) in zip(self._handle_artists, pts):
                    ln.set_data([hx], [hy])
        self.fig.canvas.draw_idle()

    def _set_slider_silently(self, slider, cid_attr, callback, val):
        """프로그램이 슬라이더 값을 세팅할 때 쓴다 — on_changed 콜백을 물리적으로
        끊었다가 세팅 후 다시 연결하므로, 콜백이 언제 실행되든(동기/지연 상관없이)
        이 세팅 때문에 _on_slider_w/h 가 불릴 일이 없다."""
        slider.disconnect(getattr(self, cid_attr))
        slider.set_val(val)
        setattr(self, cid_attr, slider.on_changed(callback))

    def _refresh_obstacle_sliders(self):
        oid = self.selected_obstacle_id
        art = self._obstacle_artists.get(oid) if oid is not None else None
        if art is None:
            self.slider_w.ax.set_visible(False)
            self.slider_h.ax.set_visible(False)
        else:
            shape = art["shape"]
            self.slider_w.ax.set_visible(True)
            if isinstance(shape, Rect):
                self.slider_h.ax.set_visible(True)
                self.slider_w.label.set_text("w")
                self.slider_h.label.set_text("h")
                self._set_slider_silently(
                    self.slider_w, "_cid_slider_w", self._on_slider_w,
                    min(max(shape.x1 - shape.x0, MIN_SIZE), self.slider_w.valmax))
                self._set_slider_silently(
                    self.slider_h, "_cid_slider_h", self._on_slider_h,
                    min(max(shape.y1 - shape.y0, MIN_SIZE), self.slider_h.valmax))
            else:
                self.slider_h.ax.set_visible(False)
                self.slider_w.label.set_text("r")
                self._set_slider_silently(
                    self.slider_w, "_cid_slider_w", self._on_slider_w,
                    min(max(shape.r, MIN_SIZE), self.slider_w.valmax))
        self.fig.canvas.draw_idle()

    def _schedule_rebuild(self):
        """무거운 _rebuild(A* 재계산 + 전체 redraw)를 디바운스한다 — 마지막
        변경으로부터 REBUILD_DEBOUNCE_MS 동안 추가 변경이 없을 때 한 번만
        돈다. 슬라이더를 빠르게 왕복해도 tick 마다 A*가 안 돌게 하는 핵심."""
        self._rebuild_timer.stop()
        self._rebuild_timer.start()

    def _debounced_rebuild(self):
        self._rebuild_timer.stop()
        self._rebuild()

    def _on_slider_w(self, val):
        if self.selected_obstacle_id is None:
            return
        oid = self.selected_obstacle_id
        shape = self._obstacle_artists[oid]["shape"]
        new_shape = (Rect(shape.x0, shape.y0, shape.x0 + val, shape.y1)
                     if isinstance(shape, Rect) else Circle(shape.cx, shape.cy, val))
        self._push_obstacle_shape(oid, new_shape)   # 가벼움: geometry만 갱신
        self._schedule_rebuild()                     # 무거운 A* 는 디바운스

    def _on_slider_h(self, val):
        if self.selected_obstacle_id is None:
            return
        oid = self.selected_obstacle_id
        shape = self._obstacle_artists[oid]["shape"]
        if isinstance(shape, Rect):
            self._push_obstacle_shape(oid, Rect(shape.x0, shape.y0, shape.x1, shape.y0 + val))
            self._schedule_rebuild()

    # =====================================================================
    # 마우스 이벤트
    # =====================================================================
    def _clear_placing_artists(self):
        for ln in self._placing_artists:
            try:
                ln.remove()
            except Exception:
                pass
        self._placing_artists = []

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        x, y = event.xdata, event.ydata

        if self.placing is not None:
            # 에이전트 3-클릭 배치 모드 — 기존 self.mode 별 처리보다 먼저
            # 소비한다(클릭 핸들러를 새로 붙이지 않고 여기 한 곳에서 처리).
            if event.button != 1:
                return
            pt = self._snap_point(x, y)
            self.placing["clicks"].append(pt)
            n = len(self.placing["clicks"])
            aid = self.placing["id"]
            marker = ("o", "D", "^")[n - 1]
            ln, = self.ax.plot(pt[0], pt[1], marker=marker, color="#7f8c8d",
                                markersize=10, zorder=7, markeredgecolor="black")
            self._placing_artists.append(ln)
            if n == 1:
                self._set_status(f"에이전트 a{aid} 배치 — 캔버스를 클릭하세요: 2/3 (waypoint). Esc로 취소.")
                self.fig.canvas.draw_idle()
            elif n == 2:
                self._set_status(f"에이전트 a{aid} 배치 — 캔버스를 클릭하세요: 3/3 (goal). Esc로 취소.")
                self.fig.canvas.draw_idle()
            else:
                s, w, g = self.placing["clicks"]
                self.data["agents"].append({
                    "id": aid, "name": f"a{aid}",
                    "start": [s[0], s[1]], "waypoint": [w[0], w[1]], "goal": [g[0], g[1]],
                    "route_waypoints": [], "path_override": None,
                })
                self.selected_agent_id = aid
                self.placing = None
                self._clear_placing_artists()
                self._refresh_agent_radio()
                self._rebuild()
                self._set_status(f"에이전트 a{aid} 추가됨 (start/waypoint/goal 지정 완료).")
            return

        if self.mode == "장애물":
            kind, oid, extra = self._hit_test(x, y)
            if kind == "handle":
                self.drag = {"kind": "resize", "id": oid, "handle": extra}
            elif kind == "body":
                self.selected_obstacle_id = oid
                self._refresh_obstacle_sliders()
                self._rebuild()
                self.drag = {"kind": "move", "id": oid, "anchor": (x, y)}
            else:
                self.selected_obstacle_id = None
                self._refresh_obstacle_sliders()
                self._rebuild()
                self.drag = {"kind": "new_rect", "start": (x, y), "patch": None}

        elif self.mode in ("start", "waypoint", "goal"):
            if self.selected_agent_id is None:
                self._set_status("배치할 에이전트가 없음 — 먼저 '에이전트 추가'.")
                return
            a = self._find_agent_spec(self.selected_agent_id)
            if a is None:
                return
            pt = self._snap_point(x, y)
            a[self.mode] = [pt[0], pt[1]]
            self._rebuild()

        elif self.mode == "dependency":
            aid = self._agent_at_point(x, y)
            if aid is None:
                self._set_status("에이전트의 start/goal/waypoint 근처를 클릭하세요.")
                return
            if self._dep_first_click is None:
                self._dep_first_click = aid
                self._set_status(f"predecessor = agent {aid} 선택됨 — successor 를 클릭하세요.")
            else:
                pred, succ = self._dep_first_click, aid
                self._dep_first_click = None
                self._add_dependency(pred, succ)

    def _on_motion(self, event):
        if self.drag is None or event.inaxes != self.ax or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        kind = self.drag["kind"]

        if kind == "move":
            oid = self.drag["id"]
            art = self._obstacle_artists.get(oid)
            if art is None:
                return
            shape = art["shape"]
            ax0, ay0 = self.drag["anchor"]
            dx, dy = x - ax0, y - ay0
            self.drag["anchor"] = (x, y)
            if isinstance(shape, Rect):
                new_shape = Rect(shape.x0 + dx, shape.y0 + dy, shape.x1 + dx, shape.y1 + dy)
            else:
                new_shape = Circle(shape.cx + dx, shape.cy + dy, shape.r)
            self._push_obstacle_shape(oid, new_shape)

        elif kind == "resize":
            oid, handle = self.drag["id"], self.drag["handle"]
            art = self._obstacle_artists.get(oid)
            if art is None:
                return
            new_shape = self._resize_shape(art["shape"], handle, x, y)
            self._push_obstacle_shape(oid, new_shape)

        elif kind == "new_rect":
            sx, sy = self.drag["start"]
            x0, x1 = sorted((sx, x)); y0, y1 = sorted((sy, y))
            x1, y1 = max(x0 + 0.02, x1), max(y0 + 0.02, y1)
            if self.drag["patch"] is None:
                patch, _ = draw_obstacle(self.ax, Rect(x0, y0, x1, y1), color="#7f8c8d")
                patch.set_alpha(0.5)
                self.drag["patch"] = patch
            else:
                self.drag["patch"].set_xy((x0, y0))
                self.drag["patch"].set_width(x1 - x0)
                self.drag["patch"].set_height(y1 - y0)
            self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self.drag is None:
            return
        kind = self.drag["kind"]
        if kind == "new_rect":
            patch = self.drag["patch"]
            if patch is not None:
                w, h = patch.get_width(), patch.get_height()
                x0, y0 = patch.get_xy()
                patch.remove()
                if w > 0.05 and h > 0.05:
                    shape = self._snap_shape(Rect(x0, y0, x0 + w, y0 + h))
                    oid = self._next_id("obstacles")
                    self.data["obstacles"].append({
                        "id": oid, "type": "rect", "x": shape.x0, "y": shape.y0,
                        "w": shape.x1 - shape.x0, "h": shape.y1 - shape.y0})
                    self.selected_obstacle_id = oid
        elif kind in ("move", "resize"):
            oid = self.drag["id"]
            art = self._obstacle_artists.get(oid)
            if art is not None:
                snapped = self._snap_shape(art["shape"])
                self._push_obstacle_shape(oid, snapped)
        self.drag = None
        self._refresh_obstacle_sliders()
        self._rebuild()

    def _on_key(self, event):
        if event.key == "escape" and self.placing is not None:
            aid = self.placing["id"]
            self.placing = None
            self._clear_placing_artists()
            self.fig.canvas.draw_idle()
            self._set_status(f"에이전트 a{aid} 배치 취소됨.")
            return
        if event.key in ("delete", "backspace"):
            if self.mode == "장애물" and self.selected_obstacle_id is not None:
                self._delete_obstacle(self.selected_obstacle_id)

    def _agent_at_point(self, x, y):
        tol = self._tol()
        best, best_d = None, tol
        for a in self.data["agents"]:
            for pt in (a.get("start"), a.get("goal"), a.get("waypoint")):
                if _is_xy(pt):
                    d = float(np.hypot(pt[0] - x, pt[1] - y))
                    if d < best_d:
                        best_d, best = d, a.get("id")
        return best

    # =====================================================================
    # 버튼 콜백
    # =====================================================================
    def _delete_obstacle(self, oid):
        self.data["obstacles"] = [o for o in self.data["obstacles"] if o.get("id") != oid]
        if self.selected_obstacle_id == oid:
            self.selected_obstacle_id = None
        self._refresh_obstacle_sliders()
        self._rebuild()

    def _on_add_agent(self, _event):
        # 즉시 기본 위치로 만들지 않고 3-클릭 배치 모드로 들어간다:
        # 캔버스 클릭 1번째=start, 2번째=waypoint, 3번째=goal 로 확정.
        # _on_press 맨 위의 self.placing 분기가 이후 클릭 3번을 소비한다.
        aid = self._next_id("agents")
        self.placing = {"id": aid, "clicks": []}
        self._clear_placing_artists()
        self._set_status(f"에이전트 a{aid} 배치 — 캔버스를 클릭하세요: 1/3 (start). Esc로 취소.")

    def _on_del_agent(self, _event):
        aid = self.selected_agent_id
        if aid is None:
            self._set_status("삭제할 에이전트가 없음.")
            return
        self.data["agents"] = [a for a in self.data["agents"] if a.get("id") != aid]
        self.data["dependencies"] = [d for d in self.data["dependencies"]
                                      if d.get("predecessor") != aid and d.get("successor") != aid]
        self.selected_agent_id = None
        self._refresh_agent_radio()
        self._rebuild()
        self._set_status(f"에이전트 {aid} 삭제됨.")

    def _on_add_obstacle(self, _event):
        oid = self._next_id("obstacles")
        gw = self.data["params"].get("grid_w") or 20
        gh = self.data["params"].get("grid_h") or 20
        cx, cy = gw / 2, gh / 2
        self.data["obstacles"].append({"id": oid, "type": "rect",
                                        "x": cx - 0.5, "y": cy - 0.5, "w": 1.0, "h": 1.0})
        self.selected_obstacle_id = oid
        self._refresh_obstacle_sliders()
        self._rebuild()
        self._set_status(f"장애물 {oid} 추가됨 — 드래그로 옮기거나 크기를 조절하세요.")

    def _on_del_obstacle(self, _event):
        if self.selected_obstacle_id is None:
            self._set_status("삭제할 장애물을 먼저 선택하세요.")
            return
        self._delete_obstacle(self.selected_obstacle_id)

    def _add_dependency(self, pred, succ):
        if pred == succ:
            self._set_status("predecessor == successor — self-dependency 는 안 됨.")
            return
        existing = [(d["predecessor"], d["successor"]) for d in self.data["dependencies"]]
        if (pred, succ) in existing:
            self._set_status(f"이미 있는 dependency: {pred} -> {succ}")
            return
        ids = [a.get("id") for a in self.data["agents"]]
        if _has_cycle(ids, existing + [(pred, succ)]):
            self._set_status(f"거부됨: {pred} -> {succ} 를 추가하면 순환(cycle)이 생김.")
            return
        self.data["dependencies"].append({"predecessor": pred, "successor": succ})
        succ_spec = self._find_agent_spec(succ)
        msg = f"dependency 추가: {pred} -> {succ}"
        if succ_spec is not None and not _is_xy(succ_spec.get("waypoint")):
            msg += f"  ⚠ agent {succ} 에 waypoint 가 없음 — waypoint 모드로 지정할 것"
        self._rebuild()
        self._set_status(msg)

    def _on_mode_change(self, label):
        self.mode = label
        self._dep_first_click = None
        self._set_status(f"모드: {label}")

    def _on_snap_toggle(self, _label):
        self.grid_snap = not self.grid_snap
        self._set_status(f"grid-snap: {self.grid_snap}")

    # ---- Save / Load / Validate / Run ----
    def _on_save(self, _event):
        path = self._current_path()
        if not path:
            self._set_status("저장 경로가 비어 있음.")
            return
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._exported_data(), f, ensure_ascii=False, indent=2)
        self.path = path
        self._set_status(f"저장됨: {path}")

    def _on_load(self, _event):
        path = self._current_path()
        self.data = self._load_or_default(path)
        self.path = path
        self.selected_obstacle_id = None
        self.selected_agent_id = None
        self._refresh_obstacle_sliders()
        self._refresh_agent_radio()
        self._rebuild()
        self._set_status(f"불러옴: {path}")

    def _on_validate(self, _event):
        try:
            config.validate_and_build(self._exported_data(), path=self._current_path() or "<editor>")
        except config.ScenarioError as e:
            print(str(e))
            self._set_status(str(e).replace("\n", "  |  "))
            return
        self._set_status("검증 통과 — 문제 없음.")

    def _on_run(self, _event):
        self._on_save(None)
        try:
            config.validate_and_build(self._exported_data(), path=self._current_path())
        except config.ScenarioError as e:
            print(str(e))
            self._set_status("검증 실패라 실행 안 함 — 콘솔 출력 참고.")
            return
        path = self._current_path()
        here = os.path.dirname(os.path.abspath(__file__))
        subprocess.Popen([sys.executable, os.path.join(here, "cli.py"), "--scenario", path, "--run"])
        self._set_status(f"백그라운드에서 실행 시작: {path} (out/ 폴더에 결과 저장됨)")


def main():
    ap = argparse.ArgumentParser(description="scenario.json 마우스 편집기")
    ap.add_argument("--scenario", default=None, help="열 시나리오 JSON 경로(없으면 새로 시작)")
    args = ap.parse_args()
    Editor(args.scenario)
    plt.show()


if __name__ == "__main__":
    main()
