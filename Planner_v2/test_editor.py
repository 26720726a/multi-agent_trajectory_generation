"""test_editor.py — 마우스 없이(Agg 백엔드) editor.py 의 로직 계층을 직접
호출해서 검증하는 회귀 테스트. `python test_editor.py`로 실행.

실제 마우스 인터랙션(드래그로 장애물 크기 조절 등)은 디스플레이가 있는
로컬 데스크톱에서 육안으로 확인해야 하지만, 데이터 모델·A* 재계산·
저장→로드 왕복·dependency 순환 거부는 이걸로 충분히 확인 가능하다.
editor.py 를 고칠 때마다 이 스크립트로 로직이 안 깨졌는지 먼저 확인할 것.
"""
import matplotlib
matplotlib.use("Agg")   # 이 프로세스엔 디스플레이가 없음 — 위젯 생성만 확인

import json
import os

from editor import Editor, _has_cycle
from scene import Rect, Circle
import config

OUT = "scenarios/_editor_test.json"


class FakeEvent:
    """button_press_event 를 흉내낸다 — Editor._on_press 가 쓰는 속성만 채움."""
    def __init__(self, ax, x, y, button=1):
        self.inaxes = ax
        self.xdata = x
        self.ydata = y
        self.button = button


class FakeKeyEvent:
    def __init__(self, key):
        self.key = key

# 1) 빈 시나리오로 시작
ed = Editor(None)
assert ed.data["agents"] == []
print("[1] 빈 시나리오 시작 OK")

# 0) ResizeEvent 버그 우회 확인 — 창 리사이즈시 TextBox._resize 가 ResizeEvent
# (.inaxes 없음)를 받아도 AttributeError 없이 넘어가야 한다.
from matplotlib.backend_bases import ResizeEvent
ev = ResizeEvent("resize_event", ed.fig.canvas)
ed.fig.canvas.callbacks.process("resize_event", ev)   # 예외 나면 아래 assert 전에 여기서 죽음
print("[0] resize_event 처리 시 AttributeError 없음(ResizeEvent 버그 우회) OK")

# 2) 장애물 추가 + 슬라이더로 크기 조절 + 드래그(직접 호출)로 이동
ed._on_add_obstacle(None)
oid = ed.selected_obstacle_id
assert oid is not None
ed.slider_w.set_val(3.0)     # -> _on_slider_w 콜백 실행
ed.slider_h.set_val(2.0)
o = ed._find_obstacle_spec(oid)
assert abs(o["w"] - 3.0) < 1e-6 and abs(o["h"] - 2.0) < 1e-6, o
print("[2] 장애물 추가+슬라이더 리사이즈 OK:", o)

# 2b~2d) GUI 멈춤 버그 회귀 확인: (1) 프로그램이 set_val 로 세팅한 값은
# _on_slider_w/h 를 건드리면 안 되고, (2) 사용자가 슬라이더를 연속으로
# 조작해도 무거운 _rebuild(A*)가 매번 즉시 불리면 안 되며(디바운스 대기),
# (3) 디바운스가 만료됐을 때만 정확히 한 번 불려야 한다.
_slider_cb_calls = {"n": 0}
_orig_on_slider_w = ed._on_slider_w


def _counting_on_slider_w(val):
    _slider_cb_calls["n"] += 1
    return _orig_on_slider_w(val)


ed._on_slider_w = _counting_on_slider_w
ed.slider_w.disconnect(ed._cid_slider_w)
ed._cid_slider_w = ed.slider_w.on_changed(ed._on_slider_w)

before = _slider_cb_calls["n"]
ed._refresh_obstacle_sliders()      # 프로그램이 세팅 -> 사용자 콜백이 불리면 안 됨
assert _slider_cb_calls["n"] == before, "프로그램 set_val 이 _on_slider_w 를 건드림(가드 실패)"
print("[2b] 프로그램 set_val(_refresh_obstacle_sliders) 은 _on_slider_w 를 안 건드림 OK")

rebuild_calls = {"n": 0}
_orig_rebuild = ed._rebuild


def _counting_rebuild():
    rebuild_calls["n"] += 1
    return _orig_rebuild()


ed._rebuild = _counting_rebuild
for v in (1.5, 1.7, 1.9, 2.1, 2.3):   # 빠르게 왕복 드래그하는 상황 흉내
    ed.slider_w.set_val(v)
assert rebuild_calls["n"] == 0, "슬라이더 연속 조작 중에 _rebuild 가 즉시 불림(디바운스 안 됨) — GUI 멈춤 버그 재발"
print("[2c] 슬라이더 연속 조작 중엔 _rebuild 가 즉시 안 불림(디바운스 대기) OK")

ed._debounced_rebuild()   # 디바운스 타이머가 실제로 만료됐다고 가정하고 콜백 실행
assert rebuild_calls["n"] == 1, f"디바운스 만료 후 _rebuild 가 정확히 1회가 아님(실제 {rebuild_calls['n']}회)"
print("[2d] 디바운스 만료 후 _rebuild 정확히 1회 실행 OK")

ed._rebuild = _orig_rebuild
ed._on_slider_w = _orig_on_slider_w
ed.slider_w.disconnect(ed._cid_slider_w)
ed._cid_slider_w = ed.slider_w.on_changed(ed._on_slider_w)

# 몸통 이동 시뮬레이션 (press->motion->release 대신 내부 메서드 직접 호출)
shape = ed._obstacle_artists[oid]["shape"]
moved = Rect(shape.x0 + 1.0, shape.y0 + 1.0, shape.x1 + 1.0, shape.y1 + 1.0)
ed._push_obstacle_shape(oid, moved)
ed._rebuild()
o = ed._find_obstacle_spec(oid)
assert abs(o["x"] - (shape.x0 + 1.0)) < 1e-6
print("[3] 장애물 이동 OK:", o)

# resize 핸들 시뮬레이션: 오른쪽 에지 핸들을 오른쪽으로 드래그
shape = ed._obstacle_artists[oid]["shape"]
new_shape = ed._resize_shape(shape, "r", shape.x1 + 2.0, (shape.y0 + shape.y1) / 2)
assert abs(new_shape.x1 - (shape.x1 + 2.0)) < 1e-6
ed._push_obstacle_shape(oid, new_shape)
ed._rebuild()
print("[4] 핸들 리사이즈(r) OK:", ed._find_obstacle_spec(oid))

# circle 장애물도 확인
ed.data["obstacles"].append({"id": ed._next_id("obstacles"), "type": "circle",
                              "cx": 5.0, "cy": 5.0, "r": 1.0})
ed._rebuild()
cid = ed.data["obstacles"][-1]["id"]
ed.selected_obstacle_id = cid
ed._refresh_obstacle_sliders()
cshape = ed._obstacle_artists[cid]["shape"]
assert isinstance(cshape, Circle)
new_c = ed._resize_shape(cshape, "r", cshape.cx + 2.0, cshape.cy)
assert abs(new_c.r - 2.0) < 1e-6
print("[5] circle 장애물 + 반지름 리사이즈 OK")

# 3) 에이전트 2개를 "에이전트 추가" 버튼 + 캔버스 3-클릭(start→waypoint→goal)으로 배치
ed.data["params"]["grid_w"] = 10
ed.data["params"]["grid_h"] = 10
ed.data["obstacles"] = []      # 장애물 없이 깔끔하게


def click(x, y):
    ed._on_press(FakeEvent(ed.ax, x, y))


ed._on_add_agent(None)
assert ed.placing is not None and ed.placing["clicks"] == []
click(1.0, 5.0)    # 1/3: start
assert len(ed.placing["clicks"]) == 1
click(3.0, 5.0)    # 2/3: waypoint
assert len(ed.placing["clicks"]) == 2
click(9.0, 5.0)    # 3/3: goal -> 확정
assert ed.placing is None
a1 = ed.selected_agent_id
spec1 = ed._find_agent_spec(a1)
assert spec1["start"] == [1.0, 5.0] and spec1["waypoint"] == [3.0, 5.0] and spec1["goal"] == [9.0, 5.0], spec1
print("[6] 3-클릭 에이전트 배치(start/waypoint/goal) OK:", spec1)

# 배치 도중 Esc 취소 확인
ed._on_add_agent(None)
click(0.0, 0.0)
assert len(ed.placing["clicks"]) == 1
ed._on_key(FakeKeyEvent("escape"))
assert ed.placing is None and ed._placing_artists == []
assert len(ed.data["agents"]) == 1, "Esc 로 취소했는데 에이전트가 추가돼버림"
print("[7] 배치 도중 Esc 취소 OK")

# 두 번째 에이전트 배치(취소 후 재시도)
ed._on_add_agent(None)
click(5.0, 1.0)
click(5.0, 5.0)
click(5.0, 9.0)
a2 = ed.selected_agent_id
assert a2 != a1
spec2 = ed._find_agent_spec(a2)
assert spec2["start"] == [5.0, 1.0] and spec2["waypoint"] == [5.0, 5.0] and spec2["goal"] == [5.0, 9.0], spec2
print("[8] 두 번째 에이전트 3-클릭 배치 OK:", spec2)

# 배치 모드가 아닐 때는 기존 모드(장애물/dependency 등) 클릭 처리가 그대로 동작해야 함
assert ed.placing is None
ed.mode = "장애물"
before = len(ed.data["obstacles"])
click(2.0, 2.0)   # 빈 곳 클릭 -> 새 장애물 드래그 시작(release 전이라 아직 미확정)
assert ed.drag is not None and ed.drag["kind"] == "new_rect"
ed.drag = None    # 실제 release 시뮬레이션은 생략, 상태만 리셋
print("[9] 배치 모드 종료 후 기존 '장애물' 모드 클릭 처리 정상 동작 OK")

# 4) dependency 추가 (+ 순환 거부 확인)
ed._add_dependency(a1, a2)
assert len(ed.data["dependencies"]) == 1
ed._add_dependency(a2, a1)   # a1->a2 이미 있는데 a2->a1 추가하면 순환
assert len(ed.data["dependencies"]) == 1, "cycle 이 거부되지 않음"
print("[10] dependency 추가 + 순환 거부 OK")
assert _has_cycle([a1, a2], [(a1, a2), (a2, a1)]) is True
assert _has_cycle([a1, a2], [(a1, a2)]) is False
print("[11] _has_cycle 단위 확인 OK")

# 5) Save -> 실제 시뮬레이터 로더로 재확인
ed.textbox_path.set_val(OUT)
ed._on_save(None)
assert os.path.exists(OUT)
scene, agents, prm = config.load_scenario(OUT)
assert len(agents) == 2
print("[12] Save 후 config.load_scenario() 로 재로딩 성공:",
      [(a.id, round(a.path.length, 2)) for a in agents])

# 6) Load 로 되불러오기(왕복 확인) + Validate
ed2 = Editor(OUT)
assert len(ed2.data["agents"]) == 2
ed2._on_validate(None)
print("[13] Load 왕복 + Validate 콜백 OK. status:", ed2.status_text.get_text())

os.remove(OUT)
print("\nALL EDITOR LOGIC CHECKS PASSED")
