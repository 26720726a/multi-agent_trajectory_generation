# 분산·실시간 멀티에이전트 궤적 협동 시뮬레이터 (방법 B)

중앙 계획자 없이, 각 에이전트가 공유 board에서 이웃의 **현재** 위치·속도만
보고 반응적으로 움직이는 toy 시뮬레이터. 검증 대상은 측위·센싱·통신의
현실성이 아니라 **탈중앙 coordination 로직**이다 (설계 전제는 아래 §가정 참고).

## 실행

```
python main.py
```

`out/` 아래에 시나리오별 애니메이션 gif, baseline(완전 직렬화) gif, 지표
`metrics.csv` 가 생성된다. 콘솔에 각 시나리오의 지표 요약이 출력된다.

개별 모듈을 실험해보려면:

```python
import scenarios, sim
scene, agents, prm = scenarios.scenario_s1()
result = sim.run(scene, agents, dt=prm.dt, tau=prm.tau, delta=prm.delta,
                  tie_break=True)
print(result.success, result.collisions_agent, result.steps)
```

장면을 코드 수정 없이 JSON으로 바꿔가며 실험하려면 아래 "시나리오 JSON
설정" 절의 `cli.py`를 쓴다.

## 시나리오 JSON 설정 (코드 수정 없이 장면 바꾸기)

`scenarios.py`의 S1~S4는 하드코딩이라 장애물 하나 옮기려 해도 코드를 고쳐야
한다. `config.py` + `cli.py`는 같은 장면을 **`scenario.json` 하나**로
표현해서, 장애물 수정·에이전트 추가·경로 변경·파라미터 튜닝을 **코드를 건드리지
않고** JSON 편집만으로 할 수 있게 한다. `scenarios/S1.json`~`S4.json` 이
하드코딩 버전을 그대로 옮긴 편집 가능한 예제다.

**reference 경로는 저장하지 않는다** — 로드할 때마다 (현재 장애물 +
start/waypoint/goal)로 A\*를 다시 돌려서 만든다. 그래서 장애물이나 시작·끝
점을 고치면 경로가 자동으로 따라온다(`path_override` 를 쓴 에이전트는 예외).

### 좌표 규약

원점 `(0,0)` 좌하단, `+x` 오른쪽, `+y` 위. `params.grid_w`/`grid_h` 는 곧
세계의 폭·높이(m)다(셀 1개 = 1 m). `obstacles`의 `rect` 타입은 셀 단위
`x,y,w,h`(좌하단 + 폭·높이), `circle` 타입과 에이전트 좌표(`start`/`goal`/
`waypoint`)는 이 프레임의 실수 `[x, y]`.

### CLI — validate → preview → run

```
python cli.py --scenario scenarios\S3.json --validate   # 로드+검증만
python cli.py --scenario scenarios\S3.json --preview     # 정적 PNG만 저장
python cli.py --scenario scenarios\S3.json --run          # 시뮬레이션 + gif/지표
```

JSON 몇 군데 고치고 `--preview`로 눈으로 확인 → 이상 없으면 `--run` 순서로
반복하면 된다. `--out`으로 출력 폴더를 바꿀 수 있다(기본은 시나리오 파일
옆의 `out/`).

### 스키마 요약

```json
{
  "params": {
    "grid_w": 20, "grid_h": 20, "tau": 1.5, "dt": 0.25,
    "default_radius": 0.28, "default_v_max": 1.0, "lookahead": 0.5,
    "safety_margin_delta": 0.0, "samples_dir": 16, "samples_speed": 5,
    "progress_window": 40, "progress_eps": 0.05, "tiebreak_enabled": true,
    "astar_inflate_margin": 0.08
  },
  "obstacles": [
    { "id": 1, "type": "rect",   "x": 5, "y": 0, "w": 2, "h": 8 },
    { "id": 2, "type": "circle", "cx": 15.5, "cy": 10.0, "r": 1.5 }
  ],
  "agents": [
    { "id": 1, "start": [1, 1], "goal": [18, 18], "waypoint": [9, 2],
      "radius": 0.4, "v_max": 1.0,
      "route_waypoints": [[6, 10], [12, 10]], "path_override": null }
  ],
  "dependencies": [ { "predecessor": 1, "successor": 2 } ]
}
```

| 필드 | 의미 |
| --- | --- |
| `obstacles[].type` | `rect`(x,y,w,h, 셀 단위) 또는 `circle`(cx,cy,r, 실수) |
| `agents[].waypoint` | dependency 체크포인트. **이 에이전트가 successor 로 쓰이면 필수** — predecessor 는 이 좌표를 자기 경로에 투영해 통과 여부를 board 에 알린다 |
| `agents[].route_waypoints` | A\* 체인에 순서대로 삽입되는 경유점(형태 잡기용, dependency 무관) |
| `agents[].path_override` | 있으면 A\*를 완전히 대체하는 폴리라인(그대로 arc-length 파라미터화만 함) |
| `dependencies[]` | `{predecessor, successor}` — successor 는 자기 waypoint 직전에서 predecessor 가 그 지점을 지날 때까지 대기(event-hold, §설계) |
| `params.astar_inflate_margin` | A\* 계획용 grid 를 `radius + 이 값`만큼 부풀림. 통로가 좁아 A\*가 경로를 못 찾으면 줄여본다(실제 회피 여유는 `safety_margin_delta` 가 담당) |
| `params.tiebreak_enabled` | 끄면 §2.5c 대칭 깨기가 꺼진다 — S4.json 에서 `false` 로 바꿔 `--run` 해보면 코드 수정 없이 deadlock/livelock 재현을 볼 수 있다 |

전체 필드 타입은 `scenario.schema.json`(JSON Schema)에 정리했다 — 에디터가
지원하면 자동완성·즉시 검증에 쓸 수 있다.

### 주석 규칙

표준 JSON은 주석이 없다. `_`로 시작하는 키(`_comment`, `_obstacles_comment`
등)는 로더가 파싱 후 무시하므로 메모 용도로 자유롭게 쓰면 된다(위 예제
JSON들도 이렇게 써 뒀다).

### 편집 예시 1 — 장애물 수정

`scenarios/S4.json`의 위쪽 벽(`id: 2`)을 조금 더 두껍게:

```diff
- { "id": 2, "type": "rect", "x": 1.0, "y": 1.75, "w": 6.0, "h": 0.25 }
+ { "id": 2, "type": "rect", "x": 1.0, "y": 1.70, "w": 6.0, "h": 0.30 }
```

저장 후 `--preview`로 통로 폭이 좁아진 걸 확인하고 `--run`.

### 편집 예시 2 — 에이전트 추가

`scenarios/S1.json`의 `agents` 배열에 객체 하나만 추가하면 3번째 에이전트가
생긴다(코드 수정 없음):

```diff
  "agents": [
    { "id": 1, "name": "red",  "start": [1, 5], "goal": [9, 5] },
    { "id": 2, "name": "blue", "start": [5, 1], "goal": [5, 9] },
+   { "id": 3, "name": "green", "start": [9, 9], "goal": [1, 1] }
  ]
```

### 편집 예시 3 — 경로 변경

`scenarios/S1.json`의 blue가 곧장 가지 않고 왼쪽으로 돌아가게 하려면
`route_waypoints`만 추가하면 된다(장애물 없는 장면이라 형태만 바뀐다):

```diff
- { "id": 2, "name": "blue", "start": [5, 1], "goal": [5, 9] }
+ { "id": 2, "name": "blue", "start": [5, 1], "goal": [5, 9],
+   "route_waypoints": [[2, 5]] }
```

장애물을 옮기거나 지워도(예: S4에서 벽 rect를 삭제) 다음 `--run`/`--preview`
때 A\*가 새 장애물 배치로 경로를 다시 계획한다 — reference 경로를 손으로
안 고쳐도 된다.

## 인터랙티브 에디터 (마우스로 장면 편집)

JSON을 텍스트로 고치는 대신 **화면에서 직접** 장애물 크기를 드래그로
조절하고, start/waypoint/goal을 클릭으로 찍을 수 있다.

```
python editor.py --scenario scenarios\S4.json
python editor.py                                  # 새 시나리오로 시작
```

에디터는 `scenario.json`의 **시각 프론트엔드일 뿐이다** — 장면 표현을 새로
만들지 않고 `scene.py`/`agent.py`/`config.py`의 기존 클래스를 그대로 읽고
쓴다. 저장 파일은 위 JSON 스키마와 100% 호환이라 **에디터 편집과 텍스트
편집을 자유롭게 왕복**할 수 있다(에디터로 저장 → 텍스트 에디터로 미세조정
→ 다시 에디터에서 Load, 얼마든지 가능).

### 화면 구성

- **왼쪽 캔버스**: grid, 장애물(원본 + 반지름만큼 부풀린 점선), 에이전트별
  start(○)/goal(△)/waypoint(◇) 마커와 경로. 검증 안 된 상태(예: start가
  장애물 안, 경로 없음)는 마커가 **빨간색**으로 표시된다(하단 상태줄에도
  이유가 뜬다) — 이건 강제로 막는 게 아니라 "지금 이 상태로는 시뮬 못
  돌린다"는 경고일 뿐이니, `Validate`로 정확한 이유를 다시 확인하고
  고치면 된다.
- **오른쪽 패널**: 모드 선택(RadioButtons), 현재 에이전트 선택
  (RadioButtons — 새 에이전트를 추가하면 자동으로 갱신됨), `grid-snap`
  체크박스, 선택된 장애물의 크기 슬라이더(사각형이면 w/h, 원이면 r만),
  버튼들(`에이전트 추가/삭제`, `장애물 추가/삭제`, `Validate`, `Save`,
  `Load`, `Run`), 파일 경로 입력창(`Save`/`Load`가 이 경로를 씀).

### 조작법

| 모드 | 동작 |
| --- | --- |
| **장애물** | 빈 곳 드래그 = 새 사각 장애물 생성. 장애물 몸통 드래그 = 이동. 장애물을 클릭해 선택하면 흰 사각 핸들(코너 4개 + 에지 4개)이 뜨는데, **핸들을 드래그하면 그 방향으로 크기가 조절**된다(핵심 기능). 선택된 장애물은 오른쪽 슬라이더로도 정밀 조절 가능. 선택 후 `Delete`/`Backspace` = 삭제. 빈 곳 클릭 = 선택 해제. |
| **start / waypoint / goal** | 캔버스 클릭 = **현재 선택된 에이전트**의 해당 지점을 그 자리에 배치. 기존 에이전트의 위치를 다시 잡을 때 쓴다. |
| **dependency** | 에이전트를 순서대로 두 번 클릭(마커 근처) = `predecessor ≺ successor` 추가. 이미 있는 dependency와 합쳐 순환(cycle)이 생기면 즉시 거부하고 상태줄에 경고. successor에 waypoint가 없으면 추가는 되지만 경고가 뜬다(§JSON 설정: successor는 waypoint가 필수). |

- **에이전트 추가**: 버튼을 누르면 즉시 만들어지지 않고 **배치 모드**로
  들어간다 — 이후 캔버스 좌클릭 **3번**이 순서대로 소비된다: 1번째=start,
  2번째=waypoint, 3번째=goal(여기서 에이전트가 확정되고 A\*가 다시 돈다).
  각 클릭마다 상태줄에 "다음엔 몇 번째로 무엇을 찍어야 하는지" 안내가
  뜨고, 찍은 점마다 회색 임시 마커가 보인다. **Esc**로 배치 도중 취소
  가능(지금까지 찍은 점은 버려지고 에이전트도 생성 안 됨). 이 배치 모드는
  다른 모드(장애물/dependency 등)의 클릭 처리보다 항상 먼저 소비하므로,
  배치 중엔 다른 모드 조작이 안 먹는다 — 완료하거나 Esc로 취소한 뒤에
  다른 조작을 하면 된다.
- **grid-snap** 체크 시 장애물 생성/이동/리사이즈 결과가 정수 셀 좌표로
  스냅된다(정밀 배치용). start/waypoint/goal 클릭 배치도 grid-snap이
  켜져 있으면 정수 좌표로 스냅된다.
- **A\*는 드래그가 끝난 뒤(release)에만** 다시 돈다 — 드래그 도중(motion
  이벤트)에는 도형 geometry만 빠르게 갱신하고 경로는 그대로 둔다(렉 방지).
- **Save** = 지금 메모리 상태를 파일 경로 입력창의 경로에 JSON으로 저장
  (reference 경로는 저장 안 함 — 파생물이라 다음 로드 때 다시 계산됨).
  **Load** = 그 경로를 다시 읽어 편집 상태로 복원. **Validate** = §7 검증
  리포트를 상태줄 + 콘솔에 출력(엄격 검증 — 캔버스의 느슨한 미리보기와
  달리 여기서 걸리면 진짜 문제). **Run** = 저장 → 검증 → 통과하면
  `cli.py --scenario <경로> --run` 을 **별도 프로세스**로 백그라운드
  실행(에디터 창은 안 멈춤, 결과는 `out/`에 gif로 쌓임).

### 알아둘 것

- 마우스 이벤트는 matplotlib 인터랙티브 백엔드가 있어야 동작한다(로컬
  데스크톱에서 Tk가 기본으로 잡힌다). 서버/헤드리스 환경에는 안 맞는다 —
  이런 GUI 코드가 늘 그렇듯, 실제 드래그 동작은 화면이 있는 로컬 환경에서
  직접 확인해야 한다(`test_editor.py`가 마우스 없이 로직 계층만 확인하는
  회귀 테스트를 제공한다: `python test_editor.py`).
- `viz.py`는 headless 저장을 위해 자기 모듈에서 matplotlib 백엔드를
  Agg로 강제한다. `editor.py`는 이걸 그대로 import하면 마우스 이벤트를
  못 받게 되므로, 그리기 코드만 떼어낸 `draw_common.py`(백엔드 비의존)를
  통해서만 `viz.py`와 그리기 로직을 공유한다.
- **창을 리사이즈하면 matplotlib 자체 버그로 콘솔에
  `AttributeError: 'ResizeEvent' object has no attribute 'inaxes'`
  traceback이 뜨는 문제가 있었다(최신 matplotlib 3.11.1에서도 재현됨 —
  `TextBox._resize`가 마우스 이벤트 전용 재부모화 데코레이터에 잘못
  감싸여 있어서, `.inaxes`가 없는 `ResizeEvent`가 들어오면 죽는다). 실행이
  멈추진 않지만(콜백 예외라 무시되고 계속 동작) 노이즈였다 — `editor.py`
  맨 위에서 `TextBox._resize`를 그 데코레이터 없는 원본 함수로
  바꿔치기해서 우회했다(원본은 `self.stop_typing()`만 하므로 안전).
  `test_editor.py`의 `[0]` 항목이 이 우회가 유지되는지 회귀 확인한다.
- **장애물 슬라이더를 조작하면 GUI가 완전히 멈추는 버그가 있었다.**
  `_on_slider_w/h`가 매 tick(슬라이더 값이 바뀔 때마다)마다 `_rebuild()`
  (= `ax.clear()` + 전체 에이전트 A\* 재계산 + 전체 redraw)를 곧바로
  불렀는데, 슬라이더를 드래그하면 아주 짧은 시간에 이 무거운 작업이
  수십 번 쌓여 Tk 이벤트 루프가 밀려버렸다. 게다가 `_refresh_obstacle_sliders`
  가 `set_val`로 슬라이더 값을 프로그램이 세팅할 때 쓰던
  `self._suspend_slider_cb` bool 가드는, `set_val`이 유발하는 `on_changed`
  콜백이 지연 실행되면 이미 가드가 풀린 뒤 불려 무력화될 수 있었다.
  두 가지로 고쳤다: (1) 프로그램이 슬라이더 값을 세팅할 땐
  `_set_slider_silently`로 `on_changed` 연결을 물리적으로 끊었다가
  다시 연결한다(타이밍과 무관하게 콜백이 절대 안 불림). (2) 슬라이더
  콜백은 도형 geometry만 즉시 갱신(`_push_obstacle_shape`, 가벼움)하고
  무거운 `_rebuild()`는 `_schedule_rebuild()`로 디바운스해
  (`REBUILD_DEBOUNCE_MS`=200ms 동안 추가 변경이 없을 때) 한 번만 돈다.
  드래그(`_on_motion`)는 원래부터 geometry만 갱신했고 `_on_release`에서만
  `_rebuild()`를 불렀으니 그대로 뒀다 — 문제는 슬라이더 쪽에만 있었다.
  `test_editor.py`의 `[2b]`~`[2d]`가 이 회귀를 확인한다.

## 모듈 구성

| 파일 | 역할 |
| --- | --- |
| `scene.py` | 정적 맵(사각형 장애물), A* 용 occupancy grid, 세계 경계 벽 |
| `astar.py` | 8-connected A*, 꺾임 최소화 tie-break |
| `path.py` | 폴리라인, arc-length 파라미터화, 임의 지점의 최근접 arc-length 조회 |
| `board.py` | 공유 blackboard — 더블 버퍼(스냅샷/커밋)로 동기 스텝 보장 |
| `rvo.py` | sampling 기반 velocity-obstacle 회피 (직접 구현) |
| `agent.py` | per-agent 스텝: preferred velocity + dependency hold + tie-break + RVO |
| `sim.py` | 동기 시뮬레이션 루프, 충돌/위반/deadlock 검출 |
| `baseline.py` | 완전 직렬화 strawman (한 명씩 순서대로) |
| `metrics.py` | 지표 요약 + CSV 저장 |
| `draw_common.py` | viz.py/editor.py 공용 그리기 헬퍼(백엔드 비의존) |
| `viz.py` | matplotlib 애니메이션(gif) + 정적 미리보기(preview) |
| `scenarios.py` | S1~S4 정의(하드코딩 버전, `python main.py` 가 씀) |
| `config.py` | `scenario.json` 로더 + 검증기(장면을 코드 밖으로 뺀 버전) |
| `cli.py` | JSON 시나리오 CLI — `--validate` / `--preview` / `--run` |
| `editor.py` | 마우스로 `scenario.json` 편집하는 인터랙티브 에디터 |
| `test_editor.py` | editor.py 로직 계층 회귀 테스트(마우스 없이 실행) |

## 시나리오가 보여주는 것

- **S1 (중앙 교차)** — dependency 없이 두 에이전트가 교차 경로로 이동. 중앙
  계획 없이 국소 회피만으로 충돌 없이 교차하는 것을 확인한다.
- **S2 (A ≺ B)** — S1과 같은 교차 기하에 dependency 하나를 건다. 후행 에이전트가
  자기 WP 직전에서 board의 '선행 통과' 이벤트 플래그만 보고 대기(hold)하다,
  플래그가 서면 바로 진행하는 event-hold를 확인한다 (미래 예측 없음).
- **S3 (체인 1≺2, {1,2}≺3)** — 3 에이전트로 확장. dependency가 여러 개
  얽혀도(합성 hold) 같은 루프가 그대로 확장되는지, board가 dependency
  위반을 여전히 0으로 보장하는지 확인한다.
- **S4 (좁은 통로 정면 대치)** — reciprocity(§2.5a)만으로는 안 풀리는 완전
  대칭 head-on 상황을 만든다. **tie-break를 끄면** 두 에이전트가 서로의
  '한 스텝 전' 속도만 보고 동시에 판단하다 (a) 제자리 대치 또는 (b) 서로
  동시에 돌진→급제동을 반복하는 왕복 진동(livelock)에 빠지고, 그 과정에서
  실제 충돌도 발생한다 — 즉 **의도적으로 실패하도록 설계**한 baseline이다.
  **tie-break를 켜면** ID가 낮은 쪽이 먼저 상황을 감지해 통로 옆으로 붙어
  대기하고, 높은 쪽이 지나간 뒤 다시 출발해 둘 다 충돌 없이 도착한다.

## 실측 결과 (`out/metrics.csv`, 파라미터는 `scenarios.Params` 기본값)

| 시나리오 | 성공 | 충돌(agent/장애물) | dep. 위반 | makespan B | makespan baseline | B/baseline |
| --- | --- | --- | --- | --- | --- | --- |
| S1 (교차) | True | 0 / 0 | 0 | 51 | 66 | **0.77** |
| S2 (A≺B) | True | 0 / 0 | 0 | 54 | 66 | **0.82** |
| S3 (체인) | True | 0 / 0 | 0 | 130 | 129 | 1.01 |
| S4 tie-break **OFF** | **False** | **20** / 0 | 0 | 53 (deadlock으로 조기 종료) | — | — |
| S4 tie-break **ON** | True | 0 / 0 | 0 | 73 | 58 | 1.26 |

S1·S2에서 B가 baseline보다 makespan이 **더 짧다** — 완전 직렬화(한 번에 한
명)는 서로 안 부딪히는 구간에서도 무조건 순서를 기다리지만, B는 국소적으로
안전하기만 하면 동시에 움직이기 때문이다("국소 양보 > 전역 대기"). S3는
거의 같고(1.01), S4는 오히려 B가 baseline보다 느리다(1.26) — 통로가
좁아 실제로 한 명이 완전히 비켜서서 기다려야 하는 시나리오라 baseline과
비슷한 대가를 치르기 때문이다. **S4의 핵심 비교는 baseline이 아니라
tie-break on/off다**: 끄면 아예 실패(충돌 20건, deadlock)하고 켜면 충돌
없이 성공한다.

콘솔에 tie-break OFF일 때 `[sim] deadlock/livelock at step 54: stuck agents
= ['red', 'blue']` 가 그대로 출력된다 — 두 에이전트 모두 진행이 없어
시뮬레이션이 스스로 중단한 것이다(그 과정에서 실제 충돌도 20회 발생했다 —
아래 "설계 결정" 참고). ON일 때는 같은 초기조건에서 두 에이전트 모두
충돌 없이 목표에 도달한다.

## 핵심 설계 결정 (구현 중 실제로 부딪힌 문제들)

- **정적 장애물도 매 스텝 반응 회피** (§2.3): 사각형 장애물을 '속도 0인
  이웃'으로 놓고 이웃과 같은 ttc 공식을 쓴다. 단, 사각형을 그대로 반지름만큼
  AABB로 부풀리면 모서리 바깥까지 네모나게 막혀버려 '통로 입구에 가까이 갈수록
  벽에 다가가는 것으로 오판'하는 문제가 생겼다 — 사각형 위의 최근접 점을
  반지름 0인 정적 이웃으로 보고 원-원 ttc 공식을 그대로 적용해(= rect ⊕ disk
  의 Minkowski 합, 모서리가 자연히 둥글게 처리됨) 해결했다.
- **정적 장애물엔 짧은 horizon(`tau_obs`)을 따로 쓴다**: 이웃과 같은 긴
  `tau`를 벽에도 그대로 쓰면, 좁은 통로에 들어가려고 벽에 가까워지는 것
  자체가 매 스텝 비용으로 잡혀 '제자리에 서 있는 게 더 싸다'는 함정에
  빠진다(실제로 관찰됨). 벽은 위치가 100% 확실하니 한두 스텝만 내다보면
  충분하다.
- **속도 관성(smoothing) 항**: 동기 스냅샷 구조상 두 에이전트는 서로의
  '한 스텝 전' 속도만 보고 동시에 결정한다. 감쇠가 없으면 멀 때는 둘 다
  전속 돌진, 가까워지면 둘 다 급제동 — 을 반복하는 왕복 진동에 빠지는 걸
  실제로 관찰했다. 직전 속도에서 너무 먼 후보에 작은 페널티를 줘서 잡았다.
- **baseline(완전 직렬화)에서 dependency hold는 처음부터 충족된 것으로
  둔다.** 처음엔 dependency가 걸린 에이전트를 baseline에서도 그대로
  `hold` 로직째 혼자 실행시켰는데, 혼자만 있는 board에는 predecessor가
  없어 '통과 이벤트'가 영원히 안 서고 makespan이 수천 스텝으로 폭발했다.
  완전 직렬화 순서상 이 에이전트 차례가 왔다는 건 predecessor가 이미 다
  끝났다는 뜻이므로, `baseline.py`는 그 사실을 board에 미리 심어둔다.
- **tie-break는 비용 항이 아니라 목표 지점 전환으로 구현** (§2.5c): 처음엔
  '옆으로 붙는 방향'에 보너스를 주는 항을 비용함수에 얹었는데, 회피 목적과
  양보 목적이 같은 함수 안에서 서로 밀고 당기며 못 풀리거나 진동하는 경우가
  잦았다. 최종적으로는 tie-break가 트리거되면(§2.5c, 낮은 ID + 임박 + 거의
  정면) `agent.py`가 **preferred velocity 자체**를 '통로 옆 대피 지점'으로
  바꿔치기하고, `rvo.py`는 평소와 똑같이 '지금 원하는 속도로 안전하게
  가라'는 하나의 목적만 푼다. 한번 트리거되면 몇 스텝(기본 16 스텝) 동안
  결정을 유지하는 쿨다운도 필요했다 — 안 그러면 매 스텝 판정이 뒤집혀 같은
  왕복 진동이 재발했다.
- **사각형 장애물을 원(Circle)까지 지원하도록 `rvo.py`를 일반화했다** (§JSON
  설정: `obstacles[].type: "circle"`). ttc/침투 회복 로직을 타입별로
  분기하는 `ttc_obstacle`/`obstacle_outward` 두 함수로 정리해서, Rect든
  Circle이든 `choose_velocity`는 신경 쓰지 않는다.
- **파라미터 값 하나를 실수로 다르게 넣어서 한동안 헷갈렸다.** `S4.json`을
  만들 때 예전 버전의 `scenario_s4()`가 쓰던 `delta=0.12`를 그대로
  베꼈는데, 그 사이 코드가 `delta=0.0`으로 바뀌어 있었다. 겉보기엔 "완전히
  같은 조건인데 결과가 다르다"는 재현성 문제처럼 보였지만, 실은 두 값이
  진짜 달랐던 평범한 버그였다 — dataclass 필드를 하나하나 diff 떠서 찾았다.
  **JSON으로 옮길 때는 하드코딩 버전의 현재 값을 직접 diff로 확인하고
  옮길 것**(기억에 의존하지 말 것).
- **Windows 에디터로 저장한 JSON은 UTF-8 BOM이 붙는 경우가 흔하다.**
  `config.py`는 `utf-8-sig`로 읽어서 BOM이 있든 없든 그냥 통과시킨다.

## 지표 (§7)

`sim.SimResult` / `metrics.summarize()` 가 기록하는 것들:
agent-agent/장애물 충돌 수, dependency 위반 수, makespan, 총 경로 길이,
deadlock/livelock bool, 에이전트별 양보(yield) 횟수, tie-break 발동 횟수,
baseline 대비 makespan 비율. S1~S4 모두에서 충돌·dependency 위반은 0 이어야
하며(S4는 tie-break ON일 때), 실제로 그렇다.

## 알려진 한계 (§12)

- **Perfect relative state 가정(B0).** 실제 로봇의 상대 측위·통신은 이
  toy가 검증하는 대상이 아니다. 완화는 안전 마진 `δ`(기본 off)가 최소
  침습 버전이며, 노이즈/필터는 의도적으로 넣지 않았다. S4는 현재
  `δ=0`으로도 풀린다(대칭 깨기 자체가 핵심이라 여유 마진이 필수는 아님) —
  통로를 더 좁히는 등 여유가 필요해지면 `safety_margin_delta`를 조금씩만
  올려보며 재확인할 것(§JSON 설정 절 참고, 너무 키우면 오히려 실패할 수
  있다 — 실제로 이 값 하나 잘못 맞춰서 S4.json이 한동안 재현 안 됐었다).
- **미래 궤적을 모른다 → 반응적 → makespan 최적 보장 없음.** feasible한
  완료 시간을 얻어 baseline과 비교할 뿐, 최적이라 주장하지 않는다.
  (baseline의 makespan을 순열 탐색 등으로 최적화하지도 않았다.)
- **Deadlock 표면은 완전히 없어지지 않는다.** reciprocity + ID tie-break로
  S1~S4는 다 풀리지만, 더 좁고 적대적인 구성(예: 3대 이상이 동시에 한
  지점에서 대칭으로 마주치는 경우)에서는 여전히 막힐 수 있다. deadlock
  감지기(progress window)가 이를 로그로 드러내도록만 해뒀다.
- **동기 스텝 가정.** 비동기·가변 지연은 모델링하지 않았다. 오히려 이
  가정 때문에 '서로 한 스텝 전 상태만 보고 동시에 판단'하는 진동 실패
  모드가 실제로 나타났고(위 설계 결정 참고), 이는 동기 통신이라도 완벽한
  정보가 아니라 '약간 지연된' 정보라는 걸 보여주는 사례이기도 하다.
- **tie-break의 우선순위 규칙(ID 낮은 쪽이 항상 양보)은 정적이다.** 공정성
  (매번 같은 에이전트가 양보)은 다루지 않았다 — 3개 시나리오 규모에서는
  중요하지 않지만, 에이전트 수가 늘면 문제가 될 수 있다.
- **S3의 makespan은 tie-break가 자주(그리고 다소 보수적으로) 발동해 baseline
  대비 이득이 S1/S2만큼 크지 않다.** 3자 이상이 한 지점 근처에서 얽히면
  '임박+정면' 판정이 자주 걸려 필요 이상으로 자주 대피 모드에 들어간다 —
  안전(충돌 0)은 지키지만 효율은 덜 최적화됐다.

## Phase 9 (미구현)

방법 A(중앙집중 space-time 계획)와의 3자 비교(baseline / A / B)는 이
구현에 포함하지 않았다. 핵심 산출물은 B + baseline이다.
