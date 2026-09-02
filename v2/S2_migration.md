# S2 — 코드 이관 및 회귀 검증

원본 `Mahoi-WM/mahoi-wm` (커밋 `abf8b8a`) 의 코드를 v2 로 옮겼다.
**로직은 하나도 바꾸지 않았다** — 위치 이동과 config 참조로의 재배선뿐이다.
게이트는 "같은 시드에서 이관 전후 결과가 같은가"이고, 결과는 §4 에 있다.

작업일 2026-08-29.  커밋 `0df799b` … `621954c` (묶음별 5개).
실행 환경: `Mahoi-WM/mahoi-wm/.venv` 의 Python 3.12 / numpy 2.5.2 / pytest 9.1.1.

---

## 1. 게이트 표

| # | 게이트 | 결과 | 근거 |
|---|--------|------|------|
| 1 | `reports/S1_skeleton.md`, `reports/S2_migration.md` 존재 | **통과** | 이 파일과 옆 파일 |
| 2 | S1 소급 검증 5항목 통과 | **통과** (5/5) | `reports/S1_skeleton.md`, 요약은 §2 |
| 3 | v2 전체 테스트 통과 | **통과** | `155 passed` / `--runslow` 시 `161 passed` (§3.7) |
| 4 | 묶음 6 (a)(b) 수치 일치 | **통과** | (a) diff 0줄, (b) 15행 전 열 일치 (§4) |
| 5 | config 밖 물리 리터럴 0건 | **통과** | `tests/test_layering.py` (c1)(c2) 가 실제 코드에서 통과 (§3.8) |
| 6 | 원본 저장소 무변경 | **통과** | 9,846개 파일 md5 전부 동일 (§4.4) |

**종합: 6/6 통과.**

추가로, 게이트가 요구한 것보다 강한 결과가 나왔다: 5종 시나리오 x 3 시드의
궤적이 makespan 뿐 아니라 **전 타임스텝의 pos/vel 까지 비트 단위로 동일**하고,
smoke 배치는 makespan/성공플래그뿐 아니라 `distance` · `wait` ·
`agent_clearance` · `n_switches` · `n_feasible_modes` 를 포함한 **전 열**이
일치한다.

---

## 2. S1 소급 검증 결과 (요약)

| # | 항목 | 결과 |
|---|------|------|
| 1 | config import 성공 | 통과 |
| 2 | S0 §C 15개 개념이 config 에 전부 | 통과 (15/15) |
| 3 | `stall_window(45) > horizon_steps(40)` 경고 출력 | 통과 |
| 4 | `tests/test_layering.py` 존재·통과 | 통과 (단서 2건) |
| 5 | config 값이 현행 값 (`v_max=1.20`, `horizon_s=4.0`, `a_max=inf`) | 통과 — 목표값 미유입, **멈출 사유 없음** |

전문은 `reports/S1_skeleton.md`.  단서 2건은 (a) ROS 가 pytest 플러그인을
등록해 `PYTHONPATH=` 없이는 수집 단계에서 죽는다(원본도 동일한 환경 문제),
(b) S1 시점의 계층 테스트 (c) 는 검사할 코드가 없어 자명하게 통과했고, 실제
코드가 오자 오탐이 드러나 이번에 고쳤다 — §3.8.

---

## 3. 묶음별 이관 내역

### 3.1 묶음 1 — 리프 (`0df799b`)

| 원본 | v2 | LOC | 바꾼 것 |
|------|----|-----|---------|
| `mahoi/geometry.py` | `safety/geometry.py` | 142 → 142 | 없음 (그대로) |
| `mahoi/wm/traj.py` | `planning/traj.py` | 195 → 195 | 없음 (그대로) |
| `mahoi/validate.py` | `safety/validate.py` | 143 → 147 | import 방향만 |

**`validate.py` 에서 한 가지.**  원본은 `from .coordination import Solution` 과
`from .world import Problem` 을 갖고 있었는데, 두 모듈은 v2 에서 `planning/` 에
간다.  그대로 두면 `safety/` 가 `planning/` 을 import 하게 되어 계층 규칙
(`tests/test_layering.py` (a))을 깬다.

두 이름은 원본에서도 **타입 어노테이션 외에는 쓰이지 않았다.**  그래서 import 를
지우고 `def validate(problem: "Problem", sol: "Solution", ...)` 로 문자열
전방 참조를 썼다.  `from __future__ import annotations` 아래이므로 런타임에
평가되지 않는다 — **동작은 원본과 완전히 같다.**
`tests/test_import_smoke.py` 가 "`safety.validate` 를 import 하는 것만으로
`planning` 이 딸려 오지 않는다"를 확인한다.

테스트: 원본에 대응 테스트가 없어 import 스모크 3개만 새로 썼다
(`tests/test_import_smoke.py`).

### 3.2 묶음 2 — 안전 계층 추출 (`46500c2`)

원본 `controller.py`(499 LOC)에서 안전에 해당하는 부분만 떼어냈다.

| 원본 위치 | v2 | 바꾼 것 |
|-----------|----|---------|
| `controller.py:466-499` `_project_safe` | `safety/project.py` (51) | 없음.  이름·시그니처·기본값(`iters=12`)·계산 순서 전부 동일 |
| `controller.py:286-306` `_ttc` | `safety/vo.py` (93) | 없음 |
| `controller.py:388-411` VO 필터 (인라인) | `safety/vo.py::add_vo_cost` | 함수로 추출.  상수를 config 참조로 |
| `controller.py:126` `PHASE_*` | `safety/phase.py` (16) | 위치만 |

**왜 `PHASE_*` 가 아래로 내려갔는가.**  `_project_safe` 는 "움직여 달라고 할 수
없는 에이전트"(dwell 중이거나 도착)를 알아야 하고 `add_vo_cost` 는 "주차된
에이전트는 비켜 주지 않는다"를 알아야 한다.  `planning/` 에 두면 safety →
planning import 가 생긴다.  `planning/control.py` 가 같은 이름으로 re-export
하므로 원본의 import 경로(`from ...control import PHASE_DONE`)는 그대로 살아
있고, `tests/test_contract.py` 에 "두 벌이 아니라 같은 객체인가"를 확인하는
테스트를 하나 추가했다.

**`add_vo_cost` 가 `cost` 를 제자리에서 갱신하는 이유.**  원본은 이미 만들어진
`cost` 배열에 `cost[hit] += ...` 로 누적한다.  새 배열에 따로 모았다가 마지막에
한 번 더하면 부동소수 덧셈의 **순서**가 달라져 마지막 자리가 어긋날 수 있고,
그러면 바로 아래 `argmin` 이 동률 근처에서 뒤집혀 rollout 전체가 갈라진다.
게이트가 "원본과 같은 결과"이므로 비트 단위 동일성이 필요했다.

`add_vo_cost` 는 `Problem` 을 타입으로 받지 않고 `n / dt / agents / min_sep` 만
읽는 덕 타이핑으로 다루며, 장애물 기하는 호출부가 계산해 둔 `near` 배열로 받는다
(원본의 `scene.nearest_points(pos[i])`).  그래야 `safety/` 가 `SceneCache`
(planning 소속)를 import 하지 않는다.

**바꾸지 않은 것:** VO 의 계산식, 컷오프 조건, `aij` 처리, 반복 순서, 상수 값.

**대조 결과 (무작위 100 케이스, `==` 판정):**

| 대상 | 결과 |
|------|------|
| `add_vo_cost` vs 원본 인라인 블록 | 100/100 비트 단위 동일 (VO 항이 실제로 붙은 케이스 97) |
| `_ttc` | 100/100 동일 |
| `_project_safe` | 100/100 동일 (속도가 실제로 줄어든 케이스 다수) |

두 벌로 확인했다.  `scripts/s2_regression/compare_vo.py` 는 **원본 저장소를
직접 import** 해 원본 상수·원본 `_ttc` 로 참조 구현을 만든다.
`tests/test_vo_extraction.py` 는 원본 코드를 글자 그대로 옮긴 자립형 참조
구현(상수도 원본 리터럴을 그대로 적었다 — config 를 참조하면 "둘 다 config 를
읽으니 같다"는 동어반복이 된다)으로 같은 대조를 영구히 남긴다.

**지시서와 달라진 점:** 지시서는 묶음 2 에서 `tests/test_safety.py` 를
이관하라고 했으나, 그 파일은 `execute` / `world` 를 import 하므로 묶음 4 전에는
돌 수 없다.  "묶음마다 테스트 통과"를 지키기 위해 묶음 4 로 미뤘다.  대신 위의
`test_vo_extraction.py` 4개가 묶음 2 의 검증을 맡는다.

### 3.3 묶음 3 — world / paths (`55dfe6b`)

| 원본 | v2 | LOC |
|------|----|-----|
| `mahoi/world.py` | `planning/world.py` | 329 → 335 |
| `mahoi/paths.py` | `planning/paths.py` | 284 → 284 |

**config 참조로 바꾼 것 (값은 전부 동일):**

| 원본 위치 | 원본 값 | v2 |
|-----------|---------|----|
| `world.py:33` `AgentSpec.dwell` | 1.0 | `PHYSICS.dwell_s` |
| `world.py:34` `AgentSpec.radius` | 0.30 | `PHYSICS.robot_radius` |
| `world.py:35` `AgentSpec.v_max` | 1.20 | `PHYSICS.v_max` |
| `world.py:36` `AgentSpec.interact_radius` | 0.55 | `PHYSICS.interact_radius` |
| `world.py:75` `Problem.safety` | 0.10 | `PHYSICS.safety_margin` |
| `world.py:76` `Problem.dt` | 0.10 | `PHYSICS.dt` |
| `world.py:250` `random_problem(size=)` | 10.0 | `PHYSICS.map_size` |
| `world.py:265` `radius` 재선언 | 0.30 | `PHYSICS.robot_radius` |
| `world.py:265` `margin` | 0.45 | `PHYSICS.gen_obstacle_margin` (신설, §5) |

**바꾸지 않은 것:**

* **시나리오별 dwell 개별 지정 11곳** (`world.py:143~226` 의 1.2 / 1.4).
  시나리오 정의의 일부이지 전역 상수가 아니다 — config 로 빼면 오히려 틀린다.
* **시나리오 장애물 좌표** `Rect(2.6, 5.0, 5.4, 7.6)` 등.  같은 이유.
* `paths.py` 는 `geometry` import 경로만 `safety.geometry` 로 바꿨다.
  `speed_levels` 인자 구조도 지시대로 그대로다 (§5 에 이름 충돌 기록).

**대조 결과:**

| 대상 | 결과 |
|------|------|
| `AgentSpec` 기본값 4개, `Problem` 기본값 3개 | 전부 동일 |
| 고정 시나리오 5종 (에이전트·장애물·deps·dt·safety) | 5/5 동일 |
| `random_problem` 100 시드 (n=2,3,4) | 100/100 동일 |
| `build_prior` 20 시드의 `AgentTrack` (pts/wp_start/wp_end/length) | 전부 동일 |

스크립트: `scripts/s2_regression/compare_world.py`.

### 3.4 묶음 4 — 제어/계획 본체 (`98f90f8`)

| 원본 | v2 | LOC |
|------|----|-----|
| `mahoi/wm/controller.py` (잔여) | `planning/control.py` | 499 → 445 |
| `mahoi/wm/worldmodel.py` | `planning/worldmodel.py` | 333 → 340 |
| `mahoi/wm/planner.py` | `planning/planner.py` | 274 → 283 |
| `mahoi/wm/execute.py` | `planning/execute.py` | 330 → 312 |
| `mahoi/coordination.py` | `planning/coordination.py` | 637 → 603 |

**`control.py`** — "Tunables" 블록 11개 + `SPEED_LEVELS` / `ANGLE_LEVELS` 를
config 참조로.  **원본의 이름을 그대로 남겼다** (`TAU = CONTROLLER.tau` 처럼).
본문의 `TAU`, `SOFT_MARGIN` 같은 표현이 원본과 글자까지 같아야 diff 가 읽힌다.

`ANGLE_LEVELS` 만 계산 경로가 다르다: 원본은 `np.deg2rad([...])`, config 는
`math.radians`.  두 결과와 그 `cos`/`sin` 이 **비트 단위로 같은 것을 확인한 뒤**
바꿨다.

**`worldmodel.py`** — `stall_window` 45 (2곳), `max_modes` 16, `k_routes` 2 의
인자 기본값을 config 참조로.  `stall_window` 는 원본에서 **인자 기본값이라
config 로 조절할 수 없었던** 값이다 (P0-2 STEP 1).

`sample_modes` 의 전수 리스트 생성 문제는 **고치지 않았다** (지시대로 S4=Phase 1).
`NOTE(S2)` 주석만 달아 두었다.

**`planner.py`** — `W_HARD` 와 `CostWeights` 8개 필드 전부를 config 참조로.
지시서는 `soft_margin` 만 지정했으나, 나머지 7개도 config 에 이미 정의되어
있어 두 벌로 두면 S0 §C-2 가 지적한 사고를 그대로 재현하게 된다.  "하드코딩
리터럴을 발견하면 config 참조로 바꾸되 값은 동일하게"라는 절대 규칙에 따라
전부 옮겼다.  값은 전부 동일하다.

`n_unfin = 0 if hit_horizon` (원본 `planner.py:218`) 은 **그대로 두었다.**
`NOTE(S2)` 주석과 §6 에 기록.

**`execute.py`** — `WMConfig` 기본값 11개와 프리셋 3종(`cfg_receding` /
`cfg_reselect_full` / `cfg_static`)을 config 참조로.  원본은 같은 숫자를
`execute.py:51-62`, `execute.py:320,325,330`,
`scripts/run_wm_experiments.py:43` 세 군데에 적어 두고 있었다.

**scorer 훅 제거.**  `mahoi/wm/scorer.py`(torch)는 이관 대상이 아니므로
(미확인 분류), `scorer="nn_filter"` 경로를 지웠다.  구체적으로 제거한 것:

* `WMConfig.scorer` / `.nn_model_path` / `.nn_threshold` / `.nn_device` 4개 필드
* `run_wm_planner` 안의 `if cfg.scorer == "nn_filter" and replan_idx == 0:` 블록
  (`from .scorer import completion_prob` 포함)
* `WMResult.nn_filtered` / `.nn_kept` 2개 필드

기본값은 `scorer="rollout"` 이었으므로 **기본 경로의 동작은 바뀌지 않는다.**
저장소 전체에서 이 필드들을 읽는 코드는 없었고 (`bench/`, `tests/` 확인),
동봉된 bench config 5종 중 `scorer` 를 쓰는 것도 없다.

**`coordination.py`** — `plan_coordination_multiroute`(원본 602-637) 제거.
저장소 전체에서 호출부가 0건인 죽은 코드였다.  그 함수만 쓰던
`VisibilityGraph` / `build_route_options` import 도 함께 걷어냈다.  나머지는
import 경로만 바꿨다.

**테스트 이관:** `tests/test_safety.py` (15개), `tests/test_contract.py`
(14 → 15개, PHASE 단일 출처 확인 1개 추가).  둘 다 통과.

**대조 결과 — 이것이 묶음 4 의 핵심이다.**

5종 시나리오 x 3 시드 = 15회 실행에서 `pos` · `vel` · `wp_in` · `wp_out` ·
`done` · `team_time` · `feasible` 이 **전부 비트 단위로 동일**했다
(불일치 배열 0개).

```
  chain3.0       team_time orig= 19.40  v2= 19.40  |d|=0.0e+00  arrays=IDENTICAL
  chain3.1       team_time orig= 19.50  v2= 19.50  |d|=0.0e+00  arrays=IDENTICAL
  chain3.2       team_time orig= 25.20  v2= 25.20  |d|=0.0e+00  arrays=IDENTICAL
  corridor2.0    team_time orig= 18.10  v2= 18.10  |d|=0.0e+00  arrays=IDENTICAL
  corridor2.1    team_time orig= 18.20  v2= 18.20  |d|=0.0e+00  arrays=IDENTICAL
  corridor2.2    team_time orig= 18.20  v2= 18.20  |d|=0.0e+00  arrays=IDENTICAL
  crossing2.0    team_time orig= 16.40  v2= 16.40  |d|=0.0e+00  arrays=IDENTICAL
  crossing2.1    team_time orig= 16.40  v2= 16.40  |d|=0.0e+00  arrays=IDENTICAL
  crossing2.2    team_time orig= 16.40  v2= 16.40  |d|=0.0e+00  arrays=IDENTICAL
  deadlock2.0    team_time orig= 15.50  v2= 15.50  |d|=0.0e+00  arrays=IDENTICAL
  deadlock2.1    team_time orig= 15.40  v2= 15.40  |d|=0.0e+00  arrays=IDENTICAL
  deadlock2.2    team_time orig= 15.40  v2= 15.40  |d|=0.0e+00  arrays=IDENTICAL
  fork3.0        team_time orig= 19.40  v2= 19.40  |d|=0.0e+00  arrays=IDENTICAL
  fork3.1        team_time orig= 20.20  v2= 20.20  |d|=0.0e+00  arrays=IDENTICAL
  fork3.2        team_time orig= 19.90  v2= 19.90  |d|=0.0e+00  arrays=IDENTICAL

mismatched arrays: 0
RESULT: IDENTICAL
```

스크립트: `scripts/s2_regression/compare_traj.py`.  두 저장소를 한 프로세스에서
번갈아 import 하면 모듈 캐시가 섞이므로 각각을 **별도 프로세스**로 돌려 `.npz`
로 떨어뜨린 뒤 비교한다.

### 3.5 묶음 5 — 벤치 (`621954c`)

| 원본 | v2 | 바꾼 것 |
|------|----|---------|
| `bench/generate.py` | `bench/generate.py` (353 → 356) | import 경로 + `Axis.size` 기본값 |
| `bench/analyze.py` | `bench/analyze.py` (646 → 646) | 없음 |
| `bench/run.py` | `bench/run.py` (897 → 897) | import 경로 + 주석 한 줄 |
| `bench/configs/*.json` 5종 | 같음 | 없음 |
| `tests/test_bench_{generate,analyze,run}.py` | 같음 (33/23/47개) | 없음 |
| `tests/conftest.py` | 같음 | 없음 |

`Axis.size` 기본값 `(10.0,)` 는 `(PHYSICS.map_size,)` 로 바꿨다 — 값 동일.
난이도 격자가 8/10/12 를 쓸 때는 bench config 가 덮어쓰므로 영향이 없다.

**ROS 격리 확인 (지시서 요구):**

```
before: PYTHONPATH= /opt/ros/jazzy/lib/python3.12/site-packages
before: sys.path 에 ros 2
removed: ['/opt/ros/jazzy/...', '/opt/ros/jazzy/...']
after : PYTHONPATH= None
after : sys.path 에 ros 0
```

`strip_ros_paths` 가 v2 경로에서도 `sys.path` 2건과 `PYTHONPATH` 를 실제로
걷어낸다.  `tests/test_bench_run.py` 의 해당 테스트 2건도 통과.

### 3.6 묶음 6 — 회귀 검증 (`621954c`)

`scripts/check_wm.py` 를 포팅했다 (import 6줄만 수정, 108 → 108 LOC).
결과는 §4 에.

`tests/test_regression.py` 와 `results/baseline_v0.json` 도 함께 가져왔다.
**지시서의 이관 목록에는 없는 항목이다.**  가져온 이유는 이것이 묶음 6(a) 를
**자동화된 형태로 영구히 남기는 유일한 테스트**이기 때문이다 — 원본 저장소가
사라지면 `scripts/s2_regression/` 의 대조 스크립트 넷은 돌지 않지만, 이 테스트는
기준선을 JSON 으로 얼려 두었으므로 계속 돈다.  `pyproject.toml` 에 `slow`
마커를 등록했다 (`tests/conftest.py` 의 `--runslow` 가 의미를 가지려면 필요).

### 3.7 테스트 현황

| | 원본 | v2 | 비고 |
|---|---|---|---|
| `test_bench_generate.py` | 33 | 33 | 그대로 |
| `test_bench_analyze.py` | 23 | 23 | 그대로 |
| `test_bench_run.py` | 47 | 47 | 그대로 |
| `test_contract.py` | 14 | **15** | PHASE 단일 출처 확인 1개 추가 |
| `test_safety.py` | 15 | 15 | 그대로 |
| `test_regression.py` | 6 (slow) | 6 (slow) | 그대로 |
| **원본 소계** | **138** | **139** | |
| `test_config_validation.py` | — | 8 | S1 산출 |
| `test_layering.py` | — | 7 | S1 산출 (5 → 7, §3.8) |
| `test_import_smoke.py` | — | 3 | S2 묶음 1 |
| `test_vo_extraction.py` | — | 4 | S2 묶음 2 |
| **합계** | 138 | **161** | |

```
$ PYTHONPATH= python -m pytest -q
155 passed, 6 skipped, 1 warning in 5.74s

$ PYTHONPATH= python -m pytest -q --runslow
161 passed, 1 warning in 29.53s
```

원본에서 옮기지 않은 소스는 `mahoi/viz.py`, `mahoi/wm/viz.py`
(선택 트랙 `viz/`), `mahoi/wm/scorer.py` (§3.4), `scripts/` 의
`build_dataset.py` / `train_wm.py` / `make_wm_figs.py` /
`run_wm_experiments.py` (torch·matplotlib 트랙).  전부 지시서의 이관 목록 밖이다.

### 3.8 게이트 5 — `tests/test_layering.py` (c) 를 고친 일

**이것은 S1 산출물을 수정한 것이므로 따로 밝혀 둔다.**

S1 의 판정은 정규식이었다.  `("1.20", "0.30", "0.70", "2.6", "0.22")` 가
주석 아닌 코드 줄에 나타나면 실패시킨다.  `safety/` 와 `planning/` 이 비어
있는 동안에는 자명하게 통과했고, 실제 코드를 올리자 **오탐 8건**이 나왔다:

| 자리 | 값 | 실제 개념 |
|------|-----|-----------|
| `world.py` 의 `Rect(2.6, 5.0, 5.4, 7.6)` 등 6줄 | 2.6 | 시나리오 장애물 좌표 |
| `paths.py:232` `for cut in (0.35, 0.22, 0.12, 0.05)` | 0.22 | 필렛 축소 스케줄 |
| `worldmodel.py:46` `ALPHA_PRIVILEGED = 0.30` | 0.30 | 무차원 책임 분담 비율 |

셋 다 물리 상수가 아니다.  값만 우연히 겹친다.

판정을 AST 기반으로 바꾸고 두 축으로 나눴다.

* **(c1) 이름 기준** — config 가 이름을 가진 개념(`v_max`, `radius`, `tau`,
  `soft_margin`, `horizon_s`, `stall_window`, `SPEED_LEVELS` 등 약 40개)이
  코드 안에서 **리터럴로 정의되면** 실패.  `v_max = 1.20` 은 걸리고
  `v_max = PHYSICS.v_max` 는 통과한다.  값이 우연히 같은지가 아니라 **누가
  그 개념을 정의하는가**를 본다 — S0 §C-2 의 사고가 정확히 이것이었다.
* **(c2) 값 기준** — 위 5개 값이 config 밖에서 정의되면 실패.  이름을 바꿔
  (c1) 을 우회하는 것을 막는다.

보는 자리는 **정의가 일어나는 세 곳**으로 좁혔다: 대입, 어노테이션 대입
(dataclass 필드 기본값), 인자 기본값.  호출부의 키워드 인자
(`AgentSpec("A", ..., dwell=1.4)`)와 위치 인자(`Rect(2.6, ...)`)는 정의가
아니라 인스턴스 데이터라 보지 않는다 — 시나리오마다 다른 dwell 은 시나리오
정의의 일부이고, 그것을 config 로 빼면 오히려 틀린다 (지시서와 같은 판단).
튜플/리스트는 한 겹 열지만 호출 안으로는 들어가지 않는다.

이렇게 하면 위 오탐 8건 중 7건이 **구조적으로** 빠진다.  남은 하나
(`ALPHA_PRIVILEGED = 0.30`)는 값·이름이 모두 정의 자리라 자동으로는 구별되지
않으므로, 사유를 적은 면제로 등록했다.

```python
WAIVERS = {
    ("planning/worldmodel.py", "ALPHA_PRIVILEGED", 0.30):
        "책임 분담 비율(무차원).  robot_radius=0.30 과 값만 우연히 같다 ...",
}
CONCEPT_WAIVERS = {
    ("planning/paths.py", "speed_levels"):
        "AgentTrack 의 타임스텝당 속도 단계 **개수**(정수 K). ...",
}
```

면제는 **자리마다** 적는다 (값 하나를 통째로 빼면 그 값의 진짜 재선언까지 함께
통과한다).  그리고 `test_every_waiver_still_applies` 가 "등록됐지만 코드에는
없는" 유령 면제를 잡는다 — 면제가 소리 없이 쌓이지 않게.

현재 면제는 위 2건뿐이고, 둘 다 §5 에 발견 사항으로 기록되어 있다.
**게이트 5 의 문언("config 밖 물리 리터럴 0건")은 이 판정 아래에서 통과한다.**

---

## 4. 회귀 대조 결과

### 4.1 (a) `scripts/check_wm.py` — 5종 시나리오

원본의 동결된 기준선 `results/check_wm_v0.txt` 와 v2 의 출력
`results/check_wm_v2.txt` 를 대조했다.

```
$ diff -u Mahoi-WM/mahoi-wm/results/check_wm_v0.txt MAHOI-WM_v2/results/check_wm_v2.txt
(출력 없음 — 0줄 차이)
```

**바이트 단위로 동일하다.**  team time 뿐 아니라 feasible 후보 수
(`3/16`, `15/16`, `9/16`, `3/16`, `5/16`), 하한, 표준편차, 최악 gap 까지 전부.

| 시나리오 | 하한 | 시드 0/1/2 team time | 평균 | sd | 최악 gap | valid |
|----------|------|----------------------|------|-----|----------|-------|
| crossing2 | 14.3 | 16.4 / 16.4 / 16.4 | 16.4 | 0.00 | 0.922 | OK |
| corridor2 | 17.4 | 18.1 / 18.2 / 18.2 | 18.2 | 0.05 | 0.931 | OK |
| deadlock2 | 14.0 | 15.5 / 15.4 / 15.4 | 15.4 | 0.05 | 0.921 | OK |
| chain3 | 17.9 | 19.4 / 19.5 / 25.2 | 21.4 | 2.71 | 0.746 | OK |
| fork3 | 18.1 | 19.4 / 20.2 / 19.9 | 19.8 | 0.33 | 0.934 | OK |

`all checks passed: 0 collisions, 0 dependency violations, speed limits
respected, across every scenario and seed.`

§3.4 의 궤적 대조가 이보다 강하다 — 같은 15회 실행의 **전 타임스텝**이 동일했다.

### 4.2 (b) smoke 배치

원본 `bench/configs/smoke.json` 을 v2 에 그대로 복사해 양쪽에서 돌렸다.
원본 실행의 출력은 스크래치패드로 뺐다 (원본 저장소를 건드리지 않기 위해).

```
$ cd Mahoi-WM/mahoi-wm && python -m bench.run --config bench/configs/smoke.json \
      --out <scratch>/orig_smoke.csv --fresh
$ cd MAHOI-WM_v2   && python -m bench.run --config bench/configs/smoke.json \
      --out <scratch>/v2_smoke.csv --fresh
$ python scripts/s2_regression/compare_smoke.py <scratch>/orig_smoke.csv \
                                                <scratch>/v2_smoke.csv
```

양쪽 모두 `6 units / 15 rows`.  키 `(uid, method, planner_seed)` 로 맞춘 결과:

| uid | method | ps | status | team_time (원본) | (v2) | \|d\| | valid | 판정 |
|-----|--------|----|--------|------------------|------|-------|-------|------|
| n2_chain_15091d17 | coordination_astar | 0 | ok | 14.2 | 14.2 | 0.0e+00 | True | OK |
| n2_chain_15091d17 | lower_bound | 0 | ok | 13.9 | 13.9 | 0.0e+00 | — | OK |
| n2_chain_15091d17 | sequential | 0 | ok | 18.9 | 18.9 | 0.0e+00 | True | OK |
| n2_chain_15091d17 | wm_planner | 0 | ok | 14.8 | 14.8 | 0.0e+00 | True | OK |
| n2_chain_15091d17 | wm_planner | 1 | ok | 14.8 | 14.8 | 0.0e+00 | True | OK |
| n2_chain_7d341442 | coordination_astar | 0 | ok | 14.0 | 14.0 | 0.0e+00 | True | OK |
| n2_chain_7d341442 | lower_bound | 0 | ok | 11.2 | 11.2 | 0.0e+00 | — | OK |
| n2_chain_7d341442 | sequential | 0 | ok | 21.6 | 21.6 | 0.0e+00 | True | OK |
| n2_chain_7d341442 | wm_planner | 0 | ok | 14.1 | 14.1 | 0.0e+00 | True | OK |
| n2_chain_7d341442 | wm_planner | 1 | ok | 14.4 | 14.4 | 0.0e+00 | True | OK |
| n2_chain_e2f46c6a | coordination_astar | 0 | ok | 16.6 | 16.6 | 0.0e+00 | True | OK |
| n2_chain_e2f46c6a | lower_bound | 0 | ok | 12.7 | 12.7 | 0.0e+00 | — | OK |
| n2_chain_e2f46c6a | sequential | 0 | ok | 19.0 | 19.0 | 0.0e+00 | True | OK |
| n2_chain_e2f46c6a | wm_planner | 0 | ok | 12.8 | 12.8 | 0.0e+00 | True | OK |
| n2_chain_e2f46c6a | wm_planner | 1 | ok | 12.9 | 12.9 | 0.0e+00 | True | OK |

```
행 15개 중 불일치 0개 (비교 제외 열: ['git_commit', 'runtime_s'])
RESULT: IDENTICAL
```

**성공/실패 플래그 완전 일치, makespan 차이 0.0 (기준 < 1e-9).**  게이트가
요구한 두 열뿐 아니라 `flow_time` · `distance` · `wait` · `agent_clearance` ·
`obstacle_clearance` · `dep_violations` · `n_switches` · `n_modes` ·
`n_feasible_modes` · `chosen_is_fastest_feasible` · `ratio_to_bound` 를 포함한
**전 열**이 일치했다.  제외한 두 열은 재현 대상이 아니다 —
`runtime_s` 는 벽시계, `git_commit` 은 저장소가 다르니 당연히 다르다.

**부동소수 수준을 넘는 차이는 한 건도 없었다.**  따라서 "원인 특정" 절차를
발동할 일이 없었고, "원본의 버그를 v2 가 고쳤다"에 해당하는 사례도 없었다.

### 4.3 (c) 대조 스크립트와 산출물

저장소에 보존했다.

```
scripts/s2_regression/README.md         사용법과 한계
scripts/s2_regression/compare_vo.py     묶음 2 (원본 저장소를 직접 import)
scripts/s2_regression/compare_world.py  묶음 3
scripts/s2_regression/compare_traj.py   묶음 4 (별도 프로세스 + npz)
scripts/s2_regression/compare_smoke.py  묶음 6b (CSV 두 개를 인자로)

results/check_wm_v2.txt                 (a) 의 출력 — v0 과 diff 0줄
results/s2/orig_smoke.csv               (b) 원본 실행 결과 15행
results/s2/v2_smoke.csv                 (b) v2 실행 결과 15행
results/s2/smoke_compare.txt            (b) 대조 출력 전문
results/baseline_v0.json                test_regression.py 의 기준선 (원본 그대로)
```

네 스크립트는 **원본 저장소 경로가 안에 박혀 있고 원본을 읽기만 한다.**
일회성 감사 도구라 그렇게 두었다.  원본이 사라지면 이 넷은 돌지 않지만, 그때는
`tests/test_regression.py` (결과를 `results/baseline_v0.json` 에 얼려 둔 것)가
같은 역할을 계속한다.

### 4.4 게이트 6 — 원본 저장소 무변경

작업 시작 전에 원본 트리(`.venv` / `.git` 제외) **9,846개 파일**의 md5 를
떠 두고, 작업이 끝난 뒤 다시 떠서 비교했다.

```
$ diff orig_baseline.md5 orig_after.md5
(출력 없음)
$ git -C Mahoi-WM rev-parse HEAD
abf8b8a0cd6ab3febe6482eef2245e71baf62a9f     (시작 시점과 동일)
```

**변경 없음.**  원본에서 Python 을 실행할 때는 `PYTHONDONTWRITEBYTECODE=1` 을
걸어 `__pycache__` 가 생기지 않게 했고, 배치 출력은 `--out` 으로 저장소 밖에
떨어뜨렸다.  (원본 저장소에는 작업 전부터 있던 미커밋 변경 28건이 있는데, 그
집합도 그대로다.)

---

## 5. 이관 중 발견 사항

S0 감사에 없었거나, 예상과 달랐던 것들.

**(1) `safety/validate.py` 가 계층 규칙과 정면으로 충돌한다.**
지시서는 `validate.py` 를 "그대로" `safety/` 로 옮기라고 했지만, 그 파일은
`coordination.Solution` 과 `world.Problem` 을 import 하고 둘 다 `planning/`
소속이다.  다행히 **어노테이션 외에는 쓰이지 않아** 전방 참조로 바꿔 해결했다
(§3.1).  만약 실제로 쓰였다면 계층 배치 자체를 다시 논의해야 했다.

**(2) `PHASE_*` 는 안전 계층의 어휘이기도 하다.**  `_project_safe` 와 VO 둘 다
"이 에이전트는 움직여 달라고 할 수 없다 / 비켜 주지 않는다"를 알아야 한다.
S0 §G-2 의 배치도에는 이 의존이 나타나 있지 않았다.  `safety/phase.py` 로 내리고
`planning/control.py` 가 re-export 하는 것으로 정리했다 (§3.2).

**(3) 부동소수 덧셈 순서가 게이트를 좌우한다.**  VO 항을 별도 배열에 모았다가
합치는 "더 깔끔한" 추출은 마지막 자리를 어긋나게 하고, 그것이 `argmin` 을 뒤집어
rollout 전체를 갈라놓을 수 있다.  제자리 누적을 유지해야 했다 (§3.2).
같은 이유로 `ANGLE_LEVELS` 의 `np.deg2rad` → `math.radians` 교체도
**비트 단위 동일성을 확인한 뒤에** 했다.

**(4) `world.py:265` 의 `margin = 0.45` 는 S0 가 말한 "재선언"이 아니다.**
같은 줄의 `radius = 0.30` 은 `AgentSpec.radius` 의 재선언이 맞지만, 0.45 는
다른 어디에도 없는 생성기 전용 값이고 `paths.build_prior(margin=0.12)`(가시성
그래프 여유)와도 다른 개념이다.  `PHYSICS.gen_obstacle_margin` 이라는 이름을
새로 붙여 구분했다 (값 동일).

**(5) 값이 우연히 겹치는 자리가 셋 있다.**  §3.8 의 표.  `2.6`(장애물 좌표),
`0.22`(필렛 비율), `0.30`(책임 분담 비율).  물리 상수 판정을 값만으로 하면
반드시 오탐이 난다는 증거다.

**(6) `speed_levels` 라는 이름이 두 개다.**  `CONTROLLER.speed_levels` 는
v_max 에 곱하는 **배율 튜플** `(0.0, 0.35, 0.70, 1.0)`, `paths.build_track(
speed_levels=1)` 은 타임스텝당 속도 단계 **개수**(정수 K)다.  지시서가
"`speed_levels` 인자 구조 유지"라고 못박았으므로 그대로 두고 면제로 등록했다.

**(7) 컨트롤러 후보 속도는 65개다** (4 x 16 + 1).  S1 지시서가 적은 "6 x 6 = 36"
이 아니다.  `config/controller.py:67` 이 이미 이 사실을 적어 두고 있었고, 이관
후 실제 코드에서도 확인했다 (`_candidates` 가 격자 뒤에 `v_pref` 자신을 하나
더 붙인다).

**(8) `plan_coordination_multiroute` 는 완전히 죽은 코드였다** — 호출부 0건.
지시서가 이미 알고 있었지만, `VisibilityGraph` 와 `build_route_options` 를
`coordination.py` 에서 쓰는 곳도 **이 함수뿐이었다**는 점은 새로 확인한 것이다.

**(9) `mahoi/wm/scorer.py` 는 원본에서 미커밋 상태였다** (`git status` 에
`?? mahoi/wm/scorer.py`).  torch 의존이고 P3-4 실험용이다.  이관하지 않기로 한
결정과 정합적이다.

**(10) 원본 저장소도 `PYTHONPATH=` 없이는 pytest 가 죽는다.**  v2 만의 문제가
아니라 이 기계의 ROS 설치가 pytest 플러그인을 등록해 놓아서다.  README 에 적어
두었다.

**(11) 원본 `bench/runs/smoke.csv` 는 커밋 `ada8a46` 의 산물이라 현행 HEAD 와
비교 기준으로 쓸 수 없었다.**  그래서 §4.2 는 저장된 CSV 를 쓰지 않고 **원본을
지금 다시 돌려** 대조했다.  (참고로 그 옛 CSV 의 `n2_chain_e2f46c6a` /
`wm_planner` / `ps=1` 은 12.9 로, 이번 양쪽 결과와도 같다.)

---

## 6. "S3 이후 후보"

이관 중 발견했지만 **건드리지 않은** 것들이다.  S2 의 규칙상 값도 알고리즘도
바꿀 수 없어 그대로 두었다.

**(1) `stall_window(45) > horizon_steps(40)` — rollout 정체 판정이 죽어 있다.**
`config/planner.py` 가 이미 경고로 알리고 있다.  `Cost.stalled` 는 항상 False,
`feasible` 은 항상 True 가 되고, P1b 전 8,100행에서 실제로 그랬다.
horizon 을 10.0 으로 올리면(아래 (2)) 자동으로 해소된다.

**(2) `horizon_s` 를 4.0 → 10.0 으로.**  P0-3 에서 전수 100 인스턴스 짝지은
비교로 **24승 1패, p<0.0001, 성공률 +7.6%p** 가 이미 측정되어 있다.  대가는
cpu 1.88배.  S2 게이트 때문에 4.0 을 유지했을 뿐, 근거는 갖춰져 있다.

**(3) `planner.py` 의 `n_unfin = 0 if roll.hit_horizon`.**  지평선에 걸린
rollout 은 미완 에이전트를 **한 명도** 세지 않는다.  둘이 남았든 다섯이
남았든 하드 항이 똑같이 0 이라, "얼마나 많이 남았는가"가 순위에 전혀 반영되지
않는다.  terminal critical-path 항이 그 몫을 한다는 것이 원본의 논거지만,
그 항은 **충돌을 완화한 하한**이라 남은 인원수에 비례하지 않는다.
지시서 지시대로 그대로 두었고 `NOTE(S2)` 주석을 달아 두었다.

**(4) `worldmodel.sample_modes` 의 전수 조합 생성.**  자르기 **전에** 전체
조합을 리스트로 만든다.  agent 6명이면 138,240개, 8명이면 30,965,760개다.
`WMConfig.time_budget_s` 는 메인 루프 **안에서만** 검사되므로 여기서 멈추면
wall-clock cap 이 발동할 기회조차 없다 — `bench/run.py` 가 인스턴스마다 별도
프로세스 + 타임아웃으로 감싸는 이유가 이것이다.  지시서대로 **S4(Phase 1)** 의
일로 남겼다.

**(5) `soft_margin` 이 두 값으로 공존한다.**  `CostWeights.soft_margin=0.25`
(planner)와 `CONTROLLER.vo_soft_margin=0.22`(controller).  같은 이름, 다른 값,
다른 용도.  S2 에서는 둘 다 다른 이름으로 보존했다.  통일할지, 통일한다면
어느 쪽으로인지는 rollout 이 달라지는 변경이라 S3 이후다.

**(6) `v_nominal` 이 아직 아무 데도 쓰이지 않는다.**  현행 컨트롤러는
`SPEED_LEVELS` 배율을 `v_max` 에 곱할 뿐이라 순항 속도라는 개념이 없다.
config 에는 있고 값은 `v_max` 와 같다.  `a_max` 를 넣는 S3 에서 함께 정리하는
것이 자연스럽다.

**(7) `a_max = inf`.**  급변은 `W_TURN`(비용)으로만 억제된다 — 상한 강제가
아니다.  계획서 §2 목표 3.0 을 넣으면 `config/__init__.py` 의 규칙 (c)
(제동거리 ≤ min_sep)가 그때부터 실제로 검사된다.  `v_max=2.0, a_max=3.0` 조합은
제동거리 0.667 m vs min_sep 0.70 m 로 **아슬아슬하게** 통과한다
(`test_plan_target_a_max_passes` 가 미리 확인해 둔 값).

**(8) `Problem.min_sep` 은 쌍마다 계산되는데 현행 데이터는 전원 0.30 이다.**
`PHYSICS.min_sep`(0.70)은 사실상 그 상수판이다.  이종 로봇을 도입하면 config 의
그 property 는 기본값 역할만 하게 된다.

**(9) `bench/analyze.py`(646 LOC)에는 대응 테스트가 23개뿐이고 그마저
포매팅 위주다.**  이관 대상이 아니어서 그대로 옮겼지만, 분석 로직의 회귀를
잡을 장치는 사실상 없다.

**(10) scorer 트랙을 되살릴 때의 자리.**  §3.4 에서 지운 4개 필드와 한 블록은
`learn/` 이 채워질 때 되돌아올 자리다.  당시의 설계 근거("t=0 에서만 거른다",
"후보 수를 늘리지 않으므로 rollout 비용은 그대로")는 원본
`mahoi/wm/scorer.py` 의 모듈 docstring 에 남아 있다.
