# S1 — 골격 + config 단일 출처 (소급 검증)

S1 은 보고서 없이 커밋(`e6049b1`)만 남았다.  S2 를 시작하기 전에 S1 게이트
5항목을 지금 검증하고 그 결과를 여기 남긴다.  **검증은 코드를 고치지 않고
현행 상태 그대로 돌린 것이다** — 한 항목만 예외이며 §6 에 적었다.

검증 시점: 2026-08-29.  대상 커밋 `e6049b1` (S2 작업 시작 전).
실행 환경: `Mahoi-WM/mahoi-wm/.venv` 의 Python 3.12 / numpy 2.5.2 / pytest 9.1.1.

---

## 1. 게이트 표

| # | 항목 | 결과 |
|---|------|------|
| 1 | `python -c "from config import PHYSICS, PLANNER, CONTROLLER"` 성공 | **통과** |
| 2 | S0 §C 15개 개념이 config 3파일에 전부 있는가 | **통과** (15/15, §3) |
| 3 | `stall_window(45) > horizon_steps(40)` 경고가 실제로 출력되는가 | **통과** |
| 4 | `tests/test_layering.py` 존재 및 통과 | **통과** (단서 있음, §5) |
| 5 | config 값이 전부 현행 값인가 (계획서 목표값이 아닌가) | **통과** |

**종합: 5/5 통과. S2 를 시작해도 되는 상태.**

---

## 2. 항목 1 — import

```
$ PYTHONPATH= python -c "from config import PHYSICS, PLANNER, CONTROLLER; \
                         print(PHYSICS.v_max, PLANNER.horizon_s, PHYSICS.a_max)"
<string>:1: ConfigWarning: [config] stall_window >= horizon_steps: ...
1.2 4.0 inf
```

성공한다.  `ConfigWarning` 은 항목 3 에서 다룬다 — 의도된 경고다.

`PYTHONPATH=` 를 앞에 붙인 이유는 이 기계의 `PYTHONPATH` 가
`/opt/ros/jazzy/lib/python3.12/site-packages` 를 가리키고 있어서다.  이것 자체는
S1 의 문제가 아니다 (§5 참조).

---

## 3. 항목 2 — S0 §C 15개 개념 대조표

**먼저 밝혀 둘 것이 있다.**  S0 보고서 파일은 두 저장소 어디에도 없다
(`Mahoi-WM/`, `MAHOI-WM_v2/` 전체에서 §C 를 담은 문서를 찾지 못했다).  따라서
아래 15개 목록은 **S0 §C 를 인용한 config 의 주석들**(`config/physics.py:4,29,36,45,57`,
`config/planner.py:3,77`, `config/controller.py:47,67`)과 S1 지시서가 명시한
두 항목(`v_nominal`, `a_max=inf`)에서 역으로 복원한 것이다.  원문과 항목 순서나
표현이 다를 수 있다.

| # | 개념 | 위치 | 현행 값 | 원본에서의 상태 |
|---|------|------|---------|-----------------|
| 1 | `v_max` | `physics.v_max` | 1.20 | world.py:35 에 1.20, build_dataset.py:44 에 1.5 — **값이 어긋나 있었다** (§C-2) |
| 2 | `v_nominal` | `physics.v_nominal` | 1.20 | **코드에 없던 개념**.  현행 재현을 위해 v_max 와 같은 값 |
| 3 | `a_max` | `physics.a_max` | `inf` | **제한이 없었다** (§C-1).  급변은 `W_TURN` 으로 비용만 매긴다 |
| 4 | `dt` (시뮬레이션 타임스텝) | `physics.dt` | 0.10 | world.py:76 `Problem.dt` |
| 5 | 제어 주기 | `physics.control_dt` (파생) | 0.10 | **별도 정의 없음**.  `controller.step` 이 정확히 1 dt |
| 6 | `robot_radius` | `physics.robot_radius` | 0.30 | world.py:34 와 world.py:265 에 **두 번** (§C-2) |
| 7 | `safety_margin` | `physics.safety_margin` | 0.10 | world.py:75 `Problem.safety` |
| 8 | `min_sep` | `physics.min_sep` (파생) | 0.70 | world.py:83-84 에서 쌍마다 계산 |
| 9 | `interact_radius` | `physics.interact_radius` | 0.55 | world.py:36.  이웃 판정에는 쓰이지 않는다 |
| 10 | `map_size` | `physics.map_size` | 10.0 | world.py:250 `random_problem(size=10.0)` |
| 11 | `dwell_s` | `physics.dwell_s` | 1.0 | world.py:33.  시나리오 11곳은 1.2~1.4 를 개별 지정 |
| 12 | 제동 거리 | `physics.braking_distance` (파생) | 0.0 | 개념 자체가 없었다 (a_max 가 없으므로) |
| 13 | `horizon_s` | `planner.horizon_s` | 4.0 | execute.py:51,320,325,330 + run_wm_experiments.py:43 — **네 곳 재기입** |
| 14 | `replan_s` | `planner.replan_s` | 1.0 | 위와 같음 |
| 15 | `stall_window` | `planner.stall_window` | 45 | worldmodel.py:178,184 의 **인자 기본값**이라 config 로 조절 불가였다 |

15/15 전부 config 에 있다.  `v_nominal` 과 `a_max` 도 지시대로 들어 있고, 둘 다
"원본에 없던 것을 현행 동작이 유지되는 값으로 넣었다"는 점이 주석에 적혀 있다.

`config/controller.py` 는 위 15개에 들어가지 않는 컨트롤러 튜너블 13개
(`lookahead`, `tau`, `tau_obst`, `vo_soft_margin`, `obst_pad`, `w_ttc`,
`w_ttc_obst`, `w_turn`, `w_side`, `snap_tol`, `dep_standoff`, `speed_levels`,
`angle_levels`)를 추가로 담고 있다.  S2 에서 전부 실제로 쓰였다.

---

## 4. 항목 3 — 경고가 실제로 나오는가

```
$ PYTHONPATH= python -W always -c "import config"
<string>:1: ConfigWarning: [config] stall_window >= horizon_steps: stall_window=45 가
horizon 4.0s = 40 스텝 이상이라, rollout 의 정체 판정이 발동할 수 없다.
Cost.stalled 는 항상 False 가 되고 feasible 은 항상 True 가 된다 ...
```

나온다.  `horizon_steps(0.10) = 40`, `stall_window = 45` 이므로 규칙 (a) 가
걸린다.  이것은 **원본의 상태를 그대로 옮긴 결과**이고, import 를 막지 않는
것이 옳다 — 막으면 S2 의 회귀 재현이 불가능해진다.

`tests/test_config_validation.py` 8개가 이 동작을 별도로 고정하고 있다
(현행 위반이 이 하나뿐인지, 메시지에 45 와 40 이 들어 있는지, horizon 을 10.0 으로
올리면 사라지는지 등).

---

## 5. 항목 4 — `tests/test_layering.py`

파일은 존재하고, S1 커밋 시점의 전체 스위트(13개)가 통과한다.

```
$ PYTHONPATH= python -m pytest -q
13 passed, 1 warning in 0.01s
```

**단서 두 가지를 기록해 둔다.**

**(a) `PYTHONPATH=` 없이는 수집 단계에서 죽는다.**  ROS 가 pytest 플러그인
(`launch_testing`, `launch_ros`, `ament_*`)을 entry point 로 등록해 두는데 그
의존인 `yaml` 이 이 venv 에 없다.  **원본 저장소도 똑같이 죽는다** — 즉 S1 이
만든 문제가 아니라 환경 문제다.  v2 의 README 에 실행법을 적어 두었고, 실행
중의 격리는 `bench/run.py:strip_ros_paths` 가 맡는다 (S2 §3 묶음 5).

**(b) (c) 항목은 S1 시점에 자명하게 통과했다** — `safety/` 와 `planning/` 이
비어 있었기 때문이다.  테스트 자신의 docstring 이 그렇게 밝히고 있다.  즉
"통과"는 "규칙이 지켜졌다"가 아니라 "검사할 코드가 없었다"였다.  S2 에서 실제
코드를 올리자 이 판정이 **오탐을 낸다**는 것이 드러났고 (시나리오 장애물 좌표
`Rect(2.6, ...)` 와 필렛 비율 `(0.35, 0.22, ...)` 를 물리 상수로 오인),
S2 묶음 3 에서 판정 방식을 고쳤다.  자세한 것은 `reports/S2_migration.md` §3.

---

## 6. 항목 5 — config 값이 현행 값인가

| 항목 | 현행 값 | 계획서 §2 목표값 | 판정 |
|------|---------|------------------|------|
| `v_max` | **1.20** | 2.0 | 현행 값 |
| `horizon_s` | **4.0** | 10.0 | 현행 값 |
| `a_max` | **inf** | 3.0 | 현행 값 |

목표값은 하나도 들어가 있지 않다.  **여기서 멈출 사유 없음.**  S2 를 시작해도
좋다.

세 값의 주석이 각각 "왜 지금 올리지 않는가"(= S2 이관 게이트가 원본과 같은
결과를 요구하므로)를 적어 두고 있어, 나중에 S3 에서 올릴 때 근거를 다시 찾을
필요가 없다.  참고로 `horizon_s=10.0` 은 P0-3 에서 이미 4.0 보다 유의하게
낫다고 측정되어 있다 (짝지은 24승 1패, p<0.0001) — 승격은 S3 다.

---

## 7. S1 이 남기지 않은 것 (S2 에서 채운 것)

소급 검증 중 발견한, 게이트 항목은 아니지만 기록해 둘 것들이다.

| 항목 | 상태 | 조치 |
|------|------|------|
| `reports/` 디렉터리 | 없었다 | S2 에서 생성.  이후 모든 단계가 여기에 남긴다 |
| `results/`, `learn/`, `viz/`, `bench/configs/` | `.gitkeep` 만 | `results/`, `bench/configs/` 는 S2 에서 채웠다.  `learn/`, `viz/` 는 선택 트랙이라 그대로 |
| `scripts/` | 없었다 | S2 묶음 6 에서 생성 (`check_wm.py`, `s2_regression/`) |
| `.pytest_cache/`, `mahoi_wm_v2.egg-info/` | 작업 트리에 있었다 | `.gitignore` 가 이미 막고 있어 커밋되지는 않았다 |
| pytest `slow` 마커 | `pyproject.toml` 에 없었다 | S2 묶음 6 에서 등록 (`tests/test_regression.py` 이관과 함께) |

---

## 8. 결론

S1 게이트 5항목 전부 통과.  §5(b) 의 "자명한 통과"는 S1 의 결함이라기보다
"코드가 오기 전에 세운 가드레일"의 성격상 불가피한 것이었고, 실제 코드가
올라오자 곧바로 드러나 S2 에서 고쳤다 — 가드레일이 의도대로 작동한 셈이다.
