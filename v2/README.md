# MAHOI-WM v2

원본 `Mahoi-WM/mahoi-wm` 의 이관 대상 (S0 감사 §G-1 구조).

## 계층

    config/     물리·계획·컨트롤러 상수의 단일 출처.  어느 계층도 import 하지 않는다
    safety/     안전 계층.  planning/ 을 import 하지 않는다 (tests/test_layering.py)
    planning/   계획 계층.  safety/ 를 호출한다
    bench/      벤치 하네스
    learn/      선택 트랙 (torch).  S1 에서는 비어 있다
    viz/        선택 트랙 (matplotlib).  S1 에서는 비어 있다

## 현재 상태 (S2)

원본 `Mahoi-WM/mahoi-wm` 의 코드가 이관되어 있다.  **로직은 바꾸지 않았다** —
위치를 옮기고 상수 출처를 config 로 돌렸을 뿐이다.  같은 시드에서 원본과 같은
결과가 나오는 것이 S2 의 게이트였고, 결과는 `reports/S2_migration.md` 에 있다
(5종 시나리오 x 3 시드에서 궤적이 비트 단위로 동일, smoke 배치 15행 전열 일치).

    scripts/check_wm.py     5종 시나리오 검증 (원본 results/check_wm_v0.txt 과 대조)
    python -m bench.run --config bench/configs/smoke.json

## 테스트

    PYTHONPATH= python -m pytest -q              # 155개, 수 초
    PYTHONPATH= python -m pytest -q --runslow    # + 5종 회귀 6개, 약 30초

`PYTHONPATH=` 를 앞에 붙이는 이유는 이 기계의 `PYTHONPATH` 가 `/opt/ros/jazzy`
를 가리키고 있어서다.  ROS 가 pytest 플러그인(`launch_testing` 등)을 등록해 두는데
그 의존(`yaml`)이 이 venv 에 없어 **수집 단계에서** 죽는다.  원본 저장소도 같은
증상이며 (S2 §2 참조), 실행 중의 격리는 `bench/run.py:strip_ros_paths` 가 맡는다.

config 값은 전부 **원본 현행 값**이다.  계획서 §2 목표값(v_max=2.0,
a_max=3.0, horizon=10.0s)으로의 변경은 **S3** 이며, 지금 바꾸면 S2 의
이관 게이트(원본과 같은 결과가 나오는가)를 세울 수 없다.

    python -c "from config import PHYSICS, PLANNER, CONTROLLER"

import 시 `ConfigWarning` 이 하나 나온다 — `stall_window=45` 가
`horizon 4.0s = 40 스텝` 보다 커서 rollout 정체 판정이 발동할 수 없다는
경고다.  **이것은 원본의 상태를 그대로 옮긴 결과이며 의도된 것이다** (S3 에서 해소).
