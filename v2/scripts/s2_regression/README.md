# S2 이관 회귀 대조 스크립트

원본 `Mahoi-WM/mahoi-wm` 과 v2 가 같은 시드에서 같은 결과를 내는지 확인한 것들이다.
**원본 저장소 경로가 스크립트 안에 박혀 있고**, 원본은 읽기만 한다 (import + 실행).
S2 게이트를 위한 일회성 감사 도구라 그렇게 두었다 — 원본이 사라지면 이 넷은
돌지 않지만, 그때는 `tests/test_regression.py`(결과를 `results/baseline_v0.json`
에 얼려 둔 것)가 같은 역할을 계속한다.

    compare_vo.py      묶음 2  add_vo_cost / _ttc / _project_safe  (무작위 100 케이스)
    compare_world.py   묶음 3  AgentSpec/Problem 기본값, 시나리오 5종,
                               random_problem 100 시드, build_prior 20 시드
    compare_traj.py    묶음 4  executor 궤적 전체 (5종 x 3 시드) — 별도 프로세스로
                               각각 돌려 .npz 로 떨어뜨린 뒤 비교한다
    compare_smoke.py   묶음 6b bench CSV 행 단위 대조 (키: uid, method, planner_seed)

실행 (ROS 격리를 위해 PYTHONPATH 를 비운다):

    PYTHONPATH= python scripts/s2_regression/compare_traj.py

`compare_smoke.py` 만 CSV 두 개를 인자로 받는다.  만드는 법은
`reports/S2_migration.md` §4 참조.

## S3 이후

`compare_traj.py` 는 **`config/physics.py` 가 S2 값일 때만** 의미가 있다.
S3-A 에서 `a_max` 가 3.0 이 되었으므로 지금 그대로 돌리면 "다르다" 가 나온다 —
그것이 정상이다.  원본과의 비트 단위 동일성을 다시 확인하려면 `a_max=math.inf`
로 되돌리고 돌린다 (`tests/test_amax.py` 가 그 되돌림 경로를 지키고 있다).
