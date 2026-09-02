# S4-A′ — S3 전용 테스트 범위 정리

## 결론

**확인:** S4-A의 복원 상태 (`a_max=inf`, `brake_lookahead=None`,
`wipeout_decelerate=False`, `multistep_lookahead=False`)에서 전체 테스트는
`238 passed, 18 skipped`로 통과했다. 이 중 S4-A′가 조건부 실행으로 바꾼
S3 전용 테스트는 12건이고, 나머지 6건은 원래부터 `--runslow`가 필요한
회귀 테스트다. 물리와 무관하게 실패한 테스트는 0건이다.

**확인:** 별도 스크래치 복사본에서 S3 구성(`a_max=3.0`,
`multistep_lookahead=True`)으로 되돌린 뒤 해당 모듈을 실행해
`71 passed`를 얻었다. 즉 skip 조건은 테스트를 숨기는 장치가 아니라,
현재 물리 사양에서 의미 없는 S3 전용 측정을 보류하는 조건이다.

알고리즘 코드(`planning/`)는 변경하지 않았다. 이번 변경은 테스트의
실행 조건과 보고서뿐이다.

## 1. 실패 테스트 인벤토리

S4-A 복원 직후의 실패/무의미 판정을 전수 점검했다. 아래의 12건은 현재
`a_max=inf` 또는 `multistep_lookahead=False`와 모순되어, 판정식은 그대로
두고 조건부 skip으로 처리했다.

| 파일:테스트 | 검증 내용 | 전제 config | 처리 |
| --- | --- | --- | --- |
| `test_amax.py::test_the_shipped_a_max_is_the_plan_target` | S3-A 목표 `a_max=3.0` 및 0.24 m 정지거리 | `a_max=3.0` | 유한 `a_max`일 때만 실행 |
| `test_amax.py::test_multistep_filter_rejects_a_candidate_that_cannot_stop_before_a_wall` | B2 다단계 정지 궤적 필터 | `multistep_lookahead=True` | 해당 플래그가 켜졌을 때만 실행 |
| `test_braking.py::test_brake_speed_is_the_standard_constant_deceleration_formula` | `sqrt(2*a_max*d)` 제동식 | `a_max=3.0` | 유한 `a_max`일 때만 실행 |
| `test_braking.py::test_the_braking_term_binds_exactly_inside_the_stopping_distance` | 0.24 m 안쪽 제동 결합 | `a_max=3.0` | 유한 `a_max`일 때만 실행 |
| `test_braking.py::test_the_braking_term_still_needs_2x_a_max_at_the_very_end` | 이산 제동의 `2*a_max` 경계 | `a_max=3.0` | 유한 `a_max`일 때만 실행 |
| `test_braking.py::test_far_from_the_target_the_braking_term_does_nothing` | 정지거리 밖에서 제동항 무효 | `a_max=3.0` | 유한 `a_max`일 때만 실행 |
| `test_braking.py::test_the_violation_budget_on_a_fixed_scenario_is_pinned` | B2의 crossing2 위반 예산 | `a_max=3.0`, `multistep_lookahead=True` | 두 조건이 모두 충족될 때만 실행 |
| `test_braking.py::test_the_fan_half_angle_comes_from_a_max_not_a_magic_number` | `a_max=3.0`에서의 14.48°/30° 팬 반각 | `a_max=3.0` | 유한 `a_max`일 때만 실행 |
| `test_braking.py::test_geometric_braking_is_off_because_measurement_showed_it_useless` | S3-A의 유한 가속 제동거리 0.180 m 및 `brake_lookahead=None` | `a_max=3.0`, `brake_lookahead=None` | 유한 `a_max`일 때만 실행; `brake_lookahead` 자체는 올바르게 꺼져 있음을 계속 판정 |
| `test_braking.py::test_the_wipeout_deceleration_is_off_because_it_broke_the_collision_gate` | S3-B3의 `a_max` 감속 fallback | `a_max=3.0`, `wipeout_decelerate=False` | 유한 `a_max`일 때만 실행; fallback은 계속 꺼져 있음을 판정 |
| `test_config_validation.py::test_braking_rule_is_now_actually_checked` | 유한 가속도에서 제동거리 config 검증 | `a_max=3.0` | 유한 `a_max`일 때만 실행 |
| `test_physics_fingerprint.py::test_changing_a_physics_value_changes_the_fingerprint[{a_max=inf}]` | 현행 값과 다른 `a_max=inf`가 지문을 바꾸는지 | 현행 `a_max=3.0` | 현행 값이 유한할 때만 해당 parameter 실행 |

`brake_lookahead`와 `wipeout_decelerate`는 현재 꺼져 있는 것이 사양이다.
두 테스트는 그 상태를 검증하므로 skip 사유가 아니다. 다단계 필터만
`multistep_lookahead=True`일 때의 동작 자체를 검증하므로 별도 skip 처리했다.

## 2. 분류와 처리 방식

### (a) 유한 물리 값에 의존

11개 항목은 `pytest.mark.skipif(not math.isfinite(PHYSICS.a_max))` 또는
동등한 `Physics().a_max` 조건을 쓴다. skip 사유에는 모두 S3의 유한
`a_max=3.0` 실험/제동 구성을 다시 켜야 한다고 적었다. assertion 및 그
판정 의미는 변경하지 않았다.

### (b) 현재 꺼진 기능에 의존

다단계 필터 단위 테스트와 고정 시나리오 위반 예산 테스트는
`pytest.mark.skipif(not CONTROLLER.multistep_lookahead)`를 쓴다. 사유는
`multistep_lookahead=True`로 S3-B2 실험을 재활성화해야 한다는 것이다.

### (c) 물리와 무관한 실패

0건이다.

처음 전체 회귀에서 보인 fingerprint 실패는 알고리즘/지문 구현 문제는 아니었다.
현행 값이 이미 `a_max=inf`인데 동일 값을 다시 넣어 “지문이 바뀌어야 한다”는
S3 전제의 parameter 하나가 원인이었다. 이 parameter도 (a)로 분류했다.

## 3. skip 목록 감사

다음 명령으로 skip 사유를 매번 출력했다.

```bash
pytest -q -rs
```

출력은 S4-A′ 12건(위 표)과 기존 slow 회귀 6건뿐이었다. S4-A′ 12건의
재활성 조건은 다음과 같이 명시적으로 출력된다.

| 조건 | 건수 | 재활성 방법 |
| --- | ---: | --- |
| 유한 `a_max` | 10 | `a_max=3.0` 등 유한 값으로 S3 가속/제동 구성을 재활성화 |
| 유한 `a_max` + 다단계 | 1 | `a_max=3.0` 및 `multistep_lookahead=True` |
| 다단계 | 1 | `multistep_lookahead=True` |

따라서 skip은 조용히 누적되지 않는다. `-rs` 결과가 표의 12건 및 각 사유와
일치하지 않으면 S4-A의 테스트 범위가 다시 변한 것이다.

## 4. 역검증

원본 작업 트리를 바꾸지 않고 `/tmp/mahoi_s4a2_scratch` 복사본에서 다음 두
config만 S3 값으로 바꿨다.

```text
PHYSICS.a_max = 3.0
CONTROLLER.multistep_lookahead = True
```

그 복사본에서 실행한 명령과 결과는 다음과 같다.

```bash
pytest -q tests/test_amax.py tests/test_braking.py \
  tests/test_config_validation.py tests/test_physics_fingerprint.py
# 71 passed, 1 warning in 31.20s
```

skip 0건으로 모든 S3 전용 테스트가 실제 실행되어 통과했다. 이는 각 marker가
의도한 config 축을 정확히 감지함을 확인한다.

## 5. 최종 검증과 예상 밖 결과

```text
pytest -q -rs
238 passed, 18 skipped, 1 warning in 65.05s
```

경고는 S4-B에서 horizon을 10.0 s로 승격하기 전까지 예상되는
`stall_window >= horizon_steps` config 경고다. 테스트 실패는 없다.

예상 밖이었던 점은 `test_physics_fingerprint`의 한 parameter도 S3 전용이었다는
것이다. 처음에는 일반 지문 회귀처럼 보였지만, 현재 값과 같은 `a_max=inf`를
주입하고 지문 변경을 요구하므로 S2/S4-A 사양에서는 불가능한 판정이었다.
이를 분리한 뒤 물리와 무관한 실패는 남지 않았다.
