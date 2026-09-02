# S4-A — 물리 사양 복원

## 게이트

| 게이트 | 결과 |
|---|---|
| `compare_traj.py` 비트 단위 일치 | **통과** — 5 시나리오 × 3 시드, mismatched arrays 0 |
| `check_wm.py` | **통과** — 전 시나리오/시드에서 충돌·의존성 위반 0 |
| crossing2 FAST 안전 3건 | **통과** — `tests/test_safety.py` 15 passed (crossing2·deadlock2 포함) |
| 전체 테스트 | **미달** — S3 유한 제동 전용 테스트가 복원된 기본 `a_max=inf` 대신 3.0을 고정 기대 |

## 변경

**확인:** `a_max=math.inf`, `multistep_lookahead=False`로 복원했다.
`brake_lookahead=None`, `wipeout_decelerate=False`, `horizon_s=4.0`은 유지했다.
유한 가속도일 때만 유효한 제동 식 코드는 지우지 않았으며, `a_max=inf`에서
자동으로 무효다.

## 결론

원본 궤적 회귀와 안전 확인은 통과했다. 그러나 전체 테스트 게이트가 미달이므로
절대 규칙에 따라 테스트를 결과에 맞춰 고치거나 S4-B 배치를 시작하지 않는다.
S4-B/C는 보류다.
