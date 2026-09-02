# mode 디코드 필드 vs 학습 토큰 필드 대조 (S6-3 D1-1)

출처: `planning/worldmodel.py:58` `PlanMode` / `:64` `label()` / `learn/data.py:37`
`parse_mode()` / `learn/data.py:107` 토큰 채우기.

## PlanMode 가 담는 필드 — **전부 4개다**

```python
@dataclass(frozen=True)
class PlanMode:
    routes: Tuple[int, ...]        # 에이전트별 경로 색인
    yield_rank: Tuple[int, ...]    # 에이전트별 양보 우선순위 (0 = 먼저)
    cautious: bool = False         # 양보하는 쪽 선호속도 ×0.75
    split_side: bool = False       # 특권/양보 쪽이 서로 반대편으로 통과
```

`label()` 은 이 4개를 **손실 없이** 문자열로 만든다:
`r{routes 자릿수}|{양보순 이름들}|{c/s 태그}` → 예 `r0102|B>A>C>D|s`.

| # | PlanMode 필드 | 값 범위 (실측) | label 위치 | **토큰에 실리는가** | 위치 |
|---|---|---|---|---|---|
| 1 | `routes[i]` | 0~3 (`k_routes=2` → 다리별 2×2=4) | `r` 뒤 i번째 자리 | **구성원만** | agent 10:14 원핫 4 |
| 2 | `yield_rank[i]` | 0~n−1 | 이름 나열 순서 | **구성원만** | agent 14 = `rank[i]/(n−1)` |
| 3 | `cautious` | bool | 태그 `c` | 예 | CLS 2 |
| 4 | `split_side` | bool | 태그 `s` | 예 | CLS 3 |

**확인:** `parse_mode()` 는 label 을 완전히 역변환한다. 즉 **mode 문자열이
담은 정보 중 디코드되지 않고 버려지는 필드는 없다.**

## 토큰이 빠뜨리는 것 — mode 필드가 아니라 **클러스터 밖 에이전트**

| 빠진 정보 | 이유 |
|---|---|
| **비구성원 에이전트의 `routes`/`yield_rank`** | 토큰은 클러스터 구성원 슬롯만 채운다 (`learn/data.py:107` 의 `for j, i in enumerate(mem)`). 그런데 rollout 은 **전원 합동**이다 — 클러스터 밖 에이전트의 경로/양보가 구성원의 진행을 바꿀 수 있다. |
| `n_agents` | rank 를 `n−1` 로 나누는 데만 쓰이고 값 자체는 어디에도 없다. CLS 에 있는 것은 `k`(클러스터 크기)다. |
| route 색인이 **가리키는 기하** | 원핫 4칸은 "몇 번 경로냐"만 말한다. 그 경로가 어디로 도는지, 얼마나 긴지, 다른 에이전트 경로와 어디서 만나는지는 들어 있지 않다. `routes[i]=1` 의 뜻은 에이전트마다·장면마다 다르다. |

**클러스터가 전체의 부분집합인 그룹 비율: train 78.7 % / val 70.3 % / test 69.9 %**
(확인). 즉 대부분의 그룹에서 클러스터 밖 에이전트가 존재한다.

## 계획서 §3.2 와의 차이 — **지시서가 든 두 항목은 구현에 없다**

지시서는 "속도 프로파일(A/B/C)"과 "횡방향 오프셋 폭(±0.80/±1.20 m)"을
mode 필드로 들었다. **이 구현의 `PlanMode` 에는 둘 다 없다** (확인):

| 지시서가 든 항목 | 이 구현의 실제 모습 | 근거 |
|---|---|---|
| 속도 프로파일 A/B/C | `cautious` **이진값 하나**. 켜지면 양보 쪽 선호속도에 **0.75 고정** 배율 | `planning/worldmodel.py:85` `speed_scale()` |
| 횡 오프셋 폭 ±0.80/±1.20 m | `split_side` **이진값 하나**. 부호 ±1 만 주고, 세기는 **전역 상수** `W_SIDE = 0.16` | `planning/worldmodel.py:91` `side()`, `config/controller.py:50`, `planning/control.py:695` |

`r0102|B>A>C>D|s` 의 `B>A>C>D` 는 **속도 프로파일이 아니라 에이전트 이름의
양보 순서**다 (`label()` 의 `names`).

**추론: 계획서 §3.2 가 정의한 mode 공간과 `planning/worldmodel.py` 가 실제로
표본하는 mode 공간이 다르다.** 토큰이 mode 정보를 빠뜨린 것이 아니라,
mode 자체가 계획서보다 좁다.
