# bench/ — 대량 평가

**소유: Track B.** Track A 는 이 폴더를 건드리지 않습니다.


## 지금 상태

돌아가는 뼈대만 있습니다. CSV 스키마와 실행 루프, 재현 정보(seed + git commit)
기록까지는 되어 있고, **난이도 축과 분석은 비어 있습니다** — 그게 B1 입니다.

```bash
python -m bench.run --config bench/configs/smoke.json      # 5초, 설정 확인용
python -m bench.analyze --csv bench/runs/smoke.csv
```

## 파일

| 파일 | 역할 |
|---|---|
| `generate.py` | 난이도 격자 → `Instance` 스트림. `world.random_problem()` 을 감쌉니다 |
| `run.py` | 인스턴스 × 방법 실행 → CSV 한 줄씩. 예외는 `status` 열에 담고 계속 갑니다 |
| `analyze.py` | CSV → 요약 표 + 성공률 곡선 |
| `configs/smoke.json` | 5초. 설정이 살아있는지만 확인 |
| `configs/scale.json` | agent 2–8명 확장 곡선 |
| `configs/difficulty.json` | 난이도 축 스윕 |

출력은 `bench/runs/` 에 쌓이고 **`.gitignore` 되어 있습니다.** 요약 수치만
`results/` 에 커밋하세요.

## B1 에서 해야 할 것

**1. 난이도 축을 실제로 만들 것** (`generate.py`)

지금 노출된 축은 agent 수 · dependency 구조 · 방 크기 · 장애물 개수 넷뿐입니다.
난이도 곡선이 의미를 가지려면 최소한 이것들이 더 필요합니다.

- **통로 폭** — corridor2 에서 문 폭 1.8 m 가 두 agent(필요 간격 0.7 m)를 나란히
  통과시켜 협응 A\* 대비 −30.9 % 가 나왔습니다. 이 폭을 좁혀가며 **언제 그 이점이
  사라지는지**가 이 방법의 적용 범위를 정합니다. 지금은 측정할 수 없습니다.
- **waypoint 근접도** — `random_problem` 의 결합 거리가 `0.55` 로 하드코딩되어
  있습니다. 이걸 축으로 빼야 "HOI 지점이 얼마나 붙어야 어려워지는가"를 답합니다.
- **dependency 밀도와 구조** — 지금 `chain` / `fork` / `none` 뿐입니다.
- **혼잡도** — agent 수와 방 크기를 따로 두지 말고 `n / size²` 로 묶는 편이
  곡선이 깔끔할 수 있습니다. 확인해 보세요.

**2. seed 를 반드시 여러 개 돌릴 것**

`chain3` 가 seed 0/1/2 에서 19.4 / 19.5 / **25.2** 초입니다 (표준편차 2.71 s).
단일 실행 수치는 신뢰할 수 없습니다. 이 편차 자체가 A4 의 성공지표이므로,
seed 축을 접지 말고 **분포로 남기세요**.

**3. 4명 이상의 baseline** (→ B5)

`coordination_astar` 는 `n > 3` 에서 자동으로 `skipped:lattice_too_large` 가
됩니다. agent 를 늘리는 순간 비교 상대가 사라진다는 뜻입니다. 확장 가능한
상한(순차 · 우선순위 계획)을 세워야 합니다.

**4. 실패를 원인별로 쪼갤 것** (→ B6)

지금 `status` 는 `ok` / `no_solution` / `unfinished:...` / `error:...` 정도입니다.
A 에게 넘길 작업 목록이 되려면 **deadlock · livelock · mode 집합에 답이 없음 ·
cost 오판**을 구분해야 합니다. 셋째 항목은 작은 인스턴스에서 전수 탐색으로
확인 가능하고, 그게 A4 의 직접적인 근거가 됩니다.

## 실행 시간 감각

`wm_planner` 는 인스턴스당 2-agent 약 2초, 3-agent 약 14초입니다. A1 이 끝나기
전에는 `n ≥ 6` 에서 `sample_modes` 하나가 30초씩 잡아먹으므로, `scale.json` 은
A1 이후에 돌리는 편이 낫습니다.

1,000 인스턴스를 하룻밤에 돌리려면 인스턴스 단위 병렬 실행이 필요합니다.
`run_instance()` 가 순수 함수라 `multiprocessing.Pool` 로 감싸면 됩니다.
