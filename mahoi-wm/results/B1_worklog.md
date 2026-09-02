B1 벤치마크 하네스 — 작업 기록 정리본
1. 작업 개요
날짜: 2026-08-26
담당: Track B (kjs)
범위: B1 — 대량 벤치마크 하네스
최종 목표: 하룻밤 동안 약 1,000개 인스턴스를 실행하고, 결과를 텍스트 성적표 + 요약 그림으로 생성
B1의 중요성: B2·B3·B6 및 A4·A6의 성공지표가 B1에서 생성되는 벤치마크 결과를 전제로 함. 따라서 B1이 완료되지 않으면 이후 항목의 검증도 진행하기 어려움.
2. 현재 진행 상태
단계	내용	상태
0	실행 환경 안정화	✅ 완료
1	축/seed/UID 정리	✅ 완료
2	타임아웃 및 안전한 실행기	✅ 완료
3	병렬 실행 + 재개	✅ 완료
4	성적표 생성	✅ 완료
대량 배치	difficulty.csv 실행	🔄 진행/완료 후 분석 예정
추가 검증	makespan loss / optimal baseline	⏳ 다음 작업
통로 폭 축	room_kind / door_width	⏳ Track A와 협의 후 진행
3. 환경 문제 및 해결
발생 문제

scripts/run_wm_experiments.py 실행 시:

ImportError: numpy.core.multiarray failed to import

원인은 다음과 같음.

apt의 matplotlib → NumPy 1.x 기준으로 컴파일
~/.local의 NumPy → 2.x
서로 ABI가 맞지 않아 충돌
해결

프로젝트 전용 venv를 사용:

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
추가 문제

ROS Jazzy가 PYTHONPATH에 /opt/ros/jazzy/...를 추가하여 pytest가 외부 플러그인을 로드하면서:

No module named 'yaml'

발생.

해결 방법

테스트 실행 시:

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q

코드에서는 strip_ros_paths()를 통해 sys.path와 PYTHONPATH에서 ROS 경로를 제거.

4. 기준선 확보

scripts/check_wm.py 전체 실행 결과를:

results/check_wm_v0.txt

에 고정.

컨테이너와 로컬 환경에서 동일한 결과가 나와 OS와 라이브러리 버전이 달라도 결정론성이 유지됨을 확인했다.

기준선 결과
시나리오	하한	seed 0/1/2	SD	worst gap	feasible / 16
crossing2	14.3	16.4 / 16.4 / 16.4	0.00	0.922	3
corridor2	17.4	18.1 / 18.2 / 18.2	0.05	0.931	15
deadlock2	14.0	15.5 / 15.4 / 15.4	0.05	0.921	9
chain3	17.9	19.4 / 19.5 / 25.2	2.71	0.746	3
fork3	18.1	19.4 / 20.2 / 19.9	0.33	0.934	5
기준선에서 발견한 핵심 사항

① B2 관련 관찰

5개 시나리오 모두:

fastest == chosen

즉 flow / wait / dist / clear / turn / dev의 soft cost 6개 항이 실제 선택을 바꾸지 않았다. 당시 기준선에서는 실질적인 목적함수가 w.make 하나처럼 동작했다.

② seed 편차와 feasible density

corridor2: 16개 중 15개 feasible → SD 0.05
chain3: 16개 중 3개 feasible → SD 2.71

즉 feasible mode가 적을수록 planner seed에 따른 결과 편차가 커지는 경향을 발견했다.

③ chain3

worst gap = 0.746으로 유일하게 0.9 이하.

특히 25.2s가 chain3에서 발생했으며, 기준선에서 가장 불안정한 시나리오로 판단된다.

5. Step 1 — 벤치마크 축과 이름표 정리
5.1 Instance seed / Planner seed 분리

기존에는 seed가 두 종류인데 벤치마크에서 하나만 변화시키고 있었다.

Seed	역할
instance_seed	문제 자체 생성 — 장애물, 좌표 등
planner_seed	같은 문제에서 16개 mode 중 어떤 것을 sampling할지 결정

기존 config는:

wm_config.seed = 0

으로 고정되어 있었다.

따라서 기존 하네스로는 A4의 chain3 seed 편차 2.71s → 0.5s 목표를 제대로 측정할 수 없었다.

이를:

Axis.instance_seeds
Axis.planner_seeds

로 분리했다.

5.2 coupled_waypoints 의미 수정

기존에는 coupled_waypoints가 거리처럼 보였지만 실제로는 확률이었다.

기존: coupled_waypoints
실제 의미: coupling 발생 확률
거리: d = 0.55로 하드코딩

따라서 다음과 같이 분리:

couple_prob
couple_dist

world.py의 기본 거리 0.55는 유지했다.

6. UID 생성 문제 수정

기존 UID 생성 방식에는 문제가 있었다.

int(size)

를 사용하면서:

10.0
10.5

같은 값을 구분하지 못했다.

또한 새로운 축이 추가될 경우 UID 충돌 가능성이 있었다.

해결

dataclasses.fields 전체를 정규화한 뒤 hashlib로 UID를 생성.

필드 이름 포함
float → .6g
정수형 float → 정수 표현
모든 축 정보를 UID에 반영

이렇게 해서 같은 인스턴스의 다른 planner seed/method 결과를 안전하게 묶을 수 있게 했다.

7. Seed shorthand 버그 발견 및 수정

가장 중요한 버그 중 하나.

기대:

[0, 1]
→ (0, 1)

실제:

[0, 1]
→ range(0, 1)
→ (0,)

따라서 seed 1은 중복 제거된 것이 아니라 애초에 생성되지 않았음.

수정

이제 규칙을 명확히 분리:

[0, 1]                → literal
{"range": [0, 3]}     → range
입력	이전	수정 후
[0, 1]	(0,)	(0,1)
[0, 3]	(0,1,2)	(0,3)
{"range":[0,3]}	없음	(0,1,2)

또한 expected_count()를 추가해:

예상 생성 개수
vs
실제 생성 개수

를 직접 검증하도록 했다.

8. 조건부 축 추가

축을 단순 곱으로 처리하면 의미 없는 조합이 생성될 수 있다.

예:

room_kind = open
door_width = 0.9

open에서는 door width가 의미 없으므로 이런 조합을 만들면 성공률 통계가 왜곡된다.

따라서 조건부 축을 지원하도록 수정했다.

9. Step 2 — 죽지 않는 실행기
9.1 기존 timeout의 문제

WMConfig.time_budget_s는 execute.py 내부에서만 확인한다.

하지만 그보다 먼저:

sample_modes()

가 실행된다.

mode 개수가 급격히 증가하면:

n=6 → 138,240개
n=8 → 30,965,760개

가 생성될 수 있다.

따라서 n≥7에서는 기존 timeout이 발동하기 전에 OOM 또는 장시간 정지가 발생할 수 있었다.

해결

프로세스 레벨 timeout을 추가.

검증 결과:

n2 → 정상
n6 → timeout
n2 → 정상

timeout이 발생해도:

해당 child process만 종료
전체 batch는 계속 진행
고아 프로세스 없음
실패한 instance도 method별 row 기록

하도록 했다.

10. CSV 스키마 확장

다음 열을 추가했다.

instance_seed
planner_seed
couple_prob
couple_dist
timeout_s
n_modes
n_feasible_modes
chosen_is_fastest_feasible
note

이를 통해 이후 seed 편차와 feasible mode density를 분석할 수 있게 되었다.

11. Soft cost가 실제 선택을 바꾸는지 검증

기존에는 5개 시나리오에서 모두 True였지만, 더 넓은 테스트에서 처음으로 False가 발견됐다.

예:

chosen
total = 14.368
makespan = 12.7402

fastest
total = 14.369
makespan = 12.7340

→ False

즉 플래너가 argmin(total)을 사용하고 있기 때문에 계산 오류는 아니었다.

중요한 해석상의 문제

차이가 매우 작다.

makespan loss ≈ 6.2 ms
soft cost 차이 ≈ 7.2 ms

따라서 이것만으로:

"soft cost가 의미 있게 작동한다."

라고 결론 내리면 안 된다.

실제로는 거의 동점인 상황에서 미세한 차이가 선택을 바꾼 것일 가능성도 있다.

따라서 최종 보고서에서는:

False 비율
+
makespan loss 분포

를 함께 봐야 한다.

12. Step 3 — 병렬 실행 및 재개
12.1 Resume 기능

기존:

open(out, "w")

→ 재시작하면 CSV 전체 삭제.

수정:

append

기존 CSV에서 완료된:

(uid, method, planner_seed)

를 확인하고 없는 작업만 실행한다.

왜 instance 단위가 아닌가?

하나의 instance에서도 method별로 별도의 row가 생성된다.

따라서 실행 중 중단되면:

method A → 저장됨
method B → 저장됨
method C → 저장 안 됨

같은 상태가 가능하다.

그래서 resume 단위를:

(uid, method, planner_seed)

로 설정했다.

12.2 잘린 CSV 처리

실행 중 프로세스가 죽으면 마지막 row가 일부만 저장될 수 있다.

이를:

마지막 완전한 row까지 되돌림
잘린 row 폐기
폐기된 row 수 기록
해당 작업 재실행

하도록 구현했다.

note 안의 개행도 제거하여 한 row = 한 line을 보장했다.

13. 병렬 실행

imap_unordered를 사용해 instance 단위 병렬화를 적용.

결과는 도착하는 즉시 기록하고 flush한다.

분석에서는 실행 순서가 아니라:

uid

로 결과를 짝지으므로 row 순서는 중요하지 않다.

14. Daemonic Pool 문제

기존 multiprocessing.Pool worker는 daemonic이라 child process를 생성할 수 없었다.

결과:

AssertionError:
daemonic processes are not allowed to have children

따라서 timeout을 위해 필요한 child process를 생성할 수 없었다.

해결

Non-daemonic pool을 직접 구현:

_NoDaemonProcess
_NoDaemonContext

이를 통해:

B outer Pool
 └─ worker
     └─ timeout child process

구조를 허용했다.

15. Track A와 병렬화 계약

Track A에서도 rollout 병렬화를 진행하기 때문에:

B Pool
 └─ A Pool

구조가 되면 CPU 과다 사용이나 deadlock 문제가 생길 수 있다.

따라서 계약을 다음과 같이 설정했다.

--inner-workers
        ↓
MAHOI_INNER_WORKERS
        ↓
Track A Pool 크기

또한:

OMP_NUM_THREADS = 1
등 thread 변수 5종 = 1

로 설정.

Pool initializer에서 strip_ros_paths()도 명시적으로 실행한다.

16. Step 3 검증 결과
검증	결과
병렬 = 순차	✅ 동일
Ctrl-C → 재개	✅ 성공
완료 batch 재실행	✅ 0 remaining
--fresh	✅
schema 불일치	✅ exit 2
pool 내부 timeout	✅
row count 검산	✅
pytest	109 passed, 6 skipped
check_wm.py	byte 단위 동일

특히:

15 rows
(expected 15)

검산이 성공했다.

이 검산 덕분에 실행 중 발생한 0 != 15 문제를 즉시 발견할 수 있었다.

17. Step 4 — 성적표
핵심 원칙

PNG보다 텍스트 보고서를 1차 산출물로 사용한다.

이유는 A1/A4의 전후 성능을:

git diff

로 직접 비교할 수 있기 때문이다.

재현성을 위해 고정한 것
실행 시각 제거
절대 경로 제거
runtime 제거
정렬 순서 고정
소수점 자릿수 고정
빈 섹션도 "해당 없음" 출력
CSV row 순서에 관계없이 동일한 결과 생성

동일 CSV를 두 번 분석했을 때 MD5까지 동일함을 검증했다.

18. 성적표 구성
섹션	내용	목적
[1]	방법별 요약	전체 성능
[2]	난이도별 성공률	적용 범위
[3]	Planner seed 편차	A4 지표
[4]	같은 UID의 paired comparison	최적 대비 손해
[5]	실패 분해 + 재현 UID	Track A 작업 목록
[6]	Cost 항 관여 여부	B2

ratio_to_bound는 평균이 아니라 분위수로 출력.

이유:

chain3
19.4 / 19.5 / 25.2

처럼 일부 큰 값이 평균을 왜곡하기 때문이다.

19. 대량 배치 중간 결과

중간 스냅샷:

3,884 rows
planner seed = 3개

단, 이 값은 배치 진행 중 수치이므로 최종 결과가 아님.

19.1 Planner seed 편차

현재 중간 결과:

Agent 수	n	SD p50	SD p90	SD 최대
2	492	0.05s	0.22s	2.11s
3	196	0.08s	0.45s	0.99s
전체	688	0.05s	0.29s	2.11s

기존 A4 목표:

chain3 SD 2.71s → 0.5s 이하

현재 대량 데이터에서는:

p50 = 0.05s
p90 = 0.29s

로 이미 목표 이하.

따라서 기존 2.71s는 대표적인 평균값이 아니라 tail 값에 가까운 것으로 보인다.

→ A4 성공지표를 특정 샘플 하나가 아니라 p90 또는 최대값 기준으로 재정의하는 것을 Track A에 제안할 필요가 있음.

20. 실패 유형 분해

현재 중간 결과:

일부 seed만 성공: 36
모든 seed 실패: 53
의미

일부 seed만 실패

→ 동일한 문제에서 planner seed에 따라 성공/실패가 갈림
→ A4가 집중적으로 해결해야 할 대상

모든 seed 실패

→ 문제 자체가 어려움 또는 feasible solution이 없을 가능성
→ planner seed 안정성 문제와 구분해야 함.

이 구분은 단순히:

실패율 = XX%

라고 하는 것보다 훨씬 중요하다.

21. Cost 항 관여율

대량 배치 중간 결과:

Agent 수	판정된 행	False	관여율
2	1620	101	6.2%
3	710	57	8.0%
전체	2330	158	6.8%

기존 5개 시나리오에서는:

0%

였지만 대량 데이터에서는:

6.8%

로 증가했다.

그러나 아직 결론을 내리면 안 됨

현재 CSV에는:

makespan_loss_s

가 없다.

따라서:

"6.8%에서 soft cost가 의미 있게 작동했다."

라고 결론을 내릴 수 없다.

필요한 것은:

False 비율
+
makespan loss distribution

이다.

22. 현재 배치에서 추가하지 않은 것

현재 실행 중인 CSV에는 makespan_loss_s가 없기 때문에 이번 배치를 재실행하지 않는다.

이유:

이미 약 17시간의 실행 결과가 존재
성공률 계산 가능
seed 편차 계산 가능
paired comparison 가능
손실 분포만 다음 배치에서 추가하면 됨

따라서 현재 결과를 버리고 다시 실행하는 것은 비효율적이다.

23. Commit 섞임 문제

대량 배치 도중 코드 commit이 발생했다.

commit 2개 혼합

ada8a46 → 2246 rows
f7b5d24 → 1638 rows

변경된 부분은 run.py의 병렬/재개 로직이며 simulation 결과 자체에는 영향을 주지 않는 것으로 판단했다.

앞으로의 원칙
Commit
 ↓
Batch 실행

순서를 반드시 지킨다.

특히 CSV에 commit hash를 기록하는 구조이므로 실행 코드와 기록된 commit이 일치해야 재현성 주장이 성립한다.

24. 현재 남은 문제
24.1 makespan_loss_s 추가

run.py에 다음 열 추가:

makespan_loss_s

이미 wm_mode_stats에 두 makespan이 있으므로 계산 자체는 단순하다.

다음 batch부터:

p50
p90
max

등의 손실 분포를 분석할 수 있다.

주의: 현재 실행 중인 batch에서는 schema 변경 금지.

24.2 coordination_astar 추가

difficulty.json에:

coordination_astar

추가.

이것이 현재 필요한 최적 baseline이다.

이를 통해:

하한 대비 성능

이 아니라:

최적 baseline 대비 손해

를 말할 수 있게 된다.

24.3 A4 성공지표 재협의

기존:

chain3 SD
2.71s → 0.5s

현재 대량 결과를 보면 2.71s는 대표값이라기보다 tail에 해당한다.

따라서:

p90 ≤ 0.5s

또는

max ≤ 0.5s

등으로 지표를 다시 정의할 필요가 있다.

25. 통로 폭 실험 — 현재 보류

현재 random_problem()은 겹치지 않는 랜덤 사각형을 생성하기 때문에 구조적인 좁은 통로가 만들어지지 않는다.

즉:

n_obstacles 증가

만으로는:

좁은 통로

가 아니라:

산발적인 장애물 증가

가 된다.

그런데 corridor2의 −30.9% 성능 차이를 설명하려면 실제 통로 폭을 조절할 수 있어야 한다.

필요한 변경
room_kind
 ├─ open
 └─ corridor
       └─ door_width

agent 반경:

0.30m

safety:

0.10m

이므로 두 agent가 나란히 지나갈 수 있는 임계 폭은 대략:

1.3m

근처로 예상.

이 부근에서 성능 knee가 나타나는지 확인하면 방법의 실패 원인을 구조적으로 설명할 수 있다.

단, world.py를 수정해야 하므로 Track A와 사전 공유 필요.

26. Git Commit 기록
Commit	내용
a7922c1	check_wm v0 기준선
765116a	두 seed 분리, UID hash, seed shorthand
edabf6b	기준선 결과 — 일부 출력 잘림
53db9d4	.gitignore 정리
—	hard timeout, planner seed, mode 통계
—	병렬 실행, resume, row count 검산

추가로:

__pycache__/
*.pyc
.venv/

는 추적되지 않도록 정리 필요.

27. 최종적으로 해야 할 일
우선순위 1 — B1 결과 확정
python -m bench.analyze \
  --csv bench/runs/difficulty.csv \
  > results/bench_v0.txt

그리고:

git add results/bench_v0.txt
git commit -m "results: difficulty 배치 v0 (n=2..4)"

이 파일이 B1의 핵심 최종 산출물이다.

우선순위 2 — 다음 batch 준비
makespan_loss_s 추가
coordination_astar 추가
같은 방식으로 batch 재실행
cost 항의 실제 makespan 손실 분석
우선순위 3 — A4 재협의
2.71s

를 그대로 성공지표로 사용할지,

p90

또는

max

기준으로 바꿀지 결정.

우선순위 4 — 통로 폭 축
room_kind
door_width

를 추가하고 Track A와 world.py 변경 사항 공유.