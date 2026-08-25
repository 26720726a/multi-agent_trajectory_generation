# results/ — 요약 수치만

그림과 GIF 는 커밋하지 않습니다. 여기에는 **텍스트 요약만** 둡니다.

수십 KB 라 저장소에 부담이 없고, 텍스트라 **diff 가 됩니다** — "이 PR 에서
chain3 가 19.4 → 19.0 으로 좋아졌다" 가 리뷰 화면에 바로 보입니다.
그림이 필요하면 각자 로컬에서 `scripts/run_wm_experiments.py` 를 돌리세요.

| 파일 | 내용 |
|---|---|
| `baseline_v0.json` | v0 회귀 기준선. `tests/test_regression.py` 가 이것과 비교합니다 |

## 기준선 갱신

의도적으로 좋아졌을 때만, **단독 커밋으로**, 왜 좋아졌는지를 메시지에 적어서.

```bash
python tests/test_regression.py --update
git add results/baseline_v0.json
git commit -m "baseline: A1 conflict-graph 로 chain3 19.4 -> 19.0s"
```
